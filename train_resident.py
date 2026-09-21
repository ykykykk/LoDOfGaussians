"""Optional resident fine trainer; keeps upstream rendering and classic splits.

No new extension build is needed. The legacy trainer remains the default for
old configs. Select this runtime explicitly via training_backend='resident'.
"""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import platform
import random
import time

import torch

from utils.resident_pool import ResidentPool, capacity_for_budget
from utils.resident_selection import select_gaussians
from utils.training_profile import TrainingProfile


@dataclass
class ResidentOptions:
    pool_gib: float = 2.5
    headroom_gib: float = 4.0
    transfer_rows: int = 65536
    pin_staging: bool = True
    profile_every: int = 100
    validate_indices: bool = False

    def __post_init__(self):
        if self.pool_gib <= 0 or self.headroom_gib < 0 or self.transfer_rows <= 0 or self.profile_every < 0:
            raise ValueError("invalid resident runtime budget or profiling interval")


def validate_options(opt, runtime=None):
    settings = ResidentOptions(**(runtime or {}))
    if opt.storage_device != "cpu" or opt.densification != "classic":
        raise ValueError("resident v1 supports classic densification with CPU backing storage; use training_backend=legacy for other modes")
    if opt.prune_unused or opt.dampen_scale_grad or opt.optimize_exposure or opt.use_occlusion_culling:
        raise ValueError("resident v1 expects the standard classic RGB preset (no experimental pruning, scale damping, exposure or occlusion mode)")
    return settings


def parameter_rates(opt, xyz_lr, width, device):
    rates = torch.empty(width, dtype=torch.float32, device=device)
    rates[:3] = xyz_lr
    rates[3:6] = opt.scaling_lr
    rates[6:10] = opt.rotation_lr
    rates[10:13] = opt.feature_lr
    rates[13] = opt.opacity_lr
    rates[14:] = opt.feature_lr
    return rates.mul_(opt.lr_multiplier)


def _backward(packet, camera, g, opt, pipe, background, profile):
    from gaussian_renderer import render_gsplat
    from utils.loss_utils import l1_loss
    from fused_ssim import fused_ssim

    if camera.invdepthmap is not None:
        raise ValueError("resident v1 is RGB-only; the existing gsplat wrapper does not provide the required inverse-depth image")
    raw = packet.parameters()
    with profile.phase("render_loss"):
        pkg = render_gsplat(
            camera, raw[:, :3].contiguous(), torch.sigmoid(raw[:, 13:14]),
            torch.exp(raw[:, 3:6]), g.rotation_activation(raw[:, 6:10]),
            raw[:, 10:13].unsqueeze(1), raw[:, 14:].reshape(len(raw), -1, 3),
            pipe, background, sh_degree=g.active_sh_degree,
        )
        image = pkg["render"]
        gt = camera.original_image.cuda(non_blocking=True)
        # Preserve the effective loss of train_hierarchy.py, including its
        # unmasked SSIM term. Changing mask semantics is a separate experiment.
        predicted = image
        if camera.alpha_mask is not None:
            predicted = image * camera.alpha_mask.cuda(non_blocking=True)
        loss = (1.0 - opt.lambda_dssim) * l1_loss(predicted, gt)
        loss = loss + opt.lambda_dssim * (1.0 - fused_ssim(image[None], gt[None]))
        scalar = loss.detach().item()
        if not math.isfinite(scalar):
            raise FloatingPointError("non-finite resident training loss")
    with profile.phase("backward"):
        loss.backward()
        screen_grad = pkg["viewspace_points"].grad
        if screen_grad is None:
            raise RuntimeError("gsplat did not retain means2d gradients needed for classic splits")
        packet.accumulate_scores(pkg["packed_indices"], torch.linalg.vector_norm(screen_grad, dim=-1))
    return scalar, raw.grad


def training(dataset, opt, pipe, saving_iterations, view_graph=None, runtime=None):
    from scene import Scene, GaussianModel
    from utils.general_utils import get_expon_lr_func
    from utils.training_runtime import make_camera_loader, shutdown_camera_loader
    from utils import view_graph_utils
    from tqdm import tqdm

    settings = validate_options(opt, runtime)
    g = GaussianModel(opt.SH_degree)
    scene = Scene(dataset, g, resolution_scales=[1], create_from_hier=True, llff_hold=opt.llff_hold)
    g.max_sh_degree = opt.SH_degree
    g.active_sh_degree = min(1, opt.SH_degree)
    for name in ("_xyz", "_opacity", "_rotation", "_scaling", "_features_dc", "_features_rest"):
        getattr(g, name).requires_grad_(False)
    g.compact_gaussians("cpu", opt.cap_max, densification="classic", prune_unused_gaussians=False)
    cameras = scene.getTrainCameras()
    if not len(cameras):
        raise ValueError("no training cameras")
    base_focal = cameras[0].focal_length
    granularity = opt.target_granularity_pixels / base_focal

    def rebuild():
        g.build_hierarchical_SPT(opt.SPT_root_volume, granularity, opt.min_SPT_size,
                                use_bounding_spheres=opt.use_bounding_spheres, revive_gaussians=False)

    rebuild()
    width = g.properties.shape[1] // 3
    driver_free, _ = torch.cuda.mem_get_info()
    reusable = max(0, torch.cuda.memory_reserved() - torch.cuda.memory_allocated())
    free = driver_free + reusable
    capacity = capacity_for_budget(min(opt.cache_size, opt.cap_max), width, free,
                                   settings.pool_gib, settings.headroom_gib)
    pool = ResidentPool(g.properties, g._densification_criterium, capacity,
                        transfer_rows=settings.transfer_rows, pin_staging=settings.pin_staging)
    output = Path(dataset.output_path)
    output.mkdir(parents=True, exist_ok=True)
    profile = TrainingProfile(output / "resident_profile.jsonl", settings.profile_every)
    loader = make_camera_loader(cameras, opt, shuffle=not opt.graph_view_select)
    loader_iter = iter(loader)
    current_camera = list(view_graph.nodes())[0] if opt.graph_view_select else 0
    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                              dtype=torch.float32, device="cuda")
    lr_schedule = get_expon_lr_func(
        lr_init=opt.position_lr_init * g.spatial_lr_scale,
        lr_final=opt.position_lr_final * g.spatial_lr_scale,
        lr_delay_mult=opt.position_lr_delay_mult, max_steps=opt.position_lr_max_steps)
    (output / "resident_run.json").write_text(json.dumps({
        "runtime": asdict(settings), "pool_capacity": capacity, "property_width": width,
        "driver_free_bytes": driver_free, "reusable_reserved_bytes": reusable,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "platform": platform.platform(),
        "resolution": dataset.resolution, "iterations": opt.iterations,
        "optimization": vars(opt), "timing_note": "CUDA event spans include stream idle/waits; host_ms is enqueue time, not GPU kernel time",
    }, indent=2), encoding="utf-8")
    print(f"Resident cache: {capacity:,} slots, {capacity * (width * 3 + 1) * 4 / 2**30:.2f} GiB")
    print("Oversized cuts stream directly; no Gaussian is discarded to fit the cache.")
    progress = tqdm(total=opt.iterations + 1, desc="Resident fine training")
    ema = 0.0
    start = time.perf_counter()
    try:
        # Preserve upstream's counter and split-vs-optimizer branch timing.
        for iteration in range(opt.iterations + 1):
            profile.begin(iteration)
            with profile.phase("data_wait"):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(loader)
                    batch = next(loader_iter)
                camera = batch[0]
                if opt.graph_view_select:
                    current_camera = int(view_graph_utils.random_walk_node(view_graph, current_camera))
                    camera = cameras[current_camera]
                    if iteration % 100 == 0:
                        current_camera = random.randrange(len(cameras))
            with profile.phase("camera_upload"):
                for name in ("world_view_transform", "projection_matrix", "full_proj_transform", "camera_center"):
                    setattr(camera, name, getattr(camera, name).cuda(non_blocking=True))
            multiplier = base_focal / camera.focal_length
            if iteration % 10 and opt.vary_distance_multiplier:
                multiplier = multiplier * (1 + torch.rand(1).pow(4) * 5).cuda()
            with profile.phase("hierarchy_cut"):
                ids = select_gaussians(g, camera, opt, multiplier)
            if not len(ids):
                raise RuntimeError(f"empty hierarchy cut for camera {camera.image_name}; check camera alignment and culling")
            with profile.phase("cache_prepare"):
                packet = pool.acquire(ids, validate=settings.validate_indices or pipe.debug)
            if iteration > 0 and iteration % max(1, int(opt.iterations * opt.SH_increase_after_train_percent)) == 0:
                g.oneupSHdegree()
            scalar, grad = _backward(packet, camera, g, opt, pipe, background, profile)
            active_count = len(packet.ids)
            ema = 0.4 * scalar + 0.6 * ema
            if iteration in saving_iterations or iteration == opt.iterations:
                with profile.phase("save"):
                    pool.flush()
                    filename = opt.output_file_name if iteration == opt.iterations else f"iteration_{iteration}"
                    g.save_hierarchy(str(output), file_name=filename)
            split = opt.densify_from_iter < iteration < opt.densify_until_iter and iteration % opt.densification_interval == 0
            if iteration < opt.iterations and split:
                with profile.phase("densify_rebuild"):
                    pool.flush()
                    pool.invalidate()
                    # Release the active packet and its gradients before a large
                    # tree rebuild or reloading an expanded set of Gaussians.
                    del packet, grad
                    dead = g.properties[:g.size, 13] <= math.log(0.005 / 0.995)
                    old_size = g.size
                    g.add_new_gs(cap_max=opt.cap_max, size=g.size, densification="classic",
                                 densify_percent=opt.densify_percent, densify_threshold=opt.densify_grad_threshold)
                    mask = torch.zeros(g.size, dtype=torch.bool)
                    mask[:old_size] = dead
                    mask &= g.nodes[:g.size, 2] == 0
                    g.relocate_gs(mask, g.size, storage_device="cpu", densification="classic")
                    # Full rebuild remains necessary to refresh geometry-based
                    # bounds/ranges even when the topology happened not to split.
                    rebuild()
                    pool.reset_scores(g.size)
            else:
                if iteration < opt.iterations:
                    with profile.phase("adam"):
                        rates = parameter_rates(opt, lr_schedule(iteration), width, pool.device)
                        packet.adam_step(grad, rates, iteration, frozen_prefix=g.skybox_points)
                del packet, grad
            if iteration % 10 == 0:
                requests = pool.stats["requested_rows"]
                hit = pool.stats["hit_rows"] / max(requests, 1)
                progress.set_postfix(loss=f"{ema:.6f}", active=active_count, total=g.size, hit=f"{hit:.1%}")
            progress.update(1)
            profile.finish(active=active_count, total=g.size, loss=scalar,
                           allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
                           peak_allocated_bytes=torch.cuda.max_memory_allocated(), **pool.stats)
    finally:
        # Shutdown must not spawn a new iterator/worker set. Flush on normal and
        # exceptional exits; no half-trained file is saved as a successful run.
        shutdown_camera_loader(loader)
        pool.close()
        profile.close()
        progress.close()
    print(f"Resident fine training: {time.perf_counter() - start:.2f} seconds")
