"""Resident v2: incremental SPT, next-view streaming and indexed CUDA ops.

The FP32 reference, native and prefetch paths use the same view schedule and
classic detail threshold. No extra radius/opacity culling or LoD reduction.
"""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import platform
import time

import torch
from utils.incremental_spt import IncrementalSPT
from utils.camera_geometry import screen_scores, effective_focal
from utils.general_policy import DetailWindow, relative_spt_volume
from utils.resident_pool import capacity_for_budget
from utils.resident_pool_v2 import StreamingResidentPool
from utils.resident_selection import select_gaussians
from utils.resident_native import load_native, native_status
from utils.adam_graph import PacketAdamGraph
from utils.training_profile import TrainingProfile
from utils.view_pipeline import ThreadedViews, CameraTransfer, make_view_loader


@dataclass
class ResidentOptions:
    pool_gib: float = 2.5
    headroom_gib: float = 4.0
    transfer_rows: int = 65536
    pin_staging: bool = True
    profile_every: int = 100
    validate_indices: bool = False
    incremental_spt: bool = True
    view_prefetch: bool = True
    gaussian_prefetch_rows: int = 131072
    image_prefetch_mib: int = 512
    image_cache_gib: float = 1.0
    compact_images: bool = True
    native_ops: str = "auto"
    graph_adam: bool = True
    graph_min_reuse: int = 8
    adaptive_pool: bool = False
    checkpoint_every: int = 0

    def __post_init__(self):
        if (self.pool_gib <= 0 or self.headroom_gib < 0 or self.transfer_rows <= 0
                or self.checkpoint_every < 0 or self.profile_every < 0 or self.gaussian_prefetch_rows < 0
                or self.image_prefetch_mib < 0 or self.image_cache_gib < 0 or self.graph_min_reuse < 1):
            raise ValueError("invalid resident budget/prefetch/graph options")
        if self.native_ops not in ("auto", "cuda", "torch"):
            raise ValueError("native_ops must be auto, cuda, or torch")


def validate_options(opt, runtime=None):
    settings = ResidentOptions(**(runtime or {}))
    if opt.storage_device != "cpu" or opt.densification != "classic":
        raise ValueError("resident v2 requires classic densification and CPU backing")
    if opt.prune_unused or opt.dampen_scale_grad or opt.optimize_exposure or opt.use_occlusion_culling:
        raise ValueError("use legacy for experimental pruning, scale damping, exposure or occlusion")
    if getattr(opt, "densify_score_space", "pixel") not in ("pixel", "ndc"):
        raise ValueError("densify_score_space must be pixel or ndc")
    if opt.densify_grad_threshold <= 0 or not math.isfinite(opt.densify_grad_threshold):
        raise ValueError("detail threshold must be positive and finite")
    if not 0 <= getattr(opt, "densify_max_leaf_fraction", 0.) <= 1:
        raise ValueError("invalid per-window leaf fraction")
    if getattr(opt, "densify_max_new_nodes", 0) < 0:
        raise ValueError("invalid new-node budget")
    return settings


def _backward(packet, camera, g, opt, pipe, background, profile):
    from gaussian_renderer import render_gsplat
    from utils.loss_utils import l1_loss
    from fused_ssim import fused_ssim
    if camera.invdepthmap is not None:
        raise ValueError("resident v2 is RGB-only; select legacy for depth supervision")
    raw = packet.parameters()
    with profile.phase('render_loss'):
        pkg = render_gsplat(camera, raw[:, :3].contiguous(), raw[:, 13:14].sigmoid(),
                            raw[:, 3:6].exp(), g.rotation_activation(raw[:, 6:10]),
                            raw[:, 10:13, None].transpose(1, 2), raw[:, 14:].reshape(len(raw), -1, 3),
                            pipe, background, sh_degree=g.active_sh_degree)
        image = pkg['render']
        gt = camera.original_image
        predicted = image if camera.alpha_mask is None else image * camera.alpha_mask
        loss = (1 - opt.lambda_dssim) * l1_loss(predicted, gt)
        # Retain v1's effective unmasked SSIM term.
        loss = loss + opt.lambda_dssim * (1 - fused_ssim(image[None], gt[None]))
    with profile.phase('backward'):
        loss.backward()
        screen = pkg['viewspace_points'].grad
        if screen is None:
            raise RuntimeError('gsplat must retain means2d gradients for detail-driven splits')
        score = screen_scores(screen, camera.original_image.shape[-1], camera.original_image.shape[-2],
                              getattr(opt, 'densify_score_space', 'pixel'))
        packet.accumulate_scores(pkg['packed_indices'], score)
        tracker = getattr(g, '_detail_tracker', None)
        if tracker is not None:
            tracker.observe(packet, pkg['packed_indices'])
    return loss.detach(), raw.grad


def training(dataset, opt, pipe, saving_iterations, view_graph=None, runtime=None, resume_checkpoint=None):
    from scene import Scene, GaussianModel
    from utils.general_utils import get_expon_lr_func
    from utils.training_runtime import shutdown_camera_loader
    from train_resident import parameter_rates
    from tqdm import tqdm
    settings = validate_options(opt, runtime)
    if (settings.checkpoint_every or resume_checkpoint) and opt.vary_distance_multiplier:
        raise ValueError('Checkpoints require fixed distance multiplier; set checkpoint_every=0 for varying-distance runs')
    g = GaussianModel(opt.SH_degree)
    scene = Scene(dataset, g, resolution_scales=[1], create_from_hier=True, llff_hold=opt.llff_hold)
    g.max_sh_degree, g.active_sh_degree = opt.SH_degree, min(1, opt.SH_degree)
    for name in ('_xyz', '_opacity', '_rotation', '_scaling', '_features_dc', '_features_rest'):
        getattr(g, name).requires_grad_(False)
    g.compact_gaussians('cpu', opt.cap_max, densification='classic', prune_unused_gaussians=False)
    from utils.resident_checkpoint import load_checkpoint, save_checkpoint
    contract = (dict(source=str(Path(dataset.source_path).resolve()), resolution=dataset.resolution,
                    hierarchy=str(Path(dataset.hierarchy).resolve()), options=vars(opt), pipeline=vars(pipe))
                if settings.checkpoint_every or resume_checkpoint else {})
    restored = load_checkpoint(resume_checkpoint, g, contract, opt.iterations) if resume_checkpoint else None
    first_iteration = restored['iteration'] + 1 if restored else 0
    cameras = scene.getTrainCameras()
    if not len(cameras):
        raise ValueError('no training cameras')
    first = cameras[0]
    pixel_lod = getattr(opt, 'lod_pixel_consistent', False)
    base_focal = effective_focal(first) if pixel_lod else first.focal_length
    root_volume = relative_spt_volume(opt, g.spatial_lr_scale)
    if base_focal <= 0:
        raise ValueError('camera focal length must be positive')
    builder = IncrementalSPT(root_volume, opt.target_granularity_pixels / base_focal,
                             opt.min_SPT_size, opt.use_bounding_spheres)
    builder.refresh(g)
    native = load_native(settings.native_ops)
    width = g.properties.shape[1] // 3
    def choose_capacity(existing_bytes=0):
        free, _ = torch.cuda.mem_get_info()
        allocated = torch.cuda.memory_allocated()
        reusable = max(0, torch.cuda.memory_reserved() - allocated)
        # Retain space for observed transient render/backward allocations.
        # A new, denser view can still exceed this historical peak.
        transient = max(0, torch.cuda.max_memory_allocated() - allocated)
        headroom = max(settings.headroom_gib, (transient * 1.25 + 2**30) / 2**30)
        requested = min(opt.cache_size, opt.cap_max)
        if settings.adaptive_pool:
            requested = min(requested, g.size + max(65536, g.size // 4))
        return capacity_for_budget(requested, width, free + reusable + existing_bytes,
                                   settings.pool_gib, headroom)
    capacity = choose_capacity()
    pool = StreamingResidentPool(g.properties, g._densification_criterium, capacity,
                                 transfer_rows=settings.transfer_rows, pin_staging=settings.pin_staging,
                                 ops=native, prefetch_rows=settings.gaussian_prefetch_rows)
    output = Path(dataset.output_path)
    output.mkdir(parents=True, exist_ok=True)
    profile = TrainingProfile(output / 'resident_profile.jsonl', settings.profile_every, append=bool(restored))
    seed = restored['seed'] if restored else torch.initial_seed()
    loader = make_view_loader(cameras, opt, settings.image_cache_gib * 2**30, seed,
                              view_graph if opt.graph_view_select else None,
                              compact_images=settings.compact_images, start=first_iteration)
    views = ThreadedViews(loader, opt.iterations + 1 - first_iteration, enabled=settings.view_prefetch)
    transfer = CameraTransfer(prefetch_bytes=settings.image_prefetch_mib * 2**20)
    adam_graph = PacketAdamGraph(settings.graph_min_reuse)
    distance_rng = torch.Generator().manual_seed(seed ^ 0x19C3)
    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                              dtype=torch.float32, device='cuda')
    lr_schedule = get_expon_lr_func(lr_init=opt.position_lr_init * g.spatial_lr_scale,
        lr_final=opt.position_lr_final * g.spatial_lr_scale, lr_delay_mult=opt.position_lr_delay_mult,
        max_steps=opt.position_lr_max_steps)
    rates = parameter_rates(opt, lr_schedule(0), width, pool.device)
    (output / 'resident_run.json').write_text(json.dumps(dict(
        version=2, runtime=asdict(settings), native=native_status(), pool_capacity=capacity,
        property_width=width, torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), platform=platform.platform(), resolution=dataset.resolution,
        iterations=opt.iterations, seed=seed, optimization=vars(opt),
        scene_radius=g.spatial_lr_scale, spt_root_volume=root_volume, lod_base_focal=base_focal), indent=2), encoding='utf-8')
    print(f'Resident v2: {capacity:,} slots; native={native is not None}; incremental_spt={settings.incremental_spt}')
    progress = tqdm(total=opt.iterations+1, initial=first_iteration, desc='Resident v2 fine training')
    ticket = None
    tracker = DetailWindow(opt.cap_max, pool.device) if getattr(opt, 'detail_diagnostics', False) else None
    g._detail_tracker = tracker
    detail_file = (output / 'densification.jsonl').open('a' if restored else 'w', encoding='utf-8')
    empty_windows = 0
    ema, started = 0.0, time.perf_counter()
    if restored:
        if tracker and restored['seen'] is not None:
            tracker.seen[:g.size].copy_(restored['seen'])
            tracker.views = restored['views']
        ema, empty_windows = restored['ema'], restored['empty_windows']
        torch.set_rng_state(restored['rng'])
        torch.cuda.set_rng_state_all(restored['cuda_rng'])
        print(f'Resumed resident checkpoint after iteration {first_iteration-1}')
        del restored

    def select(ticket, iteration):
        if ticket.multiplier is None:
            focal = effective_focal(ticket.cpu) if pixel_lod else ticket.cpu.focal_length
            ticket.multiplier = base_focal / focal
            if opt.vary_distance_multiplier and iteration % 10:
                ticket.multiplier *= float(1 + torch.rand((), generator=distance_rng).pow(4) * 5)
        if ticket.ids is None or ticket.epoch != builder.generation:
            ticket.ids = select_gaussians(g, transfer.metadata(ticket), opt, ticket.multiplier, native=native)
            ticket.epoch = builder.generation
        return ticket.ids

    try:
        for iteration in range(first_iteration, opt.iterations+1):
            profile.begin(iteration)
            with profile.phase('data_wait'):
                if ticket is None:
                    ticket = transfer.submit(views.pop(), speculative=False)
            with profile.phase('camera_upload'):
                camera = transfer.ready(ticket)
            with profile.phase('hierarchy_cut'):
                ids = select(ticket, iteration)
            if not len(ids):
                raise RuntimeError(f'empty cut for {camera.image_name}; check camera alignment and culling')
            with profile.phase('cache_prepare'):
                packet = pool.acquire(ids, validate=settings.validate_indices or pipe.debug)
            # Schedule only a ready CPU lookahead. Do not stall today's GPU work
            # to fetch tomorrow's image, and do not speculate over a tree edit.
            split = (opt.densify_from_iter < iteration < opt.densify_until_iter
                     and iteration % opt.densification_interval == 0)
            next_ticket = None
            with profile.phase('prefetch'):
                if iteration < opt.iterations and settings.view_prefetch:
                    next_cpu = views.pop(block=False)
                    if next_cpu is not None:
                        next_ticket = transfer.submit(next_cpu, speculative=True)
                        if not split and settings.gaussian_prefetch_rows:
                            next_ids = select(next_ticket, iteration+1)
                            pool.prefetch(next_ids, epoch=pool.epoch)
            if iteration > 0 and iteration % max(1, int(opt.iterations * opt.SH_increase_after_train_percent)) == 0:
                g.oneupSHdegree()
            value, grad = _backward(packet, camera, g, opt, pipe, background, profile)
            # Check before Adam, but after all current backward work was queued.
            scalar = float(value)
            if not math.isfinite(scalar):
                raise FloatingPointError(f'non-finite resident loss at iteration {iteration}')
            active_count = len(packet.ids)
            ema = .4 * scalar + .6 * ema
            if iteration in saving_iterations or iteration == opt.iterations:
                with profile.phase('save'):
                    pool.flush()
                    filename = opt.output_file_name if iteration == opt.iterations else f'iteration_{iteration}'
                    g.save_hierarchy(str(output), file_name=filename)
            if iteration < opt.iterations and split:
                with profile.phase('densify_rebuild'):
                    pool.flush()
                    adam_graph.reset()
                    pool.invalidate()
                    del packet, grad
                    old_size = g.size
                    detail = tracker.report(g, opt.densify_grad_threshold, iteration) if tracker else {}
                    dead = g.properties[:old_size, 13] <= math.log(.005/.995)
                    g.add_new_gs(cap_max=opt.cap_max, size=g.size, densification='classic',
                                 densify_percent=opt.densify_percent, densify_threshold=opt.densify_grad_threshold,
                                 max_leaf_fraction=getattr(opt, 'densify_max_leaf_fraction', 0.),
                                 max_new_nodes=getattr(opt, 'densify_max_new_nodes', 0))
                    mask = torch.zeros(g.size, dtype=torch.bool)
                    mask[:old_size] = dead
                    mask &= g.nodes[:g.size, 2] == 0
                    g.relocate_gs(mask, g.size, storage_device='cpu', densification='classic')
                    if not settings.incremental_spt:
                        builder.cache.clear()  # full-refresh A/B, identical construction rules
                    builder.refresh(g)
                    pool.reset_scores(g.size)
                    if tracker is not None:
                        detail.update(new_nodes=g.size-old_size, split_parents=(g.size-old_size)//2,
                            leaf_nodes_after=int((g.nodes[:g.size,2] == 0).sum()),
                            score_space=getattr(opt, 'densify_score_space', 'pixel'))
                        detail_file.write(json.dumps(detail) + chr(10))
                        detail_file.flush()
                        print('Detail:', json.dumps(detail))
                        empty_windows = empty_windows + 1 if not detail['eligible_leaves'] else 0
                        if empty_windows == 3:
                            import warnings
                            warnings.warn('No eligible leaves in three windows; inspect visible_leaves and score_max in densification.jsonl')
                        tracker.reset()
                    if settings.adaptive_pool:
                        old_bytes = (pool.state.numel()*pool.state.element_size() + pool.scores.numel()*4) if pool.state is not None else 0
                        desired = choose_capacity(old_bytes)
                        if desired != pool.capacity:
                            pool.resize_empty(desired)

            else:
                if iteration < opt.iterations:
                    with profile.phase('adam'):
                        # Only XYZ has a schedule; retain all other rates and
                        # the vector's address instead of seven scalar writes.
                        rates[:3].fill_(lr_schedule(iteration) * opt.lr_multiplier)
                        replayed = (settings.graph_adam and adam_graph.step(
                            packet, grad, rates, iteration, frozen_prefix=g.skybox_points))
                        if not replayed:
                            packet.adam_step(grad, rates, iteration, frozen_prefix=g.skybox_points)
                del packet, grad
            ticket = next_ticket
            if settings.checkpoint_every and iteration > 0 and iteration < opt.iterations and iteration % settings.checkpoint_every == 0:
                with profile.phase('checkpoint'):
                    pool.flush()
                    save_checkpoint(output / 'resident_latest.pt', g, iteration, contract,
                                    seed, tracker, ema, empty_windows)
                    print(f'Resident checkpoint saved after iteration {iteration}', flush=True)
            if iteration % 10 == 0:
                hit = pool.stats['hit_rows'] / max(pool.stats['requested_rows'], 1)
                progress.set_postfix(loss=f'{ema:.6f}', active=active_count, total=g.size, hit=f'{hit:.1%}')
            progress.update(1)
            profile.finish(active=active_count, total=g.size, loss=scalar,
                allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), pool_capacity=pool.capacity, image_uploaded_bytes=transfer.uploaded_bytes,
                image_prefetch_uploads=transfer.prefetch_uploads, graph_captures=adam_graph.captures,
                graph_replays=adam_graph.replays, **pool.stats, **builder.stats)
    finally:
        # Join the CPU fetcher before shutting down DataLoader workers; retire
        # all DMA owners before invalidating or freeing their destination slots.
        try:
            views.close()
        finally:
            shutdown_camera_loader(loader)
            try:
                transfer.close()
            finally:
                adam_graph.reset()
                pool.close()
                profile.close()
                progress.close()
                detail_file.close()
    print(f'Resident v2 fine training: {time.perf_counter()-started:.2f} seconds')
