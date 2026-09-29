"""Resident v2: cached Gaussian training with LoD or explicit flat point storage."""
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch
from utils.incremental_spt import IncrementalSPT
from utils.camera_geometry import screen_scores, effective_focal
from utils.general_policy import DetailWindow, relative_spt_volume
from utils.resident_pool import capacity_for_budget, cuda_available_bytes
from utils.resident_pool_v2 import StreamingResidentPool
from utils.resident_selection import select_with_budget
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
    host_storage: str = 'ram'
    host_directory: str = ''
    max_active_nodes: int = 0
    representation: str = "lod"
    flat_direct: bool = False
    flat_native: bool = False

    def __post_init__(self):
        if self.representation not in ('lod', 'flat'):
            raise ValueError('representation must be lod or flat')
        if self.representation == 'flat' and self.max_active_nodes:
            raise ValueError('Flat training cannot use a LoD active-node budget; set max_active_nodes=0')
        if self.host_storage not in ('ram', 'mmap'):
            raise ValueError('host_storage must be ram or mmap')
        if self.host_storage == 'mmap' and not self.host_directory:
            raise ValueError('mmap storage requires an explicit SSD host_directory')
        if self.max_active_nodes < 0:
            raise ValueError('max_active_nodes must be nonnegative')
        if (self.pool_gib <= 0 or self.headroom_gib < 0 or self.transfer_rows <= 0
                or self.checkpoint_every < 0 or self.profile_every < 0 or self.gaussian_prefetch_rows < 0
                or self.image_prefetch_mib < 0 or self.image_cache_gib < 0 or self.graph_min_reuse < 1):
            raise ValueError("invalid resident budget/prefetch/graph options")
        if self.native_ops not in ("auto", "cuda", "torch"):
            raise ValueError("native_ops must be auto, cuda, or torch")


def validate_options(opt, runtime=None):
    settings = ResidentOptions(**(runtime or {}))
    if settings.representation == "flat" and opt.SH_degree != 1:
        raise ValueError("Flat migration currently supports SH degree 1 checkpoints")
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
        loss = loss + opt.lambda_dssim * (1 - fused_ssim(predicted[None], gt[None]))
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


def _grow_cpu_backing(g, pool, required, cap_max):
    """Grow only used CPU rows; the configured node limit is a logical ceiling."""
    current = len(g.properties)
    if required <= current:
        return
    capacity = min(cap_max, max(required, math.ceil(current * 1.25)))
    if capacity < required or pool.used or pool.active is not None or pool.dirty.any():
        raise RuntimeError('Grow resident backing only after flushing and invalidating the pool')
    properties = torch.zeros((capacity, g.properties.shape[1]), dtype=g.properties.dtype)
    properties[:g.size].copy_(g.properties[:g.size])
    g.properties = pool.host = properties
    nodes = torch.zeros((capacity, g.nodes.shape[1]), dtype=g.nodes.dtype)
    nodes[:g.size].copy_(g.nodes[:g.size])
    g.nodes = nodes
    scores = torch.zeros(capacity, dtype=g._densification_criterium.dtype)
    scores[:g.size].copy_(g._densification_criterium[:g.size])
    g._densification_criterium = pool.host_scores = scores
    pool.to_slot = np.full(capacity, -1, dtype=np.int64)
    print(f'Resident CPU backing grown from {current:,} to {capacity:,} slots', flush=True)


def training(dataset, opt, pipe, saving_iterations, view_graph=None, runtime=None,
             resume_checkpoint=None, allow_growth_resume=False, viewer_config=None):
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
    initial_capacity = opt.cap_max
    if settings.host_storage == 'ram' and getattr(opt, 'densify_max_new_nodes', 0) > 0:
        starting_size = len(g._xyz)
        if resume_checkpoint:
            preview = torch.load(resume_checkpoint, map_location='cpu', weights_only=True, mmap=True)
            starting_size = max(starting_size, int(preview['size']))
            del preview
        initial_capacity = min(opt.cap_max, starting_size + 2 * opt.densify_max_new_nodes)
    backing = None
    if settings.host_storage == 'mmap':
        from utils.resident_storage import MappedHostStorage
        backing = MappedHostStorage(settings.host_directory)
        print(f'Resident SSD backing: {backing.path}', flush=True)
    g.compact_gaussians('cpu', initial_capacity, densification='classic', prune_unused_gaussians=False,
                       allocator=backing.allocate if backing else None)
    print(f'Resident CPU backing: {initial_capacity:,} slots; limit: {opt.cap_max:,}', flush=True)
    from utils.resident_checkpoint import load_checkpoint, save_checkpoint
    contract = (dict(source=str(Path(dataset.source_path).resolve()), resolution=dataset.resolution,
                    hierarchy=str(Path(dataset.hierarchy).resolve()), options=vars(opt), pipeline=vars(pipe))
                if settings.checkpoint_every or resume_checkpoint or os.environ.get('YK_SAVE_REQUEST') else {})
    if settings.representation == 'flat':
        contract['representation'] = 'flat'
    restored = load_checkpoint(resume_checkpoint, g, contract, opt.iterations,
                               allow_growth=allow_growth_resume) if resume_checkpoint else None
    if settings.representation == 'flat' and not restored:
        raise ValueError('Flat mode requires an explicitly migrated flat checkpoint')
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
    if settings.representation == 'flat':
        from utils.flat_gaussians import FlatSelector
        builder = FlatSelector(use_frustum_culling=opt.use_frustum_culling)
    else:
        builder = IncrementalSPT(root_volume, opt.target_granularity_pixels / base_focal,
                                 opt.min_SPT_size, opt.use_bounding_spheres)
    builder.refresh(g)
    native = load_native(settings.native_ops)
    if settings.representation == 'flat' and settings.flat_native:
        builder.native = native
    width = g.properties.shape[1] // 3
    def choose_capacity(existing_bytes=0, released_store_bytes=0):
        allocated = torch.cuda.memory_allocated()
        # Retain space for observed transient render/backward allocations.
        # A new, denser view can still exceed this historical peak.
        transient = max(0, torch.cuda.max_memory_allocated() - allocated - released_store_bytes)
        headroom = max(settings.headroom_gib, (transient * 1.25 + 2**30) / 2**30)
        requested = opt.cap_max if settings.representation == 'flat' and settings.flat_direct else min(opt.cache_size, opt.cap_max)
        if settings.adaptive_pool:
            requested = min(requested, g.size + max(65536, g.size // 4))
        return capacity_for_budget(requested, width, cuda_available_bytes(existing_bytes),
                                   settings.pool_gib, headroom)
    def make_pool(capacity):
        if settings.representation == 'flat' and settings.flat_direct and native is not None and g.size <= capacity:
            from utils.direct_resident_pool import DirectResidentPool
            return DirectResidentPool(g.properties, g._densification_criterium, g.size,
                                      live_size=g.size, ops=native, transfer_rows=settings.transfer_rows)
        return StreamingResidentPool(g.properties, g._densification_criterium, capacity,
                                     transfer_rows=settings.transfer_rows, pin_staging=settings.pin_staging,
                                     ops=native, prefetch_rows=settings.gaussian_prefetch_rows)
    capacity = choose_capacity()
    pool = make_pool(capacity)
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
        scene_radius=g.spatial_lr_scale, spt_root_volume=root_volume, lod_base_focal=base_focal,
        host_backing=str(backing.path) if backing else None), indent=2), encoding='utf-8')
    print(f'Resident v2: {pool.capacity:,} slots; native={native is not None}; representation={settings.representation}')
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
        if settings.representation == 'flat':
            ticket.selected_multiplier = 1.0
            return builder.select(g, transfer.metadata(ticket))
        if ticket.multiplier is None:
            focal = effective_focal(ticket.cpu) if pixel_lod else ticket.cpu.focal_length
            ticket.multiplier = base_focal / focal
            if opt.vary_distance_multiplier and iteration % 10:
                ticket.multiplier *= float(1 + torch.rand((), generator=distance_rng).pow(4) * 5)
        if ticket.ids is None or ticket.epoch != builder.generation:
            ticket.ids, ticket.selected_multiplier = select_with_budget(
                g, transfer.metadata(ticket), opt, ticket.multiplier, native=native,
                max_active_nodes=settings.max_active_nodes)
            ticket.epoch = builder.generation
        return ticket.ids

    def preview_select(camera, capacity):
        if settings.representation == 'flat':
            ids = builder.select(g, camera)
            if len(ids) > capacity:
                raise RuntimeError('Preview exceeds resident pool capacity; zoom closer')
            return ids
        focal = effective_focal(camera) if pixel_lod else camera.focal_length
        budget = min(capacity, settings.max_active_nodes) if settings.max_active_nodes > 0 else capacity
        return select_with_budget(g, camera, opt, base_focal / focal, native=native,
                                  max_active_nodes=budget)[0]

    from utils.initial_viewer import start_initial_viewer
    viewer = None
    physical_vram = torch.cuda.mem_get_info()[1]
    try:
        viewer = start_initial_viewer(viewer_config, g, pipe, background,
                                      pool=lambda: pool, select=preview_select,
                                      before_render=adam_graph.reset)
        for iteration in range(first_iteration, opt.iterations+1):
            profile.begin(iteration)
            with profile.phase('allocator_trim'):
                # Variable-size views can leave large unused cached blocks after
                # densification ends too. Release them before WDDM starts paging.
                reserved = torch.cuda.memory_reserved()
                if (reserved > physical_vram - settings.headroom_gib * 2**30
                        and reserved - torch.cuda.memory_allocated() > 2**30):
                    torch.cuda.empty_cache()
            with profile.phase('data_wait'):
                if ticket is None:
                    ticket = transfer.submit(views.pop(), speculative=False)
            with profile.phase('camera_upload'):
                camera = transfer.ready(ticket)
            with profile.phase('hierarchy_cut'):
                ids = select(ticket, iteration)
                selected_multiplier = ticket.selected_multiplier
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
                        if not split and settings.gaussian_prefetch_rows and settings.representation != 'flat':
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
            if iteration < opt.iterations and split:
                with profile.phase('densify_rebuild'):
                    released_store_bytes = sum(t.numel()*t.element_size() for t in (pool.state, pool.scores) if t is not None)
                    pool.flush()
                    adam_graph.reset()
                    pool.invalidate()
                    del packet, grad
                    old_size = g.size
                    detail = tracker.report(g, opt.densify_grad_threshold, iteration) if tracker else {}
                    dead = g.properties[:old_size, 13] <= math.log(.005/.995)
                    if getattr(opt, 'densify_max_new_nodes', 0) > 0:
                        needed = old_size + min(opt.densify_max_new_nodes, opt.cap_max - old_size)
                        _grow_cpu_backing(g, pool, needed, opt.cap_max)
                    if settings.representation == 'flat':
                        from utils.flat_gaussians import split_flat
                        split_flat(g, opt)
                    else:
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
                        detail.update(new_nodes=g.size-old_size, split_parents=(g.size-old_size) if settings.representation == 'flat' else (g.size-old_size)//2,
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
                    if settings.flat_direct and settings.representation == 'flat':
                        # The released resident store is not render workspace.
                        desired = choose_capacity(released_store_bytes=released_store_bytes)
                        counters = dict(pool.stats)
                        pool.close()
                        pool = make_pool(desired)
                        pool.stats.update(counters)
                    elif settings.adaptive_pool:
                        old_bytes = (pool.state.numel()*pool.state.element_size() + pool.scores.numel()*4) if pool.state is not None else 0
                        desired = choose_capacity(old_bytes)
                        if desired != pool.capacity:
                            pool.resize_empty(desired)

            else:
                if iteration <= opt.iterations:
                    with profile.phase('adam'):
                        # Only XYZ has a schedule; retain all other rates and
                        # the vector's address instead of seven scalar writes.
                        rates[:3].fill_(lr_schedule(iteration) * opt.lr_multiplier)
                        replayed = (settings.graph_adam and adam_graph.step(
                            packet, grad, rates, iteration, frozen_prefix=g.skybox_points))
                        if not replayed:
                            packet.adam_step(grad, rates, iteration, frozen_prefix=g.skybox_points)
                        if settings.representation == 'flat':
                            builder.update(packet.ids, packet.state)
                del packet, grad
            ticket = next_ticket
            if iteration in saving_iterations or iteration == opt.iterations:
                with profile.phase('save'):
                    pool.flush()
                    filename = opt.output_file_name if iteration == opt.iterations else f'iteration_{iteration}'
                    if settings.representation == 'flat':
                        save_checkpoint(output / 'resident_latest.pt', g, iteration, contract,
                                        seed, tracker, ema, empty_windows)
                    else:
                        g.save_hierarchy(str(output), file_name=filename)
            from utils.training_control import checkpoint_requested, acknowledge_checkpoint
            requested = checkpoint_requested()
            if requested or (settings.checkpoint_every and iteration > 0 and (iteration == opt.iterations or iteration % settings.checkpoint_every == 0)):
                with profile.phase('checkpoint'):
                    pool.flush()
                    save_checkpoint(output / 'resident_latest.pt', g, iteration, contract,
                                    seed, tracker, ema, empty_windows)
                    acknowledge_checkpoint(output / 'resident_latest.pt', iteration, 'resident')
                    print(f'Resident checkpoint saved after iteration {iteration}', flush=True)
            if iteration % 10 == 0:
                hit = pool.stats['hit_rows'] / max(pool.stats['requested_rows'], 1)
                progress.set_postfix(loss=f'{ema:.6f}', active=active_count, total=g.size, hit=f'{hit:.1%}')
            progress.update(1)
            profile.finish(active=active_count, total=g.size, loss=scalar,
                allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), pool_capacity=pool.capacity, image_uploaded_bytes=transfer.uploaded_bytes,
                host_capacity=len(g.properties), host_storage=settings.host_storage,
                max_active_nodes=settings.max_active_nodes,
                selected_lod_multiplier=selected_multiplier,
                direct_resident=type(pool).__name__ == 'DirectResidentPool',
                image_prefetch_uploads=transfer.prefetch_uploads, graph_captures=adam_graph.captures,
                graph_replays=adam_graph.replays, **pool.stats, **builder.stats)
            if viewer is not None:
                viewer.poll(iteration)
    finally:
        if viewer is not None:
            viewer.close()
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
                if backing:
                    backing.flush()
                profile.close()
                progress.close()
                detail_file.close()
    print(f'Resident v2 fine training: {time.perf_counter()-started:.2f} seconds')
