"""Native-resolution tiled training of immutable, disk-paged flat Gaussian blocks."""
import argparse
import copy
import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import torch


def scheduled_camera(seed, group, camera_count):
    """Stateless group schedule: lookahead never consumes checkpoint RNG."""
    if camera_count <= 0 or group < 0:
        raise ValueError('Invalid camera schedule dimensions')
    digest = hashlib.sha256(f'paged-camera-v1:{seed}:{group}'.encode('ascii')).digest()
    return int.from_bytes(digest[:8], 'little') % camera_count


class CameraLookahead:
    """Single cache owner, current camera plus at most one future group."""
    def __init__(self, cameras, seed, start, end, tiles_per_camera):
        self.cameras, self.seed = cameras, seed
        self.tiles_per_camera = tiles_per_camera
        self.last_group = (end - 1) // tiles_per_camera
        self.group = None
        self.camera = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='paged-image')
        self.future_group = start // tiles_per_camera
        self.future = self.executor.submit(self._decode, self.future_group)

    def _decode(self, group):
        index = scheduled_camera(self.seed, group, len(self.cameras))
        return self.cameras[index]

    def get(self, zero_based_step):
        group = zero_based_step // self.tiles_per_camera
        if group != self.group:
            if group != self.future_group or self.future is None:
                raise RuntimeError('Camera lookahead requires sequential steps')
            self.camera = self.future.result()
            self.group = group
            self.future_group = group + 1
            self.future = (self.executor.submit(self._decode, group+1)
                           if group < self.last_group else None)
        return self.camera

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)


def fork_store(source, destination):
    from utils.gaussian_block_store import GaussianBlockStore
    source = GaussianBlockStore.open(source)
    destination = Path(destination)
    if (destination / 'manifest.json').exists():
        raise FileExistsError('Output already has a checkpoint; explicitly resume it into a new output directory')
    target = GaussianBlockStore.create(destination, copy.deepcopy(source.metadata))
    target.manifest['blocks'] = copy.deepcopy(source.blocks)
    target.blocks = target.manifest['blocks']
    for block in target.blocks:
        for key in ('file', 'scores'):
            name = block.get(key)
            if not name:
                continue
            dst = target.root / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source.root / name, dst)
            except OSError:
                shutil.copy2(source.root / name, dst)
    target.checkpoint()
    return target


def train(args):
    from fused_ssim import FusedSSIMMap
    from gaussian_renderer import render_gsplat
    from scene.dataset_readers import readColmapSceneInfo
    from utils.camera_utils import CameraDataset
    from utils.camera_tiles import choose_training_tile
    from utils.camera_geometry import screen_scores
    from utils.view_pipeline import CachedCameras, CameraTransfer
    from utils.resident_native import load_native
    from utils.paged_gaussian_pool import PagedGaussianPool, CapacityError
    from utils.block_densification import densify_blocks
    from utils.general_utils import get_expon_lr_func
    from train_resident import parameter_rates
    from utils.paged_progress import restore_progress, crossed_growth, maximum_updates
    from utils.paged_camera_prefetch import PagedCameraPrefetch

    config = json.loads(Path(args.config).read_text(encoding='utf-8-sig')) if args.config else {}
    runtime_override = config.get('paged', {})
    store = fork_store(args.checkpoint, args.output_dir)
    metadata = copy.deepcopy(store.metadata)
    runtime = dict(metadata.get('paged', {}))
    runtime.update(runtime_override)
    contract = metadata['contract']
    options = dict(contract['options'])
    options.update(config.get('options', {}))
    opt = SimpleNamespace(**options)
    pipe = SimpleNamespace(**contract['pipeline'])
    from train_resident_v2 import validate_options
    validate_options(opt, {'representation': 'flat'})
    if contract.get('representation') != 'flat' or metadata.get('representation', 'flat') not in ('flat', 'flat_blocks'):
        raise ValueError('Paged training requires an explicitly converted flat checkpoint')
    if runtime.get('depths') or config.get('depths') or runtime.get('use_depth'):
        raise ValueError('Paged training supports RGB supervision only')
    if runtime.get('representation', 'flat') != 'flat' or config.get('representation', 'flat') != 'flat':
        raise ValueError('Paged training does not support LoD representation')
    source = str(Path(args.source_path or contract['source']).resolve())
    if Path(source) != Path(contract['source']).resolve():
        raise ValueError('Source path differs from checkpoint contract')
    tile_size = int(runtime.get('tile_size', 2048))
    tiles_per_camera = int(runtime.get('tiles_per_camera', 4))
    if tile_size <= 0 or tiles_per_camera <= 0:
        raise ValueError('tile_size and tiles_per_camera must be positive')
    halo = int(runtime.get('halo', 8))
    if halo < 5:
        raise ValueError('SSIM requires a halo of at least 5 pixels')
    checkpoint_every = int(runtime.get('checkpoint_every', 1000))
    if checkpoint_every <= 0:
        raise ValueError('checkpoint_every must be positive')
    model = SimpleNamespace(source_path=source, images=runtime.get('images', 'images'),
        alpha_masks=runtime.get('alpha_masks', 'masks'), depths='', eval=True,
        train_test_exp=False, resolution=contract['resolution'], data_device='cpu')
    info = readColmapSceneInfo(source, model.images, model.alpha_masks, '', True, False, opt.llff_hold)
    camera_dataset = CameraDataset(info.train_cameras, model, 1, False)
    if not len(camera_dataset):
        raise ValueError('No training cameras')
    transfer = CameraTransfer()
    native = load_native('cuda')
    free_bytes, _ = torch.cuda.mem_get_info()
    usable_bytes = min(float(runtime.get('pool_gib', 8))*2**30,
                       free_bytes - float(runtime.get('headroom_gib', 6))*2**30)
    capacity = int(runtime.get('capacity_rows', max(0, usable_bytes) // 280))
    if opt.densify_until_iter <= 0:
        # Refinement cannot append pages; unused growth slots only steal render memory.
        capacity = min(capacity, len(store.blocks)*store.block_rows)
    pool = PagedGaussianPool(store, capacity, native)
    seed = int(metadata.get('seed', 0))
    if metadata.get('rng') is not None:
        torch.set_rng_state(metadata['rng'])
    else:
        torch.manual_seed(seed)
    if metadata.get('cuda_rng'):
        torch.cuda.set_rng_state_all(metadata['cuda_rng'])
    visits = dict(metadata.get('camera_visits', {}))
    balanced = bool(runtime.get('balanced_tiles', not bool(visits)))
    start = int(metadata['iteration'])
    camera_origin = int(runtime.get('camera_schedule_origin', 0))
    if camera_origin < 0 or camera_origin > start:
        raise ValueError('Camera schedule origin must precede the current update')
    image_progress = restore_progress(metadata, info.train_cameras, tile_size, balanced, seed)
    if image_progress >= opt.iterations:
        raise ValueError('Image coverage has reached the configured training endpoint')
    end = start + (args.steps if args.steps is not None else maximum_updates(
        opt.iterations-image_progress, info.train_cameras, model.resolution, tile_size, balanced))
    if end <= start:
        raise ValueError('Training must advance at least one step')
    schedule = get_expon_lr_func(lr_init=opt.position_lr_init*metadata['spatial_lr_scale'],
        lr_final=opt.position_lr_final*metadata['spatial_lr_scale'],
        lr_delay_mult=opt.position_lr_delay_mult, max_steps=opt.position_lr_max_steps)
    background = torch.zeros(3, device='cuda')
    contract['options'] = options
    sampling = dict(version=1, method='sha256-camera-groups', seed=seed,
        tiles_per_camera=tiles_per_camera, tile_size=tile_size, halo=halo, balanced_tiles=balanced,
        camera_schedule_origin=camera_origin,
        cameras=hashlib.sha256('\n'.join(str(c.image_name) for c in info.train_cameras).encode('utf-8')).hexdigest())
    previous_sampling = metadata.get('sampling_contract')
    if previous_sampling is not None:
        previous_sampling = dict(previous_sampling)
        previous_sampling.setdefault('balanced_tiles', False)
        previous_sampling.setdefault('camera_schedule_origin', 0)
    if metadata.get('camera_visits') and previous_sampling is None:
        old_runtime = metadata.get('paged', {})
        for name, value in (('tile_size', tile_size), ('halo', halo), ('tiles_per_camera', tiles_per_camera)):
            if name in old_runtime and old_runtime[name] != value:
                raise ValueError(f'Cannot change {name} with saved camera tile visits')
    if previous_sampling is not None and previous_sampling != sampling:
        raise ValueError('Sampling configuration differs from the checkpoint; preserve tile_size, halo, tiles_per_camera and camera order')
    metadata['sampling_contract'] = sampling
    metadata['paged'] = dict(runtime, tile_size=tile_size, halo=halo,
                             tiles_per_camera=tiles_per_camera, balanced_tiles=balanced,
                             resolved_capacity_rows=pool.capacity)
    lookahead = PagedCameraPrefetch(camera_dataset, seed, start-camera_origin, end-camera_origin, tiles_per_camera,
        cache_bytes=int(float(runtime.get('image_cache_gib', 2))*2**30),
        workers=int(runtime.get('decode_workers', 2)))
    log_path = Path(args.output_dir) / 'paged_profile.jsonl'
    started = time.perf_counter()

    def save(iteration):
        pool.flush()
        metadata.update(iteration=iteration, camera_visits=visits,
            image_equivalent_progress=image_progress, progress_clock_version=1,
            rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
            size=sum(b['count'] for b in store.blocks), representation='flat_blocks')
        store.checkpoint(metadata)

    print(json.dumps(dict(event='started', iteration=start, end=end, blocks=len(store.blocks),
                          image_equivalent_progress=image_progress,
                          points=sum(b['count'] for b in store.blocks), capacity_rows=pool.capacity)), flush=True)
    viewer = None
    try:
        if getattr(args, 'viewer', False):
            from utils.realtime_viewer import PagedViewRenderer, RealtimeViewer
            viewer = RealtimeViewer(PagedViewRenderer(pool, pipe), args.viewer_port)
            print(f'Realtime viewer: {viewer.url}', flush=True)
        with log_path.open('a', encoding='utf-8') as log:
            for iteration in range(start+1, end+1):
                torch.cuda.synchronize()
                tick = time.perf_counter()
                camera = lookahead.get(iteration-1-camera_origin)
                key = str(camera.image_name)
                tile = choose_training_tile(camera, visits.get(key, 0), seed, tile_size, halo, balanced=balanced)
                tile = transfer.ready(transfer.submit(tile, speculative=False))
                torch.cuda.synchronize()
                after_data = time.perf_counter()
                candidates = pool.candidate_blocks(tile)
                try:
                    packet = pool.acquire(candidates, tile)
                except CapacityError as exc:
                    raise RuntimeError(f'Camera {key} tile {tile.tile_index} requires {exc.required_rows} rows; '
                        f'cache holds {exc.capacity_rows}. Reduce paged.tile_size or increase capacity_rows; no points were dropped.') from exc
                torch.cuda.synchronize()
                after_load = time.perf_counter()
                raw = packet.parameters()
                if not len(raw):
                    raise RuntimeError('Tile contains no Gaussian points')
                pkg = render_gsplat(tile, raw[:, :3].contiguous(), raw[:, 13:14].sigmoid(),
                    raw[:, 3:6].exp(), torch.nn.functional.normalize(raw[:, 6:10], dim=1),
                    raw[:, 10:13, None].transpose(1, 2), raw[:, 14:].reshape(len(raw), -1, 3),
                    pipe, background, sh_degree=1)
                image, gt = pkg['render'], tile.original_image
                x, y, w, h = tile.tile_core
                core = (..., slice(y, y+h), slice(x, x+w))
                predicted = image if tile.alpha_mask is None else image*tile.alpha_mask
                l1 = (predicted[core]-gt[core]).abs().mean()
                smap = FusedSSIMMap.apply(.01**2, .03**2, predicted[None].contiguous(), gt[None].contiguous(), 'same', True, 2)
                loss = (1-opt.lambda_dssim)*l1 + opt.lambda_dssim*(1-smap[core].mean())
                # Equal tile visits with area weights recover a uniform pixel objective.
                sw, sh = tile.tile_source_size
                previous_progress = image_progress
                next_progress = previous_progress + w*h/(sw*sh)
                area_weight = tile.tile_count*w*h/(sw*sh)
                objective = loss*area_weight
                torch.cuda.synchronize()
                after_render = time.perf_counter()
                objective.backward()
                if not torch.isfinite(loss).item() or raw.grad is None or not torch.isfinite(raw.grad).all().item():
                    raise RuntimeError(f'Non-finite loss or gradient at {iteration}')
                screen = pkg['viewspace_points'].grad
                if screen is None:
                    raise RuntimeError('Renderer did not retain screen gradients')
                # The sampled objective scales pixel gradients by tile_count.
                # Undo that only for growth scores, retaining full-image threshold units.
                packet.accumulate_scores(pkg['packed_indices'],
                    screen_scores(screen, sw, sh, opt.densify_score_space) / tile.tile_count)
                torch.cuda.synchronize()
                after_backward = time.perf_counter()
                packet.adam_step(raw.grad, parameter_rates(opt, schedule(next_progress), 23, 'cuda'), iteration)
                pool.finish_step(packet)
                visits[key] = visits.get(key, 0)+1
                image_progress = next_progress
                torch.cuda.synchronize()
                after_adam = time.perf_counter()
                growth = None
                if crossed_growth(previous_progress, image_progress, opt.densify_from_iter,
                                  opt.densify_until_iter, opt.densification_interval):
                    if runtime.get('growth_backend', 'resident') == 'flush':
                        pool.clear()
                        growth = densify_blocks(store, opt)
                        pool.refresh_metadata()
                    else:
                        growth = pool.densify(opt)
                final_step = iteration == end or image_progress >= opt.iterations
                from utils.training_control import checkpoint_requested, acknowledge_checkpoint
                requested = checkpoint_requested()
                saved = requested or iteration % checkpoint_every == 0 or final_step or growth is not None
                if saved:
                    save(iteration)
                    acknowledge_checkpoint(store.root / "manifest.json", iteration, "paged")
                elapsed = time.perf_counter()-tick
                record = dict(iteration=iteration, image_equivalent_progress=image_progress,
                    loss=float(loss.detach()), tile_steps_per_second=1/elapsed,
                    core_pixels=w*h, rendered_pixels=tile.image_width*tile.image_height,
                    core_megapixels_per_second=w*h/1e6/elapsed, camera=key, tile=tile.tile_index,
                    visible_points=len(raw), candidate_blocks=len(candidates), total_points=sum(b['count'] for b in store.blocks),
                    data_s=after_data-tick, load_s=after_load-after_data, render_loss_s=after_render-after_load,
                    backward_s=after_backward-after_render, adam_bounds_s=after_adam-after_backward,
                    growth_checkpoint_s=time.perf_counter()-after_adam, elapsed_s=elapsed,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(), checkpoint=saved, growth=growth,
                    cache=dict(pool.stats))
                log.write(json.dumps(record, allow_nan=False)+'\n'); log.flush()
                if iteration == start+1 or iteration % 10 == 0 or saved:
                    print(json.dumps(record, allow_nan=False), flush=True)
                del packet, raw, pkg, image, gt, predicted, smap, loss, objective, tile
                if viewer is not None:
                    viewer.poll(iteration)
                if final_step:
                    break
        if args.export_ply:
            store.export_ply(args.export_ply)
        print(json.dumps(dict(event='completed', iteration=iteration, image_equivalent_progress=image_progress,
                              elapsed_s=time.perf_counter()-started,
                              checkpoint=str(store.root/'manifest.json'))), flush=True)
    finally:
        if viewer is not None:
            viewer.close()
        lookahead.close()
        transfer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--source-path')
    parser.add_argument('--config')
    parser.add_argument('--steps', type=int, help='Maximum optimizer/tile updates for this run; iterations in config counts image coverage')
    parser.add_argument('--export-ply')
    parser.add_argument('--viewer', action='store_true', help='Enable live browser preview between training steps')
    parser.add_argument('--viewer-port', type=int, default=8765)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
