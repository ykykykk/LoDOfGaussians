"""Read-only, full native-pixel heldout evaluation of a block checkpoint.

Tiles are a memory bound, not an image sampling scheme: every pixel appears
once. SSIM is computed with a halo before reducing over each tile core.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def tile_rectangles(width, height, size):
    if min(width, height, size) <= 0:
        raise ValueError('Image dimensions and tile size must be positive')
    for top in range(0, height, size):
        for left in range(0, width, size):
            yield left, top, min(size, width-left), min(size, height-top)


def metrics(squared_error, ssim_sum, pixels):
    if pixels <= 0:
        raise ValueError('Cannot score zero pixels')
    mse = squared_error / (3*pixels)
    return dict(pixels=pixels, mse=mse, psnr=(-10*math.log10(mse) if mse else None),
                ssim=ssim_sum/(3*pixels), perfect_match=(mse == 0))


def evaluate(args):
    import torch
    from fused_ssim import FusedSSIMMap
    from gaussian_renderer import render_gsplat
    from scene.dataset_readers import readColmapSceneInfo
    from utils.camera_utils import CameraDataset
    from utils.camera_tiles import crop_camera
    from utils.view_pipeline import CachedCameras, CameraTransfer
    from utils.gaussian_block_store import GaussianBlockStore
    from utils.resident_native import load_native
    from utils.paged_gaussian_pool import PagedGaussianPool

    if args.halo < 5 or args.camera_limit <= 0 or args.pool_gib <= 0:
        raise ValueError('Require halo >= 5, positive camera-limit and pool-gib')
    store = GaussianBlockStore.open(args.checkpoint)
    contract = store.metadata['contract']
    runtime = store.metadata.get('paged', {})
    source = contract['source']
    model = SimpleNamespace(source_path=source, images=runtime.get('images', 'images'),
        alpha_masks=runtime.get('alpha_masks', 'masks'), depths='', eval=True,
        train_test_exp=False, resolution=1, data_device='cpu')
    info = readColmapSceneInfo(source, model.images, model.alpha_masks, '', True, False,
                               contract['options']['llff_hold'])
    selected = sorted(info.test_cameras, key=lambda camera: str(camera.image_name))[:args.camera_limit]
    if not selected:
        raise ValueError('Checkpoint source has no heldout cameras')
    cameras = CachedCameras(CameraDataset(selected, model, 1, True),
                            max_bytes=0, compact_images=True)
    transfer = CameraTransfer()
    native = load_native('cuda')
    free_bytes, _ = torch.cuda.mem_get_info()
    capacity = int(min(args.pool_gib*2**30, free_bytes-6*2**30)//280)
    if capacity < store.block_rows:
        raise ValueError('Insufficient free GPU memory for the requested evaluation cache')
    pool = PagedGaussianPool(store, capacity, native)
    pipe = SimpleNamespace(**contract['pipeline'])
    background = torch.zeros(3, device='cuda')
    views = []
    total_error = total_ssim = 0.0
    total_pixels = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for i in range(len(cameras)):
            camera = cameras[i]
            preview = None
            if getattr(args, 'preview_dir', None):
                rw, rh = min(1024, camera.image_width), min(1024, camera.image_height)
                rx, ry = (camera.image_width-rw)//2, (camera.image_height-rh)//2
                preview = torch.empty((2, 3, rh, rw), dtype=torch.float32)
            error = ssim = 0.0
            pixels = tiles = 0
            view_start = time.perf_counter()
            for left, top, width, height in tile_rectangles(camera.image_width, camera.image_height, args.tile_size):
                tile = transfer.ready(transfer.submit(crop_camera(camera, left, top, width, height, args.halo),
                                                       speculative=False))
                packet = pool.acquire(pool.candidate_blocks(tile), tile)
                raw = packet.state
                if len(raw):
                    rendered = render_gsplat(tile, raw[:, :3].contiguous(), raw[:, 13:14].sigmoid(),
                        raw[:, 3:6].exp(), torch.nn.functional.normalize(raw[:, 6:10], dim=1),
                        raw[:, 10:13, None].transpose(1, 2), raw[:, 14:].reshape(len(raw), -1, 3),
                        pipe, background, sh_degree=1)
                    image, rendered_alpha = rendered['render'], rendered['alpha']
                else:
                    image = torch.zeros_like(tile.original_image)
                    rendered_alpha = torch.zeros_like(image[:1])
                gt = tile.original_image
                x, y, w, h = tile.tile_core
                core = (..., slice(y, y+h), slice(x, x+w))
                from utils.mask_loss import mask_targets
                prediction, gt, _ = mask_targets(image, gt, tile.alpha_mask, rendered_alpha, background, core)
                if preview is not None:
                    # Only disjoint tile cores contribute: no duplicated halo,
                    # interpolation or full-frame host assembly.
                    ix0, iy0 = max(left, rx), max(top, ry)
                    ix1, iy1 = min(left+w, rx+rw), min(top+h, ry+rh)
                    if ix1 > ix0 and iy1 > iy0:
                        src = (..., slice(y+iy0-top, y+iy1-top), slice(x+ix0-left, x+ix1-left))
                        dst = (..., slice(iy0-ry, iy1-ry), slice(ix0-rx, ix1-rx))
                        preview[0][dst] = gt[src].cpu()
                        preview[1][dst] = prediction[src].cpu()
                error += (prediction[core]-gt[core]).square().double().sum().item()
                ssim_map = FusedSSIMMap.apply(.01**2, .03**2, prediction[None].contiguous(),
                                             gt[None].contiguous(), 'same', False, 2)
                ssim += ssim_map[core].double().sum().item()
                pixels += w*h
                tiles += 1
                pool.finish_step(packet)
            result = dict(camera=str(camera.image_name), width=camera.image_width, height=camera.image_height,
                          tiles=tiles, seconds=time.perf_counter()-view_start, **metrics(error, ssim, pixels))
            if preview is not None:
                from PIL import Image, ImageDraw
                preview_dir = Path(args.preview_dir)
                preview_dir.mkdir(parents=True, exist_ok=True)
                pair = Image.new('RGB', (rw*2, rh+24), 'black')
                for side in range(2):
                    rgb = (preview[side].clamp(0, 1)*255).round().to(torch.uint8).permute(1, 2, 0).numpy()
                    pair.paste(Image.fromarray(rgb), (side*rw, 24))
                draw = ImageDraw.Draw(pair)
                draw.text((8, 5), 'GT - native center crop', fill='white')
                draw.text((rw+8, 5), 'Render - native center crop', fill='white')
                preview_file = preview_dir / f'{i:02d}_{Path(str(camera.image_name)).stem}_center.png'
                pair.save(preview_file)
                result.update(preview_file=str(preview_file.resolve()), preview_roi=[rx, ry, rw, rh])
            views.append(result)
            total_error += error
            total_ssim += ssim
            total_pixels += pixels
            print(json.dumps(result), flush=True)
    if pool.dirty:
        raise RuntimeError('Read-only evaluation unexpectedly modified the GPU model')
    result = dict(checkpoint=str(Path(args.checkpoint).resolve()), iteration=store.metadata['iteration'],
        points=sum(block['count'] for block in store.blocks), source=source, views=views,
        aggregate=metrics(total_error, total_ssim, total_pixels),
        settings=dict(resolution=1, tile_size=args.tile_size, halo=args.halo, camera_limit=args.camera_limit,
            camera_selection='heldout image_name ascending', capacity_rows=pool.capacity,
            psnr='pixel-weighted RGB MSE; crop compares composited RGB, ignore masks the prediction',
            ssim='pixel-weighted RGB map, halo then core; same mask convention as training'),
        seconds=time.perf_counter()-started, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        cache_stats=pool.stats)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--camera-limit', type=int, default=3)
    parser.add_argument('--tile-size', type=int, default=2048)
    parser.add_argument('--halo', type=int, default=8)
    parser.add_argument('--pool-gib', type=float, default=8)
    parser.add_argument('--preview-dir', help='Save one native center 1024x1024 GT/render pair per view')
    evaluate(parser.parse_args())
