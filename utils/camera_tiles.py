"""Native-pixel camera tiles, with an explicit loss core and rendering halo."""
from copy import copy
import hashlib
import math
import random

import torch

from utils.camera_geometry import camera_intrinsics


def crop_camera(camera, left, top, width, height, halo=8):
    """Copy a camera and crop a core rectangle, retaining a clipped pixel halo.

    ``tile_core`` is (left, top, width, height) inside the returned image;
    reduce pixel losses only over that core. Compute neighborhood losses such
    as SSIM on the halo image first, then crop their maps to the core.
    """
    values = (left, top, width, height, halo)
    if any(int(v) != v for v in values):
        raise ValueError('Tile coordinates must be integers')
    left, top, width, height, halo = map(int, values)
    sw, sh = int(camera.image_width), int(camera.image_height)
    if (min(left, top, halo) < 0 or min(width, height) <= 0
            or left + width > sw or top + height > sh):
        raise ValueError('Tile core must lie inside the source image')
    x0, y0 = max(0, left - halo), max(0, top - halo)
    x1, y1 = min(sw, left + width + halo), min(sh, top + height + halo)
    result = copy(camera)
    for name in ('original_image', 'alpha_mask', 'invdepthmap', 'depth_mask'):
        tensor = getattr(camera, name, None)
        if tensor is not None:
            if tensor.shape[-2:] != (sh, sw):
                raise ValueError(f'{name} dimensions do not match camera')
            # Own the cropped storage, so a small tile need not keep a full
            # CUDA image alive after its transfer ticket is released.
            setattr(result, name, tensor[..., y0:y1, x0:x1].clone())
    k = camera_intrinsics(camera).clone()
    k[0, 2] -= x0
    k[1, 2] -= y0
    result.K_train = k
    result.image_width, result.image_height = x1 - x0, y1 - y0
    result.resolution = (result.image_width, result.image_height)
    result.fx, result.fy = float(k[0, 0]), float(k[1, 1])
    result.cx, result.cy = float(k[0, 2]), float(k[1, 2])
    result.primx = result.cx / result.image_width
    result.primy = result.cy / result.image_height
    result.FoVx = 2 * math.atan(result.image_width / (2 * result.fx))
    result.FoVy = 2 * math.atan(result.image_height / (2 * result.fy))
    result.lod_focal_length = max(result.fx, result.fy)
    # Row-vector/transposed convention, matching getProjectionMatrix. The
    # principal point may be outside the crop, which is valid off-axis optics.
    view = camera.world_view_transform
    p = torch.zeros((4, 4), dtype=view.dtype, device=view.device)
    p[0, 0] = 2 * result.fx / result.image_width
    p[1, 1] = 2 * result.fy / result.image_height
    p[2, 0] = 2 * result.primx - 1
    p[2, 1] = 2 * result.primy - 1
    p[2, 2] = camera.zfar / (camera.zfar - camera.znear)
    p[3, 2] = -camera.zfar * camera.znear / (camera.zfar - camera.znear)
    p[2, 3] = 1
    result.projection_matrix = p
    result.full_proj_transform = view @ p
    result.full_proj_transform_inverse = result.full_proj_transform.inverse()
    result.tile_bounds = (x0, y0, x1 - x0, y1 - y0)
    result.tile_core = (left - x0, top - y0, width, height)
    result.tile_source_size = (sw, sh)
    return result


def choose_training_tile(camera, iteration, seed, tile_size=2048, halo=8, balanced=False):
    """Select one shuffled-grid tile with no repeated core within a cycle.

    ``iteration`` is a zero-based visit counter for THIS camera, not necessarily
    the global training iteration. Persist camera visit counters at checkpoints
    to preserve coverage when cameras are randomly sampled or training resumes.
    Core rectangles cover the complete image exactly once per cycle. Border
    cores can be smaller; callers must account for their area in sampling/loss
    weighting if optimizing a uniform full-image pixel objective.
    """
    if int(iteration) != iteration or iteration < 0:
        raise ValueError('iteration must be a nonnegative integer')
    if int(tile_size) != tile_size or tile_size <= 0:
        raise ValueError('tile_size must be a positive integer')
    tile_size = int(tile_size)
    w, h = int(camera.image_width), int(camera.image_height)
    nx, ny = math.ceil(w / tile_size), math.ceil(h / tile_size)
    cycle, offset = divmod(int(iteration), nx * ny)
    identity = getattr(camera, 'image_name', getattr(camera, 'uid', ''))
    key = f'{seed}:{identity}:{w}:{h}:{tile_size}:{cycle}'.encode('utf-8')
    order = list(range(nx * ny))
    random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], 'little')).shuffle(order)
    index = order[offset]
    col, row = index % nx, index // nx
    if balanced:
        x, y = col * w // nx, row * h // ny
        tw, th = (col + 1) * w // nx - x, (row + 1) * h // ny - y
    else:
        x, y = col * tile_size, row * tile_size
        tw, th = min(tile_size, w-x), min(tile_size, h-y)
    result = crop_camera(camera, x, y, tw, th, halo)
    result.tile_index, result.tile_count, result.tile_cycle = index, nx * ny, cycle
    return result
