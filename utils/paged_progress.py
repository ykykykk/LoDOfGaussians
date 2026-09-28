"""Separate optimizer updates from accumulated full-image pixel coverage."""
import math
from utils.camera_tiles import tile_rectangle
from utils.camera_geometry import image_size


def restore_progress(metadata, camera_infos, tile_size, balanced, seed):
    if 'image_equivalent_progress' in metadata:
        return float(metadata['image_equivalent_progress'])
    visits = metadata.get('camera_visits', {})
    # Old paged checkpoints counted every tile as an iteration. Preserve the
    # full-image history before migration, reconstruct only their tiled suffix.
    origin = int(metadata['iteration']) - sum(int(n) for n in visits.values())
    if origin < 0:
        raise ValueError('Camera visit counts exceed optimizer iteration')
    value = float(origin)
    infos = {str(c.image_name): c for c in camera_infos}
    resolution = metadata['contract']['resolution']
    for name, visits_count in visits.items():
        if name not in infos:
            raise ValueError(f'Saved camera is absent: {name}')
        info = infos[name]
        w, h = image_size(info.width, info.height, resolution, 1)
        count = math.ceil(w/tile_size)*math.ceil(h/tile_size)
        cycles, remainder = divmod(int(visits_count), count)
        value += cycles
        for offset in range(remainder):
            _, _, tw, th, *_ = tile_rectangle(w,h,name,cycles*count+offset,seed,tile_size,balanced)
            value += tw*th/(w*h)
    return value


def crossed_growth(previous, current, start, stop, interval):
    if interval <= 0:
        raise ValueError('Growth interval must be positive')
    boundary = (math.floor((previous+1e-8)/interval)+1)*interval
    return boundary <= current+1e-8 and start < boundary < stop


def maximum_updates(remaining_images, camera_infos, resolution, tile_size, balanced):
    fractions = []
    for c in camera_infos:
        w,h = image_size(c.width,c.height,resolution,1)
        nx,ny = math.ceil(w/tile_size),math.ceil(h/tile_size)
        mw,mh = (w//nx,h//ny) if balanced else (w-(nx-1)*tile_size,h-(ny-1)*tile_size)
        fractions.append(mw*mh/(w*h))
    return math.ceil(max(0,remaining_images)/min(fractions))+1
