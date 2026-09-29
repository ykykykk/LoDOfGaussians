"""Conservative multi-view silhouette cropping into a separate block checkpoint."""
import copy
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class TrainingMaskCropper:
    """Cache mask projections; physically prune only at training safe points."""
    def __init__(self, source):
        self.config = json.loads((Path(__file__).resolve().parents[1] / 'configs/mask_crop.json').read_text())
        self.interval = int(self.config['training_interval'])
        if self.interval <= 0:
            raise ValueError('Mask training_interval must be positive')
        self.views = load_views(source, self.config)

    def scaffold(self, model):
        import torch
        count = len(model._xyz)
        keep = np.zeros(count, dtype=bool)
        for start in range(model.skybox_points, count, 65536):
            stop = min(start + 65536, count)
            rows = torch.cat((model._xyz[start:stop], model._scaling[start:stop]), 1).detach().cpu().numpy()
            keep[start:stop] = keep_points(rows, self.views, self.config)
        if int(keep.sum()) < 10:
            raise ValueError('Mask crop leaves fewer than ten foreground points required by the hierarchy builder; check masks and camera alignment.')
        removed = count - int(keep.sum())
        if removed:
            with torch.no_grad():
                model.prune_points(torch.as_tensor(~keep, device=model._xyz.device))
        print(f'Mask crop (scaffold): {count:,} -> {int(keep.sum()):,} points; removed {removed:,}', flush=True)
        return removed


def training_cropper(source):
    return TrainingMaskCropper(source) if os.environ.get('YK_MASK_MODE') == 'crop' else None


def load_views(source, config):
    from scene.dataset_readers import readColmapSceneInfo
    info = readColmapSceneInfo(str(source), 'images', 'masks', '', False, False, 100)
    cameras = info.train_cameras + info.test_cameras
    if not cameras:
        raise ValueError('Mask crop requires registered cameras.')
    views = []
    for c in cameras:
        with Image.open(c.mask_path or c.image_path) as image:
            if image.size != (c.width, c.height):
                raise ValueError('Mask dimensions do not match camera: ' + (c.mask_path or c.image_path))
            if c.mask_path:
                image = image.convert('L')
            elif 'A' in image.getbands() or 'transparency' in image.info:
                image = image.convert('RGBA').getchannel('A')
            else:
                raise ValueError('Crop mode needs a mask or image alpha channel: ' + c.image_path)
            image.thumbnail((config['mask_max_dimension'],) * 2, Image.Resampling.NEAREST)
            foreground = np.asarray(image) >= config['foreground_threshold']
        # Distance from each background pixel to the foreground silhouette.
        distance = cv2.distanceTransform((~foreground).astype(np.uint8), cv2.DIST_L2, 5)
        h, w = foreground.shape
        views.append((np.asarray(c.R), np.asarray(c.T),
                      w / (2 * np.tan(c.FovX / 2)), h / (2 * np.tan(c.FovY / 2)),
                      c.primx * w, c.primy * h, distance))
    return views


def keep_points(rows, views, config):
    observed = np.zeros(len(rows), dtype=np.int32)
    outside = np.zeros(len(rows), dtype=np.int32)
    # Keep points whose 3-sigma support can overlap a silhouette boundary.
    radius = 3 * np.exp(rows[:, 3:6].astype(np.float64)).max(axis=1)
    for rotation, translation, fx, fy, cx, cy, distance in views:
        xyz = rows[:, :3].astype(np.float64) @ rotation + translation
        h, w = distance.shape
        front = xyz[:, 2] > radius + 1e-6
        ids = np.flatnonzero(front)
        z = xyz[ids, 2]
        u = xyz[ids, 0] * fx / z + cx
        v = xyz[ids, 1] * fy / z + cy
        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        ids, u, v, z = ids[valid], u[valid], v[valid], z[valid]
        observed[ids] += 1
        projected_radius = (max(fx, fy) * radius[ids] / (z - radius[ids])
                            * (1 + np.maximum(np.abs((u-cx)/fx), np.abs((v-cy)/fy))))
        outside[ids] += distance[v.astype(int), u.astype(int)] > (
            projected_radius + config['boundary_margin_pixels'])
    remove = (outside >= config['minimum_outside_views']) & (
        outside >= config['outside_view_fraction'] * np.maximum(observed, 1))
    return ~remove


def crop_checkpoint(checkpoint, source, output):
    from utils.gaussian_block_store import GaussianBlockStore
    config = json.loads((Path(__file__).resolve().parents[1] / 'configs/mask_crop.json').read_text())
    if not (0 < config['outside_view_fraction'] <= 1 and config['minimum_outside_views'] >= 2
            and config['mask_max_dimension'] > 0 and 0 < config['foreground_threshold'] <= 255
            and config['boundary_margin_pixels'] >= 0):
        raise ValueError('Invalid mask crop configuration')
    original = GaussianBlockStore.open(checkpoint)
    if Path(original.metadata['contract']['source']).resolve() != Path(source).resolve():
        raise ValueError('Mask crop dataset does not match checkpoint')
    views = load_views(source, config)
    result = GaussianBlockStore.create(output, copy.deepcopy(original.metadata))
    before = kept = 0
    for block in original.blocks:
        rows = original.read(block['id'])
        before += len(rows)
        keep = np.zeros(len(rows), dtype=bool) if block.get('skybox') else keep_points(rows, views, config)
        if keep.any():
            result.append(np.array(rows[keep]), skybox=False,
                          scores=np.array(original.read_scores(block['id'])[keep]))
            kept += int(keep.sum())
        print(f'Mask crop: {before:,} examined, {kept:,} kept', flush=True)
    if not kept:
        raise ValueError('Mask crop would remove the entire model; check masks and camera alignment.')
    report = dict(input_points=before, kept_points=kept, removed_points=before-kept,
                  views=len(views), config=config, source_checkpoint=str(Path(checkpoint).resolve()))
    result.metadata['mask_crop'] = report
    manifest = result.checkpoint()
    (Path(output) / 'mask_crop.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return str(manifest), report
