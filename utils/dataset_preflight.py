"""Read-only calibration/image validation and reproducible input fingerprints."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image
from utils.camera_geometry import image_size


def inspect_dataset(root, images_dir=None, masks_dir=None, resolution=2, hold=100):
    from utils.read_write_model import read_cameras_binary, read_images_binary, read_cameras_text, read_images_text
    root = Path(root).resolve()
    sparse = root / 'sparse' / '0'
    images_dir = Path(images_dir or root / 'images').resolve()
    masks_dir = Path(masks_dir).resolve() if masks_dir else None
    binary = (sparse/'cameras.bin').is_file() and (sparse/'images.bin').is_file()
    ext = '.bin' if binary else '.txt'
    camera_file, image_file = sparse/('cameras'+ext), sparse/('images'+ext)
    cameras = (read_cameras_binary if binary else read_cameras_text)(str(camera_file))
    images = (read_images_binary if binary else read_images_text)(str(image_file))
    if not images or not cameras:
        raise ValueError('COLMAP model contains no registered images/cameras')
    for camera in cameras.values():
        if camera.model not in ('PINHOLE', 'SIMPLE_PINHOLE'):
            raise ValueError(f'{camera.model}: undistort images and calibration to PINHOLE/SIMPLE_PINHOLE first')
        params = np.asarray(camera.params)
        if camera.width <= 0 or camera.height <= 0 or not np.isfinite(params).all():
            raise ValueError('Invalid calibration dimensions or parameters')
        focal_count = 2 if camera.model == 'PINHOLE' else 1
        if np.any(params[:focal_count] <= 0):
            raise ValueError('Focal length must be positive')
    ordered = sorted(images.values(), key=lambda x:x.name)
    records, decoded, compact, masks, sizes = [], 0, 0, 0, []
    for i, image in enumerate(ordered):
        if image.camera_id not in cameras or not np.isfinite(image.tvec).all() or not np.isfinite(image.qvec).all():
            raise ValueError('Invalid image extrinsics or camera reference')
        camera = cameras[image.camera_id]
        path = (images_dir/image.name).resolve()
        if not path.is_relative_to(images_dir):
            raise ValueError('Image name leaves the dataset image directory')
        if not path.is_file():
            alternatives = [path.with_suffix('.jpg'), path.with_suffix('.png')]
            path = next((p for p in alternatives if p.is_file()), path)
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as source:
            w, h = source.size
            byte_rgb = source.mode == 'RGB'
        if abs(w/h - camera.width/camera.height) > 1e-3 * camera.width/camera.height:
            raise ValueError(f'Image/calibration aspect ratio mismatch: {image.name}')
        tw, th = image_size(w, h, resolution)
        test = hold > 0 and i % hold == 0
        if not test:
            decoded += tw * th * 16 + 1024
        sizes.append((tw, th))
        stat = path.stat()
        record = [str(path), stat.st_size, stat.st_mtime_ns, w, h]
        mask = None
        if masks_dir:
            mask = next((masks_dir/Path(image.name).with_suffix(suffix) for suffix in ('.png','.JPG')
                         if (masks_dir/Path(image.name).with_suffix(suffix)).is_file()), None)
            if mask is not None:
                with Image.open(mask) as source:
                    if source.size != (w,h):
                        raise ValueError(f'Mask/image size mismatch: {mask}')
                stat = mask.stat(); record += [str(mask),stat.st_size,stat.st_mtime_ns]; masks += 1
        if not test:
            # Conservative for alpha, actual external masks and non-RGB formats.
            compact += tw * th * (4 if byte_rgb and mask is None else 16) + 1024
        records.append(record)
    training_views = sum(not (hold > 0 and i % hold == 0) for i in range(len(ordered)))
    if not training_views:
        raise ValueError('Evaluation split leaves no training views')
    digest = hashlib.sha256()
    for path in (camera_file,image_file):
        with path.open('rb') as handle:
            for block in iter(lambda:handle.read(2**20), b''):
                digest.update(block)
    points = next((sparse/('points3D'+e) for e in ('.bin','.txt','.ply') if (sparse/('points3D'+e)).is_file()),None)
    if points is None:
        raise FileNotFoundError('Missing COLMAP sparse points3D model')
    stat = points.stat()
    digest.update(json.dumps([records,str(points),stat.st_size,stat.st_mtime_ns,resolution,hold]).encode())
    return dict(schema=2, input_fingerprint=digest.hexdigest(), images=len(ordered),
        training_views=training_views, test_views=len(ordered)-training_views,
        cameras=len(cameras), masks=masks, resolution=resolution,
        training_sizes=sorted(set(sizes)), decoded_training_bytes=decoded,
        compact_training_bytes=compact,
        model_format=ext, model_path=str(sparse), images_path=str(images_dir),
        caveat='Image metadata and calibration fingerprint; does not prove pose accuracy or detect same-aspect crops')


def scaffold_manifest(inspection, options, model, seed):
    # Hash resolved coarse options, not only hand-written keys present in JSON.
    return dict(schema=2, calibration_contract='training_pixels_v1',
        input_fingerprint=inspection['input_fingerprint'], seed=seed,
        model={k:getattr(model,k,None) for k in ('resolution','skybox_num','white_background','train_test_exp')},
        coarse={k:getattr(options,k,None) for k in ('coarse_iterations','llff_hold','lr_multiplier',
            'feature_lr','opacity_lr','scaling_lr','rotation_lr','lambda_dssim','coarse_fused_ssim',
            'exposure_lr_init','exposure_lr_final','exposure_lr_delay_steps','exposure_lr_delay_mult')})
