"""Camera geometry in training-image pixels; independent of CUDA extensions."""
import math
import torch


def image_size(width, height, resolution=-1, scale=1.0):
    if width <= 0 or height <= 0 or scale <= 0:
        raise ValueError('Image dimensions and resolution scale must be positive')
    if resolution in (1, 2, 4, 8):
        return max(1, round(width / (resolution * scale))), max(1, round(height / (resolution * scale)))
    if resolution == -1:
        factor = max(1.0, width / 1600.0) * scale
    elif resolution > 0:
        factor = width / float(resolution) * scale
    else:
        raise ValueError('resolution must be -1, a downsample factor, or a positive width')
    return max(1, int(width / factor)), max(1, int(height / factor))


def intrinsics(width, height, fovx, fovy, principal_x=0.5, principal_y=0.5):
    values = (width, height, fovx, fovy, principal_x, principal_y)
    if not all(math.isfinite(float(x)) for x in values):
        raise ValueError('Camera calibration contains non-finite values')
    if width <= 0 or height <= 0 or not (0 < fovx < math.pi and 0 < fovy < math.pi):
        raise ValueError('Invalid image size or field of view')
    return (width / (2 * math.tan(fovx / 2)), height / (2 * math.tan(fovy / 2)),
            width * principal_x, height * principal_y)


def camera_intrinsics(camera, device=None):
    cached = getattr(camera, 'K_train', None)
    if cached is not None:
        return cached.to(device=device) if device is not None else cached
    fx, fy, cx, cy = intrinsics(camera.image_width, camera.image_height,
        camera.FoVx, camera.FoVy, getattr(camera, 'primx', 0.5), getattr(camera, 'primy', 0.5))
    return torch.tensor([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]],
                        dtype=torch.float32, device=device)


def effective_focal(camera):
    # Conservative under rectangular pixels / anisotropic resizing.
    value = getattr(camera, 'lod_focal_length', None)
    if value is None:
        fx, fy, _, _ = intrinsics(camera.image_width, camera.image_height,
            camera.FoVx, camera.FoVy)
        value = max(fx, fy)
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError('Invalid effective focal length')
    return float(value)


def screen_scores(gradient, width, height, space='pixel'):
    if gradient.ndim != 2 or gradient.shape[1] != 2 or width <= 0 or height <= 0:
        raise ValueError('Expected means2d gradient [N,2] and positive image dimensions')
    value = gradient.detach()
    if space == 'ndc':
        # Chain rule: dL/dx_ndc = dL/dx_pixel * W/2.
        value = value * value.new_tensor([width / 2., height / 2.])
    elif space != 'pixel':
        raise ValueError('score_space must be pixel or ndc')
    return torch.linalg.vector_norm(value, dim=-1)
