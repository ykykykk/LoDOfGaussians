"""Match a target silhouette in crop mode; ignore masked pixels otherwise."""
import json
import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def alpha_weight():
    import math
    value = float(json.loads((Path(__file__).resolve().parents[1] / 'configs/mask_loss.json').read_text())['alpha_loss_weight'])
    if not math.isfinite(value) or value <= 0:
        raise ValueError('alpha_loss_weight must be finite and positive')
    return value


def mask_targets(image, premultiplied_target, mask, rendered_alpha, background, core=Ellipsis):
    """Input target is already RGB * mask, as loaded by Camera.

    Crop compares unmasked rendered RGB with the target composited on the same
    background, and supervises differentiable alpha directly. Never multiply
    rendered alpha by the target mask: background splats must receive gradients.
    """
    if mask is None:
        return image, premultiplied_target, image.new_zeros(())
    if os.environ.get('YK_MASK_MODE') != 'crop':
        return image * mask, premultiplied_target, image.new_zeros(())
    target = premultiplied_target + (1 - mask) * background.reshape(3, 1, 1)
    loss = (rendered_alpha[core] - mask[core]).abs().mean() * alpha_weight()
    return image, target, loss
