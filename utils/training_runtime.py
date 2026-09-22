"""Training I/O and explicit write-back of the out-of-core resident state."""
from typing import Optional, Sequence

import torch
from torch.utils.data import DataLoader


CAMERA_TENSORS = (
    "K_train",
    "original_image", "alpha_mask", "invdepthmap", "depth_mask",
    "world_view_transform", "projection_matrix", "full_proj_transform",
    "full_proj_transform_inverse", "camera_center",
)


def pin_camera_tensors(camera):
    """Called by DataLoader's pinning thread, not by dataset workers."""
    for name in CAMERA_TENSORS:
        value = getattr(camera, name, None)
        if isinstance(value, torch.Tensor) and value.device.type == "cpu":
            setattr(camera, name, value.pin_memory())
    return camera


def direct_collate(batch):
    return batch


def make_camera_loader(cameras, opt, *, shuffle: bool):
    workers = int(getattr(opt, "data_workers", 4))
    prefetch = int(getattr(opt, "data_prefetch_factor", 1))
    if workers < 0 or prefetch < 1:
        raise ValueError("data_workers must be >= 0; data_prefetch_factor must be >= 1")
    if len(cameras) == 0:
        raise ValueError("The training camera dataset is empty")
    cache_gib = float(getattr(opt, "coarse_image_cache_gib", 0.0))
    if cache_gib > 0:
        from utils.view_pipeline import CachedCameras
        cameras = CachedCameras(cameras, int(cache_gib * 2**30) // max(1, workers),
                                pin_cache=workers == 0 and torch.cuda.is_available(),
                                compact_images=getattr(opt, 'coarse_compact_images', False))
    kwargs = dict(
        batch_size=1, num_workers=workers, shuffle=shuffle,
        collate_fn=direct_collate,
        pin_memory=bool(getattr(opt, "pin_memory", True)),
    )
    if workers:
        kwargs.update(prefetch_factor=prefetch, persistent_workers=True)
    return DataLoader(cameras, **kwargs)


def shutdown_camera_loader(loader):
    # Do not call _get_iterator(): that can START a second set of workers.
    iterator = getattr(loader, "_iterator", None)
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if shutdown is not None:
        shutdown()


@torch.no_grad()
def flush_resident_state(
    gaussians,
    indices: torch.Tensor,
    active: Sequence[torch.Tensor],
    cached: Sequence[torch.Tensor],
    parameters: Sequence[dict],
    *,
    scores: Optional[Sequence[torch.Tensor]] = None,
    contributed: Optional[Sequence[torch.Tensor]] = None,
    chunk_size: int = 65_536,
) -> None:
    """Write active AND inactive GPU-cache rows back before save/densification.

    Property order is xyz, log-scale, quaternion, DC, opacity-logit, SH-rest;
    followed by Adam's first and second moments in the same order. The row
    order is active (including skybox), then inactive cached Gaussians.
    Transfers to CPU are deliberately synchronous: the caller consumes the
    destination immediately. Chunking avoids a full-size temporary cache copy.
    """
    if chunk_size <= 0 or len(active) != 6 or len(cached) != 6 or len(parameters) != 6:
        raise ValueError("Expected six property groups and a positive chunk_size")
    n_active, n_cached = active[0].shape[0], cached[0].shape[0]
    total = n_active + n_cached
    if indices.ndim != 1 or indices.numel() != total:
        raise ValueError("Resident indices must match active + cached row counts")
    for group, count in ((active, n_active), (cached, n_cached)):
        if any(t.shape[0] != count for t in group):
            raise ValueError("Inconsistent resident property row counts")
    for state in parameters:
        for name in ("exp_avgs", "exp_avgs_sqs"):
            if state[name].shape[0] != total:
                raise ValueError("Adam state must cover active + cached rows")
    for pair in (scores, contributed):
        if pair is not None and (len(pair) != 2 or pair[0].shape != (n_active,) or pair[1].shape != (n_cached,)):
            raise ValueError("Statistics must match active + cached rows")
    storage = gaussians.properties.device
    ids = indices.detach().to(device=storage, dtype=torch.long)
    if total == 0:
        return
    if int(ids.min()) < 0 or int(ids.max()) >= gaussians.size:
        raise ValueError("Resident index outside the initialized Gaussian range")
    for group, base, count, part in ((active, 0, n_active, 0), (cached, n_active, n_cached, 1)):
        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            n = stop - start
            lo, hi = base + start, base + stop
            fields = [t[start:stop].detach().reshape(n, -1) for t in group]
            for name in ("exp_avgs", "exp_avgs_sqs"):
                fields.extend(p[name][lo:hi].detach().reshape(n, -1) for p in parameters)
            rows = torch.cat(fields, dim=1)
            if rows.shape[1] != gaussians.properties.shape[1]:
                raise ValueError("Resident property/Adam layout does not match backing storage")
            gaussians.properties[ids[lo:hi]] = rows.to(storage)
            if scores is not None:
                gaussians._densification_criterium[ids[lo:hi]] = scores[part][start:stop].detach().to(storage)
            if contributed is not None:
                gaussians._contributed[ids[lo:hi]] = contributed[part][start:stop].detach().to(storage)
