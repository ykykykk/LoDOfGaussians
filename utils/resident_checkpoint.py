"""Atomic resident-v2 checkpoints, including Adam moments and split scores."""
import os
from pathlib import Path
import torch


def save_checkpoint(path, g, iteration, contract, seed, tracker, ema, empty_windows):
    path = Path(path)
    state = dict(version=1, iteration=iteration, contract=contract, seed=seed,
        size=g.size, active_sh_degree=g.active_sh_degree,
        skybox_points=int(g.skybox_points), spatial_lr_scale=float(g.spatial_lr_scale),
        properties=g.properties[:g.size].clone(), nodes=g.nodes[:g.size].clone(),
        scores=g._densification_criterium[:g.size].clone(),
        seen=tracker.seen[:g.size].cpu() if tracker else None,
        views=tracker.views if tracker else 0, ema=ema, empty_windows=empty_windows,
        rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_checkpoint(path, g, contract, iterations):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('version') != 1 or state['contract'] != contract:
        raise ValueError('Checkpoint dataset/options differ from this run')
    n = state['size']
    if not 0 < n <= len(g.properties) or not 0 <= state['iteration'] < iterations:
        raise ValueError('Checkpoint capacity or iteration is incompatible')
    for key, destination in (('properties', g.properties), ('nodes', g.nodes),
                             ('scores', g._densification_criterium)):
        value = state[key]
        if value.shape != destination[:n].shape or value.dtype != destination.dtype:
            raise ValueError('Invalid checkpoint tensor: ' + key)
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError('Non-finite checkpoint tensor: ' + key)
    for key, destination in (('properties', g.properties), ('nodes', g.nodes),
                             ('scores', g._densification_criterium)):
        destination[:n].copy_(state.pop(key))
    g.size = n
    for name in ('active_sh_degree', 'skybox_points', 'spatial_lr_scale'):
        setattr(g, name, state[name])
    return state
