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


def _growth_contract_matches(old, new, iteration):
    if not isinstance(old, dict) or not isinstance(new, dict):
        return False
    if any(old.get(key) != new.get(key) for key in ('source', 'resolution', 'hierarchy', 'pipeline')):
        return False
    a, b = old.get('options', {}), new.get('options', {})
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    allowed = {'iterations', 'cap_max', 'densify_max_new_nodes',
               'densify_until_iter', 'densification_interval'}
    if a.keys() != b.keys() or any(a[key] != b[key] for key in a.keys() - allowed):
        return False
    return (b['iterations'] > a['iterations'] > iteration
            and b['cap_max'] > a['cap_max']
            and b['densify_max_new_nodes'] > a['densify_max_new_nodes']
            and iteration < b['densify_until_iter'] < b['iterations']
            and b['densification_interval'] > 0)


def load_checkpoint(path, g, contract, iterations, allow_growth=False):
    state = torch.load(path, map_location='cpu', weights_only=True)
    changed = state.get('contract') != contract
    if state.get('version') != 1 or (changed and not (allow_growth and
            _growth_contract_matches(state['contract'], contract, state['iteration']))):
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
    if changed:
        # A new split schedule starts a new score/visibility window. Adam and
        # model parameters still resume exactly from the source checkpoint.
        g._densification_criterium[:n].zero_()
        state['seen'], state['views'], state['empty_windows'] = None, 0, 0
    g.size = n
    for name in ('active_sh_degree', 'skybox_points', 'spatial_lr_scale'):
        setattr(g, name, state[name])
    return state
