"""Atomic resident-v2 checkpoints, including Adam moments and split scores."""
import os
from pathlib import Path
import torch
import psutil


def _compact_alias(value):
    # torch.save otherwise serializes the entire capacity behind a prefix view.
    # Saving is synchronous at a flushed training barrier: no snapshot copy needed.
    value = value.detach().cpu()
    if not value.is_contiguous():
        raise ValueError('Checkpoint backing must be contiguous')
    return torch.from_numpy(value.numpy())


def save_checkpoint(path, g, iteration, contract, seed, tracker, ema, empty_windows):
    path = Path(path)
    state = dict(version=1, iteration=iteration, contract=contract, seed=seed,
        size=g.size, active_sh_degree=g.active_sh_degree,
        skybox_points=int(g.skybox_points), spatial_lr_scale=float(g.spatial_lr_scale),
        properties=_compact_alias(g.properties[:g.size]), nodes=_compact_alias(g.nodes[:g.size]),
        scores=_compact_alias(g._densification_criterium[:g.size]),
        seen=_compact_alias(tracker.seen[:g.size]) if tracker else None,
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
    if any(old.get(key) != new.get(key) for key in ('source', 'resolution', 'hierarchy', 'pipeline', 'representation')):
        return False
    a, b = old.get('options', {}), new.get('options', {})
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    allowed = {'iterations', 'cap_max', 'densify_max_new_nodes',
               'densify_until_iter', 'densification_interval'}
    if a.keys() != b.keys() or any(a[key] != b[key] for key in a.keys() - allowed):
        return False
    return (b['iterations'] >= a['iterations'] > iteration
            and b['cap_max'] > a['cap_max']
            and b['densify_max_new_nodes'] > a['densify_max_new_nodes']
            and iteration < b['densify_until_iter'] < b['iterations']
            and b['densification_interval'] > 0)


def load_checkpoint(path, g, contract, iterations, allow_growth=False):
    # Sequential read-ahead avoids thousands of tiny mapped faults on cold HDDs.
    # One bounded buffer, regardless of checkpoint size.
    file_bytes = Path(path).stat().st_size
    if 64 * 2**20 < file_bytes < psutil.virtual_memory().available // 2:
        with open(path, 'rb', buffering=0) as handle:
            buffer = bytearray(16 * 2**20)
            while handle.readinto(buffer):
                pass
        del buffer
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
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
    for key, destination in (('properties', g.properties), ('nodes', g.nodes),
                             ('scores', g._densification_criterium)):
        value = state.pop(key)
        for start in range(0, n, 65536):
            block = value[start:start + 65536]
            if block.is_floating_point() and not torch.isfinite(block).all():
                raise ValueError('Non-finite checkpoint tensor: ' + key)
            destination[start:start + len(block)].copy_(block)
        del value, block
    if changed:
        # A new split schedule starts a new score/visibility window. Adam and
        # model parameters still resume exactly from the source checkpoint.
        g._densification_criterium[:n].zero_()
        state['seen'], state['views'], state['empty_windows'] = None, 0, 0
    g.size = n
    for name in ('active_sh_degree', 'skybox_points', 'spatial_lr_scale'):
        setattr(g, name, state[name])
    return state
