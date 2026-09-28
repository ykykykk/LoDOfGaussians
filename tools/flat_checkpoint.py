"""Explicit LoD-to-flat migration and portable Gaussian PLY export."""
import argparse
import json
import os
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def atomic_save(state, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    with temporary.open('wb') as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)


def migrate(source, output, config):
    from utils.flat_gaussians import compact_leaf_state
    if Path(source).resolve() == Path(output).resolve() or Path(output).exists():
        raise ValueError('Migration requires a new output file; preserve the source checkpoint')
    state = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
    data = json.loads(Path(config).read_text(encoding='utf-8-sig'))
    if data.get('resident', {}).get('representation') != 'flat':
        raise ValueError('Use a flat representation config')
    state = compact_leaf_state(state)
    # Explicit migration is the only place where representation and split units change.
    # All unrelated optimization and dataset settings remain exactly the same.
    allowed = {'cap_max', 'densify_max_new_nodes', 'densification_interval',
               'densify_until_iter', 'iterations'}
    for key, value in data.items():
        if key in state['contract']['options']:
            old = state['contract']['options'][key]
            if value != old and key not in allowed:
                raise ValueError(f'Migration cannot change unrelated option {key}: {old} -> {value}')
            state['contract']['options'][key] = value
    state['contract']['representation'] = 'flat'
    state['migration'] = {'source': str(Path(source).resolve()), 'iteration': state['iteration']}
    if state['size'] > state['contract']['options']['cap_max']:
        raise ValueError('Point cap is smaller than the migrated point set')
    atomic_save(state, output)
    print(f"Flat checkpoint: step {state['iteration']}, {state['size']:,} rows including {state['skybox_points']:,} background rows", flush=True)


def export_checkpoint(source, output):
    from tools.export_ply import finest_leaf_indices, export_gaussian_tensors_ply
    state = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
    p = state['properties']
    width = p.shape[1] // 3
    selected = finest_leaf_indices(state['nodes'])
    export_gaussian_tensors_ply(p[:, :3], p[:, 10:13], p[:, 14:width], p[:, 13:14],
                               p[:, 3:6], p[:, 6:10], Path(output), indices=selected)
    print(f"Exported step {state['iteration']}: {len(selected):,} points to {output}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['migrate', 'export'])
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config')
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.action == 'migrate':
        if not args.config:
            parser.error('migrate requires --config')
        migrate(args.input, args.output, args.config)
    else:
        export_checkpoint(args.input, args.output)
