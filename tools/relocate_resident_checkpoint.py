"""Copy a resident checkpoint after a dataset move, verifying its scaffold fingerprint."""
import argparse
import json
import os
from pathlib import Path

import torch

from utils.dataset_preflight import inspect_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'output', 'old-root', 'new-root', 'manifest'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve() == args.input.resolve():
        raise ValueError('Choose a new output checkpoint; the original must be preserved')
    state = torch.load(args.input, weights_only=True, map_location='cpu', mmap=True)
    contract = dict(state['contract'])
    old_source = contract['source']
    for key in ('source', 'hierarchy'):
        target = (args.new_root / Path(contract[key]).relative_to(args.old_root)).resolve()
        if not target.exists():
            raise FileNotFoundError(target)
        contract[key] = str(target)
    manifest = json.loads(args.manifest.read_text(encoding='utf-8-sig'))
    options = contract['options']
    source = Path(contract['source'])
    masks = source / 'masks'
    kwargs = dict(resolution=contract['resolution'], hold=options['llff_hold'],
                  masks_dir=masks if masks.is_dir() else None)
    previous = inspect_dataset(source, fingerprint_root=old_source, **kwargs)
    if previous['input_fingerprint'] != manifest['input_fingerprint']:
        raise ValueError('Moved dataset metadata/calibration differs from the saved scaffold fingerprint')
    current = inspect_dataset(source, **kwargs)
    state['contract'] = contract
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    try:
        with temporary.open('wb') as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    backup = args.manifest.with_suffix('.before_relocation.json')
    if not backup.exists():
        backup.write_bytes(args.manifest.read_bytes())
    manifest['input_fingerprint'] = current['input_fingerprint']
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(dict(iteration=state['iteration'], checkpoint=str(args.output),
                         source=contract['source'], hierarchy=contract['hierarchy']), indent=2))


if __name__ == '__main__':
    main()
