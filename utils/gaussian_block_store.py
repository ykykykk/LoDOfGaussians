"""Immutable disk blocks and atomic manifests for flat Gaussian training."""
import copy
import json
import os
import re
import uuid
from pathlib import Path

import numpy as np
import torch


class GaussianBlockStore:
    def __init__(self, root, manifest, metadata):
        self.root = Path(root)
        self.manifest = manifest
        self.blocks = manifest['blocks']
        self.metadata = metadata

    @property
    def block_rows(self):
        return int(self.metadata.get('block_rows', max((b['count'] for b in self.blocks), default=50000)))

    @classmethod
    def create(cls, directory, metadata=None):
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        if (root / 'manifest.json').exists():
            raise FileExistsError(root / 'manifest.json')
        (root / 'blocks').mkdir(exist_ok=True)
        return cls(root, dict(version=1, representation='flat_blocks', blocks=[]), dict(metadata or {}))

    @classmethod
    def open(cls, path):
        path = Path(path)
        if path.is_dir():
            path = path / 'manifest.json'
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if manifest.get('version') != 1 or manifest.get('representation') != 'flat_blocks':
            raise ValueError('Unsupported block checkpoint')
        metadata = torch.load(path.parent / manifest['metadata'], map_location='cpu', weights_only=True)
        return cls(path.parent, manifest, metadata)

    def _entry(self, block_id):
        entry = self.blocks[int(block_id)]
        if entry['id'] != int(block_id):
            raise ValueError('Non-contiguous block IDs')
        return entry

    def read(self, block_id):
        entry = self._entry(block_id)
        value = np.load(self.root / entry['file'], mmap_mode='r', allow_pickle=False)
        if value.shape != (entry['count'], 69) or value.dtype != np.float32:
            raise ValueError('Invalid block shape/dtype')
        return value

    def read_scores(self, block_id):
        entry = self._entry(block_id)
        if not entry.get("scores"):
            return np.zeros(entry["count"], dtype=np.float32)
        scores = np.load(self.root / entry["scores"], mmap_mode="r", allow_pickle=False)
        if scores.shape != (entry["count"],) or scores.dtype != np.float32:
            raise ValueError("Invalid score shape/dtype")
        return scores

    def write_scores(self, block_id, scores):
        """Write scores only, without duplicating the parameter/Adam block."""
        entry = self._entry(block_id)
        value = scores.detach().cpu().numpy() if isinstance(scores, torch.Tensor) else np.asarray(scores)
        if value.shape != (entry['count'],) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError('Expected finite float32 scores [N]')
        relative = 'blocks/' + str(block_id) + '-' + uuid.uuid4().hex + '-scores.npy'
        with (self.root / relative).open('wb') as handle:
            np.save(handle, value, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        entry['scores'] = relative

    def update_bounds(self, block_id, low, high):
        low, high = np.asarray(low), np.asarray(high)
        if low.shape != (3,) or high.shape != (3,) or not np.isfinite([low,high]).all() or np.any(low > high):
            raise ValueError("Invalid block bounds")
        self._entry(block_id).update(bounds_min=low.tolist(), bounds_max=high.tolist())

    def write(self, block_id, rows, scores=None):
        entry = self._entry(block_id)
        value = rows.detach().cpu().numpy() if isinstance(rows, torch.Tensor) else np.asarray(rows)
        if value.ndim != 2 or value.shape[1] != 69 or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError('Expected finite float32 [N,69] parameters and Adam state')
        if len(value):
            radius = 3 * np.exp(value[:, 3:6].astype(np.float64)).max(axis=1)
            low = (value[:, :3] - radius[:, None]).min(axis=0)
            high = (value[:, :3] + radius[:, None]).max(axis=0)
            if not np.isfinite(low).all() or not np.isfinite(high).all():
                raise ValueError('Non-finite Gaussian bounds')
        else:
            low = high = np.zeros(3)
        relative = 'blocks/' + str(block_id) + '-' + uuid.uuid4().hex + '.npy'
        with (self.root / relative).open('wb') as handle:
            np.save(handle, value, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        score_relative = entry.get('scores') if entry.get('count') == len(value) else None
        if scores is not None:
            scores = scores.detach().cpu().numpy() if isinstance(scores, torch.Tensor) else np.asarray(scores)
            if scores.shape != (len(value),) or scores.dtype != np.float32 or not np.isfinite(scores).all():
                raise ValueError('Expected finite float32 scores [N]')
            score_relative = 'blocks/' + str(block_id) + '-' + uuid.uuid4().hex + '-scores.npy'
            with (self.root / score_relative).open('wb') as handle:
                np.save(handle, scores, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
        entry.update(file=relative, scores=score_relative, count=len(value), bounds_min=low.tolist(), bounds_max=high.tolist())

    def append(self, rows, skybox=False, scores=None):
        block_id = len(self.blocks)
        self.blocks.append(dict(id=block_id, skybox=bool(skybox)))
        try:
            self.write(block_id, rows, scores)
        except Exception:
            self.blocks.pop()
            raise
        return block_id

    appendblock = append

    def checkpoint(self, metadata=None):
        if metadata is not None:
            self.metadata = dict(metadata)
        relative = 'metadata-' + uuid.uuid4().hex + '.pt'
        with (self.root / relative).open('wb') as handle:
            torch.save(self.metadata, handle)
            handle.flush()
            os.fsync(handle.fileno())
        snapshot = copy.deepcopy(self.manifest)
        snapshot.update(metadata=relative, count=sum(b['count'] for b in self.blocks))
        destination = self.root / 'manifest.json'
        temporary = self.root / ('manifest-' + uuid.uuid4().hex + '.tmp')
        with temporary.open('w', encoding='utf-8') as handle:
            json.dump(snapshot, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        if destination.exists():
            previous_temporary = self.root / ('manifest-previous-' + uuid.uuid4().hex + '.tmp')
            with previous_temporary.open('wb') as handle:
                handle.write(destination.read_bytes())
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(previous_temporary, self.root / 'manifest.previous.json')
        os.replace(temporary, destination)
        self.manifest.update(metadata=relative, count=snapshot['count'])
        self.garbage_collect()
        return destination

    def garbage_collect(self):
        """Retain current and previous committed checkpoints plus live dirty blocks.

        Only immutable version names emitted by this store are eligible. Open
        Windows mmap handles can defer removal until the next checkpoint.
        """
        root = self.root.resolve()
        keep = set()
        snapshots = [self.manifest]
        for name in ('manifest.json', 'manifest.previous.json'):
            path = root / name
            if path.exists():
                snapshots.append(json.loads(path.read_text(encoding='utf-8')))
        for snapshot in snapshots:
            if snapshot.get('metadata'):
                keep.add((root / snapshot['metadata']).resolve())
            for entry in snapshot['blocks']:
                for key in ('file', 'scores'):
                    if entry.get(key):
                        keep.add((root / entry[key]).resolve())
        patterns = [(root, r'metadata-[0-9a-f]{32}\.pt'),
                    (root / 'blocks', r'[0-9]+-[0-9a-f]{32}(?:-scores)?\.npy')]
        removed = 0
        for directory, pattern in patterns:
            resolved = directory.resolve()
            if resolved != root and not resolved.is_relative_to(root):
                raise ValueError('Block directory resolves outside checkpoint root')
            for path in directory.iterdir():
                if not re.fullmatch(pattern, path.name):
                    continue
                target = path.resolve()
                if not target.is_relative_to(root) or target in keep or path.is_symlink():
                    continue
                try:
                    path.unlink()
                    removed += 1
                except PermissionError:
                    pass
        return removed

    def export_ply(self, path, include_skybox=False, chunk_size=65536):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        selected = [b for b in self.blocks if include_skybox or not b['skybox']]
        names = ['x','y','z','nx','ny','nz'] + [f'f_dc_{i}' for i in range(3)] + [f'f_rest_{i}' for i in range(9)]
        names += ['opacity'] + [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)]
        temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
        try:
            with temp.open('wb') as handle:
                header = ['ply','format binary_little_endian 1.0',f"element vertex {sum(b['count'] for b in selected)}"]
                header += [f'property float {name}' for name in names] + ['end_header']
                handle.write(('\n'.join(header) + '\n').encode('ascii'))
                for entry in selected:
                    rows = self.read(entry['id'])
                    for start in range(0, len(rows), chunk_size):
                        p = rows[start:start+chunk_size, :23]
                        if not np.isfinite(p).all():
                            raise ValueError('Non-finite Gaussian parameters')
                        out = np.zeros((len(p), 26), dtype='<f4')
                        out[:,:3] = p[:,:3]
                        out[:,6:9] = p[:,10:13]
                        out[:,9:18] = p[:,14:23].reshape(-1,3,3).transpose(0,2,1).reshape(-1,9)
                        out[:,18] = p[:,13]
                        out[:,19:22] = p[:,3:6]
                        out[:,22:26] = p[:,6:10]
                        out.tofile(handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
        return path
