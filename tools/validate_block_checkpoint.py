"""Read-only bounded-memory integrity and conversion checks for block checkpoints."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from utils.gaussian_block_store import GaussianBlockStore


class Digest:
    def __init__(self):
        self.count = 0
        self.sums = np.zeros(69, dtype=np.float64)
        self.hash_sum = np.uint64(0)
        self.hash_xor = np.uint64(0)
        self.weights = np.random.default_rng(57119).integers(1, np.iinfo(np.uint64).max, size=69, dtype=np.uint64) | np.uint64(1)

    def add(self, rows):
        if rows.ndim != 2 or rows.shape[1] != 69 or rows.dtype != np.float32 or not np.isfinite(rows).all():
            raise ValueError('Non-finite or invalid parameter/Adam rows')
        self.count += len(rows)
        self.sums += rows.sum(axis=0,dtype=np.float64)
        # Column-sensitive row hash preserves exact FP32 bits; sum and xor are
        # independent of spatial reordering. This is a strong corruption check,
        # not a mathematical proof of multiset equality.
        hashes = np.zeros(len(rows),dtype=np.uint64)
        bits = np.ascontiguousarray(rows).view(np.uint32)
        with np.errstate(over='ignore'):
            for column in range(69):
                hashes += bits[:,column].astype(np.uint64) * self.weights[column]
            hashes ^= hashes >> np.uint64(30)
            hashes *= np.uint64(0xbf58476d1ce4e5b9)
            hashes ^= hashes >> np.uint64(27)
            hashes *= np.uint64(0x94d049bb133111eb)
            hashes ^= hashes >> np.uint64(31)
            self.hash_sum += hashes.sum(dtype=np.uint64)
        self.hash_xor ^= np.bitwise_xor.reduce(hashes,initial=np.uint64(0))


def verify_ply(path, expected_count=None):
    path=Path(path)
    with path.open('rb') as handle:
        count=None; properties=0; binary=False
        for _ in range(256):
            line=handle.readline().decode('ascii').strip()
            if line=='format binary_little_endian 1.0': binary=True
            if line.startswith('element vertex '): count=int(line.split()[-1])
            if line.startswith('property float '): properties+=1
            if line=='end_header': break
        else: raise ValueError('PLY header not terminated')
        header_bytes=handle.tell()
    if not binary or properties!=26 or count is None:
        raise ValueError('Expected binary SH1 Gaussian PLY')
    if expected_count is not None and count!=expected_count:
        raise ValueError('PLY point count mismatch')
    if path.stat().st_size!=header_bytes+count*26*4:
        raise ValueError('PLY byte length mismatch')
    return dict(path=str(path),count=count,bytes=path.stat().st_size)


def validate(checkpoint, source=None, ply=None):
    store=GaussianBlockStore.open(checkpoint)
    digest=Digest(); sky=0
    for entry in store.blocks:
        rows=store.read(entry['id']); scores=store.read_scores(entry['id'])
        if not np.isfinite(scores).all(): raise ValueError('Non-finite split scores')
        if entry['skybox']: sky+=len(rows)
        for start in range(0,len(rows),65536):
            chunk=rows[start:start+65536]
            digest.add(chunk)
            if len(chunk):
                radius=3*np.exp(chunk[:,3:6].astype(np.float64)).max(axis=1)
                lower=(chunk[:,:3]-radius[:,None]).min(axis=0)
                upper=(chunk[:,:3]+radius[:,None]).max(axis=0)
                if np.any(lower<np.asarray(entry['bounds_min'])-1e-7) or np.any(upper>np.asarray(entry['bounds_max'])+1e-7):
                    raise ValueError('Block bounds do not contain Gaussian supports')
    if digest.count!=store.manifest['count']: raise ValueError('Manifest count mismatch')
    result=dict(checkpoint=str(checkpoint),iteration=store.metadata['iteration'],count=digest.count,scene_count=digest.count-sky,skybox_count=sky,blocks=len(store.blocks),finite=True,bounds=True,parameter_adam_hash_sum=hex(int(digest.hash_sum)),parameter_adam_hash_xor=hex(int(digest.hash_xor)))
    if source:
        state=torch.load(source,map_location='cpu',weights_only=True,mmap=True)
        reference=Digest()
        rows=state['properties'].numpy()
        for start in range(0,len(rows),65536): reference.add(rows[start:start+65536])
        if digest.count!=reference.count or digest.hash_sum!=reference.hash_sum or digest.hash_xor!=reference.hash_xor:
            raise ValueError('Source parameter/Adam row multiset hash mismatch')
        if not np.allclose(digest.sums,reference.sums,rtol=1e-10,atol=1e-7):
            raise ValueError('Source per-column float64 sums differ')
        if state['iteration']!=store.metadata['iteration']: raise ValueError('Iteration differs')
        result.update(source=str(source),source_parameters_and_adam_preserved=True,max_column_sum_error=float(np.max(np.abs(digest.sums-reference.sums))))
    if ply: result['ply']=verify_ply(ply,digest.count-sky)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--ply',type=Path)
    args=parser.parse_args()
    print(json.dumps(validate(args.checkpoint,args.source,args.ply),indent=2),flush=True)

if __name__=='__main__': main()
