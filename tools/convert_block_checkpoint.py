"""Convert a flat resident checkpoint to spatial, immutable disk blocks (CPU only)."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from utils.gaussian_block_store import GaussianBlockStore


def morton_keys(xyz, low, high):
    q = np.clip((xyz.astype(np.float64) - low) / np.maximum(high-low, 1e-12) * 1023, 0, 1023).astype(np.uint32)
    result = np.zeros(len(xyz), dtype=np.uint32)
    for bit in range(10):
        for axis in range(3):
            result |= ((q[:,axis] >> bit) & 1) << (3 * bit + axis)
    return result


def convert(source, output, block_size=50000, radius_bands=True):
    if block_size <= 0:
        raise ValueError('block_size must be positive')
    state = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
    n = int(state['size'])
    rows = state['properties'].numpy()
    if rows.shape != (n,69) or rows.dtype != np.float32:
        raise ValueError('Expected SH1 FP32 resident properties [N,69]')
    nodes = state['nodes']
    sky = int(state['skybox_points'])
    # Reject a LoD tree: silently copying ancestors would duplicate the scene.
    for start in range(sky,n,262144):
        if torch.any(nodes[start:start+262144,2] != 0):
            raise ValueError('Conversion requires flat or leaf-only checkpoint')
    metadata = {k:v for k,v in state.items() if k not in ('properties','nodes','scores','seen')}
    metadata.update(source_checkpoint=str(Path(source).resolve()), storage_representation='flat_blocks', scores_reset=False, block_rows=block_size)
    store = GaussianBlockStore.create(output, metadata)
    for start in range(0,sky,block_size):
        store.append(rows[start:min(start+block_size,sky)], skybox=True, scores=state['scores'][start:min(start+block_size,sky)].numpy().reshape(-1))
    low, high = np.full(3,np.inf), np.full(3,-np.inf)
    for start in range(sky,n,262144):
        xyz = rows[start:start+262144,:3]
        low = np.minimum(low,xyz.min(axis=0)); high = np.maximum(high,xyz.max(axis=0))
    # A handful of very large supports must not inflate otherwise compact blocks.
    # Derive a scale-independent band base from a bounded, deterministic sample.
    sample = rows[sky:n:max(1, (n-sky)//100000),3:6]
    base = float(12*np.exp(np.median(sample.max(axis=1)))) if len(sample) else 1.0
    metadata.update(radius_bands=bool(radius_bands), radius_band_base=base)
    keys = np.empty(n-sky, dtype=np.uint64)
    for start in range(sky,n,262144):
        chunk = rows[start:start+262144]
        spatial = morton_keys(chunk[:,:3],low,high).astype(np.uint64)
        bands = np.maximum(0, np.ceil((np.log(3)+chunk[:,3:6].max(axis=1)-np.log(base))/np.log(2))).astype(np.uint64) if radius_bands else np.zeros(len(chunk),dtype=np.uint64)
        if bands.max(initial=0) > 63:
            raise ValueError('Gaussian support range exceeds supported radius bands')
        keys[start-sky:min(start+262144,n)-sky] = (bands << np.uint64(30)) | spatial
    order = np.argsort(keys, kind='stable')
    # Never allow a block to straddle size bands, even for a short final block.
    bands, counts = np.unique(keys >> np.uint64(30), return_counts=True)
    del keys
    offset = 0
    for band, count in zip(bands,counts):
        end = offset+int(count)
        for start in range(offset,end,block_size):
            indices = order[start:min(start+block_size,end)] + sky
            block_id=store.append(rows[indices], scores=state['scores'].numpy()[indices].reshape(-1))
            store.blocks[block_id]['radius_band']=int(band)
        offset=end
        print(f'Converted {end:,}/{len(order):,} scene points; radius band {band}',flush=True)
    metadata.update(size=n, views=0, empty_windows=0)
    path = store.checkpoint(metadata)
    print(f'Block checkpoint: {path}; {n:,} points; {len(store.blocks)} blocks',flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--block-size',type=int,default=50000)
    parser.add_argument('--export-ply',type=Path)
    parser.add_argument('--no-radius-bands',action='store_true',help='Use legacy Morton-only partition for comparisons')
    args=parser.parse_args()
    path=convert(args.input,args.output,args.block_size,not args.no_radius_bands)
    if args.export_ply:
        GaussianBlockStore.open(path).export_ply(args.export_ply)

if __name__ == '__main__':
    main()
