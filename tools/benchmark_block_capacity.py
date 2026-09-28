"""Synthetic capacity test; repeated blocks do not prove real-scene quality."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
import uuid
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from utils.gaussian_block_store import GaussianBlockStore
from utils.paged_gaussian_pool import PagedGaussianPool


class RepeatedBlocks:
    def __init__(self, source, repeats):
        self.source = source
        self.block_rows = source.block_rows
        self.blocks = [dict(b, id=i * len(source.blocks) + b['id'])
                       for i in range(repeats) for b in source.blocks]

    def read(self, bid):
        return self.source.read(bid % len(self.source.blocks))

    def read_scores(self, bid):
        return self.source.read_scores(bid % len(self.source.blocks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--materialize', help='Create an independent hardlinked synthetic 5x checkpoint')
    args = parser.parse_args()
    source = GaussianBlockStore.open(args.checkpoint)
    if args.materialize:
        metadata = copy.deepcopy(source.metadata)
        metadata.update(synthetic_capacity_test=True, synthetic_repeats=5,
                        size=source.manifest['count']*5,
                        skybox_points=int(metadata['skybox_points'])*5)
        target = GaussianBlockStore.create(args.materialize, metadata)
        for _ in range(5):
            for original in source.blocks:
                bid = len(target.blocks)
                entry = copy.deepcopy(original)
                entry['id'] = bid
                for key in ('file', 'scores'):
                    if not original.get(key):
                        continue
                    name = f'blocks/{bid}-{uuid.uuid4().hex}' + ('-scores.npy' if key == 'scores' else '.npy')
                    os.link(source.root / original[key], target.root / name)
                    entry[key] = name
                target.blocks.append(entry)
        print(target.checkpoint())
        return
    for repeats in (1, 5):
        store = RepeatedBlocks(source, repeats)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        pool = PagedGaussianPool(store, 30 * store.block_rows)
        start = time.perf_counter()
        for offset in (0, 20, len(store.blocks) - 20, 0):
            pool.ensure(list(range(offset, offset + 20)))
        torch.cuda.synchronize()
        print(json.dumps(dict(synthetic=True, repeats=repeats,
            logical_points=sum(b['count'] for b in store.blocks),
            cache_rows=pool.capacity, seconds=time.perf_counter()-start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), **pool.stats)), flush=True)
        del pool, store


if __name__ == '__main__':
    main()
