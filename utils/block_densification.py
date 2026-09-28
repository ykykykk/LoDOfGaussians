"""Bounded-memory, parent-free growth of immutable Gaussian disk blocks."""
from types import SimpleNamespace
from collections import defaultdict, deque
import math

import numpy as np
import torch

from utils.flat_gaussians import split_flat


def _morton_order(xyz):
    """Stable local spatial ordering; memory proportional to one page."""
    low = xyz.min(axis=0)
    span = np.maximum(xyz.max(axis=0) - low, 1e-12)
    q = np.clip((xyz - low) / span * 1023, 0, 1023).astype(np.uint64)
    codes = np.zeros(len(xyz), dtype=np.uint64)
    for bit in range(10):
        for axis in range(3):
            codes |= ((q[:, axis] >> bit) & 1) << (3 * bit + axis)
    return np.argsort(codes, kind='stable')


def densify_blocks(store, opt):
    """Split scored scene rows through a disk store or coherent cache adapter.

    All point-sized temporaries are bounded by two block capacities. Budget
    allocation uses only O(number-of-blocks) metadata. No checkpoint is committed
    here: callers atomically checkpoint the complete growth transaction afterward.
    """
    blocks = [dict(entry) for entry in store.blocks]
    page_rows = int(store.block_rows)
    if page_rows <= 0:
        raise ValueError('block_rows must be positive')
    threshold = float(opt.densify_grad_threshold)
    fraction = float(getattr(opt, 'densify_max_leaf_fraction', 0))
    if not math.isfinite(threshold) or not math.isfinite(fraction) or fraction < 0:
        raise ValueError('Invalid densification threshold/fraction')
    total = sum(b['count'] for b in blocks)
    scene = sum(b['count'] for b in blocks if not b['skybox'])
    eligible, positive = [], []
    for entry in blocks:
        if entry['count'] > page_rows:
            raise ValueError('Source block exceeds block_rows')
        scores = store.read_scores(entry['id'])
        if not np.isfinite(scores).all():
            raise ValueError('Non-finite block densification scores')
        positive.append(bool(np.any(scores > 0)))
        eligible.append(0 if entry['skybox'] else int(np.count_nonzero(scores > threshold)))
        del scores
    count = sum(eligible)
    budget = min(count, max(0, int(opt.cap_max) - total))
    limit = int(getattr(opt, 'densify_max_new_nodes', 0))
    if limit > 0:
        budget = min(budget, limit)
    if fraction > 0:
        budget = min(budget, int(scene * fraction))
    allocations = [budget * n // count if count else 0 for n in eligible]
    remainder = budget - sum(allocations)
    ranked = sorted(range(len(blocks)), key=lambda i: (-(budget * eligible[i] % max(count, 1)), i))
    for i in ranked[:remainder]:
        allocations[i] += 1
    # Spatially nearby source blocks tend to emit into the same child page.
    if blocks:
        centers = np.array([(np.asarray(b['bounds_min']) + np.asarray(b['bounds_max'])) * .5
                            for b in blocks], dtype=np.float64)
        order = _morton_order(centers)
        order = sorted(order, key=lambda i: blocks[i].get('radius_band', -1))
    else:
        order = []
    pending = torch.empty((page_rows, 69), dtype=torch.float32) if budget else None
    used, added = 0, 0
    current_band = None
    tails = defaultdict(deque)

    def register_tail(block_id):
        item = store.blocks[block_id]
        if not item["skybox"] and item["count"] < page_rows:
            tails[item.get("radius_band")].append(block_id)

    def emit(size):
        rows = pending[:size]
        permutation = _morton_order(rows[:, :3].numpy().astype(np.float64))
        rows = rows[torch.from_numpy(permutation)]
        offset = 0
        queue = tails[current_band]
        # Only processed source pages or newly appended pages are registered:
        # extending an unprocessed page would invalidate this window's snapshot.
        while queue and offset < size:
            block_id = queue[0]
            count = store.blocks[block_id]['count']
            take = min(page_rows-count, size-offset)
            merged = np.empty((count+take,69),dtype=np.float32)
            merged[:count] = store.read(block_id)
            merged[count:] = rows[offset:offset+take].numpy()
            scores = np.zeros(count+take,dtype=np.float32)
            scores[:count] = store.read_scores(block_id)
            store.write(block_id,merged,scores=scores)
            offset += take
            if count+take == page_rows:
                queue.popleft()
        if offset < size:
            block_id = store.append(rows[offset:],skybox=False,
                                    scores=np.zeros(size-offset,dtype=np.float32))
            if current_band is not None:
                store.blocks[block_id]['radius_band'] = current_band
            register_tail(block_id)

    for index in order:
        entry, amount = blocks[index], allocations[index]
        n, block_id = entry['count'], entry['id']
        if not amount:
            if positive[index]:
                store.write_scores(block_id, np.zeros(n, dtype=np.float32))
            register_tail(block_id)
            continue
        band = entry.get('radius_band')
        if used and band != current_band:
            emit(used)
            used = 0
        current_band = band
        props = torch.empty((n + amount, 69), dtype=torch.float32)
        props[:n].copy_(torch.from_numpy(np.array(store.read(block_id), copy=True)))
        scores = torch.zeros(n + amount)
        scores[:n].copy_(torch.from_numpy(np.array(store.read_scores(block_id), copy=True)))
        local = SimpleNamespace(properties=props, nodes=torch.zeros((n + amount, 6), dtype=torch.int32),
                                size=n, skybox_points=0, _densification_criterium=scores)
        local_opt = SimpleNamespace(cap_max=n + amount, densify_max_new_nodes=amount,
                                    densify_max_leaf_fraction=0, densify_grad_threshold=threshold)
        actual = split_flat(local, local_opt)
        if actual != amount:
            raise RuntimeError('Block growth allocation changed unexpectedly')
        store.write(block_id, props[:n], scores=np.zeros(n, dtype=np.float32))
        register_tail(block_id)
        offset = 0
        while offset < amount:
            take = min(page_rows - used, amount - offset)
            pending[used:used+take].copy_(props[n+offset:n+offset+take])
            used += take
            offset += take
            if used == page_rows:
                emit(used)
                used = 0
        added += actual
    if used:
        emit(used)
    return dict(net_added=added, total_points=total + added, eligible_points=count)
