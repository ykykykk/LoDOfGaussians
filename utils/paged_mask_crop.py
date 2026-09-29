"""Silhouette pruning of the working block store, retaining complete Adam rows."""
import numpy as np

from utils.mask_crop import keep_points


def crop_pool(pool, views, config):
    # Flush the latest parameters, moments and growth scores before classification.
    # Store writes are immutable: the input checkpoint and published manifest stay valid.
    pool.clear()
    store = pool.store
    selections = []
    before = kept = 0
    for block in store.blocks:
        rows = store.read(block['id'])
        keep = (np.zeros(len(rows), dtype=bool) if block.get('skybox')
                else keep_points(rows, views, config))
        selections.append(keep)
        before += len(rows)
        kept += int(keep.sum())
    if not kept:
        raise ValueError('Mask crop would remove the entire model; check masks and camera alignment.')
    for block, keep in zip(store.blocks, selections):
        if not keep.all():
            bid = block['id']
            store.write(bid, np.array(store.read(bid)[keep]),
                        scores=np.array(store.read_scores(bid)[keep]))
    # Empty pages must not participate in frustum selection or bounds reductions.
    store.blocks[:] = [block for block in store.blocks if block['count']]
    for bid, block in enumerate(store.blocks):
        block['id'] = bid
    pool.refresh_metadata()
    return dict(input_points=before, kept_points=kept, removed_points=before-kept)
