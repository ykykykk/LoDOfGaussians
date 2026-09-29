"""Interactive no-LoD Gaussian checkpoint viewer (local browser UI)."""
import argparse
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='Paged checkpoint directory or manifest.json')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--pool-gib', type=float, default=4)
    args = parser.parse_args()
    import torch
    from utils.gaussian_block_store import GaussianBlockStore
    from utils.paged_gaussian_pool import PagedGaussianPool
    from utils.resident_native import load_native
    from utils.realtime_viewer import PagedViewRenderer, RealtimeViewer
    if args.pool_gib <= 0:
        parser.error('--pool-gib must be positive')
    store = GaussianBlockStore.open(args.checkpoint)
    free, _ = torch.cuda.mem_get_info()
    capacity = int(min(args.pool_gib*2**30, free-6*2**30)//280)
    if capacity < store.block_rows:
        raise RuntimeError('Insufficient free GPU memory for preview cache plus rendering headroom')
    pool = PagedGaussianPool(store, capacity, load_native('cuda'))
    renderer = PagedViewRenderer(pool, SimpleNamespace(**store.metadata['contract']['pipeline']))
    viewer = RealtimeViewer(renderer, args.port)
    print(f'Realtime viewer: {viewer.url}', flush=True)
    try:
        viewer.serve()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
