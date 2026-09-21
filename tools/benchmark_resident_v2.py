"""Isolated FP32 Adam microbenchmark; not an end-to-end training speed claim.

Run from the project root: python -m tools.benchmark_resident_v2 --rows 100000
Native build and warmup are excluded. Both variants start from identical state.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from utils.resident_pool_v2 import StreamingResidentPool
from utils.resident_native import load_native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, default=100000)
    parser.add_argument('--degree', type=int, choices=(0,1,2,3), default=1)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--output', type=Path, default=Path('resident_adam_benchmark.json'))
    args = parser.parse_args()
    if args.rows <= 0 or args.steps <= 0 or args.warmup < 0:
        parser.error('rows/steps must be positive; warmup must be nonnegative')
    if not torch.cuda.is_available():
        raise RuntimeError('This benchmark requires a CUDA GPU; no CPU speed ratio is substituted')
    native = load_native('cuda')
    width = 14 + 3*((args.degree+1)**2-1)
    free, _ = torch.cuda.mem_get_info()
    if args.rows * width * 4 * 14 > free - 2*2**30:
        raise ValueError('Requested benchmark is too large for the current free VRAM; reduce --rows')
    torch.manual_seed(19)
    source = torch.zeros(args.rows, 3*width)
    source[:, :width] = torch.rand(args.rows, width)
    gradient = torch.rand(args.rows, width, device='cuda') - .5
    rates = torch.linspace(.00002, .05, width, device='cuda')
    results, final = {}, []
    for name, ops in [('torch', None), ('indexed_cuda', native)]:
        host = source.clone()
        pool = StreamingResidentPool(host, torch.zeros(args.rows), args.rows,
                                     ops=ops, prefetch_rows=0)
        try:
            packet = pool.acquire(np.arange(args.rows, dtype=np.int64))
            for step in range(args.warmup):
                packet.adam_step(gradient, rates, step)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            begin = time.perf_counter()
            for step in range(args.warmup, args.warmup+args.steps):
                packet.adam_step(gradient, rates, step)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-begin
            results[name] = dict(ms_per_adam=elapsed*1000/args.steps,
                                 peak_allocated_bytes=torch.cuda.max_memory_allocated())
            pool.flush()
            final.append(host)
        finally:
            pool.close()
        del packet, pool
    torch.testing.assert_close(final[0], final[1], rtol=5e-4, atol=3e-5)
    report = dict(kind='isolated_adam_only', rows=args.rows, degree=args.degree, steps=args.steps,
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                  results=results, parity='passed',
                  adam_speed_ratio=results['torch']['ms_per_adam']/results['indexed_cuda']['ms_per_adam'],
                  note='Excludes rasterization, backward, image I/O, hierarchy and cache misses. Not total training speed.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
