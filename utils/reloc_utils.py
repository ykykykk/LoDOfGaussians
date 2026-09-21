"""Lazy relocation constants: importing a Windows worker must not start CUDA."""
from functools import lru_cache
import math

import torch

N_max = 51


@lru_cache(maxsize=None)
def _binomial_table(device: str):
    # Build once on CPU and upload in one operation, rather than performing
    # 1,326 individual scalar writes to CUDA on every module import.
    values = [[math.comb(n, k) if k <= n else 0 for k in range(N_max)]
              for n in range(N_max)]
    return torch.tensor(values, dtype=torch.float32, device=device)


def compute_relocation_cuda(opacity_old, scale_old, N):
    from gaussian_hierarchy._C import compute_relocation

    binoms = _binomial_table(str(opacity_old.device))
    return compute_relocation(opacity_old, scale_old, N.int(), binoms, N_max)
