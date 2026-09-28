"""Optional SSD-backed CPU tensors; GPU residency remains independently bounded."""
from pathlib import Path
import tempfile

import numpy as np
import torch


class MappedHostStorage:
    def __init__(self, directory):
        base = Path(directory).expanduser().resolve()
        base.mkdir(parents=True, exist_ok=True)
        self.path = Path(tempfile.mkdtemp(prefix='resident-', dir=base))
        self.maps = []

    def allocate(self, shape, dtype):
        numpy_dtype = torch.empty((), dtype=dtype).numpy().dtype
        path = self.path / f'{len(self.maps)}.bin'
        # A fresh extended file reads as zero; do not touch every reserved page.
        mapped = np.memmap(path, dtype=numpy_dtype, mode='w+', shape=tuple(shape))
        self.maps.append(mapped)
        return torch.from_numpy(mapped)

    def flush(self):
        for mapped in self.maps:
            mapped.flush()
