"""Lazy Windows/Linux CUDA extension. No CUDA work in DataLoader imports."""
from pathlib import Path
import os
import warnings
import torch

_MODULE = None
_ERROR = None


def load_native(mode="auto"):
    global _MODULE, _ERROR
    if mode not in ("auto", "cuda", "torch"):
        raise ValueError("native_ops must be auto, cuda, or torch")
    if mode == "torch":
        return None
    if _MODULE is not None:
        return _MODULE
    if _ERROR is None:
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable")
            os.environ.setdefault("MAX_JOBS", "4")
            from torch.utils.cpp_extension import load
            root = Path(__file__).resolve().parents[1] / "csrc"
            _MODULE = load(
                name="alod_resident_ops_v2", sources=[str(root / "resident_ops.cpp"), str(root / "resident_ops.cu")],
                extra_cflags=["/O2"] if os.name == "nt" else ["-O3"],
                # Keep FP32 arithmetic; do not enable fast-math/FTZ.
                extra_cuda_cflags=["-O3", "--fmad=false"], verbose=False,
            )
            return _MODULE
        except Exception as exc:
            _ERROR = str(exc)
            if mode == "auto":
                warnings.warn("ALoD native ops unavailable; using the Torch reference: " + _ERROR, RuntimeWarning)
    if mode == "cuda":
        raise RuntimeError("ALoD CUDA backend could not be loaded: " + str(_ERROR))
    return None


def native_status():
    return {"loaded": _MODULE is not None, "build_error": _ERROR}
