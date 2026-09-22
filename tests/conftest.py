"""Run CUDA tests under the same Windows gsplat adapter as training."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.gsplat_compat import prepare_gsplat_windows
prepare_gsplat_windows()
