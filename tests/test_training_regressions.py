"""CPU regressions: python -m pytest tests/test_training_regressions.py -q"""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.export_ply import export_gaussian_tensors_ply, finest_leaf_indices
from utils.training_runtime import flush_resident_state, make_camera_loader, shutdown_camera_loader


ROOT = Path(__file__).resolve().parents[1]


def read_ply(path):
    names = []
    with path.open("rb") as f:
        assert f.readline().strip() == b"ply"
        assert f.readline().strip() == b"format binary_little_endian 1.0"
        count = int(f.readline().split()[-1])
        while True:
            line = f.readline().decode("ascii").strip()
            if line == "end_header":
                break
            assert line.startswith("property float ")
            names.append(line.split()[-1])
        result = np.fromfile(f, dtype=np.dtype([(name, "<f4") for name in names]))
    assert len(result) == count
    return result


def tensors(count=5, degree=1):
    rest = (degree + 1) ** 2 - 1
    return (
        torch.arange(count * 3, dtype=torch.float32).reshape(count, 3),
        torch.full((count, 3), 0.125),
        torch.arange(count * rest * 3, dtype=torch.float32).reshape(count, rest, 3),
        torch.linspace(0.1, 0.9, count).reshape(count, 1),
        torch.full((count, 3), -2.0),
        torch.tensor([[1., 0., 0., 0.]]).repeat(count, 1),
    )


def test_leaf_selection_is_an_adaptive_cut_not_maximum_depth():
    nodes = torch.tensor([
        [0, -1, 2, 1, 0, 0], [1, 0, 0, 0, 2, 0],
        [1, 0, 2, 3, 0, 0], [2, 2, 0, 0, 4, 0],
        [2, 2, 0, 0, 0, 0], [-99, -99, -99, -99, -99, 0],
    ], dtype=torch.int32)
    assert finest_leaf_indices(nodes).tolist() == [1, 3, 4]
    assert finest_leaf_indices(nodes, include_skybox=True).tolist() == [1, 3, 4, 5]


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
@pytest.mark.parametrize("chunk_size", [1, 7])
def test_ply_roundtrip_fields_and_chunking(tmp_path, degree, chunk_size):
    values = tensors(degree=degree)
    indices = torch.tensor([1, 3, 4])
    path = tmp_path / "leaves.ply"
    export_gaussian_tensors_ply(*values, path, chunk_size, opacity_activated=True, indices=indices)
    data = read_ply(path)
    np.testing.assert_allclose(data["x"], values[0][indices, 0])
    np.testing.assert_allclose(data["opacity"], torch.logit(values[3][indices, 0]), rtol=1e-6)
    np.testing.assert_allclose(data["scale_0"], -2.0)
    np.testing.assert_allclose(data["rot_0"], 1.0)
    expected = values[2][indices].permute(0, 2, 1).reshape(3, -1)
    for i in range(expected.shape[1]):
        np.testing.assert_allclose(data[f"f_rest_{i}"], expected[:, i])


def test_ply_atomic_on_invalid_later_chunk(tmp_path):
    values = tensors()
    values[0][4, 0] = float("nan")
    path = tmp_path / "existing.ply"
    path.write_bytes(b"previous valid output")
    with pytest.raises(ValueError, match="Non-finite xyz"):
        export_gaussian_tensors_ply(*values, path, chunk_size=2)
    assert path.read_bytes() == b"previous valid output"
    assert not list(tmp_path.glob("*.tmp"))


def test_unselected_parents_are_not_exported_or_validated(tmp_path):
    values = tensors()
    values[0][0] = float("nan")
    path = tmp_path / "leaf.ply"
    export_gaussian_tensors_ply(*values, path, indices=torch.tensor([1, 2]))
    assert len(read_ply(path)) == 2


@pytest.mark.parametrize("bad", [torch.ones(3), torch.zeros(2, 6), torch.zeros(3, 5, dtype=torch.int32)])
def test_invalid_nodes_rejected(bad):
    with pytest.raises(ValueError):
        finest_leaf_indices(bad)


@pytest.mark.parametrize("chunk_size", [1, 4, 64])
@pytest.mark.parametrize("rest_width", [0, 9, 45])
def test_flush_preserves_property_order_moments_and_scores(chunk_size, rest_width):
    widths = [3, 3, 4, 3, 1, rest_width]
    active = [torch.arange(2 * w).reshape(2, w).float() + i * 100 for i, w in enumerate(widths)]
    cached = [torch.arange(3 * w).reshape(3, w).float() + 1000 + i * 100 for i, w in enumerate(widths)]
    parameters = [{"exp_avgs": torch.full((5, w), 2000. + i), "exp_avgs_sqs": torch.full((5, w), 3000. + i)} for i, w in enumerate(widths)]
    ids = torch.tensor([7, 2, 5, 1, 4], dtype=torch.int32)
    g = SimpleNamespace(size=8, properties=torch.full((8, sum(widths) * 3), -1.),
                        _densification_criterium=torch.full((8,), -1.), _contributed=torch.zeros(8, dtype=torch.bool))
    scores = (torch.tensor([0.25, 0.5]), torch.tensor([0.75, 1., 2.]))
    contributed = (torch.tensor([True, False]), torch.tensor([False, True, True]))
    flush_resident_state(g, ids, active, cached, parameters, scores=scores, contributed=contributed, chunk_size=chunk_size)
    expected = torch.cat([torch.cat([a, c]) for a, c in zip(active, cached)] +
                         [p["exp_avgs"] for p in parameters] + [p["exp_avgs_sqs"] for p in parameters], dim=1)
    torch.testing.assert_close(g.properties[ids.long()], expected)
    torch.testing.assert_close(g._densification_criterium[ids.long()], torch.cat(scores))
    assert torch.equal(g._contributed[ids.long()], torch.cat(contributed))
    assert (g.properties[0] == -1).all()
    assert (g._densification_criterium[ids.long()] > 0.6).sum().item() == 3


def test_loader_zero_workers_and_shutdown_does_not_create_workers():
    opt = SimpleNamespace(data_workers=0, pin_memory=False)
    loader = make_camera_loader([1, 2], opt, shuffle=False)
    assert list(loader) == [[1], [2]]
    shutdown_camera_loader(loader)
    marker = []
    mock = SimpleNamespace(_iterator=SimpleNamespace(_shutdown_workers=lambda: marker.append(1)))
    shutdown_camera_loader(mock)
    assert marker == [1]


def test_loader_rejects_empty_dataset_and_invalid_worker_count():
    with pytest.raises(ValueError, match="empty"):
        make_camera_loader([], SimpleNamespace(), shuffle=False)
    with pytest.raises(ValueError, match="data_workers"):
        make_camera_loader([1], SimpleNamespace(data_workers=-1), shuffle=False)


def test_classic_densification_restores_upstream_threshold_selection():
    path = ROOT / "scene" / "gaussian_model.py"
    if not path.exists():
        pytest.skip("Full repository source is required for this integration guard")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "add_new_gs")
    source = ast.unparse(method)
    assert "topk" not in source
    assert "alive_indices[self._densification_criterium[alive_indices] > densify_threshold]" in source


def test_training_flushes_before_save_and_before_densification_reset():
    path = ROOT / "train_hierarchy.py"
    if not path.exists():
        pytest.skip("Full repository source is required for this integration guard")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    saves = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "save_hierarchy"]
    flushes = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "flush_resident_state"]
    assert len(flushes) == 2
    assert saves and flushes[0] < min(saves)
    resets = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "densification_criterium_cache" for t in n.targets)]
    assert max(resets) > flushes[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required for fused-SSIM parity")
def test_coarse_fused_ssim_matches_reference_value_and_gradient():
    from fused_ssim import fused_ssim
    from utils.loss_utils import ssim
    torch.manual_seed(17)
    image = torch.rand(1, 3, 32, 48, device="cuda", requires_grad=True)
    target = torch.rand_like(image)
    reference = ssim(image, target)
    fused = fused_ssim(image, target)
    reference_grad, = torch.autograd.grad(reference, image, retain_graph=True)
    fused_grad, = torch.autograd.grad(fused, image)
    torch.testing.assert_close(fused, reference, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(fused_grad, reference_grad, rtol=1e-3, atol=1e-4)
