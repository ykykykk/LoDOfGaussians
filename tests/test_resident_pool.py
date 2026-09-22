from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import math

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.resident_pool import ResidentPool, capacity_for_budget
from utils.resident_selection import split_upper_cut
from utils.training_profile import TrainingProfile


def backing(n=12, d=23):
    generator = torch.Generator().manual_seed(13)
    host = torch.zeros(n, 3*d)
    host[:, :d] = torch.randn(n, d, generator=generator)
    return host, torch.zeros(n)


def reference_step(state, grad, rates, iteration):
    d = grad.shape[1]
    p, m, v = state.split(d, dim=1)
    m.mul_(.9).add_(grad, alpha=1-.9)
    v.mul_(.999).addcmul_(grad, grad, value=1-.999)
    denom = v.sqrt().div_(math.sqrt(1-.999**(iteration+1))).add_(1e-8)
    # Same scalar addcdiv as upstream, independently for each property column.
    for c in range(d):
        p[:, c].addcdiv_(m[:, c], denom[:, c], value=-float(rates[c])/(1-.9**(iteration+1)))


def test_same_packet_reuses_storage_without_upload_or_gather():
    h, s = backing()
    pool = ResidentPool(h, s, 8, device="cpu")
    p = pool.acquire([3, 1, 7])
    ptr = p.state.data_ptr()
    p.adam_step(torch.ones(3, 23), torch.full((23,), .01), 0)
    p2 = pool.acquire([3, 1, 7])
    assert p2 is p and p2.state.data_ptr() == ptr
    assert pool.stats["uploaded_rows"] == 3
    assert pool.stats["packet_gathers"] == 1
    assert pool.stats["same_packet_hits"] == 1
    pool.close()


@pytest.mark.parametrize("d", [14, 23, 38, 59])
def test_eviction_save_reload_matches_unpaged_reference(d):
    h, scores = backing(30, d)
    ref = h.clone()
    ref_scores = scores.clone()
    pool = ResidentPool(h, scores, 7, device="cpu", transfer_rows=2)
    rng = np.random.default_rng(12)
    rates = torch.linspace(.00001, .03, d)
    for iteration in range(45):
        count = [4, 4, 9, 7, 2][iteration % 5]  # Includes a cut > capacity.
        ids = rng.choice(len(h), count, replace=False)
        packet = pool.acquire(ids)
        torch.testing.assert_close(packet.state, ref[ids], rtol=2e-5, atol=2e-7)
        torch.testing.assert_close(packet.scores, ref_scores[ids])
        grad = torch.from_numpy(rng.normal(size=(count, d)).astype(np.float32))
        score = grad[:, :2].norm(dim=-1)
        packet.accumulate_scores(torch.arange(count), score)
        ref_scores[ids] = torch.maximum(ref_scores[ids], score)
        packet.adam_step(grad, rates, iteration)
        rows = ref[ids].clone()
        reference_step(rows, grad, rates, iteration)
        ref[ids] = rows
        if iteration % 11 == 0:
            pool.flush()
            pool.invalidate()
    pool.flush()
    torch.testing.assert_close(h, ref, rtol=2e-5, atol=2e-7)
    torch.testing.assert_close(scores, ref_scores)
    assert pool.stats["overflow_steps"] > 0
    assert pool.stats["evicted_rows"] > 0
    pool.close()


def test_cold_rows_do_not_move_or_decay_adam():
    h, s = backing()
    pool = ResidentPool(h, s, 6, device="cpu")
    pool.acquire([0, 1, 2]).adam_step(torch.ones(3, 23), torch.ones(23)*.01, 0)
    p = pool.acquire([3, 4])
    slots = pool.to_slot[:3].copy()
    cold = pool.state[slots].clone()
    p.adam_step(torch.ones(2, 23), torch.ones(23)*.01, 1)
    pool.acquire([4, 5])
    assert np.array_equal(slots, pool.to_slot[:3])
    torch.testing.assert_close(pool.state[slots], cold)
    pool.close()


def test_topology_barrier_reloads_updated_host_rows():
    h, s = backing()
    pool = ResidentPool(h, s, 4, device="cpu")
    packet = pool.acquire([0, 1])
    packet.accumulate_scores(torch.tensor([0, 1]), torch.tensor([2., 3.]))
    with pytest.raises(RuntimeError, match="flush"):
        pool.invalidate()
    pool.flush()
    pool.invalidate()
    h[1, 0] = 100.
    s[1] = 5.
    packet = pool.acquire([1, 6])
    assert packet.state[0, 0] == 100.
    assert packet.scores[0] == 5.
    pool.reset_scores(len(h))
    assert packet.scores.sum() == 0 and s.sum() == 0
    pool.close()


def test_partial_scores_and_frozen_skybox():
    h, s = backing()
    old = h.clone()
    pool = ResidentPool(h, s, 4, device="cpu")
    packet = pool.acquire([0, 1, 5])
    packet.accumulate_scores(torch.tensor([2, 2, 0]), torch.tensor([1., 3., 2.]))
    packet.adam_step(torch.ones(3, 23), torch.ones(23)*.01, 0, frozen_prefix=2)
    pool.close()
    torch.testing.assert_close(h[:2], old[:2])
    assert not torch.equal(h[5], old[5])
    assert s[5] == 3. and s[0] == 2. and s[1] == 0.


def test_overflow_preserves_full_cut_and_releases_pool():
    h, s = backing()
    pool = ResidentPool(h, s, 2, device="cpu")
    pool.acquire([1, 3]).adam_step(torch.ones(2, 23), torch.ones(23)*.01, 0)
    p = pool.acquire([1, 3, 4, 6])
    assert p.slots is None and p.state.shape == (4, 69)
    assert pool.state is None and pool.used == 0
    p.adam_step(torch.ones(4, 23), torch.ones(23)*.01, 1)
    p = pool.acquire([6, 3])
    torch.testing.assert_close(p.state, h[[6, 3]])
    pool.close()


@pytest.mark.parametrize("capacity", [0, 1, 12])
def test_empty_cut_and_capacities(capacity):
    h, s = backing()
    pool = ResidentPool(h, s, capacity, device="cpu")
    assert pool.acquire(np.empty(0, dtype=np.int64)).state.shape == (0, 69)
    pool.acquire([2])
    pool.close()


@pytest.mark.parametrize("ids,exception", [([1, 1], ValueError), ([-1], IndexError), ([12], IndexError), ([1.5], ValueError), ([[1]], ValueError)])
def test_invalid_ids(ids, exception):
    h, s = backing()
    with pytest.raises(exception):
        ResidentPool(h, s, 4, device="cpu").acquire(ids)


def test_request_hits_are_never_evicted():
    h, s = backing()
    pool = ResidentPool(h, s, 3, device="cpu")
    pool.acquire([0, 1, 2])
    kept = pool.to_slot[[0, 2]].copy()
    pool.acquire([0, 2, 3])
    assert np.array_equal(kept, pool.to_slot[[0, 2]])
    assert pool.to_slot[1] == -1
    pool.close()


def test_parameter_leaf_does_not_allocate_full_pool_gradient():
    h, s = backing(50)
    pool = ResidentPool(h, s, 40, device="cpu")
    packet = pool.acquire([3, 9])
    leaf = packet.parameters()
    leaf.square().mean().backward()
    assert leaf.grad.shape == (2, 23)
    assert pool.state.grad is None and not pool.state.requires_grad
    pool.close()


def test_memory_budget_is_bounded():
    n = capacity_for_budget(8_000_000, 23, 24*2**30, 2.5, 4.)
    assert n == 8_000_000
    assert capacity_for_budget(8_000_000, 23, 3*2**30, 2.5, 4.) == 0
    assert capacity_for_budget(100, 59, 24*2**30, 2.5, 4.) == 100


def test_shared_cuda_cache_is_not_extra_physical_vram(monkeypatch):
    from utils.resident_pool import cuda_available_bytes
    gib = 2**30
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda: (2*gib, 24*gib))
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda: 6*gib)
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda: 36*gib)
    assert cuda_available_bytes() == 18*gib
    assert cuda_available_bytes(4*gib) == 22*gib
    assert cuda_available_bytes(8*gib) == 24*gib
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda: 8*gib)
    assert cuda_available_bytes() == 4*gib


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_resize_releases_old_store():
    host = torch.zeros(131072, 69)
    pool = ResidentPool(host, torch.zeros(len(host)), 65536, device='cuda')
    packet = pool.acquire([1, 3])
    packet.adam_step(torch.ones(2, 23, device='cuda'), torch.ones(23, device='cuda')*.01, 0)
    pool.flush(); pool.invalidate()
    del packet
    before = torch.cuda.memory_reserved()
    pool.resize_empty(131072)
    assert torch.cuda.memory_reserved() < before
    torch.testing.assert_close(pool.acquire([3, 1]).state.cpu(), host[[3, 1]])
    pool.close()


def test_spt_zero_and_terminal_parents_are_disjoint():
    # SPT 0, ordinary leaf, terminal coarse internal node, SPT 1.
    nodes = torch.tensor([[0, -1, 0, 0, 0, 10], [1, 0, 0, -1, 0, 11],
                          [1, 0, 2, 4, 0, 12], [1, 0, 0, 1, 0, 13]])
    spt_nodes, direct = split_upper_cut(nodes, torch.arange(4))
    assert spt_nodes.tolist() == [0, 3]
    assert direct.tolist() == [11, 12]


def test_profile_records_cpu_without_cuda(tmp_path):
    p = TrainingProfile(tmp_path / "profile.jsonl", every=2, device="cpu")
    for i in range(3):
        p.begin(i)
        with p.phase("cache_prepare"):
            sum(range(20))
        p.finish(active=10, uploaded_rows=2)
    p.close()
    rows = [json.loads(x) for x in (tmp_path / "profile.jsonl").read_text().splitlines()]
    assert [r["iteration"] for r in rows] == [0, 2]
    assert rows[0]["phases"]["cache_prepare"]["host_ms"] >= 0
    assert rows[0]["phases"]["cache_prepare"]["cuda_span_ms"] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_cuda_pinned_chunk_reuse_and_adam_equivalence():
    h, scores = backing(64)
    ref = h.clone()
    pool = ResidentPool(h, scores, 19, transfer_rows=3, pin_staging=True)
    rates = torch.linspace(.00001, .02, 23)
    for i, ids in enumerate([np.arange(17), np.arange(5, 25), np.arange(4, 20), np.arange(4, 20)]):
        p = pool.acquire(ids)
        torch.testing.assert_close(p.state.cpu(), ref[ids], rtol=3e-5, atol=1e-6)
        grad = torch.ones(len(ids), 23)
        p.adam_step(grad.cuda(), rates.cuda(), i)
        rows = ref[ids].clone()
        reference_step(rows, grad, rates, i)
        ref[ids] = rows
    pool.close()
    torch.testing.assert_close(h, ref, rtol=3e-5, atol=1e-6)


def test_relocation_constants_are_lazy_and_match_original():
    from utils.reloc_utils import _binomial_table, N_max
    _binomial_table.cache_clear()
    # A CPU import and the CPU reference table must never require CUDA.
    assert _binomial_table.cache_info().currsize == 0
    table = _binomial_table("cpu")
    original = torch.zeros(N_max, N_max)
    for n in range(N_max):
        for k in range(n + 1):
            original[n, k] = math.comb(n, k)
    torch.testing.assert_close(table, original, rtol=0, atol=0)
    assert _binomial_table("cpu").data_ptr() == table.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA + gsplat required")
def test_cuda_gsplat_packet_forward_gradient_equivalence():
    gsplat = pytest.importorskip("gsplat")
    d, n = 23, 8
    h, scores = backing(n, d)
    h[:, :3] *= .1
    h[:, 2] += 3
    h[:, 3:6] = -2
    h[:, 6:10] = torch.tensor([1., 0., 0., 0.])
    h[:, 13] = 0
    pool = ResidentPool(h, scores, n, device="cuda")
    packet = pool.acquire(np.arange(n))
    packed = packet.parameters()
    # Six independent parameter groups reproduce the legacy render inputs.
    reference = [h[:, a:b].cuda().contiguous().requires_grad_(True)
                 for a, b in [(0, 3), (3, 6), (6, 10), (10, 13), (13, 14), (14, d)]]

    def render(groups):
        xyz, scale, quat, dc, opacity, rest = groups
        view = torch.eye(4, device="cuda")[None]
        k = torch.tensor([[40., 0., 16.], [0., 40., 16.], [0., 0., 1.]], device="cuda")[None]
        return gsplat.rasterization(
            means=xyz.contiguous(), quats=torch.nn.functional.normalize(quat, dim=-1),
            scales=scale.exp(), opacities=opacity.sigmoid().squeeze(-1),
            colors=torch.cat([dc[:, None], rest.reshape(n, 3, 3)], dim=1),
            viewmats=view, Ks=k, width=32, height=32, sh_degree=1, packed=True,
        )[0]
    out1 = render([packed[:, a:b] for a, b in [(0, 3), (3, 6), (6, 10), (10, 13), (13, 14), (14, d)]])
    out2 = render(reference)
    out1.square().mean().backward()
    out2.square().mean().backward()
    torch.testing.assert_close(out1, out2, rtol=1e-5, atol=1e-6)
    grads = torch.cat([x.grad for x in reference], dim=1)
    torch.testing.assert_close(packed.grad, grads, rtol=2e-4, atol=1e-6)
    pool.close()


def test_trainer_control_flow_flushes_splits_and_saves(tmp_path, monkeypatch):
    """Execute the real trainer/packet/Adam flow; fake only scene and CUDA render.

    This is an integration/control-flow test, NOT a CUDA end-to-end benchmark.
    """
    import types
    import train_resident as tr
    instances = []
    shutdowns = []

    class Model:
        def __init__(self, degree):
            self.max_sh_degree = self.active_sh_degree = degree
            self.skybox_points = 1
            self.spatial_lr_scale = 1.
            self.size = 8
            self.nodes = torch.zeros(32, 6, dtype=torch.int32)
            self.properties = torch.zeros(32, 69)
            self.properties[:, 6] = 1.
            self._densification_criterium = torch.zeros(32)
            self.rebuilds = 0
            self.saves = []
            self.splits = 0
            for name in ("_xyz", "_opacity", "_rotation", "_scaling", "_features_dc", "_features_rest"):
                setattr(self, name, torch.zeros(1))
            instances.append(self)
        def compact_gaussians(self, *a, **kw):
            pass
        def build_hierarchical_SPT(self, *a, **kw):
            self.rebuilds += 1
        def rotation_activation(self, x):
            return torch.nn.functional.normalize(x, dim=-1)
        def oneupSHdegree(self):
            self.active_sh_degree = min(self.active_sh_degree + 1, self.max_sh_degree)
        def add_new_gs(self, **kw):
            assert self._densification_criterium.sum() > 0
            self.splits += 1
            self.properties[self.size:self.size + 2] = self.properties[2]
            self.size += 2
        def relocate_gs(self, mask, *a, **kw):
            assert mask.shape == (self.size,)
        def save_hierarchy(self, *a, **kw):
            self.saves.append((kw["file_name"], self.properties.clone()))

    camera = SimpleNamespace(focal_length=100., invdepthmap=None, image_name="fixture",
                             original_image=torch.zeros(3, 2, 2), alpha_mask=torch.ones(1, 2, 2),
                             world_view_transform=torch.eye(4), projection_matrix=torch.eye(4),
                             full_proj_transform=torch.eye(4), camera_center=torch.zeros(3))
    class Scene:
        def __init__(self, *a, **kw):
            pass
        def getTrainCameras(self):
            return [camera, camera]

    def render(cam, xyz, opacity, scale, rot, dc, rest, *args, **kw):
        screen = xyz[:, :2]
        screen.retain_grad()
        value = (screen.sum() + xyz[:, 2].sum() + opacity.sum() + scale.sum()
                 + rot.sum() + dc.sum() + rest.sum()) * .01
        return {"render": value.sigmoid().expand(3, 2, 2), "viewspace_points": screen,
                "packed_indices": torch.arange(len(xyz))}

    monkeypatch.setitem(sys.modules, "scene", SimpleNamespace(Scene=Scene, GaussianModel=Model))
    monkeypatch.setitem(sys.modules, "gaussian_renderer", SimpleNamespace(render_gsplat=render))
    monkeypatch.setitem(sys.modules, "utils.loss_utils", SimpleNamespace(l1_loss=lambda a, b: (a-b).abs().mean()))
    monkeypatch.setitem(sys.modules, "fused_ssim", SimpleNamespace(fused_ssim=lambda a, b: 1-(a-b).square().mean()))
    monkeypatch.setitem(sys.modules, "utils.general_utils", SimpleNamespace(get_expon_lr_func=lambda **kw: lambda i: kw["lr_init"]))
    monkeypatch.setitem(sys.modules, "utils.view_graph_utils", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "utils.training_runtime", SimpleNamespace(
        make_camera_loader=lambda cameras, opt, **kw: [[c] for c in cameras],
        shutdown_camera_loader=lambda loader: shutdowns.append(True)))
    real_pool = tr.ResidentPool
    monkeypatch.setattr(tr, "ResidentPool", lambda *a, **kw: real_pool(*a, **kw, device="cpu"))
    real_profile = tr.TrainingProfile
    monkeypatch.setattr(tr, "TrainingProfile", lambda *a: real_profile(*a, device="cpu"))
    monkeypatch.setattr(tr, "select_gaussians", lambda *a: torch.tensor([0, 2, 5]))
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **kw: self)
    real_tensor = torch.tensor
    def tensor(*a, **kw):
        if kw.get("device") == "cuda":
            kw["device"] = "cpu"
        return real_tensor(*a, **kw)
    monkeypatch.setattr(torch, "tensor", tensor)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (24*2**30, 24*2**30))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "CPU-test-renderer")
    for name in ("memory_allocated", "memory_reserved", "max_memory_allocated"):
        monkeypatch.setattr(torch.cuda, name, lambda: 0)
    opt = SimpleNamespace(storage_device="cpu", densification="classic", prune_unused=False,
        dampen_scale_grad=False, optimize_exposure=False, use_occlusion_culling=False,
        SH_degree=1, cap_max=32, llff_hold=100, target_granularity_pixels=2, SPT_root_volume=25,
        min_SPT_size=256, use_bounding_spheres=False, cache_size=6, graph_view_select=False,
        position_lr_init=.01, position_lr_final=.001, position_lr_delay_mult=.01,
        position_lr_max_steps=6, iterations=6, vary_distance_multiplier=False,
        SH_increase_after_train_percent=.5, output_file_name="result.dhier", lambda_dssim=.2,
        densify_from_iter=0, densify_until_iter=5, densification_interval=2, densify_percent=1.02,
        densify_grad_threshold=1e-6, scaling_lr=.005, rotation_lr=.001, feature_lr=.0025,
        opacity_lr=.05, lr_multiplier=1.)
    dataset = SimpleNamespace(output_path=str(tmp_path), white_background=False, resolution=1)
    tr.training(dataset, opt, SimpleNamespace(debug=True), saving_iterations=[3],
                runtime={"profile_every": 1})
    g = instances[0]
    assert g.splits == 2 and g.rebuilds == 3
    assert [name for name, _ in g.saves] == ["iteration_3", "result.dhier"]
    assert g.saves[-1][1][2, 0] != 0  # Latest active GPU/packet parameters were saved.
    assert g.properties[0, 0] == 0  # Skybox stayed frozen.
    assert shutdowns == [True]
    assert len((tmp_path / "resident_profile.jsonl").read_text().splitlines()) == 7


def test_small_eviction_does_not_sort_or_scan_entire_large_pool():
    h, s = backing(10000, 14)
    pool = ResidentPool(h, s, 8192, device="cpu")
    pool.acquire(np.arange(8192))
    slots = pool.to_slot[[5000, 6000]].copy()
    pool.acquire([5000, 6000, 9000])
    assert np.array_equal(slots, pool.to_slot[[5000, 6000]])
    assert pool.stats["eviction_scan_rows"] == 4096
    assert pool.stats["evicted_rows"] == 1
    pool.close()


def test_float_tensor_ids_are_not_silently_truncated():
    h, s = backing()
    pool = ResidentPool(h, s, 4, device="cpu")
    with pytest.raises(ValueError, match="integer"):
        pool.acquire(torch.tensor([1.5]))
    pool.close()
