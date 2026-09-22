from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
import math
import time
import numpy as np
import pytest
import torch

from utils.incremental_spt import IncrementalSPT
from utils.resident_pool import ActivePacket
from utils.resident_pool_v2 import StreamingResidentPool, IndexedPacket
from utils.view_pipeline import CachedCameras, ThreadedViews, CameraTransfer, camera_bytes
from utils.adam_graph import tensor_adam_step, PacketAdamGraph


class TorchOps:
    @staticmethod
    def gather_parameters(state, slots):
        return state[slots, :state.shape[1] // 3].contiguous()

    @staticmethod
    def indexed_adam(state, slots, raw, gradient, rates, frozen, c1, c2):
        d = raw.shape[1]
        rows = state.index_select(0, slots)
        grad = gradient.masked_fill(frozen[:, None], 0)
        m, v = rows[:, d:2*d], rows[:, 2*d:]
        m.mul_(0.9).add_(grad, alpha=0.1)
        v.mul_(0.999).addcmul_(grad, grad, value=0.001)
        raw.sub_(m / (v.sqrt() / c2 + 1e-8) * (rates / c1))
        rows[:, :d] = raw
        state.index_copy_(0, slots, rows)


def backing(n=20, d=23):
    gen = torch.Generator().manual_seed(2)
    data = torch.randn(n, d * 3, generator=gen)
    data[:, 2*d:] = data[:, 2*d:].abs()
    return data, torch.zeros(n)


@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("capacity", [0, 3, 8, 20])
def test_random_train_prefetch_evict_reload_matches_reference(indexed, capacity):
    host, scores = backing()
    expected, expected_scores = host.clone(), scores.clone()
    pool = StreamingResidentPool(host, scores, capacity, device="cpu", transfer_rows=2,
                                 prefetch_rows=4, ops=TorchOps if indexed else None)
    rng = np.random.default_rng(99)
    gen = torch.Generator().manual_seed(90)
    cuts = [rng.choice(20, size=int(rng.integers(1, 8)), replace=False) for _ in range(25)]
    for step, ids in enumerate(cuts):
        packet = pool.acquire(ids)
        if step + 1 < len(cuts):
            pool.prefetch(cuts[step + 1], epoch=pool.epoch)
        grad = torch.randn(len(ids), 23, generator=gen)
        rates = torch.linspace(0.00001, 0.05, 23)
        visible = torch.arange(len(ids))
        detail = torch.rand(len(ids), generator=gen)
        packet.accumulate_scores(visible, detail)
        packet.adam_step(grad, rates, step, frozen_prefix=1)
        ref = ActivePacket(ids, None, expected[ids].clone(), expected_scores[ids].clone())
        ref.accumulate_scores(visible, detail)
        ref.adam_step(grad, rates, step, frozen_prefix=1)
        expected[ids] = ref.state
        expected_scores[ids] = ref.scores
        if step % 7 == 0:
            pool.flush()
            torch.testing.assert_close(host, expected, rtol=2e-6, atol=2e-7)
    pool.close()
    torch.testing.assert_close(host, expected, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(scores, expected_scores)


def test_indexed_reuses_packet_without_moment_gather():
    host, scores = backing()
    pool = StreamingResidentPool(host, scores, 8, device="cpu", ops=TorchOps)
    a = pool.acquire([3, 1, 4])
    assert isinstance(a, IndexedPacket)
    assert a.state.shape == (3, 23)
    a.adam_step(torch.ones_like(a.state), torch.ones(23)*0.002, 7)
    assert pool.acquire([3, 1, 4]) is a
    assert pool.stats['indexed_packet_gathers'] == 1
    assert pool.stats['avoided_moment_gather_bytes'] == 3*23*8
    pool.close()


def test_prefetch_preserves_active_and_dirty_slots_and_epoch():
    host, scores = backing()
    pool = StreamingResidentPool(host, scores, 4, device="cpu", ops=TorchOps,
                                 transfer_rows=2, prefetch_rows=4)
    first = pool.acquire([0, 1])
    first.adam_step(torch.ones_like(first.state), torch.ones(23)*0.001, 1)
    pool.acquire([2, 3])
    assert pool.dirty[pool.to_slot[[0, 1]]].all()
    assert pool.prefetch([4, 5]) == 0
    assert (pool.to_slot[:4] >= 0).all()
    assert pool.prefetch([4], epoch=pool.epoch + 1) == 0
    pool.flush()
    assert pool.prefetch([4, 5]) == 2
    assert (pool.to_slot[[2, 3]] >= 0).all()
    old_epoch = pool.epoch
    pool.invalidate()
    assert pool.epoch == old_epoch + 1
    assert pool.prefetch([7], epoch=old_epoch) == 0
    assert (pool.to_slot < 0).all()
    pool.close()


def test_prefetch_dedup_and_bounds():
    host, scores = backing()
    pool = StreamingResidentPool(host, scores, 8, device="cpu", transfer_rows=2, prefetch_rows=100)
    pool.acquire([0])
    assert pool.prefetch([1, 1, 2, 3, 4, 5, 6]) == 4
    assert pool.used == 5
    with pytest.raises(IndexError):
        pool.prefetch([-1])
    with pytest.raises(ValueError):
        pool.acquire([1, 1])
    pool.close()


def tree(depth=4):
    n = 2**(depth+1)-1
    props = torch.zeros(n + 100, 69)
    nodes = torch.zeros(n + 100, 6, dtype=torch.int32)
    gen = torch.Generator().manual_seed(30)
    props[:n, :3] = torch.rand(n, 3, generator=gen)*0.1
    for i in range(n):
        level = int(math.log2(i+1))
        nodes[i, 0] = level
        nodes[i, 1] = (i-1)//2 if i else -1
        props[i, 3:6] = math.log(4.0 if i == 0 else 0.5 / 2**level)
        if 2*i+2 < n:
            nodes[i, 2:4] = torch.tensor([2, 2*i+1])
            nodes[2*i+1, 4] = 2*i+2
    return SimpleNamespace(properties=props, nodes=nodes, size=n, skybox_points=0)


FIELDS = ('SPT_gaussian_indices', 'SPT_min', 'SPT_max', 'SPT_starts',
          'upper_tree_nodes', 'upper_tree_xyz', 'upper_tree_scaling', 'min_distance_squared',
          'upper_cut_order', 'bounding_sphere_radii')


def assert_full_equal(g, manager):
    fresh = deepcopy(g)
    full = IncrementalSPT(manager.root_volume, manager.granularity, manager.min_size,
                          manager.bounding_spheres, device="cpu", scan_rows=7)
    full.refresh(fresh)
    for name in FIELDS:
        if hasattr(fresh, name):
            torch.testing.assert_close(getattr(g, name), getattr(fresh, name), rtol=0, atol=0)


@pytest.mark.parametrize('spheres', [False, True])
def test_incremental_clean_geometry_split_and_nonlocal_reparent(spheres):
    g = tree()
    manager = IncrementalSPT(1, 0.01, 2, spheres, device='cpu', scan_rows=7)
    manager.refresh(g)
    assert manager.stats['spt_rebuilt'] == 2
    assert_full_equal(g, manager)
    manager.refresh(g)
    assert manager.stats['spt_reused'] == 2
    assert manager.stats['spt_uploaded_rows'] == 0
    # Editing color/rotation/opacity cannot change an SPT's spatial ranges.
    g.properties[17, 6:14] += 0.2
    manager.refresh(g)
    assert manager.stats['spt_reused'] == 2
    # A descendant geometry edit invalidates its entire owning subtree only.
    g.properties[17, 0] += 0.8
    manager.refresh(g)
    assert manager.stats['spt_rebuilt'] == 1
    assert manager.stats['spt_layout_repack'] == 0
    assert_full_equal(g, manager)
    # A true leaf split appends nodes and changes the old parent.
    old = g.size
    g.nodes[17, 2:4] = torch.tensor([2, old])
    g.nodes[old:old+2, 1] = 17
    g.nodes[old:old+2, 0] = 5
    g.nodes[old, 4] = old+1
    g.properties[old:old+2] = g.properties[17]
    g.size += 2
    manager.refresh(g)
    assert manager.stats['spt_rebuilt'] == 1
    assert_full_equal(g, manager)
    # Swap complete subtrees 3 and 5 across SPT roots 1 and 2.
    g.nodes[1, 3], g.nodes[2, 3] = 5, 3
    g.nodes[5, 1], g.nodes[3, 1] = 1, 2
    g.nodes[5, 4], g.nodes[3, 4] = 4, 6
    manager.refresh(g)
    assert manager.stats['spt_rebuilt'] == 2
    assert_full_equal(g, manager)


def test_incremental_partition_threshold_and_small_trees():
    g = tree()
    manager = IncrementalSPT(1, .02, 2, True, device='cpu')
    manager.refresh(g)
    g.properties[1, 3:6] = math.log(2)
    manager.refresh(g)
    assert manager.stats['spt_layout_repack'] == 1
    assert_full_equal(g, manager)
    g = tree(0)
    manager = IncrementalSPT(1, .02, 100, True, device='cpu')
    manager.refresh(g)
    assert g.SPT_starts.tolist() == [0]
    assert g.upper_tree_nodes[0, 3].item() == -1
    assert_full_equal(g, manager)


def test_incremental_cpu_scene_has_nonzero_root():
    g = tree(3)
    offset = 3
    g.properties[offset:offset+g.size] = g.properties[:g.size].clone()
    shifted = g.nodes[:g.size].clone()
    shifted[:, 1] += offset
    shifted[0, 1] = -1
    shifted[shifted[:, 2] > 0, 3] += offset
    shifted[shifted[:, 4] > 0, 4] += offset
    g.nodes[offset:offset+g.size] = shifted
    g.nodes[:offset] = -99
    g.skybox_points, g.size = offset, g.size+offset
    manager = IncrementalSPT(1, .02, 2, True, device='cpu')
    manager.refresh(g)
    assert g.upper_tree_nodes[0, 5] == offset
    assert_full_equal(g, manager)


def camera(i=0):
    return SimpleNamespace(uid=i, original_image=torch.ones(3, 8, 8), alpha_mask=torch.ones(1, 8, 8),
                           world_view_transform=torch.eye(4), projection_matrix=torch.eye(4),
                           full_proj_transform=torch.eye(4), camera_center=torch.ones(3), invdepthmap=None)


class Cameras:
    def __len__(self): return 5
    def __getitem__(self, i): return camera(i)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Pinned memory requires CUDA")
def test_pinned_cache_reuses_storage_and_upload_preserves_pixels():
    size = camera_bytes(camera())
    data = CachedCameras(Cameras(), size * 2, pin_cache=True)
    first = data[0]
    assert first.original_image.is_pinned()
    assert first.original_image.data_ptr() == data[0].original_image.data_ptr()
    transfer = CameraTransfer()
    try:
        uploaded = transfer.ready(transfer.submit(first, False))
        torch.testing.assert_close(uploaded.original_image.cpu(), first.original_image)
    finally:
        transfer.close()
    data[1]; data[2]
    assert data.used_bytes <= size * 2 and set(data.cache) == {1, 2}


def test_decoded_cache_bounded_and_object_mutations_isolated():
    size = camera_bytes(camera())
    data = CachedCameras(Cameras(), size*2)
    first = data[0]
    first.original_image = torch.zeros(3,8,8)
    assert data[0].original_image.sum() == 192
    data[1]; data[0]; data[2]
    assert set(data.cache) == {0,2}
    assert data.used_bytes <= size*2
    assert not data.__getstate__()['cache']
    tiny = CachedCameras(Cameras(), 1)
    tiny[0]
    assert not tiny.cache


@pytest.mark.parametrize('threaded', [False, True])
def test_cpu_view_lookahead_order_and_exhaustion(threaded):
    views = ThreadedViews([[camera(i)] for i in range(3)], count=8, enabled=threaded)
    assert [views.pop().uid for _ in range(8)] == [0,1,2,0,1,2,0,1]
    assert views.pop() is None
    views.close()


def test_camera_upload_cpu_reference_does_not_mutate_cached_camera():
    transfer = CameraTransfer(device='cpu')
    original = camera(4)
    ticket = transfer.submit(original)
    result = transfer.ready(ticket)
    result.original_image = torch.zeros(3,8,8)
    assert original.original_image.sum() == 192
    assert transfer.metadata(ticket).uid == 4
    transfer.close()


def test_adam_graph_math_matches_reference_dynamic_rates_steps():
    state, scores = backing(4)
    expected = state.clone()
    p = ActivePacket(np.arange(4), None, expected, scores)
    for step in (0,1,50,10000):
        gradient = torch.randn(4,23)
        rates = torch.linspace(1e-7, .001, 23)
        p.adam_step(gradient, rates, step)
        tensor_adam_step(state, gradient, rates,
                         torch.tensor([1-.9**(step+1), math.sqrt(1-.999**(step+1))]))
        torch.testing.assert_close(state, expected, rtol=2e-6, atol=2e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA and a C++/CUDA toolchain')
def test_cuda_indexed_adam_and_async_prefetch_values():
    from utils.resident_native import load_native
    native = load_native('cuda')
    host, scores = backing()
    expected = host.clone()
    pool = StreamingResidentPool(host, scores, 8, device='cuda', ops=native,
                                 transfer_rows=2, prefetch_rows=4)
    cuts = ([0,1,2], [2,3,4], [4,7,8], [0,9,10])
    for step, ids in enumerate(cuts):
        packet = pool.acquire(ids)
        if step+1 < len(cuts):
            pool.prefetch(cuts[step+1])
        grad = torch.randn(len(ids),23)
        rates = torch.linspace(.00001,.003,23)
        packet.adam_step(grad.cuda(), rates.cuda(), step, 1)
        ref = ActivePacket(np.array(ids), None, expected[ids].clone(), torch.zeros(len(ids)))
        ref.adam_step(grad, rates, step, 1)
        expected[ids] = ref.state
    pool.close()
    torch.testing.assert_close(host, expected, rtol=2e-5, atol=3e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_cuda_adam_graph_replay_updates_bias_and_lr():
    state, scores = backing(4)
    packet = ActivePacket(np.arange(4), None, state.cuda(), scores.cuda())
    reference = ActivePacket(np.arange(4), None, state.clone(), scores.clone())
    graph = PacketAdamGraph(minimum_reuse=1)
    for step in (0,1,5,23):
        grad, rates = torch.randn(4,23), torch.rand(23)*.001
        assert graph.step(packet, grad.cuda(), rates.cuda(), step, 1)
        reference.adam_step(grad, rates, step, 1)
        torch.testing.assert_close(packet.state.cpu(), reference.state, rtol=2e-5, atol=3e-6)
    assert graph.captures == 1 and graph.replays == 4
    graph.reset()


def load_upstream_spt_reference(monkeypatch):
    """Run repository methods with CPU allocation, not a second v2 builder.

    CUDA changes are limited to device placement; hierarchy/range algorithms
    are compiled unchanged from the existing GaussianModel source.
    """
    import ast
    source = (Path(__file__).resolve().parents[1] / 'scene/gaussian_model.py').read_text(encoding='utf-8')
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'GaussianModel')
    names = {'build_hierarchical_SPT', 'get_min_distance', 'cut_hierarchy_on_condition'}
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    class CPU(ast.NodeTransformer):
        def visit_Constant(self, n):
            return ast.copy_location(ast.Constant('cpu'), n) if n.value == 'cuda' else n
    module = CPU().visit(ast.Module(body=[cls], type_ignores=[]))
    ast.fix_missing_locations(module)
    env = dict(torch=torch, scales1=3, scales2=6, xyz1=0, xyz2=3,
               hierarchy_node_depth=0, hierarchy_node_parent=1, hierarchy_node_child_count=2,
               hierarchy_node_first_child=3, hierarchy_node_next_sibling=4, hierarchy_node_max_side_length=5)
    exec(compile(module, '<upstream SPT CPU reference>', 'exec'), env)
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **k: self)
    return env['GaussianModel']


@pytest.mark.parametrize('volume,min_size,bounds', [(3., 2, False), (1., 2, True), (100., 2, True), (1., 10000, True)])
def test_incremental_ranges_match_existing_model_builder(monkeypatch, volume, min_size, bounds):
    original = load_upstream_spt_reference(monkeypatch)
    g = tree()
    # The old method's default root is 100000. Supply the real root in the
    # test wrapper; this is independent of the new traversal implementation.
    ref = original()
    ref.properties, ref.nodes = g.properties.clone(), g.nodes.clone()
    ref.size, ref.skybox_points = g.size, 0
    ref.scaling_activation = torch.exp
    method = ref.cut_hierarchy_on_condition
    ref.cut_hierarchy_on_condition = lambda *a, **k: method(*a, root_node=0, **k)
    ref.build_hierarchical_SPT(volume, .05, min_size, use_bounding_spheres=bounds)
    manager = IncrementalSPT(volume, .05, min_size, bounds, device='cpu')
    manager.refresh(g)
    torch.testing.assert_close(g.upper_tree_nodes, ref.upper_tree_nodes)
    torch.testing.assert_close(g.min_distance_squared, ref.min_distance_squared)
    torch.testing.assert_close(g.SPT_starts, ref.SPT_starts)
    # The old sort is unstable for equal max-distance keys; compare each
    # subtree by global ID, not unspecified tie order.
    for a, b in zip(g.SPT_starts[:-1], g.SPT_starts[1:]):
        a, b = int(a), int(b)
        oi = torch.argsort(g.SPT_gaussian_indices[a:b]) + a
        ri = torch.argsort(ref.SPT_gaussian_indices[a:b]) + a
        torch.testing.assert_close(g.SPT_gaussian_indices[oi], ref.SPT_gaussian_indices[ri])
        torch.testing.assert_close(g.SPT_min[oi], ref.SPT_min[ri])
        torch.testing.assert_close(g.SPT_max[oi], ref.SPT_max[ri])
    if bounds:
        torch.testing.assert_close(g.bounding_sphere_radii, ref.bounding_sphere_radii)


def test_view_schedule_does_not_consume_global_rng_and_repeats_deterministically():
    from utils.view_pipeline import ViewSchedule
    before = torch.get_rng_state().clone()
    a = list(ViewSchedule(5, 17, 123))
    b = list(ViewSchedule(5, 17, 123))
    assert a == b and len(a) == 17
    assert sorted(a[:5]) == list(range(5))
    assert sorted(a[5:10]) == list(range(5))
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA device required')
def test_native_upper_cut_matches_reference_bfs_including_order():
    from utils.resident_native import load_native
    ops = load_native('cuda')
    g = tree()
    builder = IncrementalSPT(1., .05, 2, True, device='cuda')
    builder.refresh(g)
    nodes = g.upper_tree_nodes
    planes = torch.tensor([[1., 0., 0., 1.], [-1., 0., 0., 1.],
                           [0., 1., 0., 1.], [0., -1., 0., 1.]], device='cuda')
    for distance in (.1, 1., 10., 1000.):
        for cull in (False, True):
            center = torch.tensor([0., 0., distance], device='cuda')
            expected = []
            queue = torch.tensor([0], dtype=torch.long, device='cuda')
            while len(queue):
                visible = ((g.upper_tree_xyz[queue] @ planes[:, :3].T + planes[:, 3]
                            + g.bounding_sphere_radii[queue, None]) >= 0).all(1)
                queue = queue[visible] if cull else queue
                leaf = nodes[queue, 2] == 0
                expected.extend(queue[leaf].tolist())
                inner = queue[~leaf]
                expand = g.min_distance_squared[inner] > (g.upper_tree_xyz[inner] - center).square().sum(1)
                expected.extend(inner[~expand].tolist())
                first = nodes[inner[expand], 3].long()
                queue = torch.cat((first, nodes[first, 4].long()))
            mask = ops.upper_cut(nodes.contiguous(), g.upper_tree_xyz.contiguous(),
                                 g.bounding_sphere_radii.contiguous(), g.min_distance_squared.contiguous(),
                                 planes.contiguous(), center.contiguous(), 1., cull)
            order = g.upper_cut_order
            assert order[mask[order]].tolist() == expected


def test_v2_real_training_control_flow_prefetch_matches_reference(tmp_path, monkeypatch):
    """Run the real v2 trainer; mock scene/rasterizer, not cache/SPT/Adam/data.

    Both runs use the same private schedule. These are CPU integration tests,
    not evidence that real gsplat kernels or Windows CUDA compilation ran.
    """
    import sys
    import train_resident_v2 as tr
    instances, order = [], []

    class Model:
        def __init__(self, degree):
            fixture = tree()
            self.__dict__.update(vars(fixture))
            self.properties[:, 6] = 1.
            self._densification_criterium = torch.zeros(len(self.properties))
            self.max_sh_degree = self.active_sh_degree = degree
            self.spatial_lr_scale = 1.
            self.saves = []
            self.splits = 0
            for n in ('_xyz', '_opacity', '_rotation', '_scaling', '_features_dc', '_features_rest'):
                setattr(self, n, torch.zeros(1))
            instances.append(self)
        def compact_gaussians(self, *a, **k):
            pass
        def rotation_activation(self, x):
            return torch.nn.functional.normalize(x, dim=-1)
        def oneupSHdegree(self):
            self.active_sh_degree = min(self.max_sh_degree, self.active_sh_degree+1)
        def add_new_gs(self, **kw):
            assert kw['densification'] == 'classic'
            candidates = torch.where((self.nodes[:self.size, 2] == 0)
                & (self._densification_criterium[:self.size] > kw['densify_threshold']))[0]
            assert len(candidates)
            p = int(candidates[0])
            a, b = self.size, self.size+1
            self.properties[a:b+1] = self.properties[p]
            self.properties[a:b+1, 23:] = 0
            self.nodes[p, 2:4] = torch.tensor([2, a])
            self.nodes[a] = torch.tensor([int(self.nodes[p, 0])+1, p, 0, 0, b, 0])
            self.nodes[b] = torch.tensor([int(self.nodes[p, 0])+1, p, 0, 0, 0, 0])
            self.size += 2
            self.splits += 1
        def relocate_gs(self, mask, *a, **k):
            assert mask.shape == (self.size,)
        def save_hierarchy(self, *a, **k):
            self.saves.append((k['file_name'], self.properties.clone(), self._densification_criterium.clone()))

    cams = [camera(i) for i in range(4)]
    for c in cams:
        c.focal_length, c.invdepthmap = 100., None
        c.image_name = str(c.uid)
        c.original_image = torch.zeros(3, 4, 4)
        c.alpha_mask = torch.ones(1, 4, 4)
    class Scene:
        def __init__(self, *a, **k):
            pass
        def getTrainCameras(self):
            return cams

    def render(cam, xyz, opacity, scales, rotations, dc, rest, *a, **k):
        order.append(cam.image_name)
        screen = xyz[:, :2]
        screen.retain_grad()
        v = .003 * (screen.sum() + xyz[:, 2].sum() + opacity.sum() + scales.sum()
                    + rotations.sum() + dc.sum() + rest.sum())
        return dict(render=v.sigmoid().expand(3, 4, 4), viewspace_points=screen,
                    packed_indices=torch.arange(len(xyz)))

    monkeypatch.setitem(sys.modules, 'scene', SimpleNamespace(Scene=Scene, GaussianModel=Model))
    monkeypatch.setitem(sys.modules, 'gaussian_renderer', SimpleNamespace(render_gsplat=render))
    monkeypatch.setitem(sys.modules, 'utils.loss_utils', SimpleNamespace(l1_loss=lambda a,b:(a-b).abs().mean()))
    monkeypatch.setitem(sys.modules, 'fused_ssim', SimpleNamespace(fused_ssim=lambda a,b:1-(a-b).square().mean()))
    monkeypatch.setitem(sys.modules, 'utils.general_utils', SimpleNamespace(
        get_expon_lr_func=lambda **k: lambda i:k['lr_init']*.99**i))
    real_builder, real_pool, real_transfer, real_profile = tr.IncrementalSPT, tr.StreamingResidentPool, tr.CameraTransfer, tr.TrainingProfile
    monkeypatch.setattr(tr, 'IncrementalSPT', lambda *a, **k:real_builder(*a, **k, device='cpu'))
    monkeypatch.setattr(tr, 'StreamingResidentPool', lambda *a, **k:real_pool(*a, **k, device='cpu'))
    monkeypatch.setattr(tr, 'CameraTransfer', lambda *a, **k:real_transfer(*a, **k, device='cpu'))
    monkeypatch.setattr(tr, 'TrainingProfile', lambda *a, **k:real_profile(*a, **k, device='cpu'))
    monkeypatch.setattr(tr, 'load_native', lambda mode:None)
    # Vary the selected leaf subset by camera, and let the real builder's epoch
    # force an updated selection after every split.
    def select(g, cam, *a, **k):
        leaves = torch.where(g.nodes[:g.size, 2] == 0)[0]
        i = int(cam.image_name)
        return leaves[i::4].to(torch.int32)
    monkeypatch.setattr(tr, 'select_gaussians', select)
    real_tensor = torch.tensor
    def tensor(*a, **k):
        if k.get('device') == 'cuda':
            k['device'] = 'cpu'
        return real_tensor(*a, **k)
    monkeypatch.setattr(torch, 'tensor', tensor)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda:(24*2**30,24*2**30))
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda:'CPU control-flow fixture')
    for name in ('memory_allocated','memory_reserved','max_memory_allocated'):
        monkeypatch.setattr(torch.cuda,name,lambda:0)
    opt = SimpleNamespace(storage_device='cpu', densification='classic', prune_unused=False,
        dampen_scale_grad=False,optimize_exposure=False,use_occlusion_culling=False,SH_degree=1,
        cap_max=100,llff_hold=100,target_granularity_pixels=2,SPT_root_volume=1.,min_SPT_size=2,
        use_bounding_spheres=False,cache_size=8,graph_view_select=False,position_lr_init=.002,
        position_lr_final=.00001,position_lr_delay_mult=.01,position_lr_max_steps=16,iterations=16,
        vary_distance_multiplier=True,SH_increase_after_train_percent=.25,output_file_name='result.dhier',
        lambda_dssim=.2,densify_from_iter=0,densify_until_iter=15,densification_interval=4,
        densify_percent=1.02,densify_grad_threshold=1e-7,scaling_lr=.005,rotation_lr=.001,
        feature_lr=.0025,opacity_lr=.05,lr_multiplier=1.,data_workers=0,data_prefetch_factor=1,pin_memory=False)
    for enabled in (False,True):
        torch.manual_seed(8)
        tr.training(SimpleNamespace(output_path=str(tmp_path/str(enabled)),white_background=False,resolution=1),
                    opt,SimpleNamespace(debug=True),saving_iterations=[3],runtime=dict(profile_every=1,
                    native_ops='torch',view_prefetch=enabled,incremental_spt=enabled,
                    image_cache_gib=.001,gaussian_prefetch_rows=8,transfer_rows=4))
    assert order[:17] == order[17:]
    a,b = instances
    assert a.splits == b.splits == 3
    assert [name for name,_,_ in a.saves] == ['iteration_3','result.dhier']
    torch.testing.assert_close(a.properties,b.properties,rtol=0,atol=0)
    torch.testing.assert_close(a.saves[-1][1],a.properties,rtol=0,atol=0)
    assert a.saves[-1][2].sum() > 0
    assert len((tmp_path/'True'/'resident_profile.jsonl').read_text().splitlines()) == 17


def test_prefetch_protects_next_view_hits_not_only_current_view():
    h,s=backing()
    pool=StreamingResidentPool(h,s,4,device='cpu',prefetch_rows=4,transfer_rows=2)
    pool.acquire([0,1,2,3])  # Clean resident rows.
    pool.acquire([0])
    old=pool.to_slot[1:4].copy()
    assert pool.prefetch([1,2,3,4]) == 0
    assert np.array_equal(pool.to_slot[1:4],old)
    pool.close()
