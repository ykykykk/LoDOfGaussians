"""CPU reference coverage of paged cache mutation and visibility contracts."""
from types import SimpleNamespace
import copy

import numpy as np
import pytest
import torch

from utils.gaussian_block_store import GaussianBlockStore
from utils.paged_gaussian_pool import CapacityError, PagedGaussianPool


def make_store(tmp_path, background=False):
    store=GaussianBlockStore.create(tmp_path, {'block_rows':4})
    for i in range(3):
        rows=np.zeros((4,69),dtype=np.float32)
        rows[:,0]=i*10
        rows[:,3:6]=-5
        rows[:,6]=1
        rows[:,23:46]=.2
        rows[:,46:]=.4
        store.append(rows,skybox=background and i==0,scores=np.zeros(4,dtype=np.float32))
    return store


def test_dirty_eviction_roundtrips_parameters_moments_and_scores(tmp_path):
    store=make_store(tmp_path)
    pool=PagedGaussianPool(store,4,device='cpu')
    packet=pool.acquire([0])
    packet.adam_step(torch.full((4,23),.1),torch.full((23,),.001),10)
    packet.accumulate_scores(torch.tensor([0,1,1]),torch.tensor([.3,.5,.7]))
    expected=pool.state[:4].clone()
    pool.acquire([1])
    np.testing.assert_array_equal(store.read(0),expected.numpy())
    np.testing.assert_array_equal(store.read_scores(0),np.array([.3,.7,0,0],dtype=np.float32))
    again=pool.acquire([0])
    torch.testing.assert_close(again.state,expected[:,:23])
    torch.testing.assert_close(again.scores,torch.tensor([.3,.7,0,0]))
    assert pool.stats['evictions']==2
    assert pool.stats['downloaded_rows']==4


def test_capacity_failure_preserves_active_dirty_packet_and_disk(tmp_path):
    store=make_store(tmp_path)
    pool=PagedGaussianPool(store,4,device='cpu')
    packet=pool.acquire([0])
    packet.accumulate_scores(torch.tensor([0]),torch.tensor([1.]))
    resident=list(pool.resident.items())
    stats=copy.deepcopy(pool.stats)
    entries=copy.deepcopy(store.blocks)
    with pytest.raises(CapacityError) as error:
        pool.acquire([1,2])
    assert (error.value.required_rows,error.value.capacity_rows)==(8,4)
    assert pool.active is packet and packet.dirty
    assert list(pool.resident.items())==resident
    assert pool.stats==stats
    assert store.blocks==entries
    pool.flush()
    assert store.read_scores(0)[0]==1


def test_background_freezes_all_parameters_and_nonzero_adam(tmp_path):
    store=make_store(tmp_path,background=True)
    original=np.array(store.read(0),copy=True)
    original_scene=np.array(store.read(1),copy=True)
    pool=PagedGaussianPool(store,8,device='cpu')
    packet=pool.acquire([1,0])  # Frozen block need not be a prefix.
    packet.adam_step(torch.ones((8,23)),torch.full((23,),.01),25)
    pool.flush()
    np.testing.assert_array_equal(store.read(0),original)
    np.testing.assert_array_equal(packet.state[4:],original[:,:23])
    assert not np.array_equal(store.read(1),original_scene)


def test_candidate_culling_and_per_row_culling_keep_background(tmp_path):
    store=make_store(tmp_path)
    # Identity clip transform yields x/y in [-1,1], independent of z.
    camera=SimpleNamespace(full_proj_transform=torch.eye(4))
    store.blocks[2]['skybox']=True
    pool=PagedGaussianPool(store,8,device='cpu')
    assert pool.candidate_blocks(camera)==[0,2]
    packet=pool.acquire([0,2],camera)
    assert len(packet.state)==8
    assert packet._frozen.tolist()==[False]*4+[True]*4
    pool.clear()
    assert not pool.resident and pool.active is None


def test_duplicate_requests_do_not_consume_extra_pages(tmp_path):
    pool=PagedGaussianPool(make_store(tmp_path),4,device='cpu')
    packet=pool.acquire([0,0,0])
    assert len(packet.state)==4
    assert list(pool.resident)==[0]


def test_growth_keeps_resident_pages_and_matches_flush_clear_reference(tmp_path, monkeypatch):
    from utils.block_densification import densify_blocks
    stores = [make_store(tmp_path / name, background=True) for name in ('reference', 'cached')]
    pools = [PagedGaussianPool(store, 16, device='cpu') for store in stores]
    for pool in pools:
        packet = pool.acquire([0, 1, 2])
        packet.adam_step(torch.full((12,23), .1), torch.full((23,), .001), 20)
        packet.accumulate_scores(torch.tensor([0,4,5,8]), torch.tensor([9.,1.,2.,3.]))
    opt = SimpleNamespace(cap_max=100, densify_max_new_nodes=3,
                          densify_max_leaf_fraction=0, densify_grad_threshold=.5)
    pools[0].clear()
    expected = densify_blocks(stores[0], opt)
    cached = pools[1]
    pages = dict(cached.resident)
    sky = cached.state[cached._slice(0)].clone()
    writes = []
    real_write = stores[1].write
    def track(bid, rows, scores=None):
        writes.append(bid)
        return real_write(bid, rows, scores)
    monkeypatch.setattr(stores[1], 'write', track)
    assert cached.densify(opt) == expected
    assert dict(cached.resident) == pages
    assert cached.stats['evictions'] == 0
    torch.testing.assert_close(cached.state[cached._slice(0)], sky)
    # Resident pages are changed in place and written only once at checkpoint.
    assert 1 not in writes and 2 not in writes
    cached.flush()
    assert writes.count(1) == writes.count(2) == 1
    for b in stores[0].blocks:
        bid = b['id']
        np.testing.assert_array_equal(stores[1].read(bid), stores[0].read(bid))
        np.testing.assert_array_equal(stores[1].read_scores(bid), stores[0].read_scores(bid))
    stores[1].checkpoint()
    reopened = GaussianBlockStore.open(stores[1].root)
    assert sum(b['count'] for b in reopened.blocks) == 15
    cached.acquire([3])  # Newly appended pages are immediately addressable.


def test_growth_extends_resident_tail_and_preserves_unmodified_dirty_bounds(tmp_path):
    from utils.block_densification import densify_blocks
    stores=[]
    for name in ('reference','cached'):
        store=GaussianBlockStore.create(tmp_path/name, {'block_rows':4})
        for n in (4,1):
            rows=np.zeros((n,69),dtype=np.float32)
            rows[:,6]=1
            rows[:,23:]=7
            store.append(rows,scores=np.zeros(n,dtype=np.float32))
        stores.append(store)
    pools=[PagedGaussianPool(store,12,device='cpu') for store in stores]
    opt=SimpleNamespace(cap_max=100,densify_max_new_nodes=2,
                        densify_max_leaf_fraction=0,densify_grad_threshold=.5)
    for window in range(3):
        for pool in pools:
            packet=pool.acquire([0,1])
            packet.accumulate_scores(torch.tensor([0]),torch.tensor([2.]))
            pool.finish_step(packet)
        pools[0].clear()
        densify_blocks(stores[0],opt)
        # Reference refresh is what the old trainer did after growth.
        pools[0].refresh_metadata()
        page_before=pools[1].resident[1]
        pools[1].densify(opt)
        assert pools[1].resident[1]==page_before
        pools[1].flush()
        for b in stores[0].blocks:
            np.testing.assert_array_equal(stores[0].read(b['id']),stores[1].read(b['id']))
    assert stores[1].blocks[1]['count']==4


def test_topology_refresh_retains_live_bounds_and_score_only_writes(tmp_path, monkeypatch):
    pool=PagedGaussianPool(make_store(tmp_path),12,device='cpu')
    pool.acquire([0,1])
    # Simulate an Adam update far beyond the disk AABB with no growth score.
    pool.state[pool._slice(1),0]=100
    pool.active.dirty=True
    pool.finish_step()
    old_bounds=pool.bounds[1].clone()
    files=[b['file'] for b in pool.store.blocks]
    opt=SimpleNamespace(cap_max=12,densify_max_new_nodes=2,
                        densify_max_leaf_fraction=0,densify_grad_threshold=.5)
    pool.densify(opt)
    torch.testing.assert_close(pool.bounds[1],old_bounds)
    assert files==[b['file'] for b in pool.store.blocks]
    assert pool.candidate_blocks(SimpleNamespace(full_proj_transform=torch.eye(4)))==[0]
