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
