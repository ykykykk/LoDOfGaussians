from types import SimpleNamespace
import numpy as np
import torch

from utils.block_densification import densify_blocks


class Store:
    block_rows = 4

    def __init__(self):
        self.blocks=[]
        self.rows={}
        self.scores={}
        self.writes=[]

    def append(self, rows, skybox=False, scores=None):
        i=len(self.blocks)
        self.blocks.append(dict(id=i,count=len(rows),skybox=skybox,bounds_min=[i,0,0],bounds_max=[i+1,1,1]))
        self.rows[i]=np.array(rows,copy=True)
        self.scores[i]=np.array(scores,copy=True)
        return i

    def read(self,i): return self.rows[i]
    def read_scores(self,i): return self.scores[i]
    def write_scores(self,i,scores): self.scores[i]=np.array(scores,copy=True)
    def write(self,i,rows,scores=None):
        self.writes.append(i)
        self.rows[i]=np.array(rows,copy=True)
        self.blocks[i]["count"]=len(rows)
        self.write_scores(i,scores)


def fixture():
    store=Store()
    for sky in (True,False,False):
        rows=np.zeros((4,69),dtype=np.float32)
        rows[:,6]=1
        rows[:,23:]=7
        rows[:,0]=np.arange(4)
        store.append(rows,skybox=sky,scores=np.array([1,2,3,4],dtype=np.float32))
    opt=SimpleNamespace(cap_max=100,densify_max_new_nodes=5,densify_max_leaf_fraction=0,densify_grad_threshold=.5)
    return store,opt


def test_budget_replacement_adam_and_coalesced_children():
    store,opt=fixture()
    sky=store.rows[0].copy()
    result=densify_blocks(store,opt)
    assert result==dict(net_added=5,total_points=17,eligible_points=8)
    np.testing.assert_array_equal(store.rows[0],sky)
    assert store.writes==[1,2]
    assert [b['count'] for b in store.blocks]==[4,4,4,4,1]
    # Largest remainder gives the first scene page 3 splits, second 2.
    np.testing.assert_array_equal(store.rows[1][0,23:],7)
    assert np.all(store.rows[1][1:,23:]==0)
    assert np.all(store.rows[2][:2,23:]==7)
    assert np.all(store.rows[2][2:,23:]==0)
    assert all(np.all(s==0) for s in store.scores.values())
    assert all(np.all(store.rows[i][:,23:]==0) for i in (3,4))


def test_cap_fraction_and_no_model_rewrites_when_full():
    store,opt=fixture()
    opt.cap_max=14
    assert densify_blocks(store,opt)['net_added']==2
    store,opt=fixture()
    opt.densify_max_leaf_fraction=.25
    assert densify_blocks(store,opt)['net_added']==2
    store,opt=fixture()
    opt.cap_max=12
    assert densify_blocks(store,opt)['net_added']==0
    assert not store.writes


def test_point_allocations_are_block_bounded(monkeypatch):
    store,opt=fixture()
    empty,zeros=torch.empty,torch.zeros
    sizes=[]
    def track(factory):
        def call(shape,*a,**kw):
            sizes.append(shape[0] if isinstance(shape,tuple) else shape)
            return factory(shape,*a,**kw)
        return call
    monkeypatch.setattr(torch,'empty',track(empty))
    monkeypatch.setattr(torch,'zeros',track(zeros))
    densify_blocks(store,opt)
    assert max(sizes)<=2*store.block_rows


def test_repeated_growth_reuses_tail_pages_without_mixing_bands():
    store=Store()
    for band in (0,1):
        rows=np.zeros((4,69),dtype=np.float32)
        rows[:,6]=1
        rows[:,10]=band+10
        rows[:,23:]=7
        block=store.append(rows,scores=np.ones(4,dtype=np.float32))
        store.blocks[block]['radius_band']=band
    opt=SimpleNamespace(cap_max=100,densify_max_new_nodes=2,densify_max_leaf_fraction=0,densify_grad_threshold=.5)
    for window in range(4):
        # Only first point in each original block is eligible. Existing children
        # and unsplit rows retain parameters/Adam while tails fill over windows.
        for b in store.blocks:
            store.scores[b['id']]=np.zeros(b['count'],dtype=np.float32)
        store.scores[0][0]=store.scores[1][0]=1
        assert densify_blocks(store,opt)['net_added']==2
        assert sum(b['count'] for b in store.blocks)==8+2*(window+1)
        assert all(b['count']<=4 for b in store.blocks)
        for b in store.blocks:
            assert np.all(store.rows[b['id']][:,10]==b['radius_band']+10)
        np.testing.assert_array_equal(store.rows[0][1:,23:],7)
        np.testing.assert_array_equal(store.rows[1][1:,23:],7)
    assert [b['count'] for b in store.blocks]==[4,4,4,4]


def test_future_source_tail_is_not_extended_before_processing():
    store=Store()
    for n,marker in ((4,10),(1,20)):
        rows=np.zeros((n,69),dtype=np.float32)
        rows[:,6]=1
        rows[:,10]=marker
        rows[:,23:]=9
        store.append(rows,scores=np.ones(n,dtype=np.float32))
    opt=SimpleNamespace(cap_max=100,densify_max_new_nodes=5,densify_max_leaf_fraction=0,densify_grad_threshold=.5)
    result=densify_blocks(store,opt)
    assert result['total_points']==10
    assert sum(b['count'] for b in store.blocks)==10
    points=np.concatenate(list(store.rows.values()))
    assert np.count_nonzero(points[:,10]==10)==8
    assert np.count_nonzero(points[:,10]==20)==2
    assert np.all(points[:,23:]==0)
