import json
import numpy as np
import torch
from utils.gaussian_block_store import GaussianBlockStore
from tools.convert_block_checkpoint import convert


def rows(n):
    a = np.zeros((n,69),dtype=np.float32)
    a[:,0] = np.arange(n)
    a[:,6] = 1
    return a


def test_atomic_manifest_and_scores(tmp_path):
    store = GaussianBlockStore.create(tmp_path, {'iteration':7,'rng':torch.get_rng_state()})
    store.append(rows(3),scores=np.arange(3,dtype=np.float32))
    path = store.checkpoint()
    original = store.read(0).copy()
    update = rows(4)
    store.write(0,update,scores=np.ones(4,dtype=np.float32))
    np.testing.assert_equal(GaussianBlockStore.open(path).read(0), original)
    store.checkpoint({'iteration':8})
    reopened = GaussianBlockStore.open(path)
    np.testing.assert_equal(reopened.read(0), update)
    np.testing.assert_equal(reopened.read_scores(0),np.ones(4,dtype=np.float32))
    assert reopened.metadata['iteration'] == 8
    assert reopened.blocks[0]['bounds_min'][0] == -3


def test_conversion_and_ply(tmp_path):
    p = rows(12)
    p[:,23:] = np.arange(46)[None,:]
    source = tmp_path/'source.pt'
    torch.save(dict(size=12,properties=torch.from_numpy(p),nodes=torch.zeros(12,6,dtype=torch.int32),scores=torch.arange(12,dtype=torch.float32),skybox_points=2,iteration=47000,contract={'representation':'flat'}),source)
    path = convert(source,tmp_path/'converted',block_size=3)
    store = GaussianBlockStore.open(path)
    actual=np.concatenate([store.read(b['id']) for b in store.blocks])
    np.testing.assert_equal(actual[np.argsort(actual[:,0])],p)
    for b in store.blocks:
        np.testing.assert_equal(store.read_scores(b['id']),store.read(b['id'])[:,0])
    output=store.export_ply(tmp_path/'scene.ply')
    data=output.read_bytes(); header,payload=data.split(b'end_header\n',1)
    assert b'element vertex 10' in header
    assert len(payload)==10*26*4


def test_checkpoint_gc_keeps_two_generations_and_dirty(tmp_path):
    store=GaussianBlockStore.create(tmp_path)
    store.append(rows(3))
    store.checkpoint({'iteration':1})
    old=store.blocks[0]['file']
    store.write_scores(0,np.ones(3,dtype=np.float32))
    assert store.blocks[0]['file']==old
    store.checkpoint({'iteration':2})
    second_score=store.blocks[0]['scores']
    store.write(0,rows(4))
    dirty=store.blocks[0]['file']
    orphan=tmp_path/'blocks'/('99-'+'a'*32+'.npy')
    np.save(orphan,rows(1))
    unrelated=tmp_path/'blocks'/'source.npy'
    np.save(unrelated,rows(1))
    store.garbage_collect()
    assert not orphan.exists()
    assert unrelated.exists() and (tmp_path/dirty).exists()
    store.checkpoint({'iteration':3})
    previous=GaussianBlockStore.open(tmp_path/'manifest.previous.json')
    assert previous.metadata['iteration']==2
    np.testing.assert_equal(previous.read_scores(0),np.ones(3,dtype=np.float32))
    assert (tmp_path/old).exists()
    store.checkpoint({'iteration':4})
    assert not (tmp_path/old).exists()
    assert not (tmp_path/second_score).exists()
    assert GaussianBlockStore.open(tmp_path).read(0).shape==(4,69)
    assert GaussianBlockStore.open(tmp_path/'manifest.previous.json').read(0).shape==(4,69)


def test_conversion_separates_large_supports(tmp_path):
    p=rows(20)
    p[::5,3:6]=np.log(100)
    source=tmp_path/'source.pt'
    torch.save(dict(size=20,properties=torch.from_numpy(p),nodes=torch.zeros(20,6,dtype=torch.int32),scores=torch.zeros(20),skybox_points=0,iteration=1),source)
    store=GaussianBlockStore.open(convert(source,tmp_path/'converted',block_size=10))
    assert len(set(b['radius_band'] for b in store.blocks))==2
    for b in store.blocks:
        values=store.read(b['id'])[:,3]
        assert np.all(values==values[0])
    actual=np.concatenate([store.read(b['id']) for b in store.blocks])
    np.testing.assert_equal(actual[np.argsort(actual[:,0])],p)
