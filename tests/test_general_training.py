import argparse
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch
from PIL import Image
from utils.camera_geometry import image_size, intrinsics, camera_intrinsics, effective_focal, screen_scores
from utils.general_policy import resolve_plan, relative_spt_volume, eligible_parents, DetailWindow
from utils.dataset_preflight import inspect_dataset, scaffold_manifest
from utils.resident_pool import ResidentPool, ActivePacket

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('w,h', [(6004,4010),(4010,6004),(13,7),(1024,1024)])
@pytest.mark.parametrize('factor', [1,2,4,8])
def test_projection_and_score_units(w,h,factor):
    tw,th = image_size(w,h,factor)
    assert (tw,th) == (max(1,round(w/factor)),max(1,round(h/factor)))
    fovx,fovy = 1.1,.8
    k=intrinsics(tw,th,fovx,fovy,.37,.61)
    base=intrinsics(w,h,fovx,fovy,.37,.61)
    np.testing.assert_allclose(k, np.array(base)*[tw/w,th/h,tw/w,th/h])
    ndc=torch.tensor([[.01,-.03],[.002,.004]])
    pixels=ndc / torch.tensor([tw/2,th/2])
    torch.testing.assert_close(screen_scores(pixels,tw,th,'ndc'),ndc.norm(dim=-1))
    torch.testing.assert_close(screen_scores(pixels,tw,th,'pixel'),pixels.norm(dim=-1))

@pytest.mark.parametrize('px,py',[(.5,.5),(.31,.7),(.7,.31)])
def test_frustum_projection_matches_pinhole_matrix(px,py):
    from utils.graphics_utils import getProjectionMatrix
    w,h,fx,fy=320,240,220.,240.
    fovx,fovy=2*math.atan(w/(2*fx)),2*math.atan(h/(2*fy))
    p=getProjectionMatrix(.01,100.,fovx,fovy,px,py)
    x=torch.tensor([.2,-.1,2.,1.])
    clip=p@x; projected=(clip[:2]/clip[3]+1)*torch.tensor([w/2,h/2])
    torch.testing.assert_close(projected,torch.tensor([fx*.2/2+px*w,fy*(-.1)/2+py*h]))
    cam=NS(image_width=w,image_height=h,FoVx=fovx,FoVy=fovy,primx=px,primy=py)
    k=camera_intrinsics(cam)
    assert k[0,2] == px*w and effective_focal(cam)==fy

@pytest.mark.parametrize('scale',[.001,1.,1000.])
def test_relative_partition_is_unit_covariant(scale):
    opt=NS(SPT_relative_volume=.001,SPT_root_volume=25)
    assert relative_spt_volume(opt,2*scale)==pytest.approx(relative_spt_volume(opt,2)*scale**3)

def test_split_budget_only_selects_qualified_leaves():
    scores=torch.tensor([.2,.6,.9,.01,.8])
    leaves=torch.tensor([0,1,2,3])
    assert set(eligible_parents(scores,leaves,.5,100).tolist())=={1,2}
    assert eligible_parents(scores,leaves,.5,100,.25).tolist()==[2]
    assert eligible_parents(scores,leaves,.95,100).numel()==0
    assert eligible_parents(scores,leaves,.01,1).numel()==0
    assert len(eligible_parents(scores,leaves,.01,100,0,3))==1
    with pytest.raises(FloatingPointError):
        eligible_parents(torch.tensor([float('nan')]),torch.tensor([0]),.1,10)

def test_zero_gradient_visibility_is_not_lost():
    tracker=DetailWindow(8,'cpu')
    packet=ActivePacket(np.array([1,4,7]),None,torch.zeros(3,69),torch.zeros(3))
    tracker.observe(packet,torch.tensor([0,2]))
    g=NS(size=8,nodes=torch.zeros(8,6,dtype=torch.int32),_densification_criterium=torch.zeros(8))
    g._densification_criterium[7]=.3
    report=tracker.report(g,.2,100)
    assert report['visible_leaves']==2 and report['positive_gradient_leaves']==1
    assert report['eligible_leaves']==1
    tracker.reset();assert not tracker.seen.any()

@pytest.mark.parametrize('views',[1,42,1500,20000])
def test_bounded_schedule_and_input_immutability(views):
    c=json.loads((ROOT/'configs/general_balanced.json').read_text());before=copy.deepcopy(c)
    plan=resolve_plan(c,{'training_views':views,'decoded_training_bytes':4*2**30},32*2**30)
    assert c==before
    assert 20000<=plan['iterations']<=120000 and plan['data_workers']==0
    assert plan['position_lr_max_steps']==plan['iterations']
    assert plan['densify_until_iter']<=.81*plan['iterations']
    assert plan['resolved_policy']['cache_budget_bytes']<=8*2**30
    small=resolve_plan(c,{'training_views':views,'decoded_training_bytes':80*2**30},8*2**30)
    assert small['data_workers']==4 and small['resolved_policy']['cache_budget_bytes']<=2*2**30


def test_continuation_keeps_original_lr_schedule_and_extended_split_window():
    c=json.loads((ROOT/'configs/general_balanced.json').read_text())
    c.update(iterations=60, position_lr_max_steps=30, densify_from_iter=5,
             densify_until_iter=48, densification_interval=5)
    c['general_policy']['preserve_fine_schedule']=True
    p=resolve_plan(c,{'training_views':42,'decoded_training_bytes':4*2**30},32*2**30)
    assert [p[k] for k in ('position_lr_max_steps','densify_from_iter',
                            'densify_until_iter','densification_interval')]==[30,5,48,5]

def test_explicit_budgets():
    c=json.loads((ROOT/'configs/general_balanced.json').read_text());c.update(iterations=1234,coarse_iterations=17)
    p=resolve_plan(c,{'training_views':42,'decoded_training_bytes':1000},2**30)
    assert p['iterations']==1234 and p['coarse_iterations']==17
    assert p['position_lr_max_steps']==1234


def test_full_resolution_byte_cache_fits_without_changing_quality_settings():
    c=json.loads((ROOT/'configs/general_balanced.json').read_text())
    dataset=dict(training_views=42,decoded_training_bytes=16*2**30,compact_training_bytes=4*2**30)
    plan=resolve_plan(c,dataset,32*2**30)
    assert plan['data_workers']==0 and plan['coarse_compact_images']
    assert plan['resolved_policy']['cached_training_bytes']==4*2**30
    assert plan['densify_grad_threshold']==c['densify_grad_threshold']
    c['resident']['compact_images']=False
    reference=resolve_plan(c,dataset,32*2**30)
    assert reference['data_workers']==4 and not reference['coarse_compact_images']

def test_resident_resize_barrier_preserves_values():
    host=torch.zeros(12,69);scores=torch.zeros(12)
    pool=ResidentPool(host,scores,4,device='cpu')
    p=pool.acquire([1,3]);p.adam_step(torch.ones(2,23),torch.ones(23)*.01,0)
    with pytest.raises(RuntimeError):pool.resize_empty(8)
    pool.flush();pool.invalidate();pool.resize_empty(8)
    torch.testing.assert_close(pool.acquire([3,1]).state,host[[3,1]])
    pool.close()

def fixture_dataset(path):
    sparse=path/'sparse/0';sparse.mkdir(parents=True)
    images=path/'images';images.mkdir()
    (sparse/'cameras.txt').write_text('0 PINHOLE 12 8 10 11 4 5'+chr(10))
    records=[]
    for i in range(4):
        name=f'{i}.png';Image.new('RGB',(12,8)).save(images/name)
        records.extend([f'{i} 1 0 0 0 {i} 0 0 0 {name}', ''])
    (sparse/'images.txt').write_text(chr(10).join(records)+chr(10))
    (sparse/'points3D.txt').write_text('0 0 0 1 255 255 255 0'+chr(10))
    return images,sparse

def test_preflight_fingerprints_and_size_validation(tmp_path):
    images,sparse=fixture_dataset(tmp_path)
    a=inspect_dataset(tmp_path,resolution=2,hold=3)
    assert a['training_views']==2 and a['training_sizes']==[(6,4)]
    assert a['compact_training_bytes']==2*(6*4*4+1024)
    empty_masks=tmp_path/'masks';empty_masks.mkdir()
    assert inspect_dataset(tmp_path,masks_dir=empty_masks,resolution=2,hold=3)['compact_training_bytes']==a['compact_training_bytes']
    b=inspect_dataset(tmp_path,resolution=1,hold=3)
    assert b['input_fingerprint']!=a['input_fingerprint']
    Image.new('RGB',(12,12)).save(images/'0.png')
    with pytest.raises(ValueError,match='aspect ratio'):inspect_dataset(tmp_path)

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_cuda_render_uses_off_center_principal_point():
    from gaussian_renderer import render_gsplat
    w,h=64,48
    camera=NS(image_width=w,image_height=h,FoVx=1.,FoVy=.9,primx=.31,primy=.68,
              world_view_transform=torch.eye(4,device='cuda'))
    xyz=torch.tensor([[.02,-.01,2.]],device='cuda',requires_grad=True)
    pkg=render_gsplat(camera,xyz,torch.ones(1,1,device='cuda')*.5,torch.ones(1,3,device='cuda')*.02,
        torch.tensor([[1.,0,0,0]],device='cuda'),torch.zeros(1,1,3,device='cuda'),
        torch.zeros(1,3,3,device='cuda'),NS(),torch.zeros(3,device='cuda'),sh_degree=1)
    fx,fy,cx,cy=intrinsics(w,h,1.,.9,.31,.68)
    expected=torch.tensor([[fx*.01+cx,fy*(-.005)+cy]],device='cuda')
    torch.testing.assert_close(pkg['viewspace_points'],expected,rtol=1e-5,atol=1e-4)
    pkg['render'].sum().backward(); assert torch.isfinite(xyz.grad).all()


def test_camera_dataset_import_is_safe_in_fresh_spawn_process():
    import subprocess, sys
    result=subprocess.run([sys.executable, '-c',
        "import sys; import utils.camera_utils; assert 'scene' not in sys.modules"],
        cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode==0, result.stdout+result.stderr
