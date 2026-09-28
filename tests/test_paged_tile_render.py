"""Real CUDA raster checks: a native-pixel tile preserves the full-view core."""
from types import SimpleNamespace

import pytest
import torch

from utils.camera_tiles import crop_camera


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA rasterization required')
@pytest.mark.parametrize('core', [(35,27,80,61), (0,0,77,59), (151,105,41,39)])
def test_native_tile_matches_full_render_and_ssim_gradient(core):
    from gaussian_renderer import render_gsplat
    from fused_ssim import FusedSSIMMap
    torch.manual_seed(714)
    device='cuda'
    camera=SimpleNamespace(image_width=192,image_height=144,
        original_image=torch.zeros(3,144,192,device=device), alpha_mask=None,
        invdepthmap=None,depth_mask=None,
        K_train=torch.tensor([[155.,0.,87.],[0.,160.,79.],[0.,0.,1.]],device=device),
        world_view_transform=torch.eye(4,device=device), znear=.01,zfar=100.)
    camera.world_view_transform[3,:3]=torch.tensor([.15,-.1,.3],device=device)
    n=96
    means=torch.rand(n,3,device=device)
    means[:,:2]=(means[:,:2]-.5)*4
    means[:,2]=means[:,2]*2+2
    means.requires_grad_(True)
    scales=torch.full((n,3),.06,device=device)
    rotations=torch.zeros(n,4,device=device)
    rotations[:,0]=1
    colors=torch.rand(n,1,3,device=device)
    rest=torch.zeros(n,3,3,device=device)
    opacity=torch.full((n,1),.65,device=device)
    background=torch.tensor([.07,.09,.12],device=device)

    def render(view):
        return render_gsplat(view,means,opacity,scales,rotations,colors,rest,
                             SimpleNamespace(),background,sh_degree=1)['render']

    full=render(camera)
    tile=crop_camera(camera,*core,halo=8)
    cropped=render(tile)
    x,y,w,h=core
    cx,cy,cw,ch=tile.tile_core
    actual=cropped[:,cy:cy+ch,cx:cx+cw]
    expected=full[:,y:y+h,x:x+w]
    error=(actual-expected).abs()
    print(f'core={core} render_max_abs={error.max().item():.9g} mean_abs={error.mean().item():.9g}')
    torch.testing.assert_close(actual,expected,atol=3e-5,rtol=3e-5)
    # Exercise actual fused SSIM map through the rendered halo into 3D means.
    target=(cropped.detach()*.93+.02).clamp(0,1)
    maps=FusedSSIMMap.apply(.01**2,.03**2,cropped[None].contiguous(),
                            target[None].contiguous(),'same',True,2)
    loss=1-maps[...,cy:cy+ch,cx:cx+cw].mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert means.grad is not None and bool(torch.isfinite(means.grad).all())
    assert bool((means.grad.abs()>0).any())
