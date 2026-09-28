from types import SimpleNamespace

import pytest
import torch

from utils.camera_tiles import crop_camera, choose_training_tile


def camera():
    w, h = 19, 13
    return SimpleNamespace(image_width=w, image_height=h,
        original_image=torch.arange(3*h*w).reshape(3,h,w).float(),
        alpha_mask=torch.rand(1,h,w), invdepthmap=torch.rand(1,h,w),
        depth_mask=torch.ones(1,h,w, dtype=torch.bool),
        K_train=torch.tensor([[24.,0.,7.],[0.,28.,8.],[0.,0.,1.]]),
        world_view_transform=torch.eye(4), znear=.01, zfar=100., image_name='test')


@pytest.mark.parametrize('rectangle', [(0,0,8,8), (7,4,8,6), (16,10,3,3)])
def test_crop_masks_halo_and_original_unchanged(rectangle):
    source = camera()
    original = source.original_image.clone()
    tile = crop_camera(source, *rectangle, halo=2)
    x,y,w,h = tile.tile_bounds
    for name in ('original_image','alpha_mask','invdepthmap','depth_mask'):
        torch.testing.assert_close(getattr(tile,name), getattr(source,name)[...,y:y+h,x:x+w])
    cx,cy,cw,ch = tile.tile_core
    left,top,width,height = rectangle
    torch.testing.assert_close(tile.original_image[...,cy:cy+ch,cx:cx+cw],
                               source.original_image[...,top:top+height,left:left+width])
    tile.original_image.zero_()
    torch.testing.assert_close(source.original_image, original)
    assert source.image_width == 19


def test_projection_matches_pixel_crop_with_offcenter_principal_point():
    source = camera()
    source.world_view_transform[3,:3] = torch.tensor([.3,-.2,1.])
    tile = crop_camera(source, 14, 9, 5, 4, halo=0)
    points = torch.tensor([[.1,.2,2.,1.],[-.4,.6,4.,1.]])
    view = points @ source.world_view_transform
    pixel = (view[:,:3] @ source.K_train.T)
    pixel = pixel[:,:2] / pixel[:,2:3] - torch.tensor([14.,9.])
    clip = points @ tile.full_proj_transform
    ndc = clip[:,:2] / clip[:,3:4]
    projected = (ndc + 1) * torch.tensor([tile.image_width/2,tile.image_height/2])
    torch.testing.assert_close(projected,pixel)
    torch.testing.assert_close(tile.full_proj_transform @ tile.full_proj_transform_inverse, torch.eye(4), atol=2e-5,rtol=2e-5)
    assert tile.cx < 0 and tile.cy < 0


def test_schedule_covers_image_once_per_cycle_and_is_reproducible():
    source = camera()
    coverage = torch.zeros(13,19,dtype=torch.int32)
    indices=[]
    for step in range(6):
        tile=choose_training_tile(source,step,123,tile_size=8,halo=2)
        repeat=choose_training_tile(source,step,123,tile_size=8,halo=2)
        assert tile.tile_bounds == repeat.tile_bounds
        indices.append(tile.tile_index)
        x,y,_,_=tile.tile_bounds
        cx,cy,cw,ch=tile.tile_core
        coverage[y+cy:y+cy+ch,x+cx:x+cx+cw]+=1
    assert len(set(indices)) == 6
    assert bool((coverage==1).all())
    assert choose_training_tile(source,6,123,8).tile_cycle == 1


def test_small_image_and_absent_masks():
    source=camera()
    source.alpha_mask=source.invdepthmap=source.depth_mask=None
    tile=choose_training_tile(source,0,0)
    assert tile.tile_core == (0,0,19,13)
    assert tile.alpha_mask is None
    torch.testing.assert_close(tile.K_train,source.K_train)


def test_balanced_grid_has_no_tiny_border_strip():
    source = camera()
    coverage = torch.zeros(13, 19, dtype=torch.int32)
    sizes = []
    for step in range(6):
        tile = choose_training_tile(source, step, 123, 8, 2, balanced=True)
        x, y, _, _ = tile.tile_bounds
        cx, cy, w, h = tile.tile_core
        sizes.append((w, h))
        coverage[y+cy:y+cy+h, x+cx:x+cx+w] += 1
    assert bool((coverage == 1).all())
    assert max(w for w,h in sizes)-min(w for w,h in sizes) <= 1
    assert max(h for w,h in sizes)-min(h for w,h in sizes) <= 1


@pytest.mark.parametrize('rect', [(-1,0,2,2),(0,0,0,2),(18,0,2,2)])
def test_invalid_bounds(rect):
    with pytest.raises(ValueError):
        crop_camera(camera(),*rect)
