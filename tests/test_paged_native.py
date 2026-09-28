import pytest
import torch
from utils.resident_native import load_native

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_paged_visibility_and_page_bounds_match_reference():
    ops=load_native('cuda')
    torch.manual_seed(73)
    rows=513
    state=torch.randn(5*rows,69,device='cuda')
    state[:,3:6]-=4
    counts=torch.tensor([513,301,0,511,513],device='cuda',dtype=torch.int64)
    requested=torch.tensor([True,True,True,False,True],device='cuda')
    sky=torch.tensor([False,True,False,False,False],device='cuda')
    planes=torch.tensor([[1.,0.,0.,.5],[-1.,0.,0.,.5],[0.,1.,0.,.5],[0.,-1.,0.,.5]],device='cuda')
    expected=torch.zeros(len(state),device='cuda',dtype=torch.bool)
    for page,n in enumerate(counts.tolist()):
        if not requested[page]: continue
        raw=state[page*rows:page*rows+n]
        radius=3*raw[:,3:6].amax(1).exp()
        visible=((raw[:,:3]@planes[:,:3].T+planes[:,3]+radius[:,None])>=0).all(1)
        expected[page*rows:page*rows+n]=True if sky[page] else visible
    actual=ops.paged_visible(state,counts,requested,sky,rows,planes)
    assert torch.equal(actual,expected)
    mapping=torch.tensor([2,0,-1,1,3],device='cuda',dtype=torch.int64)
    bounds=torch.full((5,6),999.,device='cuda')
    reference=bounds.clone()
    for page,n in enumerate(counts.tolist()):
        if not requested[page] or not n or mapping[page]<0: continue
        raw=state[page*rows:page*rows+n]
        radius=3*raw[:,3:6].amax(1,keepdim=True).exp()
        reference[mapping[page],:3]=(raw[:,:3]-radius).amin(0)
        reference[mapping[page],3:]=(raw[:,:3]+radius).amax(0)
    ops.recompute_page_bounds(state,counts,mapping,requested,rows,bounds)
    torch.testing.assert_close(bounds,reference,atol=1e-6,rtol=1e-6)
    print('bounds_max_abs',float((bounds-reference).abs().max()))


def test_paged_native_rejects_metadata_shape_mismatch():
    ops=load_native('cuda')
    state=torch.zeros(8,69,device='cuda')
    counts=torch.tensor([4],device='cuda')
    flags=torch.ones(2,device='cuda',dtype=torch.bool)
    with pytest.raises(RuntimeError,match='page counts'):
        ops.paged_visible(state,counts,flags,flags,4,torch.zeros(4,4,device='cuda'))
