import pytest
import torch
from tools.tomato_spatial_readout import spatial_pool,SpatialReadout


def test_equal_region_mean_matches_original_mean():
    x=torch.randn(2,2048,12);a=spatial_pool(x)
    assert a.shape==(2,16,12)
    torch.testing.assert_close(a.mean(1),x.mean(1),atol=1e-6,rtol=1e-5)
    with pytest.raises(ValueError):spatial_pool(x[:,:2000])


def test_matched_methods_use_identical_parameters_and_only_spatial_layout_differs():
    m=SpatialReadout().eval();x=torch.randn(2,3,16,1024);q=torch.randn(2,16,58)
    a=m(x,q,'global_mean_control');b=m(x,q,'spatial_4x4')
    assert a.shape==b.shape==(2,4)
    assert not torch.allclose(a,b)
    flat=x.mean(2,keepdim=True).expand_as(x)
    torch.testing.assert_close(m(flat,q,'global_mean_control'),m(flat,q,'spatial_4x4'))
