import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'/'geometry_map'))
from decompose_action_curvature import diagnose_path,direction_scale_estimate


def circle():
    t=torch.tensor([-1.,-.5,0.,.5,1.],dtype=torch.float64)
    u=torch.tensor([1.,-1.,0.],dtype=torch.float64)/2**.5
    v=torch.tensor([1.,1.,-2.],dtype=torch.float64)/6**.5
    return (3+2*(torch.cos(t[:,None])*u+torch.sin(t[:,None])*v))[:,None,:]


def test_sphere_mean_radius_direction_reconstructs_held_midpoint():
    x=circle();estimate,fallback=direction_scale_estimate(x[[0,1,3,4]])
    torch.testing.assert_close(estimate,x[2],rtol=0,atol=1e-14)
    assert fallback==0
    report=diagnose_path(x)
    assert report['normal_energy_fractions']['token_radius']==pytest.approx(1)
    assert report['normal_energy_fractions']['remaining_direction']<1e-25
    assert report['midpoint_mse']['linear_equal_data']>0.01


def test_nonradial_bending_remains_and_energy_is_orthogonal():
    t=torch.tensor([-1.,-.5,0.,.5,1.],dtype=torch.float64)
    # Mutually orthogonal zero-mean radius/tangent/bending directions in R4.
    u=torch.tensor([1.,-1.,0.,0.],dtype=torch.float64)/2**.5
    v=torch.tensor([0.,0.,1.,-1.],dtype=torch.float64)/2**.5
    w=torch.tensor([1.,1.,-1.,-1.],dtype=torch.float64)/2
    x=(u+t[:,None]*v+t[:,None]**2*w)[:,None,:]
    result=diagnose_path(x)
    assert result['normal_energy_fractions']['remaining_direction']==pytest.approx(1)
    assert result['orthogonal_decomposition_error']<1e-12


def test_constant_field_degenerate_path_and_donor_validation():
    x=torch.full((5,2,4),3.)
    estimate,fallback=direction_scale_estimate(x[[0,1,3,4]])
    assert torch.equal(estimate,x[2]) and fallback==2
    result=diagnose_path(x)
    assert result['degenerate_endpoint_chord']
    assert all(v is None for v in result['normal_energy_fractions'].values())
    with pytest.raises(ValueError):direction_scale_estimate(x)
    with pytest.raises(ValueError):direction_scale_estimate(torch.full((4,2,4),float('nan')))
