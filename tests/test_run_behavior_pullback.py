import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'/'geometry_map'))
from run_behavior_pullback import spline_weights,sphere_log,sphere_exp,behavior_curve,projected_replacement,hellinger2


def test_natural_spline_exact_knots_and_linear():
    x=torch.tensor([-4.,-2.,2.,4.],dtype=torch.float64)
    torch.testing.assert_close(spline_weights(x,x),torch.eye(4,dtype=torch.float64))
    q=torch.linspace(-4,4,20,dtype=torch.float64)
    torch.testing.assert_close(spline_weights(x,q)@x,q)


def test_ten_control_spline_gradients():
    c=torch.randn(10,4,dtype=torch.float64,requires_grad=True)
    w=spline_weights(torch.linspace(0,1,10),torch.linspace(0,1,20))
    assert torch.autograd.gradcheck(lambda v:(w@v).square().mean(),(c,))


def test_sphere_log_exp_and_degenerate_identity():
    torch.manual_seed(5);p=torch.rand(4,9,dtype=torch.float64);p/=p.sum(-1,keepdim=True)
    x=p.sqrt();b=x.mean(0);b/=b.norm()
    torch.testing.assert_close(sphere_exp(b,sphere_log(b,x)),x,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(sphere_log(b,b),torch.zeros_like(b),atol=1e-12,rtol=0)
    torch.testing.assert_close(sphere_exp(b,torch.zeros_like(b)),b)


def test_behavior_spline_knots_and_probability():
    p=torch.softmax(torch.randn(4,9,dtype=torch.float64),-1)
    decoded,_=behavior_curve(p,[-4.,-2.,2.,4.])
    torch.testing.assert_close(decoded,p,atol=1e-12,rtol=1e-12)
    curved,_=behavior_curve(p,torch.linspace(-4,4,20))
    torch.testing.assert_close(curved.sum(-1),torch.ones(20,dtype=torch.float64))


def test_shared_support_preserves_orthogonal_complement_and_gradient():
    torch.manual_seed(3);r=torch.randn(9,2,8,dtype=torch.float64);m=torch.randn(16,dtype=torch.float64)
    b=torch.linalg.qr(torch.randn(16,4,dtype=torch.float64)).Q.T
    c=torch.randn(4,dtype=torch.float64,requires_grad=True)
    out=projected_replacement(r,m,b,c);d=(out-r).flatten(1)
    torch.testing.assert_close(d-(d@b.T)@b,torch.zeros_like(d),atol=1e-12,rtol=0)
    assert torch.autograd.gradcheck(lambda v:projected_replacement(r,m,b,v),(c,))


def test_nine_way_behavior_jacobian_rank_bound():
    matrix=torch.randn(9,32,dtype=torch.float64);x=torch.randn(32,dtype=torch.float64,requires_grad=True)
    jac=torch.autograd.functional.jacobian(lambda v:(matrix@v).softmax(-1),x)
    assert torch.linalg.matrix_rank(jac,atol=1e-10)<=8
    p=(matrix@x).softmax(-1)
    torch.testing.assert_close(hellinger2(p,p),torch.tensor(0.,dtype=torch.float64))


def test_dimensionless_controls_preserve_initial_physical_coordinates():
    torch.manual_seed(7);physical=torch.randn(10,32,dtype=torch.float32)*200
    offset=torch.randn(32,dtype=torch.float64);scale=torch.linspace(2,300,32,dtype=torch.float64)
    controls=((physical.double()-offset)/scale).requires_grad_(True)
    recovered=offset+controls*scale
    torch.testing.assert_close(recovered,physical.double(),atol=1e-12,rtol=1e-12)
    loss=recovered.float().square().mean();loss.backward()
    torch.testing.assert_close(controls.grad,2*physical.double()*scale/physical.numel(),atol=1e-4,rtol=1e-6)
