import torch
from scripts.geometry_map.diagnose_p5_suffix import decomposition, layernorm_jvp, suffix, contrast_rank


def test_decomposition_orthogonal_and_complete():
    torch.manual_seed(12)
    x=torch.randn(3,5,400,dtype=torch.float64)
    delta=torch.randn_like(x)
    mean,radial,tangent=decomposition(x,delta)
    torch.testing.assert_close(mean+radial+tangent,delta)
    for a,b in ((mean,radial),(mean,tangent),(radial,tangent)):
        assert float((a*b).sum(-1).abs().max())<1e-11


def test_layernorm_jvp_matches_autograd():
    torch.manual_seed(13)
    x=torch.randn(2,4,400,dtype=torch.float64)
    delta=torch.randn_like(x)
    weight=torch.randn(400,dtype=torch.float64)
    _,expected=torch.autograd.functional.jvp(lambda z:torch.nn.functional.layer_norm(z,(400,),weight,eps=1e-6),x,delta)
    torch.testing.assert_close(layernorm_jvp(x,delta,weight),expected,rtol=1e-11,atol=1e-11)


def test_mean_null_and_radial_epsilon_effect():
    torch.manual_seed(14)
    x=torch.randn(2,3,400,dtype=torch.float64)
    weight=torch.ones(400,dtype=torch.float64)
    assert layernorm_jvp(x,torch.ones_like(x),weight).abs().max()==0
    radial=x-x.mean(-1,keepdim=True)
    observed=layernorm_jvp(x,radial,weight)
    expected=radial*1e-6/(radial.square().mean(-1,keepdim=True)+1e-6).pow(1.5)
    torch.testing.assert_close(observed,expected,rtol=1e-8,atol=1e-12)


def test_exact_split_and_projection_location():
    torch.manual_seed(15)
    x=torch.randn(2,3,400)
    params=dict(norm_weight=torch.ones(400),norm_bias=torch.zeros(400),projection_weight=2*torch.eye(384),projection_bias=torch.ones(384))
    norm,visual,proprio=suffix(x,params)
    torch.testing.assert_close(visual,2*norm[...,:384]+1)
    torch.testing.assert_close(proprio,norm[...,384:])
    torch.testing.assert_close(suffix(x+2,params)[0],norm,atol=2e-6,rtol=2e-6)


def test_nine_plan_design_rank_bound():
    torch.manual_seed(16)
    values=torch.randn(9,60,dtype=torch.float64)
    report=contrast_rank(values)
    assert report['rank_relative1e_6']==8
    assert contrast_rank(torch.ones(9,60))['rank_relative1e_6']==0
