import torch
from scripts.geometry_map.run_normalization_aware_subspace import fit_action_subspaces,weighted_rms,first_crossing_match,SubspacePatch


def test_rank4_energy_and_random_input_norm_match():
    torch.manual_seed(71);d=torch.randn(5,8,12);d[0]=0
    b,q,p,r,s=fit_action_subspaces(d,17)
    assert b.shape==q.shape==(12,4)
    assert s['random_vs_action_subspace_maxabs']<1e-12
    assert 0<s['retained_contrast_energy']<1
    torch.testing.assert_close(p.double().flatten(1).norm(dim=1),r.double().flatten(1).norm(dim=1),rtol=1e-6,atol=1e-6)
    assert torch.equal(p[0],torch.zeros_like(p[0]))


def test_exact_rank4_contrast_reconstruction():
    torch.manual_seed(72);b=torch.linalg.qr(torch.randn(12,4,dtype=torch.float64)).Q
    d=torch.randn(5,8,4,dtype=torch.float64)@b.T
    _,_,p,_,s=fit_action_subspaces(d,7)
    torch.testing.assert_close(p,d,rtol=1e-10,atol=1e-10)
    assert abs(s['retained_contrast_energy']-1)<1e-12


def test_weighted_RMS_units_not_raw_channel_count():
    v=torch.ones(3,256,384)*2;p=torch.ones(3,256,16)*3
    torch.testing.assert_close(weighted_rms(v,p),torch.full((3,),4.9**.5,dtype=torch.float64))


def test_first_bracket_not_global_monotonic_assumption():
    target=torch.tensor([.5],dtype=torch.float64);radius=torch.ones_like(target)
    fn=lambda x:torch.sin(torch.pi*x).abs()
    amp,valid,info=first_crossing_match(fn,target,radius)
    assert valid.all() and abs(float(amp)-1/6)<1e-6
    assert info['upward_crossings']==[1] and info['downward_crossings']==[1]


def test_infeasible_returns_zero_without_increasing_radius():
    target=torch.tensor([2.,0.],dtype=torch.float64);radius=torch.ones_like(target)
    amp,valid,info=first_crossing_match(lambda x:x,target,radius)
    assert not valid.any() and torch.equal(amp,torch.zeros_like(amp))
    assert info['feasible']==[False,False]


def test_multiple_crossings_choose_first():
    target=torch.tensor([.5],dtype=torch.float64);radius=torch.ones_like(target)
    amp,valid,info=first_crossing_match(lambda x:torch.sin(3*torch.pi*x).abs(),target,radius)
    assert valid.all() and abs(float(amp)-1/18)<1e-6
    assert info['upward_crossings']==[3]


def test_off_subspace_residual_preserved_in_real_arithmetic():
    torch.manual_seed(2);b=torch.linalg.qr(torch.randn(12,4,dtype=torch.float64)).Q
    x=torch.randn(2,8,12,dtype=torch.float64);d=torch.randn(2,8,4,dtype=torch.float64)@b.T
    before=x-(x@b)@b.T;after=(x+d)-((x+d)@b)@b.T
    torch.testing.assert_close(before,after,rtol=1e-12,atol=1e-12)
