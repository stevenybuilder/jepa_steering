import torch
from scripts.geometry_map.run_acpc_image_nuisance import noisy_display,paired_spread,initial_rgb,rank_indices


def test_native_stable_ranking_keyword_contract():
    costs=torch.tensor([[2.,1.,1.],[0.,2.,1.]])
    assert rank_indices(costs).tolist()==[[1,2,0],[0,2,1]]


def test_native_observation_singleton_preserves_exact_pixels():
    frame=torch.arange(48).reshape(1,3,4,4).to(torch.uint8)
    raw=initial_rgb(frame)
    assert raw.shape==(1,1,3,4,4)
    assert torch.equal(raw[0],frame.float())


def test_zero_noise_identity_and_no_mutation():
    x=torch.arange(48).reshape(1,1,3,4,4).to(torch.uint8);before=x.clone()
    y,meta=noisy_display(x,1,0.)
    assert torch.equal(y,x.float()) and torch.equal(x,before)
    assert meta['realized_display_RMS']==0


def test_rng_fixed_independent_of_global_rng():
    x=torch.full((1,1,3,8,8),127.)
    a,_=noisy_display(x,100);torch.manual_seed(999);torch.randn(60)
    b,_=noisy_display(x,100);c,_=noisy_display(x,101)
    assert torch.equal(a,b) and not torch.equal(a,c)


def test_clipping_and_pixel_units():
    x=torch.zeros(1,1,3,40,40)
    y,m=noisy_display(x,3,.01)
    assert 0<=y.min() and y.max()<=255
    assert .45<m['clipped_fraction']<.55
    assert 0<m['realized_display_RMS']<.01
    assert not m['requantized_to_uint8']


def test_normalization_is_RMS_not_MSE_ratio():
    clean=torch.ones(6,9,2);clean[:,0]=0
    noisy=clean+2
    result=paired_spread(clean,noisy)
    assert result['RMS_over_native_action_RMS']==[2.]*6
    assert result['mean_mse_by_horizon']==[4.]*6


def test_no_action_response_flagged_not_inflated():
    clean=torch.zeros(6,9,2)
    result=paired_spread(clean,clean+1)
    assert result['RMS_over_native_action_RMS']==[None]*6
    assert all(result['denominator_tiny'])
