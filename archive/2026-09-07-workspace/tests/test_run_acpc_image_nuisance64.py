import torch
from scripts.geometry_map.run_acpc_image_nuisance64 import spread,mse_rows,tensor_sha
from scripts.geometry_map.run_acpc_image_nuisance import noisy_display,SEED


def test_full64_RMS_denominator():
    clean=torch.ones(6,64,2);clean[:,0]=0
    result=spread(clean,clean+2)
    assert result['RMS_over_native_action_RMS']==[2.]*6
    assert result['mean_mse_by_horizon']==[4.]*6


def test_broader_bank_does_not_change_original9_denominator():
    x=torch.ones(6,64,2);x[:,0]=0;x[:,9:]=10
    a=spread(x,x+2,range(9));b=spread(x,x+2)
    assert a['RMS_over_native_action_RMS']==[2.]*6
    assert all(v<.3 for v in b['RMS_over_native_action_RMS'])


def test_tiny_denominator_not_inflated():
    x=torch.zeros(6,64,3);r=spread(x,x+1)
    assert r['RMS_over_native_action_RMS']==[None]*6 and all(r['denominator_tiny'])


def test_full_spatial_MSE_not_pooled():
    x=torch.zeros(6,64,2,2);y=x.clone();y[:,:,0]=1;y[:,:,1]=-1
    assert torch.equal(mse_rows(x,y),torch.ones(6,64,dtype=torch.float64))


def test_same_draw_independent_of_candidate_batch():
    x=torch.full((1,1,3,8,8),255.);a,m=noisy_display(x,SEED+2)
    torch.randn(64,60);b,n=noisy_display(x,SEED+2)
    assert torch.equal(a,b) and m==n and tensor_sha(a)==tensor_sha(b)


def test_hash_covers_shape_and_dtype_and_first9():
    x=torch.arange(64*60).reshape(64,30,2).float()
    assert tensor_sha(x[:9])==tensor_sha(x[:9].clone())
    assert tensor_sha(x)!=tensor_sha(x.reshape(64,60))
    assert tensor_sha(x)!=tensor_sha(x.double())
