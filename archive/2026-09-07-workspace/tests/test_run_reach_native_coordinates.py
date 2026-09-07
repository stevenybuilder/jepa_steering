import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'/'geometry_map'))
from run_reach_native_coordinates import action_basis, correction, fit_error_target, rank_metrics


def test_action_basis_support_and_orthogonality():
    basis=action_basis()
    assert basis.shape==(4,60)
    assert torch.equal(basis.reshape(4,15,4)[:,:,3],torch.zeros(4,15))
    torch.testing.assert_close(basis@basis.T,torch.eye(4))


def test_rotation_does_not_invent_patch_change():
    torch.manual_seed(3)
    basis=torch.linalg.qr(torch.randn(12,3,dtype=torch.float64)).Q.T
    response=torch.randn(3,20,dtype=torch.float64);target=torch.randn(20,dtype=torch.float64)
    rotation=torch.linalg.qr(torch.randn(3,3,dtype=torch.float64)).Q
    a,_=correction(basis,response,target);b,_=correction(rotation@basis,rotation@response,target)
    torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)


def test_target_fit_never_accepts_test_labels():
    torch.manual_seed(4)
    train=torch.randn(10,8);error=torch.randn(10,6);test=torch.randn(5,8)
    predicted,basis,metadata=fit_error_target(train,error,test)
    assert predicted.shape==(5,6) and basis.shape==(3,8) and metadata['training_examples']==10


def test_ranking_ties_are_explicit():
    result=rank_metrics(torch.ones(5),torch.arange(5.))
    assert result['spearman'] is None and result['predicted_min_ties']==5
