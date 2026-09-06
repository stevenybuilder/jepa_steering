import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/geometry_map'))
import torch
from run_action_ranking_jacobian import whiten_delta,action_tangent,metric_vector

class JacobianTests(unittest.TestCase):
    def test_isotropic_metric_preserves_edit(self):
        u=torch.eye(3,dtype=torch.float64)[:,:2][None];delta=torch.tensor([[1.,2.,3.]])
        result,_=whiten_delta(delta,u,torch.eye(2,dtype=torch.float64)[None]*3)
        torch.testing.assert_close(result,delta)
    def test_anisotropy_same_norm_and_basis_rotation_equivariance(self):
        u=torch.eye(3,dtype=torch.float64)[:,:2][None];delta=torch.tensor([[1.,1.,1.]])
        gram=torch.diag(torch.tensor([1.,9.],dtype=torch.float64))[None]
        result,_=whiten_delta(delta,u,gram)
        self.assertGreater(float(result[0,0]),float(result[0,1]));torch.testing.assert_close(result.norm(),delta.norm())
        q=torch.tensor([[0.,-1.],[1.,0.]],dtype=torch.float64)
        rotated,_=whiten_delta(delta,u@q,q.T@gram@q)
        torch.testing.assert_close(rotated,result)
    def test_metric_matches_native_modality_weighting(self):
        v=torch.ones(2,9,1,2);p=torch.ones(2,9,1,3)*2
        result=metric_vector({'visual':v,'proprio':p},.1,False)
        torch.testing.assert_close(result.square().sum(1),torch.full((9,),1.4,dtype=torch.float64))
        summed=metric_vector({'visual':v,'proprio':p},.1,True)
        torch.testing.assert_close(summed.square().sum(1),torch.full((9,),2.8,dtype=torch.float64))
    def test_tangent_rank_and_zero_unidentifiable_metric(self):
        donors=torch.zeros(8,9,2,3)
        donors[1::2,:,:,0]=1;donors[0::2,:,:,0]=-1
        u,s,active=action_tangent(donors)
        self.assertTrue(torch.equal(active.sum(1),torch.ones(9,dtype=torch.int64)))
        delta=torch.randn(9,6);result,_=whiten_delta(delta,u,torch.zeros(9,4,4,dtype=torch.float64))
        torch.testing.assert_close(result,delta)
        gram=torch.zeros(9,4,4,dtype=torch.float64);gram[:,0,0]=3
        result,_=whiten_delta(delta,u,gram)
        torch.testing.assert_close(result,delta)

if __name__=='__main__':unittest.main()
