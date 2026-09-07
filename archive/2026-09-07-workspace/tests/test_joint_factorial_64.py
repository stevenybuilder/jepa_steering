import sys
from pathlib import Path
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/geometry_map'))
from prepare_joint_factorial_64 import same
from run_joint_visual_action_factorial import input_edits, vector_interaction, metric_decomposition
from run_joint_factorial_64 import FACTORS


class Tests(unittest.TestCase):
    def test_batch64_visual_boundary_zero_and_ownership(self):
        torch.manual_seed(7)
        v=torch.randn(64,2,1,16,16,384);a=torch.randn(64,2,10);p=torch.randn(64,2,256,16)
        d=v[:,-1]-v[0:1,-1];before=v.clone()
        out=input_edits((v,a,p),d,'joint',.1)
        self.assertEqual(out[0].shape,v.shape);self.assertTrue(torch.equal(v,before))
        self.assertTrue(torch.equal(out[0][0],v[0]));self.assertTrue(torch.equal(out[0][:,:-1],v[:,:-1]))
        self.assertIs(out[1],a);self.assertIs(out[2],p)
        self.assertIs(input_edits((v,a,p),d,'joint',0.)[0],v)
        self.assertEqual(1+2*len(FACTORS),11)
    def test_nested_physics_mismatch_rejected(self):
        x={'states':np.zeros((31,7),dtype=np.float32),'physics':[{'step':0,'body':{'position':[1.,2.]}}]}
        same(x,x,'identity')
        y={'states':x['states'].copy(),'physics':[{'step':0,'body':{'position':[1.,2.00001]}}]}
        with self.assertRaises(ValueError):same(x,y,'physics')
    def test_all64_metric_additive_null(self):
        torch.manual_seed(9);b=torch.randn(6,64,3);v=b+.25;a=b+.5;j=b+.75
        r,_,_=vector_interaction(b,v,a,j)
        self.assertLess(float(r.abs().max()),5e-7)
        m=metric_decomposition(b,v,a,j,torch.ones_like(b));self.assertLess(m['identity_decomposition_maxabs'],1e-10)


if __name__=='__main__':unittest.main()
