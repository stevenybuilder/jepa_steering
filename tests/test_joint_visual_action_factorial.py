import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/geometry_map'))
from run_joint_visual_action_factorial import input_edits,vector_interaction,metric_decomposition


class Tests(unittest.TestCase):
    def test_true_visual_field_and_ownership(self):
        torch.manual_seed(1);v=torch.randn(3,2,1,16,16,384);a=torch.randn(3,2,10);p=torch.randn(3,2,256,16);d=v[:,-1]-v[0:1,-1];saved=v.clone()
        changed=input_edits((v,a,p),d,'joint',.1)
        self.assertIs(changed[1],a);self.assertIs(changed[2],p);self.assertTrue(torch.equal(v,saved))
        self.assertTrue(torch.equal(changed[0][:,:-1],v[:,:-1]));self.assertTrue(torch.equal(changed[0][0],v[0]))
        self.assertGreater(float((changed[0][1:]-v[1:]).norm()),0.)
    def test_zero_and_action_do_not_touch_visual(self):
        v=torch.randn(3,2,1,16,16,384);args=(v,torch.randn(3,2,10),torch.randn(3,2,256,16));d=v[:,-1]-v[0:1,-1]
        self.assertIs(input_edits(args,d,'joint',0.),args);self.assertIs(input_edits(args,d,'action',.1),args)
    def test_spatial_permutation_norm_match(self):
        torch.manual_seed(3);v=torch.randn(3,2,1,16,16,384);args=(v,torch.randn(3,2,10),torch.randn(3,2,256,16));d=v[:,-1]-v[0:1,-1]
        dense=input_edits(args,d,'visual',.1)[0]-v;sham=input_edits(args,d,'permuted_visual',.1)[0]-v
        torch.testing.assert_close(dense.double().flatten(1).norm(dim=-1),sham.double().flatten(1).norm(dim=-1),rtol=2e-6,atol=2e-6)
    def test_additive_vectors_nonzero_metric_cross_term(self):
        base=torch.zeros(1,2,3);visual=base+1;action=base+2;joint=base+3
        r,stats,additive=vector_interaction(base,visual,action,joint)
        self.assertTrue(torch.equal(r,torch.zeros_like(r)))
        d=metric_decomposition(base,visual,action,joint,base)
        self.assertEqual(d['metric_interaction'],[[4.,4.]]);self.assertEqual(d['nonlinear_output_coupling_term'],[[0.,0.]])
    def test_bilinear_coupling_and_metric_identity(self):
        x=torch.tensor([[[2.,3.]]]);z=torch.tensor([[[4.,5.]]]);dx=.1*x;dz=.1*z
        b=x*z;v=(x+dx)*z;a=x*(z+dz);joint=(x+dx)*(z+dz)
        r,_,_=vector_interaction(b,v,a,joint)
        torch.testing.assert_close(r,(dx*dz).double(),rtol=2e-5,atol=2e-6)
        d=metric_decomposition(b,v,a,joint,torch.ones_like(b));self.assertLess(d['identity_decomposition_maxabs'],1e-10)


if __name__=='__main__':unittest.main()
