from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'/'geometry_map'))
from probe_action_interactions import factorial_actions,interaction_vectors,CaptureStages,summarize_stages

class ActionInteractionTests(unittest.TestCase):
    def test_factorial_gripper_future_exact_no_clipping(self):
        original=torch.randn(30,4)
        original[15,0]=3.5
        actions,rows=factorial_actions(original,'native_future_fixed')
        self.assertEqual(len(rows),17)
        self.assertTrue(torch.equal(actions[:,5:],original[5:].expand(17,-1,-1)))
        self.assertTrue(torch.equal(actions[:,:,3],original[:,3].expand(17,-1)))
        self.assertEqual(float(actions[0,:5,:3].abs().max()),0)
        self.assertAlmostEqual(float(actions[:,:5,:3].abs().max()),.05,places=7)
        primary,_=factorial_actions(original)
        self.assertTrue(primary[:,5:].eq(0).all());self.assertTrue(primary[:,:,3].eq(0).all())
    def test_bilinear_signed_interaction(self):
        x,z=.025,-.05
        fn=lambda a,b:torch.tensor([[2*a+3*b+7*a*b]],dtype=torch.float64)
        value=interaction_vectors(fn(0,0),fn(x,0),fn(0,z),fn(x,z))
        self.assertAlmostEqual(float(value),7*x*z,places=14)
    def test_quadratic_curvature_and_linear_zero_interaction(self):
        _,rows=factorial_actions(torch.zeros(30,4,dtype=torch.float64))
        values=torch.tensor([2*r['raw_amplitude']*r['x_sign']+3*(r['raw_amplitude']*r['x_sign'])**2+4*r['raw_amplitude']*r['z_sign'] for r in rows],dtype=torch.float64)
        stages={'P0':values[None,:,None,None].expand(6,-1,256,1)}
        metrics,signed=summarize_stages(stages,rows)
        self.assertAlmostEqual(float(signed['P0/1.0/x/curvature'][0]),6*.05**2,places=13)
        self.assertTrue(all(r['mixed_difference_l2']<1e-12 for r in metrics if r['kind']=='xz_interaction'))
    def test_observer_hooks_identity_and_cleanup(self):
        block=torch.nn.Identity();value=torch.randn(17,512,400)
        with CaptureStages({'P3':block}) as cap:self.assertIs(block(value),value)
        self.assertTrue(torch.equal(cap.values['P3'][0],value[:,-256:]))
        self.assertEqual(len(block._forward_hooks),0)
    def test_bad_shape_rejected(self):
        with self.assertRaises(ValueError):factorial_actions(torch.zeros(15,4))

if __name__=='__main__':unittest.main()
