from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/geometry_map'))
from capture_action_patch_geometry import head_geometry

class HeadGeometryTests(unittest.TestCase):
    def test_sum_plus_shared_bias(self):
        torch.manual_seed(4);x=torch.randn(2,7,8);w=torch.randn(8,8);b=torch.randn(8);gate=torch.randn(2,7,8)
        stats,total=head_geometry(x,w,b,gate,2)
        torch.testing.assert_close(total,torch.nn.functional.linear(x,w,b)*gate)
        self.assertEqual(stats['head_cosine'].shape,(2,2,2));self.assertEqual(stats['head_norm'].shape,(2,2))
    def test_zero_gate_no_fake_norm(self):
        x=torch.ones(1,2,8);stats,total=head_geometry(x,torch.eye(8),None,torch.zeros_like(x),2)
        self.assertEqual(float(stats['head_norm'].sum()),0.);self.assertEqual(float(total.abs().sum()),0.)
    def test_input_unchanged(self):
        x=torch.randn(1,256,8);saved=x.clone();head_geometry(x,torch.eye(8),None,torch.ones_like(x),2)
        self.assertTrue(torch.equal(x,saved))

if __name__=='__main__':unittest.main()
