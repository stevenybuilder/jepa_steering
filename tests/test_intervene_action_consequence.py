from pathlib import Path
import sys
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/geometry_map'))
from intervene_action_consequence import normalized_delta,settings

class InterventionTests(unittest.TestCase):
    def test_equal_native_norm_and_subspace(self):
        torch.manual_seed(7);b=torch.linalg.qr(torch.randn(400,2)).Q;r=torch.randn(400,384);e=torch.randn(3,384)
        d=normalized_delta(e,b,r,.4)
        torch.testing.assert_close(d.norm(dim=-1),torch.ones(3)*.4)
        torch.testing.assert_close(d,(d@b)@b.T,atol=1e-6,rtol=1e-5)
    def test_frozen_controls_and_gates(self):
        rows=settings();self.assertEqual(len(rows),16);self.assertEqual(len({r['name'] for r in rows}),16)
        self.assertEqual(rows[0]['name'],'unsteered');self.assertEqual(rows[1]['dose'],0)
        self.assertEqual(sum(r['basis']=='random' for r in rows),4)
    def test_zero_error_no_fabricated_direction(self):
        b=torch.eye(400)[:,:2];r=torch.randn(400,384)
        self.assertEqual(normalized_delta(torch.zeros(1,384),b,r,1).abs().sum(),0)

if __name__=='__main__':unittest.main()
