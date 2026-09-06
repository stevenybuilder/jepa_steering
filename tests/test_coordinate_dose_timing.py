from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'/'geometry_map'))
from run_coordinate_dose_timing import DoseTimingHook,shuffled_signs,ARMS
from coordinate_calibration_operator import CoordinateCalibration,CoordinateCalibrationHook
from test_coordinate_calibration import fixture

class Fake(torch.nn.Module):
    def __init__(self):super().__init__();self.block=torch.nn.Identity();self.last=None
    def forward_pred(self,v):
        self.block(torch.ones(len(v),256,400,dtype=v.dtype));self.last=(v,torch.ones(1),torch.ones(1));return self.last

class DoseTests(unittest.TestCase):
    def test_independent_exact_norm_shams(self):
        v=torch.randn(300,384)
        for seed in (6101,6102,6103):
            signs=shuffled_signs(seed);self.assertEqual(int((signs<0).sum()),192)
            self.assertTrue(torch.equal(v.norm(dim=-1),(v*signs).norm(dim=-1)))
        self.assertFalse(torch.equal(shuffled_signs(6101),shuffled_signs(6102)))
    def test_reference_matches_original_and_quarter_after_cap(self):
        wm=Fake();op=CoordinateCalibration(fixture(),dtype=torch.float64);v=torch.randn(2,1,1,16,16,384,dtype=torch.float64)
        with CoordinateCalibrationHook(wm,wm.block,op):old=wm.forward_pred(v)
        with DoseTimingHook(wm,wm.block,op,1) as hook:
            hook.reset(record=True);new=wm.forward_pred(v);full=hook.records[0]['delivered_delta']
        self.assertTrue(torch.equal(old[0],new[0]))
        with DoseTimingHook(wm,wm.block,op,.25) as hook:
            hook.reset(record=True);wm.forward_pred(v);quarter=hook.records[0]['delivered_delta']
        self.assertTrue(torch.equal(quarter,full*.25))
    def test_first_only_and_reset_and_beta_zero_identity(self):
        wm=Fake();op=CoordinateCalibration(fixture(),dtype=torch.float64);v=torch.randn(2,1,1,16,16,384,dtype=torch.float64)
        with DoseTimingHook(wm,wm.block,op,.25,True) as hook:
            hook.reset();self.assertFalse(torch.equal(wm.forward_pred(v)[0],v))
            for _ in range(5):self.assertIs(wm.forward_pred(v),wm.last)
            hook.reset();self.assertFalse(torch.equal(wm.forward_pred(v)[0],v))
        self.assertEqual(len(wm.block._forward_hooks),0)
        with DoseTimingHook(wm,wm.block,op,0) as hook:
            hook.reset();self.assertIs(wm.forward_pred(v),wm.last)
    def test_frozen_arm_count(self):
        self.assertEqual(len(ARMS),7)
        self.assertEqual([x[1] for x in ARMS],[0.,1.,1.,.25,.25,.25,.25])

if __name__=='__main__':unittest.main()
