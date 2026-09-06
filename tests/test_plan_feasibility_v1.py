from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/geometry_map'))
from capture_plan_feasibility_v1 import xyz_clip,clip_metrics,MEAN,STD

class FeasibilityTests(unittest.TestCase):
    def test_only_confirmed_xyz_clipped(self):
        raw=torch.zeros(30,4);raw[0]=torch.tensor([2.,-3.,.5,9.]);old=raw.clone()
        clipped=xyz_clip(raw)
        self.assertTrue(torch.equal(raw,old));self.assertTrue(torch.equal(clipped[0],torch.tensor([1.,-1.,.5,9.])))
        self.assertTrue(torch.equal(xyz_clip(clipped),clipped))
    def test_official_affine_coordinates_and_metric_counts(self):
        raw=torch.zeros(30,4);raw[0,:2]=torch.tensor([2.,-2.])
        normalized=((raw-torch.tensor(MEAN))/torch.tensor(STD)).reshape(6,20)
        recovered,clipped,m=clip_metrics(normalized)
        torch.testing.assert_close(recovered,raw);self.assertEqual(m['raw_xyz_clipped_count'],2)
        self.assertEqual(m['normalized_changed_translation_coordinate_indices'],[0,1])
        self.assertTrue(m['raw_gripper_unchanged'])

if __name__=='__main__':unittest.main()
