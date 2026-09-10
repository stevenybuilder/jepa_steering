import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/geometry_map'))
from analyze_xyz_projected_planner_v1 import action_metrics


class ActionMetricsTests(unittest.TestCase):
    def test_xyz_only_metrics(self):
        r=action_metrics(torch.tensor([[2.,0.,-2.,3.],[0.,0.,0.,0.]]))
        self.assertAlmostEqual(r['raw_xyz_out_of_bounds_fraction'],1/3)
        self.assertAlmostEqual(r['raw_xyz_l2'],2*(2**.5))
        self.assertAlmostEqual(r['effective_xyz_l2'],2**.5)
        self.assertEqual(r['raw_gripper_out_of_bounds_fraction_not_projected'],.5)


if __name__=='__main__':unittest.main()
