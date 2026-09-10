import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/geometry_map'))
from analyze_bounded_elite_v1 import roundtrip_gripper_error


class GripperDiagnosticTests(unittest.TestCase):
    def test_xyz_change_does_not_count_as_gripper(self):
        n=torch.zeros(6,3,20);e=n.clone();e[...,0::4]=1
        self.assertEqual(roundtrip_gripper_error(n,e),0)
    def test_reports_gripper_change(self):
        n=torch.zeros(6,3,20);e=n.clone();e[...,3::4]=1
        self.assertGreater(roundtrip_gripper_error(n,e),.7)


if __name__=='__main__':unittest.main()
