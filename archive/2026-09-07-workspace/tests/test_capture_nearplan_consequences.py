from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/geometry_map'))
from capture_nearplan_consequences import local_actions

class NearPlanTests(unittest.TestCase):
    def test_fixed_radius_antithetic(self):
        center=torch.ones(30,2)*.12;bank=local_actions(center,0)
        self.assertTrue(torch.equal(bank[0],center));self.assertEqual(bank.shape,(9,30,2))
        for i in (1,3,5,7):
            torch.testing.assert_close(bank[i]+bank[i+1],2*center)
            self.assertAlmostEqual(float((bank[i]-center).square().mean().sqrt()),.01,places=7)
    def test_seed_repeat_and_devguard(self):
        center=torch.zeros(30,2);self.assertTrue(torch.equal(local_actions(center,1),local_actions(center,1)))
        self.assertFalse(torch.equal(local_actions(center,1),local_actions(center,2)))
        with self.assertRaises(ValueError):local_actions(center,4)

if __name__=='__main__':unittest.main()
