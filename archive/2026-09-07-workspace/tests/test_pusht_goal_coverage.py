import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/geometry_map'))
from diagnose_pusht_goal_coverage import coverage,polygon


class GoalCoverageTests(unittest.TestCase):
    def test_identity_and_disjoint(self):
        self.assertAlmostEqual(coverage([13,24,.8],[13,24,.8]),1,places=12)
        self.assertEqual(coverage([0,0,0],[1000,1000,0]),0)
    def test_rotation_matters(self):
        self.assertLess(coverage([256,256,0],[256,256,np.pi/2]),.9)
    def test_translation_rotation_invariance_and_area(self):
        self.assertAlmostEqual(polygon([0,0,0]).area,6300)
        self.assertAlmostEqual(coverage([0,0,0],[10,20,0]),coverage([40,-10,.5],[40+10*np.cos(.5)-20*np.sin(.5),-10+10*np.sin(.5)+20*np.cos(.5),.5]),places=12)


if __name__=='__main__':unittest.main()
