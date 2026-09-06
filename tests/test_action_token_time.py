import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts/geometry_map"))
from analyze_action_token_time import spatial_response, stability, denominator_floor


class SpatialTests(unittest.TestCase):
    def conditions(self):
        signs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
        return [{"scale":0.,"x_sign":0,"z_sign":0}]+[{"scale":s,"x_sign":x,"z_sign":z} for s in (.5,1.) for x,z in signs]

    def values(self, nonlinear=False):
        conditions = self.conditions(); values = np.zeros((6,17,256,2))
        for i, row in enumerate(conditions):
            x,z = row["scale"]*row["x_sign"], row["scale"]*row["z_sign"]
            values[:,i,:,0] = x+(x*x if nonlinear else 0)
            values[:,i,:,1] = z+(x*z if nonlinear else 0)
        return values,conditions

    def test_linear_curvature_and_mixed_zero(self):
        response = spatial_response(*self.values(), 1.)
        self.assertEqual(response["curvature_numerator_l2"].max(),0)
        self.assertEqual(response["mixed_numerator_l2"].max(),0)

    def test_quadratic_and_bilinear_detected(self):
        response = spatial_response(*self.values(True), 1.)
        self.assertAlmostEqual(float(response["curvature_numerator_l2"][0,0]),2)
        self.assertAlmostEqual(float(response["mixed_numerator_l2"][0,0]),1)

    def test_zero_denominator_floor_is_finite(self):
        self.assertTrue((denominator_floor(np.zeros((6,256)), np.zeros((6,256)))>0).all())
        values, conditions = self.values(); values[:]=0
        self.assertTrue(np.isfinite(spatial_response(values,conditions,1.)["curvature_ratio"]).all())

    def test_rank_stability_identical(self):
        response=stability(np.tile(np.arange(256),(6,1)))
        self.assertEqual(response["comparisons"][1]["intersection_fraction"],1.)


if __name__ == "__main__":
    unittest.main()
