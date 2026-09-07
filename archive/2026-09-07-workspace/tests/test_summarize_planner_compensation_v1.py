from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/geometry_map'))
from summarize_planner_compensation_v1 import jacobian_alignment

class JacobianTests(unittest.TestCase):
    def test_exact_linear_inverse_compatible(self):
        j=np.eye(3);b=np.array([1.,2.,3.]);a=-b
        r=jacobian_alignment(j,b,a,j@a)
        self.assertAlmostEqual(r['observed_action_vs_minimum_norm_inverse_bias_cosine'],1.)
        self.assertAlmostEqual(r['linearized_bias_cancellation_projection'],1.)
        self.assertEqual(r['linearization_error_relative_to_actual_action_effect'],0.)
    def test_aligned_disruption_not_cancellation(self):
        r=jacobian_alignment(np.eye(2),np.ones(2),np.ones(2),np.ones(2))
        self.assertAlmostEqual(r['observed_action_vs_minimum_norm_inverse_bias_cosine'],-1.)
        self.assertAlmostEqual(r['observed_bias_cancellation_projection'],-1.)
    def test_shape_and_zero_ambiguity(self):
        with self.assertRaises(ValueError):jacobian_alignment(np.eye(2),np.zeros(3),np.zeros(2),np.zeros(3))
        r=jacobian_alignment(np.eye(2),np.zeros(2),np.zeros(2),np.zeros(2))
        self.assertIsNone(r['observed_action_vs_minimum_norm_inverse_bias_cosine'])
    def test_nullspace_can_hide_exact_feature_compensation(self):
        r=jacobian_alignment(np.array([[1.,0.]]),np.array([1.]),np.array([-1.,100.]),np.array([-1.]))
        self.assertLess(r['observed_action_vs_minimum_norm_inverse_bias_cosine'],.011)
        self.assertAlmostEqual(r['rowspace_action_vs_inverse_bias_cosine'],1.)
        self.assertAlmostEqual(r['linearized_forecast_change_vs_negative_bias_cosine'],1.)
        self.assertGreater(r['action_nullspace_norm_fraction'],.999)

if __name__=='__main__':unittest.main()
