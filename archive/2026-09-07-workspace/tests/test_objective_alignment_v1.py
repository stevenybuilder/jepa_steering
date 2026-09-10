import importlib.util
from pathlib import Path
import unittest
import numpy as np

SOURCE=Path(__file__).parents[1]/'scripts/geometry_map/run_objective_alignment_v1.py'
spec=importlib.util.spec_from_file_location('objective_alignment_v1',SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class ObjectiveAlignmentTest(unittest.TestCase):
    def test_rank_ties(self):
        np.testing.assert_array_equal(m.ranks([4,1,1,3]),[3,.5,.5,2])
        self.assertIsNone(m.rho([1,1],[2,3]))
    def test_pair_direction_and_ties(self):
        r=m.paired_order([1,2,2],[3,1,1])
        self.assertEqual(r['agreement'],1);self.assertEqual(r['cost_ties'],1)
        self.assertEqual(m.paired_order([3,2,1],[3,2,1])['agreement'],0)
    def test_oracle_not_physical(self):
        r=m.horizon_summary([1,2,3],[3,1,2],[.8,.2,.9],[3,2,1])
        self.assertEqual(r['native_predicted']['index'],0)
        self.assertEqual(r['actual_encoding_cost_oracle']['index'],1)
        self.assertEqual(r['requested_coverage_oracle']['index'],2)
        self.assertAlmostEqual(r['actual_encoding_cost_oracle']['coverage_regret'],.7)
    def test_identity_margin_certificate(self):
        r=m.horizon_summary([1,2,4],[1,2,4],[.9,.5,.1],[1,2,3])
        self.assertTrue(r['sufficient_native_winner_certificate'])
        self.assertEqual(r['sufficient_pair_cost_rank_certificate_fraction'],1)
    def test_fixed_cost_weight_and_goal_zero(self):
        import torch
        v=torch.ones(6,2,3);p=torch.full((6,4),2.)
        np.testing.assert_allclose(m.native_cost(v,p,torch.zeros(2,3),torch.zeros(4)),1.4)
        np.testing.assert_array_equal(m.native_cost(v,p,v[0],p[0]),np.zeros(6))
    def test_cost_tie_reports_outcome_range(self):
        r=m.selection([1,1,2],[.3,.9,.5],[3,1,2])
        self.assertEqual(r['index'],0);self.assertEqual(r['tie_count'],2)
        self.assertEqual(r['tied_coverage_range'],[.3,.9])


if __name__=='__main__':unittest.main()
