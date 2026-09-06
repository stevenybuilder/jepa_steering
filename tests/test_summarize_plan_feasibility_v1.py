import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/geometry_map'))
from summarize_plan_feasibility_v1 import forecast_error,rank_agreement


class FeasibilitySummaryTests(unittest.TestCase):
    def test_ranks_and_ties(self):
        r=rank_agreement([1,2,3],[3,2,1])
        self.assertAlmostEqual(r['spearman'],-1)
        self.assertEqual(r['discordant_pair_fraction'],1)
        self.assertIsNone(rank_agreement([1,1,1],[1,2,3])['spearman'])
    def test_full_spatial_metric_excludes_context(self):
        v={'predicted_visual':np.zeros((7,1,2,3)), 'predicted_proprio':np.zeros((7,1,4)),
           'actual_encoded':{'visual':np.ones((6,6)), 'proprio':np.ones((6,4))*2}}
        v['predicted_visual'][0]=100
        np.testing.assert_allclose(forecast_error(v),1.4)
    def test_shape_guard(self):
        with self.assertRaises(ValueError):rank_agreement([1,2],[1])


if __name__=='__main__':unittest.main()
