from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/geometry_map'))
from analyze_local_action_ranking import metrics

class RankingTests(unittest.TestCase):
    def test_inverse_order_and_regret(self):
        x=np.arange(4.);r=metrics(x,x[::-1],x,x,x)
        self.assertAlmostEqual(r['rank_actual_encoded'],-1.);self.assertEqual(r['encoded_regret'],3.)
        self.assertEqual(r['pair_order_accuracy_non_ties'],0.)
    def test_ties_not_counted_as_correct(self):
        x=np.ones(3);r=metrics(x,x,x,x,x)
        self.assertEqual(r['ties_at_abs_1e_minus6'],3);self.assertIsNone(r['pair_order_accuracy_non_ties'])

if __name__=='__main__':unittest.main()
