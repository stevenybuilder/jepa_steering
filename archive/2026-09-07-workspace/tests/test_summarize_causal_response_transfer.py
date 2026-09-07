from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/geometry_map'))
from summarize_causal_response_transfer import response_errors

class SummaryResponseTests(unittest.TestCase):
    def test_exact_response(self):
        a=np.ones((7,2,384));r=response_errors(a,a)
        self.assertEqual(r['relative_error'],0.);self.assertAlmostEqual(r['cosine'],1.)
    def test_opposite_response(self):
        a=np.ones((7,2,384));r=response_errors(-a,a)
        self.assertAlmostEqual(r['relative_error'],2.);self.assertAlmostEqual(r['cosine'],-1.)

if __name__=='__main__':unittest.main()
