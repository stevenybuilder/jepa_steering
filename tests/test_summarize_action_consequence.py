from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/geometry_map'))
from summarize_action_consequence import ci
class SummaryTests(unittest.TestCase):
    def test_group_n_and_pairing(self):
        self.assertEqual(ci([1.,2.,3.,4.])['n_initialstates'],4)
        self.assertEqual(ci([1.,2.,3.,4.])['mean'],2.5)
    def test_exact_identity(self):self.assertEqual(ci([0.]*8)['interval95'],[0.,0.])
if __name__=='__main__':unittest.main()
