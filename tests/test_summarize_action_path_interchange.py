import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'/'geometry_map'))
from summarize_action_path_interchange import effect_fraction,summarize


class InterchangeSummaryTests(unittest.TestCase):
    def test_fraction_and_tiny_denominator(self):
        self.assertEqual(effect_fraction(.25,1),.75)
        self.assertEqual(effect_fraction(2,1),-1)
        self.assertIsNone(effect_fraction(0,0))
        self.assertIsNone(effect_fraction(0,1e-13))
    def test_incomplete_rejected(self):
        with self.assertRaises(ValueError):summarize([])
