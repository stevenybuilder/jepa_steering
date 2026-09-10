import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'/'geometry_map'))
from summarize_actual_context_curvature import summarize


class ActualContextSummaryTests(unittest.TestCase):
    def test_no_partial_panel_claim(self):
        with self.assertRaises(ValueError):summarize([])
