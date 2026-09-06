import importlib.util
from pathlib import Path
import unittest

PATH=Path(__file__).resolve().parents[1]/'scripts'/'geometry_map'/'summarize_action_path_curvature.py'
spec=importlib.util.spec_from_file_location('summary',PATH);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class SummaryTests(unittest.TestCase):
    def test_interval_units_are_sources(self):
        out=module.episode_interval([1,1,3,3])
        self.assertEqual(out['n_initial_states'],4)
        self.assertEqual(out['mean'],2)
        self.assertEqual(out['initial_state_values'],[1,1,3,3])

    def test_incomplete_panel_is_rejected(self):
        with self.assertRaises(ValueError): module.summarize([])
