import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("specificity", Path(__file__).resolve().parents[3] / "analysis/mechanism/steering_specificity.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class DecompositionTests(unittest.TestCase):
    def test_constant_all_candidates_and_calls(self):
        r = m.decompose([np.ones((300, 4)), np.ones((300, 4))])
        self.assertEqual(r["scenario_mean_fraction"], 1)
        self.assertEqual(r["candidate_centered_fraction"], 0)

    def test_candidate_variation_only(self):
        c = np.tile([1., 0, 0, 0], (300, 1))
        c[:150] *= -1
        r = m.decompose([c, c])
        self.assertEqual(r["candidate_centered_fraction"], 1)

    def test_changing_call_mean_only(self):
        r = m.decompose([np.ones((300, 4)), -np.ones((300, 4))])
        self.assertEqual(r["between_call_fraction"], 1)

    def test_identity_and_basis_rotation(self):
        rng = np.random.default_rng(1)
        c = [rng.normal(size=(300, 4)) + i for i in range(3)]
        q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        a, b = m.decompose(c), m.decompose([x @ q for x in c])
        self.assertAlmostEqual(sum(a[k] for k in ("candidate_centered_fraction", "between_call_fraction", "scenario_mean_fraction")), 1)
        for key in a:
            self.assertAlmostEqual(a[key], b[key], places=9)

    def test_bad_input_rejected(self):
        for c in ([], [np.ones((299, 4))], [np.zeros((300, 4))], [np.full((300, 4), np.nan)]):
            with self.assertRaises(ValueError):
                m.decompose(c)

    def test_common_output_shift_can_reverse_goal_ranking(self):
        z = np.array([[1.], [2.]])
        g, d = np.array([0.]), np.array([-2.])
        native = np.square(z - g).sum(axis=1)
        shifted = np.square(z + d - g).sum(axis=1)
        predicted_change = 2 * ((z - g) @ d) + d @ d
        np.testing.assert_allclose(shifted - native, predicted_change)
        self.assertNotEqual(native.argmin(), shifted.argmin())

    def test_committed_data_matches_summary(self):
        frame = pd.read_csv(m.DATA / "candidate_specificity_scenarios.csv")
        report = json.loads((m.DATA / "candidate_specificity.json").read_text())
        self.assertEqual(len(frame), 768)
        self.assertEqual(len(report["raw_sha256"]), 768)
        self.assertFalse(frame.duplicated(["task", "arm", "scenario"]).any())
        np.testing.assert_allclose(frame.candidate_centered_fraction + frame.common_to_candidates_fraction, 1)
        np.testing.assert_allclose(frame.between_call_fraction + frame.scenario_mean_fraction, frame.common_to_candidates_fraction)
        self.assertLess(frame.max_relative_coefficient_vs_requested_norm_error.max(), 2e-5)
        self.assertEqual(report["protocol_sha256"], m.sha(m.PROTOCOL))
        self.assertEqual(report["analysis_source_sha256"], m.sha(m.__file__))
        for (task, arm), f in frame.groupby(["task", "arm"]):
            self.assertEqual(len(f), 96)
            for key in ("candidate_centered_fraction", "common_to_candidates_fraction", "between_call_fraction", "scenario_mean_fraction"):
                self.assertAlmostEqual(f[key].mean(), report["summary"][task][arm][key]["mean"], places=12)

    def test_layer_grid_matches_original_export(self):
        grid = pd.read_csv(m.DATA / "layer_mechanism_grid.csv")
        source = pd.read_csv(m.DATA / "all_task_ablation_metrics.csv")
        source = source[source.category == "distribution_layer"]
        self.assertEqual(len(grid), 576)
        self.assertFalse(grid.duplicated(["task", "precision", "arm", "endpoint", "horizon"]).any())
        self.assertTrue((grid[grid.horizon < 3].reduction_vs_native_percent.abs() < 1e-10).all())
        for r in grid.itertuples():
            f = source[(source.task == r.task) & (source.precision == r.precision) & (source.arm == r.arm) & (source.endpoint == f"{r.endpoint}_h{r.horizon}")]
            self.assertEqual(len(f), 1)
            self.assertAlmostEqual(r.learned_mean, f.iloc[0].lineage_weighted_mean, places=12)


if __name__ == "__main__":
    unittest.main()
