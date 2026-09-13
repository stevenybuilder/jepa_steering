import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("pathway_geometry", Path(__file__).resolve().parents[1]/"analysis/mechanism/pathway_geometry.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class PathwayGeometryTests(unittest.TestCase):
    def test_additive_outputs_can_have_loss_interaction(self):
        value = m.factorial_terms([0, 0], [1, 2], [3, 4], [4, 6], [8, 9])
        self.assertEqual(value["output_interaction_mse"], 0)
        self.assertEqual(value["factorial_mse_interaction"], 11)
        self.assertEqual(value["additive_quadratic_cross"], 11)
        self.assertEqual(value["nonadditive_mse_remainder"], 0)

    def test_nonlinear_interaction_matches_exact_expansion(self):
        n, v, a, j, y = np.random.default_rng(7).normal(size=(5, 12))
        value = m.factorial_terms(n, v, a, j, y)
        r = j-v-a+n
        expected = 2*np.mean((n-y+v-n+a-n)*r)+np.mean(r*r)
        self.assertAlmostEqual(value["nonadditive_mse_remainder"], expected)
        self.assertGreater(value["output_interaction_mse"], 0)

    def test_group_records_reject_duplicates_missing_and_nonfinite(self):
        r = {"lineage_group": "a", "metrics": {"mse": 1.}}
        for records, ids in (([r, r], None), ([r], ["a", "b"]),
                             ([{"lineage_group": "a", "metrics": {"mse": float("nan")}}], None)):
            with self.assertRaises(ValueError):
                m.group_records(records, ids)

    def test_paired_ratio_bootstrap_preserves_constant_effect(self):
        denominator = np.array([1., 10., 100., 1000.])
        w = m.bootstrap_weights(4, 1, 1000)
        value = m.estimate(.1*denominator, w, denominator)
        self.assertAlmostEqual(value["percent_of_reference"], 10)
        self.assertAlmostEqual(value["percent_95_low"], 10)
        self.assertAlmostEqual(value["percent_95_high"], 10)

    def test_constant_correlation_is_explicitly_undefined(self):
        value = m.correlation([1, 1, 1], [1, 2, 3])
        self.assertIsNone(value["pearson"])
        self.assertIsNone(value["spearman"])

    def test_committed_source_bindings_and_full_coverage(self):
        report = json.loads((m.DATA/"pathway_geometry.json").read_text())
        protocol = json.loads(m.PROTOCOL.read_text())
        self.assertEqual(report["protocol_sha256"], m.sha(m.PROTOCOL))
        self.assertEqual(report["analysis_source_sha256"], m.sha(m.__file__))
        self.assertEqual([s["id"] for s in report["source_audit"]], protocol["source_ids"])
        self.assertEqual(len(report["source_audit"]), 17)
        for path, receipt in report["outputs"].items():
            self.assertEqual(m.sha(m.ROOT/path), receipt["sha256"])
            self.assertEqual(len(pd.read_csv(m.ROOT/path)), receipt["rows"])

    def test_saved_factorial_terms_and_missing_horizons(self):
        frame = pd.read_csv(m.DATA/"pathway_geometry_coupling_lineages.csv")
        self.assertFalse(frame.duplicated(["source_id", "lineage_group", "modality", "horizon"]).any())
        saved = frame.dropna(subset=["factorial_mse_interaction"])
        means = pd.read_csv(m.DATA/"pathway_geometry_coupling_summary.csv")
        means = means.drop_duplicates(["source_id", "modality", "horizon"])
        saved = saved.merge(means[["source_id", "modality", "horizon", "native_mse"]],
                            on=["source_id", "modality", "horizon"], suffixes=("", "_reference"))
        error = np.abs(saved.factorial_mse_interaction-saved.additive_quadratic_cross-saved.nonadditive_mse_remainder)
        self.assertTrue((error <= 1e-12+1e-6*saved.native_mse_reference).all())
        self.assertTrue(frame[frame.horizon.isin([2, 4, 5])].output_interaction_mse.isna().all())
        self.assertTrue((saved.output_interaction_mse >= 0).all())
        self.assertTrue(frame[frame.task == "wall"].native_mse.isna().all())
        self.assertEqual(set(frame.task), {"reach", "reach-wall", "pusht", "wall"})
        counts = frame.groupby(["task", "precision"]).lineage_group.nunique()
        for (task, _), n in counts.items():
            self.assertEqual(n, {"reach": 33, "reach-wall": 27, "pusht": 21, "wall": 192}[task])

    def test_geometry_coverage_and_distinct_endpoints(self):
        frame = pd.read_csv(m.DATA/"pathway_geometry_geometry_lineages.csv")
        self.assertFalse(frame.duplicated(["source_id", "lineage_group", "arm"]).any())
        self.assertEqual(set(frame.task), {"reach", "reach-wall", "pusht", "wall", "pointmaze"})
        self.assertEqual(set(frame.arm), set(m.GEOMETRY_ARMS))
        for (task, _, _), n in frame.groupby(["task", "precision", "arm"]).lineage_group.nunique().items():
            self.assertEqual(n, {"reach": 33, "reach-wall": 27, "pusht": 21, "wall": 192, "pointmaze": 200}[task])
        self.assertTrue((frame.omitted_activation_mse >= 0).all())
        self.assertFalse(np.allclose(frame.raw_native_visual_fidelity_mse_h6,
                                     frame.recorded_visual_mse_h6))


if __name__ == "__main__":
    unittest.main()
