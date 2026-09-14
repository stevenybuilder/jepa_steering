import unittest
import json

import numpy as np

from analysis.mechanism.decision_geometry import ROOT, PLAN, compare_scores, score_entropy, digest


class DecisionGeometryTests(unittest.TestCase):
    def test_common_cost_shift_preserves_order_and_entropy(self):
        x = np.arange(300, dtype=float)
        result = compare_scores(x, x + 2)
        self.assertEqual(result["changed"], 0)
        self.assertEqual(result["no_flip_certified"], 1)
        self.assertEqual(result["centered_delta_energy_fraction"], 0)
        self.assertAlmostEqual(result["score_entropy_delta_nats"], 0)

    def test_small_range_certifies_no_flip(self):
        result = compare_scores([0, 2, 4], [.1, 1.9, 3.9])
        self.assertEqual(result["no_flip_certified"], 1)
        self.assertEqual(result["changed"], 0)

    def test_crossing_detects_change(self):
        result = compare_scores([0, .01, 1], [.02, 0, 1])
        self.assertEqual(result["strict_margin_crossings"], 1)
        self.assertEqual(result["changed"], 1)
        self.assertEqual(result["no_flip_certified"], 0)

    def test_tie_break_is_lowest_candidate(self):
        result = compare_scores([1, 0, 2], [0, 0, 2])
        self.assertEqual(result["changed"], 1)
        self.assertEqual(result["strict_margin_crossings"], 0)

    def test_undefined_zero_scale(self):
        result = compare_scores([1, 1], [1, 1])
        self.assertIsNone(result["range_to_margin"])
        self.assertIsNone(result["score_entropy_delta_nats"])
        self.assertIsNone(result["centered_delta_energy_fraction"])
        self.assertIsNone(score_entropy(np.ones(2), 0))

    def test_reject_invalid_vectors(self):
        for a, b in (([1], [2]), ([1, 2], [3]), ([0, np.nan], [1, 2])):
            with self.assertRaises(ValueError):
                compare_scores(a, b)

    def test_public_receipts_and_counts(self):
        data = json.loads((ROOT / "paper/data/decision_geometry.json").read_text())
        self.assertEqual(data["plan_sha256"], digest(PLAN))
        from offline_study._paths import frozen_analysis_path
        self.assertEqual(data["analysis_source_sha256"], digest(
            frozen_analysis_path("decision_geometry.py", data["analysis_source_sha256"])))
        self.assertEqual(len(data["record_sha256"]), 192)
        self.assertEqual(data["paired_arm_comparisons"], 768)
        for group in data["summaries"]:
            self.assertLessEqual(group["changed"] + group["no_flip_certified"], 96)


if __name__ == "__main__":
    unittest.main()
