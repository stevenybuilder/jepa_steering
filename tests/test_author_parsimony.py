import unittest
from unittest.mock import patch

from offline_study.author_parsimony import audit, interval_between


class ParsimonyTests(unittest.TestCase):
    def report(self):
        return {"precision": "bfloat16", "contrasts": [
            {"candidate": "rank1", "control": "rank4", "endpoint": "proprio_mse_h6",
             "simultaneous_95_difference_interval": [.006, .013]},
            {"candidate": "rank4", "control": "rank8", "endpoint": "proprio_mse_h6",
             "simultaneous_95_difference_interval": [-.003, .001]}]}

    def gates(self):
        return {"statistically_eligible_arms": ["rank1", "rank4", "rank8"],
                "arms": {name: {"all_delivered_energies_within_frozen_fp32_tolerance": True,
                                  "native_error_reduction_percent": value}
                         for name, value in (("rank1", 1.9), ("rank4", 2.9), ("rank8", 2.8))}}

    def test_explicit_rank_rule_does_not_use_non_significance_as_equivalence(self):
        protocol = {"category": "operator_rank", "development_analysis_plan": {"equivalence_margin": .009}}
        with patch("offline_study.author_parsimony.qualify", return_value=self.gates()):
            result = audit(self.report(), protocol)
        self.assertEqual(result["selected_arm"], "rank4")
        self.assertFalse(result["equivalence_to_observed_best"]["rank1"]["equivalence_established"])
        self.assertFalse(result["planning_or_confirmation_authorized"])

    def test_reverse_contrast_and_missing_comparison(self):
        self.assertEqual(interval_between(self.report(), "rank8", "rank4"), [-.001, .003])
        self.assertIsNone(interval_between(self.report(), "rank1", "rank8"))

    def test_precision_and_unregistered_tie_breaker_cannot_select(self):
        report = self.report()
        with self.assertRaises(ValueError):
            audit({**report, "precision": "float32"}, {})
        with patch("offline_study.author_parsimony.qualify", return_value=self.gates()):
            result = audit(report, {"category": "distribution_layer", "development_analysis_plan": {"equivalence_margin": .009}})
        self.assertIsNone(result["selected_arm"])
        self.assertEqual(result["decision"], "multiple_eligible_choices_no_unique_frozen_tie_rule")
