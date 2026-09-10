import copy
import unittest

from offline_study.combined_advancement import decide_combination
from offline_study.combined_operator import PAIRS, RECIPE_ARMS


def fixture():
    rows = []
    for a, b in PAIRS:
        for endpoint in ("visual_mse_h6", "proprio_mse_h6"):
            interval = [-1.2, -.8] if b == "combined_rank1" else [-3., -2.]
            rows.append({"candidate": a, "control": b, "endpoint": endpoint,
                "simultaneous_95_difference_interval": interval,
                "minimum_useful_error_reduction": 1., "beats_control_by_frozen_minimum": interval[1] < -1.})
    return {"precision": "bfloat16", "contrasts": rows,
            "arms": {r[0]: {"realized_energy_match_fraction": 1.} for r in RECIPE_ARMS}}


class CombinedAdvancementTests(unittest.TestCase):
    def test_retains_rank4_without_claiming_minimum_rank_superiority(self):
        result = decide_combination(fixture(), 1.)
        self.assertEqual(result["selected_nonrouted_recipe"], "combined")
        self.assertFalse(result["rank1_equivalence_established"])
        self.assertFalse(result["rank4_beats_rank1_by_frozen_minimum"])
        self.assertFalse(result["confirmation_or_planning_launch_authorized"])

    def test_rank1_simplification_requires_positive_equivalence_evidence(self):
        report = fixture()
        for row in report["contrasts"]:
            if row["control"] == "combined_rank1":
                row["simultaneous_95_difference_interval"] = [-.5, .5]
        self.assertEqual(decide_combination(report, 1.)["selected_nonrouted_recipe"], "combined_rank1")

    def test_rejects_incomplete_family_changed_margin_energy_or_precision(self):
        original = fixture()
        variants = []
        report = copy.deepcopy(original)
        report["contrasts"].pop()
        variants.append(report)
        report = copy.deepcopy(original)
        report["precision"] = "float32"
        variants.append(report)
        report = copy.deepcopy(original)
        report["arms"]["combined"]["realized_energy_match_fraction"] = .9
        variants.append(report)
        for report in variants:
            with self.assertRaises(ValueError):
                decide_combination(report, 1.)
        with self.assertRaises(ValueError):
            decide_combination(original, 2.)

    def test_no_selection_if_combination_fails_its_control(self):
        report = fixture()
        for row in report["contrasts"]:
            if row["candidate"] == "combined" and row["control"] == "matched_random_combined":
                row.update(simultaneous_95_difference_interval=[-1., 1.], beats_control_by_frozen_minimum=False)
        self.assertIsNone(decide_combination(report, 1.)["selected_nonrouted_recipe"])
