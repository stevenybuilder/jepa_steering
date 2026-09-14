import unittest

import torch

from offline_study.evaluation.checkpoint.author_analyze import analyze
from offline_study.evaluation.checkpoint.author_evaluate import check_coverage, planned_diagnostics


class AuthorEvaluationTests(unittest.TestCase):
    def test_coupling_diagnostics_distinguish_output_from_squared_error_interaction(self):
        names = ["native", "visual_only", "action_condition_only", "joint"]
        pred = {key: torch.tensor([0., 1., 2., 3.]).reshape(1, 4, 1).repeat(7, 1, 1)
                for key in ("visual", "proprio")}
        target = {key: torch.zeros(4, 7, 1) for key in pred}
        rows = planned_diagnostics(pred, target, names, "vision_action_coupling", 1)
        self.assertEqual(rows[0]["visual_output_interaction_mse_h6"], 0.)
        self.assertEqual(rows[0]["visual_factorial_mse_interaction_h6"], 4.)
        self.assertEqual(rows[0]["visual_nonadditive_mse_remainder_h6"], 0.)

    def test_geometry_preserves_reconstruction_and_adds_native_fidelity(self):
        names = ["native", "cubic"]
        pred = {key: torch.tensor([2., 3.]).reshape(1, 2, 1).repeat(7, 1, 1)
                for key in ("visual", "proprio")}
        target = {key: torch.zeros(2, 7, 1) for key in pred}
        rows = planned_diagnostics(pred, target, names, "action_response_geometry", 1,
                                   [{"omitted_activation_error": .001}])
        self.assertEqual(rows[0]["omitted_activation_error"], .001)
        self.assertEqual(rows[0]["cubic_visual_native_fidelity_mse_h6"], 1.)
        self.assertEqual(rows[0]["native_visual_native_fidelity_mse_h6"], 0.)

    def test_missing_duplicate_or_extra_arm_measurements_fail(self):
        meta = [{"trajectory_id": "a", "start": 0}]
        measurements = [{**meta[0], "arm": name} for name in ("native", "edit")]
        check_coverage(measurements, meta, ["native", "edit"])
        for invalid in (measurements[:1], measurements + measurements[:1],
                        measurements + [{**meta[0], "arm": "unregistered"}]):
            with self.assertRaises(ValueError):
                check_coverage(invalid, meta, ["native", "edit"])

    def test_analysis_preserves_units_sign_and_joint_endpoint_family(self):
        rows = []
        for family in range(5):
            for start in range(3):
                for name in ("native", "zero_dose", "edit"):
                    error = 1 + family * .1 + start * .01
                    value = error * (.9 if name == "edit" else 1.)
                    rows.append({"trajectory_id": str(family), "lineage_group": str(family),
                        "task": "task", "start": start, "arm": name,
                        "metrics": {"visual_mse_h6": value, "proprio_mse_h6": value / 10},
                        "requested_l2": float(name == "edit"), "realized_l2": float(name == "edit"),
                        "realized_energy_within_fp32_tolerance": True})
        protocol = {"arms": [{"name": name} for name in ("native", "zero_dose", "edit")],
                    "primary_contrasts": [{"name": "edit_vs_native", "candidate": "edit", "control": "native"}]}
        receipt = {"fit_native_visual_mse_h6": 1., "fit_native_proprio_mse_h6": .1}
        result = analyze(rows, protocol, receipt, "task")
        self.assertEqual(result["independent_lineage_groups"], 5)
        self.assertEqual(len(result["contrasts"]), 2)
        for contrast in result["contrasts"]:
            self.assertAlmostEqual(contrast["error_reduction_percent_of_native"], 10)
            self.assertTrue(contrast["beats_control"])
        self.assertEqual(result["arms"]["zero_dose"]["error_reduction_percent_vs_native"]["visual_mse_h6"], 0)
        self.assertFalse(result["fresh_confirmation"])


if __name__ == "__main__":
    unittest.main()
