import unittest

from offline_study.development_analysis import analyze_task_report, bootstrap_mean_summary
from offline_study.operator_fit import make_protocol


class DevelopmentAnalysisTests(unittest.TestCase):
    def test_bootstrap_is_deterministic_and_uses_independent_values(self):
        first = bootstrap_mean_summary([-1., 0., 2., 3.], seed=9, replicates=1000)
        second = bootstrap_mean_summary([-1., 0., 2., 3.], seed=9, replicates=1000)
        self.assertEqual(first, second)
        self.assertEqual(first["independent_lineage_groups"], 4)
        self.assertEqual(first["mean"], 1.)
        self.assertEqual(first["fraction_positive"], .5)

    def test_task_analysis_rejects_confirmation_and_summarizes_contrasts(self):
        protocol = make_protocol(
            "mw-reach", "a" * 64, "b" * 64, "c" * 64, .2, .3,
            "2026-09-07T00:00:00+00:00")
        contrast_rows = [{
            **contrast,
            "per_lineage_group": [
                {"task": "mw-reach", "lineage_group": "g1",
                 "candidate_minus_control": {"proprio_mse_h6": 1.}},
                {"task": "mw-reach", "lineage_group": "g2",
                 "candidate_minus_control": {"proprio_mse_h6": -1.}},
            ],
        } for contrast in protocol["primary_contrasts"]]
        report = {
            "status": "real_multigpu_intervention_development_complete",
            "gpu_execution_valid": True,
            "scientific_confirmation": False,
            "category": "vision_action_coupling",
            "rollout_trajectories": 2,
            "independent_lineage_groups": 2,
            "windows": 8,
            "aggregation": {
                "primary_contrasts": contrast_rows,
                "per_arm": {"native": {"per_task_group_weighted": {
                    "mw-reach": {"lineage_groups": 2, "metrics": {"proprio_mse_h6": 4.}}
                }}},
            },
            "mechanism_diagnostics": {"per_lineage_group": [
                {"task": "mw-reach", "lineage_group": "g1",
                 "metrics": {"visual_output_interaction_mse_h6": .1}},
                {"task": "mw-reach", "lineage_group": "g2",
                 "metrics": {"visual_output_interaction_mse_h6": .2}},
            ]},
        }
        result = analyze_task_report(report, protocol, seed=11, replicates=1000)
        self.assertEqual(result["task"], "mw-reach")
        self.assertEqual(result["primary_contrasts"][0]["metrics"]["proprio_mse_h6"]["mean"], 0.)
        report["scientific_confirmation"] = True
        with self.assertRaisesRegex(ValueError, "Invalid development-only"):
            analyze_task_report(report, protocol, seed=11, replicates=1000)


if __name__ == "__main__":
    unittest.main()
