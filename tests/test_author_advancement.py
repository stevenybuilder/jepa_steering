import unittest

from offline_study.author_advancement import qualify, required_controls


class AdvancementTests(unittest.TestCase):
    def test_missing_matched_control_never_qualifies(self):
        protocol = {"category": "operator_rank", "development_analysis_plan": {
            "primary_forecast_endpoint": "proprio_mse_h6", "smallest_useful_effect": .01,
            "mechanism_controls": {"rank1": ["native", "matched_random_rank1"]}}}
        report = {"category": "operator_rank", "contrasts": [{"candidate": "rank1", "control": "native",
            "endpoint": "proprio_mse_h6", "simultaneous_95_difference_interval": [-.2, -.1]}],
            "arms": {name: {"realized_energy_match_fraction": 1.,
                "error_reduction_percent_vs_native": {"proprio_mse_h6": 10.}} for name in ("rank1", "matched_random_rank1")}}
        self.assertEqual(qualify(report, protocol)["decision"], "retain_native")
        report["contrasts"].append({**report["contrasts"][0], "control": "matched_random_rank1"})
        self.assertEqual(qualify(report, protocol)["statistically_eligible_arms"], ["rank1"])
        report["arms"]["rank1"]["realized_energy_match_fraction"] = .8
        self.assertFalse(qualify(report, protocol)["arms"]["rank1"]["exact_equal_energy_mechanism_claim_supported"])

    def test_single_pathway_does_not_borrow_joint_random_control(self):
        controls = required_controls({"category": "vision_action_coupling"})
        self.assertNotIn("visual_only", controls)
        self.assertNotIn("action_condition_only", controls)
