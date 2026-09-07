import unittest

from offline_study.behavioral_analysis import NAMES, TASKS, analyze, exact_discordance, holm


def fixture():
    return {task: {arm: [{"initial_state_vector": [float(i)], "seconds": 2.,
        "result": {"goal_sha256": str(i), "native_success": False,
                   "native_state_distance": 1., "native_reward": 0.}}
        for i in range(96)] for arm in NAMES} for task in TASKS}


class BehavioralAnalysisTests(unittest.TestCase):
    def test_exact_discordance_and_full_holm_family(self):
        self.assertEqual(exact_discordance(6, 0), .03125)
        self.assertEqual(exact_discordance(0, 0), 1.)
        self.assertEqual(exact_discordance(2, 2), 1.)
        self.assertEqual(holm([.03125] + [1.] * 11)[0], .375)
        with self.assertRaises(ValueError):
            holm([.01])

    def test_null_complete_panel_retains_native(self):
        result = analyze(fixture())
        self.assertEqual(len(result["contrasts"]), 12)
        for task in TASKS:
            self.assertEqual(result["tasks"][task]["selected_for_later_confirmation_freeze"], "native")
        self.assertFalse(result["confirmation_outcomes_authorized"])

    def test_positive_candidate_requires_own_control(self):
        panel = fixture()
        for task in TASKS:
            for arm in ("coupling_only", "matched_random_coupling", "rank4_only"):
                for row in panel[task][arm]:
                    row["result"]["native_success"] = True
        result = analyze(panel)
        for task in TASKS:
            self.assertEqual(result["tasks"][task]["eligible_candidates"], ["rank4_only"])
            self.assertEqual(result["tasks"][task]["selected_for_later_confirmation_freeze"], "rank4_only")

    def test_partial_panel_rejected_and_duplicate_scenarios_clustered(self):
        panel = fixture()
        panel["reach"]["combined"].pop()
        with self.assertRaises(ValueError):
            analyze(panel)
        panel = fixture()
        for task in TASKS:
            for arm in NAMES:
                for i, row in enumerate(panel[task][arm]):
                    row["initial_state_vector"] = [float(i // 2)]
                    row["result"]["goal_sha256"] = str(i // 2)
        result = analyze(panel)
        self.assertEqual(result["tasks"]["reach"]["scenario_clusters"], 48)


if __name__ == "__main__":
    unittest.main()
