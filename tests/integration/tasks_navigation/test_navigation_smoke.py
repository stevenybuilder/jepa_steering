import unittest

from offline_study.tasks.navigation.navigation_smoke import validate_complete, comparison_record


class NavigationSmokeTests(unittest.TestCase):
    def test_full_budget_required(self):
        result = {"elementary_steps": 30, "published_candidate_count": 300,
                  "planning_calls": [{"iterations": 30, "returned_model_actions": 6}]}
        schedule = [(6, 300), (6, 1)] * 30
        validate_complete(result, schedule)
        for wrong in (schedule[:-1], [(6, 32), (6, 1)] * 30, schedule[::-1]):
            with self.assertRaises(ValueError):
                validate_complete(result, wrong)
        result["elementary_steps"] = 29
        with self.assertRaises(ValueError):
            validate_complete(result, schedule)

    def test_pairing_includes_actions_and_all_outcomes_but_not_timings(self):
        result = {"initial_sha256": "a", "goal_sha256": "b", "native_success": False,
                  "native_state_distance": 1., "native_reward": 0., "elementary_steps": 30,
                  "observed_frames": 31, "planning_calls": [{"seconds": 1.}]}
        first = comparison_record(result, ["action1"])
        result["planning_calls"][0]["seconds"] = 2.
        self.assertEqual(first, comparison_record(result, ["action1"]))
        self.assertNotEqual(first, comparison_record(result, ["action2"]))
