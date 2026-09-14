import copy
import unittest

from offline_study.evaluation.behavioral_development import schedule
from offline_study.evaluation.behavioral_repair_plan import changed_streams


class RepairPlanTests(unittest.TestCase):
    def rows(self):
        return [{**row, "initial_state_vector": [row["episode"]],
                 "result": {"initial_sha256": "i", "goal_sha256": "g", "native_success": 0}}
                for row in schedule()]

    def test_complete_stream_not_modulo_episode_and_no_outcome_selection(self):
        original = self.rows()
        candidates = copy.deepcopy(original)
        for episode in (20, 35):
            candidates[episode]["result"]["goal_sha256"] = "render-jitter"
        ranks, mismatches = changed_streams(original, candidates)
        self.assertEqual(ranks, [1, 2])
        self.assertEqual([r["episode"] for r in mismatches], [20, 35])
        for row in candidates:
            row["result"]["native_success"] = 1
        self.assertEqual(changed_streams(original, candidates), (ranks, mismatches))

    def test_other_input_mismatch_and_duplicates_fail(self):
        original = self.rows()
        for field in ("initial_state_vector", "environment_seed", "logical_rank"):
            candidate = copy.deepcopy(original)
            candidate[20][field] = -1
            with self.assertRaises(ValueError):
                changed_streams(original, candidate)
        candidate = copy.deepcopy(original)
        candidate[20]["result"]["initial_sha256"] = "wrong"
        with self.assertRaises(ValueError):
            changed_streams(original, candidate)
        with self.assertRaises(ValueError):
            changed_streams(original, original + original[:1])
        self.assertEqual(changed_streams(original, original), ([], []))
