import copy
import unittest

import numpy as np

from offline_study.decision_diagnostic import (ARMS, TASKS, analyze, candidate_seed,
                                               rank_agreement, score_summary, validate_records)


class DecisionDiagnosticTests(unittest.TestCase):
    def fixture(self):
        schedule = [{"episode": i, "environment_seed": i + 10, "logical_rank": 0, "local_seed": 1} for i in range(4)]
        records = []
        for task in TASKS:
            for row in schedule:
                common = {"initial_sha256": "i", "physics_sha256": "p", "candidate_sha256": "a", "goal_sha256": "g"}
                arms = {arm: {**common, "scores": list(range(300)), "selected": 0,
                              "elementary_steps": 15, "terminal_ee_distance": 1.} for arm in ARMS}
                records.append({**row, **common, "task": task, "protocol_sha256": "x", "arms": arms})
        return records, schedule

    def test_stable_tie_and_invalid_scores(self):
        self.assertEqual(score_summary(np.ones(300))["selected"], 0)
        for scores in (np.zeros(299), np.full(300, np.nan)):
            with self.assertRaises(ValueError):
                score_summary(scores)

    def test_rank_ties(self):
        self.assertAlmostEqual(rank_agreement([1, 1, 2], [2, 2, 1]), -1.)
        self.assertIsNone(rank_agreement([1, 1], [1, 2]))

    def test_seeds_fixed_and_task_specific(self):
        self.assertEqual(candidate_seed("reach", 12), candidate_seed("reach", 12))
        self.assertNotEqual(candidate_seed("reach", 12), candidate_seed("reach-wall", 12))

    def test_incomplete_duplicate_and_unpaired_rejected(self):
        rows, schedule = self.fixture()
        for changed in (rows[:-1], rows + [rows[0]]):
            with self.assertRaises(ValueError):
                validate_records(changed, schedule, "x")
        for field in ("initial_sha256", "physics_sha256", "candidate_sha256", "goal_sha256"):
            changed = copy.deepcopy(rows)
            changed[0]["arms"]["coupling_only"][field] = "different"
            with self.assertRaises(ValueError):
                validate_records(changed, schedule, "x")

    def test_null_pairing_and_unit(self):
        rows, schedule = self.fixture()
        report = analyze(rows, schedule, "x")
        self.assertEqual(report["scenarios"], 8)
        self.assertEqual(report["condition_prefixes"], 40)
        self.assertEqual(len(report["contrasts"]), 8)
        for item in report["contrasts"]:
            self.assertEqual(item["mean_distance_gain_m"], 0)
            self.assertEqual(item["simultaneous_95_interval_m"], [0, 0])
            self.assertEqual(item["n_scenarios"], 4)

    def test_population_interception_restores_native_callback(self):
        import torch
        from offline_study.decision_runtime import capture_population
        class Planner:
            def __init__(self):
                self.local_generator = torch.Generator()
            def cost_function(self, actions, context):
                raise AssertionError("Real forecast should not run during capture")
            def plan(self, context, steps_left):
                actions = torch.randn(6, 300, 20, generator=self.local_generator)
                actions[:, 0] = 0
                self.cost_function(actions, context)
                raise AssertionError("Must stop before adaptive CEM update")
        planner = Planner()
        original = planner.cost_function
        first = capture_population(planner, None, 12)
        self.assertEqual(planner.cost_function, original)
        self.assertTrue(torch.equal(first, capture_population(planner, None, 12)))

    def test_positive_distance_gain_direction(self):
        rows, schedule = self.fixture()
        for row in rows:
            row["arms"]["fixed_rank4"]["terminal_ee_distance"] = .9
        result = analyze(rows, schedule, "x")
        item = result["contrasts"][0]
        self.assertAlmostEqual(item["mean_distance_gain_m"], .1)
        self.assertGreater(item["simultaneous_95_interval_m"][0], 0)


if __name__ == "__main__":
    unittest.main()
