import copy
import unittest

from offline_study.evaluation.behavioral_development import assigned_rows, schedule
from offline_study.tasks.navigation.navigation_replication import validate_records


class NavigationReplicationTests(unittest.TestCase):
    def test_logical_streams_are_disjoint_and_complete(self):
        self.assertEqual(assigned_rows(schedule(), [0, 1, 2, 3]) +
                         assigned_rows(schedule(), [4, 5, 6, 7]), schedule())
        with self.assertRaises(ValueError):
            assigned_rows(schedule(), [0, 0])

    def test_full_actions_and_candidate_schedule(self):
        expected = schedule()[:1]
        row = {**expected[0], "result": {"elementary_steps": 30,
            "published_candidate_count": 300, "native_success": False,
            "planning_calls": [{"iterations": 30, "returned_model_actions": 6}]},
            "unroll_calls": [(6, 300), (6, 1)] * 30,
            "planned_actions": [[[0., 0.] for _ in range(6)]]}
        validate_records([row], expected)
        for key, value in (("planned_actions", []), ("unroll_calls", [(6, 1)]),
                           ("local_seed", 1)):
            wrong = copy.deepcopy(row)
            wrong[key] = value
            with self.assertRaises(ValueError):
                validate_records([wrong], expected)
        with self.assertRaises(ValueError):
            validate_records([row, row], expected)
