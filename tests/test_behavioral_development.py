import copy
import unittest

from offline_study.behavioral_development import ARMS, CONTRASTS, assigned_rows, schedule, validate_coverage
from offline_study.planning_contract import seed_schedule


class BehavioralDevelopmentTests(unittest.TestCase):
    def test_development_and_reserved_confirmation_are_disjoint(self):
        rows = schedule()
        self.assertEqual(len(rows), 96)
        self.assertFalse({r["environment_seed"] for r in rows} & {r["environment_seed"] for r in seed_schedule(1)})
        self.assertEqual([len(assigned_rows(rows, [rank])) for rank in range(8)], [12] * 8)
        self.assertEqual(assigned_rows(rows, [0, 1, 2, 3]) + assigned_rows(rows, [4, 5, 6, 7]), rows)

    def test_fail_closed_duplicate_rank_and_tampered_identity(self):
        rows = schedule()
        for ranks in ([], [0, 0], [8], [-1]):
            with self.assertRaises(ValueError):
                assigned_rows(rows, ranks)
        bad = copy.deepcopy(rows)
        bad[0]["environment_seed"] += 1
        with self.assertRaises(ValueError):
            assigned_rows(bad, [0])

    def test_exact_coverage_and_complete_episode(self):
        expected = assigned_rows(schedule(), [0])
        records = [{**row, "result": {"elementary_steps": 100, "native_success": False}} for row in expected]
        validate_coverage(records, expected)
        for bad in (records[:-1], records + records[:1], list(reversed(records))):
            with self.assertRaises(ValueError):
                validate_coverage(bad, expected)
        records[0]["result"]["elementary_steps"] = 99
        with self.assertRaises(ValueError):
            validate_coverage(records, expected)

    def test_finite_preexisting_panel_and_contrasts(self):
        names = {arm[0] for arm in ARMS}
        self.assertEqual(len(names), 7)
        self.assertEqual(len(CONTRASTS), 6)
        self.assertTrue(all(a in names and b in names for a, b in CONTRASTS))


if __name__ == "__main__":
    unittest.main()
