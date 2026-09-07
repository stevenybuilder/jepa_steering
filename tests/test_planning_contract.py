import unittest
from pathlib import Path

from offline_study.planning_contract import prepare, seed_schedule


class PlanningContractTests(unittest.TestCase):
    def test_seed_schedule_has_exact_coverage_and_no_padding(self):
        rows = seed_schedule()
        self.assertEqual([row["episode"] for row in rows], list(range(96)))
        self.assertEqual(len({row["environment_seed"] for row in rows}), 96)
        for rank in range(8):
            subset = [row for row in rows if row["logical_rank"] == rank]
            self.assertEqual(len(subset), 12)
            self.assertEqual(subset[0]["local_seed"], 1 + 6000 * rank)
        uneven = seed_schedule(episodes=97)
        self.assertEqual(len(uneven), 97)

    def test_paper_scale_and_distinct_planning_settings(self):
        vendor = Path(__file__).resolve().parents[1] / "vendor/jepa-wms"
        if not vendor.is_dir():
            self.skipTest("Pinned upstream source not installed")
        for task in ("pusht", "reach", "reach-wall"):
            contract = prepare(vendor, task)
            self.assertEqual(contract["planned_replication_episodes_per_condition"], 96)
            self.assertEqual(contract["planning_context"], 2)
            self.assertFalse(contract["fresh_family_confirmation"])
            self.assertFalse(contract["matches_three_training_seeds_and_last_ten_epochs"])
            self.assertEqual(contract["config"]["planner"]["planning_objective"]["alpha"], .1)
            self.assertEqual(contract["source_config_episodes"], 96 if task == "pusht" else 48)

    def test_invalid_seed_configuration_fails(self):
        with self.assertRaises(ValueError):
            seed_schedule(base_seed=0)
        with self.assertRaises(ValueError):
            seed_schedule(episodes=4, logical_ranks=8)


if __name__ == "__main__":
    unittest.main()
