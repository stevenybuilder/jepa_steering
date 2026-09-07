import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("summary", Path(__file__).parents[1]/"scripts/geometry_map/summarize_residual_search_physics.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class SummaryTests(unittest.TestCase):
    def rows(self):
        return [{"episode": e, "arm": a, "hand_goal_progress_m": p+e, "native_ever_success": e == 0}
                for e in (0, 4, 7) for a, p in (("unsteered", 1), ("projected-unsteered", 0),
                    ("projected-x-edit", .5), ("projected-x-sham", .2))]

    def test_keeps_both_baselines_and_sham(self):
        r = next(r for r in M.summarize(self.rows(), [0, 4, 7])["rows"] if r["arm"] == "projected-x-edit")
        self.assertAlmostEqual(r["comparisons"]["unsteered"]["hand_goal_progress_m"]["mean_difference"], -.5)
        self.assertAlmostEqual(r["comparisons"]["projected-unsteered"]["hand_goal_progress_m"]["mean_difference"], .5)
        self.assertAlmostEqual(r["comparisons"]["projected-x-sham"]["hand_goal_progress_m"]["mean_difference"], .3)
        self.assertEqual(r["metrics"]["native_final_success"]["status"], "not_captured")
        self.assertEqual(r["metrics"]["native_ever_success"]["count"], 1)

    def test_rejects_incomplete_or_duplicate(self):
        for rows in (self.rows()[:-1], self.rows()+self.rows()[:1]):
            with self.assertRaises(ValueError): M.summarize(rows, [0, 4, 7])

    def test_refuses_held_data(self):
        with self.assertRaises(ValueError): M.summarize([], [12])
        with self.assertRaises(ValueError): M.summarize([], [12], seen_development_repeat=True)
        with self.assertRaises(ValueError): M.summarize([], [8])

    def test_explicit_seen_group_keeps_real_episode_ids(self):
        rows = [{"episode": e, "arm": "unsteered", "hand_goal_progress_m": .1} for e in (8, 9, 10, 11)]
        result = M.summarize(rows, [8, 9, 10, 11], seen_development_repeat=True)
        self.assertEqual(result["episodes"], [8, 9, 10, 11])
        self.assertTrue(result["seen_development_repeat"])

    def test_forecasts_episode_weighted_not_replan_weighted(self):
        rows = self.rows()
        for row in rows:
            count, error = (1, 1.) if row["episode"] == 0 else (3, 4.)
            row["replans"] = [{"prediction_errors": [{"imagined_horizon": h, "visual_mse": error,
                                                       "proprio_mse": error*2} for h in (1, 2, 3)]} for _ in range(count)]
        values = M.summarize(rows, [0, 4, 7])["rows"][0]["executed_prefix_forecast_errors"]
        self.assertEqual(values[0]["visual_mse"], 3.)
        self.assertEqual(values[0]["proprio_mse"], 6.)


if __name__ == "__main__": unittest.main()
