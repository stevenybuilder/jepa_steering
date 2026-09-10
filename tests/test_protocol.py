import unittest

from offline_study.protocol import shard_for, study_split, summarize_metrics, window_starts


class ProtocolTests(unittest.TestCase):
    def test_full_scale_split_and_order_independence(self):
        ids = [f"trajectory-{i}" for i in range(18500)]
        split = study_split(ids)
        self.assertEqual(sum(s != "holdout" for s in split.values()), 16650)
        self.assertEqual(sum(s == "holdout" for s in split.values()), 1850)
        self.assertEqual(split, study_split(ids[::-1]))

    def test_duplicate_trajectory_rejected(self):
        with self.assertRaises(ValueError):
            study_split(["a", "a"])

    def test_windows_include_final_future_without_padding(self):
        self.assertEqual(window_starts(30), [])
        self.assertEqual(window_starts(31), [0])
        starts = window_starts(99)
        self.assertEqual(len(starts), 4)
        self.assertEqual(starts[-1] + 6 * 5, 98)

    def test_eight_shards_cover_each_trajectory_once(self):
        ids = [str(i) for i in range(31100)]
        shards = [{i for i in ids if shard_for(i, 8) == s} for s in range(8)]
        self.assertEqual(set.union(*shards), set(ids))
        self.assertEqual(sum(map(len, shards)), len(ids))

    def test_equal_trajectory_weighting(self):
        rows = [{"task": "reach", "trajectory_id": "a", "lineage_group": "g1",
                 "start": i, "metrics": {"error": 0.}} for i in range(4)]
        rows.append({"task": "reach", "trajectory_id": "b", "lineage_group": "g1",
                     "start": 0, "metrics": {"error": 10.}})
        rows.append({"task": "reach", "trajectory_id": "c", "lineage_group": "g2",
                     "start": 0, "metrics": {"error": 20.}})
        summary = summarize_metrics(rows)
        self.assertEqual(summary["per_task"]["reach"]["metrics"]["error"], 10.)
        self.assertEqual(summary["per_task"]["reach"]["trajectories"], 3)
        self.assertEqual(summary["per_task_group_weighted"]["reach"]["metrics"]["error"], 12.5)
        self.assertEqual(summary["per_task_group_weighted"]["reach"]["lineage_groups"], 2)
        with self.assertRaises(ValueError):
            summarize_metrics(rows + [rows[0]])


if __name__ == "__main__":
    unittest.main()
