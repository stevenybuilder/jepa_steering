import tempfile
import unittest
from pathlib import Path

from offline_study.planning.planning_contract import seed_schedule
from offline_study.core.protocol import sha256, write_json
from offline_study.tasks.droid.verify_droid_replication import verify


class DroidReplicationVerificationTests(unittest.TestCase):
    def fixture(self, root):
        rows = seed_schedule(1, 64, 8, 3)
        write_json(root / "protocol.json", {"assigned_episodes": rows, "fresh_confirmation": False,
                                           "native_baseline_only": True})
        files = {}
        for row in rows:
            frames = [row["episode"] * 8 + i * 8 for i in range(5)]
            result = {"unroll_calls": [[3, 300], [3, 1]] * 15,
                "dummy_success_intentionally_omitted": True,
                "planned_actions": [[0.] * 7] * 3, "recorded_actions": [[0.] * 7] * 3,
                "metrics": {"action_error_xyz": 0., "action_error_orientation": 0., "action_error_gripper": 0.},
                "dataset_sample": {"path": f"recording-{row['episode'] % 15}",
                    "raw_frame_indices": frames, "goal_segment_offset": 1,
                    "goal_segment_raw_frame_indices": frames[1:]}}
            path = root / f"episode-{row['episode']:03d}.json"
            write_json(path, {**row, "arm": "native", "result": result})
            files[path.name] = sha256(path)
        report = {"protocol_sha256": sha256(root / "protocol.json"),
            "status": "released_droid_native_replication_shard_complete", "episodes": 64,
            "robot_executions": 0, "episode_files_sha256": files,
            "official_checkpoint_score_if_complete": 80.}
        self.update_report(root, report)
        return report

    def update_report(self, root, report):
        write_json(root / "report.json", report)
        write_json(root / "DONE.json", {"report_sha256": sha256(root / "report.json")})

    def test_full_reconstruction_and_reused_families(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            result = verify(root)
            self.assertEqual(result["sampled_recording_families"], 15)
            self.assertEqual(result["unique_recording_and_goal_segments"], 64)
            self.assertEqual(result["official_checkpoint_score"], 80.)
            self.assertFalse(result["intervention_comparison_complete"])

    def test_fail_closed_on_missing_episodes_or_bad_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = self.fixture(root)
            report["official_checkpoint_score_if_complete"] = 100.
            self.update_report(root, report)
            with self.assertRaises(ValueError):
                verify(root)
            report["official_checkpoint_score_if_complete"] = 80.
            report["episode_files_sha256"].pop("episode-003.json")
            self.update_report(root, report)
            with self.assertRaises(ValueError):
                verify(root)
