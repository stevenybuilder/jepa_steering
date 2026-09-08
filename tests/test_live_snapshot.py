import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/vast"
sys.path.insert(0, str(SCRIPTS))
from snapshot_live_results import select_files


class LiveSnapshotTests(unittest.TestCase):
    def test_missing_worker_roots_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "Missing explicit"):
                select_files(folder, "metaworld")

    def test_selects_only_complete_episode_and_bound_traces(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "fixed-response-code-20260908-v4").mkdir()
            for gpu in range(8):
                p = root / f"fixed-response-worker-checks-20260908-v1/gpu-{gpu}"
                p.mkdir(parents=True); (p / "DONE.json").write_text("{}")
            p = root / "fixed-response-behavior-20260908-v1/reach/native/shard-gpu0"
            p.mkdir(parents=True)
            (p / "protocol.json").write_text("{}")
            (p / "progress.json").write_text("{}")
            (p / "episode-000.json").write_text('{"episode": 0}')
            (p / "episode-001.json").write_text('{"episode":')
            calls = p / "calls-000"; calls.mkdir()
            for name in ("unroll_calls.json", "action_trace.json", "progress.json"):
                (calls / name).write_text("[]")
            files = select_files(root, "metaworld")
            self.assertIn(str((p / "episode-000.json").relative_to(root)), files)
            self.assertIn(str((calls / "action_trace.json").relative_to(root)), files)
            self.assertFalse(any("progress" in f or "episode-001" in f for f in files))
            (calls / "action_trace.json").unlink()
            with self.assertRaisesRegex(ValueError, "missing snapshot"):
                select_files(root, "metaworld")

    def test_incomplete_engineering_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "fixed-response-code-20260908-v4").mkdir()
            with self.assertRaisesRegex(ValueError, "incomplete"):
                select_files(root, "metaworld")

    def test_pusht_snapshot_retains_source_receipts_not_raw_videos_or_inflight_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("pusht-planning-code-20260908-v1", "pusht-planning-evidence-20260908-v1"):
                (root / name).mkdir()
            base = root / "pusht-planning-native-20260908-v1"
            for name in ("engineering", "simulator-check"):
                p = base / name
                p.mkdir(parents=True); (p / "DONE.json").write_text("{}")
            (base / "freeze").mkdir(); (base / "freeze/FROZEN.json").write_text("{}")
            (base / "LAUNCH.json").write_text("{}")
            (root / "run_pusht_native_queue_v1.py").write_text("# queue")
            assets = root / "pusht-planning-assets-20260908-v3"
            (assets / "source/data/pusht_noise/train").mkdir(parents=True)
            for name in ("receiving_report.json", "RECEIVING_DONE.json"):
                (assets / name).write_text("{}")
            for name in ("protocol.json", "files.json", "report.json", "DONE.json"):
                (assets / "source" / name).write_text("{}")
            (assets / "source/data/pusht_noise/train/unused.pth").write_bytes(b"not a receipt")
            shard = base / "native/shard-0"; shard.mkdir(parents=True)
            for name in ("protocol", "report", "DONE"):
                (shard / (name + ".json")).write_text("{}")
            (shard / "episode-000.json").write_text('{"episode":0}')
            (shard / "episode-001.json").write_text('{"episode":')
            selected = select_files(root, "pusht")
            self.assertIn(str((shard / "episode-000.json").relative_to(root)), selected)
            self.assertIn(str((shard / "DONE.json").relative_to(root)), selected)
            self.assertFalse(any("episode-001" in p or "unused.pth" in p for p in selected))


if __name__ == "__main__":
    unittest.main()
