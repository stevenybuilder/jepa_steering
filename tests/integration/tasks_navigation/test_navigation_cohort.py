import tempfile
import unittest
from pathlib import Path

from offline_study.evaluation.checkpoint.author_runtime import examples
from offline_study.tasks.navigation.navigation_cohort import COUNTS, verify_inputs
from offline_study.core.protocol import sha256, write_json


class NavigationCohortTests(unittest.TestCase):
    def test_official_clip_and_prefix_counts_not_training_sample_counts(self):
        config = {"data": {"validation": {"num_frames_val": 8}, "custom": {"frameskip": 5}}}
        for task, length in (("wall", 50), ("pointmaze", 100)):
            rows = [{"trajectory_id": f"{task}/{i}", "lineage_group": f"{task}/{i}",
                     "task": task, "length": length} for i in range(COUNTS[task][2])]
            self.assertEqual(len(examples(rows, config)), 2 * COUNTS[task][3])
            self.assertEqual(len(examples(rows[:128], config, fitting=True)), 512)

    def test_cohort_input_hashes_fail_closed_before_dataset_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir()
            write_json(root / "source_manifest.json", [])
            write_json(raw / "video.pth", {"synthetic": True})
            write_json(root / "input_files.json", {"video.pth": sha256(raw / "video.pth")})
            write_json(root / "cohort.json", {
                "source_input_files_sha256": sha256(root / "input_files.json"),
                "source_manifest_sha256": sha256(root / "source_manifest.json")})
            write_json(root / "FROZEN.json", {"cohort_sha256": sha256(root / "cohort.json")})
            write_json(raw / "video.pth", {"synthetic": "changed"})
            with self.assertRaisesRegex(ValueError, "raw input changed"):
                verify_inputs(root / "cohort.json", raw)
