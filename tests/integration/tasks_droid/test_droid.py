import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from offline_study.tasks.droid.droid_assets import validate_manifest, verify_download
from offline_study.tasks.droid.droid_contract import action_metrics, checkpoint_score, late_checkpoints, prepare


ROOT = Path(__file__).resolve().parents[3]


class DroidTests(unittest.TestCase):
    def manifest(self):
        return json.loads((ROOT / "configs/droid_assets.json").read_text())

    def test_scope_exact_recordings_and_checkpoint_no_robocasa(self):
        spec = self.manifest()
        self.assertEqual(validate_manifest(spec), 4852175564)
        for change in ("duplicate", "traversal", "robocasa", "revision", "checksum"):
            bad = copy.deepcopy(spec)
            if change == "duplicate":
                bad["assets"][1] = bad["assets"][0]
            elif change == "traversal":
                bad["assets"][0]["filename"] = "../escape"
            elif change == "robocasa":
                bad["assets"][0]["filename"] = "robocasa/episode.h5"
            elif change == "revision":
                bad["dataset_revision"] = "main"
            else:
                bad["assets"][0]["checksum"] = None
            with self.subTest(change=change), self.assertRaises((ValueError, TypeError)):
                validate_manifest(bad)

    def test_small_git_blob_and_lfs_content_hash_verification(self):
        payload = b"recording companion fixture"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trajectory.hdf5"
            path.write_bytes(payload)
            for kind, checksum in (("sha256", hashlib.sha256(payload).hexdigest()),
                    ("git_blob_sha1", hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest())):
                entry = {"filename": path.name, "size": len(payload), "checksum_type": kind,
                         "checksum": checksum}
                result = verify_download(path, entry)
                self.assertEqual(result["verified_content_sha256"], hashlib.sha256(payload).hexdigest())
                with self.assertRaisesRegex(ValueError, "checksum"):
                    verify_download(path, {**entry, "checksum": "0" * len(checksum)})

    def test_score_uses_displacement_sum_not_stepwise_absolute_mean(self):
        planned = torch.zeros(2, 3, 7, dtype=torch.float64)
        recorded = torch.zeros_like(planned)
        planned[0, 0, 0], planned[0, 1, 0] = .02, -.02
        planned[0, :, 3:] = 50.  # Orientation/gripper are not the primary metric.
        planned[1, 0, 0] = .11
        result = action_metrics(planned, recorded)
        torch.testing.assert_close(result["action_error_xyz"], torch.tensor([0., .11], dtype=torch.float64))
        self.assertAlmostEqual(float(checkpoint_score(result["action_error_xyz"], expected_episodes=2)), 36.)
        self.assertEqual(float(result["action_error_xyz"][0]), 0.)
        single = action_metrics(planned[0], recorded[0])
        self.assertEqual(float(single["action_error_xyz"]), 0.)
        planned[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            action_metrics(planned, recorded)
        with self.assertRaisesRegex(ValueError, "three-step"):
            action_metrics(torch.zeros(6, 7), torch.zeros(6, 7))

    def test_contract_preserves_native_budget_and_exposes_paper_code_gaps(self):
        vendor = ROOT / "vendor/jepa-wms"
        if not vendor.exists():
            self.skipTest("Pinned vendor checkout is not available")
        contract = prepare(vendor, self.manifest())
        self.assertEqual(len(contract["episodes"]), 64)
        self.assertEqual([row["local_seed"] for row in contract["episodes"][::8]],
                         [1 + rank * 3000 for rank in range(8)])
        self.assertEqual(len(contract["released_config_recordings"]), 15)
        self.assertEqual(contract["unselected_released_recordings"],
                         ["franka_custom/data/pick/v0/run_0001/episode.h5"])
        self.assertFalse(contract["paper_population_exactly_matched"])
        self.assertEqual(contract["training"]["required_late_epoch_metadata"], list(range(217, 314, 6)) + [315])
        self.assertEqual(len(late_checkpoints()), 18)
        self.assertEqual(contract["training"]["global_batch"], 256)
        self.assertEqual(contract["training"]["updates_per_seed"], 94500)
        self.assertEqual(contract["config"]["planner"]["max_norm_dims"], [[0, 1, 2, 3, 4, 5], [6]])
        self.assertEqual(contract["config"]["planner"]["planning_objective"]["alpha"], 0)
        self.assertFalse(contract["config"]["planner"]["decode_each_iteration"])
        self.assertEqual(contract["model_runs"], 0)
        self.assertFalse(contract["fresh_confirmation"])

    def test_checkpoint_score_rejects_partial_and_nonfinite_episodes(self):
        self.assertEqual(float(checkpoint_score(torch.zeros(64))), 80.)
        self.assertEqual(float(checkpoint_score(torch.full((64,), .2))), 0.)
        for values in (torch.zeros(63), torch.full((64,), float("nan")), -torch.ones(64)):
            with self.assertRaises(ValueError):
                checkpoint_score(values)


if __name__ == "__main__":
    unittest.main()
