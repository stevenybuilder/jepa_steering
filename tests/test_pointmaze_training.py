import copy
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, DistributedSampler

from offline_study.droid_native import assert_same
from offline_study.navigation_input_check import rng_state, restore_rng
from offline_study.pointmaze_training_history import (
    ValidationFrames, restore_checkpoint, training_config, validation_batches, UPDATES, SAMPLER_POLICY)
from offline_study.pointmaze_training_inputs import required_files, verify_inputs, ARCHIVE_SHA, REVISION
from offline_study.protocol import sha256, write_json
from offline_study.training_history import checkpoint_payload
from offline_study.training_pilot import VirtualRankBatchSampler
from offline_study.vendor import use_vendor


class EmptyScaler:
    def state_dict(self):
        return {}
    def load_state_dict(self, state):
        if state != {}:
            raise ValueError("Unexpected scaler")


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.predictor = torch.nn.Linear(2, 2)
        self.action_encoder = None
        self.proprio_encoder = torch.nn.Linear(2, 2)
        self.optimizer = torch.optim.AdamW(self.parameters())
        self.scaler = EmptyScaler()


class PointMazeTrainingTests(unittest.TestCase):
    def test_actual_native_loader_schedule_includes_partial_validation(self):
        batches = validation_batches()
        self.assertEqual(len(batches), 191)
        self.assertTrue(all(sum(map(len, batch)) == 64 for batch in batches[:-1]))
        self.assertEqual([len(ids) for ids in batches[-1]], [3]*16)
        for rank in range(16):
            native = DataLoader(range(12200), batch_size=4, drop_last=False,
                sampler=DistributedSampler(range(12200), num_replicas=16, rank=rank, shuffle=False))
            self.assertEqual([x.tolist() for x in native], [x[rank] for x in batches])
        self.assertEqual(sum(len(ids) for batch in batches for ids in batch), 12208)
        self.assertEqual(len(VirtualRankBatchSampler(145800)), UPDATES)
        self.assertEqual(UPDATES, 1139)

    def test_pointmaze_config_retains_native_training_budget(self):
        vendor = Path(__file__).resolve().parents[1] / "vendor/jepa-wms"
        cfg = training_config(vendor)
        self.assertEqual(cfg["data"]["datasets"], ["PointMaze"])
        self.assertEqual(cfg["optimization"]["transition_model"]["num_epochs"], 50)
        self.assertFalse(cfg["data"]["validation"]["val_dataset_drop_last"])
        self.assertEqual(cfg["evals"]["eval_episodes"], 96)

    def test_resume_rejects_wall_cursors_and_changed_seed(self):
        model = TinyModel()
        cfg = {"model": {"action_encoder": {}, "proprio_encoder": {}}}
        schedule, wd = SimpleNamespace(_step=1139), SimpleNamespace(_step=1139)
        rngs = [torch.Generator().manual_seed(i).get_state() for i in range(16)]
        loader = torch.Generator().manual_seed(6)
        binding = {"task": "pointmaze", "seed": 234, "sampler_policy": SAMPLER_POLICY}
        data = copy.deepcopy(checkpoint_payload(model, cfg, 1, rngs, rngs, schedule, wd, loader, binding, 5))
        cpu, cuda, events = restore_checkpoint(data, model, cfg, schedule, wd, loader, binding)
        assert_same(cpu, rngs); assert_same(cuda, rngs)
        self.assertEqual(events, 5)
        for key, value in (("scheduler_step", 418), ("wd_step", 418), ("validation_events", 2),
                           ("epoch_boundary_only", False)):
            bad = copy.deepcopy(data)
            bad["study_resume"][key] = value
            with self.assertRaises(ValueError):
                restore_checkpoint(bad, model, cfg, schedule, wd, loader, binding)
        with self.assertRaises(ValueError):
            restore_checkpoint(data, model, cfg, schedule, wd, loader, {**binding, "seed": 235})
        historical = copy.deepcopy(data)
        del historical['study_resume']['binding']['sampler_policy']
        with self.assertRaisesRegex(ValueError, 'historical extra-shuffled'):
            restore_checkpoint(historical, model, cfg, schedule, wd, loader, binding)
        changed = copy.deepcopy(data)
        changed['study_resume']['binding']['sampler_policy']['train_shuffle'] = True
        with self.assertRaisesRegex(ValueError, 'historical extra-shuffled'):
            restore_checkpoint(changed, model, cfg, schedule, wd, loader, binding)

    def test_inputs_require_complete_population_and_source_bound_bytes(self):
        self.assertEqual(len(required_files()), 2003)
        self.assertEqual(len(set(required_files())), 2003)
        self.assertIn("obses/episode_1999.pth", required_files())
        self.assertNotIn("door_locations.pth", required_files())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, receipt = root / "raw", root / "receipt"
            raw.mkdir(); receipt.mkdir()
            (raw / "states.pth").write_bytes(b"fixture")
            write_json(receipt / "protocol.json", {"archive_sha256": ARCHIVE_SHA, "revision": REVISION})
            write_json(receipt / "files.json", {"states.pth": {"bytes": 7, "sha256": sha256(raw / "states.pth")}})
            write_json(receipt / "report.json", {"status": "complete_pointmaze_inputs_match_pinned_archive",
                "protocol_sha256": sha256(receipt / "protocol.json"), "files_sha256": sha256(receipt / "files.json")})
            write_json(receipt / "DONE.json", {"report_sha256": sha256(receipt / "report.json")})
            with self.assertRaises(ValueError):
                verify_inputs(raw, receipt)
            with patch("offline_study.pointmaze_training_inputs.required_files", return_value=["states.pth"]):
                verify_inputs(raw, receipt)
                (raw / "states.pth").write_bytes(b"corrupt")
                with self.assertRaises(ValueError):
                    verify_inputs(raw, receipt)

    def test_actual_native_validation_pixels_actions_nested_rows_and_rng(self):
        if importlib.util.find_spec("torchvision") is None:
            self.skipTest("Native image parity is executed on the receiving worker")
        vendor = Path(__file__).resolve().parents[1] / "vendor/jepa-wms"
        use_vendor(vendor)
        from app.plan_common.datasets.point_maze_dset import PointMazeDataset
        from app.plan_common.datasets.traj_dset import TrajSubset, TrajSlicerDataset
        from app.plan_common.datasets.transforms import make_transforms
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "obses").mkdir()
            generator = torch.Generator().manual_seed(27)
            for name in ("states", "actions"):
                torch.save(torch.randn(3, 45, 2, generator=generator), root / f"{name}.pth")
            torch.save(torch.tensor([45, 45, 45]), root / "seq_lengths.pth")
            for i in range(3):
                torch.save(torch.randint(0, 256, (45, 32, 48, 3), generator=generator,
                    dtype=torch.uint8), root / f"obses/episode_{i:03d}.pth")
            dataset = PointMazeDataset(str(root), normalize_action=True,
                transform=make_transforms(img_size=32, random_horizontal_flip=False,
                    random_resize_aspect_ratio=(1., 1.), random_resize_scale=(1.777, 1.777)))
            nested = TrajSubset(TrajSubset(dataset, [2, 0, 1]), [1, 0])
            native = TrajSlicerDataset(nested, 8, frameskip=5, generator=generator)
            reader = ValidationFrames(native)
            for i in (0, 1, len(native)-1):
                before = rng_state()
                expected, after = native[i], rng_state()
                restore_rng(before)
                actual = reader[i]
                assert_same(expected, actual); assert_same(after, rng_state())
            native.num_frames = 4
            with self.assertRaises(ValueError):
                ValidationFrames(native)


if __name__ == "__main__":
    unittest.main()
