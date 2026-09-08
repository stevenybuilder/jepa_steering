"""CPU source/mechanics fixtures only; never real Push-T input-parity receipts."""
import abc
import ast
import copy
import inspect
import json
import math
import os
import pickle
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional
from unittest.mock import patch

import numpy as np
import torch
from einops import rearrange

from offline_study import robotics_training_batch as b
from offline_study import robotics_training_pilot as p
from offline_study.protocol import sha256, write_json

VENDOR = Path(__file__).resolve().parents[1] / "vendor/jepa-wms"


def definitions(path, names, namespace):
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise AssertionError("Missing actual pinned source definition")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def native_fixture_namespace(images):
    """Unchanged native Python methods; fake MP4 backend is only a test fixture."""
    calls = []

    class FixtureVideoReader:
        def __init__(self, path, num_threads):
            self.row = int(Path(path).stem.split("_")[1])
            calls.append((str(path), num_threads))

        def get_batch(self, frames):
            return images[self.row, list(frames)].clone()

    ns = {"torch": torch, "abc": abc, "Dataset": torch.utils.data.Dataset,
          "Optional": Optional, "Callable": Callable, "Path": Path, "pickle": pickle,
          "rearrange": rearrange, "log": SimpleNamespace(info=lambda *a: None),
          "decord": SimpleNamespace(bridge=SimpleNamespace(set_bridge=lambda _: None)),
          "VideoReader": FixtureVideoReader}
    definitions(VENDOR / "app/plan_common/datasets/traj_dset.py",
                ("TrajDataset", "TrajSlicerDataset"), ns)
    path = VENDOR / "app/plan_common/datasets/pusht_dset.py"
    constant_nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.Assign)
                      and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                      and n.targets[0].id.isupper()]
    exec(compile(ast.Module(body=constant_nodes, type_ignores=[]), str(path), "exec"), ns)
    definitions(path, ("PushTDataset", "load_pusht_slice_train_val"), ns)
    ns["fixture_calls"] = calls
    return ns


def native_fixture_transform():
    """Actual crop, normalization and VideoTransform; unused branches are stubs."""
    crop = {"torch": torch, "random": random, "np": np, "math": math}
    definitions(VENDOR / "src/datasets/utils/video/transforms.py",
                ("_get_param_spatial_crop", "random_resized_crop"), crop)
    video = SimpleNamespace(random_resized_crop=crop["random_resized_crop"],
        create_random_augment=lambda **kw: None)
    ns = {"torch": torch, "video_transforms": video, "RandomErasing": lambda *a, **kw: None}
    definitions(VENDOR / "app/plan_common/datasets/transforms.py",
                ("VideoTransform", "make_transforms", "_tensor_normalize_inplace"), ns)
    cfg = p.native_contract(VENDOR, "pusht")["config"]
    return b._native_transform(VENDOR, cfg, ns["make_transforms"])


class NativeBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = p.native_contract(VENDOR, "pusht")
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        self.ambient = b._rng()
        self.addCleanup(b._restore, self.ambient)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self):
        train = self.root / "train"
        train.mkdir()
        raw = {"states": torch.arange(2 * 24 * 5).reshape(2, 24, 5).float() / 9,
               "actions": torch.arange(2 * 24 * 2).reshape(2, 24, 2).float() - 100,
               "velocities": torch.arange(2 * 24 * 2).reshape(2, 24, 2).float() / 3}
        for key, name in (("states", "states.pth"), ("actions", "rel_actions.pth"),
                          ("velocities", "velocities.pth")):
            torch.save(raw[key], train / name)
        (train / "seq_lengths.pkl").write_bytes(pickle.dumps([23, 24]))
        image = (torch.arange(2 * 24 * 8 * 8 * 3).reshape(2, 24, 8, 8, 3) % 251).to(torch.uint8)
        ns, transform = native_fixture_namespace(image), native_fixture_transform()
        _, _, call = p.native_loader_probe(VENDOR, "pusht")
        original = b._native_training_branch(VENDOR / "app/plan_common/datasets/pusht_dset.py",
                                             ns, call, transform, self.root)
        return raw, image, ns, transform, original

    def test_actual_training_branch_never_constructs_validation(self):
        raw, images, ns, transform, original = self.fixture()
        self.assertFalse((self.root / "val").exists())
        self.assertEqual(original.dataset.data_path, self.root / "train")
        self.assertEqual(len(original), 9)
        self.assertEqual((original.num_frames, original.frameskip, original.action_skip), (4, 5, 1))
        self.assertEqual(original.process_actions, "concat")
        self.assertTrue(original.dataset.normalize_action)
        self.assertTrue(original.dataset.with_velocity)
        self.assertEqual(original.dataset.shapes, ["T", "T"])
        # No decoder call during construction, either train or validation.
        self.assertEqual(ns["fixture_calls"], [])

    def test_rank_major_indices_from_actual_caller(self):
        before = b._rng()
        indices = b._first_update_indices(self.contract, 1031)
        self.assertEqual(indices, [rank + 32 * offset for rank in range(32) for offset in range(8)])
        self.assertEqual(sorted(indices), list(range(256)))
        b.assert_same(before, b._rng())

    def test_independent_prefix_mapping_matches_actual_native_enumeration(self):
        _, _, _, _, original = self.fixture()
        indices = list(range(9))
        identities, count = b._independent_slices([23, 24], indices, 4, 5, 234)
        self.assertEqual(count, len(original))
        self.assertEqual(identities, original.slices)
        # Empty/too-short trajectories must not shift subsequent row identities.
        wanted, count = b._independent_slices([1, 20, 3, 21], [0, 1, 2], 4, 5, 234)
        self.assertEqual(count, 3)
        self.assertEqual(set(wanted), {(1, 0, 20), (3, 0, 20), (3, 1, 21)})
        with self.assertRaises(ValueError): b._independent_slices([10], [0], 4, 5, 234)

    def test_actual_native_transform_and_modalities_bitwise_plus_rng(self):
        raw, images, ns, transform, original = self.fixture()
        constants = {name: ns[name] for name in ("ACTION_MEAN", "ACTION_STD", "PROPRIO_MEAN", "PROPRIO_STD")}
        before = b._rng()
        for clip in (0, 4, 8):
            identity = original.slices[clip]
            row, start, end = identity
            image = images[row, list(range(start, end, 5))].clone()
            sample, record = b._compare_clip(lambda: original[clip],
                lambda: b._selected_sample(raw, constants, identity, image, transform), identity)
            self.assertEqual(record["sample_sha256"], p._batch_digest(sample))
            self.assertNotEqual(record["rng_before_sha256"], record["rng_after_sha256"])
            self.assertEqual(sample[1].shape, (4, 10))
            self.assertEqual(sample[2].shape, (4, 7))
            self.assertTrue(torch.equal(sample[2][:, :5], raw["states"][row, list(range(start, end, 5))]))
            self.assertTrue(torch.equal(sample[3], torch.zeros(4)))
        # Native no-augmentation crop still consumes Python + NumPy RNG.
        after = b._rng()
        self.assertNotEqual(before[0], after[0])
        self.assertFalse(np.array_equal(before[1][1], after[1][1]))
        self.assertTrue(torch.equal(before[2], after[2]))
        self.assertTrue(all("/train/" in path for path, _ in ns["fixture_calls"]))

    def test_rng_mismatch_and_changed_selected_decode_reject(self):
        sample = ({"visual": torch.ones(4, 3, 2, 2), "proprio": torch.zeros(4, 4)},
                  torch.zeros(4, 10), torch.zeros(4, 7), torch.zeros(4))
        before = b._rng()

        def wrong_rng():
            np.random.rand()
            return copy.deepcopy(sample)

        with self.assertRaises(ValueError):
            b._compare_clip(lambda: sample, wrong_rng, (0, 0, 20))
        b.assert_same(before, b._rng())
        changed = copy.deepcopy(sample)
        changed[0]["visual"][0, 0, 0, 0] += 1
        with self.assertRaises(ValueError):
            b._compare_clip(lambda: sample, lambda: changed, (0, 0, 20))
        b.assert_same(before, b._rng())

    def test_signed_zero_is_not_bitwise_equal(self):
        sample = ({"visual": torch.zeros(1), "proprio": torch.zeros(1)},
                  torch.zeros(1), torch.zeros(1), torch.zeros(1))
        changed = copy.deepcopy(sample)
        changed[0]["visual"].neg_()
        self.assertTrue(torch.equal(sample[0]["visual"], changed[0]["visual"]))
        with self.assertRaisesRegex(ValueError, "bytes changed"):
            b._compare_clip(lambda: sample, lambda: changed, (0, 0, 20))

    def test_sample_storage_shape_dtype_and_native_order(self):
        sample = ({"visual": torch.ones(4, 3, 224, 224), "proprio": torch.ones(4, 4)},
                  torch.ones(4, 10), torch.ones(4, 7), torch.zeros(4))
        batch = {}
        b._append_sample(batch, sample, 0, count=2)
        changed = copy.deepcopy(sample)
        changed[1].mul_(2)
        b._append_sample(batch, changed, 1, count=2)
        self.assertTrue(torch.equal(b._batch_tuple(batch)[1][0], sample[1]))
        self.assertTrue(torch.equal(b._batch_tuple(batch)[1][1], changed[1]))
        changed[0]["visual"] = changed[0]["visual"].bfloat16()
        with self.assertRaisesRegex(ValueError, "modality changed"):
            b._append_sample({}, changed, 0)

    def test_training_hash_guard_does_not_open_validation(self):
        (self.root / "train").mkdir()
        path = self.root / "train/states.pth"
        path.write_bytes(b"fixture metadata, not production data")
        manifest = {"train/states.pth": {"bytes": path.stat().st_size, "sha256": sha256(path)},
                    "val/states.pth": {"bytes": 1, "sha256": "0" * 64}}
        self.assertEqual(b._verify_training_members(self.root, manifest, ["train/states.pth"]),
                         {"train/states.pth": manifest["train/states.pth"]})
        with self.assertRaisesRegex(ValueError, "Only manifest-bound training"):
            b._verify_training_members(self.root, manifest, ["val/states.pth"])
        path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "bytes changed"):
            b._verify_training_members(self.root, manifest, ["train/states.pth"])

    def test_optional_shapes_and_member_symlinks_reject(self):
        (self.root / "train").mkdir()
        shapes = self.root / "train/shapes.pkl"
        shapes.write_bytes(b"unexpected")
        with self.assertRaisesRegex(ValueError, "optional shapes"):
            b._verify_training_members(self.root, {}, [])
        shapes.unlink()
        (self.root / "target").write_bytes(b"outside")
        (self.root / "train/states.pth").symlink_to(self.root / "target")
        with self.assertRaisesRegex(ValueError, "symlink"):
            b._verify_training_members(self.root, {"train/states.pth": {}}, ["train/states.pth"])

    def test_public_producer_has_no_fixture_or_reader_injection(self):
        parameters = set(inspect.signature(b.produce_pusht_batch).parameters)
        self.assertEqual(parameters, {"vendor", "data_root", "raw_inputs_receipt", "provenance", "output", "model_seed"})
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            with self.assertRaisesRegex(ValueError, "Dedicated CPU"):
                b.produce_pusht_batch(VENDOR, self.root, raw_inputs_receipt=self.root,
                    provenance=self.root / "not_real.json", output=self.root / "never_created")
        self.assertFalse((self.root / "never_created").exists())

    def test_self_hashed_fixture_raw_receipt_cannot_create_real_batch_proof(self):
        # Even matching raw report fields are not enough: the public producer
        # requires the original hard-pinned manifest before importing/decoding.
        receipt = self.root / "raw"
        receipt.mkdir()
        write_json(receipt / "protocol.json", {"fixture_only": True})
        report = {"status": "complete_robotics_raw_inputs_verified", "task": "pusht",
            "data_root": str(self.root), "data_revision": b.DATA_REVISION,
            "raw_manifest_sha256": b.MANIFEST_HASHES["pusht"], "training_rows": 18685,
            "validation_rows": 21, "training_or_validation_authorized": False,
            "native_reader_parity_established": False, "model_or_data_loader_initialized": False,
            "protocol_sha256": sha256(receipt / "protocol.json")}
        write_json(receipt / "report.json", report)
        write_json(receipt / "DONE.json", {"report_sha256": sha256(receipt / "report.json")})
        manifest = self.root / "fake_manifest.json"
        manifest.write_text("{}")
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}), patch.object(b, "use_vendor") as imported:
            with self.assertRaisesRegex(ValueError, "pinned download/extraction manifest"):
                b.produce_pusht_batch(VENDOR, self.root, raw_inputs_receipt=receipt,
                    provenance=manifest, output=self.root / "never_created")
            imported.assert_not_called()
        self.assertFalse((self.root / "never_created").exists())

    def test_actual_crop_sources_remain_pinned(self):
        for relative, expected in b.EXTRA_SOURCES.items():
            self.assertEqual(sha256(VENDOR / relative), expected)


if __name__ == "__main__":
    unittest.main()
