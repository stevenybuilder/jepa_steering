"""CPU-only synthetic input tests plus actual pinned native reader methods."""
import ast
import copy
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from einops import rearrange
import numpy as np
import torch

from offline_study import fixed_combined_stimulus as s
from offline_study.inventory import _metaworld_trajectory_group


class StimulusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)
        cls.vendor = Path(__file__).resolve().parents[1] / "vendor/jepa-wms"
        source = (cls.vendor / s.READER_SOURCE).read_text()
        tree = ast.parse(source)
        klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MetaworldHFDataset")
        methods = [n for n in klass.body if isinstance(n, ast.FunctionDef) and
                   n.name in ("get_frames", "get_seq_length", "__getitem__")]
        cls.reader_namespace = {"np": np, "torch": torch, "rearrange": rearrange}
        exec(compile(ast.Module(body=methods, type_ignores=[]), s.READER_SOURCE, "exec"), cls.reader_namespace)
        cls.reader = type("NativeMethodsOnly", (), {n.name: cls.reader_namespace[n.name] for n in methods})

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def fixture(self):
        states = torch.arange(3900, dtype=torch.float32).reshape(100, 39) / 1000
        actions = torch.arange(396, dtype=torch.float32).reshape(99, 4) / 500
        selected = {"task": "mw-reach", "source_seed": 1, "source_episode": 88, "length": 99,
            "lineage_group": _metaworld_trajectory_group("mw-reach", states[:-1], actions)}
        row = {"task": "mw-reach", "seed": 1, "episode": 88, "states": states.tolist(),
            "actions": actions.tolist(), "video": {"bytes": b"fitting-video-fixture"}}
        return row, selected

    def test_published_stats_exact_source_and_float32(self):
        stats = s.published_stats(self.vendor / s.STATS_SOURCE)
        self.assertEqual(set(stats), set(s.NORMALIZATION_KEYS))
        self.assertEqual(stats["action_mean"][0].item(), 0.005723577458411455)
        self.assertEqual(stats["state_std"].shape, (39,))
        self.assertTrue(all(t.dtype == torch.float32 for t in stats.values()))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "stats.py"
            path.write_text((self.vendor / s.STATS_SOURCE).read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "source changed"):
                s.published_stats(path)

    def test_original_global_row_arithmetic_and_not_renumbered(self):
        counts = {f"train-{i:05d}-of-00126.parquet": 100 for i in range(126)}
        self.assertEqual(s.locate_row(10587, counts), {"path": "train-00105-of-00126.parquet", "local_row": 87})
        for index in (-1, 12600, 1.5, True):
            with self.assertRaises(ValueError): s.locate_row(index, counts)
        with self.assertRaises(ValueError): s.locate_row(0, {"a": 0})

    def test_raw_row_exact_original_lineage_and_alignment(self):
        row, selected = self.fixture()
        states, actions = s.validate_raw_row(row, selected)
        self.assertEqual(states.shape, (99, 39)); self.assertEqual(actions.shape, (99, 4))
        for key, value in (("task", "mw-reach-wall"), ("seed", 2), ("episode", 89)):
            bad = {**row, key: value}
            with self.assertRaisesRegex(ValueError, "original fitting lineage"):
                s.validate_raw_row(bad, selected)
        bad = copy.deepcopy(row); bad["actions"][0][0] += .001
        with self.assertRaises(ValueError): s.validate_raw_row(bad, selected)
        bad = {**row, "states": row["states"][:-1]}
        with self.assertRaisesRegex(ValueError, "alignment"): s.validate_raw_row(bad, selected)

    def test_video_auto_decoder_and_nonfinite_states_refused(self):
        row, selected = self.fixture()
        with self.assertRaisesRegex(ValueError, "MP4"):
            s.validate_raw_row({**row, "video": object()}, selected)
        row["states"][0][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"): s.validate_raw_row(row, selected)

    def test_native_reader_parity_and_rng_restoration_with_original_methods(self):
        row, selected = self.fixture(); states, actions = s.validate_raw_row(row, selected)
        frames = (np.arange(100 * 4 * 4 * 3).reshape(100, 4, 4, 3) % 255).astype(np.uint8)
        seen = []
        def decode(raw):
            seen.append(raw); return frames.copy()
        def transform(pixels):
            return pixels + torch.rand(1).item()
        self.reader_namespace["decode_video_frames"] = decode
        stats = s.published_stats(self.vendor / s.STATS_SOURCE)
        before = s.rng_digest(s.rng_state())
        value, parity = s.selected_native_parity(row, states, actions, stats, transform, self.reader, decode)
        self.assertEqual(s.rng_digest(s.rng_state()), before)
        self.assertEqual(seen, [row["video"]["bytes"]] * 2)
        self.assertTrue(parity["selected_native_reader_bytes_equal"])
        self.assertFalse(parity["full_dataset_constructor_parity"])
        self.assertEqual(value["observations"]["visual"].shape, (1, 18, 3, 4, 4))
        expected = ((actions - stats["action_mean"]) / stats["action_std"])[:90].reshape(1, 18, 20)
        s.assert_bytes(value["actions"], expected, "all concatenated actions")

    def test_native_reader_mismatch_is_not_tolerated_and_rng_restored(self):
        row, selected = self.fixture(); states, actions = s.validate_raw_row(row, selected)
        frames = np.zeros((100, 4, 4, 3), dtype=np.uint8)
        self.reader_namespace["decode_video_frames"] = lambda raw: frames
        class Changed(self.reader):
            def __getitem__(self, index):
                obs, action, state, reward, misc = super().__getitem__(index)
                obs["proprio"] = obs["proprio"] + 1
                return obs, action, state, reward, misc
        before = s.rng_digest(s.rng_state())
        with self.assertRaisesRegex(ValueError, "Byte parity"):
            s.selected_native_parity(row, states, actions, s.published_stats(self.vendor / s.STATS_SOURCE),
                lambda v: v, Changed, lambda raw: frames)
        self.assertEqual(s.rng_digest(s.rng_state()), before)

    def test_cpu_only_requirement_before_output_mutation(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            out = Path(temp) / "new"
            with self.assertRaisesRegex(ValueError, "CUDA hidden"):
                s.prepare(types.SimpleNamespace(output=out))
            self.assertFalse(out.exists())

    def test_wrong_raw_proof_stops_before_dataset_or_decoder(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
            root = Path(temp); source = root / "cohort.json"; source.write_text("{}")
            args = types.SimpleNamespace(output=root / "out", cohort=source, cohort_sha256=s.sha256(source), raw_proof=source)
            with self.assertRaisesRegex(ValueError, "proof changed"): s.prepare(args)
            self.assertTrue((args.output / "FAILED.json").exists())
            self.assertFalse((args.output / "STIMULUS.json").exists())
            with self.assertRaises(FileExistsError): s.prepare(args)


if __name__ == "__main__": unittest.main()
