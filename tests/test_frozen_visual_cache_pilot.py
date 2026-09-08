"""Local CPU fixtures for the receiving pilot; never execute a CUDA model."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from offline_study import frozen_visual_cache as c
from offline_study import frozen_visual_cache_pilot as p
from offline_study.training_pilot import VirtualRankBatchSampler
from offline_study.vendor import use_vendor


VENDOR = Path(__file__).resolve().parents[1] / "vendor/jepa-wms"


class ImageEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.tensor(2.))
        self.calls = 0
        self.eval()

    def forward(self, images):
        self.calls += 1
        output = torch.zeros(len(images), 3, 3)
        output[:, 1:] = images.flatten(1)[:, :6].reshape(len(images), 2, 3) * self.weight
        return output[:, 1:]


class TinyTraining(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ImageEncoder()
        self.predictor = torch.nn.Linear(6, 1)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=.001)
        self.scaler = SimpleNamespace(state_dict=lambda: {})

    def optimization_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.rng = c.rng_snapshot()

    def tearDown(self):
        c.restore_rng(self.rng)

    def test_actual_upstream_caller_selects_exact_two_corrected_updates(self):
        cfg = p.training_config(VENDOR)
        updates, val = p.actual_native_schedule(VENDOR, cfg, range(145800), range(12200), range(1800), range(200))
        self.assertEqual(len(updates), 2)
        self.assertEqual(updates, list(VirtualRankBatchSampler(145800, shuffle=False))[:2])
        self.assertEqual(updates[0][:8], [0, 16, 32, 48, 64, 80, 96, 112])
        self.assertEqual(updates[1][:8], [128, 144, 160, 176, 192, 208, 224, 240])
        self.assertEqual(sum(map(len, val)), 64)
        self.assertEqual(val[0], [0, 16, 32, 48])

    def test_previous_extra_shuffle_is_not_accepted_by_native_gate(self):
        cfg = p.training_config(VENDOR)
        with patch.object(p, "VirtualRankBatchSampler", side_effect=lambda n, shuffle: VirtualRankBatchSampler(n, shuffle=True)):
            with self.assertRaisesRegex(ValueError, "actual upstream"):
                p.actual_native_schedule(VENDOR, cfg, range(145800), range(12200), range(1800), range(200))

    def test_native_nested_subset_frame_ids_preserve_original_clip_order(self):
        use_vendor(VENDOR)
        from app.plan_common.datasets.traj_dset import TrajSubset
        class Dataset:
            pass
        nested = TrajSubset(TrajSubset(Dataset(), [2, 0, 1]), [1, 0])
        slicer = SimpleNamespace(dataset=nested, slices=[(0, 3, 23), (1, 4, 24)], num_frames=4, frameskip=5)
        files = {f"obses/episode_{i:03d}.pth": {"sha256": str(i + 1) * 64} for i in range(3)}
        frames = p.ordered_frames(slicer, [0, 1] * 64, files)
        self.assertEqual(len(frames), 512)
        self.assertEqual([(f.relative_path, f.index) for f in frames[:8]],
            [("obses/episode_000.pth", i) for i in (3, 8, 13, 18)] +
            [("obses/episode_002.pth", i) for i in (4, 9, 14, 19)])
        with self.assertRaisesRegex(ValueError, "complete"):
            p.ordered_frames(slicer, [0], files)

    def test_independent_compositions_change_neighbors_and_preserve_every_frame(self):
        orders = p.composition_orders()
        self.assertEqual(list(orders), ["frame_major_cross_batch", "reverse"])
        for order in orders.values():
            self.assertEqual(sorted(order), list(range(1024)))
            self.assertNotEqual(order, list(range(1024)))
        self.assertEqual(orders["frame_major_cross_batch"][:32], list(range(0, 1024, 32)))
        self.assertEqual(orders["reverse"][:32], list(range(1023, 991, -1)))

    def make_cache(self, root, encoder, frames):
        binding = c.Binding(task="pointmaze", source_sha256="a" * 64, encoder_sha256=c.encoder_digest(encoder),
            transform_sha256="b" * 64, input_manifest_sha256="c" * 64,
            frame_inventory_sha256=c.digest(c.frame_inventory(frames)),
            runtime_sha256=c.digest(c.runtime_identity("cpu")), precision_sha256=c.digest(c.precision_identity("cpu")),
            input_shape=(32, 3, 2, 2), input_stride=(12, 4, 2, 1), input_dtype="torch.float32",
            output_shape=(32, 2, 3), output_stride=(9, 3, 1), output_dtype="torch.float32")
        return c.CacheStore.create(root, binding, frames, max_disk_bytes=16 << 20, min_free_bytes=0, max_memory_bytes=4096)

    def test_encoder_route_is_exactly16_full_batches_and_restores_native_validation(self):
        encoder = ImageEncoder()
        frames = [c.Frame("obses/episode_000.pth", "d" * 64, i) for i in range(512)]
        images = torch.arange(512 * 12, dtype=torch.float32).reshape(512, 3, 2, 2)
        with tempfile.TemporaryDirectory() as temporary, self.make_cache(Path(temporary) / "cache", encoder, frames) as store:
            with c.EncoderCacheSession(encoder, store, mode="record") as record:
                for rank in range(16):
                    with record.frames(frames[rank*32:(rank+1)*32]):
                        encoder(images[rank*32:(rank+1)*32])
            native_calls = encoder.calls
            with p.routed_encoder(encoder, frames, store) as timing:
                for rank in range(16):
                    actual = encoder(images[rank*32:(rank+1)*32])
                    self.assertEqual(actual.stride(), (9, 3, 1))
            self.assertEqual(encoder.calls, native_calls)
            self.assertEqual(timing.report()["encoder_calls"], 16)
            self.assertNotIn("forward", encoder.__dict__)
            encoder(images[:32])  # Validation must bypass cached training route.
            self.assertEqual(encoder.calls, native_calls + 1)
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                with p.routed_encoder(encoder, frames, store):
                    encoder(images[:32])
            self.assertNotIn("forward", encoder.__dict__)

    def test_gradient_observation_does_not_change_optimizer_or_objective(self):
        torch.manual_seed(21)
        original = TinyTraining()
        import copy
        observed = copy.deepcopy(original)
        for model in (original, observed):
            model.predictor(torch.ones(2, 6)).square().mean().backward()
        original.optimization_step()
        with p.gradient_evidence(observed, True) as evidence:
            observed.optimization_step()
        c.assert_bitwise(original.state_dict(), observed.state_dict())
        c.assert_bitwise(original.optimizer.state_dict(), observed.optimizer.state_dict())
        self.assertEqual(len(evidence), 1)
        self.assertIn("predictor.weight", evidence[0])
        self.assertIsNone(evidence[0]["encoder.weight"] if "encoder.weight" in evidence[0] else None)
        self.assertNotIn("optimization_step", observed.__dict__)

    def test_full_update_driver_reuses_objective_and_uncached_native_validation(self):
        model = TinyTraining()
        initial = (model, SimpleNamespace(_step=0), SimpleNamespace(_step=0))
        visual = torch.arange(128 * 4 * 12, dtype=torch.float32).reshape(128, 4, 3, 2, 2) / 1000
        batch = ({"visual": visual}, torch.zeros(128, 4, 2), torch.zeros(128, 4, 2), torch.zeros(128, 4))
        batches = [batch, batch]
        frames = [c.Frame("obses/episode_000.pth", "d" * 64, i) for i in range(512)]
        calls = []
        def objective(current, scheduler, wd, supplied, cpu):
            calls.append("unchanged_objective")
            scheduler._step += 1; wd._step += 1
            total = 0.
            for rank in range(16):
                torch.set_rng_state(cpu[rank])
                images = supplied[0]["visual"][rank*8:(rank+1)*8].flatten(0, 1)
                features = current.encoder(images)
                loss = current.predictor(features.flatten(1)).square().mean() * (torch.randint(2, ()).item() + 1)
                cpu[rank] = torch.get_rng_state(); (loss / 16).backward(); total += float(loss.detach()) / 16
            current.optimization_step()
            return {"loss": total}
        def monitoring(run, validation, schedule, event, cpu, cuda):
            calls.append("native_validation")
            self.assertEqual(event, 0)
            self.assertNotIn("forward", run.encoder.__dict__)
            return {"global_clips": 64, "diagnostic": torch.rand(1).item()}
        cpu = [torch.get_rng_state().clone() for _ in range(16)]
        with tempfile.TemporaryDirectory() as temporary, self.make_cache(Path(temporary) / "cache", model.encoder, frames) as store:
            with c.EncoderCacheSession(model.encoder, store, mode="record") as record:
                for rank in range(16):
                    with record.frames(frames[rank*32:(rank+1)*32]):
                        model.encoder(visual[rank*8:(rank+1)*8].flatten(0, 1))
            with patch.object(p, "two_batches", return_value=batches), patch.object(p, "accumulated_update", side_effect=objective), \
                    patch.object(p, "validation_step", side_effect=lambda vendor, model, *args: model), \
                    patch.object(p, "monitor", side_effect=monitoring), patch.object(torch.cuda, "synchronize"):
                def run(store):
                    return p.run_updates(initial, None, [[0]*128]*2, [frames]*2, p.tensor_evidence(batches),
                        cpu, cpu, VENDOR, {}, None, None, store, capture_gradients=True)[0]
                checked = c.paired_update_check(lambda: run(None), lambda: run(store))
            self.assertTrue(checked["native_and_cached_state_and_rng_bitwise"])
            self.assertEqual(calls, ["unchanged_objective", "native_validation", "unchanged_objective"] * 2)

    def test_scratch_and_time_caps_never_silently_reduce_scientific_work(self):
        frame = c.Frame("obses/episode_000.pth", "d" * 64, 0)
        with tempfile.TemporaryDirectory() as temporary, self.make_cache(Path(temporary) / "cache", ImageEncoder(), [frame]) as store:
            binding = replace(store.binding, output_shape=(32, 256, 384), output_stride=(98688, 384, 1))
        frames = [replace(frame, index=i) for i in range(1024)]
        result = p.check_budget(binding, frames)
        self.assertLess(result["conservative_scratch_bytes"], 1 << 30)
        with self.assertRaisesRegex(ValueError, "no truncation"):
            p.check_budget(binding, [replace(frame, index=i) for i in range(3000)])
        with self.assertRaisesRegex(ValueError, "FP32"):
            p.check_budget(replace(binding, output_dtype="torch.bfloat16"), frames)
        self.assertLess(p.GPU_STAGE_LIMIT_SECONDS, 600)
        self.assertEqual(p.REQUIRED_OUTER_TIMEOUT_SECONDS, 600)

    def test_input_failure_leaves_failed_receipt_without_any_gpu_initialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "pilot"
            argv = ["pilot", "--vendor", str(VENDOR), "--data-root", str(Path(temporary)/"raw"),
                "--input-receipt", str(Path(temporary)/"receipt"), "--input-check", str(Path(temporary)/"check"),
                "--output", str(output)]
            with patch("sys.argv", argv), patch.object(p, "verify_inputs", side_effect=ValueError("fixture input failure")), \
                    patch.object(torch.cuda, "set_device") as gpu, self.assertRaisesRegex(ValueError, "fixture"):
                p.main()
            gpu.assert_not_called()
            self.assertFalse((output / "DONE.json").exists())
            self.assertFalse(json.loads((output / "FAILED.json").read_text())["scientific_activation"])


if __name__ == "__main__":
    unittest.main()
