"""Small CPU fixtures only; no DINO model, GPU, network or scientific cache."""
from dataclasses import replace
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from offline_study.models import frozen_visual_cache as c


class FixtureEncoder(torch.nn.Module):
    def __init__(self, batch_dependent=False):
        super().__init__()
        self.register_buffer("weight", torch.tensor(2.))
        self.batch_dependent = batch_dependent
        self.batches = []
        self.eval()

    def forward(self, value):
        self.batches.append(len(value))
        result = value.reshape(len(value), -1)[:, :6].reshape(len(value), 2, 3) * self.weight
        if self.batch_dependent:
            result = result + value.mean()
        # Match DINO's gap after the class token, not a contiguous tensor.
        padded = torch.zeros(len(value), 3, 3, dtype=value.dtype)
        padded[:, 1:] = result
        return padded[:, 1:]


class ModeEncoder(torch.nn.Module):
    """Small source-bound deterministic fixture, not a generic whitelist bypass."""
    def __init__(self, *, mode_sensitive=False, stochastic=False, stateful=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.), requires_grad=False)
        self.drop = torch.nn.Dropout(0.)
        self.mode_sensitive, self.stochastic, self.stateful = mode_sensitive, stochastic, stateful
        self.counter = 0
        self.eval()

    def forward(self, value):
        if self.training and self.stochastic:
            torch.rand(())
        if self.training and self.stateful:
            self.counter += 1
        output = torch.zeros(len(value), 3, 3)
        output[:, 1:] = self.drop(value.flatten(1)[:, :6].reshape(len(value), 2, 3) * self.weight)
        if self.training and self.mode_sensitive:
            output[:, 1:] += 1
        return output[:, 1:]


class ModePolicyTests(unittest.TestCase):
    def setUp(self):
        self.before = c.rng_snapshot()
        self.tmp = tempfile.TemporaryDirectory()
        self.encoder = ModeEncoder()
        self.frames = tuple(c.Frame("obses/episode_000.pth", "d" * 64, i) for i in range(4))
        self.images = torch.arange(48, dtype=torch.float32).reshape(4, 3, 2, 2)
        self.binding = c.Binding(task="pointmaze", source_sha256="a" * 64,
            encoder_sha256=c.encoder_digest(self.encoder), transform_sha256="b" * 64,
            input_manifest_sha256="c" * 64, frame_inventory_sha256=c.digest(c.frame_inventory(self.frames)),
            runtime_sha256=c.digest(c.runtime_identity("cpu")), precision_sha256=c.digest(c.precision_identity("cpu")),
            input_shape=(2, 3, 2, 2), input_stride=(12, 4, 2, 1), input_dtype="torch.float32",
            output_shape=(2, 2, 3), output_stride=(9, 3, 1), output_dtype="torch.float32")
        self.store = c.CacheStore.create(Path(self.tmp.name) / "cache", self.binding, self.frames,
            max_disk_bytes=1 << 20, min_free_bytes=0)
        self.orders = {"original": [0, 1, 2, 3], "cross": [0, 2, 1, 3], "reverse": [3, 2, 1, 0]}

    def tearDown(self):
        self.store.close(); self.tmp.cleanup(); c.restore_rng(self.before)

    def policy(self, encoder=None, **kwargs):
        encoder = encoder or self.encoder
        identity = c.class_identity(type(encoder))
        args = {"audited_classes": {identity["class"]: identity["file_sha256"]},
            "source_receipt": {"source_sha256": self.binding.source_sha256, "scope": "CPU fixture"}}
        return c.EngineeringModePolicy(encoder, self.binding, **{**args, **kwargs})

    def batches(self, order):
        for start in range(0, 4, 2):
            selected = order[start:start+2]
            yield [self.frames[i] for i in selected], self.images[selected]

    def populate(self, encoder=None):
        encoder = encoder or self.encoder
        with c.EncoderCacheSession(encoder, self.store, mode="record") as session:
            for frames, images in self.batches(self.orders["original"]):
                with session.frames(frames):
                    encoder(images)

    def prove(self, policy, encoder=None):
        return policy.prove(encoder or self.encoder, self.store, self.frames, self.orders, self.batches)

    def test_complete_mode_and_composition_proof_preserves_native_modes_then_allows_replay(self):
        self.populate(); policy = self.policy()
        with self.assertRaisesRegex(ValueError, "proof"):
            with c.EncoderCacheSession(self.encoder, self.store, mode="engineering_replay", mode_policy=policy):
                pass
        before = c.rng_snapshot()
        report = self.prove(policy)
        self.assertEqual(len(report["checks"]), 6)
        self.assertEqual({row["training"] for row in report["checks"]}, {True, False})
        self.assertFalse(self.encoder.training)
        c.assert_bitwise(before, c.rng_snapshot())
        self.encoder.train()  # Stand-in for the upstream validation exit, not forced by cache.
        expected = self.encoder(self.images[:2])
        with c.EncoderCacheSession(self.encoder, self.store, mode="engineering_replay", mode_policy=policy) as session:
            with session.frames(self.frames[:2]):
                c.assert_bitwise(expected, self.encoder(self.images[:2]))
        self.assertTrue(self.encoder.training)
        with self.assertRaisesRegex(ValueError, "eval-mode"):
            with c.EncoderCacheSession(self.encoder, self.store, mode="record"):
                pass
        with self.assertRaisesRegex(ValueError, "Scientific activation"):
            c.EncoderCacheSession(self.encoder, self.store, mode="science", mode_policy=policy)

    def test_source_weight_binding_and_unknown_components_are_fail_closed(self):
        identity = c.class_identity(type(self.encoder))
        with self.assertRaisesRegex(ValueError, "source"):
            self.policy(source_receipt={"source_sha256": "f" * 64})
        with self.assertRaisesRegex(ValueError, "source"):
            self.policy(audited_classes={identity["class"]: "f" * 64})
        self.encoder.weight.data.add_(1)
        with self.assertRaisesRegex(ValueError, "weights"):
            self.policy()
        self.encoder.weight.data.sub_(1)
        self.encoder.extra = torch.nn.BatchNorm2d(3, track_running_stats=False, affine=False)
        self.encoder.extra.eval()
        with self.assertRaisesRegex(ValueError, "Unaudited"):
            self.policy()

    def test_nonzero_dropout_attention_drop_path_and_buffers_are_rejected(self):
        self.encoder.drop.p = .1
        with self.assertRaisesRegex(ValueError, "stochastic"):
            self.policy()
        self.encoder.drop.p = 0.
        for name in ("attn_drop", "sample_drop_ratio", "drop_prob"):
            setattr(self.encoder, name, .1)
            with self.assertRaisesRegex(ValueError, "stochastic"):
                self.policy()
            delattr(self.encoder, name)
        self.encoder.register_buffer("hidden", torch.zeros(1), persistent=False)
        with self.assertRaisesRegex(ValueError, "buffer"):
            self.policy()

    def test_mode_sensitive_stochastic_or_state_mutating_encoder_never_passes(self):
        for option in ("mode_sensitive", "stochastic", "stateful"):
            with self.subTest(option=option):
                encoder = ModeEncoder(**{option: True})
                self.populate(encoder); policy = self.policy(encoder)
                before = c.rng_snapshot()
                with self.assertRaises(ValueError):
                    self.prove(policy, encoder)
                self.assertIsNone(policy._proof)
                self.assertFalse(encoder.training)
                c.assert_bitwise(before, c.rng_snapshot())

    def test_partial_compositions_missing_frames_and_mixed_modes_are_rejected(self):
        policy = self.policy()
        with self.assertRaisesRegex(ValueError, "prepopulated"):
            self.prove(policy)
        self.populate()
        with self.assertRaisesRegex(ValueError, "permutations"):
            policy.prove(self.encoder, self.store, self.frames, {**self.orders, "reverse": [0,1,2,3]}, self.batches)
        def partial(order):
            yield next(self.batches(order))
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            policy.prove(self.encoder, self.store, self.frames, self.orders, partial)
        self.assertIsNone(policy._proof)
        self.encoder.drop.train()
        with self.assertRaisesRegex(ValueError, "Mixed"):
            self.policy()

    def test_post_proof_configuration_and_within_session_mode_mutations_rejected(self):
        self.populate(); policy = self.policy(); self.prove(policy)
        self.encoder.mode_sensitive = True
        with self.assertRaisesRegex(ValueError, "changed"):
            with c.EncoderCacheSession(self.encoder, self.store, mode="engineering_replay", mode_policy=policy):
                pass
        self.encoder.mode_sensitive = False
        with self.assertRaisesRegex(ValueError, "mode changed"):
            with c.EncoderCacheSession(self.encoder, self.store, mode="engineering_replay", mode_policy=policy) as session:
                self.encoder.train()
                with session.frames(self.frames[:2]):
                    self.encoder(self.images[:2])
        self.assertNotIn("forward", self.encoder.__dict__)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.initial_rng = c.rng_snapshot()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "cache"
        self.encoder = FixtureEncoder()
        self.frames = tuple(c.Frame(f"obses/episode_{i:03d}.pth", str(i + 1) * 64, 0) for i in range(4))
        self.images = torch.arange(4 * 3 * 2 * 2, dtype=torch.float32).reshape(4, 3, 2, 2)
        self.binding = c.Binding(task="wall", source_sha256="a" * 64,
            encoder_sha256=c.encoder_digest(self.encoder), transform_sha256="b" * 64,
            input_manifest_sha256="c" * 64, frame_inventory_sha256=c.digest(c.frame_inventory(self.frames)),
            runtime_sha256=c.digest(c.runtime_identity("cpu")), precision_sha256=c.digest(c.precision_identity("cpu")),
            input_shape=(2, 3, 2, 2), input_stride=(12, 4, 2, 1), input_dtype="torch.float32",
            output_shape=(2, 2, 3), output_stride=(9, 3, 1), output_dtype="torch.float32")
        self.stores = []

    def tearDown(self):
        for store in self.stores:
            store.close()
        self.tmp.cleanup()
        c.restore_rng(self.initial_rng)

    def store(self, **options):
        options.setdefault("max_disk_bytes", 1 << 20)
        options.setdefault("min_free_bytes", 0)
        store = c.CacheStore.create(self.root, self.binding, self.frames, **options)
        self.stores.append(store)
        return store

    def populate(self, store):
        with c.EncoderCacheSession(self.encoder, store, mode="record") as session:
            for indices in ((0, 1), (2, 3)):
                with session.frames([self.frames[i] for i in indices]):
                    self.encoder(self.images[list(indices)])

    def test_independent_composition_exact_dtype_bits_and_original_stride(self):
        store = self.store(max_memory_bytes=48)
        before = c.rng_snapshot()
        self.populate(store)
        # Overlapping original/new batch pairs and both image positions.
        for indices in ((3, 0), (1, 2), (0, 0), (2, 1)):
            images = self.images[list(indices)]
            expected = self.encoder(images)
            with c.EncoderCacheSession(self.encoder, store, mode="record") as record:
                with record.frames([self.frames[i] for i in indices]):
                    c.assert_bitwise(expected, self.encoder(images))
            calls_before_replay = len(self.encoder.batches)
            with c.EncoderCacheSession(self.encoder, store, mode="engineering_replay") as replay:
                with replay.frames([self.frames[i] for i in indices]):
                    actual = self.encoder(images)
            c.assert_bitwise(expected, actual)
            self.assertEqual(len(self.encoder.batches), calls_before_replay)
        self.assertTrue(all(size == 2 for size in self.encoder.batches))
        c.assert_bitwise(before, c.rng_snapshot())
        self.assertLessEqual(store.memory_bytes, 48)

    def test_batch_dependent_encoder_refused_on_independent_composition(self):
        self.encoder = FixtureEncoder(batch_dependent=True)
        store = self.store()
        self.populate(store)
        with c.EncoderCacheSession(self.encoder, store, mode="record") as session:
            with self.assertRaisesRegex(ValueError, "batch composition"):
                with session.frames([self.frames[3], self.frames[0]]):
                    self.encoder(self.images[[3, 0]])

    def test_replay_never_computes_missing_frames(self):
        store = self.store()
        with c.EncoderCacheSession(self.encoder, store, mode="engineering_replay") as session:
            with self.assertRaises(KeyError), session.frames(self.frames[:2]):
                self.encoder(self.images[:2])
        self.assertEqual(self.encoder.batches, [])

    def test_original_transform_rng_path_runs_before_both_modes(self):
        store = self.store()
        start = c.rng_snapshot()
        def transform():
            random.random(); np.random.random(); torch.rand(1)
            return self.images[:2].clone()
        images = transform()
        with c.EncoderCacheSession(self.encoder, store, mode="record") as session:
            with session.frames(self.frames[:2]):
                expected = self.encoder(images)
        recorded = c.rng_snapshot()
        c.restore_rng(start)
        images = transform()
        with c.EncoderCacheSession(self.encoder, store, mode="engineering_replay") as session:
            with session.frames(self.frames[:2]):
                actual = self.encoder(images)
        c.assert_bitwise(expected, actual)
        c.assert_bitwise(recorded, c.rng_snapshot())

    def test_no_quantization_and_signed_zero_preserved(self):
        self.binding = replace(self.binding, input_dtype="torch.bfloat16", output_dtype="torch.bfloat16")
        store = self.store()
        feature = torch.tensor([[0., -0., 1.003], [3.25, -8.5, 0.125]], dtype=torch.bfloat16)
        store.put(self.frames[0], "d" * 64, feature)
        actual = store.get(self.frames[0], "d" * 64)
        c.assert_bitwise(feature, actual)
        with self.assertRaises(ValueError):
            store.put(self.frames[1], "d" * 64, feature.float())
        changed = feature.clone(); changed[0, 1] = 0.
        with self.assertRaises(ValueError):
            c.assert_bitwise(feature, changed)

    def test_content_pixel_and_input_inventory_checks(self):
        store = self.store()
        self.populate(store)
        pixel = c.tensor_digest(self.images[0])
        with self.assertRaisesRegex(ValueError, "checksum"):
            store.get(self.frames[0], "f" * 64)
        with self.assertRaisesRegex(ValueError, "absent"):
            store.get(c.Frame("obses/new.pth", "e" * 64, 4), pixel)
        path = store.path(self.frames[0])
        payload = torch.load(path, weights_only=True)
        payload["tensor"][0, 0] += 1
        torch.save(payload, path)
        with self.assertRaisesRegex(ValueError, "checksum"):
            store.get(self.frames[0], pixel)

    def test_memory_lru_is_bounded_and_does_not_alias_returned_values(self):
        store = self.store(max_memory_bytes=24)
        self.populate(store)
        original = store.get(self.frames[0], c.tensor_digest(self.images[0]))
        original.fill_(123)
        reread = store.get(self.frames[0], c.tensor_digest(self.images[0]))
        self.assertFalse(torch.equal(original, reread))
        store.get(self.frames[1], c.tensor_digest(self.images[1]))
        self.assertEqual(store.memory_bytes, 24)
        self.assertEqual(len(store.memory), 1)

    def test_frozen_weights_eval_and_hook_guards_restore_original_forward(self):
        store = self.store()
        with self.assertRaisesRegex(ValueError, "frozen"):
            self.encoder.train()
            with c.EncoderCacheSession(self.encoder, store, mode="record"):
                pass
        self.encoder.eval()
        self.encoder.register_parameter("learned", torch.nn.Parameter(torch.tensor(1.)))
        with self.assertRaisesRegex(ValueError, "frozen"):
            with c.EncoderCacheSession(self.encoder, store, mode="record"):
                pass
        del self.encoder.learned
        hook = self.encoder.register_forward_hook(lambda *args: None)
        with self.assertRaisesRegex(ValueError, "hooks"):
            with c.EncoderCacheSession(self.encoder, store, mode="record"):
                pass
        hook.remove()
        with self.assertRaisesRegex(ValueError, "changed"):
            with c.EncoderCacheSession(self.encoder, store, mode="record") as session:
                self.encoder.weight.add_(1)
                with session.frames(self.frames[:2]):
                    self.encoder(self.images[:2])
        self.assertNotIn("forward", self.encoder.__dict__)

    def test_runtime_precision_shape_and_one_call_scope_guards(self):
        store = self.store()
        with c.EncoderCacheSession(self.encoder, store, mode="record") as session:
            with self.assertRaisesRegex(ValueError, "ordered"):
                self.encoder(self.images[:2])
            with self.assertRaisesRegex(ValueError, "exactly one"):
                with session.frames(self.frames[:2]):
                    pass
            with self.assertRaisesRegex(ValueError, "contract"), session.frames(self.frames[:2]):
                self.encoder(self.images[:1])
            with patch.object(c, "runtime_identity", return_value={}), self.assertRaisesRegex(ValueError, "contract"):
                with session.frames(self.frames[:2]):
                    self.encoder(self.images[:2])
            with torch.autocast("cpu", dtype=torch.bfloat16), self.assertRaisesRegex(ValueError, "contract"):
                with session.frames(self.frames[:2]):
                    self.encoder(self.images[:2])

    def test_writer_lock_quota_and_atomic_failure_leave_old_entries_intact(self):
        store = self.store(max_disk_bytes=9000)
        feature = torch.zeros(2, 3)
        store.put(self.frames[0], "d" * 64, feature)
        path = store.path(self.frames[0]); original = path.read_bytes()
        with self.assertRaises(BlockingIOError):
            c.CacheStore(self.root, self.binding, writable=True)
        with self.assertRaises(OSError):
            store.put(self.frames[1], "d" * 64, feature)
        with self.assertRaisesRegex(ValueError, "batch composition"):
            store.put(self.frames[0], "d" * 64, feature + 1)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.root.rglob(".pending-*")), [])

    def test_failed_atomic_publication_removes_only_owned_tempfile(self):
        store = self.store()
        with patch.object(c.torch, "save", side_effect=OSError("fixture storage error")), self.assertRaises(OSError):
            store.put(self.frames[0], "d" * 64, torch.zeros(2, 3))
        self.assertFalse(store.path(self.frames[0]).exists())
        self.assertEqual(list(self.root.rglob(".pending-*")), [])

    def test_encoder_rng_consumption_cannot_publish_features(self):
        store = self.store()
        original = self.encoder.forward
        def stochastic(images):
            torch.rand(1)
            return original(images)
        self.encoder.forward = stochastic
        with self.assertRaises(ValueError):
            with c.EncoderCacheSession(self.encoder, store, mode="record") as session:
                with session.frames(self.frames[:2]):
                    self.encoder(self.images[:2])
        self.assertFalse(store.path(self.frames[0]).exists())
        self.assertIs(self.encoder.forward, stochastic)

    def test_global_activation_hooks_rejected_and_symlink_never_followed(self):
        store = self.store()
        handle = torch.nn.modules.module.register_module_forward_hook(lambda *args: None)
        try:
            with self.assertRaisesRegex(ValueError, "hooks"):
                with c.EncoderCacheSession(self.encoder, store, mode="record"):
                    pass
        finally:
            handle.remove()
        (self.root / "entries").symlink_to(Path(self.tmp.name), target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symbolic"):
            store.put(self.frames[0], "d" * 64, torch.zeros(2, 3))

    def test_bound_manifest_reopen_and_science_activation_refused(self):
        store = self.store()
        for field in ("task", "source_sha256", "encoder_sha256", "transform_sha256", "input_manifest_sha256",
                      "runtime_sha256", "precision_sha256"):
            changed = replace(self.binding, **{field: "pointmaze" if field == "task" else "f" * 64})
            with self.assertRaises(ValueError):
                c.CacheStore(self.root, changed)
        with self.assertRaisesRegex(ValueError, "Scientific activation"):
            c.EncoderCacheSession(self.encoder, store, mode="science")
        metadata = json.loads((self.root / "manifest.json").read_text())
        self.assertFalse(metadata["scientific_activation"])

    def test_full_update_rng_parity_requires_state_coverage(self):
        before = c.rng_snapshot()
        def complete():
            parameter = torch.nn.Parameter(torch.tensor(1.))
            opt = torch.optim.Adam([parameter], lr=.01)
            losses = []
            for _ in range(2):
                loss = parameter.square() * torch.rand(())
                loss.backward(); losses.append(float(loss.detach())); opt.step(); opt.zero_grad()
            return {"losses": losses, "parameters": parameter.detach().clone(), "gradients": {},
                "optimizer": opt.state_dict(), "scaler": {}, "scheduler": 2,
                "logical_rngs": [torch.get_rng_state()], "loader_rng": torch.get_rng_state(),
                "validation_events": 1, "updates": 2}
        result = c.paired_update_check(complete, complete)
        self.assertTrue(result["native_and_cached_state_and_rng_bitwise"])
        c.assert_bitwise(before, c.rng_snapshot())
        def wrong_rng():
            result = complete(); random.random(); return result
        with self.assertRaises(ValueError):
            c.paired_update_check(complete, wrong_rng)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            c.paired_update_check(lambda: {"losses": []}, complete)
        c.assert_bitwise(before, c.rng_snapshot())

    def test_complete_cached_training_updates_match_native_predictor_and_rng(self):
        store = self.store()
        self.populate(store)
        def run(cached):
            predictor = torch.nn.Linear(6, 1)
            optimizer = torch.optim.Adam(predictor.parameters(), lr=.001)
            losses, gradients = [], []
            loader = torch.Generator().manual_seed(123)
            session = c.EncoderCacheSession(self.encoder, store, mode="engineering_replay")
            from contextlib import nullcontext
            with session if cached else nullcontext():
                for indices in ((3, 0), (1, 2)):
                    # Keep preprocessing and its worker/main RNG consumption.
                    random.random(); np.random.random(); torch.rand(1, generator=loader)
                    images = self.images[list(indices)].clone()
                    with session.frames([self.frames[i] for i in indices]) if cached else nullcontext():
                        features = self.encoder(images)
                    prefix = torch.randint(2, (1,)).item()
                    loss = predictor(features.flatten(1)).square().mean() * (prefix + 1)
                    loss.backward()
                    gradients.append({k: v.grad.clone() for k, v in predictor.named_parameters()})
                    losses.append(float(loss.detach())); optimizer.step(); optimizer.zero_grad()
                # Native-style validation has its own random action draw.
                validation = torch.rand(3)
            return {"losses": losses, "parameters": predictor.state_dict(), "gradients": gradients,
                "optimizer": optimizer.state_dict(), "scaler": {}, "scheduler": 2,
                "logical_rngs": [torch.get_rng_state()], "loader_rng": loader.get_state(),
                "validation_events": 1, "validation": validation, "updates": 2}
        before = c.rng_snapshot()
        result = c.paired_update_check(lambda: run(False), lambda: run(True))
        self.assertEqual(result["complete_updates"], 2)
        c.assert_bitwise(before, c.rng_snapshot())

    def test_second_run_cannot_mutate_first_run_evidence_to_hide_failure(self):
        shared = torch.ones(1)
        def result():
            return {"losses": [], "parameters": shared, "gradients": {}, "optimizer": {}, "scaler": {},
                "scheduler": 2, "logical_rngs": [], "loader_rng": [], "validation_events": 1, "updates": 2}
        def changed():
            shared.add_(1)
            return result()
        with self.assertRaises(ValueError):
            c.paired_update_check(result, changed)


if __name__ == "__main__":
    unittest.main()
