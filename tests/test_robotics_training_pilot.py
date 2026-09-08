"""CPU mechanics/source tests only; never receiving CUDA numerical evidence."""
import copy
import json
import random
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from offline_study import robotics_training_pilot as p

VENDOR = Path(__file__).resolve().parents[1] / "vendor/jepa-wms"


class FakeCudaState:
    """Separate CPU generator stands in for CUDA state only in private tests."""
    def __init__(self):
        self.generator = torch.Generator().manual_seed(41)

    def get(self):
        return self.generator.get_state().clone()

    def set(self, state):
        self.generator.set_state(state)

    def draw(self):
        return torch.rand((), generator=self.generator)


def streams():
    backend = FakeCudaState()
    states = []
    for rank in range(32):
        states.append({"cpu": torch.Generator().manual_seed(70 + rank).get_state(),
            "cuda": torch.Generator().manual_seed(140 + rank).get_state(),
            "python": random.Random(210 + rank).getstate(),
            "numpy": np.random.RandomState(280 + rank).get_state()})
    return p.LogicalRNGStreams(states, _backend=backend)


class TinyModel(torch.nn.Module):
    def __init__(self, backend):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.25))
        self.predictor = torch.nn.Identity()
        self.heads = {}
        self.cuda_test_backend = backend
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=.0005, betas=(.9, .999), eps=1e-8)
        self.scaler = TinyScaler()
        self.updates = 0

    def encode(self, obs, action):
        # Exercise every logical RNG, not just CPU prefix sampling.
        offset = torch.rand(()) + self.cuda_test_backend.draw() + random.random() + float(np.random.rand())
        return obs["visual"] + offset, obs["proprio"], action

    def forward_pred(self, visual, action, proprio):
        return visual * self.weight, None, proprio * self.weight

    def compute_loss(self, pv, pp, visual, proprio, shift):
        return {"loss": (pv - visual).square().mean() + (pp - proprio).square().mean()}

    def rollout(self, **kwargs):
        loss = kwargs["pred_video_features"][:, kwargs["t"]].square().mean()
        return {"loss": loss[None]}, loss, None, None

    def backward(self, loss):
        self.scaler.scale(loss).backward()

    def optimization_step(self):
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        self.updates += 1
        return None, None


class TinyScaler:
    """Local Torch lacks CPU GradScaler; this is explicitly only a test double."""
    def __init__(self): self.updates = 0
    def scale(self, loss): return loss * 65536
    def unscale_(self, optimizer):
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None: param.grad.div_(65536)
    def step(self, optimizer): optimizer.step()
    def update(self): self.updates += 1
    def state_dict(self): return {"scale": 65536, "updates": self.updates}


def schedules(model):
    namespace = {}
    path = VENDOR / "src/utils/schedulers.py"
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    lr = namespace["WarmupCosineSchedule"](model.optimizer, warmup_steps=0,
        start_lr=.0005, ref_lr=.0005, T_max=100, final_lr=.0005)
    wd = namespace["CosineWDSchedule"](model.optimizer, ref_wd=1e-7, T_max=100, final_wd=1e-6)
    return lr, wd


def batches():
    return [({"visual": torch.arange(96.).reshape(8, 4, 3) / 100 + rank / 64,
               "proprio": torch.ones(8, 4, 2)}, torch.ones(8, 4, 2),
              torch.zeros(8, 4, 2), torch.zeros(8, 4)) for rank in range(32)]


class Native32Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (VENDOR / "app/vjepa_wm/train.py").exists():
            raise RuntimeError("Pinned upstream is required for native32 source tests")
        cls.contracts = {task: p.native_contract(VENDOR, task) for task in p.CONFIGS}

    def test_actual_caller_policy_not_assumed_shared_shuffle(self):
        for task, wanted_shuffle in (("metaworld", True), ("pusht", False)):
            c = self.contracts[task]
            train, val, kwargs = p.native_loader_probe(VENDOR, task)
            self.assertEqual(train.num_workers, c["config"]["data"]["loader"]["num_workers"])
            self.assertEqual(train.sampler.shuffle, wanted_shuffle)
            self.assertEqual(val.sampler.shuffle, wanted_shuffle)
            self.assertEqual(c["sampler"]["global_batch"], 256)
            self.assertEqual((train.sampler.seed, val.sampler.seed), (0, 0))
            self.assertEqual(kwargs["random_seed"], 234)
            self.assertEqual(kwargs["process_actions"], "concat")
            self.assertEqual(kwargs["num_hist"] + kwargs["num_pred"], 4)
            if task == "pusht": self.assertTrue(kwargs["with_velocity"])

    def test_training_counts_from_actual_loaders(self):
        for task, n, v, updates in (("metaworld", 907200, 12600, 3543), ("pusht", 1981721, 1695, 7741)):
            train, val, _ = p.native_loader_probe(VENDOR, task, n, v)
            c = self.contracts[task]
            self.assertEqual(len(train), updates)
            self.assertEqual(len(p.Native32BatchSampler(c, n)), updates)
            self.assertEqual(updates // c["config"]["meta"]["light_eval_freq"], 11 if task == "metaworld" else 25)
            self.assertEqual(c["config"]["optimization"]["transition_model"]["num_epochs"], 50)

    def test_every_rank_train_order_and_epoch_match_actual_loader(self):
        for task, c in self.contracts.items():
            for epoch in (0, 1, 4):
                actual = list(p.Native32BatchSampler(c, 1031, epoch=epoch))
                for rank in range(32):
                    native, _, _ = p.native_loader_probe(VENDOR, task, 1031, 129, rank)
                    native.sampler.set_epoch(epoch)
                    self.assertEqual([list(x[rank]) for x in actual], list(native.batch_sampler))

    def test_validation_tail_and_native_padding_not_independent_samples(self):
        for task, n, count, last in (("metaworld", 12600, 99, 2), ("pusht", 1695, 14, 1)):
            c = self.contracts[task]
            batches_ = list(p.Native32BatchSampler(c, n, validation=True))
            self.assertEqual(len(batches_), count)
            self.assertEqual([len(x) for x in batches_[-1]], [last] * 32)
            for rank in (0, 15, 31):
                _, native, _ = p.native_loader_probe(VENDOR, task, 257, n, rank)
                self.assertEqual([list(x[rank]) for x in batches_], list(native.batch_sampler))
            with self.assertRaises(ValueError): p.Native32BatchSampler(c, n, epoch=1, validation=True)

    def test_code_paper_episode_distinction_is_not_gpu_multiplier(self):
        self.assertEqual(self.contracts["metaworld"]["config"]["evals"]["eval_episodes"], 48)
        self.assertEqual(self.contracts["pusht"]["config"]["evals"]["eval_episodes"], 96)

    def test_changed_task_or_mutated_contract_fails(self):
        with self.assertRaises(ValueError): p.native_contract(VENDOR, "pointmaze")
        changed = copy.deepcopy(self.contracts["pusht"])
        changed["sampler"]["train_shuffle"] = True
        with self.assertRaises(ValueError): p.Native32BatchSampler(changed, 257)
        with self.assertRaises(ValueError): p.Native32BatchSampler(self.contracts["pusht"], 0)

    def test_model_seed_does_not_change_native_data_or_sampler_seed(self):
        for seed in (234, 235, 236):
            contract = p.native_contract(VENDOR, "pusht", model_seed=seed)
            self.assertEqual(contract["model_seed"], seed)
            self.assertEqual(contract["sampler"]["data_seed"], 234)
            self.assertEqual(contract["sampler"]["sampler_seed"], 0)
            self.assertFalse(contract["authors_exact_seed_triplet_known"])
        with self.assertRaises(ValueError): p.native_contract(VENDOR, "pusht", model_seed=237)

    def test_changed_native_source_bytes_fail_before_loader_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = next(iter(p.SOURCE_HASHES))
            target = root / first
            target.parent.mkdir(parents=True)
            target.write_text("# not the pinned native training implementation\n")
            with self.assertRaisesRegex(ValueError, "Pinned native source/config changed"):
                p.native_contract(root, "pusht")

    def test_rank_rng_independence_and_ambient_restoration(self):
        rng = streams(); ambient = rng._capture(); original = rng.state_dict()
        with rng.rank(3):
            torch.rand(4); rng.backend.draw(); random.random(); np.random.rand()
        p.assert_same(ambient, rng._capture())
        for rank in range(32):
            if rank != 3: p.assert_same(original["states"][rank], rng.states[rank])
        for key in ("cpu", "cuda"):
            self.assertFalse(torch.equal(original["states"][3][key], rng.states[3][key]))
        self.assertNotEqual(original["states"][3]["python"], rng.states[3]["python"])
        self.assertFalse(np.array_equal(original["states"][3]["numpy"][1], rng.states[3]["numpy"][1]))

    def test_failed_rank_restores_ambient_and_is_not_retryable(self):
        rng = streams(); before = rng._capture()
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with rng.rank(4):
                torch.rand(4); rng.backend.draw(); raise RuntimeError("intentional")
        p.assert_same(before, rng._capture())
        self.assertTrue(rng.poisoned)
        with self.assertRaises(ValueError):
            with rng.rank(0): pass
        with self.assertRaises(ValueError): rng.clone()

    def test_require_32_complete_explicit_rng_states(self):
        states = streams().states
        with self.assertRaises(ValueError): p.LogicalRNGStreams(states[:16])
        changed = copy.deepcopy(states); del changed[2]["cuda"]
        with self.assertRaises(ValueError): p.LogicalRNGStreams(changed)

    def test_actual_native_full_update_cpu_control_flow_all_grad_rng_state(self):
        # This is a CPU unit test, not a receiving-model numerical certificate.
        for task, c in self.contracts.items():
            a, b = streams(), streams()
            reference, actual = TinyModel(a.backend), TinyModel(b.backend)
            rs, rw = schedules(reference); actual_s, actual_w = schedules(actual)
            ambient = a._capture()
            with patch("torch.amp.autocast", side_effect=lambda *args, **kwargs: nullcontext()):
                for update in range(2):
                    left = p._drive_update(c, reference, rs, rw, batches(), a, mean_after=True)
                    right = p._drive_update(c, actual, actual_s, actual_w, batches(), b)
                    p.assert_same(left, right)
                    p.assert_same(reference.state_dict(), actual.state_dict())
                    p.assert_same(reference.optimizer.state_dict(), actual.optimizer.state_dict())
                    p.assert_same(reference.scaler.state_dict(), actual.scaler.state_dict())
                    p.assert_same(a.state_dict(), b.state_dict())
                    self.assertEqual((rs._step, rw._step, actual_s._step, actual_w._step), (update + 1,) * 4)
                    self.assertEqual((reference.updates, actual.updates), (update + 1,) * 2)
            p.assert_same(ambient, a._capture())
            self.assertNotIn("backward", actual.__dict__)
            self.assertNotIn("optimization_step", actual.__dict__)

    def test_incomplete_batch_has_no_optimizer_update(self):
        rng = streams(); model = TinyModel(rng.backend); s, w = schedules(model)
        with self.assertRaises(ValueError):
            p._drive_update(self.contracts["pusht"], model, s, w, batches()[:16], rng)
        self.assertEqual((model.updates, s._step, w._step), (0, 0, 0))

    def test_nonfinite_loss_poisons_stream_without_skipped_update(self):
        rng = streams(); model = TinyModel(rng.backend); s, w = schedules(model)
        data = batches(); data[0][0]["visual"].fill_(float("nan")); before = rng._capture()
        with patch("torch.amp.autocast", side_effect=lambda *args, **kwargs: nullcontext()):
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                p._drive_update(self.contracts["pusht"], model, s, w, data, rng)
        self.assertEqual(model.updates, 0)
        self.assertTrue(rng.poisoned)
        p.assert_same(before, rng._capture())
        self.assertNotIn("backward", model.__dict__)

    def test_public_entrypoints_require_inputs_before_cuda_or_updates(self):
        rng = streams(); model = TinyModel(rng.backend); s, w = schedules(model)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                p.verify_native_update(self.contracts["pusht"], model, s, w, batches()[0], rng, input_proof=directory)
        self.assertEqual(model.updates, 0)
        with self.assertRaisesRegex(ValueError, "actual CUDA"):
            p._runtime(model, rng)

    def test_raw_bytes_alone_never_claim_native_tensor_or_access_parity(self):
        from offline_study.robotics_training_inputs import DATA_REVISION, MANIFEST_HASHES, COUNTS
        def receipt(root, report, protocol):
            root.mkdir()
            (root / "protocol.json").write_text(json.dumps(protocol))
            report = {**report, "protocol_sha256": p.sha256(root / "protocol.json")}
            (root / "report.json").write_text(json.dumps(report))
            digest = p.sha256(root / "report.json")
            (root / "DONE.json").write_text(json.dumps({"report_sha256": digest}))
            return digest
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); raw = root / "raw"; native = root / "native"
            raw_report = {"status": "complete_robotics_raw_inputs_verified", "task": "pusht",
                "data_revision": DATA_REVISION, "raw_manifest_sha256": MANIFEST_HASHES["pusht"],
                "training_rows": COUNTS["pusht"][0], "validation_rows": COUNTS["pusht"][1],
                "data_root": "/explicit/test/data/root", "training_or_validation_authorized": False,
                "native_reader_parity_established": False, "model_or_data_loader_initialized": False}
            raw_hash = receipt(raw, raw_report, {"role": "CPU_test_fixture_not_real_raw_verification"})
            receipt(native, raw_report, {"raw_inputs_receipt": str(raw),
                "raw_input_report_sha256": raw_hash, "data_root": raw_report["data_root"]})
            with self.assertRaisesRegex(ValueError, "native training-only input parity"):
                p._input_gate(self.contracts["pusht"], batches()[0], native)

    def test_wrong_raw_proof_binding_is_rejected_before_tensor_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "protocol.json").write_text(json.dumps({"raw_inputs_receipt": str(root / "absent")}))
            (root / "report.json").write_text(json.dumps({"protocol_sha256": p.sha256(root / "protocol.json")}))
            (root / "DONE.json").write_text(json.dumps({"report_sha256": p.sha256(root / "report.json")}))
            with self.assertRaises(FileNotFoundError):
                p._input_gate(self.contracts["pusht"], batches()[0], root)

    def test_bf16_batch_hash_binds_bytes_dtype_shape_and_modalities(self):
        batch = batches()[0]
        before = p._batch_digest(batch)
        cloned = copy.deepcopy(batch)
        self.assertEqual(before, p._batch_digest(cloned))
        cloned[0]["proprio"][0, 0, 0] += 1
        self.assertNotEqual(before, p._batch_digest(cloned))
        bf16 = ({k: v.to(torch.bfloat16) for k, v in batch[0].items()},
                *[v.to(torch.bfloat16) for v in batch[1:]])
        self.assertNotEqual(before, p._batch_digest(bf16))

    def test_no_global_batch_as_single_forward(self):
        rng = streams(); model = TinyModel(rng.backend); s, w = schedules(model)
        with patch("torch.amp.autocast", side_effect=lambda *args, **kwargs: nullcontext()), patch.object(model, "encode", wraps=model.encode) as encode:
            p._drive_update(self.contracts["pusht"], model, s, w, batches(), rng)
        self.assertEqual(encode.call_count, 32)
        self.assertEqual({len(call.args[1]) for call in encode.call_args_list}, {8})


if __name__ == "__main__":
    unittest.main()
