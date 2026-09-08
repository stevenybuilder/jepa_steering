"""CPU-only contract/failure/timer fixtures, not real GPU numerical evidence."""
import ast
import copy
import inspect
import json
import os
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from offline_study import robotics_training_gpu_check as g
from offline_study import robotics_training_pilot as p

VENDOR = Path(__file__).resolve().parents[1] / "vendor/jepa-wms"
READBACK = Path(__file__).resolve().parents[1] / "artifacts/offline_study/robotics-training-batch-20260908-v1/readback/evidence"
UUID = "01234567-89ab-cdef-0123-456789abcdef"


class TinyModel(torch.nn.Module):
    """Private complete-state/clone checker fixture, never a native model."""
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.25))
        self.register_buffer("buffer", torch.tensor([3.]))
        self.optimizer = torch.optim.AdamW(self.parameters())
        self.scaler = SimpleNamespace(state_dict=lambda: {"scale": 65536., "updates": 0})


class TinyRNG:
    def __init__(self):
        self.state = {"cpu": torch.arange(3, dtype=torch.uint8),
                      "numpy": np.random.RandomState(13).get_state()}

    def state_dict(self): return copy.deepcopy(self.state)
    def _capture(self): return {"ambient_fixture": torch.arange(4, dtype=torch.uint8)}


class FakeEvent:
    """Only a CPU timer-lifecycle test double, not a GPU timing measurement."""
    def __init__(self, enable_timing):
        if enable_timing is not True: raise AssertionError("Timing must be enabled")
        self.recorded = False

    def record(self): self.recorded = True
    def synchronize(self):
        if not self.recorded: raise AssertionError("Unrecorded event")
    def elapsed_time(self, other):
        if not self.recorded or not other.recorded: raise AssertionError("Missing interval")
        return 1250.0


class NativeGpuCheckTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_fixed_scope_no_count_tolerance_model_or_fixture_injection(self):
        self.assertEqual(set(inspect.signature(g.run_check).parameters),
                         {"vendor", "input_proof", "output", "expected_device_uuid"})
        self.assertEqual((g.TIMED_UPDATES, g.MODEL_SEED, g.LIMIT_SECONDS), (4, 234, 1200))
        self.assertIn("not_native_loader_lpips_history", g.RNG_SCOPE)

    def test_actual_completed_compact_receipt_pins_when_available(self):
        if not READBACK.is_dir():
            self.skipTest("Receiving suite need not stage local archival readback")
        self.assertEqual(g.sha256(READBACK / "report.json"), g.INPUT_REPORT_SHA256)
        self.assertEqual(g.sha256(READBACK / "protocol.json"), g.INPUT_PROTOCOL_SHA256)
        protocol = json.loads((READBACK / "protocol.json").read_text())
        self.assertEqual(protocol["batch_file_sha256"], g.INPUT_BATCH_FILE_SHA256)
        self.assertEqual(protocol["batch_sha256"], g.INPUT_BATCH_TENSOR_SHA256)

    def test_synthetic_receipts_and_missing_batch_fail_before_cuda(self):
        proof = self.root / "fake_proof"
        proof.mkdir()
        (proof / "report.json").write_text(json.dumps({"status": "native32_training_batch_input_parity_verified"}))
        with patch.object(g.torch.cuda, "is_available", side_effect=AssertionError("No CUDA query")):
            with self.assertRaisesRegex(ValueError, "Pinned receiving input file"):
                g._input_files(proof)

    def test_failed_or_symlinked_receipt_refused(self):
        proof = self.root / "failed"
        proof.mkdir(); (proof / "FAILED.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "completed real"):
            g._input_files(proof)
        link = self.root / "link"
        link.symlink_to(proof)
        with self.assertRaisesRegex(ValueError, "completed real"):
            g._input_files(link)

    def test_no_unassigned_or_multi_visible_device(self):
        with patch.object(g.torch.cuda, "is_available", side_effect=AssertionError("No GPU query")):
            for visibility in ("", "0,1", " 0"):
                with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": visibility}):
                    with self.assertRaisesRegex(ValueError, "explicitly assigned"):
                        g._device_identity(UUID)

    def test_device_uuid_normalization_and_actual_cuda_required(self):
        properties = SimpleNamespace(uuid=UUID, name="CPU timer fixture, not GPU evidence", total_memory=1)
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-" + UUID}), \
                patch.object(g.torch.cuda, "is_available", return_value=True), \
                patch.object(g.torch.cuda, "device_count", return_value=1), \
                patch.object(g.torch.cuda, "get_device_properties", return_value=properties):
            self.assertEqual(g._device_identity("GPU-" + UUID)["device_uuid"], UUID)
            with self.assertRaisesRegex(ValueError, "reserved device UUID"):
                g._device_identity("11234567-89ab-cdef-0123-456789abcdef")
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}), \
                patch.object(g.torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "no CPU fallback"):
                g._device_identity(UUID)

    def test_fingerprint_scalar_bf16_layout_and_signed_zero(self):
        for dtype in (torch.float32, torch.bfloat16, torch.int64):
            value = torch.tensor(2, dtype=dtype)
            self.assertEqual(g._fingerprint(value), g._fingerprint(value.clone()))
        zero = torch.tensor(0.)
        self.assertNotEqual(g._fingerprint(zero), g._fingerprint(-zero))
        x = torch.ones(2, 2)
        self.assertNotEqual(g._fingerprint(x), g._fingerprint(x.T))
        self.assertNotEqual(g._fingerprint({1: "a"}), g._fingerprint({"1": "a"}))
        with self.assertRaises(ValueError): g._fingerprint(float("nan"))

    def test_complete_state_detects_optimizer_buffers_modes_gradients_rng(self):
        model, rng = TinyModel(), TinyRNG()
        scheduler = SimpleNamespace(_step=0, optimizer=model.optimizer)
        wd = SimpleNamespace(_step=0, optimizer=model.optimizer)
        before = g._complete_state(model, scheduler, wd, rng)
        model.buffer.add_(1)
        self.assertNotEqual(before["model"], g._complete_state(model, scheduler, wd, rng)["model"])
        model.weight.grad = torch.ones_like(model.weight)
        self.assertNotEqual(before["gradients"], g._complete_state(model, scheduler, wd, rng)["gradients"])
        model.optimizer.step(); model.optimizer.zero_grad()
        after = g._complete_state(model, scheduler, wd, rng)
        self.assertNotEqual(before["optimizer"], after["optimizer"])
        model.eval(); scheduler._step += 1; rng.state["cpu"][0] = 5
        after = g._complete_state(model, scheduler, wd, rng)
        self.assertNotEqual(before["module_modes"], after["module_modes"])
        self.assertNotEqual(before["scheduler"], after["scheduler"])
        self.assertNotEqual(before["logical_rngs"], after["logical_rngs"])

    def test_clone_only_wrapper_rejects_live_mutation_without_trusting_flag(self):
        model, rng = TinyModel(), TinyRNG()
        scheduler = SimpleNamespace(_step=0, optimizer=model.optimizer)
        wd = SimpleNamespace(_step=0, optimizer=model.optimizer)
        snapshots = {}
        with patch.object(g.core, "verify_native_update", return_value={"live_model_unchanged": True}):
            g._verify_clone_only({}, model, scheduler, wd, None, rng, "fixture", snapshots)
        self.assertEqual(snapshots["before"], snapshots["after"])

        def changed(*a, **kw):
            scheduler._step += 1
            return {"live_model_unchanged": True}

        with patch.object(g.core, "verify_native_update", side_effect=changed):
            with self.assertRaises(ValueError):
                g._verify_clone_only({}, model, scheduler, wd, None, rng, "fixture", {})

    def test_clone_only_failure_preserved_and_timeout_does_not_submit_snapshot_work(self):
        model, rng = TinyModel(), TinyRNG()
        scheduler = SimpleNamespace(_step=0, optimizer=model.optimizer)
        wd = SimpleNamespace(_step=0, optimizer=model.optimizer)
        snapshots = {}
        with patch.object(g.core, "verify_native_update", side_effect=ValueError("numerical fixture failure")):
            with self.assertRaisesRegex(ValueError, "numerical fixture failure"):
                g._verify_clone_only({}, model, scheduler, wd, None, rng, "fixture", snapshots)
        self.assertTrue(snapshots["checked_after_failure"])
        snapshots = {}
        with patch.object(g.core, "verify_native_update", side_effect=TimeoutError("bounded")), \
                patch.object(g, "_complete_state", return_value={}) as capture:
            with self.assertRaises(TimeoutError):
                g._verify_clone_only({}, model, scheduler, wd, None, rng, "fixture", snapshots)
        self.assertEqual(capture.call_count, 1)
        self.assertFalse(snapshots["checked_after_failure"])

    def test_timers_call_originals_once_separate_gate_and_restore(self):
        events = []
        def gate(*a, **k): events.append("gate"); return "checked"
        def update(*a, **k): events.append("update"); return {"unchanged": True}
        with patch.object(g.core, "_input_gate", gate), patch.object(g.factory, "_input_gate", gate), \
                patch.object(g.core, "_drive_update", update), \
                patch.object(g.torch.cuda, "Event", FakeEvent), \
                patch.object(g.torch.cuda, "synchronize") as sync, \
                patch.object(g.torch.cuda, "max_memory_allocated", return_value=10), \
                patch.object(g.torch.cuda, "max_memory_reserved", return_value=20):
            with g._Timings() as timer:
                timer.phase = "fixture_timing_only"
                self.assertEqual(g.factory._input_gate(), "checked")
                self.assertEqual(g.core._drive_update(mean_after=True), {"unchanged": True})
            self.assertIs(g.core._input_gate, gate)
            self.assertIs(g.factory._input_gate, gate)
            self.assertIs(g.core._drive_update, update)
            sync.assert_called_once_with(0)
        self.assertEqual(events, ["gate", "update"])
        self.assertEqual(len(timer.gates), 1)
        self.assertEqual(timer.updates[0]["gpu_interval_seconds"], 1.25)
        self.assertTrue(timer.updates[0]["mean_after"])

    def test_failed_timer_never_synchronizes_failed_gpu_work_or_retries(self):
        def gate(*a, **k): raise ValueError("input fixture failure")
        def update(*a, **k): raise TimeoutError("GPU fixture failure")
        with patch.object(g.core, "_input_gate", gate), patch.object(g.factory, "_input_gate", gate), \
                patch.object(g.core, "_drive_update", update), \
                patch.object(g.torch.cuda, "Event", FakeEvent), \
                patch.object(g.torch.cuda, "synchronize") as sync:
            timer = g._Timings()
            with self.assertRaises(TimeoutError):
                with timer:
                    with self.assertRaises(ValueError): g.core._input_gate()
                    g.core._drive_update()
            self.assertIs(g.core._drive_update, update)
            self.assertIs(g.factory._input_gate, gate)
            sync.assert_called_once_with(0)  # before only, never after failure
        self.assertFalse(timer.gates[0]["completed"])
        self.assertFalse(timer.updates[0]["completed"])
        self.assertIsNone(timer.updates[0]["gpu_interval_seconds"])

    def test_timer_refuses_existing_alias_or_other_python_threads(self):
        with patch.object(g.factory, "_input_gate", lambda *a: None):
            with self.assertRaisesRegex(ValueError, "guard differs"):
                with g._Timings(): pass
        with patch.object(g.threading, "active_count", return_value=2):
            with self.assertRaisesRegex(ValueError, "exclusive single"):
                with g._Timings(): pass

    def test_timeout_handlers_restored_no_real_signal_sent(self):
        callbacks = {}
        def install(signum, handler): callbacks[signum] = handler
        with patch.object(g.signal, "getitimer", return_value=(0., 0.)), \
                patch.object(g.signal, "getsignal", return_value="previous_fixture_handler"), \
                patch.object(g.signal, "signal", side_effect=install), \
                patch.object(g.signal, "setitimer") as timer:
            with self.assertRaises(TimeoutError):
                with g._bounded_process():
                    callbacks[signal.SIGALRM](signal.SIGALRM, None)
            self.assertEqual(callbacks[signal.SIGALRM], "previous_fixture_handler")
            self.assertEqual(callbacks[signal.SIGTERM], "previous_fixture_handler")
            self.assertEqual(timer.call_args_list[-1].args, (signal.ITIMER_REAL, 0))

    def test_existing_alarm_not_repurposed(self):
        with patch.object(g.signal, "getitimer", return_value=(1., 0.)), \
                patch.object(g.signal, "signal") as install:
            with self.assertRaisesRegex(ValueError, "existing process timeout"):
                with g._bounded_process(): pass
            install.assert_not_called()

    def test_native_benchmark_policy_matches_both_pinned_upstream_assignments(self):
        path = VENDOR / "app/vjepa_wm/train.py"
        assignments = [n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.Assign)
            and any(isinstance(target, ast.Attribute) and ast.unparse(target) == "torch.backends.cudnn.benchmark"
                    for target in n.targets)]
        self.assertEqual(sorted(n.lineno for n in assignments), [68, 236])
        self.assertTrue(all(isinstance(n.value, ast.Constant) and n.value.value is True for n in assignments))
        self.assertEqual(g.BACKEND_POLICY, {"cudnn_benchmark": True,
            "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False})

    def test_constructor_sees_native_backends_and_failure_restores_them_cpu_only(self):
        original = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                    torch.backends.cudnn.benchmark)
        calls = []

        def stop_before_any_model(*args, **kwargs):
            calls.append({"cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32})
            raise ValueError("CPU fixture stops before model creation")

        gate = lambda *a, **k: "fixture_gate_not_evidence"
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = False
            with patch.object(g.torch, "get_num_threads", return_value=1), \
                    patch.object(g.torch, "get_num_interop_threads", return_value=1), \
                    patch.object(g, "_input_files", return_value={"fixture_only": True}), \
                    patch.object(g.torch, "load", return_value=None), \
                    patch.object(g.core, "_input_gate", gate), patch.object(g.factory, "_input_gate", gate), \
                    patch.object(g, "_device_identity", return_value={"fixture_only": True}), \
                    patch.object(g.torch.cuda, "reset_peak_memory_stats"), \
                    patch.object(g.factory, "build_receiving_model", side_effect=stop_before_any_model):
                with self.assertRaisesRegex(ValueError, "CPU fixture stops"):
                    g.run_check(VENDOR, self.root / "fixture_input", self.root / "run", expected_device_uuid=UUID)
            self.assertEqual(calls, [g.BACKEND_POLICY])
            self.assertEqual((torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                              torch.backends.cudnn.benchmark), (True, True, False))
            protocol = json.loads((self.root / "run/protocol.json").read_text())
            self.assertEqual(protocol["backend_policy"], g.BACKEND_POLICY)
            self.assertFalse((self.root / "run/DONE.json").exists())
        finally:
            (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
             torch.backends.cudnn.benchmark) = original

    def test_run_failure_receipt_no_gpu_no_retry_and_no_output_overwrite(self):
        output = self.root / "run"
        original_core, original_factory = g.core._input_gate, g.factory._input_gate
        with patch.object(g.torch, "get_num_threads", return_value=1), \
                patch.object(g.torch, "get_num_interop_threads", return_value=1), \
                patch.object(g.torch.cuda, "is_available", side_effect=AssertionError("No CUDA query")):
            with self.assertRaisesRegex(ValueError, "Pinned receiving input file"):
                g.run_check(VENDOR, self.root / "absent", output, expected_device_uuid=UUID)
        failed = json.loads((output / "FAILED.json").read_text())
        self.assertEqual(failed["completed_timed_updates"], 0)
        self.assertFalse(failed["automatic_retry"])
        self.assertFalse(failed["history_launch_authorized"])
        self.assertFalse((output / "DONE.json").exists())
        self.assertIs(g.core._input_gate, original_core)
        self.assertIs(g.factory._input_gate, original_factory)
        with self.assertRaises(FileExistsError):
            g.run_check(VENDOR, self.root / "absent", output, expected_device_uuid=UUID)

    def test_exact_four_update_lifecycle_fixed_batch_and_proof(self):
        scheduler, wd = SimpleNamespace(_step=0), SimpleNamespace(_step=0)
        batch, proof, completed, timings = object(), {"fixture_not_gpu_evidence": True}, [], SimpleNamespace()
        calls = []

        def update(*args, **kwargs):
            self.assertIs(args[4], batch)
            self.assertIs(kwargs["numerical_proof"], proof)
            self.assertEqual(kwargs["input_proof"], "fixture_input")
            scheduler._step += 1; wd._step += 1
            calls.append(timings.phase)
            return {"loss": 1.0}

        with patch.object(g.core, "accumulated_update", side_effect=update):
            g._four_updates({}, None, scheduler, wd, batch, None, "fixture_input", proof,
                            timings, completed, self.root)
        self.assertEqual(calls, [f"timed_update_{i}" for i in range(1, 5)])
        self.assertEqual([row["update"] for row in completed], [1, 2, 3, 4])
        progress = json.loads((self.root / "progress.json").read_text())
        self.assertEqual(progress["history_updates"], 0)
        self.assertFalse((self.root / "DONE.json").exists())
        with self.assertRaisesRegex(ValueError, "unchanged fresh model"):
            g._four_updates({}, None, scheduler, wd, batch, None, "fixture_input", proof,
                            timings, completed, self.root)

    def test_four_update_failure_does_not_retry_or_continue(self):
        scheduler, wd = SimpleNamespace(_step=0), SimpleNamespace(_step=0)
        completed, calls = [], []

        def update(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2: raise ValueError("fixed engineering failure")
            scheduler._step += 1; wd._step += 1
            return {"loss": 1.0}

        with patch.object(g.core, "accumulated_update", side_effect=update):
            with self.assertRaisesRegex(ValueError, "fixed engineering failure"):
                g._four_updates({}, None, scheduler, wd, None, None, "fixture", {},
                                SimpleNamespace(), completed, self.root)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(completed), 1)
        self.assertEqual(json.loads((self.root / "progress.json").read_text())["completed_timed_updates"], 1)
        self.assertFalse((self.root / "DONE.json").exists())


if __name__ == "__main__":
    unittest.main()
