"""One disposable Push-T native32 arithmetic check, never a training history.

Fixed verified first256 training batch; one bitwise reference/accumulation proof,
then exactly four timed updates of the disposable model on that same batch. The
32 logical streams duplicate the returned post-constructor RNG *for arithmetic
engineering only*. No native persistent-worker/LPIPS initialization, validation,
resume, physical-DDP reduction order, or trained-seed replication is established.

Receiving operations must first reserve one empty GPU and use an outer timeout:
timeout --signal=TERM --kill-after=5s 1210s python -m
offline_study.robotics_training_gpu_check ...
This module launches nothing remotely and must not delay an existing GPU owner.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import sys
import threading
import time
import uuid

import numpy as np
import torch

from . import robotics_training_model as factory
from . import robotics_training_pilot as core
from .droid_native import verified_report
from .protocol import sha256, write_json

INPUT_REPORT_SHA256 = "4b14268d293bb708ad8b96c074624a4d52c4b70ca57988c9858273a8eaa1b7aa"
INPUT_PROTOCOL_SHA256 = "a5975c987b5cd614de7b56e1943f3927f0365ccdd0175dd5e62c4a36fa01e0ef"
INPUT_BATCH_FILE_SHA256 = "2ada0bc241424e26899300b77da849f38c9c056ecab14295c6c99be27526bcc4"
INPUT_BATCH_TENSOR_SHA256 = "db3bd93a282349f4657c7c288560fa01d67501bbbe284dbb79fd37f5cae96df7"
TIMED_UPDATES = 4
MODEL_SEED = 234
LIMIT_SECONDS = 1200
RNG_SCOPE = "duplicated_post_constructor_state_for_arithmetic_only_not_native_loader_lpips_history"
BACKEND_POLICY = {"cudnn_benchmark": True, "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False}


def _input_files(root):
    """Bind the already completed real receipt before any CUDA query/load."""
    root = Path(root)
    if not root.is_absolute() or root.is_symlink() or (root / "FAILED.json").exists():
        raise ValueError("Explicit completed real Push-T input proof required")
    expected = {"report.json": INPUT_REPORT_SHA256, "protocol.json": INPUT_PROTOCOL_SHA256,
                "batch.pt": INPUT_BATCH_FILE_SHA256}
    for name, digest in expected.items():
        path = root / name
        if path.is_symlink() or not path.is_file() or sha256(path) != digest:
            raise ValueError("Pinned receiving input file missing or changed: " + name)
    report, digest = verified_report(root)
    protocol = json.loads((root / "protocol.json").read_text())
    if (digest != INPUT_REPORT_SHA256 or protocol.get("batch_sha256") != INPUT_BATCH_TENSOR_SHA256 or
            protocol.get("batch_file_sha256") != INPUT_BATCH_FILE_SHA256 or
            report.get("status") != "native32_training_batch_input_parity_verified" or
            report.get("validation_or_confirmation_access") is not False):
        raise ValueError("Not the verified first Push-T training-only batch")
    return {"root": str(root), "files_sha256": expected,
            "batch_tensor_sha256": INPUT_BATCH_TENSOR_SHA256}


def _device_identity(expected_uuid):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible or visible.strip() != visible:
        raise ValueError("Expose exactly one explicitly assigned receiving CUDA device")
    wanted = str(uuid.UUID(expected_uuid.removeprefix("GPU-")))
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Actual single CUDA GPU required; no CPU fallback")
    props = torch.cuda.get_device_properties(0)
    actual_uuid = str(props.uuid)
    if str(uuid.UUID(actual_uuid.removeprefix("GPU-"))) != wanted:
        raise ValueError("Visible GPU differs from the reserved device UUID")
    return {"cuda_visible_devices": visible, "logical_device": 0,
            "device_uuid": actual_uuid, "expected_device_uuid": expected_uuid,
            "name": props.name, "total_memory_bytes": props.total_memory,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "python": sys.version, "visible_device_count": 1}


def _evidence(value):
    """Compact exact state bytes, including non-parameter state and RNG."""
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        return {"tensor_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                "shape": list(value.shape), "stride": list(value.stride()),
                "dtype": str(value.dtype), "device": str(value.device)}
    if isinstance(value, np.ndarray):
        return {"array_sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
                "shape": list(value.shape), "dtype": value.dtype.str}
    if isinstance(value, dict):
        return [[type(key).__name__, str(key), _evidence(item)] for key, item in value.items()]
    if isinstance(value, (tuple, list)):
        return [type(value).__name__, [_evidence(item) for item in value]]
    if isinstance(value, np.generic):
        return _evidence(value.item())
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return {"float_hex": value.hex()}
    raise ValueError("Unsupported or nonfinite complete-state evidence")


def _fingerprint(value):
    encoded = json.dumps(_evidence(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _complete_state(model, scheduler, wd, rng):
    return {"model": _fingerprint(model.state_dict()),
            "gradients": _fingerprint({n: p.grad for n, p in model.named_parameters()}),
            "optimizer": _fingerprint(model.optimizer.state_dict()),
            "scaler": _fingerprint(model.scaler.state_dict()),
            "scheduler": _fingerprint({k: v for k, v in vars(scheduler).items() if k != "optimizer"}),
            "wd_scheduler": _fingerprint({k: v for k, v in vars(wd).items() if k != "optimizer"}),
            "module_modes": _fingerprint({n: m.training for n, m in model.named_modules()}),
            "logical_rngs": _fingerprint(rng.state_dict()), "ambient_rng": _fingerprint(rng._capture())}


def _verify_clone_only(contract, model, scheduler, wd, batch, rng, proof, snapshots):
    snapshots["before"] = _complete_state(model, scheduler, wd, rng)
    try:
        result = core.verify_native_update(contract, model, scheduler, wd, batch, rng, input_proof=proof)
    except (TimeoutError, KeyboardInterrupt):
        # Do not submit more CUDA work after the bounded timer interrupts a GPU
        # operation. The external timeout owns final process cleanup if needed.
        snapshots["checked_after_failure"] = False
        raise
    except BaseException:
        snapshots["after_failure"] = _complete_state(model, scheduler, wd, rng)
        snapshots["checked_after_failure"] = True
        core.assert_same(snapshots["before"], snapshots["after_failure"])
        raise
    snapshots["after"] = _complete_state(model, scheduler, wd, rng)
    core.assert_same(snapshots["before"], snapshots["after"])
    if result.get("live_model_unchanged") is not True:
        raise ValueError("Native verification did not attest a clone-only update")
    return result


class _Timings:
    """Observe actual calls, never replace input checking or update arithmetic.

    CUDA-event intervals exclude each CPU input gate and include native update
    CPU launch/transfer gaps: they are stream elapsed, NOT kernel-busy seconds.
    Dedicated-process wrappers are restored even when model/parity work fails.
    """
    def __init__(self):
        self.phase = "preflight"
        self.gates, self.updates = [], []

    def __enter__(self):
        if threading.current_thread() is not threading.main_thread() or threading.active_count() != 1:
            raise ValueError("Timing wrappers require an exclusive single-Python-thread process")
        self.gate, self.drive = core._input_gate, core._drive_update
        self.factory_gate = factory._input_gate
        if self.factory_gate is not self.gate:
            raise ValueError("Model constructor input guard differs from native core")

        def gate(*args, **kwargs):
            started, ok = time.perf_counter(), False
            try:
                result = self.gate(*args, **kwargs)
                ok = True
                return result
            finally:
                self.gates.append({"phase": self.phase, "cpu_wall_seconds": time.perf_counter() - started,
                                   "completed": ok})

        def drive(*args, **kwargs):
            row = {"phase": self.phase, "mean_after": kwargs.get("mean_after", False),
                   "completed": False, "gpu_interval_seconds": None}
            self.updates.append(row)
            torch.cuda.synchronize(0)
            first, last = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            started = time.perf_counter()
            first.record()
            try:
                result = self.drive(*args, **kwargs)
                last.record()
                last.synchronize()
                row.update(completed=True, gpu_interval_seconds=first.elapsed_time(last) / 1000,
                    synchronized_wall_seconds=time.perf_counter() - started,
                    peak_allocated_bytes_so_far=torch.cuda.max_memory_allocated(0),
                    peak_reserved_bytes_so_far=torch.cuda.max_memory_reserved(0))
                return result
            except BaseException as error:
                # No synchronization/retry on failed or timed-out CUDA work.
                row.update(error_type=type(error).__name__,
                           wall_seconds_until_failure=time.perf_counter() - started)
                raise

        core._input_gate = factory._input_gate = gate
        core._drive_update = drive
        return self

    def __exit__(self, *exc):
        core._input_gate, factory._input_gate = self.gate, self.factory_gate
        core._drive_update = self.drive


@contextmanager
def _bounded_process():
    """Local process alarm only; a receiving outer timeout is also required."""
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise ValueError("Do not replace an existing process timeout")
    previous = {signum: signal.getsignal(signum) for signum in (signal.SIGALRM, signal.SIGTERM)}

    def interrupted(signum, frame):
        raise TimeoutError("Bounded native32 engineering interrupted: " + str(signum))

    for signum in previous:
        signal.signal(signum, interrupted)
    signal.setitimer(signal.ITIMER_REAL, LIMIT_SECONDS)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _source_files():
    directory = Path(__file__).parent
    return {name: sha256(directory / name) for name in (
        "robotics_training_gpu_check.py", "robotics_training_model.py", "robotics_training_pilot.py",
        "robotics_training_batch.py", "robotics_training_inputs.py", "model_loader.py", "vendor.py")}


def _four_updates(contract, model, scheduler, wd, batch, rng, input_proof, parity, timings, completed, output):
    """Fixed private lifecycle; every public update still executes all real gates."""
    if completed or scheduler._step != 0 or wd._step != 0:
        raise ValueError("Four-update engineering begins from the unchanged fresh model")
    for update in range(TIMED_UPDATES):
        timings.phase = f"timed_update_{update + 1}"
        call_started = time.perf_counter()
        value = core.accumulated_update(contract, model, scheduler, wd, batch, rng,
            input_proof=input_proof, numerical_proof=parity)
        if scheduler._step != update + 1 or wd._step != update + 1:
            raise ValueError("Timed engineering scheduler skipped or repeated an update")
        completed.append({"update": update + 1, "call_wall_seconds": time.perf_counter() - call_started,
                          **value})
        write_json(output / "progress.json", {"completed_timed_updates": len(completed),
            "required_timed_updates": TIMED_UPDATES, "history_updates": 0, "engineering_only": True})


def run_check(vendor, input_proof, output, *, expected_device_uuid):
    """Production-only CLI core; accepts no model, fixture, tolerance or job count."""
    vendor, input_proof, output = Path(vendor).resolve(), Path(input_proof), Path(output)
    if (not output.is_absolute() or output.is_symlink() or
            output.resolve() == input_proof.resolve() or input_proof.resolve() in output.resolve().parents):
        raise ValueError("Fresh explicit output outside immutable input evidence required")
    output.mkdir(parents=True, exist_ok=False)
    started, timings, snapshots = time.monotonic(), _Timings(), {}
    protocol = {"role": "native32_pusht_arithmetic_and_timing_engineering", "version": 1,
        "task": "pusht", "model_seed": MODEL_SEED, "timed_updates": TIMED_UPDATES,
        "input_proof": str(input_proof), "vendor": str(vendor), "source_sha256": _source_files(),
        "expected_device_uuid": expected_device_uuid, "internal_limit_seconds": LIMIT_SECONDS,
        "required_outer_timeout_seconds": 1210, "logical_rng_scope": RNG_SCOPE,
        "logical_ranks": 32, "microbatch": 8, "global_batch": 256,
        "backend_policy": BACKEND_POLICY,
        "cudnn_benchmark_basis": "pinned app/vjepa_wm/train.py lines 68 and 236",
        "reused_fixed_training_batch": True, "numerical_tolerance": "bitwise_no_relaxation",
        "validation_or_confirmation_access": False, "history_launch_authorized": False,
        "native_loader_lpips_rng_or_resume_verified": False, "physical_ddp_allreduce_order_reproduced": False,
        "engineering_only": True, "automatic_retry": False}
    write_json(output / "protocol.json", protocol)
    parity, completed, device = None, [], None
    original_backends = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                         torch.backends.cudnn.benchmark)
    try:
        with _bounded_process(), timings:
            if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
                raise ValueError("Dedicated receiving CPU thread pools must both be one")
            inputs = _input_files(input_proof)
            load_started = time.perf_counter()
            batch = torch.load(input_proof / "batch.pt", map_location="cpu", weights_only=True)
            load_seconds = time.perf_counter() - load_started
            contract = core.native_contract(vendor, "pusht", MODEL_SEED)
            core._input_gate(contract, batch, input_proof)
            device = _device_identity(expected_device_uuid)
            # Explicit registered precision policy, not a performance candidate.
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = True  # Actual pinned native training policy.
            torch.cuda.reset_peak_memory_stats(0)
            timings.phase = "constructor"
            model, scheduler, wd, post_constructor, construction = factory.build_receiving_model(
                contract, batch, input_proof=input_proof, device=0)
            rng = core.LogicalRNGStreams([copy.deepcopy(post_constructor) for _ in range(32)], device=0)
            initial_streams = _fingerprint(rng.state_dict())
            write_json(output / "constructor.json", {**construction, "logical_rng_scope": RNG_SCOPE,
                "duplicated_stream_count": 32, "logical_rng_state_sha256": initial_streams})
            timings.phase = "clone_only_parity"
            parity = _verify_clone_only(contract, model, scheduler, wd, batch, rng, input_proof, snapshots)
            write_json(output / "clone_state.json", snapshots)
            write_json(output / "numerical_proof.json", parity)
            if scheduler._step != 0 or wd._step != 0:
                raise ValueError("Disposable verification advanced the live scheduler")
            _four_updates(contract, model, scheduler, wd, batch, rng, input_proof, parity,
                          timings, completed, output)
            timings.phase = "final_integrity"
            final_state = _complete_state(model, scheduler, wd, rng)
            core.assert_same(inputs, _input_files(input_proof))
            core.assert_same(protocol["source_sha256"], _source_files())
            if len(timings.updates) != 6 or not all(row["completed"] for row in timings.updates):
                raise ValueError("Require both proof updates and all four timed complete updates")
            intervals = [row["gpu_interval_seconds"] for row in timings.updates
                         if row["phase"].startswith("timed_update_")]
            if len(intervals) != TIMED_UPDATES or not all(math.isfinite(x) and x > 0 for x in intervals):
                raise ValueError("Missing actual receiving GPU interval measurements")
            report = {"status": "native32_pusht_disposable_gpu_arithmetic_check_complete",
                "protocol_sha256": sha256(output / "protocol.json"), "inputs": inputs, "device": device,
                "backend_policy": {"cudnn_benchmark": torch.backends.cudnn.benchmark,
                    "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32},
                "constructor_sha256": sha256(output / "constructor.json"),
                "numerical_proof_sha256": sha256(output / "numerical_proof.json"),
                "clone_state_sha256": sha256(output / "clone_state.json"),
                "live_model_rng_optimizer_scheduler_unchanged_by_verification": True,
                "timed_engineering_updates": completed, "actual_call_timings": timings.updates,
                "cpu_input_gate_timings": timings.gates, "input_file_load_seconds": load_seconds,
                "median_timed_gpu_interval_seconds": statistics.median(intervals),
                "gpu_interval_definition": "CUDA stream elapsed around update only, including CPU launch/transfer gaps; excludes CPU input gates; not kernel-busy time",
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
                "final_complete_state": final_state, "logical_rng_scope": RNG_SCOPE,
                "native_loader_lpips_rng_or_resume_verified": False,
                "physical_ddp_allreduce_order_reproduced": False,
                "validation_or_confirmation_access": False, "history_launch_authorized": False,
                "training_history_updates": 0, "engineering_only": True,
                "seconds": time.monotonic() - started}
            write_json(output / "report.json", report)
            write_json(output / "DONE.json", {"report_sha256": sha256(output / "report.json")})
            return report
    except BaseException as error:
        write_json(output / "FAILED.json", {"error": str(error), "error_type": type(error).__name__,
            "phase": timings.phase, "completed_timed_updates": len(completed),
            "actual_call_timings": timings.updates, "cpu_input_gate_timings": timings.gates,
            "clone_snapshots": snapshots, "device": device,
            "numerical_proof_returned": parity is not None, "automatic_retry": False,
            "history_launch_authorized": False, "validation_or_confirmation_access": False,
            "seconds": time.monotonic() - started})
        raise
    finally:
        (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
         torch.backends.cudnn.benchmark) = original_backends


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vendor", "input-proof", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--expected-device-uuid", required=True)
    args = parser.parse_args()
    # Dedicated CLI process only; callers of run_check must already bind these.
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    print(json.dumps(run_check(args.vendor, args.input_proof, args.output,
                              expected_device_uuid=args.expected_device_uuid)))


if __name__ == "__main__":
    main()
