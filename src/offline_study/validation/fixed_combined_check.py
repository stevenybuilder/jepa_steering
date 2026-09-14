"""Fit-stimulus-only equivalence/timing of the native-prefix combined adapter.

This checker never constructs a dataset, decodes a video, selects an intervention,
or runs a planner. A separately reviewed, hash-pinned normalized fitting stimulus
is required: the native MetaWorld loader otherwise scans every row to normalize.
Success is engineering evidence only, not behavioral or confirmation clearance.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import pickle
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch

from offline_study.evaluation.checkpoint.author_evaluate import verify_fit
from offline_study.fitting.author_fit import source_hash
from offline_study.evaluation.checkpoint.author_runtime import validate_cohort
from offline_study.models.backends import JepaBackend
from offline_study.interventions.fixed_combined import ARM_COMPONENTS, METHOD, CombinedFixedResponseIntervention, validate_coupling, validate_predictor
from offline_study.interventions.fixed_response import FixedResponseIntervention, load_fitted_bank
from offline_study.validation.fixed_response_check import CountedBackend, reference_fields
from offline_study.interventions.interventions import PredictorIntervention
from offline_study.planning.planning_intervention import static_edits
from offline_study.planning.planning_panel_engineering import H6StaticPlanningIntervention
from offline_study.planning.planning_support_check import fit_candidate_actions
from offline_study.core.protocol import sha256
from offline_study.data.robotics_training_inputs import MANIFEST_HASHES, bound_manifest, safe_member
from offline_study.interventions.support_operator import NativeFieldCapture
from offline_study.models.vendor import use_vendor


STATUS = "fit_only_native_prefix_combined_engineering_complete"
COUNTS, HORIZONS, REPEATS, MAX_SECONDS = (8, 19, 300), (2, 5, 6), 3, 1200
RAW_PROOF_SHA256 = "fe5b27b569f59c41706f4eddb62925f13102c9d51ef241c79d4310fa613058ec"
NORMALIZATION_SCOPE = "pinned_upstream_planning_statistics_reused_not_fit_only"
STATS_SOURCE_SHA256 = "d242db62989cf8cb9abe3e901473722ccba9d7b1f6d0786d5598ae57c5d6f3ac"


def exclusive_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def byte_digest(value):
    raw = value.detach().reshape(-1).contiguous().cpu().view(torch.uint8)
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def tensor_signature(value):
    return {"shape": list(value.shape), "stride": list(value.stride()),
        "dtype": str(value.dtype), "device": str(value.device),
        "storage_offset": value.storage_offset(), "version": value._version,
        "requires_grad": value.requires_grad, "sha256": byte_digest(value)}


def input_signature(context, actions):
    return {"context": {k: tensor_signature(context[k]) for k in sorted(context.keys())},
        "actions": tensor_signature(actions)}


def model_signature(model):
    return {"parameters": {k: tensor_signature(v) for k, v in model.named_parameters()},
        "buffers": {k: tensor_signature(v) for k, v in model.named_buffers()},
        "modes": {k: v.training for k, v in model.named_modules()}}


def rng_signature(device):
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch_cpu": byte_digest(torch.get_rng_state())}
    if torch.device(device).type == "cuda":
        state["torch_cuda"] = byte_digest(torch.cuda.get_rng_state(device))
    return hashlib.sha256(pickle.dumps(state, protocol=4)).hexdigest()


def hooks_clean(predictor):
    from torch.nn.modules import module as global_state
    return not (global_state._global_forward_hooks or global_state._global_forward_pre_hooks or
        any(m._forward_hooks or m._forward_pre_hooks for m in predictor.modules()))


def assert_bytes(actual, expected, label):
    if (actual.shape != expected.shape or actual.dtype != expected.dtype or
            actual.device != expected.device or
            not torch.equal(actual.detach().reshape(-1).contiguous().view(torch.uint8),
                            expected.detach().reshape(-1).contiguous().view(torch.uint8))):
        raise ValueError("Byte parity failed: " + label)


def json_tensors(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: json_tensors(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_tensors(v) for v in value]
    return value


def verify_bindings(fit, coupling_fit, task, checkpoint_hash):
    if task != "mw-reach":
        raise ValueError("Initial combined engineering is frozen to Reach")
    for directory in (fit, coupling_fit, coupling_fit.parent):
        if (directory / "FAILED.json").exists():
            raise ValueError("A failed fitting artifact cannot authorize engineering")
    bank = load_fitted_bank(fit, task=task, checkpoint_sha256=checkpoint_hash)
    cohort = json.loads((fit / "cohort.json").read_text())
    validate_cohort(cohort)
    if (cohort["task"] != task or cohort["dataset"] != "metaworld" or
            len(cohort["fit"]) != 128 or bank["binding"].get("fit_precision") != "bfloat16"):
        raise ValueError("Wrong original fitting task/population/precision")
    cohort_hash = sha256(fit / "cohort.json")
    receipt, protocol = verify_fit(coupling_fit, cohort_hash, checkpoint_hash, "bfloat16")
    expected = {"status": "author_train_fit_only_complete", "task": task,
        "fit_split": "fit", "fit_lineage_group_count": 128,
        "fit_window_count": 512,
        "fit_lineage_groups": [r["lineage_group"] for r in cohort["fit"]],
        "fit_trajectory_ids": [r["trajectory_id"] for r in cohort["fit"]],
        "development_outcomes_accessed": False, "holdout_outcomes_accessed": False,
        "model_parameters_and_buffers_unchanged": True}
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError("Coupling fit population, integrity, or outcome-access binding changed")
    if (protocol.get("tasks") != [task] or protocol.get("evaluation_split") != "development" or
            sha256(coupling_fit.parent / "cohort.json") != cohort_hash):
        raise ValueError("Require original same-cohort coupling fit, not an access extension")
    coupling = torch.load(coupling_fit / "operator_bank.pt", map_location="cpu", weights_only=True)
    if coupling.get("protocol_sha256") != sha256(coupling_fit / "protocol.json"):
        raise ValueError("Coupling operator bank does not bind its frozen protocol")
    validate_coupling(protocol, coupling)
    config = cohort["reference_config"]
    if (config["data"]["custom"]["frameskip"] != 5 or
            config["data"]["validation"]["num_frames_val"] != 18):
        raise ValueError("Changed recorded H6 fitting clip convention")
    eligible = [r for r in cohort["fit"] if r["length"] >= 90]
    if not eligible:
        raise ValueError("No eligible frozen fitting row")
    selected = eligible[0]
    if selected["source_pool"] != "all" or selected["split"] != "fit":
        raise ValueError("Require the original global MetaWorld row identity")
    files = {str(directory / name): sha256(directory / name)
        for directory, names in ((fit, ("DONE.json", "cohort.json", "operator_bank.pt")),
            (coupling_fit, ("DONE.json", "protocol.json", "operator_bank.pt", "fit_receipt.json")),
            (coupling_fit.parent, ("PARITY.json", "cohort.json"))) for name in names}
    return bank, cohort, protocol, coupling, selected, files


def verify_stimulus(directory, pinned_hash, data_root, cohort_hash, selected, checkpoint_hash):
    """Verify receipts/bytes only. No tensor load or dataset initialization here.

    STIMULUS.json is independently pinned by the launch contract. Its `files`
    maps exact filenames to hashes; required members below are the complete
    normalized first clip, normalization provenance, original manifest, full raw
    verification report, and externally audited global-row/parquet mapping.
    """
    directory, data_root = Path(directory), Path(data_root)
    if (directory / "FAILED.json").exists() or sha256(directory / "STIMULUS.json") != pinned_hash:
        raise ValueError("Missing/failed/unbound normalized fitting stimulus")
    receipt = json.loads((directory / "STIMULUS.json").read_text())
    expected = {"schema_version": 1, "status": "frozen_normalized_fit_stimulus",
        "task": "mw-reach", "checkpoint_sha256": checkpoint_hash,
        "cohort_sha256": cohort_hash,
        "selected_fit_row": selected, "normalized_fit_payload_only": True,
        "new_full_pool_state_scan": False, "development_or_protected_outcomes_accessed": False,
        "normalization_scope": NORMALIZATION_SCOPE,
        "clip_start": 0, "frames": 18, "frameskip": 5}
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError("Stimulus population/normalization/access contract differs")
    files = receipt["files"]
    required = {"stimulus.pt", "normalization.pt", "normalization_source.json",
                "official_manifest.json", "raw_input_report.json", "source_mapping.json"}
    if set(files) != required:
        raise ValueError("Incomplete or unexpected fitting stimulus file registry")
    for name, digest in files.items():
        if sha256(safe_member(directory, name)) != digest:
            raise ValueError("Stimulus member changed: " + name)
    if files["raw_input_report.json"] != RAW_PROOF_SHA256:
        raise ValueError("Require the existing independently verified complete raw-input proof")
    manifest = bound_manifest("metaworld", directory / "official_manifest.json")
    normalization = json.loads((directory / "normalization_source.json").read_text())
    if (normalization.get("normalization_sha256") != files["normalization.pt"] or
            normalization.get("raw_manifest_sha256") != MANIFEST_HASHES["metaworld"] or
            normalization.get("published_stats_source_sha256") != STATS_SOURCE_SHA256 or
            normalization.get("native_normalization_reused") is not True or
            normalization.get("normalization_recomputed") is not False):
        raise ValueError("Original native normalization must be reused with its provenance")
    mapping = json.loads((directory / "source_mapping.json").read_text())
    rows = mapping["parquet_rows"]
    if (mapping.get("raw_manifest_sha256") != MANIFEST_HASHES["metaworld"] or
            set(rows) != set(manifest) or sum(rows.values()) != 12600 or
            any(type(v) is not int or v <= 0 for v in rows.values()) or
            mapping.get("selected_fit_row") != selected):
        raise ValueError("Require verified original global-row mapping, not a renumbered subset")
    # Mapping is audited at stimulus preparation and hash-pinned by the caller.
    # Recheck its positional arithmetic against the complete original ordering.
    offset, resolved = selected["index"], None
    for name in sorted(manifest):
        if offset < rows[name]:
            resolved = {"path": name, "local_row": offset}
            break
        offset -= rows[name]
    if resolved is None or mapping.get("selected_source") != resolved:
        raise ValueError("Selected original row does not match its source parquet")
    path = safe_member(data_root, resolved["path"])
    if path.stat().st_size != manifest[resolved["path"]]["bytes"] or sha256(path) != manifest[resolved["path"]]["sha256"]:
        raise ValueError("Selected original source parquet changed")
    bound = {str(directory / "STIMULUS.json"): pinned_hash,
             **{str(directory / n): h for n, h in files.items()},
             str(path): manifest[resolved["path"]]["sha256"]}
    return receipt, bound


def load_stimulus(directory):
    """Only called after contract publication and all predecode byte gates."""
    value = torch.load(Path(directory) / "stimulus.pt", map_location="cpu", weights_only=True)
    if set(value) != {"observations", "actions"} or set(value["observations"]) != {"visual", "proprio"}:
        raise ValueError("Stimulus may contain only normalized fitting observations/actions")
    expected = {"visual": (1, 18, 3, 224, 224), "proprio": (1, 18, 4), "actions": (1, 18, 20)}
    for name, tensor in {**value["observations"], "actions": value["actions"]}.items():
        if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32 or
                tuple(tensor.shape) != expected[name] or not torch.isfinite(tensor).all()):
            raise ValueError("Invalid normalized fitting tensor: " + name)
    return value


class BytePreservingReference(PredictorIntervention):
    def __init__(self, predictor, edits, active):
        super().__init__(predictor, edits)
        self.active = active

    def _replace(self, target, delta, label):
        changed = super()._replace(target, delta, label)
        if label.startswith("block_output/"):
            return torch.where(self.active[:, None, None], changed, target)
        return changed


@torch.no_grad()
def independent_reference(backend, fixed, protocol, coupling, arm, context, actions):
    """Two complete native/static forecasts, independently counted and compiled."""
    coupling_arm, fixed_arm = ARM_COMPONENTS[arm]
    horizon, batch = actions.shape[:2]
    trace, handles = [], []
    counted = CountedBackend(backend)
    try:
        for index, block in enumerate(backend.predictor.predictor_blocks):
            handles.append(block.register_forward_hook(
                lambda module, args, output, index=index: trace.append(index)))
        with torch.autocast(device_type=backend.device.type, enabled=False):
            if horizon < 6:
                result = counted.predict(context, actions)
                record = {"backend_calls": counted.calls, "predictor_blocks": len(trace)}
            else:
                with NativeFieldCapture(backend.predictor) as capture:
                    native = counted.predict(context, actions)
                field = capture.values[3]
                fields = reference_fields(field, fixed, fixed_arm)
                spec = fixed["operators"][fixed_arm]
                scores = ((field.float().flatten(1) - fixed["mean"]) @ fixed["projection"].T) / fixed["scale"]
                raw = torch.cat([torch.ones_like(scores[:, :1]), scores], 1) @ spec["map"].T
                unscaled = (raw @ spec["basis"].flatten(1)).reshape_as(field)
                norms = unscaled.flatten(1).norm(dim=1)
                active = norms > fixed["zero_threshold"]
                factors = torch.where(active, fixed["dose"] / norms.clamp_min(fixed["zero_threshold"]), 0.)
                edits = static_edits(protocol, coupling, coupling_arm, batch, 6, backend.device)
                with BytePreservingReference(backend.predictor, edits + fields, active):
                    result = counted.predict(context, actions)
                record = {"backend_calls": counted.calls, "predictor_blocks": len(trace),
                    "coefficients": raw * factors[:, None],
                    "requested_l2": fields[0].delta.flatten(1).norm(dim=1),
                    "realized_l2": fields[0].realized_l2, "active": active,
                    "coupling": [{"site": e.site, "block": e.block, "horizon": e.horizon,
                        "requested_l2": e.delivered_l2, "realized_l2": e.realized_l2,
                        "applications": e.applications} for e in edits]}
                for key in ("visual", "proprio"):
                    assert_bytes(result[key][:3], native[key][:3], "oracle native H1/H2 " + key)
            expected_passes = 2 if horizon == 6 else 1
            if counted.calls != expected_passes or trace != list(range(6)) * horizon * expected_passes:
                raise ValueError("Independent oracle forecast/block accounting failed")
            return result, record
    finally:
        for handle in handles:
            handle.remove()


def compare_records(actual, reference, horizon, count):
    counters = {"backend_calls": 1, "response_probe_rollouts": 0,
        "full_native_shadow_rollouts": 0, "native_prefix_replays": int(horizon == 6),
        "extra_native_predictor_blocks": 4 if horizon == 6 else 0,
        "main_predictor_blocks": 6 * horizon, "rank_applications": int(horizon == 6),
        "candidates": count, "horizon": horizon, "behavioral_launch_ready": False}
    if any(actual.get(k) != v for k, v in counters.items()):
        raise ValueError("Combined call/prefix/block/probe accounting failed")
    if horizon < 6:
        if actual["coupling"]:
            raise ValueError("A shortened forecast applied coupling")
        return
    for key in ("coefficients", "requested_l2", "realized_l2", "active"):
        assert_bytes(actual[key], reference[key], "per-candidate " + key)
    if len(actual["coupling"]) != 2 or len(reference["coupling"]) != 2:
        raise ValueError("Missing coupling delivery")
    for a, b in zip(actual["coupling"], reference["coupling"]):
        for key in ("site", "block", "horizon", "requested_l2", "applications"):
            if a[key] != b[key]:
                raise ValueError("Coupling delivery changed: " + key)
        assert_bytes(a["realized_l2"], b["realized_l2"], "coupling realized dose")


@torch.no_grad()
def equivalence_case(backend, adapter, protocol, coupling, context, actions):
    before_input, before_rng = input_signature(context, actions), rng_signature(backend.device)
    expected, reference = independent_reference(backend, adapter.bank, protocol, coupling,
                                               adapter.arm, context, actions)
    counted = adapter.backend
    before_calls = counted.calls
    with torch.autocast(device_type=backend.device.type, enabled=False):
        actual = adapter(context, actions)
    if counted.calls - before_calls != 1:
        raise ValueError("Combined operator made more than one backend call")
    horizon, count = actions.shape[:2]
    for key in ("visual", "proprio"):
        if (actual[key].shape[:2] != (horizon + 1, count) or
                not torch.isfinite(actual[key]).all() or not torch.isfinite(expected[key]).all()):
            raise ValueError("Incomplete/nonfinite candidate-horizon output")
        for h in range(horizon + 1):
            assert_bytes(actual[key][h], expected[key][h], f"{key}/H{h}")
    compare_records(adapter.last_record, reference, horizon, count)
    if before_input != input_signature(context, actions) or before_rng != rng_signature(backend.device):
        raise ValueError("Engineering changed an input/layout or CPU/CUDA RNG state")
    validate_predictor(backend.predictor)
    return {"arm": adapter.arm, "horizon": horizon, "candidate_count": count,
        "all_output_horizons_byte_equal": True, "input_shape_stride_bytes_unchanged": True,
        "cpu_cuda_rng_unchanged": True, "reference_backend_calls": reference["backend_calls"],
        "reference_predictor_blocks": reference["predictor_blocks"],
        "adapter_record": json_tensors(adapter.last_record)}


def synchronized_time(call, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.monotonic()
    with torch.autocast(device_type=device.type, enabled=False):
        output = call()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.monotonic() - start, output


def measure(call, device, context, actions):
    inputs_before, rng_before = input_signature(context, actions), rng_signature(device)
    synchronized_time(call, device)  # Separate untimed warmup on the exact case.
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    seconds = []
    for _ in range(REPEATS):
        elapsed, output = synchronized_time(call, device)
        seconds.append(elapsed)
        if any(not torch.isfinite(output[k]).all() for k in ("visual", "proprio")):
            raise ValueError("Nonfinite timed forecast")
        del output
    if inputs_before != input_signature(context, actions) or rng_before != rng_signature(device):
        raise ValueError("Timing changed inputs/layout or CPU/CUDA RNG")
    return {"seconds": seconds, "mean_seconds": sum(seconds) / REPEATS,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "warmup_forecasts": 1, "repeats": REPEATS, "cuda_synchronized": device.type == "cuda"}


def validate_coverage(checks, timings):
    expected = {(a, h, n) for a in ARM_COMPONENTS for h in HORIZONS for n in COUNTS}
    for rows in (checks, timings):
        keys = [(r["arm"], r["horizon"], r["candidate_count"]) for r in rows]
        if len(keys) != len(set(keys)) or set(keys) != expected:
            raise ValueError("Required case registry incomplete or duplicated; no DONE")
    if any(len(r["seconds"]) != REPEATS or r["warmup_forecasts"] != 1 or
           any(not math.isfinite(s) or s <= 0 for s in r["seconds"]) for r in timings):
        raise ValueError("Required timing repeats/warmups incomplete")


@contextlib.contextmanager
def deadline(seconds=MAX_SECONDS):
    if not hasattr(signal, "setitimer"):
        raise RuntimeError("This bounded receiving checker requires a POSIX timer")
    previous = signal.getsignal(signal.SIGALRM)
    if signal.getitimer(signal.ITIMER_REAL) != (0., 0.):
        raise RuntimeError("Do not replace an existing process timer")
    def expired(signum, frame):
        raise TimeoutError("1,200-second fit-only engineering cap reached; no completion")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        signal.signal(signal.SIGALRM, previous)


def verify_files(files):
    if any(sha256(Path(path)) != digest for path, digest in files.items()):
        raise ValueError("Frozen source/input/artifact bytes changed")


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fit", "coupling-fit", "vendor", "checkpoint", "data-root", "stimulus", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("checkpoint-sha256", "stimulus-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--task", required=True, choices=("mw-reach",))
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    started, backend, initial_model, checks, timings, optional = time.monotonic(), None, None, [], [], []
    original_policy = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                       torch.get_float32_matmul_precision())
    bound, frozen_source = {}, source_hash()
    try:
        with deadline():
            bank, cohort, protocol, coupling, selected, bound = verify_bindings(
                args.fit, args.coupling_fit, args.task, args.checkpoint_sha256)
            stimulus, inputs_bound = verify_stimulus(args.stimulus, args.stimulus_sha256,
                args.data_root, sha256(args.fit / "cohort.json"), selected, args.checkpoint_sha256)
            bound.update(inputs_bound)
            bound[str(args.checkpoint)] = args.checkpoint_sha256
            verify_files(bound)
            use_vendor(args.vendor)
            contract = {"method": METHOD, "task": args.task, "source_sha256": frozen_source,
                "bound_files": bound, "stimulus": stimulus, "fit_trajectory": selected,
                "candidate_counts": list(COUNTS), "horizons": list(HORIZONS),
                "arms": list(ARM_COMPONENTS), "timing_repeats": REPEATS,
                "precision": "strict_float32_no_autocast_no_tf32", "planning_context": 2,
                "reference": "independent_native_capture_then_static_coupling_and_fixed_fields",
                "max_seconds": MAX_SECONDS, "required_outer_process_cap_seconds": 1210,
                "candidate_actions": "cyclic contiguous H6 normalized recorded suffixes of first eligible fit row",
                "dataset_loader_or_video_decoder_called": False,
                "full_cem_episode_run": False, "behavioral_launch_ready": False,
                "scientific_efficacy_measurement": False, "fresh_confirmation": False}
            exclusive_json(args.output / "contract.json", contract)
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise ValueError("Expose exactly one authorized receiving GPU; no CPU fallback")
            backend = JepaBackend(args.vendor, args.checkpoint, args.checkpoint_sha256,
                                  "metaworld", "cuda:0", "float32")
            if backend.model.ctxt_window != 2 or backend.allow_tf32:
                raise ValueError("Wrong model context/precision")
            validate_predictor(backend.predictor)
            initial_model = model_signature(backend.model)
            exclusive_json(args.output / "model-before.json", initial_model)
            value = load_stimulus(args.stimulus)
            observations = {k: v.to(backend.device) for k, v in value["observations"].items()}
            actions = value["actions"].to(backend.device)
            with torch.autocast(device_type="cuda", enabled=False):
                visual, proprio, _ = backend.model.model.encode(observations, actions)
            from tensordict import TensorDict
            context = TensorDict({"visual": visual[:, :1], "proprio": proprio[:, :1]}, batch_size=[])
            counted = CountedBackend(backend)
            adapters = {a: CombinedFixedResponseIntervention(counted, bank, protocol, coupling, a)
                        for a in ARM_COMPONENTS}
            for count in COUNTS:
                full_actions = fit_candidate_actions(actions, count)
                for horizon in HORIZONS:
                    suffix = full_actions[:horizon]
                    for arm, adapter in adapters.items():
                        checks.append(equivalence_case(backend, adapter, protocol, coupling, context, suffix))
                        timings.append({"arm": arm, "horizon": horizon, "candidate_count": count,
                            **measure(lambda: adapter(context, suffix), backend.device, context, suffix)})
                        exclusive_json(args.output / f"case-{len(checks):02d}.json",
                                       {"check": checks[-1], "timing": timings[-1]})
            validate_coverage(checks, timings)
            # Optional comparator timings are fixed in advance, never selected by
            # outcomes. Admission uses observed combined runtime, with a 60 s
            # reserve for final full-model/source/input integrity checks.
            for count in COUNTS:
                suffix = fit_candidate_actions(actions, count)
                estimate = max(r["mean_seconds"] for r in timings if r["candidate_count"] == count and r["horizon"] == 6)
                fixed_comparator = FixedResponseIntervention(counted, bank, "fixed_rank4")
                coupling_comparator = H6StaticPlanningIntervention(counted, protocol, coupling,
                    "joint_equal_standardized_energy")
                comparators = {"native": lambda: backend.predict(context, suffix),
                    "fixed_rank4": lambda: fixed_comparator(context, suffix),
                    "coupling_only": lambda: coupling_comparator(context, suffix)}
                for arm, call in comparators.items():
                    if MAX_SECONDS - (time.monotonic() - started) < 60 + 2 * (REPEATS + 1) * estimate:
                        optional.append({"arm": arm, "candidate_count": count,
                            "status": "not_admitted_fixed_time_budget", "scientific_coverage_required": False})
                        continue
                    optional.append({"arm": arm, "horizon": 6, "candidate_count": count,
                                     "status": "timed", **measure(call, backend.device, context, suffix)})
            validate_predictor(backend.predictor)
            final_model = model_signature(backend.model)
            exclusive_json(args.output / "model-after.json", final_model)
            if final_model != initial_model or source_hash() != frozen_source:
                raise ValueError("Model parameter/buffer bytes/modes or source changed")
            verify_files(bound)
            report = {"status": STATUS, "contract_sha256": sha256(args.output / "contract.json"),
                "checks": checks, "timings": timings, "optional_comparators": optional,
                "backend": backend.provenance, "device_name": torch.cuda.get_device_name(backend.device),
                "device_uuid": str(torch.cuda.get_device_properties(backend.device).uuid),
                "parameter_buffer_bytes_modes_unchanged": True, "hooks_clean": True,
                "model_before_sha256": sha256(args.output / "model-before.json"),
                "model_after_sha256": sha256(args.output / "model-after.json"),
                "source_and_bound_inputs_unchanged": True, "wall_seconds": time.monotonic() - started,
                "full_cem_episode_run": False, "behavioral_launch_ready": False,
                "scientific_efficacy_measurement": False, "fresh_confirmation": False,
                "development_or_protected_outcomes_accessed": False,
                "normalization_was_fit_only": False, "normalization_recomputed": False}
            exclusive_json(args.output / "report.json", report)
            exclusive_json(args.output / "DONE.json", {"status": STATUS,
                "report_sha256": sha256(args.output / "report.json")})
    except BaseException as error:
        # No expensive CUDA synchronization/hash after timeout; parent enforces
        # the outer process deadline. Context-managed hooks still unwind.
        audit = {"source_unchanged": source_hash() == frozen_source,
            "model_integrity_recheck_completed": False, "hooks_clean": None}
        if backend is not None:
            audit["hooks_clean"] = hooks_clean(backend.predictor)
            audit["all_model_modules_eval"] = not any(m.training for m in backend.model.modules())
            if initial_model is not None and not isinstance(error, TimeoutError):
                try:
                    failed_model = model_signature(backend.model)
                    exclusive_json(args.output / "model-failure.json", failed_model)
                    audit["model_failure_sha256"] = sha256(args.output / "model-failure.json")
                    audit["model_bytes_modes_unchanged"] = failed_model == initial_model
                    audit["model_integrity_recheck_completed"] = True
                except BaseException as audit_error:
                    audit["integrity_error"] = repr(audit_error)
        exclusive_json(args.output / "FAILED.json", {"error": repr(error), "audit": audit,
            "completed_checks": len(checks), "completed_timings": len(timings),
            "partial_not_eligible": True, "behavioral_launch_ready": False,
            "scientific_efficacy_measurement": False, "fresh_confirmation": False})
        raise
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = original_policy[:2]
        torch.set_float32_matmul_precision(original_policy[2])


if __name__ == "__main__":
    main()
