"""Execute a frozen development-only intervention protocol on recorded futures."""
from __future__ import annotations
from offline_study._paths import package_source_hash

import argparse
import hashlib
import json
import platform
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from offline_study import VENDOR_COMMIT
from offline_study.models.backends import JepaBackend
from offline_study.runtime.benchmark import batches, elapsed_call, filter_reviewed_development, select_rows, synchronize
from offline_study.interventions.interventions import PredictorIntervention, compile_edits, validate_frozen_protocol, validate_operator_bank
from offline_study.runtime.pipeline import PrefetchIterator
from offline_study.core.protocol import sha256, summarize_metrics, validate_manifest, write_json


# Predictor attention uses optimized CUDA kernels. Separate FP32 executions can
# differ at roundoff scale even when inputs and weights are identical. The
# same-pass native/zero-dose check below remains bitwise exact; this tolerance
# is only for the additional uninstrumented-versus-instrumented comparison.
IDENTITY_RTOL = 1e-5
IDENTITY_ATOL = 5e-5


def _model_versions(model) -> dict[str, int]:
    versions = {
        f"parameter:{name}": parameter._version
        for name, parameter in model.named_parameters()
    }
    versions.update({
        f"buffer:{name}": buffer._version
        for name, buffer in model.named_buffers()
    })
    return versions


def _edit_norms(edits, effective_batch: int) -> list[dict[str, float]]:
    rows = [dict() for _ in range(effective_batch)]
    for edit in edits:
        token_slice = "all" if edit.token_start is None and edit.token_end is None else (
            f"{edit.token_start}:{edit.token_end}")
        label = f"{edit.site}/H{edit.horizon}/P{edit.block}/tokens={token_slice}"
        if edit.delivered_l2 is None or len(edit.delivered_l2) != effective_batch:
            raise ValueError("Compiled edit is missing its CPU-staged delivered norms")
        values = edit.delivered_l2
        for index, value in enumerate(values):
            rows[index][label] = value
    return rows


def _identity_diagnostics(left: torch.Tensor, right: torch.Tensor) -> dict:
    """Return actionable diagnostics for a failed instrumentation identity check."""
    if left.shape != right.shape:
        return {"left_shape": list(left.shape), "right_shape": list(right.shape)}
    finite = torch.isfinite(left) & torch.isfinite(right)
    difference = (left.float() - right.float()).abs()
    finite_difference = difference[finite]
    return {
        "shape": list(left.shape),
        "dtype": str(left.dtype),
        "all_finite": bool(finite.all().item()),
        "different_elements": int(torch.count_nonzero(left != right).item()),
        "maximum_absolute_difference": (
            float(finite_difference.max().item()) if finite_difference.numel() else None
        ),
        "mean_absolute_difference": (
            float(finite_difference.mean().item()) if finite_difference.numel() else None
        ),
        "rtol": IDENTITY_RTOL,
        "atol": IDENTITY_ATOL,
    }


def _numerically_identical(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(torch.allclose(
        left, right, rtol=IDENTITY_RTOL, atol=IDENTITY_ATOL, equal_nan=False))


def score_intervention_predictions(
    predicted,
    target,
    arm_names: list[str],
    category: str,
    window_count: int,
    horizons=(1, 3, 6),
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Score arms and category diagnostics with one device-to-host transfer."""
    metric_names, metric_columns = [], []
    for modality in ("visual", "proprio"):
        for horizon in horizons:
            delta = predicted[modality][horizon].float() - target[modality][:, horizon].float()
            metric_names.append(f"{modality}_mse_h{horizon}")
            metric_columns.append(delta.square().flatten(1).mean(1))

    if category == "action_response_geometry":
        for modality in ("visual", "proprio"):
            for horizon in (3, 6):
                outputs = predicted[modality][horizon].float().reshape(window_count, len(arm_names), -1)
                native = outputs[:, arm_names.index("native"):arm_names.index("native") + 1]
                metric_names.append(f"{modality}_native_fidelity_mse_h{horizon}")
                metric_columns.append((outputs - native).square().mean(2).flatten())

    diagnostic_names, diagnostic_columns = [], []
    if category == "vision_action_coupling":
        indices = {
            name: arm_names.index(name)
            for name in ("native", "visual_only", "action_condition_only", "joint")
        }
        arm_count = len(arm_names)
        for modality in ("visual", "proprio"):
            for horizon in horizons:
                outputs = predicted[modality][horizon].float().reshape(window_count, arm_count, -1)
                targets = target[modality][:, horizon].float().reshape(window_count, arm_count, -1)[:, 0]
                native = outputs[:, indices["native"]]
                visual = outputs[:, indices["visual_only"]]
                action = outputs[:, indices["action_condition_only"]]
                joint = outputs[:, indices["joint"]]
                output_interaction = joint - visual - action + native
                visual_effect = visual - native
                action_effect = action - native
                factorial_mse = (
                    (joint - targets).square().mean(1)
                    - (visual - targets).square().mean(1)
                    - (action - targets).square().mean(1)
                    + (native - targets).square().mean(1)
                )
                additive_cross = 2 * (visual_effect * action_effect).mean(1)
                values = (
                    output_interaction.square().mean(1),
                    additive_cross,
                    factorial_mse,
                    factorial_mse - additive_cross,
                )
                labels = (
                    "output_interaction_mse",
                    "additive_quadratic_cross",
                    "factorial_mse_interaction",
                    "nonadditive_mse_remainder",
                )
                for label, value in zip(labels, values, strict=True):
                    diagnostic_names.append(f"{modality}_{label}_h{horizon}")
                    diagnostic_columns.append(value.repeat_interleave(arm_count))

    metric_count = len(metric_columns)
    columns = metric_columns + diagnostic_columns
    transferred = torch.stack(columns, dim=1).cpu().tolist()
    metrics = [
        dict(zip(metric_names, row[:metric_count], strict=True))
        for row in transferred
    ]
    diagnostics = [
        dict(zip(diagnostic_names, transferred[index * len(arm_names)][metric_count:], strict=True))
        for index in range(window_count)
    ] if diagnostic_names else []
    return metrics, diagnostics


def summarize_interventions(rows: list[dict], protocol: dict) -> dict:
    names = [arm["name"] for arm in protocol["arms"]]
    by_arm = {
        name: summarize_metrics([
            {key: row[key] for key in (
                "task", "trajectory_id", "lineage_group", "start", "metrics"
            )}
            for row in rows if row["arm"] == name
        ])
        for name in names
    }
    contrasts = []
    for contrast in protocol["primary_contrasts"]:
        candidate = {
            (row["task"], row["trajectory_id"]): row
            for row in by_arm[contrast["candidate"]]["per_trajectory"]
        }
        control = {
            (row["task"], row["trajectory_id"]): row
            for row in by_arm[contrast["control"]]["per_trajectory"]
        }
        if set(candidate) != set(control):
            raise ValueError(f"Unpaired trajectories in contrast: {contrast['name']}")
        per_trajectory, group_values = [], defaultdict(list)
        for key in sorted(candidate):
            if candidate[key]["lineage_group"] != control[key]["lineage_group"]:
                raise ValueError(f"Unpaired lineage groups in contrast: {contrast['name']}/{key}")
            candidate_metrics = candidate[key]["metrics"]
            control_metrics = control[key]["metrics"]
            if set(candidate_metrics) != set(control_metrics):
                raise ValueError(f"Unpaired metrics in contrast: {contrast['name']}/{key}")
            difference = {
                metric: candidate_metrics[metric] - control_metrics[metric]
                for metric in sorted(candidate_metrics)
            }
            lineage_group = candidate[key]["lineage_group"]
            per_trajectory.append({
                "task": key[0], "trajectory_id": key[1], "lineage_group": lineage_group,
                "candidate_minus_control": difference,
            })
            group_values[(key[0], lineage_group)].append(difference)
        per_lineage_group, task_group_values = [], defaultdict(list)
        for (task, lineage_group), values in sorted(group_values.items()):
            mean = {
                metric: sum(value[metric] for value in values) / len(values)
                for metric in values[0]
            }
            per_lineage_group.append({
                "task": task,
                "lineage_group": lineage_group,
                "trajectories": len(values),
                "candidate_minus_control": mean,
            })
            task_group_values[task].append(mean)
        per_task = {
            task: {
                metric: sum(value[metric] for value in values) / len(values)
                for metric in values[0]
            }
            for task, values in sorted(task_group_values.items())
        }
        contrasts.append({
            **contrast,
            "difference_definition": (
                "candidate_minus_control after windows within rollout, then rollouts within "
                "lineage group, then equal lineage-group weighting"
            ),
            "per_trajectory": per_trajectory,
            "per_lineage_group": per_lineage_group,
            "per_task_group_weighted_mean": per_task,
        })
    return {"per_arm": by_arm, "primary_contrasts": contrasts}


@torch.inference_mode()
def execute(args, backend, selected, protocol, bank):
    device = backend.device
    torch.cuda.reset_peak_memory_stats(device)
    arm_names = [arm["name"] for arm in protocol["arms"]]
    arm_count = len(arm_names)
    native_index = arm_names.index("native")
    zero_index = arm_names.index("zero_dose")
    timings = {
        "data_wait_seconds": 0., "data_decode_seconds": 0., "encode_seconds": 0.,
        "operator_stage_seconds": 0., "rollout_seconds": 0., "metrics_seconds": 0.,
        "warmup_seconds": 0., "instrumentation_reference_seconds": 0.,
    }
    measurements = []
    mechanism_rows = []
    started = time.perf_counter()
    deadline = started + args.max_runtime_seconds
    iterator = PrefetchIterator(
        batches(selected, args.batch_size, backend, args.data_root, pin_memory=True),
        args.prefetch_batches,
    )
    instrumentation_identity = None
    instrumentation_identity_diagnostics = {}
    zero_dose_identity = True
    batch_count = 0
    try:
        while True:
            if time.perf_counter() >= deadline:
                raise TimeoutError("Intervention execution reached its batch-boundary runtime cap")
            try:
                meta, visual, proprio, raw_actions = next(iterator)
            except StopIteration:
                break
            encoded, seconds = elapsed_call(
                lambda visual=visual, proprio=proprio: backend.encode(visual, proprio), device)
            timings["encode_seconds"] += seconds
            context = backend.context(encoded)
            actions = backend.normalize_actions(raw_actions)
            expanded_context = backend.expand_context(context, arm_count)
            expanded_actions = actions.repeat_interleave(arm_count, dim=1)
            expanded_target = {
                key: encoded[key].repeat_interleave(arm_count, dim=0)
                for key in ("visual", "proprio")
            }
            geometry_diagnostics = None
            if protocol["category"] == "action_response_geometry" and "geometry" in protocol:
                from offline_study.interventions.action_geometry import prepare_geometry
                (compiled, geometry_diagnostics), seconds = elapsed_call(
                    lambda: prepare_geometry(backend, context, actions, protocol, bank), device)
            elif "support_operator" in protocol:
                from offline_study.interventions.support_operator import prepare_support
                (compiled, geometry_diagnostics), seconds = elapsed_call(
                    lambda: prepare_support(backend, context, actions, protocol, bank), device)
            else:
                compiled, seconds = elapsed_call(
                    lambda protocol=protocol, bank=bank, meta=meta: compile_edits(
                        protocol, bank, meta, device), device)
            timings["operator_stage_seconds"] += seconds
            edit_norms = _edit_norms(compiled, len(meta) * arm_count)
            if batch_count == 0:
                start = time.perf_counter()
                for _ in range(args.warmup):
                    backend.predict(expanded_context, expanded_actions)
                synchronize(device)
                timings["warmup_seconds"] = time.perf_counter() - start
                reference, seconds = elapsed_call(
                    lambda expanded_context=expanded_context, expanded_actions=expanded_actions:
                    backend.predict(expanded_context, expanded_actions),
                    device,
                )
                timings["instrumentation_reference_seconds"] = seconds

            def edited_rollout(
                compiled=compiled,
                expanded_context=expanded_context,
                expanded_actions=expanded_actions,
            ):
                with PredictorIntervention(backend.predictor, compiled):
                    return backend.predict(expanded_context, expanded_actions)

            predicted, seconds = elapsed_call(edited_rollout, device)
            timings["rollout_seconds"] += seconds
            if "support_operator" in protocol:
                realized = torch.stack([edit.realized_l2 for edit in compiled], 1).square().sum(1).sqrt()
                requested = torch.tensor([
                    sum(value * value for value in norms.values()) ** .5 for norms in edit_norms
                ], device=device)
                if not torch.allclose(realized, requested, atol=1e-5, rtol=1e-3):
                    raise RuntimeError("Realized FP32 edit energy differs from the frozen requested dose")
                for norms, value in zip(edit_norms, realized.cpu().tolist(), strict=True):
                    norms["realized_total_l2"] = value
            native_positions = [i * arm_count + native_index for i in range(len(meta))]
            zero_positions = [i * arm_count + zero_index for i in range(len(meta))]
            for key in ("visual", "proprio"):
                if not torch.equal(predicted[key][:, native_positions], predicted[key][:, zero_positions]):
                    zero_dose_identity = False
                    diagnostics = _identity_diagnostics(
                        predicted[key][:, native_positions], predicted[key][:, zero_positions])
                    raise RuntimeError(f"Zero-dose identity failed for {key}: {diagnostics}")
                if batch_count == 0:
                    reference_native = reference[key][:, native_positions]
                    predicted_native = predicted[key][:, native_positions]
                    diagnostics = _identity_diagnostics(reference_native, predicted_native)
                    diagnostics["bitwise_equal"] = torch.equal(reference_native, predicted_native)
                    diagnostics["within_declared_fp32_tolerance"] = _numerically_identical(
                        reference_native, predicted_native)
                    instrumentation_identity_diagnostics[key] = diagnostics
                    if not diagnostics["within_declared_fp32_tolerance"]:
                        instrumentation_identity = False
                        raise RuntimeError(
                            f"Native instrumentation identity failed for {key}: {diagnostics}")
            if batch_count == 0:
                instrumentation_identity = True
                del reference
            start = time.perf_counter()
            values, diagnostics = score_intervention_predictions(
                predicted, expanded_target, arm_names, protocol["category"], len(meta))
            if geometry_diagnostics is not None:
                diagnostics = geometry_diagnostics
            expanded_rows = []
            for window_index, metadata in enumerate(meta):
                for arm in arm_names:
                    row = {**metadata, "arm": arm}
                    if diagnostics and arm == "native":
                        row["mechanism_diagnostics"] = diagnostics[window_index]
                    expanded_rows.append(row)
                if diagnostics:
                    mechanism_rows.append({**metadata, "metrics": diagnostics[window_index]})
            measurements.extend({
                **row,
                "metrics": value,
                "edit_l2_by_site": edit_norm,
            } for row, value, edit_norm in zip(expanded_rows, values, edit_norms, strict=True))
            timings["metrics_seconds"] += time.perf_counter() - start
            batch_count += 1
            del encoded, context, actions, expanded_context, expanded_actions
            del expanded_target, compiled, predicted
            if time.perf_counter() >= deadline:
                raise TimeoutError("Intervention execution exceeded its batch-boundary runtime cap")
    finally:
        iterator.close()
        timings["data_wait_seconds"] = iterator.consumer_wait_seconds
        timings["data_decode_seconds"] = iterator.producer_seconds
    measured = time.perf_counter() - started - timings["warmup_seconds"] - timings["instrumentation_reference_seconds"]
    if not measurements:
        raise ValueError("No eligible intervention windows")
    aggregation = summarize_interventions(measurements, protocol)
    window_count = len(measurements) // arm_count
    return {
        "status": "real_intervention_development_complete",
        "gpu_execution_valid": True,
        "scientific_confirmation": False,
        "category": protocol["category"],
        "arms": arm_names,
        "primary_tasks": protocol["tasks"],
        "rollout_trajectories": len({row["trajectory_id"] for row in measurements}),
        "independent_lineage_groups": len({row["lineage_group"] for row in measurements}),
        "independence_unit": "lineage_group",
        "windows": window_count,
        "arm_evaluations": len(measurements),
        "batches": batch_count,
        "window_batch_size": args.batch_size,
        "maximum_effective_arm_batch_size": args.batch_size * arm_count,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "max_runtime_seconds": args.max_runtime_seconds,
        "measured_pipeline_seconds": measured,
        "windows_per_second": window_count / measured,
        "arm_evaluations_per_second": len(measurements) / measured,
        "zero_dose_identity": zero_dose_identity,
        "native_instrumentation_identity": instrumentation_identity,
        "native_instrumentation_identity_diagnostics": instrumentation_identity_diagnostics,
        "timing": timings,
        "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_gpu_bytes": torch.cuda.max_memory_reserved(device),
        "task_measured_trajectory_counts": dict(Counter(
            row["task"] for row in aggregation["per_arm"]["native"]["per_trajectory"])),
        "aggregation": aggregation,
        "mechanism_diagnostics": summarize_metrics(mechanism_rows) if mechanism_rows else None,
        "backend": backend.provenance,
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "cuda": torch.version.cuda, "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
        },
        "execution": {
            "precision": "float32", "allow_tf32": False,
            "arm_batching": "window-major; every registered arm in one predictor unroll",
            "parallelism": "disjoint rollout trajectories; no model collectives",
            "prefetch_batches": args.prefetch_batches,
        },
        "limitations": [
            "Development comparison only; this is not protected-holdout confirmation",
            "Operator fitting and frozen-protocol validity are upstream of this executor",
            "Forecast embedding error is not closed-loop physical behavior",
            "No simulator, CEM, autonomous method search, or parameter training occurs",
            "A split label alone does not establish that a lineage group is historically untouched",
        ],
    }, measurements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--exposure-registry", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--operator-bank", type=Path, required=True)
    parser.add_argument("--fit-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--max-trajectories", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Windows per batch; effective predictor batch also multiplies by arm count")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prefetch-batches", type=int, default=2)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-runtime-seconds", type=int, default=1800,
                        help="Fail at a batch boundary when this wall-clock cap is reached")
    args = parser.parse_args()
    if (args.batch_size < 1 or args.max_trajectories < 1 or args.warmup < 0 or
            args.prefetch_batches < 0 or args.num_shards < 1 or
            not 0 <= args.shard_index < args.num_shards or
            args.max_runtime_seconds < 1 or not args.device.startswith("cuda:")):
        parser.error("Invalid batch, trajectory, warmup, prefetch, shard, runtime, or CUDA device")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        protocol = json.loads(args.protocol.read_text())
        validate_frozen_protocol(protocol)
        protocol_sha256 = sha256(args.protocol)
        if protocol["vendor_commit"] != VENDOR_COMMIT:
            raise ValueError("Frozen protocol names the wrong vendor commit")
        if protocol["manifest_sha256"] != sha256(args.manifest):
            raise ValueError("Frozen protocol names a different manifest")
        if protocol["checkpoint_sha256"] != args.checkpoint_sha256:
            raise ValueError("Frozen protocol names a different checkpoint")
        if protocol["fit_receipt_sha256"] != sha256(args.fit_receipt):
            raise ValueError("Frozen protocol fit receipt hash mismatch")
        if set(protocol["tasks"]) != set(args.tasks):
            raise ValueError("Requested tasks differ from the frozen task registry")
        rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
        validate_manifest(rows)
        registry = json.loads(args.exposure_registry.read_text())
        rows = filter_reviewed_development(rows, registry, sha256(args.manifest))
        selected = select_rows(
            rows, args.tasks, "development", args.max_trajectories,
            args.shard_index, args.num_shards,
        )
        if not selected:
            raise ValueError("Empty development trajectory selection")
        if len({row["dataset"] for row in selected}) != 1:
            raise ValueError("One dataset/checkpoint is required per intervention process")
        selected_meta = [
            {"trajectory_id": row["trajectory_id"], "lineage_group": row["lineage_group"],
             "task": row["task"],
             "start": start, "split": "development"}
            for row in selected for start in row["starts"]
        ]
        bank = torch.load(args.operator_bank, map_location="cpu", weights_only=True)
        validate_operator_bank(bank, protocol_sha256, selected_meta)
        shutil.copyfile(args.protocol, args.output / "protocol.json")
        shutil.copyfile(args.fit_receipt, args.output / "fit_receipt.json")
        write_json(args.output / "selection.json", selected)
        write_json(args.output / "config.json", {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        })
        setup_started = time.perf_counter()
        backend = JepaBackend(
            args.vendor, args.checkpoint, args.checkpoint_sha256, selected[0]["dataset"],
            args.device, precision="float32", allow_tf32=False,
        )
        setup_seconds = time.perf_counter() - setup_started
        versions = _model_versions(backend.model)
        report, measurements = execute(args, backend, selected, protocol, bank)
        if versions != _model_versions(backend.model):
            raise RuntimeError("Frozen model parameter or buffer versions changed during intervention execution")
        write_json(args.output / "window_metrics.json", measurements)
        code_hash = package_source_hash()
        report.update(
            model_setup_seconds=setup_seconds,
            manifest_sha256=sha256(args.manifest),
            exposure_registry_sha256=sha256(args.exposure_registry),
            protocol_sha256=protocol_sha256,
            operator_bank_sha256=sha256(args.operator_bank),
            fit_receipt_sha256=sha256(args.fit_receipt),
            selection_sha256=sha256(args.output / "selection.json"),
            harness_source_sha256=code_hash,
            parameters_and_buffers_unchanged=True,
        )
        write_json(args.output / "report.json", report)
        write_json(args.output / "DONE.json", {
            "status": report["status"],
            "report_sha256": sha256(args.output / "report.json"),
            "window_metrics_sha256": sha256(args.output / "window_metrics.json"),
            "selection_sha256": sha256(args.output / "selection.json"),
            "protocol_sha256": sha256(args.output / "protocol.json"),
            "fit_receipt_sha256": sha256(args.output / "fit_receipt.json"),
        })
        print(json.dumps({key: report[key] for key in (
            "status", "category", "arms", "windows", "rollout_trajectories",
            "independent_lineage_groups",
            "measured_pipeline_seconds")}), flush=True)
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"status": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
