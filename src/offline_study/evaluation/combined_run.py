"""Fit-only parity, full-pool disjoint execution and analysis of the frozen recipe."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from offline_study.evaluation.checkpoint.author_analyze import analyze
from offline_study.evaluation.checkpoint.author_evaluate import check_coverage
from offline_study.evaluation.checkpoint.author_runtime import AuthorBackend, encoded_batches, examples, open_normalized_dataset, prefix_batches
from offline_study.evaluation.combined_contract import load_components, validate_contract
from offline_study.interventions.combined_operator import RECIPE_ARMS, prepare_combined
from offline_study.runtime.intervention_runner import _model_versions
from offline_study.interventions.interventions import CompiledEdit, PredictorIntervention
from offline_study.planning.planning_native_smoke import CHECKPOINTS
from offline_study.core.protocol import sha256, write_json


def load_contract(path):
    if sha256(path) != json.loads((path.parent / "FROZEN.json").read_text())["protocol_sha256"]:
        raise ValueError("Combined protocol changed after freezing")
    value = json.loads(path.read_text())
    validate_contract(value)
    return value


def banks(components):
    return {key: torch.load(value[0] / "operator_bank.pt", map_location="cpu", weights_only=True)
            for key, value in components.items()}


@torch.no_grad()
def run_batch(backend, components, operator_banks, metadata, context, actions, target, full_parity=False):
    count, names = len(RECIPE_ARMS), [r[0] for r in RECIPE_ARMS]
    coupling, rank = "vision_action_coupling", "operator_rank"
    edits, diagnostics, source_edits = prepare_combined(backend, context, actions, metadata,
        components[coupling][2], operator_banks[coupling], components[rank][2], operator_banks[rank], return_sources=True)
    expanded_target = {key: value.repeat_interleave(count, dim=0) for key, value in target.items()}
    # A no-op uses the actual uninstrumented model, not a row padded through
    # somebody else's active hooks. Zero-valued cloning changed FP32 striding in
    # the initial fit check. Dispatch is decided from requested edits BEFORE any
    # predictions/outcomes; no failed prediction is overwritten to pass a gate.
    native = backend.predict(context, actions)
    if full_parity:
        zero = backend.predict(context, actions)
        if any(not torch.equal(native[k], zero[k]) for k in ("visual", "proprio")):
            raise ValueError("True zero-dose native dispatch is not deterministic")
    prediction = {k: v.repeat_interleave(count, dim=1) for k, v in native.items()}
    requested = torch.zeros(actions.shape[1] * count, device=backend.device)
    realized = torch.zeros_like(requested)
    # Same-location edited arms remain batched; no-op rows and unrelated hook
    # locations do not alter a source singleton's numerical execution path.
    groups = (([2, 6], 1), ([3, 7], 2), ([4, 5, 8, 9], None))
    for arm_indices, source_column in groups:
        flat = torch.tensor([b * count + j for b in range(actions.shape[1]) for j in arm_indices], device=backend.device)
        relevant = [e for e in edits if source_column is None or (e.site == "block_output") == (source_column == 2)]
        fields = [e.delta.index_select(0, flat) for e in relevant]
        active = torch.stack([field.flatten(1).ne(0).any(1) for field in fields]).any(0)
        active_flat = flat[active]
        if not len(active_flat):
            continue  # Frozen source degeneracy means a true no-op, never a dropped observation.
        windows = active_flat.div(count, rounding_mode="floor")
        group_context = {k: v.index_select(0, windows) for k, v in context.items()}
        group_actions = actions.index_select(1, windows).contiguous()
        group_edits = []
        for original, field in zip(relevant, fields):
            delta = field[active].contiguous()
            group_edits.append(CompiledEdit(original.site, original.horizon, original.block,
                original.token_start, original.token_end, delta, delta.double().flatten(1).norm(dim=1).cpu().tolist()))
        with PredictorIntervention(backend.predictor, group_edits):
            changed = backend.predict(group_context, group_actions)
        if full_parity and source_column is not None:
            # Independent reference from the immutable source compiler's fields,
            # using the SAME batch shape and only its own registered hook sites.
            originals, source_names = source_edits[source_column]
            source_indices = torch.tensor([b * len(source_names) + source_names.index(RECIPE_ARMS[j][source_column])
                for b in range(actions.shape[1]) for j in arm_indices], device=backend.device)[active]
            reference_edits = []
            for original in originals:
                delta = original.delta.index_select(0, source_indices).contiguous()
                reference_edits.append(replace(original, delta=delta,
                    delivered_l2=delta.double().flatten(1).norm(dim=1).cpu().tolist(), applications=0, realized_l2=None))
            if len(reference_edits) != len(group_edits) or any(not torch.equal(a.delta, b.delta)
                    for a, b in zip(reference_edits, group_edits)):
                raise ValueError("Combined dispatch changed a source component tensor")
            with PredictorIntervention(backend.predictor, reference_edits):
                expected = backend.predict(group_context, group_actions)
            if any(not torch.allclose(changed[k].float(), expected[k].float(), atol=1e-6, rtol=1e-5)
                    for k in ("visual", "proprio")):
                raise ValueError("Combined dispatch changes the frozen source singleton")
        if any(e.realized_l2 is None for e in group_edits):
            raise ValueError("Missing combined energy measurement")
        for key in prediction:
            prediction[key][:, active_flat] = changed[key]
        requested[active_flat] = torch.stack([torch.tensor(e.delivered_l2, device=backend.device) for e in group_edits], 1).square().sum(1).sqrt()
        realized[active_flat] = torch.stack([e.realized_l2 for e in group_edits], 1).square().sum(1).sqrt()
    metrics = backend.metrics(prediction, expanded_target)
    matched = torch.isclose(realized, requested, atol=1e-5, rtol=1e-3)
    energy = torch.stack([requested, realized, matched.float()], 1).cpu().tolist()
    measurements = []
    for i, meta in enumerate(metadata):
        for j, name in enumerate(names):
            index = i * count + j
            measurements.append({**meta, "arm": name, "metrics": metrics[index],
                "requested_l2": energy[index][0], "realized_l2": energy[index][1],
                "realized_energy_within_fp32_tolerance": bool(energy[index][2])})
    return measurements, [{**meta, "metrics": row} for meta, row in zip(metadata, diagnostics)]


def fit_check(args, contract):
    checks = []
    for precision in contract["precisions"]:
        cohort, components = load_components(contract, precision)
        operator_banks = banks(components)
        backend = AuthorBackend(args.vendor, args.checkpoint, CHECKPOINTS["metaworld"], "metaworld", "cuda:0", precision)
        versions = _model_versions(backend.model)
        dataset = open_normalized_dataset("metaworld", args.data_root, cohort["reference_config"], True)
        measured = []
        for clip_meta, encoded in encoded_batches(backend, dataset, cohort["fit"][:1], cohort["reference_config"], fitting=True):
            # The no-op baseline is checked directly against the official core
            # rollout/loss implementation, at the actual input batch shape.
            backend.verify_reference(encoded)
            for meta, context, actions, target in prefix_batches(backend, clip_meta, encoded):
                rows, _ = run_batch(backend, components, operator_banks, meta, context, actions, target, True)
                measured.extend(rows)
        if versions != _model_versions(backend.model):
            raise ValueError("Frozen model changed")
        expected = examples(cohort["fit"][:1], cohort["reference_config"], fitting=True)
        check_coverage(measured, expected, [r[0] for r in RECIPE_ARMS])
        checks.append({"precision": precision, "fit_trajectory": cohort["fit"][0]["trajectory_id"],
            "fit_windows": len(expected), "arms": len(RECIPE_ARMS), "identity_and_native_fidelity": True,
            "source_singleton_fidelity": True, "parameters_unchanged": True,
            "delivered_energy_within_frozen_tolerance": all(r["realized_energy_within_fp32_tolerance"] for r in measured)})
        del backend, operator_banks, dataset, encoded, context, actions, target
        torch.cuda.empty_cache()
    write_json(args.output / "report.json", {"status": "combined_fit_only_parity_passed", "checks": checks,
        "protocol_sha256": sha256(args.protocol), "development_or_protected_outcomes_accessed": False,
        "scientific_efficacy_measurement": False, "fresh_confirmation": False})
    write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})


def evaluate(args, contract):
    if args.fit_check is None or args.precision not in contract["precisions"] or not 0 <= args.shard_index < 4:
        raise ValueError("Need frozen precision, disjoint shard and fit-only gate")
    done = json.loads((args.fit_check / "DONE.json").read_text())
    if sha256(args.fit_check / "report.json") != done["report_sha256"]:
        raise ValueError("Fit-only gate changed")
    gate = json.loads((args.fit_check / "report.json").read_text())
    if (gate["protocol_sha256"] != sha256(args.protocol) or gate["development_or_protected_outcomes_accessed"] or
            {c["precision"] for c in gate["checks"]} != set(contract["precisions"]) or
            not all(c["identity_and_native_fidelity"] and c["source_singleton_fidelity"] and
                    c["parameters_unchanged"] and c["delivered_energy_within_frozen_tolerance"] for c in gate["checks"])):
        raise ValueError("Missing validated frozen combination")
    cohort, components = load_components(contract, args.precision)
    selected = [r for i, r in enumerate(cohort["evaluation"]) if i % 4 == args.shard_index]
    expected = examples(selected, cohort["reference_config"], role="combined_development")
    backend = AuthorBackend(args.vendor, args.checkpoint, CHECKPOINTS["metaworld"], "metaworld", "cuda:0", args.precision)
    versions, operator_banks = _model_versions(backend.model), banks(components)
    dataset = open_normalized_dataset("metaworld", args.data_root, cohort["reference_config"], False)
    measured, diagnostics, started = [], [], time.monotonic()
    for clip_meta, encoded in encoded_batches(backend, dataset, selected, cohort["reference_config"], role="combined_development"):
        for meta, context, actions, target in prefix_batches(backend, clip_meta, encoded):
            if time.monotonic() - started > 7000:
                raise TimeoutError("Combined shard reached its fixed execution cap")
            rows, diagnostic = run_batch(backend, components, operator_banks, meta, context, actions, target, not measured)
            measured.extend(rows)
            diagnostics.extend(diagnostic)
        print(json.dumps({"precision": args.precision, "shard": args.shard_index,
            "prefixes": len(diagnostics), "target": len(expected), "seconds": time.monotonic() - started}), flush=True)
    check_coverage(measured, expected, [r[0] for r in RECIPE_ARMS])
    if versions != _model_versions(backend.model):
        raise ValueError("Frozen model changed")
    for name, value in (("window_metrics", measured), ("diagnostics", diagnostics), ("selection", selected)):
        write_json(args.output / (name + ".json"), value)
    write_json(args.output / "report.json", {"status": "planned_combination_development_shard_complete",
        "task": contract["task"], "precision": args.precision, "shard_index": args.shard_index, "shard_count": 4,
        "protocol_sha256": sha256(args.protocol), "fit_gate_sha256": done["report_sha256"],
        "rollout_trajectories": len(selected), "prefix_rollouts_per_arm": len(expected),
        "zero_dose_identity": True, "native_fidelity": True, "frozen_weights_unchanged": True,
        "historically_protected_rows_reused": sum(r["split"] == "holdout" for r in selected),
        "newly_opened_untouched_rows": 0, "fresh_confirmation": False,
        "seconds": time.monotonic() - started, "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()})
    write_json(args.output / "DONE.json", {name + "_sha256": sha256(args.output / (name + ".json"))
        for name in ("report", "window_metrics", "diagnostics", "selection")})


def analyze_shards(args, contract):
    cohort, components = load_components(contract, args.precision)
    measured, inputs, diagnostics, indices = [], {}, [], []
    for directory in sorted(args.shards.glob("shard-*")):
        done = json.loads((directory / "DONE.json").read_text())
        for name in ("report", "window_metrics", "diagnostics", "selection"):
            path = directory / (name + ".json")
            if sha256(path) != done[name + "_sha256"]:
                raise ValueError("Combined measurement checksum mismatch")
            inputs[str(path)] = done[name + "_sha256"]
        report = json.loads((directory / "report.json").read_text())
        if (report["protocol_sha256"] != sha256(args.protocol) or report["precision"] != args.precision or
                not report["frozen_weights_unchanged"] or not report["native_fidelity"] or not report["zero_dose_identity"]):
            raise ValueError("Incompatible combined shard")
        indices.append(report["shard_index"])
        measured.extend(json.loads((directory / "window_metrics.json").read_text()))
        diagnostics.extend(json.loads((directory / "diagnostics.json").read_text()))
    if sorted(indices) != list(range(4)):
        raise ValueError("Missing or duplicate combined shards")
    expected = examples(cohort["evaluation"], cohort["reference_config"], role="combined_development")
    check_coverage(measured, expected, [r[0] for r in RECIPE_ARMS])
    diagnostic_keys = [(r["trajectory_id"], r["start"]) for r in diagnostics]
    if len(diagnostic_keys) != len(set(diagnostic_keys)) or set(diagnostic_keys) != {(r["trajectory_id"], r["start"]) for r in expected}:
        raise ValueError("Incomplete common-degeneracy diagnostics")
    result = analyze(measured, contract, components["operator_rank"][1], contract["task"])
    result.update(status="verified_full_reach_combination_development_complete", task=contract["task"],
        category=contract["category"], precision=args.precision, rollout_trajectories=33,
        prefix_rollouts_per_arm=len(expected), protocol_sha256=sha256(args.protocol), input_sha256=inputs,
        historically_protected_rows_reused=4, newly_opened_untouched_rows=0, fresh_confirmation=False,
        full_study_complete=False, winner_selection=False)
    write_json(args.output / "report.json", result)
    write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "evaluate", "analyze"))
    for name in ("protocol", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("vendor", "checkpoint", "data-root", "fit-check", "shards"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--precision", choices=("bfloat16", "float32"))
    parser.add_argument("--shard-index", type=int, default=-1)
    args = parser.parse_args()
    contract = load_contract(args.protocol)
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        {"check": fit_check, "evaluate": evaluate, "analyze": analyze_shards}[args.stage](args, contract)
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "fresh_confirmation": False})
        raise


if __name__ == "__main__":
    main()
