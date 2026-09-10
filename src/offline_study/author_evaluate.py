"""Evaluate every frozen arm on disjoint, protected-excluded official clip shards."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .action_geometry import prepare_geometry
from .author_access import verify_runtime_access
from .author_fit import source_hash
from .author_runtime import (AuthorBackend, encoded_batches, examples, open_normalized_dataset,
                            prefix_batches, validate_cohort)
from .intervention_runner import _model_versions, score_intervention_predictions, summarize_interventions
from .interventions import (PredictorIntervention, compile_edits, validate_frozen_protocol,
                           validate_operator_bank)
from .protocol import sha256, write_json
from .support_operator import prepare_support


def verify_fit(fit, cohort_hash, checkpoint_hash, precision):
    done = json.loads((fit / "DONE.json").read_text())
    for filename, key in (("protocol.json", "protocol_sha256"), ("operator_bank.pt", "operator_bank_sha256"),
                          ("fit_receipt.json", "fit_receipt_sha256")):
        if sha256(fit / filename) != done[key]:
            raise ValueError(f"Fit checksum mismatch: {filename}")
    receipt = json.loads((fit / "fit_receipt.json").read_text())
    for key, expected in (("cohort_sha256", cohort_hash), ("checkpoint_sha256", checkpoint_hash), ("precision", precision)):
        if receipt.get(key) != expected:
            raise ValueError(f"Fit compatibility mismatch: {key}")
    protocol = json.loads((fit / "protocol.json").read_text())
    validate_frozen_protocol(protocol)
    if protocol["fit_receipt_sha256"] != sha256(fit / "fit_receipt.json") or protocol["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("Protocol/receipt/checkpoint binding mismatch")
    if protocol["evaluation_split"] == "author_replication":
        source = receipt["source_frozen_artifacts"]
        if sha256(fit / "source_protocol.json") != source["protocol_sha256"] or sha256(fit / "source_fit_receipt.json") != source["fit_receipt_sha256"]:
            raise ValueError("Original frozen protocol or fit receipt changed")
        previous = json.loads((fit / "source_protocol.json").read_text())
        for key in ("arms", "primary_contrasts", "dose_budget", "geometry", "support_operator", "development_analysis_plan", "hypothesis", "tasks", "checkpoint_sha256"):
            if previous.get(key) != protocol.get(key):
                raise ValueError("Replication access changed the fixed scientific protocol: " + key)
    if protocol["author_validation"]["cohort_sha256"] != cohort_hash:
        raise ValueError("Protocol does not bind the evaluation cohort")
    if not (fit.parent / "PARITY.json").is_file():
        raise ValueError("Fit-only upstream parity receipt missing")
    return receipt, protocol


def check_coverage(measurements, expected_meta, arms):
    expected = {(m["trajectory_id"], m["start"], arm) for m in expected_meta for arm in arms}
    observed = [(m["trajectory_id"], m["start"], m["arm"]) for m in measurements]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError("Incomplete or duplicated clip/prefix/arm coverage")


def planned_diagnostics(prediction, target, names, category, count, existing=None):
    """Retain the original mechanism endpoints without replacing official losses."""
    if category not in {"vision_action_coupling", "action_response_geometry"}:
        return existing
    auxiliary, coupling = score_intervention_predictions(prediction, target, names, category, count)
    if category == "vision_action_coupling":
        return coupling
    result = [dict(row) for row in existing]
    for i in range(count):
        for j, name in enumerate(names):
            for metric, value in auxiliary[i * len(names) + j].items():
                if "native_fidelity" in metric:
                    result[i][name + "_" + metric] = value
    return result


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("vendor", "checkpoint", "cohort", "data-root", "output", "fit"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--precision", required=True, choices=["bfloat16", "float32"])
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("Invalid disjoint shard")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        started = time.monotonic()
        cohort = json.loads(args.cohort.read_text())
        validate_cohort(cohort)
        protected_access = verify_runtime_access(args.cohort, cohort)
        cohort_hash = sha256(args.cohort)
        receipt, protocol = verify_fit(args.fit, cohort_hash, args.checkpoint_sha256, args.precision)
        config = cohort["reference_config"]
        selected = [row for i, row in enumerate(cohort["evaluation"]) if i % args.shard_count == args.shard_index]
        if not selected:
            raise ValueError("Empty evaluation shard")
        role = cohort.get("measurement_role", "development")
        if protocol["evaluation_split"] != role:
            raise ValueError("Protocol and cohort access roles differ")
        if protected_access:
            authorization = cohort["access_authorization_sha256"]
            if receipt.get("access_authorization_sha256") != authorization or protocol["author_validation"].get("access_authorization_sha256") != authorization:
                raise ValueError("Fit/protocol lack the exact authorized access binding")
        expected = examples(selected, config, role=role)
        bank = torch.load(args.fit / "operator_bank.pt", map_location="cpu", weights_only=True)
        if protected_access and bank.get("access_authorization_sha256") != cohort["access_authorization_sha256"]:
            raise ValueError("Operator bank has the wrong access authorization")
        validate_operator_bank(bank, sha256(args.fit / "protocol.json"), expected)
        backend = AuthorBackend(args.vendor, args.checkpoint, args.checkpoint_sha256,
                                cohort["dataset"], args.device, args.precision)
        versions = _model_versions(backend.model)
        dataset = open_normalized_dataset(cohort["dataset"], args.data_root, config, False)
        names = [arm["name"] for arm in protocol["arms"]]
        count = len(names)
        measurements, diagnostics = [], []
        done_prefixes, native_parity = 0, None
        stage_seconds = {"operator": 0., "edited_prediction": 0., "metrics": 0.}
        for clip_meta, encoded in encoded_batches(backend, dataset, selected, config, role=role):
            for meta, context, actions, target in prefix_batches(backend, clip_meta, encoded):
                if time.monotonic() - started > 7000:
                    raise TimeoutError("Evaluation reached fixed time cap; partial results are not complete")
                timer = time.monotonic()
                diagnostic = None
                if "geometry" in protocol:
                    compiled, diagnostic = prepare_geometry(backend, context, actions, protocol, bank)
                elif "support_operator" in protocol:
                    compiled, diagnostic = prepare_support(backend, context, actions, protocol, bank)
                else:
                    compiled = compile_edits(protocol, bank, meta, backend.device)
                torch.cuda.synchronize(backend.device)
                stage_seconds["operator"] += time.monotonic() - timer
                expanded = backend.expand_context(context, count)
                expanded_actions = actions.repeat_interleave(count, dim=1)
                expanded_target = {k: v.repeat_interleave(count, dim=0) for k, v in target.items()}
                if native_parity is None:
                    reference = backend.predict(expanded, expanded_actions)
                timer = time.monotonic()
                with PredictorIntervention(backend.predictor, compiled):
                    prediction = backend.predict(expanded, expanded_actions)
                torch.cuda.synchronize(backend.device)
                stage_seconds["edited_prediction"] += time.monotonic() - timer
                for modality in ("visual", "proprio"):
                    native = prediction[modality][:, names.index("native")::count]
                    zero = prediction[modality][:, names.index("zero_dose")::count]
                    if not torch.equal(native, zero):
                        raise RuntimeError(f"Zero-dose identity failed: {modality}")
                    if native_parity is None:
                        ref = reference[modality][:, names.index("native")::count]
                        if not torch.allclose(native.float(), ref.float(), atol=1e-6, rtol=1e-5):
                            raise RuntimeError(f"Native instrumentation fidelity failed: {modality}")
                if native_parity is None:
                    native_parity = True
                    del reference
                timer = time.monotonic()
                metrics = backend.metrics(prediction, expanded_target)
                diagnostic = planned_diagnostics(prediction, expanded_target, names,
                    protocol["category"], len(meta), diagnostic)
                requested = torch.stack([torch.tensor(edit.delivered_l2, device=backend.device) for edit in compiled], 1).square().sum(1).sqrt()
                if any(edit.realized_l2 is None for edit in compiled):
                    raise ValueError("Missing realized energy instrumentation")
                realized = torch.stack([edit.realized_l2 for edit in compiled], 1).square().sum(1).sqrt()
                energy_ok = torch.isclose(realized, requested, atol=1e-5, rtol=1e-3)
                energy = torch.stack([requested, realized, energy_ok.float()], 1).cpu().tolist()
                for i, item in enumerate(meta):
                    if diagnostic:
                        diagnostics.append({**item, "metrics": diagnostic[i]})
                    for j, name in enumerate(names):
                        k = i * count + j
                        measurements.append({**item, "arm": name, "metrics": metrics[k],
                            "requested_l2": energy[k][0], "realized_l2": energy[k][1],
                            "realized_energy_within_fp32_tolerance": bool(energy[k][2])})
                done_prefixes += len(meta)
                stage_seconds["metrics"] += time.monotonic() - timer
            print(json.dumps({"task": cohort["task"], "category": protocol["category"], "precision": args.precision,
                "shard": args.shard_index, "prefixes_done": done_prefixes, "prefixes_total": len(expected),
                "seconds": time.monotonic() - started}), flush=True)
        check_coverage(measurements, expected, names)
        if versions != _model_versions(backend.model):
            raise RuntimeError("Frozen model changed during evaluation")
        write_json(args.output / "window_metrics.json", measurements)
        write_json(args.output / "mechanism_diagnostics.json", diagnostics)
        write_json(args.output / "protocol.json", protocol)
        write_json(args.output / "selection.json", selected)
        report = {"status": "author_offline_shard_complete", "task": cohort["task"],
            "category": protocol["category"], "precision": args.precision, "arms": names,
            "cohort_sha256": cohort_hash, "fit_receipt_sha256": sha256(args.fit / "fit_receipt.json"),
            "protocol_sha256": sha256(args.fit / "protocol.json"), "checkpoint_sha256": args.checkpoint_sha256,
            "shard_index": args.shard_index, "shard_count": args.shard_count,
            "rollout_trajectories": len(selected), "independent_lineage_groups": len({r["lineage_group"] for r in selected}),
            "windows": len(expected), "window_key": "clip_start * prefixes_per_clip + prefix; not a physical frame index",
            "coverage": cohort["coverage"], "protected_outcomes_accessed": protected_access, "fresh_confirmation": False,
            "evaluation_role": role, "access_authorization_sha256": cohort.get("access_authorization_sha256"),
            "zero_dose_identity": True, "native_instrumentation_fidelity": native_parity,
            "source_sha256": source_hash(), "backend": backend.provenance,
            "wall_seconds": time.monotonic() - started, "stage_seconds": stage_seconds,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(backend.device),
            "aggregation": summarize_interventions(measurements, protocol)}
        write_json(args.output / "report.json", report)
        write_json(args.output / "DONE.json", {"status": report["status"], **{
            name + "_sha256": sha256(args.output / (name + ".json")) for name in
            ("report", "window_metrics", "mechanism_diagnostics", "selection", "protocol")}})
        print(json.dumps({"status": report["status"], "seconds": report["wall_seconds"]}), flush=True)
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"status": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
