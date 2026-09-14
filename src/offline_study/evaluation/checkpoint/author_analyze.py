"""Verify complete author-correction shards and report every arm versus native."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from offline_study.evaluation.checkpoint.author_evaluate import check_coverage, verify_fit
from offline_study.evaluation.checkpoint.author_access import verify_runtime_access
from offline_study.evaluation.checkpoint.author_runtime import examples, validate_cohort
from offline_study.evaluation.development_analysis import simultaneous_contrasts
from offline_study.runtime.intervention_runner import summarize_interventions
from offline_study.core.protocol import sha256, summarize_metrics, write_json


def analyze(measurements, protocol, receipt, task):
    aggregation = summarize_interventions(measurements, protocol)
    endpoints = ("visual_mse_h6", "proprio_mse_h6")
    contrasts = aggregation["primary_contrasts"]
    vectors, labels, groups = [], [], None
    for contrast in contrasts:
        mapping = {row["lineage_group"]: row["candidate_minus_control"] for row in contrast["per_lineage_group"]}
        if groups is None:
            groups = sorted(mapping)
        if sorted(mapping) != groups:
            raise ValueError("Contrasts do not share identical lineage units")
        for endpoint in endpoints:
            labels.append((contrast, endpoint))
            vectors.append([mapping[group][endpoint] for group in groups])
    simultaneous = simultaneous_contrasts(np.asarray(vectors).T, 2026090704, 10000)
    comparison_rows = []
    for i, (contrast, endpoint) in enumerate(labels):
        control_mean = aggregation["per_arm"][contrast["control"]]["per_task_group_weighted"][task]["metrics"][endpoint]
        native_mean = aggregation["per_arm"]["native"]["per_task_group_weighted"][task]["metrics"][endpoint]
        threshold = .01 * receipt["fit_native_" + endpoint]
        lower, upper = simultaneous["lower"][i], simultaneous["upper"][i]
        comparison_rows.append({"candidate": contrast["candidate"], "control": contrast["control"], "endpoint": endpoint,
            "candidate_minus_control": simultaneous["mean"][i], "simultaneous_95_difference_interval": [lower, upper],
            "error_reduction_percent_of_control": -100 * simultaneous["mean"][i] / control_mean,
            "error_reduction_percent_of_native": -100 * simultaneous["mean"][i] / native_mean,
            "simultaneous_95_reduction_percent_of_observed_native": [-100 * upper / native_mean, -100 * lower / native_mean],
            "interval_denominator": "observed native mean; scaled difference CI, not a ratio-parameter CI",
            "beats_control": upper < 0,
            "beats_control_by_frozen_minimum": upper < -threshold,
            "minimum_useful_error_reduction": threshold})
    by_arm = defaultdict(list)
    for row in measurements:
        by_arm[row["arm"]].append(row)
    arms = {}
    for name, rows in by_arm.items():
        mean = aggregation["per_arm"][name]["per_task_group_weighted"][task]["metrics"]
        native = aggregation["per_arm"]["native"]["per_task_group_weighted"][task]["metrics"]
        arms[name] = {"lineage_weighted_metrics": mean,
            "clip_weighted_metrics": {metric: float(np.mean([row["metrics"][metric] for row in rows])) for metric in mean},
            "error_reduction_percent_vs_native": {metric: 100 * (1 - value / native[metric]) for metric, value in mean.items()},
            "requested_l2_mean": float(np.mean([row["requested_l2"] for row in rows])),
            "realized_l2_mean": float(np.mean([row["realized_l2"] for row in rows])),
            "realized_energy_match_fraction": float(np.mean([row["realized_energy_within_fp32_tolerance"] for row in rows]))}
    return {"aggregation": aggregation, "arms": arms, "contrasts": comparison_rows,
            "independent_lineage_groups": len(groups),
            "simultaneous_family": "all registered contrasts x visual/proprio H6 MSE within task/category/precision",
            "bootstrap_replicates": 10000, "bootstrap_seed": 2026090704,
            "critical_value": simultaneous["critical_value"],
            "interpretation": "Positive percentage = lower error than native; embeddings, not decoded states or task success. "
                "Quantized energy mismatch prevents an exact equal-delivered-energy interpretation.",
            "fresh_confirmation": False, "winner_selection": False}


def expected_shards(root, count):
    if count < 1:
        raise ValueError("A positive frozen shard count is required")
    directories = sorted(root.glob("shard-*"))
    if {p.name for p in directories} != {f"shard-{i}" for i in range(count)} or any(not p.is_dir() for p in directories):
        raise ValueError("Missing or unexpected fixed disjoint shards")
    return directories


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("cohort", "fit", "shards", "output"):
        parser.add_argument("--" + flag, required=True, type=Path)
    parser.add_argument("--shard-count", type=int, default=2,
                        help="Physical sharding only; complete frozen trajectory/arm coverage is still required")
    args = parser.parse_args()
    cohort = json.loads(args.cohort.read_text())
    validate_cohort(cohort)
    protected_access = verify_runtime_access(args.cohort, cohort)
    receipt = json.loads((args.fit / "fit_receipt.json").read_text())
    receipt, protocol = verify_fit(args.fit, sha256(args.cohort), receipt["checkpoint_sha256"], receipt["precision"])
    inputs, measurements, diagnostics, reports = {}, [], [], []
    directories = expected_shards(args.shards, args.shard_count)
    for directory in directories:
        done = json.loads((directory / "DONE.json").read_text())
        for name in ("report", "window_metrics", "selection", "protocol", "mechanism_diagnostics"):
            path = directory / (name + ".json")
            actual = sha256(path)
            if actual != done[name + "_sha256"]:
                raise ValueError(f"Shard checksum mismatch: {path}")
            inputs[str(path)] = actual
        report = json.loads((directory / "report.json").read_text())
        if (report["shard_count"] != args.shard_count or
                directory.name != f"shard-{report['shard_index']}" or report["task"] != cohort["task"]):
            raise ValueError("Changed physical shard identity or task")
        for key, expected in (("cohort_sha256", sha256(args.cohort)),
                              ("protocol_sha256", sha256(args.fit / "protocol.json")),
                              ("fit_receipt_sha256", sha256(args.fit / "fit_receipt.json")),
                              ("checkpoint_sha256", receipt["checkpoint_sha256"]),
                              ("precision", receipt["precision"])):
            if report.get(key) != expected:
                raise ValueError(f"Incompatible shard: {key}")
        if not report["zero_dose_identity"] or not report["native_instrumentation_fidelity"]:
            raise ValueError("Fidelity gate missing")
        if report["protected_outcomes_accessed"] != protected_access:
            raise ValueError("Shard exposure claim differs from the verified authorization")
        reports.append(report)
        measurements.extend(json.loads((directory / "window_metrics.json").read_text()))
        diagnostics.extend(json.loads((directory / "mechanism_diagnostics.json").read_text()))
    expected = examples(cohort["evaluation"], cohort["reference_config"], role=cohort.get("measurement_role", "development"))
    names = [arm["name"] for arm in protocol["arms"]]
    check_coverage(measurements, expected, names)
    expected_keys = {(m["trajectory_id"], m["start"]) for m in expected}
    diagnostic_keys = [(m["trajectory_id"], m["start"]) for m in diagnostics]
    if len(diagnostic_keys) != len(set(diagnostic_keys)) or set(diagnostic_keys) != expected_keys:
        raise ValueError("Missing or duplicated planned mechanism diagnostics; forecast-only is not sweep completion")
    category = protocol["category"]
    required = {"vision_action_coupling": "visual_output_interaction_mse_h6",
                "action_response_geometry": "cubic_visual_native_fidelity_mse_h6"}.get(category)
    if required and any(required not in row["metrics"] for row in diagnostics):
        raise ValueError("A planned mechanism endpoint is absent")
    result = analyze(measurements, protocol, receipt, cohort["task"])
    result["mechanism_diagnostics"] = summarize_metrics(diagnostics)
    result.update(status="verified_author_corrected_replication_complete" if protected_access else "verified_author_corrected_development_complete", task=cohort["task"],
                  category=protocol["category"], precision=receipt["precision"],
                  rollout_trajectories=len(cohort["evaluation"]), prefix_rollouts_per_arm=len(expected),
                  coverage=cohort["coverage"], protected_outcomes_accessed=protected_access,
                  access_authorization_sha256=cohort.get("access_authorization_sha256"),
                  input_sha256=inputs, fit_receipt_sha256=sha256(args.fit / "fit_receipt.json"),
                  cohort_sha256=sha256(args.cohort), protocol_sha256=sha256(args.fit / "protocol.json"))
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "report.json", result)
    write_json(args.output / "DONE.json", {"status": result["status"], "report_sha256": sha256(args.output / "report.json")})
    print(json.dumps({"status": result["status"], "task": result["task"], "category": result["category"],
                      "precision": result["precision"], "trajectories": result["rollout_trajectories"]}), flush=True)


if __name__ == "__main__":
    main()
