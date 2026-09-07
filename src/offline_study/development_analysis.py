"""Analyze frozen development interventions at the independent-lineage level."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .protocol import sha256, write_json


BOOTSTRAP_SEED = 2026090704


def bootstrap_mean_summary(
    values: list[float], seed: int, replicates: int = 10_000,
) -> dict[str, float | int | list[float] | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("Lineage bootstrap requires at least two finite scalar values")
    if replicates < 1000:
        raise ValueError("At least 1,000 bootstrap replicates are required")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(array), size=(replicates, len(array)))
    resampled = array[indices].mean(1)
    standard_deviation = float(array.std(ddof=1))
    mean = float(array.mean())
    return {
        "independent_lineage_groups": len(array),
        "mean": mean,
        "median": float(np.median(array)),
        "standard_deviation": standard_deviation,
        "standard_error": standard_deviation / np.sqrt(len(array)),
        "standardized_mean_difference": (
            mean / standard_deviation if standard_deviation > 0 else None),
        "two_sided_percentile_95_interval": [
            float(value) for value in np.quantile(resampled, [.025, .975])],
        "fraction_positive": float((array > 0).mean()),
        "fraction_negative": float((array < 0).mean()),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def analyze_task_report(report: dict, protocol: dict, seed: int, replicates: int) -> dict:
    tasks = protocol.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 1:
        raise ValueError("Development analysis requires one task-specific frozen protocol per run")
    task = tasks[0]
    if report.get("status") not in {
            "real_intervention_development_complete",
            "real_multigpu_intervention_development_complete"}:
        raise ValueError(f"Run is not a completed development intervention: {task}")
    if (not report.get("gpu_execution_valid") or report.get("scientific_confirmation") or
            report.get("category") != protocol.get("category")):
        raise ValueError(f"Invalid development-only execution receipt: {task}")
    registered = {item["name"]: item for item in protocol["primary_contrasts"]}
    observed = {item["name"]: item for item in report["aggregation"]["primary_contrasts"]}
    if set(registered) != set(observed):
        raise ValueError(f"Primary contrast registry mismatch: {task}")
    contrast_results = []
    for contrast_index, name in enumerate(registered):
        planned, result = registered[name], observed[name]
        if (planned["candidate"] != result.get("candidate") or
                planned["control"] != result.get("control")):
            raise ValueError(f"Primary contrast arms changed: {task}/{name}")
        groups = result.get("per_lineage_group", [])
        if not groups or {row["task"] for row in groups} != {task}:
            raise ValueError(f"Missing or cross-task lineage contrasts: {task}/{name}")
        metric_names = set(groups[0]["candidate_minus_control"])
        if any(set(row["candidate_minus_control"]) != metric_names for row in groups):
            raise ValueError(f"Inconsistent contrast metrics: {task}/{name}")
        metrics = {}
        for metric_index, metric in enumerate(sorted(metric_names)):
            values = [float(row["candidate_minus_control"][metric]) for row in groups]
            metrics[metric] = bootstrap_mean_summary(
                values, seed + 10_000 * contrast_index + metric_index, replicates)
        contrast_results.append({**planned, "metrics": metrics})
    mechanism = report.get("mechanism_diagnostics")
    mechanism_metrics = None
    if mechanism is not None:
        groups = mechanism.get("per_lineage_group", [])
        if not groups or {row["task"] for row in groups} != {task}:
            raise ValueError(f"Missing or cross-task mechanism diagnostics: {task}")
        names = set(groups[0]["metrics"])
        mechanism_metrics = {
            metric: bootstrap_mean_summary(
                [float(row["metrics"][metric]) for row in groups],
                seed + 1_000_000 + index,
                replicates,
            )
            for index, metric in enumerate(sorted(names))
        }
    native = report["aggregation"]["per_arm"]["native"]["per_task_group_weighted"]
    if set(native) != {task}:
        raise ValueError(f"Native task registry mismatch: {task}")
    return {
        "task": task,
        "category": protocol["category"],
        "rollout_trajectories": report["rollout_trajectories"],
        "independent_lineage_groups": report["independent_lineage_groups"],
        "windows": report["windows"],
        "native_group_weighted_metrics": native[task]["metrics"],
        "primary_contrasts": contrast_results,
        "mechanism_diagnostics": mechanism_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    if args.bootstrap_replicates < 1000 or args.seed < 0:
        parser.error("Invalid bootstrap replicate count or seed")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        inputs, task_results, categories, tasks = [], [], set(), set()
        for run in args.runs:
            done_path, report_path, protocol_path = (
                run / "DONE.json", run / "report.json", run / "protocol.json")
            done = json.loads(done_path.read_text())
            if done.get("report_sha256") != sha256(report_path):
                raise ValueError(f"Run report hash mismatch: {run}")
            report = json.loads(report_path.read_text())
            protocol = json.loads(protocol_path.read_text())
            if report.get("protocol_sha256") != sha256(protocol_path):
                raise ValueError(f"Protocol hash mismatch: {run}")
            task_result = analyze_task_report(
                report, protocol, args.seed + 10_000_000 * len(task_results),
                args.bootstrap_replicates)
            if task_result["task"] in tasks:
                raise ValueError(f"Duplicate task result: {task_result['task']}")
            tasks.add(task_result["task"])
            categories.add(task_result["category"])
            task_results.append(task_result)
            inputs.append({
                "run": str(run.resolve()),
                "done_sha256": sha256(done_path),
                "report_sha256": sha256(report_path),
                "protocol_sha256": sha256(protocol_path),
            })
        if len(categories) != 1:
            raise ValueError("One development analysis may cover only one intervention category")
        report = {
            "schema_version": 1,
            "status": "development_lineage_analysis_complete",
            "scientific_confirmation": False,
            "category": next(iter(categories)),
            "tasks": sorted(tasks),
            "inputs": inputs,
            "analysis": {
                "resampling_unit": "lineage_group",
                "bootstrap_replicates": args.bootstrap_replicates,
                "seed": args.seed,
                "interval": "two-sided percentile 95%",
                "task_pooling": False,
                "contrast_difference": "candidate_minus_control",
                "multiplicity_adjusted_in_development": False,
                "arm_selection_from_this_receipt": False,
            },
            "per_task": sorted(task_results, key=lambda row: row["task"]),
            "limitations": [
                "Development estimates variance and effect size; it is not confirmation",
                "Intervals are descriptive bootstrap intervals, not multiplicity-adjusted tests",
                "No task pooling, arm selection, smallest-useful-effect choice, or power claim is made",
                "Push-T remains development-only until fresh independent families exist",
                "Forecast embedding error is not closed-loop physical behavior",
            ],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json(args.output / "report.json", report)
        source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        write_json(args.output / "DONE.json", {
            "status": report["status"],
            "report_sha256": sha256(args.output / "report.json"),
            "analysis_source_sha256": source_hash,
        })
        print(json.dumps({"status": report["status"], "category": report["category"],
                          "tasks": report["tasks"]}), flush=True)
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"status": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
