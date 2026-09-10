"""Compare a strict-FP32 benchmark with a reduced-precision/TF32 candidate."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .protocol import sha256, write_json


def _load_run(path: Path) -> tuple[dict, list[dict]]:
    done = json.loads((path / "DONE.json").read_text())
    if done.get("report_sha256") != sha256(path / "report.json"):
        raise ValueError(f"Run report hash mismatch: {path}")
    if done.get("window_metrics_sha256") != sha256(path / "window_metrics.json"):
        raise ValueError(f"Run metric hash mismatch: {path}")
    report = json.loads((path / "report.json").read_text())
    if report.get("selection_sha256") != done.get("selection_sha256"):
        raise ValueError(f"Run selection receipt mismatch: {path}")
    return report, json.loads((path / "window_metrics.json").read_text())


def compare_runs(
    reference_report: dict,
    reference_rows: list[dict],
    candidate_report: dict,
    candidate_rows: list[dict],
    relative_floor: float = 1e-12,
) -> dict:
    if relative_floor <= 0:
        raise ValueError("Relative floor must be positive")
    if not reference_report.get("gpu_benchmark_valid") or not candidate_report.get("gpu_benchmark_valid"):
        raise ValueError("Precision comparison requires two real GPU benchmark runs")
    reference_execution = reference_report.get("execution", {})
    if reference_execution.get("precision") != "float32" or reference_execution.get("allow_tf32"):
        raise ValueError("Reference must be strict float32 with TF32 disabled")
    invariant_fields = ("manifest_sha256", "exposure_registry_sha256", "selection_sha256")
    for field in invariant_fields:
        if reference_report.get(field) != candidate_report.get(field):
            raise ValueError(f"Run provenance differs: {field}")
    if (reference_report.get("backend", {}).get("checkpoint_sha256") !=
            candidate_report.get("backend", {}).get("checkpoint_sha256")):
        raise ValueError("Run provenance differs: checkpoint_sha256")
    if reference_report.get("harness_source_sha256") != candidate_report.get("harness_source_sha256"):
        raise ValueError("Run provenance differs: harness_source_sha256")
    for field in ("batch_size", "prefetch_batches", "asynchronous_cache_writer", "cache_queue_depth"):
        if reference_execution.get(field) != candidate_report.get("execution", {}).get(field):
            raise ValueError(f"Execution differs beyond precision mode: {field}")
    for field in ("shard_index", "num_shards"):
        if reference_report.get(field) != candidate_report.get(field):
            raise ValueError(f"Execution differs beyond precision mode: {field}")
    for field in ("gpu", "gpu_capability", "torch", "cuda"):
        if reference_report.get("environment", {}).get(field) != candidate_report.get("environment", {}).get(field):
            raise ValueError(f"Runtime environment differs: {field}")
    def keyed(rows):
        result = {}
        for row in rows:
            if not row.get("lineage_group"):
                raise ValueError(f"Comparison row lacks lineage group: {row.get('trajectory_id')}")
            key = (row["task"], row["trajectory_id"], row["start"])
            if key in result:
                raise ValueError(f"Duplicate window in comparison: {key}")
            result[key] = {
                "lineage_group": row["lineage_group"],
                "metrics": row["metrics"],
            }
        return result
    reference = keyed(reference_rows)
    candidate = keyed(candidate_rows)
    if set(reference) != set(candidate):
        raise ValueError("Reference and candidate windows differ")
    samples = defaultdict(list)
    for key in sorted(reference):
        if reference[key]["lineage_group"] != candidate[key]["lineage_group"]:
            raise ValueError(f"Lineage group differs for {key}")
        reference_metrics = reference[key]["metrics"]
        candidate_metrics = candidate[key]["metrics"]
        if set(reference_metrics) != set(candidate_metrics):
            raise ValueError(f"Metric keys differ for {key}")
        for metric in reference_metrics:
            base = float(reference_metrics[metric])
            changed = float(candidate_metrics[metric])
            absolute = abs(changed - base)
            relative = absolute / max(abs(base), relative_floor)
            samples[metric].append((absolute, relative))
    drift = {
        metric: {
            "samples": len(values),
            "max_absolute": max(value[0] for value in values),
            "mean_absolute": sum(value[0] for value in values) / len(values),
            "max_relative": max(value[1] for value in values),
            "mean_relative": sum(value[1] for value in values) / len(values),
        }
        for metric, values in sorted(samples.items())
    }
    reference_rate = reference_report["windows_per_second"]
    candidate_rate = candidate_report["windows_per_second"]
    if reference_rate <= 0 or candidate_rate <= 0:
        raise ValueError("Run throughput must be positive")
    reference_memory = reference_report.get("peak_reserved_gpu_bytes")
    candidate_memory = candidate_report.get("peak_reserved_gpu_bytes")
    return dict(
        status="precision_candidate_measured_not_adjudicated",
        same_windows=True,
        windows=len(reference),
        rollout_trajectories=len({key[1] for key in reference}),
        independent_lineage_groups=len({
            value["lineage_group"] for value in reference.values()
        }),
        reference_execution=reference_execution,
        candidate_execution=candidate_report.get("execution", {}),
        metric_drift=drift,
        throughput_speedup=candidate_rate / reference_rate,
        peak_reserved_memory_ratio=(candidate_memory / reference_memory
                                    if candidate_memory is not None and reference_memory else None),
        limitation=("This compares end-to-end forecast-error metrics, including target encoding; "
                    "it does not establish scientific equivalence or raw latent parity."),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relative-floor", type=float, default=1e-12)
    parser.add_argument("--max-absolute-drift", type=float,
                        help="Predeclared acceptance bound; omit to measure without adjudicating")
    parser.add_argument("--max-relative-drift", type=float,
                        help="Predeclared acceptance bound; omit to measure without adjudicating")
    args = parser.parse_args()
    if (args.max_absolute_drift is None) != (args.max_relative_drift is None):
        parser.error("Supply both drift bounds or neither")
    if args.relative_floor <= 0 or any(
            value is not None and value < 0
            for value in (args.max_absolute_drift, args.max_relative_drift)):
        parser.error("Comparison floors/bounds must be nonnegative and the floor must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        reference_report, reference_rows = _load_run(args.reference)
        candidate_report, candidate_rows = _load_run(args.candidate)
        result = compare_runs(
            reference_report, reference_rows, candidate_report, candidate_rows, args.relative_floor)
        result.update(
            reference_report_sha256=sha256(args.reference / "report.json"),
            candidate_report_sha256=sha256(args.candidate / "report.json"),
        )
        if args.max_absolute_drift is not None:
            accepted = all(
                values["max_absolute"] <= args.max_absolute_drift and
                values["max_relative"] <= args.max_relative_drift
                for values in result["metric_drift"].values()
            )
            result.update(
                status="precision_candidate_accepted" if accepted else "precision_candidate_rejected",
                accepted=accepted,
                predeclared_bounds={
                    "max_absolute_drift": args.max_absolute_drift,
                    "max_relative_drift": args.max_relative_drift,
                },
            )
        write_json(args.output / "comparison.json", result)
        write_json(args.output / "DONE.json", {
            "status": result["status"],
            "comparison_sha256": sha256(args.output / "comparison.json"),
        })
        print(json.dumps({key: result[key] for key in (
            "status", "windows", "throughput_speedup", "peak_reserved_memory_ratio")}), flush=True)
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"status": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
