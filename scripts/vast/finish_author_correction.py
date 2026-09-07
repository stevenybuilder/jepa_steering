"""Bounded fixed-work completion: fill omitted diagnostics, verify, and analyze.

No new hypotheses, rentals, protected access, precision selection, or adaptive arms.
Waits for the intervention queue before reusing its six devices. Broad baselines
retain GPUs 4 and 5 and run concurrently with diagnostic completion and CPU analysis.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/workspace/jepa-runtime/author-correction-20260907")
CATEGORIES = ("vision_action_coupling", "action_response_geometry", "operator_rank",
              "distribution_layer", "distribution_spatial")


def invoke(command, timeout, log_name):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "code-v4/src"),
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    with (ROOT / "logs" / log_name).open("x") as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            if child.wait(timeout=timeout):
                raise RuntimeError(f"Completion command failed; inspect {log_name}")
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()


def compare_forecasts(previous, replacement):
    """Adding diagnostics must not change any already saved forecast/error/energy."""
    from offline_study.protocol import sha256
    receipts = []
    for shard in range(2):
        before = previous / f"shard-{shard:03d}"
        after = replacement / f"shard-{shard:03d}"
        for directory in (before, after):
            done = json.loads((directory / "DONE.json").read_text())
            if done["window_metrics_sha256"] != sha256(directory / "window_metrics.json"):
                raise ValueError("Saved forecast checksum mismatch")
        # Repeated CUDA floating point reductions can differ slightly by process.
        # Require all metadata identical and all numeric results within the existing
        # native instrumentation tolerance; record the largest observed difference.
        old = json.loads((before / "window_metrics.json").read_text())
        new = json.loads((after / "window_metrics.json").read_text())
        if len(old) != len(new):
            raise ValueError("Diagnostic completion changed forecast coverage")
        maximum = 0.
        for a, b in zip(old, new, strict=True):
            if {k: v for k, v in a.items() if k != "metrics"} != {k: v for k, v in b.items() if k != "metrics"}:
                raise ValueError("Diagnostic completion changed cohort, arms or energy")
            if set(a["metrics"]) != set(b["metrics"]):
                raise ValueError("Diagnostic completion changed metric registry")
            for metric, value in a["metrics"].items():
                difference = abs(value - b["metrics"][metric])
                maximum = max(maximum, difference)
                if difference > 1e-7 + 2e-5 * abs(value):
                    raise ValueError("Diagnostic completion changed forecast values")
        receipts.append({"shard": shard, "previous_sha256": sha256(before / "window_metrics.json"),
                         "replacement_sha256": sha256(after / "window_metrics.json"),
                         "maximum_absolute_metric_difference": maximum})
    return receipts


def main():
    from offline_study.protocol import sha256, write_json
    status = ROOT / "closure-v2"
    status.mkdir(exist_ok=False)
    start = time.monotonic()
    try:
        while True:
            if time.monotonic() - start > 9000:
                raise TimeoutError("Dependency wait exceeded fixed 2.5-hour cap")
            failures = [str(path) for folder in ("evaluation-v1", "broad-baseline-v1")
                        for path in (ROOT / folder).rglob("FAILED.json")]
            queue_path = ROOT / "logs/metaworld-eval-v1.json"
            queue = json.loads(queue_path.read_text()) if queue_path.is_file() else {}
            if failures or queue.get("status") == "failed":
                raise RuntimeError(f"Existing GPU work failed: {failures or queue}")
            if queue.get("status") == "all_scheduled_jobs_complete":
                break
            time.sleep(15)
        invoke([sys.executable, str(ROOT / "configs/run_author_correction-v4.py"),
                "--root", str(ROOT), "--source", str(ROOT / "code-v4/src"), "--stage", "evaluate",
                "--tasks", "reach", "reach-wall", "--precisions", "bfloat16", "float32",
                "--devices", "0", "1", "2", "3", "6", "7",
                "--categories", "vision_action_coupling", "action_response_geometry",
                "--evaluation-name", "diagnostic-completion-v1", "--tag", "diagnostic-completion-v1"],
               3600, "diagnostic-completion-v1-supervisor.log")
        comparisons, reports = {}, []
        for precision in ("bfloat16", "float32"):
            for task in ("reach", "reach-wall"):
                for category in CATEGORIES:
                    original = ROOT / "evaluation-v1" / precision / task / category
                    shards = original
                    if category in CATEGORIES[:2]:
                        shards = ROOT / "diagnostic-completion-v1" / precision / task / category
                        comparisons[f"{precision}/{task}/{category}"] = compare_forecasts(original, shards)
                    output = ROOT / "analysis-v2" / precision / task / category
                    invoke([sys.executable, "-m", "offline_study.author_analyze", "--cohort",
                            str(ROOT / "cohorts" / task / "cohort.json"), "--fit",
                            str(ROOT / "fits-v1" / precision / task / category), "--shards", str(shards),
                            "--output", str(output)], 600, f"analysis-v2-{precision}-{task}-{category}.log")
                    reports.append({"path": str(output / "report.json"), "sha256": sha256(output / "report.json")})
        while not all((ROOT / "broad-baseline-v1" / p / "DONE.json").is_file() for p in ("bfloat16", "float32")):
            if time.monotonic() - start > 10800 or list((ROOT / "broad-baseline-v1").rglob("FAILED.json")):
                raise RuntimeError("Broad baseline failed or exceeded completion cap")
            time.sleep(15)
        write_json(status / "report.json", {"status": "all_author_corrected_metaworld_offline_sweeps_verified",
            "requested_primary_tasks": 3, "evaluated_primary_tasks": 2, "analysis_scopes": 20,
            "pusht_status": "fit_only_protected_validation_not_opened", "protected_outcomes_accessed": False,
            "full_study_complete": False, "diagnostic_completion_forecast_checks": comparisons,
            "reports": reports, "seconds": time.monotonic() - start})
        write_json(status / "DONE.json", {"report_sha256": sha256(status / "report.json")})
    except Exception as exc:
        write_json(status / "FAILED.json", {"error": str(exc)})
        raise


if __name__ == "__main__":
    main()
