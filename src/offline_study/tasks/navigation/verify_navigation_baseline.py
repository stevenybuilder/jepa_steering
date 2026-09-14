"""Verify complete author-split navigation native forecasts, without selection."""
import argparse
import json
import math
from pathlib import Path

from offline_study.evaluation.checkpoint.author_runtime import examples, validate_cohort
from offline_study.tasks.navigation.navigation_cohort import COUNTS
from offline_study.tasks.navigation.navigation_smoke import CHECKPOINTS
from offline_study.core.protocol import sha256, summarize_metrics, write_json


def verify(root, cohort_path):
    done = json.loads((root / "DONE.json").read_text())
    report = json.loads((root / "report.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    cohort = json.loads(cohort_path.read_text())
    validate_cohort(cohort)
    task = cohort["task"]
    if (task not in COUNTS or report["status"] != "full_author_navigation_native_forecast_complete" or
            report["task"] != task or protocol["task"] != task or
            report["precision"] != protocol["precision"] or
            protocol["precision"] not in ("bfloat16", "float32") or
            protocol["checkpoint_sha256"] != CHECKPOINTS[task] or
            protocol["cohort_sha256"] != sha256(cohort_path) or
            report["rollout_trajectories"] != COUNTS[task][2] or
            report["validation_clips"] != COUNTS[task][3] or
            report["prefixes"] != COUNTS[task][3] * 2 or
            report["parameters_unchanged"] is not True or report["fresh_confirmation"] is not False):
        raise ValueError("Wrong/incomplete native navigation reference")
    for filename, expected in (("report.json", done["report_sha256"]),
            ("protocol.json", report["protocol_sha256"]), ("PARITY.json", report["parity_sha256"]),
            ("window_metrics.json", report["window_metrics_sha256"])):
        if sha256(root / filename) != expected:
            raise ValueError("Altered navigation receipt: " + filename)
    rows = json.loads((root / "window_metrics.json").read_text())
    expected = examples(cohort["evaluation"], cohort["reference_config"])
    wanted = {(r["trajectory_id"], r["start"]) for r in expected}
    actual = [(r["trajectory_id"], r["start"]) for r in rows]
    if len(actual) != len(wanted) or set(actual) != wanted or len(actual) != report["prefixes"]:
        raise ValueError("Missing, duplicated or different validation prefixes")
    for row in rows:
        if row["arm"] != "native" or any(not math.isfinite(v) for v in row["metrics"].values()):
            raise ValueError("Wrong condition or nonfinite recorded prediction metrics")
    if summarize_metrics(rows) != report["aggregation"]:
        raise ValueError("Native metric aggregation does not reconstruct")
    return {"task": task, "precision": report["precision"], "trajectories": len(cohort["evaluation"]),
        "clips": COUNTS[task][3], "H6_prefixes": len(actual), "report_sha256": done["report_sha256"],
        "seconds": report["seconds"], "peak_gpu_memory_bytes": report["peak_gpu_memory_bytes"],
        "full_prefix_coverage_and_metrics_reconstructed": True,
        "intervention_comparison_complete": False, "fresh_confirmation": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "cohorts", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    checks = [verify(args.root / task / precision, args.cohorts / task / "cohort.json")
              for task in COUNTS for precision in ("bfloat16", "float32")]
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "report.json", {"status": "all_four_navigation_native_references_verified",
        "checks": checks, "unique_validation_trajectories": 392,
        "intervention_comparisons_complete": False, "three_seed_histories_complete": False})
    write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    print(json.dumps(checks))


if __name__ == "__main__":
    main()
