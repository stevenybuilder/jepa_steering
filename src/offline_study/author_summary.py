"""Export every verified arm versus native; never choose a winner or precision."""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

from .protocol import sha256, write_json


CATEGORIES = ("vision_action_coupling", "action_response_geometry", "operator_rank",
              "distribution_layer", "distribution_spatial")
COUNTS = {"reach": 33, "reach-wall": 27, "pusht": 21}


def verified_analysis(directory, task, precision, category, checked_inputs=None):
    checked_inputs = {} if checked_inputs is None else checked_inputs
    path = directory / "report.json"
    done = json.loads((directory / "DONE.json").read_text())
    if sha256(path) != done["report_sha256"]:
        raise ValueError("Analysis checksum mismatch")
    report = json.loads(path.read_text())
    status = ("verified_author_corrected_replication_complete" if task == "pusht"
              else "verified_full_primary_author_split_replication_complete")
    if (report["status"] != status or report["precision"] != precision or
            report["task"] != (task if task == "pusht" else "mw-" + task) or report["category"] != category or
            report["rollout_trajectories"] != COUNTS[task] or not report["coverage"]["complete_author_split"] or
            report["fresh_confirmation"]):
        raise ValueError("Analysis lacks complete authorized primary task coverage")
    for filename, expected in report["input_sha256"].items():
        if filename not in checked_inputs:
            checked_inputs[filename] = sha256(Path(filename))
        if checked_inputs[filename] != expected:
            raise ValueError("Underlying verified measurement changed")
    if not report["input_sha256"]:
        raise ValueError("No underlying measurement evidence")
    return report, {"path": str(path), "sha256": done["report_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--metaworld-extension-root", type=Path, required=True)
    parser.add_argument("--pusht-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparisons, reports, evidence, checked_inputs = [], [], [], {}
    for precision in ("bfloat16", "float32"):
        for task in COUNTS:
            for category in CATEGORIES:
                directory = ((args.pusht_root / "analysis-v1") if task == "pusht"
                             else (args.metaworld_extension_root / "analysis-full-v1")) / precision / task / category
                report, binding = verified_analysis(directory, task, precision, category, checked_inputs)
                reports.append(report)
                evidence.append(binding)
                intervals = {(row["candidate"], row["endpoint"]): row for row in report["contrasts"] if row["control"] == "native"}
                for arm, values in report["arms"].items():
                    for metric in ("proprio_mse_h6", "visual_mse_h6"):
                        contrast = intervals.get((arm, metric))
                        interval = contrast["simultaneous_95_reduction_percent_of_observed_native"] if contrast else [0., 0.]
                        comparisons.append({"task": task, "precision": precision, "category": category, "arm": arm,
                            "endpoint": metric, "independent_families": report["independent_lineage_groups"],
                            "native_error": report["arms"]["native"]["lineage_weighted_metrics"][metric],
                            "edited_error": values["lineage_weighted_metrics"][metric],
                            "error_reduction_percent": values["error_reduction_percent_vs_native"][metric],
                            "simultaneous_95_lower_percent": interval[0], "simultaneous_95_upper_percent": interval[1],
                            "requested_l2_mean": values["requested_l2_mean"], "realized_l2_mean": values["realized_l2_mean"],
                            "realized_energy_match_fraction": values["realized_energy_match_fraction"]})
    broad = {}
    for precision in ("bfloat16", "float32"):
        directory = args.root / "broad-baseline-v1" / precision
        done = json.loads((directory / "DONE.json").read_text())
        for name in ("report", "selection", "window_metrics", "PARITY"):
            if done[name + "_sha256"] != sha256(directory / (name + ".json")):
                raise ValueError("Broad baseline checksum mismatch")
        broad[precision] = json.loads((directory / "report.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "report.json", {"status": "verified_three_task_author_pool_offline_results",
        "primary_tasks_requested": 3, "primary_tasks_measured": 3, "task_trajectories": COUNTS,
        "full_study_complete": False, "protected_outcomes_accessed": True, "fresh_confirmation": False,
        "unique_primary_trajectories": 81,
        "primary_official_pool_rows": 81, "primary_author_pool_coverage_percent": 100.,
        "broad_unique_metaworld_trajectories": 1120, "broad_task_count": 42,
        "comparisons": comparisons, "broad_baseline": broad, "verified_analysis_scopes": evidence})
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(comparisons[0]))
    writer.writeheader()
    writer.writerows(comparisons)
    (args.output / "all_arms_vs_native.csv").write_text(buffer.getvalue())
    lines = ["# Corrected offline three-task results", "",
        "Verified complete primary author-validation pools. Offline replication is not end-to-end study completion.", "",
        "Reach: 33/33 official validation trajectories; Reach-Wall: 27/27; Push-T: 21/21.",
        "The seven added MW rows and 21 Push-T rows were explicitly authorized for frozen replication, not fresh confirmation.",
        "Broad unsteered coverage: 1,120/1,260 rows across all 42 MetaWorld tasks; 140 protected rows excluded.", "",
        "Positive percentages mean lower error than the same unsteered checkpoint on the same inputs.",
        "These are embedding errors, not physical-state errors or task-success rates. Rows below include ALL arms and controls.",
        "Intervals are paired-family simultaneous 95% intervals within each task/sweep/precision across registered contrasts and both H6 MSE endpoints.",
        "Percent intervals scale the difference interval by the observed native mean; they are not ratio-parameter confidence intervals.",
        "bfloat16 is primary; FP32 is a prespecified sensitivity analysis. Neither precision nor a winning arm is selected here.", ""]
    for precision in ("bfloat16", "float32"):
        lines.extend([f"## {precision}", "", "| Task / sweep | Arm | Proprio H6 reduction (95% interval) | Visual H6 reduction (95% interval) | Energy-match fraction |",
                      "|---|---|---:|---:|---:|"])
        for report in reports:
            if report["precision"] != precision:
                continue
            task = report["task"].removeprefix("mw-")
            for arm in report["arms"]:
                subset = {r["endpoint"]: r for r in comparisons if r["precision"] == precision and r["task"] == task
                          and r["category"] == report["category"] and r["arm"] == arm}
                cells = []
                for metric in ("proprio_mse_h6", "visual_mse_h6"):
                    r = subset[metric]
                    cells.append(f"{r['error_reduction_percent']:+.4f}% [{r['simultaneous_95_lower_percent']:+.4f}, {r['simultaneous_95_upper_percent']:+.4f}]")
                lines.append(f"| {task} / {report['category']} | {arm} | {cells[0]} | {cells[1]} | {100 * r['realized_energy_match_fraction']:.1f}% |")
        lines.append("")
    lines.extend(["## Interpretation limits", "",
        "An improvement over native alone does not establish a selective mechanism: inspect the registered matched-random, permutation, position and capacity contrasts in the detailed analysis reports.",
        "Energy mismatch under low-precision rounding prevents an exact equal-delivered-energy interpretation. The threshold is the fixed FP32 tolerance, not an outcome-selected tolerance.",
        "No trained-seed replication, decoded-state evaluation, routing/combined recipe, fresh confirmation or closed-loop success claim is supplied by these results.", ""])
    (args.output / "METRICS.md").write_text("\n".join(lines))
    write_json(args.output / "DONE.json", {name + "_sha256": sha256(args.output / name)
        for name in ("report.json", "all_arms_vs_native.csv", "METRICS.md")})


if __name__ == "__main__":
    main()
