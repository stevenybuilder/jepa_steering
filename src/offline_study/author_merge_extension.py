"""Verify and combine the original 53 MW rows with the seven authorized additions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .author_access import verify_runtime_access
from .author_analyze import analyze
from .author_evaluate import check_coverage, verify_fit
from .author_extension import FULL_COUNTS
from .author_runtime import examples, validate_cohort
from .protocol import sha256, summarize_metrics, write_json


def diagnostic_coverage(diagnostics, expected, category):
    keys = [(row["trajectory_id"], row["start"]) for row in diagnostics]
    if len(keys) != len(set(keys)) or set(keys) != {(r["trajectory_id"], r["start"]) for r in expected}:
        raise ValueError("Missing or duplicate full-pool mechanism diagnostics")
    required = {"vision_action_coupling": "visual_output_interaction_mse_h6",
                "action_response_geometry": "cubic_visual_native_fidelity_mse_h6"}.get(category)
    if required and any(required not in row["metrics"] for row in diagnostics):
        raise ValueError("A registered mechanism endpoint is missing")


def merge(original_root, extension_root, task, precision, category, output):
    cohort_path = extension_root / "cohorts" / task / "cohort.json"
    cohort = json.loads(cohort_path.read_text())
    validate_cohort(cohort)
    if not verify_runtime_access(cohort_path, cohort):
        raise ValueError("Missing authorized seven-row extension")
    fit = extension_root / "fits-v1" / precision / task / category
    receipt = json.loads((fit / "fit_receipt.json").read_text())
    receipt, protocol = verify_fit(fit, sha256(cohort_path), receipt["checkpoint_sha256"], precision)
    old_fit = original_root / "fits-v1" / precision / task / category
    source_binding = receipt["source_frozen_artifacts"]
    for name, key in (("protocol.json", "protocol_sha256"), ("fit_receipt.json", "fit_receipt_sha256"),
                      ("operator_bank.pt", "bank_sha256")):
        if sha256(old_fit / name) != source_binding[key]:
            raise ValueError("Original scientific fit differs from the extension")
    old_analysis = original_root / "analysis-v2" / precision / task / category
    old_done = json.loads((old_analysis / "DONE.json").read_text())
    if sha256(old_analysis / "report.json") != old_done["report_sha256"]:
        raise ValueError("Original analysis was changed")
    old = json.loads((old_analysis / "report.json").read_text())
    if (old["status"] != "verified_author_corrected_development_complete" or
            old["cohort_sha256"] != cohort["replication_access"]["original_cohort_sha256"] or
            old["protocol_sha256"] != source_binding["protocol_sha256"] or
            old["fit_receipt_sha256"] != source_binding["fit_receipt_sha256"] or
            old["precision"] != precision or old["category"] != category or old["task"] != cohort["task"]):
        raise ValueError("Original and extension scientific scopes disagree")
    inputs, measurements, diagnostics = {}, [], []
    old_metric_files = 0
    for filename, expected_hash in old["input_sha256"].items():
        path = Path(filename)
        if not path.is_relative_to(original_root) or sha256(path) != expected_hash:
            raise ValueError("Original verified shard changed or is outside the source run")
        inputs[str(path)] = expected_hash
        if path.name == "window_metrics.json":
            old_metric_files += 1
            measurements.extend(json.loads(path.read_text()))
        elif path.name == "mechanism_diagnostics.json":
            diagnostics.extend(json.loads(path.read_text()))
    if old_metric_files != 2:
        raise ValueError("Original two-shard evidence is incomplete")
    names = [arm["name"] for arm in protocol["arms"]]
    config = cohort["reference_config"]
    reused = examples(cohort["reused_evaluation"], config)
    check_coverage(measurements, reused, names)
    extra_measurements, extra_diagnostics = [], []
    shard_root = extension_root / "evaluation-v1" / precision / task / category
    directories = sorted(shard_root.glob("shard-*"))
    if len(directories) != 2:
        raise ValueError("Both extension shards are required")
    for directory in directories:
        done = json.loads((directory / "DONE.json").read_text())
        for name in ("report", "window_metrics", "selection", "protocol", "mechanism_diagnostics"):
            path = directory / (name + ".json")
            if sha256(path) != done[name + "_sha256"]:
                raise ValueError("Extension shard checksum mismatch")
            inputs[str(path)] = sha256(path)
        report = json.loads((directory / "report.json").read_text())
        for key, expected in (("cohort_sha256", sha256(cohort_path)), ("protocol_sha256", sha256(fit / "protocol.json")),
                              ("fit_receipt_sha256", sha256(fit / "fit_receipt.json")), ("precision", precision),
                              ("checkpoint_sha256", receipt["checkpoint_sha256"]), ("category", category),
                              ("task", cohort["task"]), ("access_authorization_sha256", cohort["access_authorization_sha256"]),
                              ("evaluation_role", "author_replication"), ("protected_outcomes_accessed", True),
                              ("zero_dose_identity", True), ("native_instrumentation_fidelity", True)):
            if report.get(key) != expected:
                raise ValueError("Extension receipt disagrees: " + key)
        extra_measurements.extend(json.loads((directory / "window_metrics.json").read_text()))
        extra_diagnostics.extend(json.loads((directory / "mechanism_diagnostics.json").read_text()))
    added = examples(cohort["evaluation"], config, role="author_replication")
    check_coverage(extra_measurements, added, names)
    measurements.extend(extra_measurements)
    diagnostics.extend(extra_diagnostics)
    check_coverage(measurements, reused + added, names)
    diagnostic_coverage(diagnostics, reused + added, category)
    result = analyze(measurements, protocol, receipt, cohort["task"])
    result["mechanism_diagnostics"] = summarize_metrics(diagnostics)
    result.update(status="verified_full_primary_author_split_replication_complete", task=cohort["task"],
        category=category, precision=precision, rollout_trajectories=FULL_COUNTS[cohort["task"]],
        prefix_rollouts_per_arm=len(reused) + len(added),
        coverage={"evaluated_rows": FULL_COUNTS[cohort["task"]], "official_rows": FULL_COUNTS[cohort["task"]],
                  "complete_author_split": True, "reused_rows": len(cohort["reused_evaluation"]),
                  "newly_evaluated_rows": len(cohort["evaluation"])},
        protected_outcomes_accessed=True, fresh_confirmation=False, old_53_rows_repeated=False,
        access_authorization_sha256=cohort["access_authorization_sha256"],
        input_sha256=inputs, original_analysis_sha256=old_done["report_sha256"],
        original_protocol_sha256=source_binding["protocol_sha256"],
        protocol_sha256=sha256(fit / "protocol.json"), fit_receipt_sha256=sha256(fit / "fit_receipt.json"),
        extension_cohort_sha256=sha256(cohort_path))
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "report.json", result)
    write_json(output / "DONE.json", {"status": result["status"], "report_sha256": sha256(output / "report.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original-root", "extension-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("task", "precision", "category"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    merge(args.original_root, args.extension_root, args.task, args.precision, args.category, args.output)


if __name__ == "__main__":
    main()
