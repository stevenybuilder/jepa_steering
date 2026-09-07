"""Verify the relocated primary offline bundle, including raw fit captures.

This is a durability audit. It does not change scientific protocols or claim the
subsequent combined, confirmation or policy stages have finished.
"""
import argparse
import json
import time
from pathlib import Path

from .author_evaluate import verify_fit
from .author_summary import CATEGORIES, COUNTS
from .planning_native_smoke import CHECKPOINTS
from .protocol import sha256, write_json
from .verify_pusht_replica import verify as verify_pusht


ORIGINAL = Path("/workspace/jepa-runtime/author-correction-20260907")
MW = Path("/workspace/jepa-runtime/metaworld-author-extension-20260907")
PUSHT = Path("/workspace/jepa-runtime/pusht-author-replication-20260907")


def relocated(path, root):
    path = Path(path)
    if ".." in path.parts:
        raise ValueError("Unsafe artifact reference")
    for source, destination in ((ORIGINAL, root), (MW, root / MW.name), (PUSHT, root / PUSHT.name)):
        if path.is_relative_to(source):
            return destination / path.relative_to(source)
    raise ValueError("Unexpected external artifact reference: " + str(path))


def verify(root):
    checked = {}

    def check(path, expected):
        relative = str(path.relative_to(root))
        if relative not in checked:
            checked[relative] = sha256(path)
        if checked[relative] != expected:
            raise ValueError("Durable artifact checksum mismatch: " + relative)

    def report(directory):
        done = json.loads((directory / "DONE.json").read_text())
        check(directory / "report.json", done["report_sha256"])
        return json.loads((directory / "report.json").read_text())

    for precision in ("bfloat16", "float32"):
        for task in COUNTS:
            fit_root = root / "fits-v1" / precision / task
            done = json.loads((fit_root / "DONE.json").read_text())
            for name, key in (("native_fit.pt", "native_fit_sha256"), ("cohort.json", "cohort_sha256"),
                              ("fit_receipt.json", "fit_receipt_sha256")):
                check(fit_root / name, done[key])
            cohort_hash = sha256(root / "cohorts" / task / "cohort.json")
            checkpoint = CHECKPOINTS["pusht" if task == "pusht" else "metaworld"]
            for category in CATEGORIES:
                fit = fit_root / category
                verify_fit(fit, cohort_hash, checkpoint, precision)
                done = json.loads((fit / "DONE.json").read_text())
                for name, key in (("protocol.json", "protocol_sha256"), ("operator_bank.pt", "operator_bank_sha256"),
                                  ("fit_receipt.json", "fit_receipt_sha256")):
                    check(fit / name, done[key])
    # This also checks the authorized Push-T cohort and both relocated fit banks.
    for relative, expected in verify_pusht(root / PUSHT.name).items():
        check(root / PUSHT.name / relative, expected)
    for precision in ("bfloat16", "float32"):
        for task in ("reach", "reach-wall"):
            cohort = root / MW.name / "cohorts" / task / "cohort.json"
            for category in CATEGORIES:
                fit = root / MW.name / "fits-v1" / precision / task / category
                verify_fit(fit, sha256(cohort), CHECKPOINTS["metaworld"], precision)
                done = json.loads((fit / "DONE.json").read_text())
                for name, key in (("protocol.json", "protocol_sha256"), ("operator_bank.pt", "operator_bank_sha256"),
                                  ("fit_receipt.json", "fit_receipt_sha256")):
                    check(fit / name, done[key])
    closure = root / "three-task-offline-closure-20260907"
    final = report(closure)
    if final["full_study_complete"] or final["trajectories"] != COUNTS:
        raise ValueError("Closure scope changed")
    check(closure / "metrics/report.json", final["metrics_sha256"])
    check(closure / "advancement/report.json", final["advancement_sha256"])
    metrics = json.loads((closure / "metrics/report.json").read_text())
    done = json.loads((closure / "metrics/DONE.json").read_text())
    for name in ("report.json", "all_arms_vs_native.csv", "METRICS.md"):
        check(closure / "metrics" / name, done[name + "_sha256"])
    scopes = set()
    for binding in metrics["verified_analysis_scopes"]:
        path = relocated(binding["path"], root)
        check(path, binding["sha256"])
        analysis = report(path.parent)
        task = analysis["task"].removeprefix("mw-")
        scope = (task, analysis["category"], analysis["precision"])
        if scope in scopes or analysis["rollout_trajectories"] != COUNTS[task] or analysis["fresh_confirmation"]:
            raise ValueError("Wrong or repeated primary analysis scope")
        scopes.add(scope)
        for original, expected in analysis["input_sha256"].items():
            check(relocated(original, root), expected)
    expected_scopes = {(t, c, p) for t in COUNTS for c in CATEGORIES for p in ("bfloat16", "float32")}
    if scopes != expected_scopes:
        raise ValueError("Missing complete primary analysis scope")
    for precision in ("bfloat16", "float32"):
        directory = root / "broad-baseline-v1" / precision
        done = json.loads((directory / "DONE.json").read_text())
        for name in ("report", "selection", "window_metrics", "PARITY"):
            check(directory / (name + ".json"), done[name + "_sha256"])
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--wait-seconds", type=int, default=0,
                        help="Wait for atomically installed raw fit captures from the active transfer")
    args = parser.parse_args()
    if not 0 <= args.wait_seconds <= 3600:
        parser.error("Durability wait must be between zero and one hour")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        started = time.monotonic()
        captures = [args.root / "fits-v1" / precision / task / "native_fit.pt"
                    for precision in ("bfloat16", "float32") for task in COUNTS]
        while args.wait_seconds and not all(path.is_file() for path in captures):
            elapsed = time.monotonic() - started
            if elapsed > args.wait_seconds:
                raise TimeoutError("Raw capture transfer did not complete within the durability wait")
            write_json(args.output / "progress.json", {"status": "waiting_for_raw_fit_transfer",
                "installed_captures": sum(p.is_file() for p in captures), "expected_captures": 6,
                "seconds": elapsed, "durability_verified": False})
            time.sleep(10)
        hashes = verify(args.root)
        write_json(args.output / "report.json", {"status": "durable_full_primary_offline_bundle_verified",
            "local_root": str(args.root.resolve()), "artifact_sha256": hashes,
            "complete_primary_analysis_scopes": 30, "raw_native_fit_captures_verified": 6,
            "both_broad_baseline_precisions_verified": True, "full_study_complete": False,
            "fresh_confirmation": False, "sources_preserved": True})
        write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "durability_verified": False})
        raise


if __name__ == "__main__":
    main()
