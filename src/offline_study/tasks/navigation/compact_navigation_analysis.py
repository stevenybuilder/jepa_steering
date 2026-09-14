"""Preserve all aggregate navigation comparisons without committing raw nested rows."""
import argparse
import json
from pathlib import Path

from offline_study.evaluation.behavioral_development import verified_report
from offline_study.core.protocol import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report, digest = verified_report(args.source)
    if report["task"] not in ("wall", "pointmaze") or not report["status"].startswith("verified_author_corrected_"):
        raise ValueError("Expected completed verified navigation analysis")
    selected = {key: value for key, value in report.items() if key != "aggregation"}
    selected.update(full_analysis_report_sha256=digest, full_analysis_location=str(args.source),
                    raw_nested_aggregation_omitted_only=True, every_arm_and_contrast_retained=True)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "report.json", selected)
    write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json"),
        "full_analysis_report_sha256": digest, "verification_not_recomputed": True})


if __name__ == "__main__":
    main()
