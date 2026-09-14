"""Verified-input wrapper around the unchanged full offline intervention evaluator."""
import argparse
import json
import sys
from pathlib import Path

from offline_study.evaluation.checkpoint.author_evaluate import main as evaluate_main
from offline_study.tasks.navigation.navigation_cohort import COUNTS, verify_inputs
from offline_study.tasks.navigation.navigation_input_check import CONFIGS
from offline_study.tasks.navigation.navigation_smoke import CHECKPOINTS
from offline_study.core.protocol import sha256
from offline_study.models.vendor import use_vendor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vendor", "checkpoint", "cohort", "data-root", "fit", "baseline", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--precision", choices=("bfloat16", "float32"), required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    args = parser.parse_args()
    use_vendor(args.vendor)
    cohort = verify_inputs(args.cohort, args.data_root)
    task = cohort["task"]
    done = json.loads((args.baseline / "DONE.json").read_text())
    report = json.loads((args.baseline / "report.json").read_text())
    bp = json.loads((args.baseline / "protocol.json").read_text())
    if (task not in COUNTS or cohort["upstream_config_sha256"] != sha256(args.vendor / CONFIGS[task]) or
            sha256(args.baseline / "report.json") != done["report_sha256"] or
            report["status"] != "full_author_navigation_native_forecast_complete" or
            report["protocol_sha256"] != sha256(args.baseline / "protocol.json") or
            report["parity_sha256"] != sha256(args.baseline / "PARITY.json") or
            report["window_metrics_sha256"] != sha256(args.baseline / "window_metrics.json") or
            bp["cohort_sha256"] != sha256(args.cohort) or bp["checkpoint_sha256"] != CHECKPOINTS[task] or
            report["prefixes"] != COUNTS[task][3] * 2 or report["precision"] != args.precision):
        raise ValueError("Missing complete native reference and source parity for this exact task/cohort/precision")
    sys.argv = ["author_evaluate", "--vendor", str(args.vendor), "--checkpoint", str(args.checkpoint),
        "--cohort", str(args.cohort), "--data-root", str(args.data_root), "--fit", str(args.fit),
        "--output", str(args.output), "--precision", args.precision, "--device", "cuda:0",
        "--checkpoint-sha256", CHECKPOINTS[task], "--shard-index", str(args.shard_index),
        "--shard-count", str(args.shard_count)]
    evaluate_main()


if __name__ == "__main__":
    main()
