"""Verify relocated Push-T BF16 evidence before releasing its originating worker."""
import argparse
import json
from pathlib import Path

from offline_study.evaluation.checkpoint.author_access import CATEGORIES, verify_runtime_access
from offline_study.evaluation.checkpoint.author_evaluate import verify_fit
from offline_study.evaluation.checkpoint.author_runtime import validate_cohort
from offline_study.core.protocol import sha256, write_json


SOURCE_ROOT = Path("/workspace/jepa-runtime/pusht-author-replication-20260907")
CHECKPOINT = "9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb"


def verify(root):
    cohort_path = root / "cohorts/pusht/cohort.json"
    cohort = json.loads(cohort_path.read_text())
    validate_cohort(cohort)
    if not verify_runtime_access(cohort_path, cohort) or len(cohort["evaluation"]) != 21:
        raise ValueError("Not the authorized full Push-T replication pool")
    checked = {}
    def check(path, expected):
        relative = str(path.relative_to(root))
        if relative not in checked:
            checked[relative] = sha256(path)
        if checked[relative] != expected:
            raise ValueError("Copied artifact changed: " + relative)
    for precision in ("bfloat16", "float32"):
        for category in CATEGORIES:
            fit = root / "fits-v1" / precision / "pusht" / category
            verify_fit(fit, sha256(cohort_path), CHECKPOINT, precision)
            done = json.loads((fit / "DONE.json").read_text())
            for name, key in (("operator_bank.pt", "operator_bank_sha256"), ("fit_receipt.json", "fit_receipt_sha256"),
                              ("protocol.json", "protocol_sha256")):
                check(fit / name, done[key])
    for category in CATEGORIES:
        fit = root / "fits-v1/bfloat16/pusht" / category
        analysis = root / "analysis-v1/bfloat16/pusht" / category
        done = json.loads((analysis / "DONE.json").read_text())
        check(analysis / "report.json", done["report_sha256"])
        report = json.loads((analysis / "report.json").read_text())
        if (report["status"] != "verified_author_corrected_replication_complete" or report["task"] != "pusht" or
                report["precision"] != "bfloat16" or report["category"] != category or report["rollout_trajectories"] != 21 or
                report["fresh_confirmation"] or report["cohort_sha256"] != sha256(cohort_path) or
                report["protocol_sha256"] != sha256(fit / "protocol.json")):
            raise ValueError("Copied analysis scope does not match")
        for original, expected in report["input_sha256"].items():
            path = Path(original)
            if not path.is_relative_to(SOURCE_ROOT):
                raise ValueError("Unexpected external source in copied evidence")
            check(root / path.relative_to(SOURCE_ROOT), expected)
        for shard in range(2):
            directory = root / "evaluation-v1/bfloat16/pusht" / category / f"shard-{shard:03d}"
            receipt = json.loads((directory / "DONE.json").read_text())
            for name in ("report", "window_metrics", "selection", "protocol", "mechanism_diagnostics"):
                check(directory / (name + ".json"), receipt[name + "_sha256"])
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    hashes = verify(args.root)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "report.json", {"status": "relocated_pusht_bfloat16_replica_verified",
        "original_root": str(SOURCE_ROOT), "local_root": str(args.root.resolve()), "source_instance": 50159352,
        "artifacts_sha256": hashes, "bf16_analysis_scopes": 5, "bf16_completed_shards": 10,
        "both_precision_fit_banks_verified": True, "fresh_confirmation": False})
    write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})


if __name__ == "__main__":
    main()
