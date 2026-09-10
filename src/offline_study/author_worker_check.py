"""Export a fit-only input fixture and verify the real model on a new worker.

This is a runtime compatibility check, never an efficacy measurement. No validation
or confirmation dataset is opened by either command.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch

from .author_input_parity import IdentityEncoder
from .author_runtime import AuthorBackend, encoded_batches, open_normalized_dataset, validate_cohort
from .protocol import sha256, write_json
from .vendor import use_vendor


def export_fixture(vendor, cohort_path, data_root, output):
    use_vendor(vendor)
    cohort = json.loads(cohort_path.read_text())
    validate_cohort(cohort)
    selected = cohort["fit"][:1]
    dataset = open_normalized_dataset(cohort["dataset"], data_root, cohort["reference_config"], True)
    metadata, batch = next(encoded_batches(IdentityEncoder(), dataset, selected,
                                          cohort["reference_config"], fitting=True))
    if any(item["split"] != "fit" for group in metadata for item in group):
        raise ValueError("Worker fixture must contain fit-only examples")
    output.mkdir(parents=True, exist_ok=False)
    torch.save(batch, output / "inputs.pt")
    write_json(output / "cohort.json", cohort)
    write_json(output / "report.json", {
        "status": "fit_only_runtime_fixture", "dataset": cohort["dataset"],
        "metadata": metadata, "selected": selected,
        "cohort_sha256": sha256(output / "cohort.json"),
        "source_cohort_sha256": sha256(cohort_path),
        "inputs_sha256": sha256(output / "inputs.pt"),
        "development_or_protected_outcomes_accessed": False,
    })
    write_json(output / "DONE.json", {"report_sha256": sha256(output / "report.json")})


@torch.no_grad()
def check_worker(vendor, checkpoint, checkpoint_hash, fixture, output):
    done = json.loads((fixture / "DONE.json").read_text())
    if sha256(fixture / "report.json") != done["report_sha256"]:
        raise ValueError("Fixture receipt changed")
    report = json.loads((fixture / "report.json").read_text())
    for filename, key in (("inputs.pt", "inputs_sha256"), ("cohort.json", "cohort_sha256")):
        if sha256(fixture / filename) != report[key]:
            raise ValueError("Fixture checksum mismatch: " + filename)
    cohort = json.loads((fixture / "cohort.json").read_text())
    validate_cohort(cohort)
    if report["selected"] != cohort["fit"][:1] or report["development_or_protected_outcomes_accessed"]:
        raise ValueError("Fixture is not the verified first fit-only family")
    if any(item["split"] != "fit" or item["trajectory_id"] != report["selected"][0]["trajectory_id"]
           for group in report["metadata"] for item in group):
        raise ValueError("Unexpected fixture membership")
    inputs = torch.load(fixture / "inputs.pt", weights_only=True, map_location="cpu")
    output.mkdir(parents=True, exist_ok=False)
    checks = {}
    for precision in ("bfloat16", "float32"):
        backend = AuthorBackend(vendor, checkpoint, checkpoint_hash, report["dataset"], "cuda:0", precision)
        encoded = backend.encode_clip({key: inputs[key] for key in ("visual", "proprio")}, inputs["action"])
        checks[precision] = backend.verify_reference(encoded)
        del encoded, backend
        torch.cuda.empty_cache()
    write_json(output / "report.json", {
        "status": "real_fit_only_worker_rollout_and_metric_parity_passed",
        "checks": checks, "fixture_report_sha256": done["report_sha256"],
        "checkpoint_sha256": checkpoint_hash, "python": platform.python_version(),
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "development_or_protected_outcomes_accessed": False,
        "scientific_efficacy_measurement": False,
    })
    write_json(output / "DONE.json", {"report_sha256": sha256(output / "report.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["export", "check"])
    for flag in ("vendor", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    for flag in ("cohort", "data-root", "fixture", "checkpoint"):
        parser.add_argument("--" + flag, type=Path)
    parser.add_argument("--checkpoint-sha256")
    args = parser.parse_args()
    if args.stage == "export":
        if args.cohort is None or args.data_root is None:
            parser.error("export requires --cohort and --data-root")
        export_fixture(args.vendor, args.cohort, args.data_root, args.output)
    else:
        if args.fixture is None or args.checkpoint is None or args.checkpoint_sha256 is None:
            parser.error("check requires fixture and checkpoint identity")
        check_worker(args.vendor, args.checkpoint, args.checkpoint_sha256, args.fixture, args.output)


if __name__ == "__main__":
    main()
