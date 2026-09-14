"""Bind the complete Wall training/validation inputs to the pinned official archive."""
import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

from offline_study.evaluation.behavioral_development import verified_report
from offline_study.tasks.navigation.navigation_input_check import CONFIGS
from offline_study.core.protocol import sha256, write_json


def required_wall_files():
    return ["states.pth", "actions.pth", "door_locations.pth", "wall_locations.pth"] + [
        f"obses/episode_{index:03d}.pth" for index in range(1920)]


def verify_inputs(root, receipt):
    report, digest = verified_report(receipt)
    files_path = receipt / "files.json"
    if (report["status"] != "complete_wall_inputs_match_pinned_archive" or report["files_sha256"] != sha256(files_path) or
            report["protocol_sha256"] != sha256(receipt / "protocol.json")):
        raise ValueError("Missing full official training input binding")
    files = json.loads(files_path.read_text())
    if set(files) != set(required_wall_files()):
        raise ValueError("Incomplete Wall input population")
    for name, expected in files.items():
        if sha256(root / name) != expected["sha256"] or (root / name).stat().st_size != expected["bytes"]:
            raise ValueError("Staged Wall input differs from pinned source: " + name)
    return report, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "input-check", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    assets, asset_hash = verified_report(args.assets)
    inputs, input_hash = verified_report(args.input_check)
    ip = json.loads((args.input_check / "protocol.json").read_text())
    if (assets["status"] != "official_navigation_assets_staged_and_verified" or
            inputs["status"] != "navigation_metadata_and_selected_frame_parity_passed" or
            ip["assets_report_sha256"] != asset_hash or inputs["protocol_sha256"] != sha256(args.input_check / "protocol.json")):
        raise ValueError("Original source input provenance changed")
    archive = next(r for r in assets["assets"] if r["task"] == "wall" and r["kind"] == "dataset_archive")
    archive_path = Path(archive["downloaded_path"])
    if archive["sha256"] != "2b4ae4ed0ad03b337efac637f17752e7e7e27f864fec39dc25b51fef490c980d" or sha256(archive_path) != archive["sha256"]:
        raise ValueError("Official Wall archive changed")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "protocol.json", {"role": "complete_source_training_inputs_no_model_outcomes",
        "assets_report_sha256": asset_hash, "input_check_report_sha256": input_hash,
        "archive_sha256": archive["sha256"], "revision": archive["revision"],
        "task": "wall", "files": required_wall_files(), "source_sha256": sha256(Path(__file__))})
    started, files = time.monotonic(), {}
    raw = args.assets / "extracted/wall/wall_single"
    try:
        with zipfile.ZipFile(archive_path) as zipped:
            names = zipped.namelist()
            for name in required_wall_files():
                member = "wall_single/" + name
                if names.count(member) != 1:
                    raise ValueError("Missing/duplicate native source member: " + member)
                expected = hashlib.sha256()
                with zipped.open(member) as stream:
                    for chunk in iter(lambda: stream.read(8 << 20), b""):
                        expected.update(chunk)
                actual = sha256(raw / name)
                if actual != expected.hexdigest():
                    raise ValueError("Extracted file differs from actual pinned archive: " + name)
                files[name] = {"sha256": actual, "bytes": (raw / name).stat().st_size}
                if len(files) % 100 == 0:
                    progress = {"verified_files": len(files), "target": 1924, "seconds": time.monotonic() - started}
                    write_json(args.output / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
        write_json(args.output / "files.json", files)
        write_json(args.output / "report.json", {"status": "complete_wall_inputs_match_pinned_archive",
            "protocol_sha256": sha256(args.output / "protocol.json"), "files_sha256": sha256(args.output / "files.json"),
            "files": len(files), "bytes": sum(row["bytes"] for row in files.values()),
            "trajectory_videos": 1920, "training_trajectories": 1728, "validation_trajectories": 192,
            "native_config_sha256": ip["native_configs_sha256"]["wall"],
            "archive_sha256": archive["sha256"], "seconds": time.monotonic() - started,
            "validation_outcomes_accessed": False, "training_started": False})
        write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "verified_files": len(files)})
        raise


if __name__ == "__main__":
    main()
