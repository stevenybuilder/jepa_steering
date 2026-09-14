"""Verify every PointMaze training/validation file against the pinned raw ZIP.

CPU-only input provenance; no learned-model outcomes or confirmation access.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile

from offline_study.evaluation.behavioral_development import verified_report
from offline_study.core.protocol import sha256, write_json

ARCHIVE_SHA = "6c48ccf22c90b9af8dcf0e2cd70849aec8dd8e214ac5f1f09552bf8bc9494acc"
REVISION = "6116f042ae7ae4c8e3f1fd2f194f432615664182"


def required_files():
    return ["states.pth", "actions.pth", "seq_lengths.pth"] + [
        f"obses/episode_{i:03d}.pth" for i in range(2000)]


def verify_inputs(root, receipt):
    report, digest = verified_report(receipt)
    protocol = json.loads((receipt / "protocol.json").read_text())
    if (report["status"] != "complete_pointmaze_inputs_match_pinned_archive" or
            report["protocol_sha256"] != sha256(receipt / "protocol.json") or
            report["files_sha256"] != sha256(receipt / "files.json") or
            protocol["archive_sha256"] != ARCHIVE_SHA or protocol["revision"] != REVISION):
        raise ValueError("Unverified complete PointMaze input binding")
    files = json.loads((receipt / "files.json").read_text())
    if set(files) != set(required_files()):
        raise ValueError("Incomplete PointMaze raw-file population")
    for name, row in files.items():
        if (root / name).stat().st_size != row["bytes"] or sha256(root / name) != row["sha256"]:
            raise ValueError("PointMaze input differs from pinned archive: " + name)
    return report, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "input-check", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    assets, assets_hash = verified_report(args.assets)
    checked, check_hash = verified_report(args.input_check)
    ip = json.loads((args.input_check / "protocol.json").read_text())
    if (assets["status"] != "official_navigation_assets_staged_and_verified" or
            checked["status"] != "navigation_metadata_and_selected_frame_parity_passed" or
            checked["protocol_sha256"] != sha256(args.input_check / "protocol.json") or
            checked["task_reports_sha256"]["pointmaze"] != sha256(args.input_check / "pointmaze.json") or
            ip["assets_report_sha256"] != assets_hash):
        raise ValueError("Original navigation input/source receipts changed")
    selected = [r for r in assets["assets"] if r["task"] == "pointmaze" and r["kind"] == "dataset_archive"]
    if len(selected) != 1:
        raise ValueError("Ambiguous PointMaze archive")
    archive = selected[0]
    archive_path = Path(archive["downloaded_path"])
    if (archive["sha256"] != ARCHIVE_SHA or archive["revision"] != REVISION or
            sha256(archive_path) != ARCHIVE_SHA or archive_path.stat().st_size != 718363945):
        raise ValueError("Pinned official PointMaze archive changed")
    args.output.mkdir(parents=True, exist_ok=False)
    started, files = time.monotonic(), {}
    write_json(args.output / "protocol.json", {
        "role": "complete_source_training_inputs_no_model_outcomes", "task": "pointmaze",
        "archive_sha256": ARCHIVE_SHA, "revision": REVISION, "source_sha256": sha256(Path(__file__)),
        "assets_report_sha256": assets_hash, "input_check_report_sha256": check_hash,
        "pointmaze_input_report_sha256": sha256(args.input_check / "pointmaze.json"), "files": required_files()})
    root = args.assets / "extracted/pointmaze/point_maze"
    try:
        with zipfile.ZipFile(archive_path) as zipped:
            names = zipped.namelist()
            for name in required_files():
                member = "point_maze/" + name
                if names.count(member) != 1:
                    raise ValueError("Missing/duplicate official member: " + member)
                expected = hashlib.sha256()
                with zipped.open(member) as stream:
                    for block in iter(lambda: stream.read(8 << 20), b""):
                        expected.update(block)
                actual = sha256(root / name)
                if actual != expected.hexdigest():
                    raise ValueError("Extracted PointMaze file differs from source: " + name)
                files[name] = {"sha256": actual, "bytes": (root / name).stat().st_size}
                if len(files) % 100 == 0:
                    progress = {"verified_files": len(files), "target": 2003, "seconds": time.monotonic()-started}
                    write_json(args.output / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
        write_json(args.output / "files.json", files)
        write_json(args.output / "report.json", {
            "status": "complete_pointmaze_inputs_match_pinned_archive",
            "protocol_sha256": sha256(args.output / "protocol.json"),
            "files_sha256": sha256(args.output / "files.json"), "files": len(files),
            "bytes": sum(r["bytes"] for r in files.values()), "trajectory_videos": 2000,
            "training_trajectories": 1800, "validation_trajectories": 200,
            "native_config_sha256": ip["native_configs_sha256"]["pointmaze"],
            "archive_sha256": ARCHIVE_SHA, "seconds": time.monotonic()-started,
            "validation_outcomes_accessed": False, "training_started": False})
        write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "verified_files": len(files)})
        raise


if __name__ == "__main__":
    main()
