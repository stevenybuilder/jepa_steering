"""Stage only pinned official Maze/Wall assets; no model or outcome evaluation."""
import argparse
import json
import os
import shutil
import stat
import time
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from offline_study.core.protocol import sha256, write_json


def safe_members(archive, max_bytes=64 * 1024**3):
    members = archive.infolist()
    names = []
    for member in members:
        path = PurePosixPath(member.filename)
        if (path.is_absolute() or ".." in path.parts or "\\" in member.filename or
                stat.S_ISLNK(member.external_attr >> 16)):
            raise ValueError("Unsafe archive member: " + member.filename)
        names.append(str(path))
    if len(names) != len(set(names)) or sum(m.file_size for m in members) > max_bytes:
        raise ValueError("Duplicated archive path or unexpected extracted size")
    return members


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-downloads", type=Path, action="append", default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    completed = []
    try:
        from huggingface_hub import hf_hub_download
        spec = json.loads(args.manifest.read_text())
        if spec["role"] != "official_asset_staging_not_evaluation_or_confirmation":
            raise ValueError("Unexpected staging scope")
        if {(x["task"], x["kind"]) for x in spec["assets"]} != {
                (task, kind) for task in ("pointmaze", "wall") for kind in ("checkpoint", "dataset_archive")} or len(spec["assets"]) != 4:
            raise ValueError("Expected exactly the four approved navigation assets")
        if shutil.disk_usage(args.output).free < 12 * 1024**3:
            raise ValueError("Need at least 12 GiB free for verified archives and extraction")
        write_json(args.output / "protocol.json", {"manifest": spec, "manifest_sha256": sha256(args.manifest),
            "scientific_evaluation_authorized": False, "checkpoint_deserialization": False,
            "source_sha256": sha256(Path(__file__))})
        for entry in spec["assets"]:
            write_json(args.output / "progress.json", {"downloading": entry["filename"], "completed": completed})
            path = args.output / "downloads" / entry["repo_type"] / entry["filename"]
            reused = None
            for root in args.reuse_downloads:
                prior = root / entry["repo_type"] / entry["filename"]
                if (prior.is_file() and prior.stat().st_size == entry["size"] and
                        sha256(prior) == entry["sha256"]):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    os.link(prior, path)  # Immutable verified input; never edit either link.
                    reused = str(prior)
                    break
            if reused is None:
                path = Path(hf_hub_download(repo_id=entry["repo_id"], repo_type=entry["repo_type"],
                    revision=entry["revision"], filename=entry["filename"], local_dir=args.output / "downloads" / entry["repo_type"]))
            if path.stat().st_size != entry["size"] or sha256(path) != entry["sha256"]:
                raise ValueError("Official asset checksum/size mismatch: " + entry["filename"])
            result = {**entry, "downloaded_path": str(path), "reused_verified_input": reused}
            if entry["kind"] == "dataset_archive":
                extraction = args.output / "extracted" / entry["task"]
                extraction.mkdir(parents=True, exist_ok=False)
                with ZipFile(path) as archive:
                    members = safe_members(archive)
                    if shutil.disk_usage(extraction).free < sum(m.file_size for m in members) + 8 * 1024**3:
                        raise ValueError("Insufficient space for actual archive expansion plus 8 GiB reserve")
                    archive.extractall(extraction, members=members)
                result.update(extracted_root=str(extraction), archive_members=len(members),
                              extracted_bytes=sum(m.file_size for m in members))
            completed.append(result)
            print(json.dumps({"verified_asset": entry["filename"], "sha256": entry["sha256"]}), flush=True)
        write_json(args.output / "report.json", {"status": "official_navigation_assets_staged_and_verified",
            "protocol_sha256": sha256(args.output / "protocol.json"), "assets": completed,
            "seconds": time.monotonic() - started, "model_runs": 0, "training_jobs": 0,
            "lineage_and_loader_audit_complete": False, "confirmation_authorized": False})
        write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "completed": completed})
        raise


if __name__ == "__main__":
    main()
