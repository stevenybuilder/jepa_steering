"""Download pinned author Franka evaluation assets, not the raw DROID training corpus.

No model/checkpoint deserialization or outcome evaluation. Small Git-hosted HDF5
companions have Git blob SHA1 identities; large LFS files have content SHA256.
Every downloaded file additionally receives a local SHA256 receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from .protocol import sha256, write_json


DATA_REVISION = "6116f042ae7ae4c8e3f1fd2f194f432615664182"
MODEL_REVISION = "9b9c41ef249466630dbf1a20e78391865d07b3b9"
CHECKPOINT_SHA256 = "daa69198aef764932f1cb809239a4e19c71da20a93c6a0b9f3869cb30a13f4aa"


def validate_manifest(spec):
    if (spec.get("role") != "official_droid_asset_staging_not_evaluation_or_confirmation" or
            spec.get("dataset_repo_id") != "facebook/jepa-wms" or
            spec.get("model_repo_id") != "facebook/jepa-wms" or
            spec.get("dataset_revision") != DATA_REVISION or
            spec.get("model_revision") != MODEL_REVISION):
        raise ValueError("Unexpected DROID asset scope or revision")
    entries = spec["assets"]
    identities = [(entry["repo_type"], entry["filename"]) for entry in entries]
    if len(entries) != 33 or len(set(identities)) != 33:
        raise ValueError("Expected 32 unique recording files and one checkpoint")
    recordings, companions = set(), set()
    for entry in entries:
        path = PurePosixPath(entry["filename"])
        if (path.is_absolute() or ".." in path.parts or "\\" in entry["filename"] or
                str(path) != entry["filename"] or entry["size"] <= 0):
            raise ValueError("Unsafe asset path or size")
        checksum_type = entry["checksum_type"]
        length = {"sha256": 64, "git_blob_sha1": 40}.get(checksum_type)
        if length is None or re.fullmatch(f"[0-9a-f]{{{length}}}", entry["checksum"]) is None:
            raise ValueError("Invalid asset checksum")
        if entry["repo_type"] == "model":
            if (entry["filename"] != "jepa_wm_droid.pth.tar" or entry["kind"] != "checkpoint" or
                    checksum_type != "sha256" or entry["checksum"] != CHECKPOINT_SHA256):
                raise ValueError("Unexpected checkpoint")
        elif entry["repo_type"] == "dataset" and entry["filename"].startswith("franka_custom/data/"):
            if path.name == "episode.h5" and entry["kind"] == "evaluation_recording":
                recordings.add(str(path.parent))
            elif path.name == "trajectory.hdf5" and entry["kind"] == "recording_companion":
                companions.add(str(path.parent))
            else:
                raise ValueError("Unexpected recording file")
        else:
            raise ValueError("RoboCasa/full-training downloads are outside this manifest")
    if len(recordings) != 16 or recordings != companions:
        raise ValueError("Missing or duplicated recording companion")
    return sum(entry["size"] for entry in entries)


def verify_download(path, entry):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != entry["size"]:
        raise ValueError("Asset type/size mismatch: " + entry["filename"])
    content_hash = hashlib.sha256()
    git_hash = hashlib.sha1(f"blob {entry['size']}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            content_hash.update(chunk)
            git_hash.update(chunk)
    actual = content_hash.hexdigest() if entry["checksum_type"] == "sha256" else git_hash.hexdigest()
    if actual != entry["checksum"]:
        raise ValueError("Asset checksum mismatch: " + entry["filename"])
    return {**entry, "downloaded_path": str(path), "verified_content_sha256": content_hash.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    spec = json.loads(args.manifest.read_text())
    total_bytes = validate_manifest(spec)
    args.output.mkdir(parents=True, exist_ok=False)
    completed = []
    started = time.monotonic()
    try:
        from huggingface_hub import hf_hub_download
        if shutil.disk_usage(args.output).free < total_bytes + 8 * 1024**3:
            raise ValueError("Need full download size plus 8 GiB free-space reserve")
        write_json(args.output / "protocol.json", {"manifest": spec,
            "manifest_sha256": sha256(args.manifest), "source_sha256": sha256(Path(__file__)),
            "parallel_downloads": 2, "model_runs": 0, "checkpoint_deserialization": False,
            "training_corpus_downloaded": False, "confirmation_authorized": False})

        def download(entry):
            kind = entry["repo_type"]
            path = Path(hf_hub_download(repo_id=spec[kind + "_repo_id"], repo_type=kind,
                revision=spec[kind + "_revision"], filename=entry["filename"],
                local_dir=args.output / "downloads" / kind))
            return verify_download(path, entry)

        # Separate files and bounded workers; no shared mutable HDF5 datasets.
        with ThreadPoolExecutor(max_workers=2) as pool:
            for result in pool.map(download, spec["assets"]):
                completed.append(result)
                write_json(args.output / "progress.json", {"verified_files": len(completed),
                    "requested_files": len(spec["assets"]), "last_verified": result["filename"]})
                print(json.dumps({"verified_asset": result["filename"]}), flush=True)
        write_json(args.output / "report.json", {"status": "official_droid_inputs_staged_and_verified",
            "protocol_sha256": sha256(args.output / "protocol.json"), "assets": completed,
            "bytes": total_bytes, "recordings": 16, "checkpoints": 1,
            "seconds": time.monotonic() - started, "model_runs": 0,
            "training_corpus_downloaded": False, "training_histories_available": False,
            "recording_lineage_audit_complete": False, "confirmation_authorized": False})
        write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    except Exception as exc:
        message = str(exc)
        for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            if os.environ.get(name):
                message = message.replace(os.environ[name], "[redacted]")
        write_json(args.output / "FAILED.json", {"error": message, "completed": completed})
        raise RuntimeError("DROID staging failed; inspect the redacted failure receipt") from None


if __name__ == "__main__":
    main()
