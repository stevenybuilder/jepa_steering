"""Stream a result snapshot to Drive and GCS, verify it, then optionally offload bulk duplicates.

No local archive, new credentials, public sharing, recursive deletion, or GPU job.
The two remote objects must read back identically; Drive's archive members are
also individually checked against the pre-upload local manifest.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[2]
BUNDLE = REPO / "artifacts/offline_study/primary-durable-20260907"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


class HashReader:
    def __init__(self, stream):
        self.stream = stream
        self.hash = hashlib.sha256()
        self.size = 0

    def read(self, size=-1):
        data = self.stream.read(size)
        self.hash.update(data)
        self.size += len(data)
        return data


def verify_archive(command, expected_hash, expected_size, manifest=None):
    proc = subprocess.Popen(command, stdout=subprocess.PIPE)
    reader = HashReader(proc.stdout)
    seen = set()
    try:
        if manifest is not None:
            with tarfile.open(fileobj=reader, mode="r|gz") as archive:
                for member in archive:
                    if member.isdir():
                        continue
                    name = member.name.removeprefix("./")
                    if not member.isfile() or name not in manifest or name in seen:
                        raise ValueError("Unexpected archive member: " + name)
                    h = hashlib.sha256()
                    stream = archive.extractfile(member)
                    for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
                        h.update(chunk)
                    if (h.hexdigest() != manifest[name]["sha256"] or
                            member.size != manifest[name]["bytes"]):
                        raise ValueError("Archive member differs from local manifest: " + name)
                    seen.add(name)
            if seen != set(manifest):
                raise ValueError("Archive omitted source files")
        while reader.read(4 * 1024**2):
            pass
        if proc.wait() != 0 or reader.hash.hexdigest() != expected_hash or reader.size != expected_size:
            raise ValueError("Remote archive readback differs from upload stream")
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait()
        proc.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--remove-verified-local-bulk", action="store_true")
    args = parser.parse_args()
    tracked = set(subprocess.check_output(["git", "ls-files", "-z"], cwd=REPO).decode().split("\0"))
    paths = sorted(p for p in BUNDLE.rglob("*") if p.is_file())
    if any(p.is_symlink() or p.name in (".env", "rclone.conf") or p.suffix in (".pem", ".key") for p in paths):
        raise ValueError("Unexpected symlink or credential-like file in result bundle")
    targets = [p for p in paths if p.name == "window_metrics.json" and
               p.stat().st_size > 5 * 1024**2 and str(p.relative_to(REPO)) not in tracked]
    print(json.dumps({"status": "inventory", "files": len(paths),
        "source_bytes": sum(p.stat().st_size for p in paths),
        "eligible_local_duplicates": len(targets), "eligible_bytes": sum(p.stat().st_size for p in targets)}), flush=True)
    if not args.execute:
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    proof = REPO / "artifacts/offline_study" / ("google-storage-relocation-" + stamp)
    proof.mkdir(exist_ok=False)
    drive = "gdrive:Research-Archives/JEPA-WM/" + stamp
    cloud = "gs://rgt-jepa-archive-2026/rep_geometry_transcoder/" + stamp
    manifest = {str(p.relative_to(BUNDLE)): {"bytes": p.stat().st_size, "sha256": digest(p)} for p in paths}
    write_json(proof / "FILES.json", manifest)
    write_json(proof / "PLAN.json", {"source": str(BUNDLE), "drive": drive, "cloud": cloud,
        "targets": [str(p.relative_to(BUNDLE)) for p in targets],
        "remove_requested": args.remove_verified_local_bulk, "complete_study_claim": False})
    subprocess.run(["rclone", "mkdir", drive], check=True)
    archive_name = "primary-durable-results.tar.gz"
    tar = subprocess.Popen(["tar", "-czf", "-", "-C", str(BUNDLE), "."],
        stdout=subprocess.PIPE, env={**os.environ, "COPYFILE_DISABLE": "1"})
    uploads = [subprocess.Popen(command, stdin=subprocess.PIPE) for command in (
        ["rclone", "rcat", drive + "/" + archive_name, "--drive-chunk-size", "16M"],
        ["gcloud", "storage", "cp", "--if-generation-match=0", "-", cloud + "/" + archive_name])]
    h, size, last = hashlib.sha256(), 0, time.monotonic()
    try:
        while True:
            data = tar.stdout.read(1024**2)
            if not data:
                break
            h.update(data)
            size += len(data)
            for proc in uploads:
                proc.stdin.write(data)
            if time.monotonic() - last > 15:
                print(json.dumps({"status": "streaming_to_drive_and_gcs", "archive_bytes": size}), flush=True)
                last = time.monotonic()
        for proc in uploads:
            proc.stdin.close()
        codes = [tar.wait()] + [proc.wait() for proc in uploads]
        if any(codes):
            raise RuntimeError("Incomplete archive/upload: " + repr(codes))
    finally:
        for proc in [tar] + uploads:
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
    archive_hash = h.hexdigest()
    print(json.dumps({"status": "uploaded_verification_pending", "archive_bytes": size,
        "archive_sha256": archive_hash, "drive": drive, "cloud": cloud}), flush=True)
    verify_archive(["rclone", "cat", drive + "/" + archive_name], archive_hash, size, manifest)
    verify_archive(["gcloud", "storage", "cat", cloud + "/" + archive_name], archive_hash, size)
    receipt = {"status": "two_google_copies_readback_verified", "archive_name": archive_name,
        "archive_sha256": archive_hash, "archive_bytes": size, "source_files": len(manifest),
        "drive": drive, "cloud": cloud, "manifest_sha256": digest(proof / "FILES.json"),
        "archive_members_verified": True, "original_worker_files_untouched": True,
        "local_bulk_removal_requested": args.remove_verified_local_bulk}
    write_json(proof / "VERIFIED.json", receipt)
    # Preserve recovery maps remotely before any local unlink.
    subprocess.run(["rclone", "copy", str(proof), drive + "/receipt", "--immutable"], check=True)
    subprocess.run(["gcloud", "storage", "cp", "--if-generation-match=0",
        str(proof / "FILES.json"), str(proof / "PLAN.json"), str(proof / "VERIFIED.json"), cloud + "/receipt/"], check=True)
    removed = []
    if args.remove_verified_local_bulk:
        for path in targets:
            relative = str(path.relative_to(BUNDLE))
            if path.is_symlink() or digest(path) != manifest[relative]["sha256"]:
                raise ValueError("Local duplicate changed after snapshot: " + relative)
            # Recheck Git ownership immediately before each deletion.
            if subprocess.check_output(["git", "ls-files", "--", str(path.relative_to(REPO))], cwd=REPO):
                raise ValueError("Target is now tracked; preserve local copy")
            path.unlink()
            removed.append(relative)
    write_json(proof / "DONE.json", {"verified_sha256": digest(proof / "VERIFIED.json"),
        "removed_local_duplicates": removed,
        "freed_bytes": sum(manifest[n]["bytes"] for n in removed)})
    print(json.dumps({**receipt, "removed_files": len(removed),
        "freed_bytes": sum(manifest[n]["bytes"] for n in removed), "receipt": str(proof)}), flush=True)


if __name__ == "__main__":
    main()
