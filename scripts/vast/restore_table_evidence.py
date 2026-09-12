"""Restore grounded, checksum-bound result/source archives; never execute them.

Uses existing Drive credentials through the established helper. No permission
changes or scientific work. Streams compressed archives without a local spool.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile

import urllib.request

FOLDER = "14zoPlI5qViiF5DwqVkCyKUu2O-Bj8u48"


def request(file_id, suffix):
    if os.environ.get("JEPA_DRIVE_ACCESS_TOKEN"):
        return urllib.request.urlopen(urllib.request.Request(
            "https://www.googleapis.com/drive/v3/files/" + file_id + suffix,
            headers={"Authorization": "Bearer " + os.environ["JEPA_DRIVE_ACCESS_TOKEN"],
                     "X-Goog-User-Project": "project-flash-490419"}), timeout=60)
    from final_preservation_drive import request as existing_request
    return existing_request(file_id, suffix)

PROJECT = Path(__file__).resolve().parents[2]
BASE = PROJECT / "artifacts/offline_study/final-storage-release-20260910-v1"
OUTPUT = Path(os.environ.get("JEPA_TABLE_RESTORE_ROOT", str(
    PROJECT / "artifacts/offline_study/table-completion-20260911-v1/restored")))
TARGETS = ((50231985, "batch-000"), (50231985, "batch-001"),
           (50259194, "batch-000"), (50245262, "batch-000"),
           (50239185, "batch-000"))


class HashedReader:
    def __init__(self, stream, limit):
        self.stream, self.limit = stream, limit
        self.count, self.hashed = 0, hashlib.sha256()

    def read(self, size):
        if not 0 <= size <= 4 << 20:
            raise ValueError("Require bounded reads")
        block = self.stream.read(size)
        self.count += len(block)
        if self.count > self.limit:
            raise ValueError("Archive exceeds frozen size")
        self.hashed.update(block)
        return block


def safe_path(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Unsafe archived path")
    return path


def restore(target):
    instance, batch = target
    proof = json.loads((BASE / str(instance) / batch / "DRIVE_VERIFIED.json").read_text())
    manifest = proof["manifest"]
    size = sum(v["bytes"] for v in manifest.values())
    root = OUTPUT / str(instance) / batch
    if (root / "RESTORED_VERIFIED.json").exists():
        # Revalidate restored bytes on reuse; a marker alone is insufficient.
        for name, spec in manifest.items():
            path = root / safe_path(name)
            if path.is_symlink() or path.stat().st_size != spec["bytes"]:
                raise ValueError("Existing restored member differs")
            h = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(4 << 20), b""):
                    h.update(block)
            if h.hexdigest() != spec["sha256"]:
                raise ValueError("Existing restored hash differs")
        return {"instance": instance, "batch": batch, "reverified": len(manifest)}
    if size > 400 << 20 or shutil.disk_usage(OUTPUT.parent).free < size + (2 << 30):
        raise ValueError("Incomplete prior root or insufficient bounded disk reserve")
    file_id = proof["drive_file_id"]
    # Ground MIME type, identity and parent before retrieving raw bytes.
    with request(file_id, "?fields=id,name,size,sha256Checksum,parents,trashed,mimeType") as stream:
        metadata = json.load(stream)
    if (metadata["id"] != file_id or metadata["name"] != proof["drive_name"] or
            int(metadata["size"]) != proof["bytes"] or metadata["parents"] != [FOLDER] or
            metadata.get("trashed") or metadata["sha256Checksum"] != proof["sha256"] or
            metadata["mimeType"].startswith("application/vnd.google-apps.")):
        raise ValueError("Drive object differs from preserved proof")
    root.mkdir(parents=True, exist_ok=True)
    seen = set()
    with request(file_id, "?alt=media") as response:
        reader = HashedReader(response, proof["bytes"])
        with tarfile.open(fileobj=reader, mode="r|gz") as archive:
            for member in archive:
                name = member.name
                safe_path(name)
                if not member.isfile():
                    raise ValueError("Unexpected non-file archive member")
                if name not in manifest:
                    # Original archive creation includes the manifest itself.
                    if name != "FINAL_PRESERVATION_FILES.json":
                        raise ValueError("Unexpected archive member: " + name)
                    stream = archive.extractfile(member)
                    if json.load(stream) != manifest:
                        raise ValueError("Embedded manifest differs")
                    continue
                if name in seen or member.size != manifest[name]["bytes"]:
                    raise ValueError("Duplicate or wrong-sized member")
                dest = root / safe_path(name)
                dest.parent.mkdir(parents=True, exist_ok=True)
                h = hashlib.sha256()
                # Preserve existing partial restoration, but require its bytes
                # to match both frozen manifest and newly received archive.
                existing = dest.exists()
                if existing:
                    if dest.is_symlink() or dest.stat().st_size != member.size:
                        raise ValueError("Partial restored file differs")
                    with dest.open("rb") as old:
                        for block in iter(lambda: old.read(4 << 20), b""):
                            h.update(block)
                    if h.hexdigest() != manifest[name]["sha256"]:
                        raise ValueError("Existing partial hash differs; preserve it")
                    h = hashlib.sha256()
                    with archive.extractfile(member) as source:
                        for block in iter(lambda: source.read(4 << 20), b""):
                            h.update(block)
                else:
                    with archive.extractfile(member) as source, dest.open("xb") as output:
                        for block in iter(lambda: source.read(4 << 20), b""):
                            h.update(block)
                            output.write(block)
                if h.hexdigest() != manifest[name]["sha256"]:
                    raise ValueError("Restored member hash differs")
                seen.add(name)
        while reader.read(4 << 20):
            pass
        if (reader.count != proof["bytes"] or reader.hashed.hexdigest() != proof["sha256"] or
                seen != set(manifest)):
            raise ValueError("Incomplete or altered archive readback")
    result = {"instance": instance, "batch": batch, "files": len(seen),
        "bytes": size, "archive_sha256": proof["sha256"], "drive_file_id": file_id,
        "metadata": metadata, "source_and_drive_unchanged": True}
    with (root / "RESTORED_VERIFIED.json").open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(restore, TARGETS):
            print(json.dumps({k: v for k, v in result.items() if k != "metadata"}), flush=True)
