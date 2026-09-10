"""Move six exact local duplicate captures to two verified existing US disks.

User-authorized disk-full fallback. No new GPU rental; no raw experiment loss.
Only enumerated local native_fit.pt duplicates can be unlinked, AFTER verification
against their existing durable receipt and BOTH remote copies. Reports stay local.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


LOCAL = Path("/Users/stevenyang/Documents/rep_geometry_transcoder/artifacts/offline_study")
BUNDLE = LOCAL / "primary-durable-20260907"
ORIGINAL = "/workspace/jepa-runtime/author-correction-20260907"
BACKUP = "/workspace/jepa-runtime/durable-offline-native-fit-20260907-v1"
KEY = "/tmp/jepa_vast_50123620_ed25519"
SOURCE_SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15",
              "-i", KEY, "-p", "45353", "root@209.146.116.50"]
BACKUP_SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15",
              "-i", KEY, "-p", "40083", "root@98.86.102.84"]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remove-verified-local-duplicates", action="store_true")
    args = parser.parse_args()
    instances = json.loads(subprocess.check_output([
        "/Users/stevenyang/.local/share/uv/tools/vastai/bin/vastai", "show", "instances", "--raw"], text=True))
    by_id = {row["id"]: row for row in instances}
    for instance in (50125440, 50189244):
        if (not by_id[instance]["geolocation"].endswith(", US") or
                by_id[instance]["actual_status"] != "running"):
            raise ValueError("Existing authorized US replicas must be accessible")
    proof = LOCAL / "primary-durable-verification-20260907-v1"
    done = json.loads((proof / "DONE.json").read_text())
    if digest(proof / "report.json") != done["report_sha256"]:
        raise ValueError("Original local durability receipt changed")
    hashes = json.loads((proof / "report.json").read_text())["artifact_sha256"]
    receipt_dir = LOCAL / "fit-capture-storage-relocation-20260907-v1"
    receipt_dir.mkdir(exist_ok=True)
    for precision in ("bfloat16", "float32"):
        for task in ("reach", "reach-wall", "pusht"):
            relative = f"fits-v1/{precision}/{task}/native_fit.pt"
            path = BUNDLE / relative
            receipt_path = receipt_dir / f"{precision}-{task}.json"
            if not path.exists():
                if not receipt_path.exists():
                    raise ValueError("Missing local file without a relocation receipt")
                continue
            expected = hashes[relative]
            if digest(path) != expected:
                raise ValueError("Local capture differs from previously verified artifact")
            source_hash = subprocess.check_output(SOURCE_SSH + [f"sha256sum {ORIGINAL}/{relative}"], text=True).split()[0]
            if source_hash != expected:
                raise ValueError("Original remote copy changed; preserve local copy")
            destination = BACKUP + "/" + relative
            subprocess.run(BACKUP_SSH + [f"mkdir -p {BACKUP}/fits-v1/{precision}/{task}"], check=True)
            subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                "-o", "ServerAliveInterval=15", "-i", KEY, "-P", "40083", str(path),
                "root@98.86.102.84:" + destination], check=True, timeout=1800)
            backup_hash = subprocess.check_output(BACKUP_SSH + [f"sha256sum {destination}"], text=True).split()[0]
            if backup_hash != expected:
                raise ValueError("New backup verification failed; preserve local file")
            receipt = {"sha256": expected, "bytes": path.stat().st_size,
                "local_path": str(path), "primary_instance": 50125440, "primary_path": ORIGINAL + "/" + relative,
                "backup_instance": 50189244, "backup_path": destination,
                "both_remote_copies_verified": True, "local_duplicate_removal_requested": args.remove_verified_local_duplicates,
                "recoverable_from_two_us_instances": True, "reason": "user_authorized_local_disk_full_storage_fallback"}
            # Write the recovery map BEFORE removing the known duplicate.
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
            subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-i", KEY, "-P", "40083",
                            str(receipt_path), "root@98.86.102.84:" + destination + ".relocation.json"], check=True)
            if args.remove_verified_local_duplicates:
                if path.is_symlink() or digest(path) != expected:
                    raise ValueError("Local target changed after backup")
                path.unlink()
            print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
