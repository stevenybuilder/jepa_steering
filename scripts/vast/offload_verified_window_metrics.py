"""Disk-full fallback for receipt-bound bulk metrics; preserve two US copies.

Only previously audited window_metrics.json files above20MiB are eligible.
Reports, protocols, statistical summaries and fitted banks stay local. Default
is a read-only inventory; removal requires the explicit execution flag.
"""
import argparse
import json
import subprocess
from pathlib import Path, PurePosixPath

from offload_verified_fit_captures import (
    LOCAL, BUNDLE, ORIGINAL, KEY, SOURCE_SSH, BACKUP_SSH, digest,
)

BACKUP = "/workspace/jepa-runtime/durable-offline-window-metrics-20260907-v1"


def source_path(relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or path.name != "window_metrics.json":
        raise ValueError("Unsafe or non-metric target")
    if path.parts[0] in ("metaworld-author-extension-20260907", "pusht-author-replication-20260907"):
        return "/workspace/jepa-runtime/" + str(path)
    if path.parts[0] in ("evaluation-v1", "diagnostic-completion-v1", "broad-baseline-v1"):
        return ORIGINAL + "/" + str(path)
    raise ValueError("Unrecognized original location")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remove-verified-local-duplicates", action="store_true")
    args = parser.parse_args()
    proof = LOCAL / "primary-durable-verification-20260907-v1"
    expected_report = json.loads((proof / "DONE.json").read_text())["report_sha256"]
    if digest(proof / "report.json") != expected_report:
        raise ValueError("Historical durability proof changed")
    hashes = json.loads((proof / "report.json").read_text())["artifact_sha256"]
    targets = []
    for relative, expected in hashes.items():
        local = BUNDLE / relative
        if local.name == "window_metrics.json" and local.exists() and local.stat().st_size > 20 * 1024**2:
            if local.is_symlink():
                raise ValueError("Symlink is not an eligible duplicate")
            targets.append((local.stat().st_size, relative, expected, source_path(relative)))
    targets.sort(reverse=True)
    print(json.dumps({"files": len(targets), "bytes": sum(t[0] for t in targets),
        "targets": [t[1] for t in targets], "removal_requested": args.remove_verified_local_duplicates}), flush=True)
    if not args.remove_verified_local_duplicates:
        return
    instances = json.loads(subprocess.check_output([
        "/Users/stevenyang/.local/share/uv/tools/vastai/bin/vastai", "show", "instances", "--raw"], text=True))
    by_id = {row["id"]: row for row in instances}
    for instance in (50125440, 50189244):
        if not by_id[instance]["geolocation"].endswith(", US") or by_id[instance]["actual_status"] != "running":
            raise ValueError("Authorized US replicas must be accessible")
    receipts = LOCAL / "window-metrics-storage-relocation-20260907-v1"
    receipts.mkdir(exist_ok=True)
    for size, relative, expected, original in targets:
        local = BUNDLE / relative
        if digest(local) != expected:
            raise ValueError("Local metrics differ from the audited copy")
        if subprocess.check_output(SOURCE_SSH + [f"sha256sum {original}"], text=True).split()[0] != expected:
            raise ValueError("Original remote metrics changed; preserve local")
        destination = BACKUP + "/" + relative
        subprocess.run(BACKUP_SSH + [f"mkdir -p {PurePosixPath(destination).parent}"], check=True)
        subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=15",
            "-i", KEY, "-P", "40083", str(local), "root@98.86.102.84:" + destination], check=True, timeout=600)
        if subprocess.check_output(BACKUP_SSH + [f"sha256sum {destination}"], text=True).split()[0] != expected:
            raise ValueError("Backup mismatch; preserve local")
        receipt = {"local_path": str(local), "bytes": size, "sha256": expected,
            "original_instance": 50125440, "original_path": original,
            "backup_instance": 50189244, "backup_path": destination,
            "historical_durability_report_sha256": expected_report,
            "two_us_copies_verified": True, "local_duplicate_removal_requested": True,
            "reason": "user_authorized_disk_full_storage_fallback", "scientific_data_discarded": False}
        receipt_path = receipts / (relative.replace("/", "__") + ".relocation.json")
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-i", KEY, "-P", "40083", str(receipt_path),
            "root@98.86.102.84:" + destination + ".relocation.json"], check=True)
        if local.is_symlink() or digest(local) != expected:
            raise ValueError("Local target changed after verification")
        local.unlink()
        print(json.dumps({"relocated": relative, "bytes": size, "sha256": expected}), flush=True)


if __name__ == "__main__":
    main()
