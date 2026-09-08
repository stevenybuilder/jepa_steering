"""Spool only the pinned, completed Indiana geometry panel; never write remotely.

Requires the exact board reservation, owned running Indiana and account <=$7/h.
No Drive/provider mutation, scientific imports, GPU calls, producer signals,
extraction or removal. Failed attempts and partial archives remain in place.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import time

from backup_results_to_google import digest, verify_archive
from navigation_geometry_archive_selection import select_files
from snapshot_live_results import HASH_FILES


PROJECT = Path(__file__).resolve().parents[2]
OUTPUT = PROJECT / "artifacts/offline_study/navigation-geometry-closure-preservation-20260908-v1"
BOARD = Path("/Users/stevenyang/Documents/GPU_RESOURCE_BOARD.md")
INSTANCE = 50205763
RAW_BYTES = 959303006
RESERVE = 4 << 30
LOW_PRIORITY = ["env", "CUDA_VISIBLE_DEVICES=", "PYTHONDONTWRITEBYTECODE=1",
                "ionice", "-c", "3", "nice", "-n", "19"]


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(name, value):
    with (OUTPUT / name).open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def connection():
    board = BOARD.read_text()
    if (OUTPUT.name not in board or "rep_geometry_transcoder/root" not in board or
            "2026-09-08T15:20Z complete navigation geometry preservation" not in board):
        raise ValueError("Exact geometry preservation board authorization required")
    rows = json.loads(subprocess.check_output(["vastai", "show", "instances", "--raw"],
                                             text=True, timeout=45))
    rates = [float(row["instance"]["totalHour"]) for row in rows]
    if any(not math.isfinite(rate) or rate < 0 for rate in rates) or sum(rates) > 7:
        raise ValueError("Authorized account-wide $7/h rate cap exceeded or unverifiable")
    matching = [row for row in rows if row["id"] == INSTANCE]
    if len(matching) != 1:
        raise ValueError("Exact owned Indiana instance missing")
    worker = matching[0]
    expected = {"label": "jepa-navigation-offline-indiana", "geolocation": "Indiana, US",
                "actual_status": "running", "intended_status": "running"}
    if any(worker.get(key) != value for key, value in expected.items()):
        raise ValueError("Indiana ownership, geography or running state changed")
    ssh = ["ssh", "-i", "/tmp/jepa_vast_50123620_ed25519", "-o", "BatchMode=yes",
           "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4",
           "-p", str(worker["ports"]["22/tcp"][0]["HostPort"]), "root@" + worker["public_ipaddr"]]
    return ssh, {"checked_at": utc(), "instance": INSTANCE, **expected,
                 "aggregate_usd_hour": sum(rates), "rate_basis": "sum of provider instance.totalHour, including stopped storage",
                 "usage_bandwidth_not_in_hourly_cap": True, "board_sha256": hashlib.sha256(board.encode()).hexdigest(),
                 "owner": "rep_geometry_transcoder/root", "provider_mutations": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    stage, archive, transfer = "intent", OUTPUT / "navigation-geometry-results.tar.gz", None
    try:
        utility_names = ("archive_navigation_geometry.py", "navigation_geometry_archive_selection.py",
                         "snapshot_live_results.py", "backup_results_to_google.py")
        utilities = {name: digest(Path(__file__).parent / name) for name in utility_names}
        write("INTENT.json", {"created_at": utc(), "instance": INSTANCE, "expected_files": 431,
              "expected_raw_bytes": RAW_BYTES, "output": str(OUTPUT), "reserve_bytes": RESERVE,
              "scope": "32 completed geometry shards; four aggregates/fits; two cohorts; original eval/analysis sources",
              "excluded": ["other categories", "datasets", "checkpoints", "caches", "logs", "credentials", "symlinks"],
              "operations_source_sha256": utilities, "remote_writes": 0, "gpu_calls": 0,
              "scientific_recomputation": False, "drive_operations": 0, "all_study_complete": False})
        stage = "authority"
        ssh, authority = connection()
        write("AUTHORITY.json", authority)
        if shutil.disk_usage(OUTPUT).free < RAW_BYTES + (64 << 20) + RESERVE:
            raise ValueError("Insufficient local space for raw-size upper bound plus 4 GiB reserve")

        def remote(code, payload=None):
            return json.loads(subprocess.check_output(ssh + [shlex.join(
                LOW_PRIORITY + ["/usr/bin/python3", "-c", code])], input=payload,
                text=True, timeout=180))

        selection_code = inspect.getsource(select_files) + (
            "\nimport json; print(json.dumps(select_files('/workspace/jepa-runtime')))\n")
        stage = "source_before"
        names = remote(selection_code)
        if len(names) != 431 or names != sorted(set(names)):
            raise ValueError("Exact 431-member production selection required")
        manifest = remote(HASH_FILES, json.dumps(names))
        if set(manifest) != set(names) or sum(item["bytes"] for item in manifest.values()) != RAW_BYTES:
            raise ValueError("Registered source inventory changed")
        write("FILES.json", manifest)
        print(json.dumps({"stage": stage, "files": len(names), "raw_bytes": RAW_BYTES,
                          "aggregate_usd_hour": authority["aggregate_usd_hour"]}), flush=True)
        # Refresh immediately before actual byte transfer, not only inventory.
        refreshed_ssh, authority = connection()
        if refreshed_ssh != ssh:
            raise ValueError("Provider connection changed during preflight")
        write("TRANSFER_AUTHORITY.json", authority)
        stage = "archive_stream"
        command = LOW_PRIORITY + ["tar", "-C", "/workspace/jepa-runtime", "--hard-dereference",
                                  "--no-recursion", "--null", "-czf", "-", "-T", "-"]
        h, size, last = hashlib.sha256(), 0, time.monotonic()
        with tempfile.TemporaryFile() as listing, archive.open("xb") as output:
            listing.write(b"\0".join(name.encode() for name in names) + b"\0")
            listing.seek(0)
            transfer = subprocess.Popen(ssh + [shlex.join(command)], stdin=listing, stdout=subprocess.PIPE)
            try:
                for chunk in iter(lambda: transfer.stdout.read(1 << 20), b""):
                    if shutil.disk_usage(OUTPUT).free < RESERVE + len(chunk):
                        raise ValueError("4 GiB reserve reached; partial archive retained")
                    if size + len(chunk) > RAW_BYTES + (64 << 20):
                        raise ValueError("Unexpected archive size; partial archive retained")
                    output.write(chunk)
                    h.update(chunk)
                    size += len(chunk)
                    if time.monotonic() - last > 20:
                        print(json.dumps({"stage": stage, "received_bytes": size}), flush=True)
                        last = time.monotonic()
            finally:
                # Close only our pipe. Never signal a producer or remote process.
                transfer.stdout.close()
            if transfer.wait(timeout=90) != 0:
                raise ValueError("Archive stream failed; original source and partial archive retained")
        write("ARCHIVE_STREAM.json", {"finished_at": utc(), "archive_bytes": size,
              "archive_sha256": h.hexdigest(), "command": command, "remote_writes": 0})
        stage = "full_local_readback"
        print(json.dumps({"stage": stage, "archive_bytes": size}), flush=True)
        verify_archive(["cat", str(archive)], h.hexdigest(), size, manifest)
        write("LOCAL_READBACK.json", {"verified_at": utc(), "compressed_sha256_verified": True,
              "compressed_size_verified": True, "every_member_sha256_and_size_verified": True, "files": len(names)})
        stage = "source_after"
        if remote(selection_code) != names:
            raise ValueError("Production selection changed after transfer")
        after = remote(HASH_FILES, json.dumps(names))
        write("FILES_AFTER.json", after)
        if after != manifest:
            raise ValueError("Selected source bytes changed during preservation")
        if any(digest(Path(__file__).parent / name) != value for name, value in utilities.items()):
            raise ValueError("Local operations source changed during preservation")
        if shutil.disk_usage(OUTPUT).free < RESERVE:
            raise ValueError("Local 4 GiB reserve violated")
        source_bindings = {name: item["sha256"] for name, item in manifest.items()
                           if name.endswith(("/protocol.json", "/fit_receipt.json", "/cohort.json"))
                           and "navigation-comparisons-" not in name}
        write("VERIFIED.json", {"status": "geometry_closure_local_archive_all_members_verified",
              "verified_at": utc(), "instance": INSTANCE, "archive": str(archive), "archive_sha256": h.hexdigest(),
              "archive_bytes": size, "files": len(names), "raw_bytes": RAW_BYTES,
              "manifest_sha256": digest(OUTPUT / "FILES.json"), "source_before_after_identical": True,
              "all_members_verified": True, "source_bindings": source_bindings,
              "operations_source_sha256": utilities, "minimum_free_reserve_bytes": RESERVE,
              "original_source_disks_retained": True, "remote_writes": 0, "gpu_calls": 0,
              "producer_signals": 0, "drive_operations": 0, "all_study_complete": False})
        print(json.dumps({"status": "verified", "archive": str(archive), "archive_sha256": h.hexdigest(),
                          "archive_bytes": size, "files": len(names)}), flush=True)
    except BaseException as error:
        write("FAILED.json", {"failed_at": utc(), "stage": stage, "error_type": type(error).__name__,
              "error": str(error), "archive_present": archive.exists(),
              "partial_archive_bytes": archive.stat().st_size if archive.exists() else 0,
              "local_transfer_pid": transfer.pid if transfer else None,
              "all_sources_preserved": True, "no_verified_completion_claim": True,
              "remote_writes": 0, "gpu_calls": 0, "producer_signals": 0})
        raise


if __name__ == "__main__":
    main()
