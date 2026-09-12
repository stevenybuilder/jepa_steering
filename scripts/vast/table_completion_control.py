"""Bounded activation/backstop for the explicitly owned table-completion worker.

Never destroys storage, resumes confirmation, or touches another rental.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "artifacts/offline_study/table-completion-20260911-v1"
VAST = "/Users/stevenyang/.local/bin/vastai"
INSTANCE = 50546172
LABEL = "jepa-confirmation-0911-reach-0"
LAUNCH_LABEL = "com.steven.jepa.table-completion.budget.20260911"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)


def provider():
    return json.loads(subprocess.check_output(
        [VAST, "show", "instances", "--raw"], text=True, timeout=45))


def owned(rows):
    row = next(r for r in rows if r["id"] == INSTANCE)
    if row["label"] != LABEL or row["geolocation"] != "Washington, US":
        raise ValueError("Worker identity or geography differs")
    return row


def activate():
    rows = provider()
    row = owned(rows)
    running = sum(r["dph_total"] for r in rows if r["cur_state"] == "running")
    if row["cur_state"] != "stopped" or row["actual_status"] != "exited":
        raise ValueError("Expected explicitly reserved stopped worker")
    if running + row["dph_total"] > 7:
        raise ValueError("Aggregate running fleet would exceed $7/hour")
    board = Path("/Users/stevenyang/Documents/GPU_RESOURCE_BOARD.md").read_text()
    if "2026-09-11T06:22Z table-completion reservation" not in board:
        raise ValueError("Missing explicit owner reservation")
    write(ROOT / "LEASE.json", {"instance": INSTANCE, "label": LABEL,
        "owner": "rep_geometry_transcoder/root", "geolocation": row["geolocation"],
        "output_root": "/workspace/table-completion-20260911-v1",
        "purpose": "restore existing table evidence and finish missing development cells",
        "activated_utc": datetime.now(timezone.utc).isoformat(),
        "deadline_timestamp": time.time() + 6 * 3600,
        "hourly_quote_usd": row["dph_total"], "hourly_fleet_cap_usd": 7,
        "confirmation_restart": False, "destroy_storage": False})
    # Install the independent lifecycle backstop BEFORE starting paid compute.
    subprocess.run(["launchctl", "submit", "-l", LAUNCH_LABEL,
        "--", str(PROJECT / ".venv/bin/python"), str(Path(__file__).resolve()),
        "guard"], check=True, timeout=30)
    subprocess.run(["launchctl", "list", LAUNCH_LABEL], check=True,
                   stdout=subprocess.DEVNULL, timeout=30)
    result = subprocess.check_output(
        [VAST, "start", "instance", str(INSTANCE), "--raw"], text=True, timeout=90)
    write(ROOT / "START_RESPONSE.json", {"response": result})
    print(json.dumps({"instance": INSTANCE, "start_response": result,
        "hourly_quote_usd": row["dph_total"], "guard_registered": True}), flush=True)


def guard():
    lease = json.loads((ROOT / "LEASE.json").read_text())
    while time.time() < lease["deadline_timestamp"]:
        if (ROOT / "STOPPED_VERIFIED.json").exists():
            return
        time.sleep(min(30, max(.1, lease["deadline_timestamp"] - time.time())))
    # Retain source disks even if the application/archival supervisor failed.
    for attempt in range(10):
        try:
            row = owned(provider())
            if row["cur_state"] != "stopped":
                result = subprocess.check_output([VAST, "stop", "instance",
                    str(INSTANCE), "--raw"], text=True, timeout=90)
                write(ROOT / f"BACKSTOP_STOP_{attempt}.json", {"response": result,
                    "source_storage_retained": True, "scientific_completion": False})
            row = owned(provider())
            if row["cur_state"] == "stopped":
                write(ROOT / "BACKSTOP_STOPPED.json", {"instance": INSTANCE,
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "state": row["cur_state"], "actual_status": row["actual_status"],
                    "source_storage_retained": True})
                return
        except Exception as error:
            write(ROOT / f"BACKSTOP_ERROR_{attempt}.json", {"type": type(error).__name__})
        time.sleep(30)
    raise RuntimeError("Provider stop unverified; retained bounded error receipts")


def rent():
    """One replacement only; source data on the unavailable old disk stays put."""
    if (ROOT / "REPLACEMENT_LEASE.json").exists():
        raise ValueError("Replacement already recorded; refuse duplicate rental")
    board = Path("/Users/stevenyang/Documents/GPU_RESOURCE_BOARD.md").read_text()
    if "2026-09-11T06:37Z replacement reservation" not in board:
        raise ValueError("Missing replacement owner reservation")
    rows = provider()
    if owned(rows)["cur_state"] != "stopped":
        raise ValueError("Old queued restart not cancelled")
    query = "geolocation=US rentable=True rented=False verified=True num_gpus>=2 num_gpus<=4 gpu_name=RTX_4090 inet_down>=500 disk_space>=100"
    offers = json.loads(subprocess.check_output([VAST, "search", "offers", query,
        "--order", "dph_total", "--limit", "40", "--raw"], text=True, timeout=45))
    ready = []
    for row in offers:
        # Search reports a small default disk quote; explicitly price100GB.
        cost = float(row["dph_base"]) + 100 * float(row["storage_cost"]) / 30 / 24
        if (row["geolocation"].endswith(", US") and row["num_gpus"] in (2, 4) and
                int(row["driver_version"].split(".")[0]) >= 570 and cost <= 2.10):
            ready.append((cost / row["num_gpus"], -row["num_gpus"], row, cost))
    if not ready:
        raise ValueError("No eligible bounded US offer")
    # The first freshly listed offer raced with another renter. Keep its
    # reservation as evidence and exclude that physical machine on this retry.
    previous = ROOT / "REPLACEMENT_RESERVATION.json"
    if previous.exists():
        excluded = json.loads(previous.read_text())["machine_id"]
        ready = [item for item in ready if item[2]["machine_id"] != excluded]
    if not ready:
        raise ValueError("No alternative machine after the rental conflict")
    _, _, spec, cost = min(ready, key=lambda x: x[:2])
    current = sum(r["dph_total"] for r in rows if r["cur_state"] == "running")
    if current + cost > 7:
        raise ValueError("Fleet cap would be exceeded")
    started = time.time()
    label = "jepa-table-completion-0911-v1"
    reservation_path = ROOT / ("REPLACEMENT_RESERVATION-v2.json" if previous.exists() else "REPLACEMENT_RESERVATION.json")
    write(reservation_path, {"owner": "rep_geometry_transcoder/root",
        "offer_id": spec["id"], "machine_id": spec["machine_id"], "label": label,
        "geolocation": spec["geolocation"], "num_gpus": spec["num_gpus"],
        "disk_gb": 100, "hourly_quote_usd": cost,
        "initial_all_in_budget_usd": 4,
        "deadline_timestamp": started + min(6 * 3600, 4 / (cost + .12) * 3600),
        "inet_down": spec["inet_down"], "inet_up": spec["inet_up"],
        "inet_down_cost": spec["inet_down_cost"], "inet_up_cost": spec["inet_up_cost"],
        "output_root": "/workspace/table-completion-20260911-v1"})
    result = json.loads(subprocess.check_output([VAST, "create", "instance", str(spec["id"]),
        "--image", "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime", "--disk", "100",
        "--label", label, "--ssh", "--direct", "--cancel-unavail", "--raw"],
        text=True, timeout=180))
    write(ROOT / "REPLACEMENT_CREATE_RESPONSE.json", result)
    if not result.get("success") or not result.get("new_contract"):
        raise ValueError("Replacement creation did not succeed")
    lease = json.loads(reservation_path.read_text())
    lease["instance"] = result["new_contract"]
    write(ROOT / "REPLACEMENT_LEASE.json", lease)
    try:
        subprocess.run(["launchctl", "submit", "-l", LAUNCH_LABEL + ".replacement",
            "--", str(PROJECT / ".venv/bin/python"), str(Path(__file__).resolve()),
            "replacement-guard"], check=True, timeout=30)
        subprocess.run(["launchctl", "list", LAUNCH_LABEL + ".replacement"], check=True,
                       stdout=subprocess.DEVNULL, timeout=30)
    except Exception:
        subprocess.run([VAST, "stop", "instance", str(lease["instance"]), "--raw"],
                       check=True, timeout=90)
        raise
    print(json.dumps({"instance": lease["instance"], "gpus": lease["num_gpus"],
        "geolocation": lease["geolocation"], "hourly_quote_usd": cost,
        "backstop_registered": True}), flush=True)


def replacement_guard():
    lease = json.loads((ROOT / "REPLACEMENT_LEASE.json").read_text())
    amendment = ROOT / "WALL_RESUME_BUDGET.json"
    if amendment.exists():
        updated = json.loads(amendment.read_text())
        if (updated["instance"] != lease["instance"] or updated["hourly_fleet_cap_usd"] != 7 or
                updated["deadline_timestamp"] != 1789128000.0):
            raise ValueError("Unexpected Wall continuation budget")
        lease["deadline_timestamp"] = updated["deadline_timestamp"]
    while time.time() < lease["deadline_timestamp"]:
        if (ROOT / "REPLACEMENT_STOPPED_VERIFIED.json").exists():
            return
        time.sleep(min(30, max(.1, lease["deadline_timestamp"] - time.time())))
    for attempt in range(10):
        try:
            row = next(r for r in provider() if r["id"] == lease["instance"])
            if row["label"] != lease["label"] or row["geolocation"] != lease["geolocation"]:
                raise ValueError("Replacement identity changed")
            if row["cur_state"] != "stopped":
                result = subprocess.check_output([VAST, "stop", "instance",
                    str(lease["instance"]), "--raw"], text=True, timeout=90)
                write(ROOT / f"REPLACEMENT_BACKSTOP_{attempt}.json", {"response": result,
                    "source_storage_retained": True, "scientific_completion": False})
            row = next(r for r in provider() if r["id"] == lease["instance"])
            if row["cur_state"] == "stopped":
                write(ROOT / "REPLACEMENT_BACKSTOP_STOPPED.json", {"instance": lease["instance"],
                    "utc": datetime.now(timezone.utc).isoformat(), "state": row["cur_state"],
                    "actual_status": row["actual_status"], "source_storage_retained": True})
                return
        except Exception as error:
            write(ROOT / f"REPLACEMENT_BACKSTOP_ERROR_{attempt}.json", {"type": type(error).__name__})
        time.sleep(30)
    raise RuntimeError("Replacement stop unverified; retained error receipts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("activate", "guard", "rent", "replacement-guard"))
    args = parser.parse_args()
    {"activate": activate, "guard": guard, "rent": rent,
     "replacement-guard": replacement_guard}[args.command]()
