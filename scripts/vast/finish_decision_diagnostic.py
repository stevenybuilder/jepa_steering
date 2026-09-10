"""Finish ONLY worker50531754: mirror, verify, Drive archive, then release.

Existing historical rentals are never operated on. Failures retain the source.
This script can outlive the interactive client; it does not launch experiments.
"""
import json
import math
from pathlib import Path
import subprocess
import time

from decision_preserve import verify as verify_members
from final_preservation_drive import upload_direct, verify as verify_drive, FOLDER

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "artifacts/offline_study/decision-closeout-20260910-v1"
RUN_NAME = "decision-diagnostic-20260910-v2-retry1"
REMOTE = "/workspace/decision/artifacts/offline_study/" + RUN_NAME
LOCAL = PROJECT / "artifacts/offline_study" / RUN_NAME
INSTANCE = 50531754
LABEL = "jepa-decision-bounded-20260910"
VAST = "/Users/stevenyang/.local/bin/vastai"
SSH = ["ssh", "-i", "/tmp/jepa_vast_50123620_ed25519", "-o", "BatchMode=yes",
       "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
       "-p", "15561", "root@97.70.195.221"]


def write(name, value):
    path = ROOT / name
    with path.open("x") as output:
        json.dump(value, output, indent=2, sort_keys=True)


def command(argv, timeout=180):
    return subprocess.check_output(argv, text=True, timeout=timeout)


def authority():
    row = json.loads(command([VAST, "show", "instance", str(INSTANCE), "--raw"]))
    if (row["id"] != INSTANCE or row["label"] != LABEL or row["geolocation"] != "Michigan, US" or
            row["public_ipaddr"] != "97.70.195.221"):
        raise ValueError("New-worker identity no longer matches the exclusive lease")
    return {k: row.get(k) for k in ("id", "label", "geolocation", "actual_status", "cur_state", "public_ipaddr")}


def mirror():
    subprocess.run(["rsync", "-a", "--timeout=90", "-e",
        "ssh -i /tmp/jepa_vast_50123620_ed25519 -o BatchMode=yes -p 15561",
        "root@97.70.195.221:" + REMOTE, str(LOCAL.parent)], check=True, timeout=180)


def close_numeric(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and set(actual) == set(expected) and all(close_numeric(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(close_numeric(a, b) for a, b in zip(actual, expected))
    if isinstance(expected, float):
        return isinstance(actual, (float, int)) and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12)
    return actual == expected


def render_report(analysis):
    path = PROJECT / "reports/PLANNER_DECISION_DIAGNOSTIC_RESULTS.md"
    if len(analysis["contrasts"]) != 8 or len(analysis["rankings"]) != 10:
        raise ValueError("Cannot render an incomplete primary family")
    positive = sum(row["simultaneous_95_interval_m"][0] > 0 for row in analysis["contrasts"])
    negative = sum(row["simultaneous_95_interval_m"][1] < 0 for row in analysis["contrasts"])
    lines = ["# Completed common-action planner-decision diagnostic", "",
        "This is a **development diagnostic**, not a new full-task success trial.",
        "Reach and Reach-Wall each contribute96 canonical scenarios and five conditions.",
        "Each model scores the same initial300-candidate H6 population; its selected",
        "first15 elementary actions are executed without replanning. Identical selected",
        "candidates may share the same verified physical outcome within a scenario.", "",
        f"Of eight simultaneous primary comparisons, {positive} have intervals entirely",
        f"above zero (better short-prefix progress) and {negative} entirely below zero.",
        "These counts do not establish improved full-episode task success.", "",
        "## All frozen physical-progress contrasts", "",
        "Positive values mean the treatment ends closer to the expert end-effector",
        "goal than its reference. Units below are millimeters; the bootstrap unit",
        "is a whole scenario, not a candidate action or repeated condition.", "",
        "| Task | Treatment | Reference | Mean distance gain, mm | Simultaneous95% interval, mm | Changed choices /96 |",
        "|---|---|---|---:|---:|---:|"]
    for row in analysis["contrasts"]:
        lo, hi = row["simultaneous_95_interval_m"]
        lines.append(f"| {row['task']} | {row['treatment']} | {row['reference']} | "
            f"{1000*row['mean_distance_gain_m']:+.4f} | [{1000*lo:+.4f}, {1000*hi:+.4f}] | {row['changed_choices']} |")
    lines += ["", "## Candidate-ranking diagnostics", "",
        "These summaries compare each condition with native. Rank agreement is",
        "descriptive and does not identify the best physical action among all300.", "",
        "| Task | Condition | Changed selections /96 | Mean Spearman | Mean top10 overlap |",
        "|---|---|---:|---:|---:|"]
    for row in analysis["rankings"]:
        rho = "undefined" if row["mean_spearman"] is None else f"{row['mean_spearman']:.6f}"
        lines.append(f"| {row['task']} | {row['arm']} | {row['changed_from_native']} | {rho} | {row['mean_top10_overlap']:.4f} |")
    lines += ["", "## Interpretation boundaries", "",
        "- This tests the first CEM population, not its final optimized elite mean.",
        "- H6 ranking and15-step physical progress have different horizons.",
        "- The cohort was already used for development; it is not fresh confirmation.",
        "- One released checkpoint is not three training seeds or a checkpoint history.",
        "- This cannot by itself explain the separate full-rollout successes/failures.",
        "- The [completed960-episode panel](CORE_METAWORLD_BEHAVIORAL_RESULTS.md)",
        "  did not establish learned-intervention task-success improvements.", "",
        "All raw inputs, scores, chosen actions, physical trajectories, failed setup",
        "attempts and runtime provenance are included in the separate preservation",
        "archive. See `artifacts/offline_study/decision-closeout-20260910-v1/` for",
        "independent CPU verification, Drive readback, and rental-release receipts.", "",
        f"Protocol SHA256: `{analysis['protocol_sha256']}`.", ""]
    with path.open("x") as output:
        output.write("\n".join(lines))
    return path


def main():
    intent = {"instance": INSTANCE, "scope": LABEL, "drive_folder": FOLDER,
        "destroy_only_after_full_drive_bytes_verified": True, "other_instances_authorized": False}
    if ROOT.exists():
        if json.loads((ROOT / "INTENT.json").read_text()) != intent:
            raise ValueError("Cannot resume a different closeout intent")
        if (ROOT / "DONE.json").exists():
            return
        if (ROOT / "FAILED.json").exists():
            raise ValueError("Prior substantive closeout failure requires review")
    else:
        ROOT.mkdir(parents=True, exist_ok=False)
        write("INTENT.json", intent)
    lease = authority()
    if not (ROOT / "LEASE_CHECK.json").exists():
        write("LEASE_CHECK.json", lease)
    # Bound monitoring. The GPU job also has its own hard timeout. If any
    # operation fails, preserve source and emit FAILED rather than destroy it.
    deadline = time.monotonic() + 3 * 3600
    while time.monotonic() < deadline:
        state = command(SSH + [f"if test -f {REMOTE}/FAILED.json; then echo FAILED; "
                f"elif test -f {REMOTE}/ANALYSIS_DONE.json; then echo COMPLETE; else echo RUNNING; fi"]).strip().splitlines()[-1]
        mirror()
        count = len(list(LOCAL.glob("*/episode-*/DONE.json")))
        print(json.dumps({"state": state, "mirrored_scenarios": count, "expected": 192}), flush=True)
        if state == "FAILED":
            raise RuntimeError("Diagnostic failed; mirrored data and worker retained for diagnosis")
        if state == "COMPLETE":
            break
        time.sleep(45)
    else:
        raise TimeoutError("Monitoring deadline; preserve source and review the bounded worker")
    original = json.loads((LOCAL / "analysis.json").read_text())
    write("WORKER_ANALYSIS.json", original)
    subprocess.run([str(PROJECT / ".venv/bin/python"), "-m", "offline_study.decision_diagnostic", str(LOCAL)],
                   cwd=PROJECT, stdout=subprocess.DEVNULL, check=True, timeout=180)
    independent = json.loads((LOCAL / "analysis.json").read_text())
    if not close_numeric(independent, original):
        raise ValueError("Independent CPU analysis does not match the worker")
    write("SCIENCE_VERIFIED.json", {"scenarios": 192, "condition_prefixes": 960,
        "all_raw_done_receipts_checked": True, "independent_cpu_recomputed": True,
        "absolute_float_comparison_tolerance": 1e-12, "statistical_thresholds_unchanged": True})
    report = render_report(original)
    command(SSH + ["mkdir -p /workspace/decision/reports /workspace/decision/closeout_pre_release"])
    scp = ["scp", "-q", "-i", "/tmp/jepa_vast_50123620_ed25519", "-o", "BatchMode=yes", "-P", "15561"]
    subprocess.run(scp + [str(report), "root@97.70.195.221:/workspace/decision/reports/"], check=True, timeout=90)
    subprocess.run(scp + [str(ROOT / "SCIENCE_VERIFIED.json"), str(ROOT / "WORKER_ANALYSIS.json"),
        "root@97.70.195.221:/workspace/decision/closeout_pre_release/"], check=True, timeout=90)
    name = "JEPA-decision-diagnostic-50531754-20260910.tar.gz"
    remote_archive = "/workspace/" + name
    packed = command(SSH + ["python3 /workspace/decision/scripts/vast/decision_preserve.py pack " + remote_archive], timeout=300)
    info = json.loads(packed.strip().splitlines()[-1])
    write("REMOTE_ARCHIVE.json", info)
    archive = ROOT / name
    subprocess.run(["scp", "-q", "-i", "/tmp/jepa_vast_50123620_ed25519", "-o", "BatchMode=yes",
        "-P", "15561", "root@97.70.195.221:" + remote_archive, str(archive)], check=True, timeout=600)
    proof = verify_members(archive)
    if proof["sha256"] != info["sha256"] or proof["bytes"] != info["bytes"]:
        raise ValueError("Source-to-local archive checksum differs")
    write("LOCAL_MEMBERS_VERIFIED.json", proof)
    subprocess.run(["rclone", "about", "gdrive:", "--json"], stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, check=True, timeout=60)
    uploaded = upload_direct(archive, name, ROOT / "DRIVE_UPLOAD.json")
    checked = verify_drive(uploaded["id"], name, proof["bytes"], proof["sha256"], stream=True)
    write("DRIVE_VERIFIED.json", {**proof, **checked, "file_id": uploaded["id"], "folder_id": FOLDER})
    # Never broaden deletion by looping over all rentals. Exactly this newly
    # created disposable instance, whose complete archive has just been read back.
    write("FINAL_LEASE_CHECK.json", authority())
    destroyed = json.loads(command([VAST, "destroy", "instance", str(INSTANCE), "--raw"]))
    write("DESTROY_RESPONSE.json", destroyed)
    remaining = json.loads(command([VAST, "show", "instances", "--raw"]))
    if any(row["id"] == INSTANCE for row in remaining):
        raise RuntimeError("Provider has not confirmed this worker absent")
    write("DONE.json", {"status": "diagnostic_verified_archived_and_new_worker_released",
        "instance": INSTANCE, "drive_file_id": uploaded["id"], "archive_sha256": proof["sha256"],
        "remaining_instance_ids": [row["id"] for row in remaining]})
    print(json.dumps({"status": "done", "instance_released": INSTANCE, "drive_file_id": uploaded["id"]}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if ROOT.exists() and not (ROOT / "FAILED.json").exists():
            write("FAILED.json", {"error_type": type(error).__name__, "error": str(error), "source_destroyed_by_failure_handler": False})
        raise
