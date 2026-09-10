"""Split untouched FP32 work onto the primary host; drain existing BF16 work safely.

No scientific settings, trajectories, arms or fits change. Existing workers finish;
only the two-GPU scheduler is paused and superseded, preventing duplicate FP32 jobs.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from offline_study.protocol import sha256, write_json

ROOT = Path("/workspace/jepa-runtime/pusht-author-replication-20260907")
ORIGINAL = Path("/workspace/jepa-runtime/author-correction-20260907")


def live(pid):
    try:
        lines = (Path("/proc") / str(pid) / "status").read_text().splitlines()
    except FileNotFoundError:
        return False
    return next(line.split()[1] for line in lines if line.startswith("State:")) not in ("Z", "X")


def run_worker(parent):
    command = (Path("/proc") / str(parent) / "cmdline").read_bytes().decode().split("\0")
    if "run_author_correction-with-analysis.py" not in " ".join(command) or str(ROOT) not in command:
        raise ValueError("Refusing to supersede an unrecognized scheduler")
    if (ROOT / "evaluation-v1/float32").exists():
        raise ValueError("FP32 was already started here; rebalance must be re-audited")
    os.kill(parent, signal.SIGSTOP)
    children = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdecimal():
            continue
        try:
            lines = (directory / "status").read_text().splitlines()
            ppid = int(next(line.split()[1] for line in lines if line.startswith("PPid:")))
            if ppid != parent:
                continue
            args = (directory / "cmdline").read_bytes().decode().split("\0")
            if not args[0]:  # Already exited child; its DONE is checked by reuse-completed.
                continue
            output = Path(args[args.index("--output") + 1])
            if ("offline_study.author_evaluate" not in args or
                    not output.is_relative_to(ROOT / "evaluation-v1/bfloat16")):
                raise ValueError("Unexpected owned worker; leave paused and inspect")
            children.append((int(directory.name), output))
        except FileNotFoundError:
            continue
    write_json(ROOT / "logs/rebalance-worker-v1.json", {"status": "old_scheduler_paused_workers_draining",
        "superseded_scheduler_pid": parent, "workers": [{"pid": p, "output": str(d)} for p, d in children],
        "fp32_assignment": "primary US host after its MetaWorld diagnostic closure", "scientific_settings_changed": False})
    started = time.monotonic()
    while any(live(pid) for pid, _ in children):
        if time.monotonic() - started > 7200:
            raise TimeoutError("Owned workers did not drain within two hours")
        time.sleep(5)
    for _, output in children:
        done = json.loads((output / "DONE.json").read_text())
        for name in ("report", "window_metrics", "selection", "protocol", "mechanism_diagnostics"):
            if done[name + "_sha256"] != sha256(output / (name + ".json")):
                raise ValueError("Drained worker artifacts failed verification")
    os.kill(parent, signal.SIGTERM)
    os.kill(parent, signal.SIGCONT)  # Deliver pending termination; no worker remains live.
    started = time.monotonic()
    while live(parent):
        if time.monotonic() - started > 30:
            raise TimeoutError("Old scheduler did not exit")
        time.sleep(.2)
    command = [sys.executable, str(ROOT / "configs/run_author_correction-rebalance.py"),
        "--root", str(ROOT), "--source", str(ORIGINAL / "code-v10/src"), "--stage", "evaluate",
        "--tasks", "pusht", "--precisions", "bfloat16", "--devices", "0", "1",
        "--tag", "pusht-bfloat16-tail-v2", "--reuse-completed", "--analyze-after"]
    subprocess.run(command, check=True, timeout=10800)
    write_json(ROOT / "logs/rebalance-worker-DONE.json", {"status": "bfloat16_complete_fp32_on_primary",
        "original_scheduler_superseded_without_repeating_completed_shards": True})


def run_primary(source):
    closure = ORIGINAL / "closure-v2"
    started = time.monotonic()
    while not (closure / "DONE.json").exists():
        if (closure / "FAILED.json").exists():
            raise RuntimeError("Original MetaWorld diagnostic closure failed")
        if time.monotonic() - started > 10800:
            raise TimeoutError("Primary GPU handoff dependency cap")
        time.sleep(10)
    done = json.loads((closure / "DONE.json").read_text())
    if done["report_sha256"] != sha256(closure / "report.json"):
        raise ValueError("Invalid primary GPU handoff receipt")
    if (ROOT / "evaluation-v1/float32").exists():
        raise FileExistsError("FP32 target is not pristine")
    command = [sys.executable, str(ROOT / "configs/run_author_correction-rebalance.py"),
        "--root", str(ROOT), "--source", str(source), "--stage", "evaluate",
        "--tasks", "pusht", "--precisions", "float32", "--devices", "0", "1", "2", "3", "6", "7",
        "--tag", "pusht-float32-primary-v2", "--analyze-after"]
    subprocess.run(command, check=True, timeout=10800)
    write_json(ROOT / "logs/rebalance-primary-DONE.json", {"status": "float32_complete_bfloat16_on_worker"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["worker", "primary"], required=True)
    parser.add_argument("--parent-pid", type=int)
    parser.add_argument("--source", type=Path, default=ORIGINAL / "code-v10/src")
    args = parser.parse_args()
    (ROOT / "logs").mkdir(exist_ok=True)
    try:
        if args.role == "worker":
            run_worker(args.parent_pid)
        else:
            run_primary(args.source)
    except Exception as exc:
        write_json(ROOT / "logs" / ("rebalance-" + args.role + "-FAILED.json"), {"error": str(exc)})
        raise


if __name__ == "__main__":
    main()
