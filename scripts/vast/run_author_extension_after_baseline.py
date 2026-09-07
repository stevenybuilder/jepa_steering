"""Reuse a specific baseline GPU once its verified job has actually exited."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def process_is_running(pid, proc_root=Path("/proc")):
    """A zombie has exited and released CUDA even if its parent waits in order."""
    try:
        status = (proc_root / str(pid) / "status").read_text()
    except FileNotFoundError:
        return False
    state = next(line.split()[1] for line in status.splitlines() if line.startswith("State:"))
    return state not in ("Z", "X")


def main():
    from offline_study.protocol import sha256, write_json
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=["bfloat16", "float32"], required=True)
    parser.add_argument("--device", choices=["4", "5"], required=True)
    parser.add_argument("--baseline-pid", type=int, required=True)
    parser.add_argument("--receipt-suffix", default="")
    args = parser.parse_args()
    if (args.precision, args.device) not in (("bfloat16", "4"), ("float32", "5")):
        raise ValueError("Do not overlap the live GPU assignments")
    original = Path("/workspace/jepa-runtime/author-correction-20260907")
    root = Path("/workspace/jepa-runtime/metaworld-author-extension-20260907")
    baseline = original / "broad-baseline-v1" / args.precision
    tag = "extension-" + args.precision
    if args.receipt_suffix and not args.receipt_suffix.replace("-", "").isalnum():
        raise ValueError("Invalid receipt suffix")
    status = root / (tag + "-handoff" + args.receipt_suffix + ".json")
    if status.exists():
        raise FileExistsError(status)
    start = time.monotonic()
    write_json(status, {"status": "waiting_for_verified_baseline_exit", "pid": args.baseline_pid,
                        "gpu": args.device, "precision": args.precision})
    try:
        while True:
            if (baseline / "FAILED.json").exists():
                raise RuntimeError("Baseline failed; preserve its artifacts and inspect before handoff")
            if time.monotonic() - start > 10800:
                raise TimeoutError("Three-hour dependency cap")
            live = process_is_running(args.baseline_pid)
            if (baseline / "DONE.json").exists() and not live:
                done = json.loads((baseline / "DONE.json").read_text())
                for name in ("report", "selection", "window_metrics", "PARITY"):
                    if done[name + "_sha256"] != sha256(baseline / (name + ".json")):
                        raise ValueError("Baseline artifact integrity check failed")
                break
            if not live and not (baseline / "DONE.json").exists():
                raise RuntimeError("Baseline exited without a completion receipt")
            time.sleep(10)
        write_json(status, {"status": "running_seven_row_extension", "gpu": args.device})
        command = [sys.executable, str(root / "configs/run_author_correction.py"),
            "--root", str(root), "--source", str(original / "code-v7/src"), "--stage", "evaluate",
            "--tasks", "reach", "reach-wall", "--precisions", args.precision, "--devices", args.device,
            "--tag", tag]
        child = subprocess.Popen(command, start_new_session=True)
        try:
            if child.wait(timeout=7200):
                raise RuntimeError("Extension scheduler failed")
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        write_json(status, {"status": "extension_precision_shards_complete_merge_still_required", "gpu": args.device,
                            "precision": args.precision, "seconds": time.monotonic() - start})
    except Exception as exc:
        write_json(status, {"status": "failed", "error": str(exc), "gpu": args.device})
        raise


if __name__ == "__main__":
    main()
