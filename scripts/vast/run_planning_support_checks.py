"""Run only fit-only planner checks as the existing offline shards release GPUs."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from offline_study.protocol import sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    for name in ("source", "combined-root", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assignments = [("reach", 1), ("reach-wall", 2)]
    waiting, running, completed = assignments.copy(), [], []
    started = time.monotonic()
    env = {**os.environ, "PYTHONPATH": str(args.source), "JEPA_VERIFIED_LOCAL_DINO": "1",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "LD_LIBRARY_PATH": "/opt/conda/lib"}
    def interrupted(signum, frame):
        raise RuntimeError("Fit-check queue interrupted")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        while waiting or running:
            if time.monotonic() - started > 2700:
                raise TimeoutError("Fit-check queue safety cap")
            for task, gpu in waiting.copy():
                directory = args.combined_root / "evaluation-v1/bfloat16" / f"shard-{gpu:03d}"
                if not (directory / "DONE.json").exists():
                    if (args.combined_root / "execution-v1/FAILED.json").exists():
                        raise ValueError("Source coordinator failed; do not assume its GPUs are free")
                    continue
                done = json.loads((directory / "DONE.json").read_text())
                if sha256(directory / "report.json") != done["report_sha256"]:
                    raise ValueError("Released GPU's completion report changed")
                memory = int(subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used",
                                                      "--format=csv,noheader,nounits"], text=True).strip())
                if memory > 512:
                    continue  # DONE is written just before process teardown; do not overlap.
                command = [sys.executable, "-m", "offline_study.planning_support_check",
                    "--vendor", "/workspace/jepa_steering/vendor/jepa-wms",
                    "--checkpoint", "/workspace/jepa-runtime/checkpoints/jepa_wm_metaworld.pth.tar",
                    "--fixture", f"/workspace/jepa-runtime/{task}-worker-fit-fixture-v1",
                    "--fit-root", "/workspace/jepa-runtime/author-correction-20260907/fits-v1",
                    "--output", str(args.output / task)]
                log = (args.output / (task + ".log")).open("x")
                process = subprocess.Popen(command, env={**env, "CUDA_VISIBLE_DEVICES": str(gpu)},
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                running.append((process, log, task, gpu, time.monotonic()))
                waiting.remove((task, gpu))
                print(json.dumps({"event": "fit_check_started", "task": task, "gpu": gpu, "pid": process.pid}), flush=True)
            for process, log, task, gpu, task_start in running.copy():
                if time.monotonic() - task_start > 1200:
                    raise TimeoutError("Bounded fit-only check exceeded 20 minutes: " + task)
                code = process.poll()
                if code is None:
                    continue
                log.close()
                running.remove((process, log, task, gpu, task_start))
                if code:
                    raise RuntimeError(f"Fit-only transfer check failed: {task}, exit={code}")
                done = json.loads((args.output / task / "DONE.json").read_text())
                if sha256(args.output / task / "report.json") != done["report_sha256"]:
                    raise ValueError("Fit-check output checksum mismatch")
                completed.append({"task": task, "gpu": gpu, **done})
                print(json.dumps({"event": "fit_check_completed", "task": task}), flush=True)
            write_json(args.output / "progress.json", {"waiting": waiting,
                "running": [{"task": r[2], "gpu": r[3], "pid": r[0].pid} for r in running], "completed": completed})
            time.sleep(2)
        write_json(args.output / "DONE.json", {"status": "both_primary_rank_transfers_fit_checked",
            "completed": completed, "scientific_efficacy_measurement": False, "fresh_confirmation": False})
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "completed": completed})
        raise
    finally:
        for process, log, task, gpu, task_start in running:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            log.close()


if __name__ == "__main__":
    main()
