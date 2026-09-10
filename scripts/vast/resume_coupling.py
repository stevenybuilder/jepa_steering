"""Run the frozen ten-arm extension on eight exclusively assigned GPUs."""
import json
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/workspace/jepa-runtime")
OUTPUT = ROOT / "resume-20260907/coupling"
TASKS = [("reach", "reach-v2", ["0"]), ("reach-wall", "reach-wall-v2", ["1"]),
         ("pusht", "pusht-v1", ["2", "3", "4", "5", "6", "7"])]


def command(name, previous, devices, output, cap=100000):
    config = json.loads((ROOT / "vision-action-development" / previous / "launch_config.json").read_text())
    cmd = [sys.executable, "-m", "offline_study.intervention_multigpu"]
    for key in ("vendor", "checkpoint", "checkpoint_sha256", "manifest", "exposure_registry", "data_root"):
        cmd.extend(["--" + key.replace("_", "-"), config[key]])
    fit = OUTPUT / "fits" / name
    for flag, file in (("protocol", "protocol.json"), ("operator-bank", "operator_bank.pt"),
                       ("fit-receipt", "fit_receipt.json")):
        cmd.extend(["--" + flag, str(fit / file)])
    cmd.extend(["--output", str(output), "--tasks", *config["tasks"], "--devices", *devices,
                "--batch-size", "12", "--warmup", "1", "--prefetch-batches", "2",
                "--max-trajectories", str(cap), "--max-runtime-seconds", "1050",
                "--supervisor-grace-seconds", "180"])
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metaworld-recovery", action="store_true")
    args = parser.parse_args()
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
           "LD_LIBRARY_PATH": "/opt/conda/lib:" + os.environ.get("LD_LIBRARY_PATH", "")}
    subprocess.run([sys.executable, "-c", "import cv2, torchcodec; print('Video runtime verified')"],
                   env=env, check=True)
    # The earlier eight-arm suite established 1/2/8-GPU scaling. Check the new
    # effective batch (12 windows x 10 arms) before allocating the full suite.
    probe = OUTPUT / "runtime-check-v2"
    if not args.metaworld_recovery:
        subprocess.run(command("pusht", "pusht-v1", ["0"], probe, 8), env=env, check=True)
    check = json.loads((probe / "report.json").read_text())
    print(json.dumps({"runtime_check_complete": True, "windows": check["windows"],
                      "wall_seconds": check["wall_seconds"],
                      "per_shard": check["per_shard"]}), flush=True)
    processes = []
    handles = []
    try:
        tasks = TASKS[:2] if args.metaworld_recovery else TASKS
        run_dir = "development-recovery" if args.metaworld_recovery else "development"
        for name, previous, devices in tasks:
            suffix = "-recovery" if args.metaworld_recovery else ""
            log = (OUTPUT / (name + suffix + ".log")).open("x")
            handles.append(log)
            proc = subprocess.Popen(command(name, previous, devices, OUTPUT / run_dir / name),
                                    env=env, stdout=log, stderr=subprocess.STDOUT)
            processes.append((name, proc))
            print(json.dumps({"task": name, "pid": proc.pid, "devices": devices}), flush=True)
        while any(proc.poll() is None for _, proc in processes):
            time.sleep(5)
        codes = {name: proc.returncode for name, proc in processes}
        print(json.dumps({"task_exit_codes": codes}), flush=True)
        if any(codes.values()):
            raise RuntimeError("At least one task failed; preserve all task outputs")
        if args.metaworld_recovery:
            return
        subprocess.run([sys.executable, "-m", "offline_study.development_analysis", "--runs",
                        *(str(OUTPUT / "development" / name) for name, _, _ in TASKS),
                        "--output", str(OUTPUT / "analysis")], env=env, check=True)
    finally:
        for name, proc in processes:
            if proc.poll() is None:
                proc.terminate()
        for _, proc in processes:
            if proc.poll() is None:
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
        for log in handles:
            log.close()


if __name__ == "__main__":
    main()
