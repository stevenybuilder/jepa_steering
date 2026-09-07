"""Execute the registered geometry fits and development suite on the owned worker."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/workspace/jepa-runtime")
OUTPUT = ROOT / "resume-20260907/geometry-v2"
TASKS = [("reach", "reach-v2", "mw-reach", ["0"]),
         ("reach-wall", "reach-wall-v2", "mw-reach-wall", ["1"]),
         ("pusht", "pusht-v1", "pusht", ["2", "3", "4", "5", "6", "7"])]


def shared(previous):
    config = json.loads((ROOT / "vision-action-development" / previous / "launch_config.json").read_text())
    result = []
    for key in ("vendor", "checkpoint", "checkpoint_sha256", "manifest", "exposure_registry", "data_root"):
        result.extend(["--" + key.replace("_", "-"), config[key]])
    return result


def development_command(name, previous, task, devices, output, cap=100000):
    fit = OUTPUT / "fits" / name
    return [sys.executable, "-m", "offline_study.intervention_multigpu", *shared(previous),
            "--protocol", str(fit / "protocol.json"), "--operator-bank", str(fit / "operator_bank.pt"),
            "--fit-receipt", str(fit / "fit_receipt.json"), "--tasks", task,
            "--devices", *devices, "--output", str(output), "--max-trajectories", str(cap),
            "--batch-size", "12", "--warmup", "1", "--prefetch-batches", "2",
            "--max-runtime-seconds", "900", "--supervisor-grace-seconds", "180"]


def run_parallel(jobs, env, cap):
    processes, logs = [], []
    try:
        for name, command in jobs:
            log = (OUTPUT / (name + ".log")).open("x")
            logs.append(log)
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            processes.append((name, process))
            print(json.dumps({"job": name, "pid": process.pid, "command": command}), flush=True)
        deadline = time.monotonic() + cap
        while any(process.poll() is None for _, process in processes):
            if time.monotonic() > deadline:
                raise TimeoutError("Geometry stage reached registered runtime cap")
            time.sleep(5)
        codes = {name: process.returncode for name, process in processes}
        print(json.dumps({"exit_codes": codes}), flush=True)
        if any(codes.values()):
            raise RuntimeError("Geometry stage has failed jobs; preserve all original outputs")
    finally:
        for _, process in processes:
            if process.poll() is None:
                os.killpg(process.pid, 15)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, 9)
                    process.wait()
        for log in logs:
            log.close()


def main():
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "LD_LIBRARY_PATH": "/opt/conda/lib:" + os.environ.get("LD_LIBRARY_PATH", "")}
    subprocess.run([sys.executable, "-c", "import cv2, torchcodec; print('Video runtime verified')"], env=env, check=True)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    run_parallel([(name + "-fit", [sys.executable, "-m", "offline_study.geometry_fit", *shared(previous),
                                  "--task", task, "--device", "cuda:" + str(index), "--batch-size", "8",
                                  "--output", str(OUTPUT / "fits" / name)])
                  for index, (name, previous, task, _) in enumerate(TASKS)], env, 300)
    probe = development_command("pusht", "pusht-v1", "pusht", ["0"], OUTPUT / "runtime-check", 8)
    subprocess.run(probe, env=env, check=True, timeout=240)
    run_parallel([(name + "-development", development_command(name, previous, task, devices,
                                                              OUTPUT / "development" / name))
                  for name, previous, task, devices in TASKS], env, 1100)
    subprocess.run([sys.executable, "-m", "offline_study.development_analysis", "--runs",
                    *(str(OUTPUT / "development" / name) for name, _, _, _ in TASKS),
                    "--output", str(OUTPUT / "analysis")], env=env, check=True)


if __name__ == "__main__":
    main()
