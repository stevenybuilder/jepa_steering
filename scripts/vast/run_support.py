"""One bounded fixed-protocol support stage on an explicitly assigned GPU set."""
import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--category", choices=["operator_rank", "distribution_layer", "distribution_spatial"])
    parser.add_argument("--phase", choices=["fit", "smoke", "development"], required=True)
    parser.add_argument("--devices", nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--native-fit", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    shared = []
    for key in ("vendor", "checkpoint", "checkpoint_sha256", "manifest", "exposure_registry", "data_root"):
        shared.extend(["--" + key.replace("_", "-"), config[key]])
    if args.phase == "fit":
        command = [sys.executable, "-m", "offline_study.support_fit", *shared,
                   "--task", args.task, "--output", str(args.fit),
                   "--device", "cuda:" + args.devices[0], "--batch-size", "8"]
        cap = 600
        if args.native_fit:
            command.extend(["--native-fit", str(args.native_fit)])
    else:
        if args.output is None or args.category is None:
            parser.error("Evaluation requires category and a fresh output")
        fit = args.fit / args.category
        cap = 360 if args.phase == "smoke" else 5400
        command = [sys.executable, "-m", "offline_study.intervention_multigpu", *shared,
                   "--protocol", str(fit / "protocol.json"), "--operator-bank", str(fit / "operator_bank.pt"),
                   "--fit-receipt", str(fit / "fit_receipt.json"), "--tasks", args.task,
                   "--output", str(args.output), "--devices", *args.devices,
                   "--max-trajectories", "4" if args.phase == "smoke" else "100000",
                   "--batch-size", str(args.batch_size), "--warmup", "1", "--prefetch-batches", "2",
                   "--max-runtime-seconds", str(cap - 120), "--supervisor-grace-seconds", "90"]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "LD_LIBRARY_PATH": "/opt/conda/lib:" + os.environ.get("LD_LIBRARY_PATH", "")}
    process = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        code = process.wait(timeout=cap)
        if code:
            raise RuntimeError(f"Stage failed with exit code {code}; original output retained")
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


if __name__ == "__main__":
    main()
