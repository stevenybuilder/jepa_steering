"""Execute the fixed eight-shard planned combination on the existing US worker."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from offline_study.combined_run import load_contract
from offline_study.protocol import sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--fit-check", required=True, type=Path)
    args = parser.parse_args()
    protocol = args.root / "freeze-v1/protocol.json"
    contract = load_contract(protocol)
    gate = args.fit_check
    done = json.loads((gate / "DONE.json").read_text())
    if sha256(gate / "report.json") != done["report_sha256"]:
        raise ValueError("Fit-only gate changed")
    receipt = json.loads((gate / "report.json").read_text())
    if receipt["protocol_sha256"] != sha256(protocol) or receipt["development_or_protected_outcomes_accessed"]:
        raise ValueError("Wrong fit-only gate")
    utilization = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
    devices = [line.split(",") for line in utilization.strip().splitlines()]
    if len(devices) != 8 or any(int(memory) > 512 for _, memory in devices):
        raise ValueError("Expected eight currently available GPUs; do not collide with other work")
    logs = args.root / "execution-v1"
    logs.mkdir(exist_ok=False)
    env = {**os.environ, "PYTHONPATH": str(args.source), "JEPA_VERIFIED_LOCAL_DINO": "1",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "LD_LIBRARY_PATH": "/opt/conda/lib"}
    running, completed = [], []
    def interrupted(signum, frame):
        raise RuntimeError("Coordinator interrupted; terminate only owned child groups")
    signal.signal(signal.SIGTERM, interrupted)
    started = time.monotonic()
    try:
        for index, (precision, shard) in enumerate((p, s) for p in contract["precisions"] for s in range(4)):
            command = [sys.executable, "-m", "offline_study.combined_run", "evaluate",
                "--protocol", str(protocol), "--fit-check", str(gate), "--precision", precision,
                "--shard-index", str(shard), "--vendor", "/workspace/jepa_steering/vendor/jepa-wms",
                "--checkpoint", "/workspace/jepa-runtime/checkpoints/jepa_wm_metaworld.pth.tar",
                "--data-root", "/workspace/jepa-runtime/data/Metaworld/data",
                "--output", str(args.root / "evaluation-v1" / precision / f"shard-{shard:03d}")]
            log = (logs / f"{precision}-{shard}.log").open("x")
            process = subprocess.Popen(command, env={**env, "CUDA_VISIBLE_DEVICES": str(index)},
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            item = {"precision": precision, "shard": shard, "gpu": index, "pid": process.pid}
            running.append((process, log, item))
            print(json.dumps({"event": "started", **item}), flush=True)
        while running:
            if time.monotonic() - started > 7400:
                raise TimeoutError("Combined stage cap")
            for process, log, item in list(running):
                code = process.poll()
                if code is not None:
                    log.close()
                    running.remove((process, log, item))
                    if code:
                        raise RuntimeError(f"Combined shard failed: {item}, exit={code}")
                    completed.append(item)
                    print(json.dumps({"event": "completed", **item}), flush=True)
            write_json(logs / "progress.json", {"completed": completed, "running": [r[2] for r in running],
                "seconds": time.monotonic() - started, "fresh_confirmation": False})
            time.sleep(2)
        for precision in contract["precisions"]:
            command = [sys.executable, "-m", "offline_study.combined_run", "analyze", "--protocol", str(protocol),
                "--precision", precision, "--shards", str(args.root / "evaluation-v1" / precision),
                "--output", str(args.root / "analysis-v1" / precision)]
            with (logs / f"analysis-{precision}.log").open("x") as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
        write_json(logs / "DONE.json", {"status": "planned_combination_both_precisions_verified",
            "protocol_sha256": sha256(protocol), "analysis_sha256": {p: sha256(args.root / "analysis-v1" / p / "report.json")
            for p in contract["precisions"]}, "fresh_confirmation": False, "full_study_complete": False})
    except Exception as exc:
        write_json(logs / "FAILED.json", {"error": str(exc), "completed": completed})
        raise
    finally:
        for process, log, item in running:
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
