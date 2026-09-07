"""Bounded native baselines on two explicitly assigned GPUs, no rental API."""
import os
import signal
import subprocess
import sys
from pathlib import Path


def main():
    root = Path("/workspace/jepa-runtime/author-correction-20260907")
    processes = []
    try:
        for device, precision in (("4", "bfloat16"), ("5", "float32")):
            cmd = [sys.executable, "-m", "offline_study.author_baseline", "--vendor", "/workspace/jepa_steering/vendor/jepa-wms",
                "--checkpoint", "/workspace/jepa-runtime/checkpoints/jepa_wm_metaworld.pth.tar", "--checkpoint-sha256",
                "c3297772c7af4e84c28bd0c0937e22398a888848305014bb3ab3f1508a310bd8",
                "--manifest", str(root / "inputs/metaworld-manifest.jsonl"), "--registry", str(root / "inputs/metaworld-registry.json"),
                "--parity-cohort", str(root / "cohorts/reach/cohort.json"), "--data-root", "/workspace/jepa-runtime/data/Metaworld/data",
                "--precision", precision, "--output", str(root / "broad-baseline-v1" / precision)]
            env = {**os.environ, "PYTHONPATH": str(root / "code-v3/src"), "CUDA_VISIBLE_DEVICES": device,
                   "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "LD_LIBRARY_PATH": "/opt/conda/lib"}
            log = (root / "logs" / ("broad-baseline-" + precision + ".log")).open("x")
            process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append((process, log))
        for process, _ in processes:
            code = process.wait(timeout=7200)
            if code:
                raise RuntimeError(f"Broad baseline failed: {code}")
    finally:
        for process, log in processes:
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
