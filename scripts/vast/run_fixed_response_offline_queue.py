"""One exclusively assigned GPU/task; all frozen precision/trajectory shards."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess

ROOT = Path("/workspace/jepa-runtime")
PYTHON = "/workspace/jepa-planning-python/bin/python"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 ** 2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("reach", "reach-wall"), required=True)
    parser.add_argument("--attempt", choices=("v1", "v2"), required=True,
                        help="Separate retry root; never overwrite failed receipts")
    args = parser.parse_args()
    gpu = {"reach": "0", "reach-wall": "1"}[args.task]
    code = ROOT / "fixed-response-code-20260908-v3"
    evidence = ROOT / "fixed-response-evidence-20260908-v1"
    protocol = evidence / "offline-freeze-v2" / args.task / "protocol.json"
    frozen = json.loads(protocol.read_text())
    digest = hashlib.sha256()
    for path in sorted((code / "src/offline_study").glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    if digest.hexdigest() != frozen["source_sha256"]:
        raise ValueError("Offline source changed after freezing")
    assets = ROOT / "fixed-response-assets-20260908-v1"
    manifest = json.loads((assets / "DOWNLOAD_MANIFEST.json").read_text())
    done = json.loads((assets / "DONE.json").read_text())
    if done["manifest_sha256"] != sha(assets / "DOWNLOAD_MANIFEST.json") or len(manifest["files"]) != 127:
        raise ValueError("Incomplete original input population")
    for row in manifest["files"]:
        path = assets / row["path"]
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError("Transferred input checksum mismatch")
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu, JEPA_VERIFIED_LOCAL_DINO="1",
        PYTHONPATH=str(code / "src") + ":/workspace/jepa-python/lib/python3.10/site-packages",
        LD_LIBRARY_PATH="/opt/conda/lib", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    subprocess.run([PYTHON, "-c", "import cv2, torchcodec; print('Video dependencies ready')"], check=True)
    running = None
    def interrupt(signum, frame):
        raise RuntimeError("Owned offline queue interrupted; preserve partial outputs")
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    fit = evidence / "fits" / args.task
    base = [PYTHON, "-u", "-m", "offline_study.fixed_response_offline"]
    try:
        for precision in ("bfloat16", "float32"):
            shard_root = ROOT / ("fixed-response-offline-20260908-" + args.attempt) / args.task / precision
            for index in range(4):
                output = shard_root / f"shard-{index}"
                log = ROOT / f"fixed-offline-{args.task}-{precision}-{index}-20260908-{args.attempt}.log"
                command = base + ["evaluate", "--protocol", str(protocol), "--fit", str(fit),
                    "--precision", precision, "--shard-index", str(index), "--output", str(output),
                    "--vendor", "/workspace/jepa_steering/vendor/jepa-wms", "--checkpoint",
                    str(assets / "jepa_wm_metaworld.pth.tar"), "--data-root", str(assets / "metaworld/data")]
                print(json.dumps({"launch": str(output), "physical_gpu": gpu}), flush=True)
                with log.open("x") as stream:
                    running = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    if running.wait(timeout=7500) != 0:
                        raise RuntimeError("Offline shard failed; no partial promotion")
                running = None
            output = ROOT / ("fixed-response-offline-analysis-20260908-" + args.attempt) / args.task / precision
            with (ROOT / f"fixed-offline-analysis-{args.task}-{precision}-20260908-{args.attempt}.log").open("x") as stream:
                running = subprocess.Popen(base + ["analyze", "--protocol", str(protocol), "--fit", str(fit),
                    "--precision", precision, "--shards", str(shard_root), "--output", str(output)],
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                if running.wait(timeout=1800) != 0:
                    raise RuntimeError("Complete paired offline analysis failed")
            running = None
            print(json.dumps({"completed_analysis": str(output), "fresh_confirmation": False}), flush=True)
    finally:
        if running is not None and running.poll() is None:
            os.killpg(running.pid, signal.SIGTERM)
            try:
                running.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(running.pid, signal.SIGKILL)
                running.wait()


if __name__ == "__main__":
    main()
