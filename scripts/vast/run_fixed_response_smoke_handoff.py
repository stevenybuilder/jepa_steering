"""Exclusive full-episode handoff after an existing complete scientific shard."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path("/workspace/jepa-runtime")
PYTHON = "/workspace/jepa-planning-python/bin/python"
DISPATCHER = b"/workspace/jepa-runtime/nonrank-queue-code-v2/run_navigation_nonrank_remaining.py"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 ** 2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=int, required=True)
    parser.add_argument("--child", type=int, required=True)
    parser.add_argument("--prior-output", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    args = parser.parse_args()
    def identity():
        if DISPATCHER not in Path(f"/proc/{args.parent}/cmdline").read_bytes():
            raise ValueError("Owned navigation dispatcher changed")
    def interrupt(signum, frame):
        raise RuntimeError("Bounded CEM handoff interrupted")
    identity()
    child_command = Path(f"/proc/{args.child}/cmdline").read_bytes()
    if (b"offline_study.navigation_evaluate" not in child_command or
            str(args.prior_output).encode() not in child_command or
            int(Path(f"/proc/{args.child}/stat").read_text().split()[3]) != args.parent):
        raise ValueError("Current scientific shard identity differs")
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    running = None
    os.kill(args.parent, signal.SIGSTOP)
    try:
        started = time.monotonic()
        while Path(f"/proc/{args.child}").exists():
            fields = Path(f"/proc/{args.child}/stat").read_text().split()
            if int(fields[3]) != args.parent:
                raise ValueError("Prior shard PID reused")
            if fields[2] == "Z":
                break
            if Path(f"/proc/{args.child}/cmdline").read_bytes() != child_command:
                raise ValueError("Prior shard command changed")
            if time.monotonic() - started > 1800:
                raise TimeoutError("Prior shard still running; release reservation")
            time.sleep(5)
        done = json.loads((args.prior_output / "DONE.json").read_text())
        for name in ("report", "window_metrics", "mechanism_diagnostics", "selection", "protocol"):
            if sha(args.prior_output / (name + ".json")) != done[name + "_sha256"]:
                raise ValueError("Prior full scientific shard did not verify")
        if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]).strip():
            raise ValueError("GPU not exclusively available")
        code = ROOT / "fixed-response-code-20260908-v2"
        digest = hashlib.sha256()
        for path in sorted((code / "src/offline_study").glob("*.py")):
            digest.update(path.name.encode() + b"\0" + path.read_bytes())
        if digest.hexdigest() != args.source_sha256:
            raise ValueError("Staged CEM source differs from tested source")
        os.environ.update(CUDA_VISIBLE_DEVICES="0", JEPA_VERIFIED_LOCAL_DINO="1", MUJOCO_GL="egl",
            PYOPENGL_PLATFORM="egl", PYTHONPATH=str(code / "src") + ":/workspace/jepa-python/lib/python3.10/site-packages",
            LD_LIBRARY_PATH="/opt/conda/lib", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        for task in ("reach", "reach-wall"):
            output = ROOT / "fixed-response-cem-20260908-v1" / task
            command = [PYTHON, "-u", "-m", "offline_study.fixed_response_smoke",
                "--task", task, "--vendor", "/workspace/jepa_steering/vendor/jepa-wms",
                "--checkpoint", str(ROOT / "fixed-response-assets-20260908-v1/jepa_wm_metaworld.pth.tar"),
                "--fit", str(ROOT / "fixed-response-fit-20260908-v1" / task),
                "--numerical-check", str(ROOT / "fixed-response-check-20260908-v1" / task),
                "--checked-source", str(ROOT / "fixed-response-code-20260908-v1/src/offline_study"),
                "--output", str(output)]
            print(json.dumps({"launch": str(output), "scientific_efficacy": False}), flush=True)
            with (ROOT / f"fixed-response-cem-{task}-20260908-v1.log").open("x") as stream:
                running = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                if running.wait(timeout=3600) != 0:
                    raise RuntimeError("Full-CEM engineering failed; do not promote partial results")
            running = None
            if not (output / "DONE.json").exists() or (output / "FAILED.json").exists():
                raise ValueError("Missing full-CEM completion receipt")
        print(json.dumps({"status": "full_cem_checks_complete_not_behavioral_efficacy"}), flush=True)
    finally:
        if running is not None and running.poll() is None:
            os.killpg(running.pid, signal.SIGTERM)
            try:
                running.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(running.pid, signal.SIGKILL)
                running.wait()
        identity()
        os.kill(args.parent, signal.SIGCONT)
        print(json.dumps({"nonrank_dispatcher_resumed": args.parent}), flush=True)


if __name__ == "__main__":
    main()
