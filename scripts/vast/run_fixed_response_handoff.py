"""Bounded exclusive handoff; always restore the untouched non-rank dispatcher."""
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
PARENT, PREVIOUS_CHILD = 3722, 5740


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024**2), b""):
            h.update(part)
    return h.hexdigest()


def parent_identity():
    command = Path(f"/proc/{PARENT}/cmdline").read_bytes()
    if b"/workspace/jepa-runtime/nonrank-queue-code-v2/run_navigation_nonrank_remaining.py" not in command:
        raise ValueError("Non-rank dispatcher identity changed")


def execute(command, log, limit):
    with log.open("x") as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            if child.wait(timeout=limit) != 0:
                raise RuntimeError("Bounded job failed: " + str(log))
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            raise


def interrupted(signum, frame):
    raise RuntimeError("Handoff interrupted; terminate owned child and restore dispatcher")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sha256", required=True)
    args = parser.parse_args()
    parent_identity()
    if "T (stopped)" not in Path(f"/proc/{PARENT}/status").read_text():
        raise ValueError("Dispatcher must be paused before handoff")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        started = time.monotonic()
        while Path(f"/proc/{PREVIOUS_CHILD}").exists():
            stat = Path(f"/proc/{PREVIOUS_CHILD}/stat").read_text().split()
            if int(stat[3]) != PARENT:
                raise ValueError("Original shard parent changed")
            if stat[2] == "Z":
                break
            if b"offline_study.navigation_evaluate" not in Path(f"/proc/{PREVIOUS_CHILD}/cmdline").read_bytes():
                raise ValueError("Original shard identity changed")
            if time.monotonic() - started > 1200:
                raise TimeoutError("Current shard did not complete in handoff wait budget")
            time.sleep(5)
        previous = ROOT / "navigation-comparisons-20260907-v1/bfloat16/pointmaze/action_response_geometry/shard-6"
        previous_done = json.loads((previous / "DONE.json").read_text())
        for name in ("report", "window_metrics", "mechanism_diagnostics", "selection", "protocol"):
            if sha(previous / (name + ".json")) != previous_done[name + "_sha256"]:
                raise ValueError("Previous completed scientific shard changed")
        if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]).strip():
            raise ValueError("GPU not exclusively free after the prior full shard")
        assets = ROOT / "fixed-response-assets-20260908-v1"
        manifest = json.loads((assets / "DOWNLOAD_MANIFEST.json").read_text())
        receipt = json.loads((assets / "DONE.json").read_text())
        if (receipt["manifest_sha256"] != sha(assets / "DOWNLOAD_MANIFEST.json") or
                manifest["data_revision"] != "6116f042ae7ae4c8e3f1fd2f194f432615664182" or
                len(manifest["files"]) != 127):
            raise ValueError("Downloaded input manifest is not the full pinned source")
        for row in manifest["files"]:
            path = assets / row["path"]
            if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
                raise ValueError("Transferred source file differs: " + row["path"])
        code = ROOT / "fixed-response-code-20260908-v1"
        h = hashlib.sha256()
        for path in sorted((code / "src/offline_study").glob("*.py")):
            h.update(path.name.encode() + b"\0" + path.read_bytes())
        if h.hexdigest() != args.source_sha256:
            raise ValueError("Staged scientific source differs from tested source")
        os.environ.update(CUDA_VISIBLE_DEVICES="0", JEPA_VERIFIED_LOCAL_DINO="1",
            PYTHONPATH=str(code / "src") + ":/workspace/jepa-python/lib/python3.10/site-packages",
            LD_LIBRARY_PATH="/opt/conda/lib", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        checkpoint = assets / "jepa_wm_metaworld.pth.tar"
        common = ["--vendor", "/workspace/jepa_steering/vendor/jepa-wms", "--checkpoint", str(checkpoint),
            "--checkpoint-sha256", "c3297772c7af4e84c28bd0c0937e22398a888848305014bb3ab3f1508a310bd8",
            "--data-root", str(assets / "metaworld/data")]
        for task in ("reach", "reach-wall"):
            source = ROOT / "fixed-response-source-fits-20260908-v1" / task
            output = ROOT / "fixed-response-fit-20260908-v1" / task
            print(json.dumps({"launch": str(output), "old_rank_jobs_disabled": True}), flush=True)
            execute([PYTHON, "-u", "-m", "offline_study.fixed_response_fit", *common,
                "--config", str(code / "configs/fixed_response_rank4.json"), "--cohort", str(source / "cohort.json"),
                "--fit", str(source / "operator_rank"), "--device", "cuda:0", "--output", str(output)],
                ROOT / f"fixed-response-fit-{task}-20260908-v1.log", 4100)
            if not (output / "DONE.json").is_file() or (output / "FAILED.json").exists():
                raise ValueError("Fit lacks a valid completion receipt")
        for task in ("reach", "reach-wall"):
            output = ROOT / "fixed-response-check-20260908-v1" / task
            print(json.dumps({"launch": str(output), "scientific_efficacy": False}), flush=True)
            execute([PYTHON, "-u", "-m", "offline_study.fixed_response_check", *common,
                "--fit", str(ROOT / "fixed-response-fit-20260908-v1" / task), "--task", "mw-" + task,
                "--output", str(output)], ROOT / f"fixed-response-check-{task}-20260908-v1.log", 600)
        print(json.dumps({"status": "bounded_successor_fit_and_checks_complete", "full_cem_not_yet_run": True}), flush=True)
    finally:
        parent_identity()
        os.kill(PARENT, signal.SIGCONT)
        print(json.dumps({"nonrank_dispatcher_resumed": PARENT}), flush=True)


if __name__ == "__main__":
    main()
