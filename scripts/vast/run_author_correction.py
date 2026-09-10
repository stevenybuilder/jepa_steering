"""Bounded local worker scheduling on an already-authorized US GPU instance."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=["fit", "evaluate"])
    parser.add_argument("--devices", required=True, nargs="+")
    parser.add_argument("--tasks", nargs="+", default=["reach", "reach-wall", "pusht"])
    parser.add_argument("--precisions", nargs="+", default=["bfloat16", "float32"])
    parser.add_argument("--fits-name", default="fits-v1")
    parser.add_argument("--evaluation-name", default="evaluation-v1")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--analyze-after", action="store_true",
                        help="Verify both completed shards and analyze each fixed sweep before closure")
    parser.add_argument("--analysis-name", default="analysis-v1")
    parser.add_argument("--reuse-completed", action="store_true",
                        help="Verify immutable completed shards and schedule only never-started outputs")
    parser.add_argument("--categories", nargs="+", choices=["vision_action_coupling", "action_response_geometry",
        "operator_rank", "distribution_layer", "distribution_spatial"],
        default=["vision_action_coupling", "action_response_geometry", "operator_rank", "distribution_layer", "distribution_spatial"])
    args = parser.parse_args()
    runtime = args.root.parent
    hashes = {"metaworld": "c3297772c7af4e84c28bd0c0937e22398a888848305014bb3ab3f1508a310bd8",
              "pusht": "9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb"}
    categories = args.categories
    jobs = []
    for precision in args.precisions:
        for task in args.tasks:
            kind = "pusht" if task == "pusht" else "metaworld"
            shared = ["--vendor", "/workspace/jepa_steering/vendor/jepa-wms", "--checkpoint",
                str(runtime / "checkpoints" / ("jepa_wm_" + kind + ".pth.tar")),
                "--checkpoint-sha256", hashes[kind], "--cohort", str(args.root / "cohorts" / task / "cohort.json"),
                "--data-root", str(runtime / "data" / ("pusht_noise" if kind == "pusht" else "Metaworld/data")),
                "--device", "cuda:0", "--precision", precision]
            fit = args.root / args.fits_name / precision / task
            if args.stage == "fit":
                jobs.append((f"fit-{precision}-{task}", [sys.executable, "-m", "offline_study.author_fit",
                             *shared, "--output", str(fit)], 1800))
            else:
                if not (fit / "DONE.json").is_file():
                    raise ValueError(f"Fit is not complete: {fit}")
                for category in categories:
                    for shard in range(2):
                        output = args.root / args.evaluation_name / precision / task / category / f"shard-{shard:03d}"
                        if args.reuse_completed and output.exists():
                            from offline_study.protocol import sha256
                            done = json.loads((output / "DONE.json").read_text())
                            for filename in ("report", "window_metrics", "selection", "protocol", "mechanism_diagnostics"):
                                if done[filename + "_sha256"] != sha256(output / (filename + ".json")):
                                    raise ValueError("Completed shard changed; cannot reuse")
                            report = json.loads((output / "report.json").read_text())
                            if (report["protocol_sha256"] != sha256(fit / category / "protocol.json") or
                                    report["fit_receipt_sha256"] != sha256(fit / category / "fit_receipt.json") or
                                    report["precision"] != precision or not report["zero_dose_identity"] or
                                    not report["native_instrumentation_fidelity"]):
                                raise ValueError("Completed shard belongs to a different experiment")
                            continue
                        jobs.append((f"eval-{precision}-{task}-{category}-{shard}",
                            [sys.executable, "-m", "offline_study.author_evaluate", *shared, "--fit", str(fit / category),
                             "--output", str(output), "--shard-index", str(shard), "--shard-count", "2"], 7200))
    running, done = {}, []
    logs = args.root / "logs"
    logs.mkdir(exist_ok=True)
    receipt_path = logs / (args.tag + ".json")
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    def interrupted(signum, frame):
        raise RuntimeError(f"Scheduler interrupted by signal {signum}; closing its owned workers")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        while jobs or running:
            for device in args.devices:
                if not jobs or device in running:
                    continue
                name, command, cap = jobs.pop(0)
                log = (logs / (name + "-" + args.tag + ".log")).open("x")
                env = {**os.environ, "PYTHONPATH": str(args.source), "CUDA_VISIBLE_DEVICES": device,
                       "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                       "LD_LIBRARY_PATH": "/opt/conda/lib:" + os.environ.get("LD_LIBRARY_PATH", "")}
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                running[device] = (name, process, log, time.monotonic(), cap)
                print(json.dumps({"event": "started", "job": name, "gpu": device, "pid": process.pid}), flush=True)
            for device, (name, process, log, start, cap) in list(running.items()):
                if time.monotonic() - start > cap:
                    raise TimeoutError(f"Stage time cap: {name}")
                code = process.poll()
                if code is not None:
                    log.close()
                    del running[device]
                    if code:
                        raise RuntimeError(f"Job failed: {name}, exit {code}")
                    done.append({"job": name, "gpu": device, "seconds": time.monotonic() - start})
                    print(json.dumps({"event": "completed", **done[-1]}), flush=True)
            time.sleep(.5)
        analyses = []
        if args.analyze_after:
            if args.stage != "evaluate":
                raise ValueError("Analysis requires an evaluation stage")
            for precision in args.precisions:
                for task in args.tasks:
                    for category in categories:
                        output = args.root / args.analysis_name / precision / task / category
                        command = [sys.executable, "-m", "offline_study.author_analyze",
                            "--cohort", str(args.root / "cohorts" / task / "cohort.json"),
                            "--fit", str(args.root / args.fits_name / precision / task / category),
                            "--shards", str(args.root / args.evaluation_name / precision / task / category),
                            "--output", str(output)]
                        env = {**os.environ, "PYTHONPATH": str(args.source), "OMP_NUM_THREADS": "1",
                               "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
                        with (logs / f"analysis-{precision}-{task}-{category}-{args.tag}.log").open("x") as log:
                            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           check=True, timeout=600)
                        analyses.append(str(output))
        receipt_path.write_text(json.dumps({"status": "all_scheduled_jobs_complete", "jobs": done,
                                           "verified_analyses": analyses}, indent=2))
    except Exception as exc:
        receipt_path.write_text(json.dumps({"status": "failed", "error": str(exc), "completed_jobs": done}, indent=2))
        raise
    finally:
        for name, process, log, _, _ in running.values():
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
