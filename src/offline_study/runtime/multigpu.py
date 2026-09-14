"""Launch bounded, communication-free JEPA trajectory shards on local GPUs."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from offline_study.runtime.benchmark import filter_reviewed_development, select_rows
from offline_study.core.protocol import sha256, summarize_metrics, write_json


def normalize_device(value: str) -> str:
    if value.isdigit():
        return f"cuda:{value}"
    if value.startswith("cuda:") and value[5:].isdigit():
        return value
    raise ValueError(f"Expected a CUDA index or cuda:N, got {value!r}")


def benchmark_command(args, device: str, shard_index: int, num_shards: int) -> list[str]:
    output = args.output / f"shard-{shard_index:03d}"
    command = [
        sys.executable, "-m", "offline_study.runtime.benchmark",
        "--backend", "jepa",
        "--vendor", str(args.vendor),
        "--checkpoint", str(args.checkpoint),
        "--checkpoint-sha256", args.checkpoint_sha256,
        "--manifest", str(args.manifest),
        "--exposure-registry", str(args.exposure_registry),
        "--data-root", str(args.data_root),
        "--output", str(output),
        "--tasks", *args.tasks,
        "--split", args.split,
        "--max-trajectories", str(args.max_trajectories),
        "--batch-size", str(args.batch_size),
        "--warmup", str(args.warmup),
        "--device", device,
        "--precision", args.precision,
        "--prefetch-batches", str(args.prefetch_batches),
        "--cache-queue-depth", str(args.cache_queue_depth),
        "--num-shards", str(num_shards),
        "--shard-index", str(shard_index),
    ]
    if args.allow_tf32:
        command.append("--allow-tf32")
    if args.write_cache:
        command.append("--write-cache")
    return command


def _canonical_child_config(config: dict) -> dict:
    ignored = {"device", "output", "shard_index"}
    return {key: value for key, value in config.items() if key not in ignored}


def aggregate_shards(output: Path, shard_count: int, wall_seconds: float) -> dict:
    reports, configs, selections, windows = [], [], [], []
    for index in range(shard_count):
        directory = output / f"shard-{index:03d}"
        done = json.loads((directory / "DONE.json").read_text())
        if done.get("report_sha256") != sha256(directory / "report.json"):
            raise ValueError(f"Shard {index} report hash mismatch")
        if done.get("window_metrics_sha256") != sha256(directory / "window_metrics.json"):
            raise ValueError(f"Shard {index} metric hash mismatch")
        if done.get("selection_sha256") != sha256(directory / "selection.json"):
            raise ValueError(f"Shard {index} selection hash mismatch")
        report = json.loads((directory / "report.json").read_text())
        config = json.loads((directory / "config.json").read_text())
        selection = json.loads((directory / "selection.json").read_text())
        measured = json.loads((directory / "window_metrics.json").read_text())
        if report.get("shard_index") != index or report.get("num_shards") != shard_count:
            raise ValueError(f"Shard {index} identity mismatch")
        if not report.get("gpu_benchmark_valid"):
            raise ValueError(f"Shard {index} is not a valid real GPU benchmark")
        if report.get("selection_sha256") != done.get("selection_sha256"):
            raise ValueError(f"Shard {index} report/selection receipt mismatch")
        selected_ids = {row["trajectory_id"] for row in selection}
        if not {row["trajectory_id"] for row in measured}.issubset(selected_ids):
            raise ValueError(f"Shard {index} contains a measurement outside its selection")
        reports.append(report)
        configs.append(config)
        selections.extend(selection)
        windows.extend(measured)
    reference_config = _canonical_child_config(configs[0])
    if any(_canonical_child_config(config) != reference_config for config in configs[1:]):
        raise ValueError("Shard configurations differ beyond device/output/index")
    trajectory_ids = [row["trajectory_id"] for row in selections]
    if len(trajectory_ids) != len(set(trajectory_ids)):
        raise ValueError("Trajectory selections overlap across GPU shards")
    if reference_config.get("backend") == "jepa":
        manifest_path = Path(reference_config["manifest"])
        source_rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
        registry_path = Path(reference_config["exposure_registry"])
        registry = json.loads(registry_path.read_text())
        source_rows = filter_reviewed_development(source_rows, registry, sha256(manifest_path))
        expected = select_rows(
            source_rows,
            reference_config["tasks"],
            reference_config["split"],
            reference_config["max_trajectories"],
            0,
            1,
        )
        if set(trajectory_ids) != {row["trajectory_id"] for row in expected}:
            raise ValueError("GPU shard selections do not cover the exact global selection")
    aggregation = summarize_metrics(windows)
    measured_ids = {row["trajectory_id"] for row in windows}
    pipeline_seconds = max(report["measured_pipeline_seconds"] for report in reports)
    task_counts = Counter(row["task"] for row in aggregation["per_trajectory"])
    report = dict(
        status="real_multigpu_baseline_benchmark_complete",
        gpu_benchmark_valid=True,
        efficacy_claim=False,
        devices=[item["environment"]["device"] for item in reports],
        precision=reports[0]["execution"]["precision"],
        allow_tf32=reports[0]["execution"]["allow_tf32"],
        shards=shard_count,
        rollout_trajectories=len(measured_ids),
        independent_lineage_groups=len({row["lineage_group"] for row in windows}),
        independence_unit="lineage_group",
        windows=len(windows),
        wall_seconds=wall_seconds,
        measured_concurrent_pipeline_seconds=pipeline_seconds,
        end_to_end_windows_per_second=len(windows) / wall_seconds,
        pipeline_windows_per_second=len(windows) / pipeline_seconds,
        task_measured_trajectory_counts=dict(task_counts),
        aggregation=aggregation,
        per_shard=[dict(
            shard_index=index,
            report=f"shard-{index:03d}/report.json",
            report_sha256=sha256(output / f"shard-{index:03d}" / "report.json"),
            windows=item["windows"],
            rollout_trajectories=item["rollout_trajectories"],
            independent_lineage_groups=item["independent_lineage_groups"],
            measured_pipeline_seconds=item["measured_pipeline_seconds"],
            peak_reserved_gpu_bytes=item["peak_reserved_gpu_bytes"],
        ) for index, item in enumerate(reports)],
        manifest_sha256=reports[0]["manifest_sha256"],
        exposure_registry_sha256=reports[0]["exposure_registry_sha256"],
        limitations=[
            "Unedited baseline only; intervention throughput is not measured",
            "GPU shards communicate no tensors and contain disjoint trajectories",
            "End-to-end throughput includes per-process model setup; pipeline throughput does not",
            "Do not assume scaling beyond the measured device count",
            "A split label alone does not establish that a lineage group is historically untouched",
        ],
    )
    write_json(output / "window_metrics.json", windows)
    write_json(output / "report.json", report)
    return report


def _stop_processes(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and any(process.poll() is None for process in processes):
        time.sleep(0.1)
    for process in processes:
        if process.poll() is None:
            process.kill()
    for process in processes:
        process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--exposure-registry", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--split", choices=["fit", "development"], default="development")
    parser.add_argument("--max-trajectories", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--devices", nargs="+", default=["0"],
                        help="CUDA indices, for example: --devices 0 1")
    parser.add_argument("--precision", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--prefetch-batches", type=int, default=2)
    parser.add_argument("--cache-queue-depth", type=int, default=2)
    parser.add_argument("--write-cache", action="store_true")
    parser.add_argument("--max-runtime-seconds", type=int, default=1800,
                        help="Hard wall-clock cap for the initial benchmark processes")
    args = parser.parse_args()
    try:
        devices = [normalize_device(device) for device in args.devices]
    except ValueError as exc:
        parser.error(str(exc))
    if len(devices) != len(set(devices)):
        parser.error("Each device may have at most one benchmark process")
    if (args.batch_size < 1 or args.max_trajectories < 0 or args.warmup < 0 or
            args.prefetch_batches < 0 or args.cache_queue_depth < 1 or args.max_runtime_seconds < 1):
        parser.error("Invalid batch, trajectory, pipeline, or runtime bound")
    if args.allow_tf32 and args.precision != "float32":
        parser.error("--allow-tf32 is valid only with --precision float32")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "launch_config.json", {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "devices": devices,
        "parallelism": "one process per GPU; disjoint trajectory shards; no collectives",
    })
    processes, logs = [], []
    started = time.monotonic()
    timed_out = False
    try:
        for index, device in enumerate(devices):
            log = (args.output / f"shard-{index:03d}.log").open("w")
            logs.append(log)
            processes.append(subprocess.Popen(
                benchmark_command(args, device, index, len(devices)),
                stdout=log,
                stderr=subprocess.STDOUT,
            ))
        while True:
            return_codes = [process.poll() for process in processes]
            if any(code not in (None, 0) for code in return_codes):
                _stop_processes(processes)
                break
            if all(code == 0 for code in return_codes):
                break
            if time.monotonic() - started >= args.max_runtime_seconds:
                timed_out = True
                _stop_processes(processes)
                break
            time.sleep(0.25)
        return_codes = [process.poll() for process in processes]
        if timed_out or any(code != 0 for code in return_codes):
            failure = {
                "status": "failed",
                "timed_out": timed_out,
                "max_runtime_seconds": args.max_runtime_seconds,
                "return_codes": return_codes,
                "logs": [f"shard-{index:03d}.log" for index in range(len(devices))],
            }
            write_json(args.output / "FAILED.json", failure)
            raise RuntimeError(f"GPU shard launch failed: {failure}")
        elapsed = time.monotonic() - started
        report = aggregate_shards(args.output, len(devices), elapsed)
        write_json(args.output / "DONE.json", {
            "status": report["status"],
            "report_sha256": sha256(args.output / "report.json"),
            "window_metrics_sha256": sha256(args.output / "window_metrics.json"),
        })
        print(json.dumps({key: report[key] for key in (
            "status", "devices", "windows", "rollout_trajectories",
            "independent_lineage_groups",
            "wall_seconds", "end_to_end_windows_per_second")}), flush=True)
    except BaseException as exc:
        if any(process.poll() is None for process in processes):
            _stop_processes(processes)
        failed_path = args.output / "FAILED.json"
        if not failed_path.exists():
            write_json(failed_path, {
                "status": "failed",
                "error": str(exc),
                "return_codes": [process.poll() for process in processes],
            })
        raise
    finally:
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
