"""Launch and aggregate communication-free intervention shards on local GPUs."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from offline_study.runtime.benchmark import filter_reviewed_development, select_rows
from offline_study.runtime.intervention_runner import summarize_interventions
from offline_study.runtime.multigpu import _stop_processes, normalize_device
from offline_study.core.protocol import sha256, summarize_metrics, write_json


def intervention_command(args, device: str, shard_index: int, num_shards: int) -> list[str]:
    return [
        sys.executable, "-m", "offline_study.runtime.intervention_runner",
        "--vendor", str(args.vendor),
        "--checkpoint", str(args.checkpoint),
        "--checkpoint-sha256", args.checkpoint_sha256,
        "--manifest", str(args.manifest),
        "--exposure-registry", str(args.exposure_registry),
        "--data-root", str(args.data_root),
        "--protocol", str(args.protocol),
        "--operator-bank", str(args.operator_bank),
        "--fit-receipt", str(args.fit_receipt),
        "--output", str(args.output / f"shard-{shard_index:03d}"),
        "--tasks", *args.tasks,
        "--max-trajectories", str(args.max_trajectories),
        "--batch-size", str(args.batch_size),
        "--warmup", str(args.warmup),
        "--device", device,
        "--prefetch-batches", str(args.prefetch_batches),
        "--shard-index", str(shard_index),
        "--num-shards", str(num_shards),
        "--max-runtime-seconds", str(args.max_runtime_seconds),
    ]


def _canonical_child_config(config: dict) -> dict:
    ignored = {"device", "output", "shard_index", "vendor", "checkpoint", "manifest",
               "exposure_registry", "data_root", "protocol", "operator_bank", "fit_receipt"}
    return {key: value for key, value in config.items() if key not in ignored}


def aggregate_shards(output: Path, shard_count: int, wall_seconds: float) -> dict:
    protocol_path = output / "protocol.json"
    manifest_path = output / "manifest.jsonl"
    registry_path = output / "exposure_registry.json"
    protocol = json.loads(protocol_path.read_text())
    reports, configs, selections, measurements = [], [], [], []
    for index in range(shard_count):
        directory = output / f"shard-{index:03d}"
        done = json.loads((directory / "DONE.json").read_text())
        hashes = {
            "report_sha256": directory / "report.json",
            "window_metrics_sha256": directory / "window_metrics.json",
            "selection_sha256": directory / "selection.json",
            "protocol_sha256": directory / "protocol.json",
            "fit_receipt_sha256": directory / "fit_receipt.json",
        }
        for field, path in hashes.items():
            if done.get(field) != sha256(path):
                raise ValueError(f"Shard {index} {field} mismatch")
        report = json.loads((directory / "report.json").read_text())
        config = json.loads((directory / "config.json").read_text())
        selection = json.loads((directory / "selection.json").read_text())
        rows = json.loads((directory / "window_metrics.json").read_text())
        if report.get("shard_index") != index or report.get("num_shards") != shard_count:
            raise ValueError(f"Shard {index} identity mismatch")
        if not report.get("gpu_execution_valid") or report.get("scientific_confirmation"):
            raise ValueError(f"Shard {index} is not a development-only GPU intervention")
        if report.get("protocol_sha256") != sha256(protocol_path):
            raise ValueError(f"Shard {index} protocol differs from aggregate input")
        if [arm["name"] for arm in protocol["arms"]] != report.get("arms"):
            raise ValueError(f"Shard {index} arm registry mismatch")
        selected_ids = {row["trajectory_id"] for row in selection}
        if not {row["trajectory_id"] for row in rows}.issubset(selected_ids):
            raise ValueError(f"Shard {index} measured outside its selection")
        reports.append(report)
        configs.append(config)
        selections.extend(selection)
        measurements.extend(rows)
    reference_config = _canonical_child_config(configs[0])
    if any(_canonical_child_config(config) != reference_config for config in configs[1:]):
        raise ValueError("Shard configurations differ beyond local file/device paths")
    trajectory_ids = [row["trajectory_id"] for row in selections]
    if len(trajectory_ids) != len(set(trajectory_ids)):
        raise ValueError("Trajectory selections overlap across intervention shards")
    source_rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    registry = json.loads(registry_path.read_text())
    source_rows = filter_reviewed_development(source_rows, registry, sha256(manifest_path))
    expected = select_rows(source_rows, reference_config["tasks"], "development",
                           reference_config["max_trajectories"], 0, 1)
    if set(trajectory_ids) != {row["trajectory_id"] for row in expected}:
        raise ValueError("GPU shards do not cover the exact global trajectory selection")
    arm_count = len(protocol["arms"])
    expected_evaluations = sum(len(row["starts"]) for row in expected) * arm_count
    if len(measurements) != expected_evaluations:
        raise ValueError("Intervention shards do not contain every expected window/arm evaluation")
    aggregation = summarize_interventions(measurements, protocol)
    mechanism_rows = [
        {"task": row["task"], "trajectory_id": row["trajectory_id"],
         "lineage_group": row["lineage_group"], "start": row["start"],
         "metrics": row["mechanism_diagnostics"]}
        for row in measurements if row["arm"] == "native" and "mechanism_diagnostics" in row
    ]
    concurrent_pipeline_seconds = max(report["measured_pipeline_seconds"] for report in reports)
    windows = len(measurements) // arm_count
    report = {
        "status": "real_multigpu_intervention_development_complete",
        "gpu_execution_valid": True,
        "scientific_confirmation": False,
        "category": protocol["category"],
        "arms": [arm["name"] for arm in protocol["arms"]],
        "primary_tasks": protocol["tasks"],
        "devices": [item["environment"]["device"] for item in reports],
        "shards": shard_count,
        "rollout_trajectories": len(set(trajectory_ids)),
        "independent_lineage_groups": len({row["lineage_group"] for row in measurements}),
        "independence_unit": "lineage_group",
        "windows": windows,
        "arm_evaluations": len(measurements),
        "wall_seconds": wall_seconds,
        "measured_concurrent_pipeline_seconds": concurrent_pipeline_seconds,
        "end_to_end_windows_per_second": windows / wall_seconds,
        "pipeline_windows_per_second": windows / concurrent_pipeline_seconds,
        "task_measured_trajectory_counts": dict(Counter(row["task"] for row in expected)),
        "aggregation": aggregation,
        "mechanism_diagnostics": summarize_metrics(mechanism_rows) if mechanism_rows else None,
        "protocol_sha256": sha256(protocol_path),
        "manifest_sha256": sha256(manifest_path),
        "exposure_registry_sha256": sha256(registry_path),
        "per_shard": [{
            "shard_index": index,
            "report": f"shard-{index:03d}/report.json",
            "report_sha256": sha256(output / f"shard-{index:03d}" / "report.json"),
            "windows": item["windows"],
            "rollout_trajectories": item["rollout_trajectories"],
            "independent_lineage_groups": item["independent_lineage_groups"],
            "measured_pipeline_seconds": item["measured_pipeline_seconds"],
            "peak_reserved_gpu_bytes": item["peak_reserved_gpu_bytes"],
        } for index, item in enumerate(reports)],
        "limitations": [
            "Development comparison only; this is not protected-holdout confirmation",
            "GPU shards communicate no tensors and contain disjoint trajectories",
            "End-to-end throughput includes per-process model setup",
            "Forecast embedding error is not closed-loop physical behavior",
            "No simulator, CEM, autonomous method search, or parameter training occurs",
        ],
    }
    write_json(output / "window_metrics.json", measurements)
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--exposure-registry", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--operator-bank", type=Path, required=True)
    parser.add_argument("--fit-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--max-trajectories", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--devices", nargs="+", default=["0"])
    parser.add_argument("--prefetch-batches", type=int, default=2)
    parser.add_argument("--shard-offset", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=0,
                        help="Global shard count; 0 means the number of local devices")
    parser.add_argument("--max-runtime-seconds", type=int, default=5400,
                        help="Per-child measured execution cap")
    parser.add_argument("--supervisor-grace-seconds", type=int, default=600,
                        help="Additional allowance for model setup and receipt writing")
    args = parser.parse_args()
    try:
        devices = [normalize_device(device) for device in args.devices]
    except ValueError as exc:
        parser.error(str(exc))
    num_shards = args.num_shards or len(devices)
    shard_indices = list(range(args.shard_offset, args.shard_offset + len(devices)))
    if (len(devices) != len(set(devices)) or args.batch_size < 1 or
            args.max_trajectories < 1 or args.warmup < 0 or args.prefetch_batches < 0 or
            args.max_runtime_seconds < 1 or args.supervisor_grace_seconds < 0 or
            args.shard_offset < 0 or num_shards < 1 or
            args.shard_offset + len(devices) > num_shards):
        parser.error("Invalid devices, batch, trajectory, pipeline, runtime, or shard range")
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.manifest, args.output / "manifest.jsonl")
    shutil.copyfile(args.exposure_registry, args.output / "exposure_registry.json")
    shutil.copyfile(args.protocol, args.output / "protocol.json")
    shutil.copyfile(args.fit_receipt, args.output / "fit_receipt.json")
    write_json(args.output / "launch_config.json", {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "devices": devices,
        "shard_indices": shard_indices,
        "global_num_shards": num_shards,
        "parallelism": "one process per GPU; disjoint trajectory shards; no collectives",
    })
    processes, logs = [], []
    started = time.monotonic()
    timed_out = False
    try:
        for device, shard_index in zip(devices, shard_indices, strict=True):
            log = (args.output / f"shard-{shard_index:03d}.log").open("w")
            logs.append(log)
            processes.append(subprocess.Popen(
                intervention_command(args, device, shard_index, num_shards),
                stdout=log, stderr=subprocess.STDOUT,
            ))
        hard_cap = args.max_runtime_seconds + args.supervisor_grace_seconds
        while True:
            codes = [process.poll() for process in processes]
            if any(code not in (None, 0) for code in codes):
                _stop_processes(processes)
                break
            if all(code == 0 for code in codes):
                break
            if time.monotonic() - started >= hard_cap:
                timed_out = True
                _stop_processes(processes)
                break
            time.sleep(.25)
        codes = [process.poll() for process in processes]
        if timed_out or any(code != 0 for code in codes):
            failure = {"status": "failed", "timed_out": timed_out,
                       "supervisor_hard_cap_seconds": hard_cap, "return_codes": codes,
                       "shard_indices": shard_indices}
            write_json(args.output / "FAILED.json", failure)
            raise RuntimeError(f"GPU intervention shard launch failed: {failure}")
        elapsed = time.monotonic() - started
        if shard_indices == list(range(num_shards)):
            report = aggregate_shards(args.output, num_shards, elapsed)
            done = {"status": report["status"],
                    "report_sha256": sha256(args.output / "report.json"),
                    "window_metrics_sha256": sha256(args.output / "window_metrics.json")}
        else:
            done = {"status": "real_intervention_partial_shards_complete",
                    "scientific_result_complete": False,
                    "shard_indices": shard_indices, "global_num_shards": num_shards,
                    "wall_seconds": elapsed,
                    "shard_done_sha256": {
                        str(index): sha256(args.output / f"shard-{index:03d}" / "DONE.json")
                        for index in shard_indices}}
        write_json(args.output / "DONE.json", done)
        print(json.dumps(done), flush=True)
    except BaseException as exc:
        if any(process.poll() is None for process in processes):
            _stop_processes(processes)
        if not (args.output / "FAILED.json").exists():
            write_json(args.output / "FAILED.json", {"status": "failed", "error": str(exc),
                                                     "return_codes": [p.poll() for p in processes]})
        raise
    finally:
        for log in logs:
            log.close()


def aggregate_main():
    parser = argparse.ArgumentParser(description="Aggregate a complete gathered intervention shard set")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--wall-seconds", type=float, required=True)
    args = parser.parse_args()
    report = aggregate_shards(args.output, args.num_shards, args.wall_seconds)
    write_json(args.output / "DONE.json", {
        "status": report["status"],
        "report_sha256": sha256(args.output / "report.json"),
        "window_metrics_sha256": sha256(args.output / "window_metrics.json"),
    })
    print(json.dumps({key: report[key] for key in (
        "status", "windows", "rollout_trajectories", "independent_lineage_groups",
        "wall_seconds", "end_to_end_windows_per_second")}), flush=True)


if __name__ == "__main__":
    main()
