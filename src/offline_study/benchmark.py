"""Measure a frozen recorded-action rollout; never run CEM or a simulator."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from .backends import JepaBackend, ToyBackend
from .protocol import sha256, shard_for, summarize_metrics, validate_manifest, window_starts, write_json
from .vendor import open_dataset


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def elapsed_call(fn, device):
    synchronize(device)
    start = time.perf_counter()
    value = fn()
    synchronize(device)
    return value, time.perf_counter() - start


def score_predictions(predicted, target, horizons=(1, 3, 6)):
    """Prediction has [time,batch,...], target has [batch,time,...]; index 0 is context."""
    metrics = {}
    for modality in ("visual", "proprio"):
        for h in horizons:
            delta = predicted[modality][h].float() - target[modality][:, h].float()
            metrics[f"{modality}_mse_h{h}"] = delta.square().flatten(1).mean(1).cpu().tolist()
    batch = len(next(iter(metrics.values())))
    return [{key: values[i] for key, values in metrics.items()} for i in range(batch)]


def select_rows(rows, tasks, split, max_trajectories, shard_index, num_shards):
    selected = [r for r in rows if r["split"] == split and (tasks == ["all"] or r["task"] in tasks)]
    if tasks != ["all"]:
        missing = set(tasks) - {r["task"] for r in selected}
        if missing:
            raise ValueError(f"Requested tasks missing from selected split: {sorted(missing)}")
    # Round-robin across tasks prevents a small timing panel containing only the first task.
    groups = defaultdict(list)
    for row in sorted(selected, key=lambda r: r["trajectory_id"]):
        groups[row["task"]].append(row)
    selected = []
    for i in range(max((len(g) for g in groups.values()), default=0)):
        for task in sorted(groups):
            if i < len(groups[task]):
                selected.append(groups[task][i])
    if max_trajectories:
        if max_trajectories < len(groups):
            raise ValueError("max-trajectories is smaller than requested task coverage")
        selected = selected[:max_trajectories]
    return [r for r in selected if shard_for(r["trajectory_id"], num_shards) == shard_index]


def toy_rows(count):
    return [dict(trajectory_id=f"toy:{i}", dataset="toy", source_pool="fixture", index=i,
                 task=("mw-reach", "mw-reach-wall", "pusht")[i % 3], split="development",
                 length=61, horizon=6, stride=5, windows_requested=4,
                 starts=window_starts(61), exposure_status="synthetic") for i in range(count)]


def batches(rows, batch_size, backend, data_root):
    buffers = []
    datasets = {}
    for row in rows:
        if backend.synthetic:
            generator = torch.Generator().manual_seed(row["index"])
            visual = torch.randint(0, 256, (row["length"], 3, 16, 16), generator=generator, dtype=torch.uint8)
            proprio = torch.randn(row["length"], 4, generator=generator)
            actions = torch.randn(row["length"], 4, generator=generator) * .01
        else:
            key = (row["dataset"], row["source_pool"])
            if key not in datasets:
                datasets[key] = open_dataset(key[0], data_root, key[1])
            dset = datasets[key]
            if int(dset.get_seq_length(row["index"])) != row["length"]:
                raise ValueError("Source length no longer matches inventory")
            if row["dataset"] == "metaworld" and dset.dataset[row["index"]]["task"] != row["task"]:
                raise ValueError("Source task/row order no longer matches inventory")
            obs, actions, _, _, _ = dset[row["index"]]
            visual = (obs["visual"] * 255).round().clamp(0, 255).to(torch.uint8)
            proprio = obs["proprio"]
        for start in row["starts"]:
            frames = [start + i * row["stride"] for i in range(row["horizon"] + 1)]
            suffix = actions[start:start + row["horizon"] * row["stride"]]
            if len(suffix) != row["horizon"] * row["stride"]:
                raise ValueError("Incomplete recorded action suffix")
            buffers.append((row, start, visual[frames], proprio[frames],
                            suffix.reshape(row["horizon"], row["stride"], -1)))
            if len(buffers) == batch_size:
                yield collate(buffers)
                buffers = []
    if buffers:
        yield collate(buffers)


def collate(items):
    meta = [{"trajectory_id": x[0]["trajectory_id"], "task": x[0]["task"], "start": x[1]} for x in items]
    return meta, torch.stack([x[2] for x in items]), torch.stack([x[3] for x in items]), torch.stack([x[4] for x in items])


@torch.inference_mode()
def execute(args, backend, rows):
    device = backend.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timings = dict(load_seconds=0., encode_seconds=0., rollout_seconds=0.,
                   metrics_seconds=0., cache_write_seconds=0., warmup_seconds=0.)
    measurements = []
    iterator = iter(batches(rows, args.batch_size, backend, args.data_root))
    identity, batch_id = None, 0
    started = time.perf_counter()
    for_batch_start = started
    while True:
        try:
            meta, visual, proprio, raw_actions = next(iterator)
        except StopIteration:
            break
        timings["load_seconds"] += time.perf_counter() - for_batch_start
        if batch_id == 0:
            start = time.perf_counter()
            encoded = backend.encode(visual, proprio)
            actions = backend.normalize_actions(raw_actions)
            context = backend.context(encoded)
            reference = backend.predict(context, actions)
            zero = backend.predict(context, actions, instrument=True)
            identity = all(torch.equal(reference[k], zero[k]) for k in ("visual", "proprio"))
            if not identity:
                raise RuntimeError("Zero-dose instrumentation changed predictions")
            for _ in range(args.warmup):
                backend.predict(context, actions)
            synchronize(device)
            timings["warmup_seconds"] = time.perf_counter() - start
            del encoded, context, actions, reference, zero
        encoded, seconds = elapsed_call(lambda: backend.encode(visual, proprio), device)
        timings["encode_seconds"] += seconds
        # Include normalization/context preparation in measured rollout time.
        predicted, seconds = elapsed_call(lambda: backend.predict(backend.context(encoded), backend.normalize_actions(raw_actions)), device)
        timings["rollout_seconds"] += seconds
        start = time.perf_counter()
        values = score_predictions(predicted, encoded)
        measurements.extend({**m, "metrics": value} for m, value in zip(meta, values))
        timings["metrics_seconds"] += time.perf_counter() - start
        if args.write_cache:
            start = time.perf_counter()
            torch.save({"windows": meta, "encoded": {k: encoded[k].cpu() for k in ("visual", "proprio")}},
                       args.output / f"encoded-{batch_id:05d}.pt")
            timings["cache_write_seconds"] += time.perf_counter() - start
        batch_id += 1
        del encoded, predicted
        for_batch_start = time.perf_counter()
    measured = time.perf_counter() - started - timings["warmup_seconds"]
    if not measurements:
        raise ValueError("No eligible windows; inspect short trajectories or sharding")
    write_json(args.output / "window_metrics.json", measurements)
    actual_units = len({r["trajectory_id"] for r in measurements})
    report = dict(
        status="synthetic_smoke_only" if backend.synthetic else "real_baseline_benchmark_complete",
        gpu_benchmark_valid=not backend.synthetic and device.type == "cuda",
        efficacy_claim=False, backend=backend.provenance, independent_trajectories=actual_units,
        windows=len(measurements), batches=batch_id, horizon=6, initial_observed_frames=1,
        frame_stride=5, actual_future_frames_fed_to_predictor=False,
        zero_dose_identity=identity, measured_pipeline_seconds=measured, timing=timings,
        windows_per_second=len(measurements) / measured,
        rollout_windows_per_second=len(measurements) / timings["rollout_seconds"],
        peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
        environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                         device=str(device), cuda=torch.version.cuda,
                         gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                         cpu_threads=torch.get_num_threads()),
        shard_index=args.shard_index, num_shards=args.num_shards,
        task_trajectory_counts=dict(Counter(r["task"] for r in rows)),
        task_measured_trajectory_counts=dict(Counter(r["task"] for r in summarize_metrics(measurements)["per_trajectory"])),
        aggregation=summarize_metrics(measurements),
        limitations=["Unedited baseline only; ablation throughput is not measured",
                     "Startup/model loading are recorded separately; downloads are excluded",
                     "Cache timing covers encoded tensors only when --write-cache is enabled",
                     "Synthetic timings must not be extrapolated to JEPA-WM or GPUs"])
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["jepa", "toy"], default="jepa")
    p.add_argument("--vendor", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--checkpoint-sha256")
    p.add_argument("--manifest", type=Path)
    p.add_argument("--exposure-registry", type=Path,
                   help="Reviewed source-ID registry declaring which trajectories may be used for development")
    p.add_argument("--data-root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tasks", nargs="+", default=["all"])
    p.add_argument("--split", choices=["fit", "development"], default="development",
                   help="Timing/development only; protected holdout is deliberately not exposed")
    p.add_argument("--max-trajectories", type=int, default=12, help="0 means all in chosen split/tasks")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--write-cache", action="store_true")
    args = p.parse_args()
    if args.batch_size < 1 or args.max_trajectories < 0 or args.warmup < 0 or not 0 <= args.shard_index < args.num_shards:
        p.error("Invalid batch, trajectory, warmup, or shard parameters")
    if args.backend == "jepa" and any(getattr(args, key) is None for key in
                                    ("vendor", "checkpoint", "checkpoint_sha256", "manifest", "data_root", "exposure_registry")):
        p.error("Real JEPA requires vendor, checkpoint+checksum, manifest, data-root, and exposure-registry")
    if args.backend == "toy" and args.device != "cpu":
        p.error("Toy smoke fixture runs on CPU only")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        rows = toy_rows(12) if args.backend == "toy" else [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
        validate_manifest(rows)
        if any(r["horizon"] != 6 or r["stride"] != 5 for r in rows):
            raise ValueError("Current benchmark requires H6 and stride5")
        selected = select_rows(rows, args.tasks, args.split, args.max_trajectories, args.shard_index, args.num_shards)
        if not selected:
            raise ValueError("Empty trajectory selection/shard")
        if args.backend == "jepa" and (len({r["dataset"] for r in selected}) != 1 or selected[0]["dataset"] == "toy"):
            raise ValueError("One real dataset/checkpoint per process")
        if args.backend == "jepa":
            registry = json.loads(args.exposure_registry.read_text())
            if registry.get("manifest_sha256") != sha256(args.manifest) or not registry.get("review_basis"):
                raise ValueError("Exposure registry must identify this manifest and its historical review basis")
            for row in selected:
                entry = registry.get("trajectories", {}).get(row["trajectory_id"], {})
                if entry.get("use") != "development" or not entry.get("evidence"):
                    raise ValueError(f"Unreviewed/protected source cannot enter benchmark: {row['trajectory_id']}")
        write_json(args.output / "selection.json", selected)
        write_json(args.output / "config.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
        start = time.perf_counter()
        backend = ToyBackend("cpu") if args.backend == "toy" else JepaBackend(
            args.vendor, args.checkpoint, args.checkpoint_sha256, selected[0]["dataset"], args.device)
        setup_seconds = time.perf_counter() - start
        report = execute(args, backend, selected)
        report.update(model_setup_seconds=setup_seconds,
                      manifest_sha256=None if args.backend == "toy" else sha256(args.manifest),
                      exposure_registry_sha256=None if args.backend == "toy" else sha256(args.exposure_registry),
                      selection_sha256=sha256(args.output / "selection.json"),
                      config_sha256=sha256(args.output / "config.json"))
        code_hash = hashlib.sha256()
        for source in sorted(Path(__file__).parent.glob("*.py")):
            code_hash.update(source.name.encode() + b"\0" + source.read_bytes())
        report["harness_source_sha256"] = code_hash.hexdigest()
        write_json(args.output / "report.json", report)
        write_json(args.output / "DONE.json", {"status": report["status"], "report_sha256": sha256(args.output / "report.json")})
        print(json.dumps({k: report[k] for k in ("status", "gpu_benchmark_valid", "windows", "independent_trajectories", "measured_pipeline_seconds")}))
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"status": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
