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
from .pipeline import AsyncCacheWriter, PrefetchIterator
from .protocol import (
    sha256,
    shard_for,
    summarize_metrics,
    validate_manifest,
    window_starts,
    write_json,
)
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
    names, columns = [], []
    for modality in ("visual", "proprio"):
        for h in horizons:
            delta = predicted[modality][h].float() - target[modality][:, h].float()
            names.append(f"{modality}_mse_h{h}")
            columns.append(delta.square().flatten(1).mean(1))
    # Transfer once; six separate .cpu().tolist() calls force six CUDA synchronizations.
    values = torch.stack(columns, dim=1).cpu().tolist()
    return [dict(zip(names, row, strict=True)) for row in values]


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


def filter_reviewed_development(rows, registry, manifest_sha256):
    """Remove anything not explicitly cleared for development before selection/sharding."""
    if registry.get("manifest_sha256") != manifest_sha256 or not registry.get("review_basis"):
        raise ValueError("Exposure registry must identify this manifest and its historical review basis")
    cleared = []
    for row in rows:
        entry = registry.get("trajectories", {}).get(row["trajectory_id"], {})
        if entry.get("use") == "development" and entry.get("evidence"):
            cleared.append(row)
    return cleared


def toy_rows(count):
    return [dict(trajectory_id=f"toy:{i}", dataset="toy", source_pool="fixture", index=i,
                 task=("mw-reach", "mw-reach-wall", "pusht")[i % 3], split="development",
                 length=61, horizon=6, stride=5, windows_requested=4,
                 starts=window_starts(61), exposure_status="synthetic") for i in range(count)]


def batches(rows, batch_size, backend, data_root, pin_memory=False):
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
                yield collate(buffers, pin_memory=pin_memory)
                buffers = []
    if buffers:
        yield collate(buffers, pin_memory=pin_memory)


def collate(items, pin_memory=False):
    meta = [{"trajectory_id": x[0]["trajectory_id"], "task": x[0]["task"], "start": x[1]} for x in items]
    tensors = tuple(torch.stack([x[i] for x in items]) for i in (2, 3, 4))
    if pin_memory:
        tensors = tuple(value.pin_memory() for value in tensors)
    return (meta, *tensors)


def _rate(units, seconds):
    return units / seconds if seconds > 0 else None


def _cuda_environment(device):
    if device.type != "cuda":
        return {}
    properties = torch.cuda.get_device_properties(device)
    with torch.cuda.device(device):
        bf16_supported = torch.cuda.is_bf16_supported()
    return {
        "gpu": torch.cuda.get_device_name(device),
        "gpu_capability": list(torch.cuda.get_device_capability(device)),
        "gpu_total_memory_bytes": properties.total_memory,
        "visible_cuda_devices": torch.cuda.device_count(),
        "bf16_supported": bf16_supported,
        "tf32_matmul_allowed": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn_allowed": torch.backends.cudnn.allow_tf32,
    }


@torch.inference_mode()
def execute(args, backend, rows):
    device = backend.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timings = dict(load_seconds=0., encode_seconds=0., rollout_seconds=0.,
                   metrics_seconds=0., cache_stage_seconds=0., cache_write_seconds=0.,
                   cache_wait_seconds=0., data_decode_seconds=0., warmup_seconds=0.)
    measurements = []
    started = time.perf_counter()
    iterator = PrefetchIterator(
        batches(rows, args.batch_size, backend, args.data_root, pin_memory=device.type == "cuda"),
        args.prefetch_batches,
    )
    cache_writer = AsyncCacheWriter(args.cache_queue_depth) if args.write_cache else None
    identity, batch_id = None, 0
    try:
        while True:
            try:
                meta, visual, proprio, raw_actions = next(iterator)
            except StopIteration:
                break
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
            encoded, seconds = elapsed_call(
                lambda visual=visual, proprio=proprio: backend.encode(visual, proprio), device)
            timings["encode_seconds"] += seconds
            # Include normalization/context preparation in measured rollout time.
            predicted, seconds = elapsed_call(
                lambda encoded=encoded, raw_actions=raw_actions: backend.predict(
                    backend.context(encoded), backend.normalize_actions(raw_actions)),
                device,
            )
            timings["rollout_seconds"] += seconds
            start = time.perf_counter()
            values = score_predictions(predicted, encoded)
            measurements.extend(
                {**meta_row, "metrics": value}
                for meta_row, value in zip(meta, values, strict=True)
            )
            timings["metrics_seconds"] += time.perf_counter() - start
            if cache_writer is not None:
                start = time.perf_counter()
                payload = {
                    "windows": meta,
                    "encoded": {k: encoded[k].detach().cpu() for k in ("visual", "proprio")},
                }
                timings["cache_stage_seconds"] += time.perf_counter() - start
                cache_writer.submit(args.output / f"encoded-{batch_id:05d}.pt", payload)
            batch_id += 1
            del encoded, predicted
    finally:
        iterator.close()
        timings["load_seconds"] = iterator.consumer_wait_seconds
        timings["data_decode_seconds"] = iterator.producer_seconds
        if cache_writer is not None:
            cache_writer.close()
            timings["cache_write_seconds"] = cache_writer.writer_seconds
            timings["cache_wait_seconds"] = cache_writer.consumer_wait_seconds
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
        windows_per_second=_rate(len(measurements), measured),
        encoder_frames_per_second=_rate(7 * len(measurements), timings["encode_seconds"]),
        rollout_windows_per_second=_rate(len(measurements), timings["rollout_seconds"]),
        peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
        environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                         device=str(device), cuda=torch.version.cuda, cpu_threads=torch.get_num_threads(),
                         **_cuda_environment(device)),
        execution=dict(
            precision=args.precision,
            allow_tf32=args.allow_tf32,
            batch_size=args.batch_size,
            encoder_frames_per_batch=7 * args.batch_size,
            pinned_host_batches=device.type == "cuda",
            prefetch_batches=args.prefetch_batches,
            asynchronous_cache_writer=cache_writer is not None,
            cache_queue_depth=args.cache_queue_depth if cache_writer is not None else 0,
            parallelism="independent_trajectory_process_sharding; no model collectives",
            precision_parity_status=("reference_float32" if args.precision == "float32" and not args.allow_tf32
                                     else "candidate_requires_separate_float32_parity_comparison"),
        ),
        shard_index=args.shard_index, num_shards=args.num_shards,
        task_trajectory_counts=dict(Counter(r["task"] for r in rows)),
        task_measured_trajectory_counts=dict(Counter(r["task"] for r in summarize_metrics(measurements)["per_trajectory"])),
        aggregation=summarize_metrics(measurements),
        limitations=["Unedited baseline only; ablation throughput is not measured",
                     "Startup/model loading are recorded separately; downloads are excluded",
                     "Cache timing covers encoded tensors only when --write-cache is enabled",
                     "Reduced precision or TF32 requires a separately reviewed float32 parity comparison",
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
    p.add_argument("--precision", choices=["float32", "bfloat16", "float16"], default="float32")
    p.add_argument("--allow-tf32", action="store_true",
                   help="Opt-in CUDA TensorFloat-32 matmuls; treated as a separate parity candidate")
    p.add_argument("--prefetch-batches", type=int, default=2,
                   help="Bounded producer queue; 0 disables decode/data prefetch")
    p.add_argument("--cache-queue-depth", type=int, default=2,
                   help="Maximum queued cache writes when --write-cache is set")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--write-cache", action="store_true")
    args = p.parse_args()
    if (args.batch_size < 1 or args.max_trajectories < 0 or args.warmup < 0 or
            args.prefetch_batches < 0 or args.cache_queue_depth < 1 or
            not 0 <= args.shard_index < args.num_shards):
        p.error("Invalid batch, trajectory, warmup, pipeline, or shard parameters")
    if args.backend == "jepa" and any(getattr(args, key) is None for key in
                                    ("vendor", "checkpoint", "checkpoint_sha256", "manifest", "data_root", "exposure_registry")):
        p.error("Real JEPA requires vendor, checkpoint+checksum, manifest, data-root, and exposure-registry")
    if args.backend == "toy" and args.device != "cpu":
        p.error("Toy smoke fixture runs on CPU only")
    if args.backend == "toy" and (args.precision != "float32" or args.allow_tf32):
        p.error("Toy smoke fixture supports only strict float32")
    if args.allow_tf32 and args.precision != "float32":
        p.error("--allow-tf32 is valid only with --precision float32")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        rows = toy_rows(12) if args.backend == "toy" else [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
        validate_manifest(rows)
        if any(r["horizon"] != 6 or r["stride"] != 5 for r in rows):
            raise ValueError("Current benchmark requires H6 and stride5")
        if args.backend == "jepa":
            registry = json.loads(args.exposure_registry.read_text())
            rows = filter_reviewed_development(rows, registry, sha256(args.manifest))
        selected = select_rows(rows, args.tasks, args.split, args.max_trajectories, args.shard_index, args.num_shards)
        if not selected:
            raise ValueError("Empty trajectory selection/shard")
        if args.backend == "jepa" and (len({r["dataset"] for r in selected}) != 1 or selected[0]["dataset"] == "toy"):
            raise ValueError("One real dataset/checkpoint per process")
        write_json(args.output / "selection.json", selected)
        write_json(args.output / "config.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
        start = time.perf_counter()
        backend = ToyBackend("cpu", args.precision, args.allow_tf32) if args.backend == "toy" else JepaBackend(
            args.vendor, args.checkpoint, args.checkpoint_sha256, selected[0]["dataset"], args.device,
            args.precision, args.allow_tf32)
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
        report["cache_files"] = [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(args.output.glob("encoded-*.pt"))
        ]
        write_json(args.output / "report.json", report)
        write_json(args.output / "DONE.json", {
            "status": report["status"],
            "report_sha256": sha256(args.output / "report.json"),
            "window_metrics_sha256": sha256(args.output / "window_metrics.json"),
            "selection_sha256": sha256(args.output / "selection.json"),
            "config_sha256": sha256(args.output / "config.json"),
        })
        print(json.dumps({k: report[k] for k in ("status", "gpu_benchmark_valid", "windows", "independent_trajectories", "measured_pipeline_seconds")}))
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"status": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
