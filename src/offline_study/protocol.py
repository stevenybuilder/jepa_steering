"""Deterministic trajectory membership, window selection, and aggregation."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    """Replace only within an exclusively created run directory."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def window_starts(length: int, horizon: int = 6, stride: int = 5, count: int = 4) -> list[int]:
    if min(horizon, stride, count) <= 0:
        raise ValueError("Horizon, stride, and window count must be positive")
    last = length - 1 - horizon * stride
    if last < 0:
        return []
    if count == 1:
        return [0]
    return sorted({round(i * last / (count - 1)) for i in range(count)})


def study_split(ids: list[str], seed: int = 234) -> dict[str, str]:
    """Exact study 90/10; reserve 10% of the outer fitting pool for development.

    This hash ordering is deliberately named as OUR split, not the authors'
    torch.randperm split. It is stable across Python/Torch versions and input order.
    """
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate trajectory IDs")
    order = sorted(ids, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).digest())
    outer_fit = int(0.9 * len(order))
    inner_fit = int(0.9 * outer_fit)
    return {key: ("fit" if i < inner_fit else "development" if i < outer_fit else "holdout")
            for i, key in enumerate(order)}


def shard_for(trajectory_id: str, shards: int) -> int:
    if shards < 1:
        raise ValueError("Shard count must be positive")
    return int.from_bytes(hashlib.sha256(trajectory_id.encode()).digest()[:8], "big") % shards


def validate_manifest(rows: list[dict]) -> None:
    seen = set()
    for row in rows:
        key = row["trajectory_id"]
        if key in seen:
            raise ValueError(f"Duplicate trajectory ID: {key}")
        seen.add(key)
        if row["split"] not in {"fit", "development", "holdout", "external_reserve"}:
            raise ValueError("Unknown split")
        if row["dataset"] not in {"metaworld", "pusht", "toy"}:
            raise ValueError("Unknown dataset")
        if not row["task"] or row["index"] < 0 or row["length"] < 1:
            raise ValueError("Invalid trajectory metadata")
        expected = window_starts(row["length"], row["horizon"], row["stride"], row["windows_requested"])
        if row["starts"] != expected:
            raise ValueError(f"Unregistered window selection: {key}")


def summarize_metrics(window_rows: list[dict]) -> dict:
    """Equal trajectory weighting, never count overlapping windows as independent."""
    by_trajectory = defaultdict(list)
    seen = set()
    for row in window_rows:
        key = (row["trajectory_id"], row["start"])
        if key in seen:
            raise ValueError(f"Duplicate measured window: {key}")
        seen.add(key)
        by_trajectory[(row["task"], row["trajectory_id"])].append(row["metrics"])
    per_trajectory, by_task = [], defaultdict(list)
    for (task, trajectory_id), values in sorted(by_trajectory.items()):
        names = set(values[0])
        if any(set(v) != names for v in values):
            raise ValueError("Inconsistent metric keys")
        means = {k: sum(v[k] for v in values) / len(values) for k in sorted(names)}
        per_trajectory.append(dict(task=task, trajectory_id=trajectory_id, windows=len(values), metrics=means))
        by_task[task].append(means)
    task_metrics = {
        task: {"trajectories": len(values), "metrics": {
            k: sum(v[k] for v in values) / len(values) for k in values[0]}}
        for task, values in sorted(by_task.items())
    }
    return {"per_trajectory": per_trajectory, "per_task": task_metrics,
            "independent_units": "trajectory IDs; replay-source family independence remains unverified"}
