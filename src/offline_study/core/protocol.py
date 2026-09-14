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
    lineage_splits = {}
    lineage_tasks = {}
    for row in rows:
        key = row["trajectory_id"]
        if key in seen:
            raise ValueError(f"Duplicate trajectory ID: {key}")
        seen.add(key)
        if row["split"] not in {"fit", "development", "holdout", "external_reserve"}:
            raise ValueError("Unknown split")
        if row["dataset"] not in {"metaworld", "pusht", "toy"}:
            raise ValueError("Unknown dataset")
        if not row["task"] or not row.get("lineage_group") or row["index"] < 0 or row["length"] < 1:
            raise ValueError("Invalid trajectory metadata")
        lineage_key = (row["dataset"], row["lineage_group"])
        prior_split = lineage_splits.setdefault(lineage_key, row["split"])
        if prior_split != row["split"]:
            raise ValueError(
                f"Lineage group crosses study splits: {row['lineage_group']}"
            )
        prior_task = lineage_tasks.setdefault(lineage_key, row["task"])
        if prior_task != row["task"]:
            raise ValueError(
                f"Lineage group crosses tasks: {row['lineage_group']}"
            )
        expected = window_starts(row["length"], row["horizon"], row["stride"], row["windows_requested"])
        if row["starts"] != expected:
            raise ValueError(f"Unregistered window selection: {key}")


def summarize_metrics(window_rows: list[dict]) -> dict:
    """Aggregate windows within rollouts and rollouts within lineage groups."""
    by_trajectory = defaultdict(list)
    trajectory_groups = {}
    seen = set()
    for row in window_rows:
        key = (row["trajectory_id"], row["start"])
        if key in seen:
            raise ValueError(f"Duplicate measured window: {key}")
        seen.add(key)
        trajectory_key = (row["task"], row["trajectory_id"])
        lineage_group = row.get("lineage_group")
        if not lineage_group:
            raise ValueError(f"Measured window lacks lineage group: {row['trajectory_id']}")
        if trajectory_key in trajectory_groups and trajectory_groups[trajectory_key] != lineage_group:
            raise ValueError(f"Trajectory has inconsistent lineage groups: {row['trajectory_id']}")
        trajectory_groups[trajectory_key] = lineage_group
        by_trajectory[trajectory_key].append(row["metrics"])
    per_trajectory, by_task, by_group = [], defaultdict(list), defaultdict(list)
    for (task, trajectory_id), values in sorted(by_trajectory.items()):
        names = set(values[0])
        if any(set(v) != names for v in values):
            raise ValueError("Inconsistent metric keys")
        means = {k: sum(v[k] for v in values) / len(values) for k in sorted(names)}
        lineage_group = trajectory_groups[(task, trajectory_id)]
        per_trajectory.append(dict(task=task, trajectory_id=trajectory_id, lineage_group=lineage_group,
                                   windows=len(values), metrics=means))
        by_task[task].append(means)
        by_group[(task, lineage_group)].append(means)
    task_metrics = {
        task: {"trajectories": len(values), "metrics": {
            k: sum(v[k] for v in values) / len(values) for k in values[0]}}
        for task, values in sorted(by_task.items())
    }
    per_lineage_group, task_group_values = [], defaultdict(list)
    for (task, lineage_group), values in sorted(by_group.items()):
        means = {k: sum(v[k] for v in values) / len(values) for k in values[0]}
        per_lineage_group.append({
            "task": task, "lineage_group": lineage_group,
            "trajectories": len(values), "metrics": means,
        })
        task_group_values[task].append(means)
    per_task_group_weighted = {
        task: {"lineage_groups": len(values), "metrics": {
            key: sum(value[key] for value in values) / len(values)
            for key in values[0]
        }}
        for task, values in sorted(task_group_values.items())
    }
    return {"per_trajectory": per_trajectory, "per_task": task_metrics,
            "per_lineage_group": per_lineage_group,
            "per_task_group_weighted": per_task_group_weighted,
            "independent_units": "lineage groups; windows aggregate within trajectory, trajectories aggregate within group"}
