#!/usr/bin/env python3
"""Choose 48 independent discovery snapshots nearest the Reach-Wall obstacle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from protocol import file_sha256, write_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected_episodes = set(range(0, 48, 3))
    rows = []
    for directory in args.capture_dirs:
        done = json.loads((directory / "DONE.json").read_text())
        for item in done["outputs"]:
            if int(item["episode"]) not in selected_episodes:
                continue
            path = directory / item["path"]
            if file_sha256(path) != item["sha256"]:
                raise RuntimeError(f"Hash mismatch {path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload["meta"]["split"] != "discovery":
                raise RuntimeError(f"Non-discovery episode selected: {path}")
            labels = payload["labels"]
            wall_distance = labels["wall_signed_distance"][:19].float().numpy()
            step = int(np.argmin(np.abs(wall_distance)))
            rows.append(
                {
                    "task": "mw-reach-wall",
                    "seed": int(payload["meta"]["seed"]),
                    "episode": int(payload["meta"]["episode"]),
                    "row": int(payload["meta"]["row"]),
                    "parquet": payload["meta"]["parquet"],
                    "parquet_sha256": payload["meta"]["parquet_sha256"],
                    "model_step": step,
                    "frame_index": int(labels["frame_index"][step]),
                    "wall_signed_distance": float(labels["wall_signed_distance"][step]),
                    "goal_distance": float(labels["goal_distance"][step]),
                    "progress": float(labels["progress"][step]),
                    "straight_path_intersects_wall": bool(labels["straight_path_intersects_wall"][step]),
                    "expert_detour_gate": bool(labels["expert_detour_gate"][step]),
                }
            )
    rows.sort(key=lambda row: (row["seed"], row["episode"]))
    counts = {seed: sum(row["seed"] == seed for row in rows) for seed in (1, 2, 3)}
    if len(rows) != 48 or counts != {1: 16, 2: 16, 3: 16}:
        raise RuntimeError(f"Snapshot balance failure: n={len(rows)}, counts={counts}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    receipt = {
        "complete": True,
        "task": "mw-reach-wall",
        "scope": "discovery only",
        "selection": "episodes 0,3,...,45 per seed; transition minimizing absolute TCP-to-wall signed distance",
        "count": len(rows),
        "seed_counts": counts,
        "manifest_sha256": file_sha256(args.output),
    }
    write_json_atomic(args.output.with_name("SNAPSHOT_MANIFEST_DONE.json"), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
