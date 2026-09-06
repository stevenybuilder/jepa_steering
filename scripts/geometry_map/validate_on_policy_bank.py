#!/usr/bin/env python3
"""Validate and summarize cached on-policy JEPA-WM episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from protocol import file_sha256, write_json_atomic


def check_episode(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    if file_sha256(path) != expected["sha256"]:
        raise RuntimeError(f"Hash mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    episode = int(payload["episode"])
    if episode != int(expected["episode"]):
        raise RuntimeError(f"Episode mismatch in {path}")
    if payload["schema_version"] != 1 or payload["arm"] != "unsteered_frozen_jepa_wm":
        raise RuntimeError(f"Unexpected schema or arm in {path}")
    replans = payload["replans"]
    if int(payload["replan_count"]) != len(replans) or not replans:
        raise RuntimeError(f"Invalid replan count in {path}")
    actions = payload["actions_raw"]
    if actions.ndim != 2 or actions.shape[1] != 4 or not torch.isfinite(actions).all():
        raise RuntimeError(f"Invalid action tensor in {path}")
    elapsed = []
    for index, replan in enumerate(replans):
        if int(replan["replan_index"]) != index:
            raise RuntimeError(f"Non-sequential replan index in {path}")
        if replan["observation_visual"].dtype != torch.uint8:
            raise RuntimeError(f"Visual observation is not uint8 in {path}")
        physics = replan["simulator"]["physics"]
        if "qpos" not in physics or "qvel" not in physics:
            raise RuntimeError(f"Missing restorable MuJoCo state in {path}")
        if not replan["planner_rng_before"].numel():
            raise RuntimeError(f"Missing planner RNG state in {path}")
        trace = replan["chosen_plan_trace"]
        if int(trace["block"]) != 3:
            raise RuntimeError(f"Unexpected causal block in {path}")
        if not torch.isfinite(trace["block_pooled"]).all():
            raise RuntimeError(f"Non-finite block trace in {path}")
        elapsed.append(int(replan["elapsed_steps"]))
    if elapsed != sorted(set(elapsed)):
        raise RuntimeError(f"Replan steps are not strictly increasing in {path}")
    return {
        "episode": episode,
        "path": str(path),
        "sha256": expected["sha256"],
        "environment_seed": int(payload["environment_seed"]),
        "planner_seed": int(payload["planner_seed"]),
        "final_success": bool(payload["final_success"]),
        "ever_success": bool(payload["ever_success"]),
        "total_reward": float(payload["total_reward"]),
        "final_state_distance_to_expert_goal": float(payload["final_state_distance_to_expert_goal"]),
        "environment_steps": int(payload["environment_steps"]),
        "replan_count": len(replans),
        "duration_seconds": float(payload["duration_seconds"]),
        "size_bytes": path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--expected-episodes", type=int, nargs="*")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = []
    receipts = []
    for directory in args.input_dirs:
        done_path = directory / "DONE.json"
        receipt = json.loads(done_path.read_text())
        if not receipt.get("complete"):
            raise RuntimeError(f"Incomplete receipt: {done_path}")
        receipts.append({"path": str(done_path), "sha256": file_sha256(done_path)})
        for expected in receipt["outputs"]:
            rows.append(check_episode(directory / expected["path"], expected))
    episodes = [row["episode"] for row in rows]
    if len(set(episodes)) != len(episodes):
        raise RuntimeError("Duplicate episode IDs across shards")
    if args.expected_episodes is not None and set(episodes) != set(args.expected_episodes):
        raise RuntimeError(
            f"Expected episodes {sorted(args.expected_episodes)}, found {sorted(episodes)}"
        )
    report = {
        "schema_version": 1,
        "complete": True,
        "episode_count": len(rows),
        "episodes": sorted(episodes),
        "final_success_count": sum(row["final_success"] for row in rows),
        "ever_success_count": sum(row["ever_success"] for row in rows),
        "final_success_rate": float(np.mean([row["final_success"] for row in rows])),
        "ever_success_rate": float(np.mean([row["ever_success"] for row in rows])),
        "mean_total_reward": float(np.mean([row["total_reward"] for row in rows])),
        "mean_final_state_distance": float(
            np.mean([row["final_state_distance_to_expert_goal"] for row in rows])
        ),
        "total_gpu_episode_seconds": float(sum(row["duration_seconds"] for row in rows)),
        "total_size_bytes": sum(row["size_bytes"] for row in rows),
        "receipts": receipts,
        "rows": sorted(rows, key=lambda row: row["episode"]),
    }
    if args.output:
        write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
