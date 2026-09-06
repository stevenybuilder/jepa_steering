"""Frozen trajectory splits and physical-coordinate labels for Reach-Wall."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


WALL_CENTER = np.array([0.10, 0.75, 0.06], dtype=np.float32)
WALL_HALF_SIZE = np.array([0.12, 0.01, 0.06], dtype=np.float32)


def split_for_episode(episode: int) -> str:
    if 0 <= episode < 50:
        return "discovery"
    if episode < 75:
        return "validation"
    if episode < 100:
        return "confirmation"
    raise ValueError(f"Episode {episode} is outside the official 0..99 range")


def shard_for(seed: int, episode: int) -> int:
    if seed not in (1, 2, 3):
        raise ValueError(f"Unexpected collection seed {seed}")
    return (seed - 1) * 4 + episode // 25


def box_signed_distance(points: np.ndarray) -> np.ndarray:
    q = np.abs(points - WALL_CENTER) - WALL_HALF_SIZE
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.max(q, axis=-1), 0.0)
    return outside + inside


def segment_intersects_box(start: np.ndarray, end: np.ndarray, margin: float = 0.03) -> np.ndarray:
    lower = WALL_CENTER - WALL_HALF_SIZE - margin
    upper = WALL_CENTER + WALL_HALF_SIZE + margin
    direction = end - start
    result = np.zeros(start.shape[0], dtype=bool)
    for row, (origin, delta) in enumerate(zip(start, direction)):
        t_min, t_max = 0.0, 1.0
        valid = True
        for axis in range(3):
            if abs(float(delta[axis])) < 1e-12:
                valid = bool(lower[axis] <= origin[axis] <= upper[axis])
                if not valid:
                    break
            else:
                a = float((lower[axis] - origin[axis]) / delta[axis])
                b = float((upper[axis] - origin[axis]) / delta[axis])
                t_min = max(t_min, min(a, b))
                t_max = min(t_max, max(a, b))
                if t_min > t_max:
                    valid = False
                    break
        result[row] = valid
    return result


def reach_wall_labels(states: np.ndarray, actions: np.ndarray, rewards: np.ndarray) -> dict[str, np.ndarray]:
    frame_indices = np.arange(0, 100, 5, dtype=np.int64)
    sampled = states[frame_indices]
    hand = sampled[:, :3].astype(np.float32)
    goal = sampled[:, -3:].astype(np.float32)
    goal_vector = goal - hand
    goal_distance = np.linalg.norm(goal_vector, axis=-1).astype(np.float32)
    progress = (goal_distance[0] - goal_distance).astype(np.float32)
    expert_detour_gate = (
        (hand[:, 0] >= -0.10)
        & (hand[:, 0] <= 0.30)
        & (hand[:, 1] >= 0.60)
        & (hand[:, 1] <= 0.80)
        & (hand[:, 2] < 0.25)
    )

    transition_count = len(frame_indices) - 1
    chunks = actions[: transition_count * 5].reshape(transition_count, 5, 4).astype(np.float32)
    reward_chunks = rewards[: transition_count * 5].reshape(transition_count, 5).astype(np.float32)
    realized_delta = np.diff(hand, axis=0).astype(np.float32)
    return {
        "frame_index": frame_indices,
        "time_fraction": (frame_indices / frame_indices[-1]).astype(np.float32),
        "hand_xyz": hand,
        "goal_xyz": goal,
        "goal_vector": goal_vector.astype(np.float32),
        "goal_distance": goal_distance,
        "progress": progress,
        "success_now": (goal_distance <= 0.05),
        "wall_signed_distance": box_signed_distance(hand).astype(np.float32),
        "height_above_wall_top": (hand[:, 2] - (WALL_CENTER[2] + WALL_HALF_SIZE[2])).astype(np.float32),
        "straight_path_intersects_wall": segment_intersects_box(hand, goal),
        "expert_detour_gate": expert_detour_gate,
        "action_mean": chunks.mean(axis=1).astype(np.float32),
        "action_translation_sum": chunks[:, :, :3].sum(axis=1).astype(np.float32),
        "action_translation_magnitude": np.linalg.norm(chunks[:, :, :3], axis=(1, 2)).astype(np.float32),
        "realized_hand_delta": realized_delta,
        "realized_hand_delta_magnitude": np.linalg.norm(realized_delta, axis=-1).astype(np.float32),
        "reward_sum": reward_chunks.sum(axis=1).astype(np.float32),
        "reward_max": reward_chunks.max(axis=1).astype(np.float32),
    }


def file_sha256(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
