#!/usr/bin/env python3
"""Discovery-only linear and geometric screen of pooled JEPA-WM activations.

No validation or confirmation episode is opened by this script.  Predictive scores
use leave-one-collection-seed-out folds, so correlated timesteps from one trajectory
never cross the train/test boundary through their collection seed.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.utils.extmath import randomized_svd

from protocol import file_sha256, split_for_episode, write_json_atomic


RIDGE_ALPHA = 100.0


TARGETS: list[tuple[str, str, int | None]] = [
    ("goal_distance", "regression", None),
    ("progress", "regression", None),
    ("hand_x", "regression", 0),
    ("hand_y", "regression", 1),
    ("hand_z", "regression", 2),
    ("goal_vector_x", "regression", 0),
    ("goal_vector_y", "regression", 1),
    ("goal_vector_z", "regression", 2),
    ("wall_signed_distance", "regression", None),
    ("height_above_wall_top", "regression", None),
    ("straight_path_intersects_wall", "binary", None),
    ("expert_detour_gate", "binary", None),
    ("action_dx", "regression", 0),
    ("action_dy", "regression", 1),
    ("action_dz", "regression", 2),
    ("action_translation_magnitude", "regression", None),
    ("realized_dx", "regression", 0),
    ("realized_dy", "regression", 1),
    ("realized_dz", "regression", 2),
    ("realized_hand_delta_magnitude", "regression", None),
    ("reward_sum", "regression", None),
]


def target_array(labels: dict[str, torch.Tensor], name: str, component: int | None) -> np.ndarray:
    source = {
        "hand_x": "hand_xyz",
        "hand_y": "hand_xyz",
        "hand_z": "hand_xyz",
        "goal_vector_x": "goal_vector",
        "goal_vector_y": "goal_vector",
        "goal_vector_z": "goal_vector",
        "action_dx": "action_translation_sum",
        "action_dy": "action_translation_sum",
        "action_dz": "action_translation_sum",
        "realized_dx": "realized_hand_delta",
        "realized_dy": "realized_hand_delta",
        "realized_dz": "realized_hand_delta",
    }.get(name, name)
    value = labels[source].cpu().numpy()
    if len(value) == 20:
        value = value[:19]
    if component is not None:
        value = value[:, component]
    return value.astype(np.float64)


def load_discovery(capture_dirs: list[Path]) -> dict[str, Any]:
    families: dict[str, list[np.ndarray]] = {"encoder": [], "predictor": []}
    target_rows: list[np.ndarray] = []
    seeds: list[np.ndarray] = []
    episodes: list[np.ndarray] = []
    times: list[np.ndarray] = []
    sources = []
    for directory in capture_dirs:
        done_path = directory / "DONE.json"
        done = json.loads(done_path.read_text())
        sources.append({"done": str(done_path), "sha256": file_sha256(done_path)})
        for item in done["outputs"]:
            # Enforce the split before opening tensors, not after loading held-outs.
            if split_for_episode(int(item["episode"])) != "discovery":
                continue
            path = directory / item["path"]
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload["meta"]["split"] != "discovery":
                continue
            if file_sha256(path) != item["sha256"]:
                raise RuntimeError(f"Hash mismatch {path}")
            features = payload["features"]
            families["encoder"].append(features["encoder"][:19].float().numpy())
            families["predictor"].append(features["predictor"].float().numpy())
            target_rows.append(
                np.stack([target_array(payload["labels"], name, component) for name, _kind, component in TARGETS], axis=1)
            )
            seeds.append(np.full(19, int(payload["meta"]["seed"]), dtype=np.int64))
            episodes.append(np.full(19, int(payload["meta"]["episode"]), dtype=np.int64))
            times.append(payload["labels"]["time_fraction"][:19].float().numpy())
    return {
        "features": {name: np.concatenate(values, axis=0) for name, values in families.items()},
        "targets": np.concatenate(target_rows, axis=0),
        "seeds": np.concatenate(seeds),
        "episodes": np.concatenate(episodes),
        "time": np.concatenate(times).astype(np.float64),
        "sources": sources,
    }


def metric(kind: str, truth: np.ndarray, prediction: np.ndarray) -> float:
    if kind == "binary":
        if np.unique(truth).size < 2:
            return float("nan")
        return float(roc_auc_score(truth, prediction))
    return float(r2_score(truth, prediction))


def grouped_predictions(x: np.ndarray, y: np.ndarray, seeds: np.ndarray, alpha: float) -> tuple[np.ndarray, list[dict[str, Any]]]:
    predictions = np.full_like(y, np.nan, dtype=np.float64)
    folds = []
    for held_seed in (1, 2, 3):
        train = seeds != held_seed
        test = seeds == held_seed
        mean = x[train].mean(axis=0, keepdims=True)
        scale = x[train].std(axis=0, keepdims=True)
        scale[scale < 1e-6] = 1.0
        x_train = (x[train] - mean) / scale
        x_test = (x[test] - mean) / scale
        model = Ridge(alpha=alpha, solver="cholesky")
        model.fit(x_train, y[train])
        predictions[test] = model.predict(x_test)
        folds.append({"held_seed": held_seed, "train_n": int(train.sum()), "test_n": int(test.sum())})
    return predictions, folds


def geometry_summary(x: np.ndarray, seeds: np.ndarray, episodes: np.ndarray) -> dict[str, float]:
    centered = x.astype(np.float64) - x.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    trace = float(np.trace(covariance))
    participation = trace * trace / max(float(np.square(covariance).sum()), 1e-12)
    _u, singular, _vt = randomized_svd(centered, n_components=min(32, x.shape[1] - 1), random_state=0)
    variance = np.square(singular)
    top10 = float(variance[:10].sum() / max(trace * (len(centered) - 1), 1e-12))
    adjacent = []
    for seed in (1, 2, 3):
        for episode in range(50):
            values = x[(seeds == seed) & (episodes == episode)]
            a, b = values[:-1], values[1:]
            denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
            adjacent.extend((np.sum(a * b, axis=1) / np.maximum(denominator, 1e-12)).tolist())
    return {
        "participation_ratio": participation,
        "top10_variance_fraction": top10,
        "adjacent_timestep_cosine_mean": float(np.mean(adjacent)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ridge-alpha", type=float, default=RIDGE_ALPHA)
    args = parser.parse_args()
    data = load_discovery(args.capture_dirs)
    y = data["targets"]
    seeds = data["seeds"]
    episodes = data["episodes"]
    time_features = np.stack([data["time"], np.square(data["time"]), np.power(data["time"], 3)], axis=1)
    time_prediction, folds = grouped_predictions(time_features, y, seeds, alpha=1.0)
    baselines = {
        name: metric(kind, y[:, index], time_prediction[:, index])
        for index, (name, kind, _component) in enumerate(TARGETS)
    }

    rows = []
    for family, values in data["features"].items():
        layer_count = values.shape[1]
        for layer in range(layer_count):
            x = values[:, layer].astype(np.float64)
            prediction, _ = grouped_predictions(x, y, seeds, alpha=args.ridge_alpha)
            geometry = geometry_summary(x, seeds, episodes)
            for index, (name, kind, _component) in enumerate(TARGETS):
                score = metric(kind, y[:, index], prediction[:, index])
                rows.append(
                    {
                        "family": family,
                        "layer": layer,
                        "target": name,
                        "metric": "auroc" if kind == "binary" else "r2",
                        "score": score,
                        "time_baseline": baselines[name],
                        "delta_vs_time": score - baselines[name],
                        **geometry,
                    }
                )

    shortlist: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for name, _kind, _component in TARGETS:
        shortlist[name] = {}
        for family, limit in (("encoder", 2), ("predictor", 3)):
            candidates = [row for row in rows if row["target"] == name and row["family"] == family]
            candidates.sort(key=lambda row: row["delta_vs_time"], reverse=True)
            shortlist[name][family] = candidates[:limit]

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "scope": "discovery only: episodes 0-49 within collection seeds 1,2,3",
        "sample_count": int(len(y)),
        "episode_count": int(len(set(zip(seeds.tolist(), episodes.tolist())))),
        "ridge_alpha": args.ridge_alpha,
        "cross_validation": "leave one collection seed out; whole episodes stay within seed",
        "time_baseline": "cubic time polynomial with identical held-seed folds",
        "folds": folds,
        "sources": data["sources"],
        "baselines": baselines,
        "rows": rows,
        "shortlist": shortlist,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "discovery_screen.json"
    write_json_atomic(output, report)
    summary_lines = [
        "# Reach-Wall discovery screen",
        "",
        f"Samples: {report['sample_count']} transitions from {report['episode_count']} whole trajectories.",
        "Validation and confirmation episodes were not loaded.",
        "",
        "## Strongest layer per target",
        "",
        "| Target | Encoder | Score / over time | Predictor | Score / over time |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, _kind, _component in TARGETS:
        enc = shortlist[name]["encoder"][0]
        pred = shortlist[name]["predictor"][0]
        summary_lines.append(
            f"| {name} | {enc['layer']} | {enc['score']:.3f} / {enc['delta_vs_time']:+.3f} | "
            f"{pred['layer']} | {pred['score']:.3f} / {pred['delta_vs_time']:+.3f} |"
        )
    (args.output_dir / "discovery_screen.md").write_text("\n".join(summary_lines) + "\n")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
