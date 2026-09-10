#!/usr/bin/env python3
"""Test task-geometry emergence and subspace dimensionality on discovery data.

This adapts only the transferable diagnostics from the Physics Emergence Zone work:
layerwise transition curves, circular direction targets, iterative orthogonal probes,
and subspace angles/overlap.  It does not assume a universal one-third-depth layer.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from protocol import write_json_atomic
from screen_pooled import TARGETS, grouped_predictions, load_discovery


RIDGE_ALPHA = 100.0


def target_index(name: str) -> int:
    return next(index for index, row in enumerate(TARGETS) if row[0] == name)


def circular_score(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    score = float(r2_score(truth, prediction, multioutput="variance_weighted"))
    true_angle = np.arctan2(truth[:, 1], truth[:, 0])
    pred_angle = np.arctan2(prediction[:, 1], prediction[:, 0])
    difference = np.arctan2(np.sin(pred_angle - true_angle), np.cos(pred_angle - true_angle))
    return {"r2": score, "circular_mae_degrees": float(np.degrees(np.abs(difference)).mean())}


def regression_score(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {"r2": float(r2_score(truth, prediction))}


def emergence_curve(
    features: np.ndarray,
    target: np.ndarray,
    seeds: np.ndarray,
    mask: np.ndarray,
    circular: bool,
    alpha: float,
) -> dict[str, Any]:
    rows = []
    for layer in range(features.shape[1]):
        prediction, _folds = grouped_predictions(
            features[mask, layer].astype(np.float64), target[mask], seeds[mask], alpha
        )
        scores = circular_score(target[mask], prediction) if circular else regression_score(target[mask], prediction)
        rows.append({"layer": layer, **scores})
    peak = max(row["r2"] for row in rows)
    near_peak = [row["layer"] for row in rows if row["r2"] >= peak - 0.05]
    deltas = [rows[index]["r2"] - rows[index - 1]["r2"] for index in range(1, len(rows))]
    return {
        "rows": rows,
        "peak_layer": max(rows, key=lambda row: row["r2"])["layer"],
        "peak_r2": peak,
        "near_peak_layers_within_0.05": near_peak,
        "largest_positive_jump_after_layer": int(np.argmax(deltas)) if deltas else None,
        "largest_positive_jump": float(max(deltas)) if deltas else None,
    }


def orthogonal_probe_sequence(
    x: np.ndarray,
    y: np.ndarray,
    seeds: np.ndarray,
    mask: np.ndarray,
    alpha: float,
    max_steps: int,
    circular: bool,
) -> list[dict[str, float]]:
    predictions = [np.full_like(y, np.nan, dtype=np.float64) for _ in range(max_steps)]
    for held_seed in (1, 2, 3):
        train = mask & (seeds != held_seed)
        test = mask & (seeds == held_seed)
        mean = x[train].mean(axis=0, keepdims=True)
        scale = x[train].std(axis=0, keepdims=True)
        scale[scale < 1e-6] = 1.0
        x_train = (x[train] - mean) / scale
        x_test = (x[test] - mean) / scale
        cumulative = np.empty((x.shape[1], 0), dtype=np.float64)
        for step in range(max_steps):
            model = Ridge(alpha=alpha, solver="cholesky").fit(x_train, y[train])
            predictions[step][test] = model.predict(x_test)
            weights = np.atleast_2d(model.coef_).T
            if cumulative.shape[1]:
                weights = weights - cumulative @ (cumulative.T @ weights)
            q, r = np.linalg.qr(weights)
            keep = np.abs(np.diag(r)) > 1e-9
            q = q[:, keep]
            if not q.shape[1]:
                break
            cumulative = np.linalg.qr(np.concatenate([cumulative, q], axis=1))[0]
            x_train = x_train - (x_train @ q) @ q.T
            x_test = x_test - (x_test @ q) @ q.T
    rows = []
    for step, prediction in enumerate(predictions):
        valid = mask & np.all(np.isfinite(prediction), axis=1) if prediction.ndim == 2 else mask & np.isfinite(prediction)
        if not valid.any():
            break
        score = circular_score(y[valid], prediction[valid]) if circular else regression_score(y[valid], prediction[valid])
        rows.append({"iteration": step, "directions_removed_before_fit": step * (2 if circular else 1), **score})
    return rows


def fit_full_basis(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    alpha: float,
    steps: int,
) -> np.ndarray:
    mean = x[mask].mean(axis=0, keepdims=True)
    scale = x[mask].std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    residual = (x[mask] - mean) / scale
    cumulative = np.empty((x.shape[1], 0), dtype=np.float64)
    for _step in range(steps):
        model = Ridge(alpha=alpha, solver="cholesky").fit(residual, y[mask])
        weights = np.atleast_2d(model.coef_).T
        if cumulative.shape[1]:
            weights = weights - cumulative @ (cumulative.T @ weights)
        q, r = np.linalg.qr(weights)
        q = q[:, np.abs(np.diag(r)) > 1e-9]
        if not q.shape[1]:
            break
        cumulative = np.linalg.qr(np.concatenate([cumulative, q], axis=1))[0]
        residual = residual - (residual @ q) @ q.T
    return cumulative


def subspace_comparison(a: np.ndarray, b: np.ndarray, ambient_dim: int) -> dict[str, Any]:
    if not a.shape[1] or not b.shape[1]:
        return {"error": "empty subspace"}
    singular = np.linalg.svd(a.T @ b, compute_uv=False)
    singular = np.clip(singular, 0.0, 1.0)
    angles = np.degrees(np.arccos(singular))
    shared_energy = float(np.square(a.T @ b).sum())
    overlap_a_captured_by_b = shared_energy / a.shape[1]
    overlap_b_captured_by_a = shared_energy / b.shape[1]
    return {
        "dim_a": int(a.shape[1]),
        "dim_b": int(b.shape[1]),
        "principal_angle_mean_degrees": float(angles.mean()),
        "principal_angle_min_degrees": float(angles.min()),
        "grassmann_distance_radians": float(np.linalg.norm(np.radians(angles))),
        "overlap_a_captured_by_b": overlap_a_captured_by_b,
        "overlap_b_captured_by_a": overlap_b_captured_by_a,
        "random_baseline_a_captured_by_b": float(b.shape[1] / ambient_dim),
        "random_baseline_b_captured_by_a": float(a.shape[1] / ambient_dim),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ridge-alpha", type=float, default=RIDGE_ALPHA)
    parser.add_argument("--max-orthogonal-steps", type=int, default=12)
    parser.add_argument("--predictor-layer", type=int, default=3)
    args = parser.parse_args()
    data = load_discovery(args.capture_dirs)
    y = data["targets"]
    seeds = data["seeds"]
    realized = np.stack(
        [y[:, target_index("realized_dx")], y[:, target_index("realized_dz")]], axis=1
    )
    realized_xz_magnitude = np.linalg.norm(realized, axis=1)
    nonzero_threshold = float(np.quantile(realized_xz_magnitude, 0.25))
    realized_mask = realized_xz_magnitude > nonzero_threshold
    realized_direction = realized / np.maximum(realized_xz_magnitude[:, None], 1e-8)
    goal = np.stack(
        [y[:, target_index("goal_vector_x")], y[:, target_index("goal_vector_z")]], axis=1
    )
    goal_magnitude = np.linalg.norm(goal, axis=1)
    goal_mask = goal_magnitude > 1e-8
    goal_direction = goal / np.maximum(goal_magnitude[:, None], 1e-8)
    scalar_targets = {
        "motion_magnitude": (y[:, target_index("realized_hand_delta_magnitude")], np.ones(len(y), dtype=bool)),
        "wall_signed_distance": (y[:, target_index("wall_signed_distance")], np.ones(len(y), dtype=bool)),
        "progress": (y[:, target_index("progress")], np.ones(len(y), dtype=bool)),
    }
    direction_targets = {
        "realized_xz_direction": (realized_direction, realized_mask),
        "goal_xz_direction": (goal_direction, goal_mask),
    }

    curves = {}
    for family, features in data["features"].items():
        curves[family] = {}
        for name, (target, mask) in direction_targets.items():
            curves[family][name] = emergence_curve(
                features, target, seeds, mask, True, args.ridge_alpha
            )
        for name, (target, mask) in scalar_targets.items():
            curves[family][name] = emergence_curve(
                features, target, seeds, mask, False, args.ridge_alpha
            )

    x = data["features"]["predictor"][:, args.predictor_layer].astype(np.float64)
    probe_targets = {**direction_targets, **scalar_targets}
    sequences = {}
    bases = {}
    for name, (target, mask) in probe_targets.items():
        circular = target.ndim == 2
        sequence = orthogonal_probe_sequence(
            x, target, seeds, mask, args.ridge_alpha, args.max_orthogonal_steps, circular
        )
        threshold = 0.1 if circular else 0.05
        useful_steps = 0
        for row in sequence:
            if row["r2"] < threshold:
                break
            useful_steps += 1
        lower_bound = bool(
            useful_steps == args.max_orthogonal_steps
            and sequence
            and sequence[-1]["r2"] >= threshold
        )
        sequences[name] = {
            "rows": sequence,
            "stopping_r2": threshold,
            "useful_probe_iterations": useful_steps,
            "estimated_dimension": useful_steps * (2 if circular else 1),
            "dimension_is_lower_bound": lower_bound,
        }
        bases[name] = fit_full_basis(x, target, mask, args.ridge_alpha, useful_steps)

    comparisons = {}
    names = sorted(bases)
    for index, name_a in enumerate(names):
        for name_b in names[index + 1 :]:
            comparisons[f"{name_a}__vs__{name_b}"] = subspace_comparison(
                bases[name_a], bases[name_b], x.shape[1]
            )

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "scope": "discovery-only task-geometry emergence diagnostics",
        "paper_adaptation": "Diagnostics adapted from arXiv:2602.07050; no universal PEZ location assumed.",
        "sample_count": int(len(y)),
        "episode_count": int(len(set(zip(data["seeds"].tolist(), data["episodes"].tolist())))),
        "cross_validation": "leave one collection seed out; whole episodes remain grouped",
        "realized_direction_filter": {
            "definition": "XZ displacement magnitude above discovery-set 25th percentile",
            "threshold": nonzero_threshold,
            "kept_samples": int(realized_mask.sum()),
        },
        "ridge_alpha": args.ridge_alpha,
        "curves": curves,
        "orthogonal_probe_layer": {"family": "predictor", "layer": args.predictor_layer},
        "orthogonal_probe_sequences": sequences,
        "subspace_comparisons": comparisons,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_dir / "emergence_geometry.json", report)
    lines = [
        "# Task-geometry emergence screen",
        "",
        "Discovery-only, leave-one-collection-seed-out analysis. A one-third-depth emergence layer was not assumed.",
        "",
        "| Family | Coordinate | Peak layer | Peak R² | Largest upward jump | Near-peak layers |",
        "|---|---|---:|---:|---:|---|",
    ]
    for family in ("encoder", "predictor"):
        for name, curve in curves[family].items():
            lines.append(
                f"| {family} | {name} | {curve['peak_layer']} | {curve['peak_r2']:.3f} | "
                f"{curve['largest_positive_jump']:+.3f} | {curve['near_peak_layers_within_0.05']} |"
            )
    lines.extend(
        [
            "",
            f"## Predictor block {args.predictor_layer}: iterative orthogonal probe dimensions",
            "",
            "| Coordinate | Estimated dimension | Initial R² | R² after first removal |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, sequence in sequences.items():
        rows = sequence["rows"]
        dimension = str(sequence["estimated_dimension"])
        if sequence["dimension_is_lower_bound"]:
            dimension = f">={dimension}"
        lines.append(
            f"| {name} | {dimension} | "
            f"{rows[0]['r2'] if rows else float('nan'):.3f} | "
            f"{rows[1]['r2'] if len(rows) > 1 else float('nan'):.3f} |"
        )
    lines.append("")
    (args.output_dir / "emergence_geometry.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
