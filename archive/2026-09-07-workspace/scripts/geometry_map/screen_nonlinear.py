#!/usr/bin/env python3
"""Cheap local-geometry check on discovery-only shortlisted pooled sites."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor

from protocol import write_json_atomic
from screen_pooled import TARGETS, load_discovery, metric


def grouped_knn(x: np.ndarray, y: np.ndarray, seeds: np.ndarray, components: int, neighbors: int) -> np.ndarray:
    prediction = np.full_like(y, np.nan, dtype=np.float64)
    for held_seed in (1, 2, 3):
        train = seeds != held_seed
        test = seeds == held_seed
        mean = x[train].mean(axis=0, keepdims=True)
        scale = x[train].std(axis=0, keepdims=True)
        scale[scale < 1e-6] = 1.0
        x_train = (x[train] - mean) / scale
        x_test = (x[test] - mean) / scale
        pca = PCA(n_components=min(components, x.shape[1]), svd_solver="randomized", random_state=0, whiten=True)
        x_train = pca.fit_transform(x_train)
        x_test = pca.transform(x_test)
        model = KNeighborsRegressor(n_neighbors=neighbors, weights="distance", n_jobs=-1)
        model.fit(x_train, y[train])
        prediction[test] = model.predict(x_test)
    return prediction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--pooled-screen", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder-blocks", type=int, nargs="+", default=[4, 6, 8, 11])
    parser.add_argument("--predictor-blocks", type=int, nargs="+", default=[2, 3, 5])
    parser.add_argument("--components", type=int, default=64)
    parser.add_argument("--neighbors", type=int, default=25)
    args = parser.parse_args()
    data = load_discovery(args.capture_dirs)
    linear = json.loads(args.pooled_screen.read_text())
    linear_lookup = {
        (row["family"], int(row["layer"]), row["target"]): float(row["score"]) for row in linear["rows"]
    }
    rows = []
    for family, blocks in (("encoder", args.encoder_blocks), ("predictor", args.predictor_blocks)):
        for block in blocks:
            prediction = grouped_knn(
                data["features"][family][:, block].astype(np.float64),
                data["targets"],
                data["seeds"],
                args.components,
                args.neighbors,
            )
            for index, (name, kind, _component) in enumerate(TARGETS):
                score = metric(kind, data["targets"][:, index], prediction[:, index])
                linear_score = linear_lookup[(family, block, name)]
                rows.append(
                    {
                        "family": family,
                        "block": block,
                        "target": name,
                        "metric": "auroc" if kind == "binary" else "r2",
                        "knn_score": score,
                        "linear_score": linear_score,
                        "nonlinear_gain": score - linear_score,
                    }
                )
    priority = [
        "progress",
        "wall_signed_distance",
        "straight_path_intersects_wall",
        "realized_hand_delta_magnitude",
        "reward_sum",
        "goal_vector_x",
        "goal_vector_z",
    ]
    best = {}
    for target in priority:
        selected = [row for row in rows if row["target"] == target]
        selected.sort(key=lambda row: row["knn_score"], reverse=True)
        best[target] = selected[:3]
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "scope": "discovery only; leave one collection seed out",
        "method": f"{args.neighbors}-NN after train-fold PCA({args.components}) and whitening",
        "rows": rows,
        "best": best,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_dir / "nonlinear_screen.json", report)
    lines = [
        "# Reach-Wall nonlinear discovery screen",
        "",
        "| Target | Family | Block | kNN | Linear | Gain |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for target in priority:
        row = best[target][0]
        lines.append(
            f"| {target} | {row['family']} | {row['block']} | {row['knn_score']:.3f} | "
            f"{row['linear_score']:.3f} | {row['nonlinear_gain']:+.3f} |"
        )
    (args.output_dir / "nonlinear_screen.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
