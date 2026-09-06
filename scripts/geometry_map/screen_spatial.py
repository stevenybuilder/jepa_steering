#!/usr/bin/env python3
"""Discovery-only token localization from deterministic projected activations."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from protocol import file_sha256, write_json_atomic
from screen_pooled import TARGETS, target_array


RIDGE_ALPHA = 100.0


def load_projected(capture_dirs: list[Path]) -> dict[str, Any]:
    family_rows: dict[str, list[np.ndarray]] = {"encoder": [], "predictor": []}
    ids: dict[str, list[int] | None] = {"encoder": None, "predictor": None}
    targets = []
    seeds = []
    episodes = []
    sources = []
    for directory in capture_dirs:
        done_path = directory / "DONE.json"
        done = json.loads(done_path.read_text())
        sources.append({"done": str(done_path), "sha256": file_sha256(done_path)})
        for item in done["outputs"]:
            path = directory / item["path"]
            if file_sha256(path) != item["sha256"]:
                raise RuntimeError(f"Hash mismatch {path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload["meta"]["split"] != "discovery" or payload["mode"] != "projected":
                raise RuntimeError(f"Unexpected payload scope/mode in {path}")
            for family in family_rows:
                observed_ids = [int(value) for value in payload["block_ids"][family]]
                if ids[family] is None:
                    ids[family] = observed_ids
                elif ids[family] != observed_ids:
                    raise RuntimeError(f"Block mismatch for {family}: {ids[family]} vs {observed_ids}")
            family_rows["encoder"].append(payload["features"]["encoder"][:19].float().numpy())
            family_rows["predictor"].append(payload["features"]["predictor"].float().numpy())
            targets.append(
                np.stack([target_array(payload["labels"], name, component) for name, _kind, component in TARGETS], axis=1)
            )
            seeds.append(np.full(19, int(payload["meta"]["seed"]), dtype=np.int64))
            episodes.append(np.full(19, int(payload["meta"]["episode"]), dtype=np.int64))
    return {
        "features": {family: np.concatenate(rows, axis=0) for family, rows in family_rows.items()},
        "block_ids": ids,
        "targets": np.concatenate(targets, axis=0),
        "seeds": np.concatenate(seeds),
        "episodes": np.concatenate(episodes),
        "sources": sources,
    }


def batched_token_ridge(x: np.ndarray, y: np.ndarray, seeds: np.ndarray, alpha: float) -> np.ndarray:
    # x: samples, tokens, projected_dimensions; y: samples, targets
    prediction = np.full((len(x), x.shape[1], y.shape[1]), np.nan, dtype=np.float32)
    identity = np.eye(x.shape[2], dtype=np.float32)[None, :, :]
    for held_seed in (1, 2, 3):
        train = seeds != held_seed
        test = seeds == held_seed
        x_train = x[train].astype(np.float32)
        x_test = x[test].astype(np.float32)
        mean = x_train.mean(axis=0, keepdims=True)
        scale = x_train.std(axis=0, keepdims=True)
        scale[scale < 1e-6] = 1.0
        x_train = (x_train - mean) / scale
        x_test = (x_test - mean) / scale
        y_train = y[train].astype(np.float32)
        y_mean = y_train.mean(axis=0, keepdims=True)
        y_train = y_train - y_mean
        xtx = np.einsum("ntd,nte->tde", x_train, x_train, optimize=True)
        xty = np.einsum("ntd,nk->tdk", x_train, y_train, optimize=True)
        coefficients = np.linalg.solve(xtx + alpha * identity, xty)
        prediction[test] = np.einsum("ntd,tdk->ntk", x_test, coefficients, optimize=True) + y_mean
    return prediction


def token_scores(kind: str, truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    if kind == "binary":
        return np.asarray(
            [roc_auc_score(truth, prediction[:, token]) if np.unique(truth).size > 1 else np.nan for token in range(256)]
        )
    residual = np.square(truth[:, None] - prediction).sum(axis=0)
    total = np.square(truth - truth.mean()).sum()
    return 1.0 - residual / max(total, 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--pooled-screen", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ridge-alpha", type=float, default=RIDGE_ALPHA)
    args = parser.parse_args()
    data = load_projected(args.capture_dirs)
    pooled = json.loads(args.pooled_screen.read_text())
    time_baselines = pooled["baselines"]
    rows = []
    maps = {}
    concentration = []
    for family, features in data["features"].items():
        block_ids = data["block_ids"][family]
        assert block_ids is not None
        for local_index, block_id in enumerate(block_ids):
            prediction = batched_token_ridge(features[:, local_index], data["targets"], data["seeds"], args.ridge_alpha)
            for target_index, (name, kind, _component) in enumerate(TARGETS):
                scores = token_scores(kind, data["targets"][:, target_index], prediction[:, :, target_index])
                map_key = f"{family}.block{block_id}.{name}"
                maps[map_key] = scores.tolist()
                positive = np.maximum(np.nan_to_num(scores - time_baselines[name], nan=0.0), 0.0)
                mass = float(positive.sum())
                probability = positive / mass if mass > 0 else np.zeros_like(positive)
                nonzero = probability[probability > 0]
                entropy = float(-(nonzero * np.log(nonzero)).sum() / np.log(256)) if len(nonzero) else 0.0
                concentration.append(
                    {
                        "family": family,
                        "block": block_id,
                        "target": name,
                        "score_median": float(np.nanmedian(scores)),
                        "score_q90": float(np.nanquantile(scores, 0.9)),
                        "score_max": float(np.nanmax(scores)),
                        "positive_delta_entropy": entropy,
                        "top10_positive_mass_fraction": float(np.sort(positive)[-10:].sum() / mass) if mass > 0 else 0.0,
                    }
                )
                order = np.argsort(np.nan_to_num(scores, nan=-np.inf))[::-1]
                for rank, token in enumerate(order[:10], start=1):
                    rows.append(
                        {
                            "family": family,
                            "block": block_id,
                            "target": name,
                            "metric": "auroc" if kind == "binary" else "r2",
                            "token": int(token),
                            "row": int(token // 16),
                            "column": int(token % 16),
                            "rank_within_block": rank,
                            "score": float(scores[token]),
                            "time_baseline": float(time_baselines[name]),
                            "delta_vs_time": float(scores[token] - time_baselines[name]),
                        }
                    )
    best = {}
    for name, _kind, _component in TARGETS:
        candidates = [row for row in rows if row["target"] == name and row["rank_within_block"] == 1]
        candidates.sort(key=lambda row: row["delta_vs_time"], reverse=True)
        best[name] = candidates
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "scope": "discovery only: episodes 0-49 within collection seeds 1,2,3",
        "sample_count": int(len(data["targets"])),
        "episode_count": int(len(set(zip(data["seeds"].tolist(), data["episodes"].tolist())))),
        "cross_validation": "leave one collection seed out",
        "projection": "deterministic Gaussian 64-D projection, fixed by family and block",
        "ridge_alpha": args.ridge_alpha,
        "sources": data["sources"],
        "rows": rows,
        "maps": maps,
        "concentration": concentration,
        "best": best,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_dir / "spatial_screen.json", report)
    priority = [
        "progress",
        "wall_signed_distance",
        "straight_path_intersects_wall",
        "realized_hand_delta_magnitude",
        "reward_sum",
    ]
    lines = [
        "# Reach-Wall spatial discovery screen",
        "",
        "Validation and confirmation episodes were not loaded.",
        "",
        "| Target | Family | Block | Token (row, col) | Score | Over time |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name in priority:
        for result in best[name][:2]:
            lines.append(
                f"| {name} | {result['family']} | {result['block']} | {result['token']} "
                f"({result['row']}, {result['column']}) | {result['score']:.3f} | {result['delta_vs_time']:+.3f} |"
            )
    (args.output_dir / "spatial_screen.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
