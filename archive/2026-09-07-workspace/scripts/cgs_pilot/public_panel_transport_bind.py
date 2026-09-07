#!/usr/bin/env python3
"""Bind a validated relational transport to a target-model Sonar operator.

The target artifact supplies its own coordinate encoder/decoder, routing model,
support metric, and dose geometry.  Only the source's outcome operator bank is
transported into target factor coordinates.  This produces a directly runnable
method comparator without confusing relational Procrustes transport with the
within-model Gaussian-OT arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from public_panel_sonar_math import gaussian_ot_affine


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def map_mean(mean: np.ndarray, source_center: np.ndarray, target_center: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return target_center + (np.asarray(mean) - source_center) @ rotation


def map_covariance(covariance: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return rotation.T @ np.asarray(covariance) @ rotation


def map_operator(operator: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return rotation.T @ np.asarray(operator) @ rotation


def bind_transport(source_path: Path, target_path: Path, modules_path: Path, out: Path) -> dict:
    with np.load(source_path) as raw:
        source = {key: np.asarray(raw[key]) for key in raw.files}
    with np.load(target_path) as raw:
        target = {key: np.asarray(raw[key]) for key in raw.files}
    with np.load(modules_path) as raw:
        modules = {key: np.asarray(raw[key]) for key in raw.files}
    required = {"transport_rotation", "transport_mean_source", "transport_mean_target", "transport_eligible"}
    missing = sorted(required - set(modules))
    if missing:
        raise ValueError(f"transport module artifact lacks {missing}")
    if not bool(modules["transport_eligible"].item()):
        raise ValueError("relational transport failed its held-out admission gate")
    rotation = modules["transport_rotation"].astype(np.float64)
    source_center = modules["transport_mean_source"].astype(np.float64)
    target_center = modules["transport_mean_target"].astype(np.float64)
    rank = int(target["coordinate_rank"].item())
    if rotation.shape != (rank, rank) or int(source["coordinate_rank"].item()) != rank:
        raise ValueError("transport and Sonar factor ranks disagree")

    source_regime_in_target = np.stack([
        map_mean(value, source_center, target_center, rotation) for value in source["regime_means"]
    ])
    cost = np.linalg.norm(
        source_regime_in_target[:, None, :] - target["regime_means"][None, :, :], axis=2
    )
    source_index, target_index = linear_sum_assignment(cost)
    source_for_target = np.empty(len(target_index), dtype=np.int64)
    source_for_target[target_index] = source_index
    output = dict(target)

    for outcome in ("success", "failure"):
        output[f"{outcome}_weights"] = source[f"{outcome}_weights"][source_for_target]
        output[f"{outcome}_means"] = np.stack([
            map_mean(source[f"{outcome}_means"][index], source_center, target_center, rotation)
            for index in source_for_target
        ]).astype(np.float32)
        output[f"{outcome}_covariances"] = np.stack([
            map_covariance(source[f"{outcome}_covariances"][index], rotation)
            for index in source_for_target
        ]).astype(np.float32)

    for key in (
        "local_contrastive_conceptors", "local_success_conceptors", "local_failure_conceptors",
        "local_matched_spectrum_conceptors", "local_label_shuffled_conceptors",
    ):
        if key in source:
            output[key] = np.stack([
                map_operator(source[key][index], rotation) for index in source_for_target
            ]).astype(np.float32)
    for key in (
        "global_contrastive_conceptor", "success_conceptor", "matched_spectrum_conceptor",
        "global_label_shuffled_conceptor",
    ):
        if key in source:
            output[key] = map_operator(source[key], rotation).astype(np.float32)
    for key in ("local_success_centers", "local_failure_centers"):
        if key in source:
            output[key] = np.stack([
                map_mean(source[key][index], source_center, target_center, rotation)
                for index in source_for_target
            ]).astype(np.float32)
    for key in ("global_success_center", "global_failure_center"):
        if key in source:
            output[key] = map_mean(source[key], source_center, target_center, rotation).astype(np.float32)

    ot_affine, ot_offset = [], []
    floor = float(target.get("covariance_floor", np.asarray(1e-3)).item())
    for regime in range(len(source_for_target)):
        affine, offset = gaussian_ot_affine(
            output["failure_means"][regime], output["failure_covariances"][regime],
            output["success_means"][regime], output["success_covariances"][regime], floor=floor,
        )
        ot_affine.append(affine)
        ot_offset.append(offset)
    output["ot_affine"] = np.stack(ot_affine).astype(np.float32)
    output["ot_offset"] = np.stack(ot_offset).astype(np.float32)
    output["relational_transport_eligible"] = np.asarray(True)
    output["relational_transport_source_sha256"] = np.asarray(sha256_file(source_path))
    output["relational_transport_target_sha256"] = np.asarray(sha256_file(target_path))
    output["relational_transport_modules_sha256"] = np.asarray(sha256_file(modules_path))
    output["relational_transport_source_for_target_regime"] = source_for_target.astype(np.int32)
    output["relational_transport_rotation"] = rotation.astype(np.float32)
    output["relational_transport_mean_source"] = source_center.astype(np.float32)
    output["relational_transport_mean_target"] = target_center.astype(np.float32)

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite transported operator: {out}")
    np.savez_compressed(out, **output)
    return {
        "source": str(source_path.resolve()),
        "target": str(target_path.resolve()),
        "modules": str(modules_path.resolve()),
        "out": str(out.resolve()),
        "out_sha256": sha256_file(out),
        "source_for_target_regime": source_for_target.tolist(),
        "mean_regime_alignment_distance": float(cost[source_index, target_index].mean()),
        "row_vector_convention": "x_target=(x_source-mu_source)@Q+mu_target; C_target=Q.T@C_source@Q",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sonar", required=True, type=Path)
    parser.add_argument("--target-sonar", required=True, type=Path)
    parser.add_argument("--transport-modules", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(bind_transport(
        args.source_sonar, args.target_sonar, args.transport_modules, args.out
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

