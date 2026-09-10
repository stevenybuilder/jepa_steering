#!/usr/bin/env python3
"""Fit once on150 discovery trajectories, freeze, then report75 validation.

No validation-driven hyperparameter or operator selection. Confirmation never opens.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Ridge

from coordinate_calibration_operator import CoordinateCalibration, calibration_matrix
from protocol import file_sha256, write_json_atomic

OBSERVER_SHA = "200feda5534ad2270a8cac7be0b2890d0c5b888198d92d735bad17aa5c2e6f8d"


def load_coordinate_split(directories, split):
    if split not in ("discovery", "validation"):
        raise ValueError("Confirmation cannot be loaded")
    episode_ids = range(50) if split == "discovery" else range(50, 75)
    fields = {name: [] for name in ("p3", "predicted", "natural", "next_xyz", "episode", "seed")}
    seen, sources = set(), []
    for directory in directories:
        receipt = json.loads((directory/"DONE.json").read_text())
        if not receipt.get("complete"):
            raise ValueError("Incomplete capture receipt")
        for entry in receipt["outputs"]:
            key = int(entry["seed"]), int(entry["episode"])
            if key[0] not in (1, 2, 3) or key[1] not in episode_ids:
                continue  # Never hash or deserialize excluded splits.
            path = directory/entry["path"]
            if key in seen or not path.resolve().is_relative_to(directory.resolve()) or file_sha256(path) != entry["sha256"]:
                raise ValueError("Duplicate, escaping, or mismatched capture")
            value = torch.load(path, map_location="cpu", weights_only=False)
            if value["meta"]["split"] != split:
                raise ValueError("Capture split disagreement")
            ids = value.get("block_ids", {}).get("predictor", list(range(6)))
            p3 = value["features"]["predictor"][:, ids.index(3)].float().numpy().astype(float)
            predicted = value["features"]["predicted_final"].float().numpy().astype(float)
            natural = value["features"]["encoded_final"].float().numpy().astype(float)
            next_xyz = value["labels"]["hand_xyz"][1:].float().numpy().astype(float)
            if p3.shape != (19, 400) or predicted.shape != (19, 384) or natural.shape != (20, 384) or next_xyz.shape != (19, 3):
                raise ValueError("Unexpected pooled feature/true NEXT XYZ alignment")
            for name, data in (("p3", p3), ("predicted", predicted), ("natural", natural), ("next_xyz", next_xyz)):
                if not np.isfinite(data).all():
                    raise ValueError("Nonfinite discovery/validation data")
                fields[name].append(data)
            fields["episode"].append(np.full(19, key[1]))
            fields["seed"].append(np.full(19, key[0]))
            sources.append({"seed": key[0], "episode": key[1], "path": str(path), "sha256": entry["sha256"]})
            seen.add(key)
    if seen != {(s, e) for s in (1, 2, 3) for e in episode_ids}:
        raise ValueError(f"Incomplete {split} trajectory set: {len(seen)}")
    return {**{name: np.concatenate(data) for name, data in fields.items()}, "sources": sources}


def fit_candidate(data, observer):
    x, y = data["p3"], data["next_xyz"]
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-6] = 1.
    q_mean, q_scale = y.mean(0), y.std(0)
    q_scale[q_scale < 1e-8] = 1.
    model = Ridge(alpha=100., solver="cholesky").fit((x-mean)/scale, (y-q_mean)/q_scale)
    cov = LedoitWolf().fit(data["natural"])
    sigma = torch.from_numpy(cov.covariance_)
    jac = torch.from_numpy(observer["coef"]/observer["scale"][None, :]/q_scale[:, None])
    matrix = calibration_matrix(sigma, jac, .1).numpy()
    radius = .01*float(np.sqrt(np.mean(np.sum(data["natural"]**2, axis=-1))))
    return {"p3_mean": mean, "p3_scale": scale, "p3_coef": model.coef_, "p3_intercept": model.intercept_,
            "q_mean": q_mean, "q_scale": q_scale, **{"observer_"+key: value for key, value in observer.items()},
            "visual_covariance": sigma.numpy(), "observer_jacobian_standardized_q": jac.numpy(),
            "correction_matrix": matrix, "cap_radius": radius, "beta": .5, "lambda": .1,
            "covariance_shrinkage": float(cov.shrinkage_), "coordinate_rank": int(np.linalg.matrix_rank(matrix))}


def error_metrics(predicted, truth):
    difference = predicted-truth
    return {"rmse_xyz_m": float(np.sqrt(np.mean(difference**2))),
            "rmse_per_axis_m": np.sqrt(np.mean(difference**2, axis=0)).tolist(),
            "mean_euclidean_error_m": float(np.linalg.norm(difference, axis=-1).mean())}


def validate_frozen(artifact, data):
    operator = CoordinateCalibration(artifact, dtype=torch.float64)
    p3, visual = torch.from_numpy(data["p3"]), torch.from_numpy(data["predicted"])
    original = operator.observer_physical(visual).numpy()
    results, arrays = {}, {"truth_next_xyz": data["next_xyz"], "observer_before_xyz": original}
    for name, sham in (("static_candidate", False), ("matched_sham", True)):
        delta, saturated, residual = operator.delta(p3, visual, sham=sham)
        after = operator.observer_physical(visual+delta).numpy()
        results[name] = {**error_metrics(after, data["next_xyz"]),
                         "cap_saturation_fraction": float(saturated.double().mean()),
                         "mean_applied_raw_l2": float(delta.norm(dim=-1).mean())}
        arrays[name+"_observer_xyz"] = after
    qhat = ((data["p3"]-artifact["p3_mean"])/artifact["p3_scale"]) @ artifact["p3_coef"].T + artifact["p3_intercept"]
    qhat = qhat*artifact["q_scale"]+artifact["q_mean"]
    return {"unsteered_observer": error_metrics(original, data["next_xyz"]),
            "p3_qhat_readout": error_metrics(qhat, data["next_xyz"]), **results}, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--observer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    if file_sha256(args.observer) != OBSERVER_SHA:
        raise ValueError("Existing independent frozen observer hash mismatch")
    observer = dict(np.load(args.observer))
    discovery = load_coordinate_split(args.capture_dirs, "discovery")
    artifact = fit_candidate(discovery, observer)
    path = args.output_dir/"coordinate_candidate.npz"
    np.savez_compressed(path, **artifact)
    frozen = {"frozen": True, "candidate_sha256": file_sha256(path), "observer_sha256": OBSERVER_SHA,
              "discovery_sources": discovery["sources"], "discovery_trajectories": 150, "training_pairs": 2850,
              "natural_reference_states": 3000, "ridge_alpha": 100., "beta": .5, "lambda": .1,
              "cap_radius": artifact["cap_radius"], "coordinate_rank": artifact["coordinate_rank"],
              "cap_definition": ".01 * sqrt(mean_discovery(sum_channels(spatially_pooled_natural_visual**2)))",
              "cap_order": "Euclidean radial cap AFTER beta times unconstrained covariance-weighted solution",
              "scope": "coordinate-calibrated Sonar pilot candidate, not native frame, full Frankenstein, COAST or HMM",
              "runtime_truth_access": False, "correction_site": "returned predicted visual at every imagined step; uniform spatial delta; proprio unchanged",
              "fit_script_sha256": file_sha256(Path(__file__)),
              "operator_script_sha256": file_sha256(Path(__file__).with_name("coordinate_calibration_operator.py"))}
    write_json_atomic(args.output_dir/"FROZEN.json", frozen)
    print(json.dumps({"event": "candidate_frozen", "sha256": frozen["candidate_sha256"], "rank": artifact["coordinate_rank"], "cap_radius": artifact["cap_radius"]}), flush=True)
    # First validation tensor access occurs only AFTER immutable fit bytes/receipt.
    validation = load_coordinate_split(args.capture_dirs, "validation")
    metrics, arrays = validate_frozen(artifact, validation)
    if file_sha256(path) != frozen["candidate_sha256"]:
        raise ValueError("Frozen candidate changed during validation")
    np.savez_compressed(args.output_dir/"validation_predictions.npz", **arrays, seeds=validation["seed"], episodes=validation["episode"])
    write_json_atomic(args.output_dir/"validation.json", {"complete": True, "metrics": metrics,
                      "trajectories": 75, "prediction_pairs": 1425, "sources": validation["sources"],
                      "hyperparameter_search": False, "candidate_selected_on_validation": False,
                      "limitation": "Fixed-action one-step observer physical prediction error; not a simulator performance result", "confirmation_opened": False})
    write_json_atomic(args.output_dir/"DONE.json", {"complete": True, "candidate_sha256": frozen["candidate_sha256"],
                      "outputs": [{"path": p.name, "sha256": file_sha256(p)} for p in sorted(args.output_dir.iterdir()) if p.is_file()]})
    print(json.dumps({"event": "validation_complete", "metrics": metrics}), flush=True)


if __name__ == "__main__":
    main()
