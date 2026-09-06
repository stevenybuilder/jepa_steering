#!/usr/bin/env python3
"""Bounded nonlinear interface screen; frozen readouts, not causal/manifold proof.

Discovery: seeds 1..3, episodes 0..49; validation: 50..74; confirmation is
never opened. All normalization/PCA/readouts are fitted on training data only.
The four interfaces are pooled, so this cannot exclude spatially nonlinear codes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import rbf_kernel

from complete_cached_geometry import corrected_labels, sha256, write_json

SITES = ("encoded_final", "predictor_block3", "predictor_block5", "predicted_final")
PROTOCOL = {
    "schema_version": 1, "sites": SITES,
    "targets": ["step_progress", "wall_clearance", "realized_xz_direction"],
    "discovery_episodes_per_seed": [0, 49], "validation_episodes_per_seed": [50, 74],
    "confirmation_never_opened": True, "pca_components": 64,
    "ridge_alpha": 100.0, "rbf_alpha": 1.0, "rbf_gamma": "1/(2 * PCA dimension)",
    "hyperparameter_search": False, "target_standardization": "training fold only",
    "causal_operator_selection": False,
    "limitations": ["pooled tokens", "fixed readout capacities and regularization",
                    "nonlinear readout gain is not proof of a curved manifold",
                    "gradient variation is an estimated readout property, not native coordinates",
                    "encoder lacks predictor action inputs; scores do not isolate depth effects"],
}


def eligible(item, split):
    bounds = {"discovery": (0, 50), "validation": (50, 75)}
    if split not in bounds:
        raise ValueError("Confirmation access is forbidden")
    lo, hi = bounds[split]
    return int(item["seed"]) in (1, 2, 3) and lo <= int(item["episode"]) < hi


def load_split(directories, split):
    import torch
    features = {name: [] for name in SITES}
    targets, masks, seeds, episodes, sources = [], [], [], [], []
    seen = set()
    for directory in directories:
        done = json.loads((directory / "DONE.json").read_text())
        if not done.get("complete"):
            raise ValueError("Incomplete capture receipt")
        for item in done["outputs"]:
            if not eligible(item, split):  # Filter BEFORE tensor access/hash.
                continue
            key = int(item["seed"]), int(item["episode"])
            if key in seen:
                raise ValueError("Duplicate episode")
            path = directory / item["path"]
            if not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError("Escaping capture path")
            if sha256(path) != item["sha256"]:
                raise ValueError("Capture hash mismatch")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload["meta"]["split"] != split:
                raise ValueError("Split mismatch")
            labels = corrected_labels({k: v.numpy() for k, v in payload["labels"].items()})
            stored = payload["features"]
            ids = payload.get("block_ids", {}).get("predictor", list(range(6)))
            for name in SITES:
                if name.startswith("predictor_block"):
                    value = stored["predictor"][:, ids.index(int(name[-1]))]
                else:
                    value = stored[name][:19]
                array = value.float().numpy()
                if array.ndim != 2 or len(array) != 19 or not np.isfinite(array).all():
                    raise ValueError("Expected finite pooled [19,channels]")
                features[name].append(array)
            targets.append(np.column_stack((labels["step_progress"], labels["wall_clearance"],
                                            labels["realized_xz_direction"])))
            masks.append(~labels["realized_xz_zero_mask"])
            seeds.extend([key[0]] * 19)
            episodes.extend([key[1]] * 19)
            seen.add(key)
            sources.append({"path": str(path), "sha256": item["sha256"]})
    expected = {(s, e) for s in (1, 2, 3) for e in
                (range(50) if split == "discovery" else range(50, 75))}
    if seen != expected:
        raise ValueError(f"Expected {len(expected)} {split} episodes, got {len(seen)}")
    return {"features": {k: np.concatenate(v).astype(float) for k, v in features.items()},
            "targets": np.concatenate(targets), "motion_mask": np.concatenate(masks),
            "seeds": np.asarray(seeds), "episodes": np.asarray(episodes), "sources": sources}


def fit_readouts(x, y, components=64):
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-6] = 1.0
    ym, ys = y.mean(0), y.std(0)
    ys[ys < 1e-8] = 1.0
    z, target = (x - mean) / scale, (y - ym) / ys
    pca = PCA(n_components=min(components, x.shape[1], len(x) - 1),
              svd_solver="randomized", random_state=0, whiten=True).fit(z)
    projected = pca.transform(z)
    gamma = 1.0 / (2 * projected.shape[1])
    models = {"full_ridge": Ridge(alpha=100, solver="cholesky").fit(z, target),
              "pca_ridge": Ridge(alpha=100, solver="cholesky").fit(projected, target),
              "rbf_ridge": KernelRidge(alpha=1, kernel="rbf", gamma=gamma).fit(projected, target)}
    return dict(mean=mean, scale=scale, target_mean=ym, target_scale=ys, pca=pca,
                models=models, train_projected=projected, gamma=gamma)


def predict(bundle, x):
    z = (x - bundle["mean"]) / bundle["scale"]
    projected = bundle["pca"].transform(z)
    return {name: model.predict(z if name == "full_ridge" else projected) * bundle["target_scale"]
            + bundle["target_mean"] for name, model in bundle["models"].items()}


def rbf_jacobian(z, reference, dual, gamma):
    weights = rbf_kernel(z, reference, gamma=gamma)
    return -2 * gamma * np.einsum("mn,mnd,ny->mdy", weights,
                                 z[:, None, :] - reference[None, :, :], dual)


def gradient_summary(bundle, x):
    # This is a learned-readout Jacobian, NOT a world-model causal Jacobian.
    z = (x - bundle["mean"]) / bundle["scale"]
    projected = bundle["pca"].transform(z)
    jac = rbf_jacobian(projected, bundle["train_projected"],
                       bundle["models"]["rbf_ridge"].dual_coef_, bundle["gamma"])
    chain = bundle["pca"].components_ / np.sqrt(bundle["pca"].explained_variance_)[:, None]
    jac = np.einsum("mky,kd->mdy", jac, chain) / bundle["scale"][None, :, None]
    jac *= bundle["target_scale"][None, None, :]
    result = {}
    for name, cols in (("step_progress", slice(0, 1)), ("wall_clearance", slice(1, 2)),
                       ("realized_xz_direction", slice(2, 4))):
        flat = jac[:, :, cols].reshape(len(x), -1)
        norm = np.linalg.norm(flat, axis=1)
        unit = flat / np.maximum(norm[:, None], 1e-12)
        cosine = unit @ unit.T
        pairs = cosine[np.triu_indices(len(x), 1)]
        result[name] = {"median_pairwise_readout_gradient_cosine": float(np.median(pairs)),
                        "gradient_norm_quantiles": np.quantile(norm, [.1, .5, .9]).tolist(),
                        "interpretation": "descriptive readout variation, not native chart discovery"}
    return result


def score(truth, prediction, motion_mask):
    values = {"step_progress": float(r2_score(truth[:, 0], prediction[:, 0])),
              "wall_clearance": float(r2_score(truth[:, 1], prediction[:, 1])),
              "realized_xz_direction": float(r2_score(truth[motion_mask, 2:], prediction[motion_mask, 2:]))}
    a, b = truth[motion_mask, 2:], prediction[motion_mask, 2:]
    angle = np.arctan2(a[:, 1], a[:, 0]) - np.arctan2(b[:, 1], b[:, 0])
    values["direction_angular_mae_degrees"] = float(np.mean(np.abs(np.arctan2(np.sin(angle), np.cos(angle)))) * 180 / np.pi)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluate-validation", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "frozen_protocol.json", PROTOCOL)
    discovery = load_split(args.capture_dirs, "discovery")
    output = {"protocol": PROTOCOL, "discovery_episodes": 150, "rows": [], "frozen_models": {}}
    frozen = {}
    for site in SITES:
        x, y = discovery["features"][site], discovery["targets"]
        predictions = {key: np.zeros_like(y) for key in ("full_ridge", "pca_ridge", "rbf_ridge")}
        for held_seed in (1, 2, 3):
            train, test = discovery["seeds"] != held_seed, discovery["seeds"] == held_seed
            bundle = fit_readouts(x[train], y[train])
            for model, pred in predict(bundle, x[test]).items():
                predictions[model][test] = pred
        row = {"site": site, "discovery_leave_seed_out": {k: score(y, v, discovery["motion_mask"])
                                                         for k, v in predictions.items()}}
        frozen[site] = fit_readouts(x, y)
        # Persist models before opening validation; useful for exact repeatability.
        import joblib
        path = args.output_dir / f"{site}_readouts.joblib"
        joblib.dump(frozen[site], path)
        output["frozen_models"][site] = {"path": path.name, "sha256": sha256(path)}
        output["rows"].append(row)
        print(json.dumps(row), flush=True)
    write_json(args.output_dir / "discovery_results.json", output)
    if args.evaluate_validation:
        validation = load_split(args.capture_dirs, "validation")
        output["validation_episodes"] = 75
        output["validation_sources"] = validation["sources"]
        for row in output["rows"]:
            site = row["site"]
            pred = predict(frozen[site], validation["features"][site])
            row["held_validation"] = {k: score(validation["targets"], v, validation["motion_mask"])
                                      for k, v in pred.items()}
            # 48 deterministic states from distinct validation episodes.
            indices = np.arange(0, len(validation["targets"]), 19)[:48]
            row["readout_gradient_variation"] = gradient_summary(frozen[site], validation["features"][site][indices])
            row["gradient_states"] = 48
            print(json.dumps({"site": site, "held_validation": row["held_validation"]}), flush=True)
    output.update(complete=True, confirmation_opened=False, recommended_operator="none_validated",
                  discovery_sources=discovery["sources"])
    write_json(args.output_dir / "nonlinear_interfaces.json", output)
    paths = [p for p in args.output_dir.iterdir() if p.is_file()]
    write_json(args.output_dir / "DONE.json", {"complete": True,
               "outputs": [{"path": p.name, "sha256": sha256(p)} for p in paths]})


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        main()
