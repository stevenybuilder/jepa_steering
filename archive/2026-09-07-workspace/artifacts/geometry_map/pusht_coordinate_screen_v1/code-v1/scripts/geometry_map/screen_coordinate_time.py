#!/usr/bin/env python3
"""Equal-input coordinate/time probes and a discovery-frozen residual candidate.

This is a readout comparison, not a nonlinear-manifold or native-coordinate claim.
Real episode time is a TARGET, never appended to the activation input. Existing
one-step captures cannot identify imagined-horizon dependence; that needs the
separate six-horizon capture. Confirmation episodes are never opened here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import rbf_kernel

from complete_cached_geometry import goal_basis, sha256, to_local, to_world, write_json
from screen_nonlinear_interfaces import eligible

FRAMES = ("world", "goal_relative", "gripper_relative", "goal_aligned")
ALPHAS = (.01, 1.)
PROTOCOL = {
    "version": 1, "site": "predictor_block3", "input": "all400 pooled residual channels",
    "frames": FRAMES, "time_targets": ["observed_episode_raw_step"],
    "time_not_in_activation_inputs": True, "imagined_horizon_in_legacy_data": "constant1, not identifiable",
    "models": ["linear", "rbf"], "alphas_each": ALPHAS,
    "linear_kernel": "standardized dot product / number_of_channels",
    "rbf_gamma": "1/(2*number_of_channels)", "pca": False,
    "selection": "three leave-seed-out discovery folds; same inputs and2 candidates per family",
    "candidate_selection": "frame with minimum discovery-CV linear world-space MSE",
    "candidate_ranks": [1, 2, 8], "candidate_not_native_coordinate_proof": True,
    "validation": "75 whole episodes, opened only after models and candidate frozen",
    "confirmation_access": False, "independent_unit": "whole trajectory, not transition or horizon",
    "frame_inverse_uses_evaluation_anchors": True,
    "object_frame": "not in legacy capture; do not invent wall pose or gripper orientation",
    "limits": ["same tuning budget does not mean equal statistical capacity",
               "a nonlinear readout gain is not intrinsic-manifold identification",
               "elapsed time can be inferred from task progress; not a native clock claim",
               "pooled tokens may conceal spatially distributed nonlinear representations",
               "held episode goals/poses are not automatically out-of-distribution"],
}


def targets_and_frames(current, following, goal, raw_step, object_xyz=None, object_rotation=None):
    current, following, goal = map(lambda a: np.asarray(a, dtype=float), (current, following, goal))
    n = len(current)
    identity = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
    rotation, valid, _ = goal_basis(goal-current)
    # A vertical/zero goal has no unique azimuth. Use a declared identity fallback
    # and retain its mask; all models and frames are compared on the SAME rows.
    rotation[~valid] = np.eye(3)
    origins = {"world": np.zeros_like(current), "goal_relative": goal,
               "gripper_relative": current, "goal_aligned": current}
    rotations = {name: rotation if name == "goal_aligned" else identity for name in FRAMES}
    if object_xyz is not None:
        origins["object_relative"] = np.asarray(object_xyz, dtype=float)
        rotations["object_relative"] = identity if object_rotation is None else np.asarray(object_rotation, dtype=float)
    targets, columns, cursor = [], {}, 0
    for name in origins:
        targets.append(to_local(following-origins[name], rotations[name]))
        columns[name] = slice(cursor, cursor+3)
        cursor += 3
    targets.append(np.asarray(raw_step, dtype=float).reshape(n, 1))
    columns["observed_episode_raw_step"] = slice(cursor, cursor+1)
    return np.concatenate(targets, axis=1), columns, origins, rotations, valid


def load_split(directories, split):
    import torch
    collected = {k: [] for k in ("x", "current", "following", "goal", "raw_step", "seed", "episode")}
    sources, seen = [], set()
    for directory in directories:
        receipt = json.loads((directory/"DONE.json").read_text())
        if not receipt.get("complete"):
            raise ValueError("Incomplete source")
        for entry in receipt["outputs"]:
            if not eligible(entry, split):  # BEFORE opening tensors, including confirmation.
                continue
            key = int(entry["seed"]), int(entry["episode"])
            path = directory/entry["path"]
            if key in seen or not path.resolve().is_relative_to(directory.resolve()) or sha256(path) != entry["sha256"]:
                raise ValueError("Duplicate, escaping, or mismatched source")
            value = torch.load(path, map_location="cpu", weights_only=False)
            if value["meta"]["split"] != split:
                raise ValueError("Split mismatch")
            ids = value.get("block_ids", {}).get("predictor", list(range(6)))
            labels = value["labels"]
            row = {"x": value["features"]["predictor"][:, ids.index(3)].float().numpy(),
                   "current": labels["hand_xyz"][:-1].numpy(), "following": labels["hand_xyz"][1:].numpy(),
                   "goal": labels["goal_xyz"][:-1].numpy(), "raw_step": labels["frame_index"][:-1].numpy(),
                   "seed": np.full(19, key[0]), "episode": np.full(19, key[1])}
            if row["x"].shape != (19, 400) or np.any(np.diff(labels["frame_index"].numpy()) != 5):
                raise ValueError("Incorrect feature/transition alignment")
            for name, array in row.items():
                if not np.isfinite(array).all():
                    raise ValueError("Nonfinite capture")
                collected[name].append(array)
            sources.append({"path": str(path), "sha256": entry["sha256"]})
            seen.add(key)
    episodes = range(50) if split == "discovery" else range(50, 75)
    if seen != {(s, e) for s in (1, 2, 3) for e in episodes}:
        raise ValueError(f"Incomplete {split}: {len(seen)}")
    data = {k: np.concatenate(v) for k, v in collected.items()}
    y, columns, origins, rotations, valid = targets_and_frames(data["current"], data["following"], data["goal"], data["raw_step"])
    return dict(**data, y=y, columns=columns, origins=origins, rotations=rotations,
                frame_valid=valid, sources=sources)


def fit_models(x, y):
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-6] = 1.
    z = (x-mean)/scale
    ym, ys = y.mean(0), y.std(0)
    ys[ys < 1e-8] = 1.
    target = (y-ym)/ys
    d = x.shape[1]
    models = {}
    for alpha in ALPHAS:
        models[f"linear:{alpha}"] = Ridge(alpha=alpha*d, solver="cholesky").fit(z, target)
        models[f"rbf:{alpha}"] = KernelRidge(alpha=alpha, kernel="rbf", gamma=1/(2*d)).fit(z, target)
    return {"mean": mean, "scale": scale, "target_mean": ym, "target_scale": ys, "models": models}


def predict_models(bundle, x):
    z = (x-bundle["mean"])/bundle["scale"]
    return {name: model.predict(z)*bundle["target_scale"]+bundle["target_mean"] for name, model in bundle["models"].items()}


def world_prediction(data, name, local, mask=None):
    index = np.ones(len(local), dtype=bool) if mask is None else mask
    return to_world(local, data["rotations"][name][index])+data["origins"][name][index]


def r2(y, pred):
    denominator = float(np.square(y-y.mean(0)).sum())
    return None if denominator < 1e-15 else float(1-np.square(y-pred).sum()/denominator)


def per_episode_ci(differences, groups, repetitions=1000):
    unique = np.unique(groups)
    means = np.asarray([differences[groups == g].mean() for g in unique])
    rng = np.random.default_rng(17)
    boot = means[rng.integers(len(means), size=(repetitions, len(means)))].mean(1)
    return {"mean": float(means.mean()), "interval95": np.quantile(boot, [.025, .975]).tolist(), "episodes": len(means)}


def choose_models(data, predictions):
    chosen = {}
    for variable, cols in data["columns"].items():
        chosen[variable] = {}
        for family in ("linear", "rbf"):
            chosen[variable][family] = min((key for key in predictions if key.startswith(family+":")),
                key=lambda key: np.square(predictions[key][:, cols]-data["y"][:, cols]).mean())
    return chosen


def evaluate(data, predictions, chosen):
    report = {}
    groups = data["seed"]*10000+data["episode"]
    for variable, cols in data["columns"].items():
        truth = data["y"][:, cols]
        errors = {}
        row = {}
        for family, key in chosen[variable].items():
            prediction = predictions[key][:, cols]
            errors[family] = np.square(prediction-truth).mean(1)
            row[family] = {"model": key, "r2": r2(truth, prediction),
                           "rmse": float(np.sqrt(errors[family].mean()))}
            if variable in data["origins"]:
                world = world_prediction(data, variable, prediction)
                row[family]["world_rmse_m"] = float(np.sqrt(np.square(world-data["following"]).mean()))
        row["nonlinear_minus_linear_mse"] = per_episode_ci(errors["rbf"]-errors["linear"], groups)
        report[variable] = row
    return report


def residual_basis(x, y, alpha, max_rank=8):
    """Orthogonal probe sequence in one RAW Euclidean residual metric.

    A rank is a candidate intervention span, not an intrinsic dimension estimate.
    """
    x = np.asarray(x, dtype=float)
    center = x.mean(0)
    residual = x-center
    basis = np.empty((x.shape[1], 0))
    for _ in range(max_rank):
        scale = residual.std(0)
        scale[scale < 1e-6] = 1.
        model = Ridge(alpha=alpha*x.shape[1], solver="cholesky").fit(residual/scale, y)
        weights = np.atleast_2d(model.coef_).T/scale[:, None]
        if basis.shape[1]:
            weights -= basis@(basis.T@weights)
        u, singular, _ = np.linalg.svd(weights, full_matrices=False)
        rank = min(int(np.sum(singular > max(singular.max(initial=0), 1e-12)*1e-8)), max_rank-basis.shape[1])
        if not rank:
            break
        basis = np.column_stack((basis, u[:, :rank]))
        residual = (x-center)-((x-center)@basis)@basis.T
        if basis.shape[1] == max_rank:
            break
    if not np.allclose(basis.T@basis, np.eye(basis.shape[1]), atol=1e-8):
        raise ValueError("Candidate is not raw-space orthonormal")
    return basis


def freeze_candidate(data, bundle, chosen, cv, directory):
    frame = min(FRAMES, key=lambda name: cv[name]["linear"]["world_rmse_m"])
    key = chosen[frame]["linear"]
    alpha = float(key.split(":")[1])
    cols = data["columns"][frame]
    basis = residual_basis(data["x"], data["y"][:, cols], alpha)
    model = bundle["models"][key]
    coefficient = model.coef_[cols]*bundle["target_scale"][cols, None]/bundle["scale"][None, :]
    intercept = model.intercept_[cols]*bundle["target_scale"][cols]+bundle["target_mean"][cols]-coefficient@bundle["mean"]
    # Fixed development dose; no validation or behavioral dose selection.
    radius = .005*float(np.sqrt(np.mean(np.sum(data["x"]**2, axis=1))))
    path = directory/"residual_candidate.npz"
    np.savez_compressed(path, **{f"basis_raw_rank{k}": basis[:, :min(k, basis.shape[1])] for k in (1, 2, 8)},
                        coordinate_coef_raw=coefficient, coordinate_intercept_raw=intercept,
                        raw_feature_mean=bundle["mean"], raw_feature_scale=bundle["scale"],
                        donor_p3=data["x"], donor_target=data["y"][:, cols], donor_current_hand=data["current"],
                        donor_goal=data["goal"], donor_real_raw_step=data["raw_step"],
                        donor_episode=data["episode"], donor_seed=data["seed"], cap_radius=radius,
                        target_name=np.asarray(frame), max_rank=np.asarray(basis.shape[1]))
    return {"path": path.name, "sha256": sha256(path), "target": frame,
            "selection": "minimum discovery-CV LINEAR world-space error; validation not opened",
            "rank": basis.shape[1], "cap_radius": radius,
            "cap_definition": ".005*RMS norm of discovery pooled P3, raw Euclidean space",
            "native_coordinates_established": False,
            "intervention": "donor subspace replacement, not optimizing the probe to a requested label"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir/"protocol.json", PROTOCOL)
    data = load_split(args.capture_dirs, "discovery")
    predictions = {f"{family}:{alpha}": np.empty_like(data["y"]) for family in ("linear", "rbf") for alpha in ALPHAS}
    mean_baseline = np.empty_like(data["y"])
    time_baseline = np.empty_like(data["y"])
    time_features = np.eye(10)[np.minimum((data["raw_step"]/10).astype(int), 9)]
    for seed in (1, 2, 3):
        train, test = data["seed"] != seed, data["seed"] == seed
        bundle = fit_models(data["x"][train], data["y"][train])
        for key, value in predict_models(bundle, data["x"][test]).items():
            predictions[key][test] = value
        mean_baseline[test] = data["y"][train].mean(0)
        time_baseline[test] = Ridge(alpha=1.).fit(time_features[train], data["y"][train]).predict(time_features[test])
        print(json.dumps({"event": "discovery_fold_complete", "held_seed": seed}), flush=True)
    chosen = choose_models(data, predictions)
    cv = evaluate(data, predictions, chosen)
    for name, cols in data["columns"].items():
        cv[name]["training_mean_baseline_rmse"] = float(np.sqrt(np.square(mean_baseline[:, cols]-data["y"][:, cols]).mean()))
        cv[name]["ten_bin_time_baseline_rmse"] = float(np.sqrt(np.square(time_baseline[:, cols]-data["y"][:, cols]).mean()))
        cv[name]["time_baseline_circular_for_time_target"] = name == "observed_episode_raw_step"
    bundle = fit_models(data["x"], data["y"])
    import joblib
    joblib.dump(bundle, args.output_dir/"readouts.joblib")
    candidate = freeze_candidate(data, bundle, chosen, cv, args.output_dir)
    frozen = {"chosen_models": chosen, "candidate": candidate,
              "readouts_sha256": sha256(args.output_dir/"readouts.joblib"),
              "discovery_results": cv, "discovery_sources": data["sources"], "confirmation_opened": False}
    write_json(args.output_dir/"FROZEN.json", frozen)
    print(json.dumps({"event": "candidate_frozen", **candidate}), flush=True)
    validation = load_split(args.capture_dirs, "validation")
    vp = predict_models(bundle, validation["x"])
    result = {"complete": True, "discovery": cv, "validation": evaluate(validation, vp, chosen),
              "candidate": candidate, "protocol": PROTOCOL, "confirmation_opened": False,
              "discovery_episodes": 150, "validation_episodes": 75,
              "goal_aligned_degenerate_rows": {"discovery": int((~data["frame_valid"]).sum()),
                                               "validation": int((~validation["frame_valid"]).sum())},
              "imagined_time_test": "awaiting separate multi-horizon capture; legacy horizon is constant1",
              "object_frame_test": "awaiting actual object pose capture; not fabricated from fixed wall constants",
              "sources": validation["sources"], "native_coordinate_claim": False}
    np.savez_compressed(args.output_dir/"validation_predictions.npz", **vp,
                        targets=validation["y"], episode=validation["episode"], seed=validation["seed"])
    write_json(args.output_dir/"coordinate_time_results.json", result)
    write_json(args.output_dir/"DONE.json", {"complete": True, "outputs": [
        {"path": p.name, "sha256": sha256(p)} for p in sorted(args.output_dir.iterdir()) if p.is_file()]})
    print(json.dumps({"event": "validation_complete", "results": result["validation"]}), flush=True)


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        main()
