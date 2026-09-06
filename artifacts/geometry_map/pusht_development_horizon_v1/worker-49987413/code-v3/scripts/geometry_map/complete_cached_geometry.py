#!/usr/bin/env python3
"""Bounded discovery-only geometry diagnostics; CPU execution, no causal claims.

Only DONE entries for seed 1/2/3 and episode 0..49 are selected, hashed, and
then deserialized. All compared subspaces use one fixed discovery mean/std per
site. Predictive comparisons instead fit their metric on training seeds only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SITES = (("encoder", 6), ("predictor", 2), ("predictor", 3), ("predictor", 5))
CORE = {"step_progress": slice(0, 1), "wall_clearance": slice(1, 2),
        "realized_xz_direction": slice(2, 4)}
EPS = 1e-8


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def normalized_direction(value, epsilon=EPS):
    value = np.asarray(value, dtype=np.float64)
    norm = np.linalg.norm(value, axis=-1)
    zero = norm <= epsilon
    direction = np.divide(value, norm[..., None], out=np.zeros_like(value),
                          where=~zero[..., None])
    return direction, zero, norm


def corrected_labels(labels):
    """State t labels align with the following five-raw-action transition."""
    arrays = {key: np.asarray(value, dtype=np.float64) for key, value in labels.items()}
    distance = arrays["goal_distance"]
    if len(distance) != 20 or not np.all(np.diff(arrays["frame_index"]) == 5):
        raise ValueError("Expected 20 sampled states separated by five raw actions")
    action_direction, action_zero, action_norm = normalized_direction(arrays["action_translation_sum"])
    motion_direction, motion_zero, motion_norm = normalized_direction(arrays["realized_hand_delta"])
    xz_direction, xz_zero, _ = normalized_direction(arrays["realized_hand_delta"][:, [0, 2]], 1e-6)
    return {
        "step_progress": distance[:-1] - distance[1:],
        "cumulative_progress": distance[0] - distance[:-1],
        "cumulative_progress_after_step": distance[0] - distance[1:],
        "goal_distance": distance[:-1], "goal_vector": arrays["goal_vector"][:-1],
        "wall_clearance": arrays["wall_signed_distance"][:-1],
        "height_above_wall_top": arrays["height_above_wall_top"][:-1],
        "time_fraction": arrays["time_fraction"][:-1],
        "frame_index": arrays["frame_index"][:-1],
        "action_direction": action_direction, "action_zero_net_mask": action_zero,
        "action_net_norm": action_norm,
        "action_chunk_frobenius_magnitude": arrays["action_translation_magnitude"],
        "realized_motion_world": arrays["realized_hand_delta"],
        "realized_direction_3d": motion_direction, "realized_motion_zero_mask": motion_zero,
        "realized_motion_magnitude": motion_norm,
        "realized_xz_direction": xz_direction, "realized_xz_zero_mask": xz_zero,
    }


def selected_entries(done):
    if done.get("complete") is not True:
        raise ValueError("Incomplete capture receipt")
    return [item for item in done["outputs"]
            if int(item["seed"]) in (1, 2, 3) and 0 <= int(item["episode"]) < 50]


def load_discovery(directories):
    import torch
    features = {f"{family}:{layer}": [] for family, layer in SITES}
    labels, seeds, episodes, sources = {}, [], [], []
    seen = set()
    for directory in directories:
        done_path = directory / "DONE.json"
        done = json.loads(done_path.read_text())
        selected = selected_entries(done)  # Filter BEFORE hashing/loading tensor files.
        source = {"done": str(done_path), "sha256": sha256(done_path), "selected_outputs": []}
        for item in selected:
            key = int(item["seed"]), int(item["episode"])
            if key in seen:
                raise ValueError(f"Duplicate discovery episode {key}")
            path = directory / item["path"]
            if not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError("Output path escapes its capture directory")
            actual_sha = sha256(path)
            if actual_sha != item["sha256"]:
                raise ValueError(f"Hash mismatch before torch.load: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            meta = payload["meta"]
            if (int(meta["seed"]), int(meta["episode"])) != key or meta["split"] != "discovery":
                raise ValueError(f"Receipt/payload split mismatch: {path}")
            physical = corrected_labels({k: v.cpu().numpy() for k, v in payload["labels"].items()})
            for family, layer in SITES:
                block_ids = payload.get("block_ids", {}).get(family, list(range(12 if family == "encoder" else 6)))
                index = block_ids.index(layer)
                value = payload["features"][family][:19, index].float().numpy()
                if value.ndim != 2 or value.shape[0] != 19 or not np.isfinite(value).all():
                    raise ValueError("Expected finite pooled features [19, channels]")
                features[f"{family}:{layer}"].append(value)
            for name, value in physical.items():
                labels.setdefault(name, []).append(value)
            seeds.extend([key[0]] * 19)
            episodes.extend([key[1]] * 19)
            seen.add(key)
            source["selected_outputs"].append({"seed": key[0], "episode": key[1],
                                                "path": item["path"], "sha256": actual_sha})
        sources.append(source)
    if seen != {(seed, episode) for seed in (1, 2, 3) for episode in range(50)}:
        raise ValueError(f"Expected all 150 discovery trajectories; found {len(seen)}")
    return {"features": {k: np.concatenate(v).astype(np.float64) for k, v in features.items()},
            "labels": {k: np.concatenate(v) for k, v in labels.items()},
            "seeds": np.asarray(seeds), "episodes": np.asarray(episodes), "sources": sources}


def common_metric(x):
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def ridge_fit(x, y, alpha=100.0):
    y = np.asarray(y).reshape(len(x), -1)
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - x_mean, y - y_mean
    gram = xc.T @ xc
    gram.flat[:: len(gram) + 1] += alpha
    weights = np.linalg.solve(gram, xc.T @ yc)
    return weights, y_mean - x_mean @ weights


def r2(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    total = float(np.square(y - y.mean(axis=0)).sum())
    return None if total <= 1e-20 else float(1 - np.square(y - prediction).sum() / total)


def basis(weights):
    weights = np.asarray(weights).reshape(len(weights), -1)
    u, singular, _ = np.linalg.svd(weights, full_matrices=False)
    rank = int(np.sum(singular > max(float(singular.max(initial=0)), 1e-12) * 1e-8))
    return u[:, :rank]


def subspace_comparison(weights_a, weights_b):
    """Inputs must already share the SAME activation metric; never rescale here."""
    a, b = basis(weights_a), basis(weights_b)
    ra, rb, d = a.shape[1], b.shape[1], a.shape[0]
    if not ra or not rb:
        return {"status": "zero_rank", "rank_a": ra, "rank_b": rb}
    singular = np.clip(np.linalg.svd(a.T @ b, compute_uv=False), 0, 1)
    angles = np.arccos(singular)
    overlap = float(np.square(singular).sum())
    return {"status": "computed", "rank_a": ra, "rank_b": rb,
            "principal_angles_deg": np.degrees(angles).tolist(),
            "projection_overlap_trace": overlap,
            "a_projected_into_b_fraction": overlap / ra,
            "b_projected_into_a_fraction": overlap / rb,
            "random_a_into_b_fraction": rb / d,
            "random_b_into_a_fraction": ra / d,
            "grassmann_geodesic_rad": float(np.linalg.norm(angles)) if ra == rb else None,
            "projector_distance_frobenius": float(np.sqrt(max(ra + rb - 2 * overlap, 0))),
            "unequal_rank_note": "Principal angles cover only min(rank_a,rank_b); projector distance includes rank mismatch."}


def goal_basis(goal_vectors):
    goal_axis, zero_goal, _ = normalized_direction(goal_vectors)
    up = np.broadcast_to(np.array([0., 0., 1.]), goal_axis.shape)
    lateral, parallel_up, _ = normalized_direction(np.cross(up, goal_axis))
    vertical = np.cross(goal_axis, lateral)
    rotation = np.stack([goal_axis, lateral, vertical], axis=-1)
    valid = ~(zero_goal | parallel_up)
    return rotation, valid, {"zero_goal_distance_count": int(zero_goal.sum()),
                             "goal_parallel_to_up_count": int((parallel_up & ~zero_goal).sum())}


def to_local(world, rotation):
    return np.einsum("nij,ni->nj", rotation, world)


def to_world(local, rotation):
    return np.einsum("nij,nj->ni", rotation, local)


def held_seed_prediction(x, y, seeds, mask, alpha):
    y = np.asarray(y).reshape(len(x), -1)
    prediction = np.full_like(y, np.nan)
    folds = []
    for held in (1, 2, 3):
        # Each fold metric is shared across target masks; only training seeds fit it.
        mean, scale = common_metric(x[seeds != held])
        z = (x - mean) / scale
        train, test = (seeds != held) & mask, (seeds == held) & mask
        weights, intercept = ridge_fit(z[train], y[train], alpha)
        prediction[test] = z[test] @ weights + intercept
        folds.append({"held_seed": held, "train_samples": int(train.sum()),
                      "test_samples": int(test.sum()), "score": r2(y[test], prediction[test])})
    return prediction, folds


def summarize_regimes(truth, prediction, data, mask, time_prediction):
    truth = np.asarray(truth).reshape(len(mask), -1)
    height, time = data["labels"]["height_above_wall_top"], data["labels"]["time_fraction"]
    rows = []
    for name, physical in (("below_wall_top", height < 0), ("at_or_above_wall_top", height >= 0)):
        for seed in (None, 1, 2, 3):
            selected = mask & physical
            if seed is not None:
                selected &= data["seeds"] == seed
            count = int(selected.sum())
            row = {"regime": name, "seed": seed, "samples": count,
                   "episodes": len(set(zip(data["seeds"][selected].tolist(), data["episodes"][selected].tolist())))}
            if count:
                row.update({"global_held_seed_mse": float(np.square(truth[selected] - prediction[selected]).mean()),
                            "time_only_mse": float(np.square(truth[selected] - time_prediction[selected]).mean()),
                            "mean_time": float(time[selected].mean()),
                            "time_quantiles": np.quantile(time[selected], [.1, .5, .9]).tolist()})
            rows.append(row)
    return {"status": "descriptive_physical_regimes", "groups": rows,
            "time_confounding": "Regimes have different time and trajectory composition; residual differences do not identify causal regimes or HMM states."}


def coordinate_frames(x, data, alpha):
    labels, seeds = data["labels"], data["seeds"]
    world = labels["realized_motion_world"]
    rotation, valid, degeneracy = goal_basis(labels["goal_vector"])
    wall = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    predictions = {name: np.full_like(world, np.nan) for name in ("world", "goal_aligned", "wall_constant_rotation")}
    folds = []
    for held in (1, 2, 3):
        mean, scale = common_metric(x[seeds != held])
        z = (x - mean) / scale
        train, test = (seeds != held) & valid, (seeds == held) & valid
        target_values = {"world": world, "goal_aligned": to_local(world, rotation),
                         "wall_constant_rotation": world @ wall}
        for name, targets in target_values.items():
            weights, intercept = ridge_fit(z[train], targets[train], alpha)
            local = z[test] @ weights + intercept
            predicted = to_world(local, rotation[test]) if name == "goal_aligned" else local @ wall.T if name == "wall_constant_rotation" else local
            predictions[name][test] = predicted
            folds.append({"held_seed": held, "frame": name, "train_samples": int(train.sum()),
                          "test_samples": int(test.sum()), "world_mse": float(np.square(world[test] - predicted).mean()),
                          "world_r2": r2(world[test], predicted)})
    scores = {name: {"world_mse": float(np.square(world[valid] - pred[valid]).mean()),
                     "world_r2": r2(world[valid], pred[valid])} for name, pred in predictions.items()}
    difference = np.square(world - predictions["world"]).mean(axis=1) - np.square(world - predictions["goal_aligned"]).mean(axis=1)
    episode_differences = []
    for seed in (1, 2, 3):
        for episode in range(50):
            selected = valid & (seeds == seed) & (data["episodes"] == episode)
            if selected.any():
                episode_differences.append({"seed": seed, "episode": episode,
                                            "world_minus_goal_world_mse": float(difference[selected].mean())})
    return {"status": "held_collection_seed_comparison", "scores": scores, "folds": folds,
            "evaluation_target": "Identical realized 3D world displacement and squared Euclidean world loss for every frame",
            "goal_basis_definition": "Columns: normalized current goal-minus-hand, normalized cross(world_up,goal_axis), cross(goal_axis,lateral)",
            "goal_frame_availability": "Requires current true hand/goal positions and world-up; cached simulator labels supply these. Runtime visual-only availability has not been established.",
            "degeneracy": {**degeneracy, "valid_samples": int(valid.sum()), "excluded_samples": int((~valid).sum())},
            "wall_rotation_max_prediction_difference": float(np.abs(predictions["world"][valid] - predictions["wall_constant_rotation"][valid]).max()),
            "wall_frame_interpretation": "A constant orthogonal target rotation is algebraically equivalent under isotropic ridge; it cannot identify a native coordinate frame.",
            "camera_frame": {"status": "unavailable", "reason": "No camera extrinsic calibration in pooled cache"},
            "native_frame": {"status": "not_identified", "reason": "Decoding compares functional parameterizations; state-dependent goal basis uses extra geometry."},
            "episode_paired_world_mse_differences": episode_differences}


def bootstrap_subspaces(z, core_y, core_mask, data, alpha, draws):
    weights, _ = ridge_fit(z[core_mask], core_y[core_mask], alpha)
    rng = np.random.default_rng(1729)
    references = {name: weights[:, columns] for name, columns in CORE.items()}
    records = {name: [] for name in CORE}
    comparisons = []
    pairs = [(a, b) for i, a in enumerate(CORE) for b in list(CORE)[i + 1:]]
    for a, b in pairs:
        comparisons.append({"target_a": a, "target_b": b, **subspace_comparison(references[a], references[b])})
    groups = {(seed, ep): np.flatnonzero((data["seeds"] == seed) & (data["episodes"] == ep) & core_mask)
              for seed in (1, 2, 3) for ep in range(50)}
    for draw in range(draws):
        # Stratified whole-episode resampling preserves repeated timesteps together.
        indices = np.concatenate([groups[(seed, int(ep))] for seed in (1, 2, 3)
                                  for ep in rng.integers(0, 50, size=50)])
        boot_weights, _ = ridge_fit(z[indices], core_y[indices], alpha)
        for name, columns in CORE.items():
            records[name].append({"draw": draw, **subspace_comparison(references[name], boot_weights[:, columns])})
    stability = {}
    for name, values in records.items():
        max_angles = [max(value["principal_angles_deg"]) for value in values if value["status"] == "computed"]
        stability[name] = {"draws": values,
                           "max_angle_deg_quantiles_025_50_975": np.quantile(max_angles, [.025, .5, .975]).tolist() if max_angles else None,
                           "sampling": "30 fixed-seed whole-episode bootstrap draws stratified by collection seed; fixed full-discovery feature metric",
                           "interpretation": "Discovery resampling reproducibility relative to the full-data fit, not independent confirmation."}
    return comparisons, stability


def regime_subspaces(z, core_y, core_mask, data, alpha):
    below = data["labels"]["height_above_wall_top"] < 0
    masks = {"below_wall_top": core_mask & below, "at_or_above_wall_top": core_mask & ~below}
    weights = {}
    counts = {}
    for regime, selected in masks.items():
        counts[regime] = {"samples": int(selected.sum()), "by_seed": []}
        for seed in (1, 2, 3):
            use = selected & (data["seeds"] == seed)
            counts[regime]["by_seed"].append({"seed": seed, "samples": int(use.sum()),
                                               "episodes": int(np.unique(data["episodes"][use]).size)})
        if selected.sum() >= 5:
            weights[regime], _ = ridge_fit(z[selected], core_y[selected], alpha)
    rows = []
    for target, columns in CORE.items():
        row = {"target": target, "status": "insufficient_samples"}
        if len(weights) == 2:
            row.update(subspace_comparison(weights["below_wall_top"][:, columns], weights["at_or_above_wall_top"][:, columns]))
        rows.append(row)
    return {"counts": counts, "comparisons": rows,
            "metric": "Same fixed full-discovery mean/std as global subspace comparisons",
            "interpretation": "Descriptive conditional fits; unequal sample sizes, time, and trajectory occupancy confound comparison. No latent-state/HMM inference."}


def natural_support(x, data):
    distances = np.empty(len(x))
    neighbor_indices = np.empty(len(x), dtype=int)
    folds = []
    for held in (1, 2, 3):
        train_idx, test_idx = np.flatnonzero(data["seeds"] != held), np.flatnonzero(data["seeds"] == held)
        mean, scale = common_metric(x[train_idx])
        train = (x[train_idx] - mean) / scale
        test = (x[test_idx] - mean) / scale
        train_sq = np.square(train).sum(axis=1)
        for offset in range(0, len(test), 128):
            block = test[offset:offset + 128]
            squared = np.maximum(np.square(block).sum(axis=1)[:, None] + train_sq[None, :] - 2 * block @ train.T, 0)
            nearest = squared.argmin(axis=1)
            rows = test_idx[offset:offset + len(block)]
            distances[rows] = np.sqrt(squared[np.arange(len(block)), nearest] / x.shape[1])
            neighbor_indices[rows] = train_idx[nearest]
        folds.append({"held_seed": held, "train_samples": len(train_idx), "test_samples": len(test_idx),
                      "rms_standardized_distance_quantiles_05_50_95": np.quantile(distances[test_idx], [.05, .5, .95]).tolist()})
    return {"status": "natural_state_support_proxy", "distance": "Euclidean distance / sqrt(channel_count), standardized with training-seed mean/std",
            "folds": folds, "sample_distances": distances.tolist(),
            "nearest_training_sample_index": neighbor_indices.tolist(),
            "interpretation": "Held-seed proximity to natural training activations; not a manifold proof or an assessment of edited states.",
            "edited_state_distance": {"status": "missing", "reason": "No causal edited activations supplied to this cached-data analysis"}}


def eigenspectrum_and_trajectories(x, z, data):
    raw_covariance = np.cov(x, rowvar=False)
    raw_eigenvalues = np.maximum(np.linalg.eigvalsh(raw_covariance)[::-1], 0)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(z, rowvar=False))
    eigenvalues, eigenvectors = np.maximum(eigenvalues[::-1], 0), eigenvectors[:, ::-1]
    # Deterministic sign convention, fixed before examining the three chosen traces.
    loadings = eigenvectors[:, :3].copy()
    for column in range(3):
        pivot = np.abs(loadings[:, column]).argmax()
        loadings[:, column] *= 1 if loadings[pivot, column] >= 0 else -1
    explained = eigenvalues / eigenvalues.sum()
    traces = []
    for episode in (0, 1, 2):
        selected = (data["seeds"] == 1) & (data["episodes"] == episode)
        traces.append({"seed": 1, "episode": episode, "sample_index": np.flatnonzero(selected).tolist(),
                       "coordinates": (z[selected] @ loadings).tolist(),
                       **{name: data["labels"][name][selected].tolist() for name in
                          ("frame_index", "time_fraction", "step_progress", "cumulative_progress", "goal_distance", "height_above_wall_top")}})
    return ({"raw_activation_covariance_eigenvalues": raw_eigenvalues.tolist(),
             "standardized_activation_covariance_eigenvalues": eigenvalues.tolist(),
             "standardized_cumulative_variance_fraction": np.cumsum(explained).tolist(),
             "participation_ratio": float(eigenvalues.sum() ** 2 / np.square(eigenvalues).sum()),
             "sample_count": len(x), "channel_count": x.shape[1],
             "interpretation": "Actual activation covariance spectrum, not target code dimension or intervention rank"},
            {"fit_scope": "All 150 discovery episodes only; descriptive PCA, not held-out prediction",
             "selection": "Predetermined seed1 episodes0,1,2; no outcome selection",
             "loadings": loadings.tolist(), "pc_variance_fraction": explained[:3].tolist(), "trajectories": traces})


def run_analysis(data, alpha, draws, output_dir):
    labels, seeds = data["labels"], data["seeds"]
    time = labels["time_fraction"]
    time_x = np.stack([time, time ** 2, time ** 3], axis=1)
    target_names = ("step_progress", "cumulative_progress", "wall_clearance", "realized_xz_direction",
                    "realized_motion_world", "action_direction", "action_net_norm", "action_chunk_frobenius_magnitude")
    masks = {name: np.ones(len(seeds), dtype=bool) for name in target_names}
    masks["realized_xz_direction"] = ~labels["realized_xz_zero_mask"]
    masks["action_direction"] = ~labels["action_zero_net_mask"]
    time_predictions = {name: held_seed_prediction(time_x, labels[name], seeds, masks[name], 1.0)[0] for name in target_names}
    core_y = np.column_stack([labels["step_progress"], labels["wall_clearance"], labels["realized_xz_direction"]])
    core_mask = ~labels["realized_xz_zero_mask"]
    rows, sites = [], []
    for family, layer in SITES:
        print(json.dumps({"event": "site_started", "family": family, "layer": layer}), flush=True)
        x = data["features"][f"{family}:{layer}"]
        mean, scale = common_metric(x)
        z = (x - mean) / scale
        comparisons, stability = bootstrap_subspaces(z, core_y, core_mask, data, alpha, draws)
        for target in target_names:
            mask = masks[target]
            truth = labels[target].reshape(len(x), -1)
            prediction, folds = held_seed_prediction(x, truth, seeds, mask, alpha)
            score = r2(truth[mask], prediction[mask])
            time_score = r2(truth[mask], time_predictions[target][mask])
            rows.append({"family": family, "layer": layer, "target": target, "hook": "block_residual_output",
                         "spatial_region": "mean_spatial_tokens", "imagined_timestep": 1 if family == "predictor" else None,
                         "metric": "r2", "score": score, "time_baseline": time_score,
                         "delta_vs_time": score - time_score if score is not None and time_score is not None else None,
                         "samples": int(mask.sum()), "folds": folds,
                         "regime_dependence": summarize_regimes(truth, prediction, data, mask, time_predictions[target]),
                         "bootstrap_stability": stability.get(target, {"status": "not_run", "reason": "Focused bootstrap covers progress, clearance and XZ direction"}),
                         "causal_edit": {"status": "missing", "reason": "Cached natural activations only"},
                         "accepted_operator": None})
        spectrum, trajectories = eigenspectrum_and_trajectories(x, z, data)
        sites.append({"family": family, "layer": layer,
                      "feature_metric": {"description": "One mean/std fitted on all 150 discovery episodes, fixed across every subspace target, regime and bootstrap draw",
                                         "mean": mean.tolist(), "std": scale.tolist()},
                      "subspace_comparisons": comparisons,
                      "subspace_fit_samples": int(core_mask.sum()),
                      "subspace_fit_mask": "Common nonzero-realized-XZ-motion rows for all three heads; metric fitted on all discovery rows",
                      "coordinate_frames": coordinate_frames(x, data, alpha),
                      "regime_subspaces": regime_subspaces(z, core_y, core_mask, data, alpha),
                      "natural_support": natural_support(x, data),
                      "eigenspectrum": spectrum, "pca_trajectories": trajectories})
        write_json(output_dir / f"site-{family}-{layer}.json", {"complete": True, "site": sites[-1],
                                                               "rows": [row for row in rows if row["family"] == family and row["layer"] == layer]})
        print(json.dumps({"event": "site_complete", "family": family, "layer": layer}), flush=True)
    return {"schema_version": 2, "created_utc": datetime.now(timezone.utc).isoformat(), "complete": True,
            "scope": "150 discovery episodes: IDs0..49 in each collection seed1,2,3; validation and confirmation tensors never opened",
            "sample_count": len(seeds), "episode_count": 150, "sources": data["sources"],
            "method": {"ridge_alpha": alpha, "bootstrap_draws": draws, "bootstrap_seed": 1729,
                       "cross_validation": "Held collection seed, with metric fit on training seeds only",
                       "multioutput_r2": "1 - summed squared world/target error divided by summed per-coordinate centered target variation",
                       "inference": "Discovery descriptive analysis; no multiplicity-adjusted confirmatory claims"},
            "label_definitions": {
                "step_progress": "goal_distance[t] - goal_distance[t+1], across five raw actions, positive toward goal",
                "cumulative_progress": "goal_distance[0] - goal_distance[t], before the corresponding transition",
                "cumulative_progress_after_step": "goal_distance[0] - goal_distance[t+1]",
                "action_direction": "sum of five raw XYZ translations divided by its L2 norm; zero-net rows stored as zero and masked from readouts",
                "action_net_norm": "L2 norm of the sum of five raw XYZ translations",
                "action_chunk_frobenius_magnitude": "sqrt(sum over five actions and XYZ coordinates of action squared); differs from net norm under cancellation",
                "realized_xz_direction": "[delta_x,delta_z]/sqrt(delta_x^2+delta_z^2), equivalent to [cos(theta),sin(theta)]; magnitude<=1e-6 excluded",
                "wall_clearance": "Cached signed distance from end effector to the wall box",
                "physical_regime": "height_above_wall_top < 0 versus >= 0 at current state"},
            "rows": rows, "sites": sites,
            "sample_arrays": {"seed": seeds.tolist(), "episode": data["episodes"].tolist(),
                              **{name: value.tolist() for name, value in labels.items()}},
            "missing_evidence": {"edited_state_manifold_distance": "No edited states in this input",
                                 "causal_specificity": "Requires separate matched causal runner outputs",
                                 "camera_coordinate_frame": "No camera calibration in pooled capture",
                                 "operator": "No operator accepted by observational geometry"},
            "accepted_operator": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    parser.add_argument("--bootstrap-draws", type=int, default=30)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--nonlinear-report", type=Path)
    parser.add_argument("--nonlinear-remote-sha256")
    args = parser.parse_args()
    if args.bootstrap_draws < 1 or args.ridge_alpha <= 0:
        parser.error("Positive ridge alpha and bootstrap count required")
    data = load_discovery(args.capture_dirs)
    stage = {"complete": True, "stage": "discovery_files_verified_and_labels_derived", "episodes": 150,
             "samples": len(data["seeds"]), "sources": data["sources"],
             "action_zero_net_rows": int(data["labels"]["action_zero_net_mask"].sum()),
             "realized_xz_zero_rows": int(data["labels"]["realized_xz_zero_mask"].sum())}
    write_json(args.output_dir / "input_verification.json", stage)
    print(json.dumps({key: value for key, value in stage.items() if key != "sources"}), flush=True)
    if args.validate_only:
        return
    report = run_analysis(data, args.ridge_alpha, args.bootstrap_draws, args.output_dir)
    if args.nonlinear_report:
        nonlinear_sha = sha256(args.nonlinear_report)
        if nonlinear_sha != args.nonlinear_remote_sha256:
            raise ValueError("Copied nonlinear report differs from its fresh remote SHA")
        report["existing_nonlinear_report"] = {
            "source_instance": 49902461,
            "source_path": "/root/geometry-map-jepawm-reach-wall-v1/canary/nonlinear-screen/nonlinear_screen.json",
            "local_filename": "nonlinear_screen.json", "remote_sha256": nonlinear_sha,
            "copied_sha256": nonlinear_sha, "verified": True, "rerun": False,
            "target_caveat": "Legacy progress means cumulative progress; nonlinear gain does not directly describe the corrected step_progress target."}
    output = args.output_dir / "cached_geometry.json"
    write_json(output, report)
    outputs = [output, args.output_dir / "input_verification.json"] + sorted(args.output_dir.glob("site-*.json"))
    write_json(args.output_dir / "DONE.json", {"complete": True, "schema_version": 2,
               "created_utc": datetime.now(timezone.utc).isoformat(), "episode_count": 150,
               "outputs": [{"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size} for path in outputs],
               "script_sha256": sha256(Path(__file__))})
    print(json.dumps({"event": "complete", "output": str(output), "rows": len(report["rows"]),
                      "sites": len(report["sites"]), "sha256": sha256(output)}), flush=True)


if __name__ == "__main__":
    main()
