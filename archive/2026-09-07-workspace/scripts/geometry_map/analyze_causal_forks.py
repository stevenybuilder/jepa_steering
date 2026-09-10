#!/usr/bin/env python3
"""Discovery-only frozen physical readout and completed causal-fork measurements.

Decoded XYZ is a probe prediction, never native model XYZ or simulator truth.
The discovery split and SHA-256 are checked before opening every tensor file.
No model execution, GPU operations, tuning, or confirmation data are used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge

from protocol import WALL_CENTER, WALL_HALF_SIZE, file_sha256, segment_intersects_box, write_json_atomic


ARMS = ("unsteered", "identity", "subspace_up", "sham_up", "subspace_down", "sham_down")
RIDGE_ALPHA = 100.0
UNITS = "simulator world-coordinate units (meters in the MuJoCo task)"


def array(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().float().numpy()
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all():
        raise RuntimeError("Nonfinite measurement input")
    return value


def load_receipt(directory):
    path = directory / "DONE.json"
    receipt = json.loads(path.read_text())
    if receipt.get("complete") is not True:
        raise RuntimeError(f"Incomplete receipt: {path}")
    return receipt


def checked_load(directory, entry):
    path = directory / entry["path"]
    if path.parent.resolve() != directory.resolve():
        raise RuntimeError("Receipt path escapes its directory")
    if file_sha256(path) != entry["sha256"]:
        raise RuntimeError(f"Hash mismatch before tensor load: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def load_discovery(directories, require_complete=True):
    trajectories, sources, seen = [], [], set()
    for directory in directories:
        receipt = load_receipt(directory)
        sources.append({"done": str(directory / "DONE.json"), "sha256": file_sha256(directory / "DONE.json")})
        for entry in receipt["outputs"]:
            # Receipt metadata controls access before hashing OR deserializing.
            episode, seed = int(entry["episode"]), int(entry["seed"])
            if not 0 <= episode < 50:
                continue
            if seed not in (1, 2, 3) or (seed, episode) in seen:
                raise RuntimeError("Unexpected seed or duplicate discovery trajectory")
            payload = checked_load(directory, entry)
            meta = payload["meta"]
            if meta["split"] != "discovery" or (int(meta["seed"]), int(meta["episode"])) != (seed, episode):
                raise RuntimeError("Discovery payload disagrees with prechecked receipt")
            if payload["mode"] != "pooled":
                raise RuntimeError("Frozen decoder requires common spatially pooled visual features")
            x = array(payload["features"]["encoded_final"])
            future = array(payload["features"]["predicted_final"])
            xyz = array(payload["labels"]["hand_xyz"])
            if x.shape != (20, 384) or future.shape != (19, 384) or xyz.shape != (20, 3):
                raise RuntimeError("Unexpected natural/predicted feature or label layout")
            trajectories.append({"seed": seed, "episode": episode, "x": x, "future": future, "xyz": xyz,
                                 "sha256": entry["sha256"], "path": entry["path"]})
            seen.add((seed, episode))
    expected = {(seed, episode) for seed in (1, 2, 3) for episode in range(50)}
    if require_complete and seen != expected:
        raise RuntimeError(f"Expected all 150 discovery trajectories; found {len(seen)}")
    return trajectories, sources


def fit_readout(x, xyz):
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-6] = 1.0
    model = Ridge(alpha=RIDGE_ALPHA, solver="cholesky").fit((x - mean) / scale, xyz)
    return {"mean": mean, "scale": scale, "coef": model.coef_, "intercept": model.intercept_}


def decode(readout, x):
    return ((x - readout["mean"]) / readout["scale"]) @ readout["coef"].T + readout["intercept"]


def errors(prediction, truth):
    difference = prediction - truth
    return {"n": int(len(truth)), "rmse_xyz": float(np.sqrt(np.mean(difference ** 2))),
            "rmse_per_axis": np.sqrt(np.mean(difference ** 2, axis=0)).tolist(),
            "rms_euclidean_error": float(np.sqrt(np.mean(np.sum(difference ** 2, axis=-1)))),
            "mean_euclidean_error": float(np.linalg.norm(difference, axis=-1).mean()), "units": UNITS}


def fit_and_validate(trajectories):
    folds, all_natural, all_predicted, all_truth, all_next, all_mean, all_hold = [], [], [], [], [], [], []
    for seed in (1, 2, 3):
        train = [row for row in trajectories if row["seed"] != seed]
        test = [row for row in trajectories if row["seed"] == seed]
        if not train or not test:
            raise RuntimeError("Each held-collection-seed fold needs train and test trajectories")
        x = np.concatenate([row["x"] for row in train])
        y = np.concatenate([row["xyz"] for row in train])
        readout = fit_readout(x, y)
        truth = np.concatenate([row["xyz"] for row in test])
        next_truth = np.concatenate([row["xyz"][1:] for row in test])
        natural = decode(readout, np.concatenate([row["x"] for row in test]))
        predicted = decode(readout, np.concatenate([row["future"] for row in test]))
        mean_baseline = np.broadcast_to(y.mean(0), truth.shape)
        hold_baseline = np.concatenate([row["xyz"][:-1] for row in test])
        folds.append({"held_collection_seed": seed, "train_trajectories": len(train), "test_trajectories": len(test),
                      "normalization_fit": "training seeds only; shared across natural and predicted features",
                      "natural_state_decoding": errors(natural, truth),
                      "natural_train_mean_baseline": errors(mean_baseline, truth),
                      "unsteered_jepa_predicted_final_to_actual_next_hand": errors(predicted, next_truth),
                      "actual_hand_persistence_baseline": errors(hold_baseline, next_truth)})
        all_natural.append(natural); all_predicted.append(predicted); all_truth.append(truth)
        all_next.append(next_truth); all_mean.append(mean_baseline); all_hold.append(hold_baseline)
    x = np.concatenate([row["x"] for row in trajectories])
    xyz = np.concatenate([row["xyz"] for row in trajectories])
    readout = fit_readout(x, xyz)
    result = {"status": "discovery_cross_validation_only; no validation or confirmation tensors opened",
              "feature": "encoded_final spatial mean, 384 dimensions; native visual representation",
              "application_feature": "predicted_final / predicted_visual_pooled, same 384-dimensional spatial mean",
              "target": "simulator hand XYZ; natural t -> XYZ[t], teacher-forced predicted t -> XYZ[t+1]",
              "predicted_label": "probe-decoded prediction; not native XYZ output or validated ground truth",
              "ridge_alpha": RIDGE_ALPHA, "final_fit_trajectories": len(trajectories), "final_fit_states": len(x),
              "scaling": "feature mean/std fitted within each training fold; final readout fitted on all discovery states",
              "natural_state_decoding": errors(np.concatenate(all_natural), np.concatenate(all_truth)),
              "natural_train_mean_baseline": errors(np.concatenate(all_mean), np.concatenate(all_truth)),
              "unsteered_jepa_predicted_final_to_actual_next_hand": errors(np.concatenate(all_predicted), np.concatenate(all_next)),
              "actual_hand_persistence_baseline": errors(np.concatenate(all_hold), np.concatenate(all_next)),
              "folds": folds}
    return readout, result


def average_ranks(values):
    values = np.asarray(values)
    _unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts + 1) / 2)[inverse]


def cost_metrics(costs, baseline):
    ranks, base_ranks = average_ranks(costs), average_ranks(baseline)
    rank_delta = ranks - base_ranks
    if np.std(ranks) == 0 or np.std(base_ranks) == 0:
        correlation = None
    else:
        correlation = float(np.corrcoef(ranks, base_ranks)[0, 1])
    base_best, best = int(baseline.argmin()), int(costs.argmin())
    top = min(30, len(costs))
    return {"candidate_count": len(costs), "argmin": best, "unsteered_argmin": base_best,
            "argmin_changed": best != base_best, "rank_changes": int(np.count_nonzero(rank_delta)),
            "mean_absolute_rank_change": float(np.abs(rank_delta).mean()),
            "maximum_absolute_rank_change": float(np.abs(rank_delta).max()), "spearman_rank_correlation": correlation,
            "cost_l2_change": float(np.linalg.norm(costs - baseline)), "cost_mean_change": float((costs-baseline).mean()),
            "cost_max_absolute_change": float(np.abs(costs-baseline).max()),
            "unsteered_best_new_rank": float(ranks[base_best]), "new_best_unsteered_rank": float(base_ranks[best]),
            "top30_overlap_count": len(set(np.argsort(costs)[:top]) & set(np.argsort(baseline)[:top])),
            "costs": costs.tolist(), "cost_changes": (costs-baseline).tolist(),
            "ranks": ranks.tolist(), "rank_changes_by_candidate": rank_delta.tolist(),
            "scope": "identical initial CEM candidates only; later iteration costs/ranks not saved"}


def vector_summary(vector):
    norm = float(np.linalg.norm(vector))
    return {"xyz": vector.tolist(), "norm": norm, "unit_direction_xyz": (vector / norm).tolist() if norm > 1e-12 else None}


def action_metrics(actions, baseline):
    if actions.shape != baseline.shape or actions.ndim != 2 or actions.shape[1] != 4:
        raise RuntimeError("Raw action pairing/layout mismatch")
    xyz, base_xyz = actions[:, :3], baseline[:, :3]
    chunks = []
    for start in range(0, len(actions), 5):
        chunk, base = xyz[start:start+5], base_xyz[start:start+5]
        chunks.append({"model_step": start // 5, "raw_steps": len(chunk),
                       "translation_frobenius_norm": float(np.linalg.norm(chunk)),
                       "translation_net": vector_summary(chunk.sum(0)),
                       "translation_net_change": vector_summary((chunk-base).sum(0)),
                       "all_action_l2_change": float(np.linalg.norm(actions[start:start+5]-baseline[start:start+5]))})
    return {"units": "raw simulator action command units; not physical displacement",
            "raw_actions": actions.tolist(), "all_action_l2_norm": float(np.linalg.norm(actions)),
            "all_action_l2_change": float(np.linalg.norm(actions-baseline)),
            "translation_frobenius_norm": float(np.linalg.norm(xyz)),
            "translation_net": vector_summary(xyz.sum(0)),
            "translation_net_change": vector_summary((xyz-base_xyz).sum(0)),
            "gripper_l2_change": float(np.linalg.norm(actions[:, 3]-baseline[:, 3])), "chunks": chunks}


def path_crossings(points, goal):
    result = {"geometry": "piecewise-linear hand-center point path in fixed axis-aligned wall box",
              "body_collision_validated": False}
    for margin in (0.0, 0.03):
        key = "point_margin_0" if margin == 0 else "point_margin_0_03"
        actual = segment_intersects_box(points[:-1], points[1:], margin=margin)
        proxy = segment_intersects_box(points, np.broadcast_to(goal, points.shape), margin=margin)
        result[key] = {"margin": margin, "segment_intersects": actual.tolist(), "any_segment_intersects": bool(actual.any()),
                       "hand_to_goal_straight_line_proxy": proxy.tolist(),
                       "interpretation": "point-path intersection" if margin == 0 else "inflated-box geometric margin proxy; not measured hand/body collision"}
    return result


def realized_metrics(fork, baseline, start, goal):
    states, base_states = array(fork["states"]), array(baseline["states"])
    points = np.concatenate([start[None], states[:, :3]])
    base_points = np.concatenate([start[None], base_states[:, :3]])
    if len(points) != len(base_points):
        raise RuntimeError("Unequal physical fork lengths require explicit censoring")
    progress = np.linalg.norm(start-goal) - np.linalg.norm(points-goal, axis=-1)
    base_progress = np.linalg.norm(start-goal) - np.linalg.norm(base_points-goal, axis=-1)
    return {"observed": True, "units": UNITS, "raw_steps": len(states), "hand_xyz_path_including_start": points.tolist(),
            "raw_step_index_including_start": list(range(len(points))), "goal_xyz": goal.tolist(),
            "displacement_xyz": (points-start).tolist(), "endpoint_displacement": vector_summary(points[-1]-start),
            "endpoint_change_from_unsteered": vector_summary(points[-1]-base_points[-1]),
            "goal_progress": progress.tolist(), "goal_progress_change_from_unsteered": (progress-base_progress).tolist(),
            "end_goal_distance": float(np.linalg.norm(points[-1]-goal)),
            "short_fork_end_success": bool(fork["end_success"]),
            "model_step_hand_xyz": points[5::5].tolist(), "crossing": path_crossings(points, goal)}


def pool_native_prediction(prediction, future_steps, candidates):
    visual = array(prediction["visual"])
    if visual.shape[:2] != (future_steps + 1, candidates) or visual.shape[-1] != 384:
        raise RuntimeError(f"Expected native visual [context+{future_steps},{candidates},...,384], got {visual.shape}")
    return visual[1:].reshape(future_steps, candidates, -1, 384).mean(2)


def prediction_arrays(xyz, start, goal):
    starts = np.broadcast_to(start, (1, xyz.shape[1], 3))
    path = np.concatenate([starts, xyz])
    result = {"decoded_hand_xyz": xyz, "decoded_displacement_xyz": xyz-start,
              "decoded_step_displacement_xyz": np.diff(path, axis=0),
              "decoded_goal_progress": np.linalg.norm(start-goal) - np.linalg.norm(xyz-goal, axis=-1)}
    for margin, key in ((0.0, "point_margin_0"), (0.03, "point_margin_0_03")):
        result[f"candidate_path_intersects_{key}"] = segment_intersects_box(path[:-1].reshape(-1, 3), path[1:].reshape(-1, 3), margin).reshape(xyz.shape[:2])
        result[f"hand_to_goal_proxy_intersects_{key}"] = segment_intersects_box(xyz.reshape(-1, 3), np.broadcast_to(goal, xyz.shape).reshape(-1, 3), margin).reshape(xyz.shape[:2])
    return result


def future_timelines(arm, latents, baseline_latents, readout, start, goal, costs, baseline_costs):
    xyz, baseline_xyz = decode(readout, latents), decode(readout, baseline_latents)
    values, base = prediction_arrays(xyz, start, goal), prediction_arrays(baseline_xyz, start, goal)
    latent_delta = latents-baseline_latents
    values.update({"latent_l2_change": np.linalg.norm(latent_delta, axis=-1),
                   "latent_standardized_l2_change": np.linalg.norm(latent_delta/readout["scale"], axis=-1),
                   "decoded_displacement_change": xyz-baseline_xyz,
                   "decoded_goal_progress_change": values["decoded_goal_progress"]-base["decoded_goal_progress"],
                   "initial_cem_costs": costs, "initial_cem_ranks": average_ranks(costs)})
    rows = []
    for step in range(len(latents)):
        row = {"arm": arm, "imagined_step": step, "nominal_raw_steps_after_start": (step+1)*5,
               "candidate_count": latents.shape[1], "observed": False,
               "physical_quantity_status": "probe-decoded prediction; candidate paths were not realized",
               "latent_l2_change": float(np.linalg.norm(latent_delta[step])),
               "latent_l2_change_mean_per_candidate": float(values["latent_l2_change"][step].mean()),
               "decoded_displacement_change": (xyz[step]-baseline_xyz[step]).mean(0).tolist(),
               "decoded_displacement_change_mean_norm": float(np.linalg.norm(xyz[step]-baseline_xyz[step], axis=-1).mean()),
               "decoded_goal_progress_mean": float(values["decoded_goal_progress"][step].mean()),
               "decoded_goal_progress_change_mean": float(values["decoded_goal_progress_change"][step].mean()),
               "same_unsteered_initial_best_candidate": int(baseline_costs.argmin()),
               "same_unsteered_initial_best_decoded_xyz": xyz[step, int(baseline_costs.argmin())].tolist(),
               "arm_initial_best_candidate": int(costs.argmin()),
               "arm_initial_best_decoded_xyz": xyz[step, int(costs.argmin())].tolist(),
               "initial_best_is_final_selected_plan": False}
        for key in ("point_margin_0", "point_margin_0_03"):
            path_key = f"candidate_path_intersects_{key}"
            proxy_key = f"hand_to_goal_proxy_intersects_{key}"
            row[f"decoded_segment_crossing_fraction_{key}"] = float(values[path_key][step].mean())
            row[f"decoded_cumulative_path_crossing_fraction_{key}"] = float(values[path_key][:step+1].any(0).mean())
            row[f"decoded_segment_crossing_fraction_change_{key}"] = float(values[path_key][step].mean()-base[path_key][step].mean())
            row[f"hand_to_goal_proxy_crossing_fraction_{key}"] = float(values[proxy_key][step].mean())
        rows.append(row)
    return rows, values


def load_causal(directory, bank_path):
    receipt = load_receipt(directory)
    if not 0 <= int(receipt["episode"]) < 12:
        raise RuntimeError("Cannot open on-policy confirmation episode")
    if tuple(receipt["arms"]) != ARMS:
        raise RuntimeError("Incomplete or unexpected causal arms")
    entries = {entry["path"]: entry for entry in receipt["outputs"]}
    required = {f"{arm}.pt" for arm in ARMS} | {"counterfactuals.pt"}
    if not required.issubset(entries) or len(entries) != len(receipt["outputs"]):
        raise RuntimeError("Causal receipt has missing or duplicate outputs")
    bank_receipt = load_receipt(bank_path.parent)
    bank_entries = [row for row in bank_receipt["outputs"] if row["path"] == bank_path.name]
    if len(bank_entries) != 1 or not 0 <= int(bank_entries[0]["episode"]) < 12:
        raise RuntimeError("Cannot open on-policy confirmation bank")
    if bank_entries[0]["sha256"] != receipt["bank_sha256"]:
        raise RuntimeError("Causal bank hash disagrees with baseline receipt")
    bank = checked_load(bank_path.parent, bank_entries[0])
    if int(bank["episode"]) != int(receipt["episode"]):
        raise RuntimeError("Causal episode disagrees with bank")
    results = {arm: checked_load(directory, entries[f"{arm}.pt"]) for arm in ARMS}
    counterfactuals = checked_load(directory, entries["counterfactuals.pt"])
    return receipt, bank, results, counterfactuals


def analyze_run(receipt, bank, results, counterfactuals, readout):
    snapshot = bank["replans"][int(receipt["replan"])]
    # The collector's generic info state predates reset_warmup at replan zero.
    # Saved observation_proprio is the actual planner-start raw hand position.
    start = array(snapshot["observation_proprio"]).reshape(-1)[:3]
    base = results["unsteered"]
    # Read the fixed target from actual fork states, not stale reset metadata.
    goal = array(base["fork"]["states"])[0, -3:]
    base_candidates = base["first_candidates"]
    baseline_latents = array(base_candidates["predicted_visual_pooled"])
    if baseline_latents.shape != (6, 300, 384):
        raise RuntimeError("Candidate futures must be [6,300,384] with context already removed")
    base_costs = array(base_candidates["costs"]).reshape(-1)
    rows, timelines, traces = [], [], {}
    for arm in ARMS:
        result = results[arm]
        if not np.allclose(array(result["fork"]["states"])[:, -3:], goal, rtol=0, atol=1e-7):
            raise RuntimeError("Fork target changed; cannot use a fixed-goal progress metric")
        candidates = result["first_candidates"]
        if not np.array_equal(array(candidates["actions"]), array(base_candidates["actions"])):
            raise RuntimeError("Initial candidates differ across arms; no paired attribution")
        latents, costs = array(candidates["predicted_visual_pooled"]), array(candidates["costs"]).reshape(-1)
        if latents.shape != baseline_latents.shape or costs.shape != (300,):
            raise RuntimeError("Candidate layout mismatch")
        timeline, values = future_timelines(arm, latents, baseline_latents, readout, start, goal, costs, base_costs)
        timelines.extend(timeline)
        traces.update({f"{arm}__{key}": value for key, value in values.items()})
        selected = pool_native_prediction(result["selected_prediction"], 6, 1)
        selected_xyz = decode(readout, selected)[:, 0]
        base_selected_xyz = decode(readout, pool_native_prediction(base["selected_prediction"], 6, 1))[:, 0]
        real = realized_metrics(result["fork"], base["fork"], start, goal)
        observed = array(real["model_step_hand_xyz"])
        n_observed = len(observed)
        selected_metrics = {"status": "probe-decoded prediction of final CEM selected full plan",
                            "decoded_hand_xyz": selected_xyz.tolist(),
                            "decoded_displacement_xyz": (selected_xyz-start).tolist(),
                            "decoded_displacement_change_from_unsteered": (selected_xyz-base_selected_xyz).tolist(),
                            "decoded_goal_progress": (np.linalg.norm(start-goal)-np.linalg.norm(selected_xyz-goal, axis=-1)).tolist(),
                            "crossing": path_crossings(np.concatenate([start[None], selected_xyz]), goal),
                            "actual_model_step_hand_xyz": observed.tolist(), "observed_model_steps": n_observed,
                            "unobserved_model_steps": list(range(n_observed, 6)),
                            "prediction_to_actual_error_first_executed_steps": errors(selected_xyz[:n_observed], observed)}
        delta = result.get("edit")
        edit = None if delta is None else array(delta)
        proprio_delta = array(candidates["predicted_proprio"])-array(base_candidates["predicted_proprio"])
        rows.append({"arm": arm, "episode": int(receipt["episode"]), "replan": int(receipt["replan"]),
                     "module": "predictor", "block": int(receipt["block"]), "hook": "block residual output",
                     "spatial_region": "uniform pooled edit across 256 visual tokens", "intervention_imagined_step": 0,
                     "candidate_rank": int(receipt["rank"]), "candidate_task_variable": "realized XZ motion direction",
                     "cost_rank": cost_metrics(costs, base_costs),
                     "selected_actions": action_metrics(array(result["fork"]["actions"]), array(base["fork"]["actions"])),
                     "realized": real, "selected_plan_prediction": selected_metrics,
                     "specificity": {"physical_specificity_validated": False,
                                     "non_target_predicted_proprio_latent_l2_change": float(np.linalg.norm(proprio_delta)),
                                     "interpretation": "proprio representation norm; not decoded unrelated physical variables"},
                     "edit": {"saved": edit is not None, "raw_delta_l2_norm": None if edit is None else float(np.linalg.norm(edit)),
                              "mean_per_candidate_delta_norm": None if edit is None else float(np.linalg.norm(edit, axis=-1).mean())},
                     "manifold_distance": {"supported": False, "reason": "actual pre/post edit block activations not saved; delta alone cannot establish natural support"},
                     "replay_checks": result["replay_checks"], "seconds": float(result["seconds"])})
    # These six action paths really were executed; compare their one-step predictions.
    cf_xyz = decode(readout, pool_native_prediction(counterfactuals["prediction"], 1, 6))[0]
    cf_rows = []
    for index, cf in enumerate(counterfactuals["candidates"]):
        actual = array(cf["states"])[-1, :3]
        cf_rows.append({"name": cf["name"], "native_cost": float(array(counterfactuals["native_costs"])[index]),
                        "raw_steps": int(cf["steps"]), "probe_decoded_predicted_next_hand_xyz": cf_xyz[index].tolist(),
                        "probe_decoded_displacement_xyz": (cf_xyz[index]-start).tolist(),
                        "actual_next_hand_xyz": actual.tolist(), "actual_displacement_xyz": (actual-start).tolist(),
                        "probe_to_actual_euclidean_error": float(np.linalg.norm(cf_xyz[index]-actual)),
                        "realized_crossing": path_crossings(np.concatenate([start[None], array(cf["states"])[:, :3]]), goal),
                        "probe_decoded_crossing": path_crossings(np.stack([start, cf_xyz[index]]), goal)})
    return rows, timelines, traces, cf_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--causal-dir", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    # A new immutable analysis root is mandatory; raw runs are never overwritten.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    trajectories, sources = load_discovery(args.capture_dirs)
    readout, validation = fit_and_validate(trajectories)
    np.savez_compressed(args.output_dir / "frozen_physical_readout.npz", **readout)
    write_json_atomic(args.output_dir / "decoder_validation.json", validation)
    print(json.dumps({"stage": "discovery_decoder_complete", "natural_rmse": validation["natural_state_decoding"]["rmse_xyz"],
                      "predicted_rmse": validation["unsteered_jepa_predicted_final_to_actual_next_hand"]["rmse_xyz"]}), flush=True)
    receipt, bank, results, counterfactuals = load_causal(args.causal_dir, args.bank)
    rows, timelines, traces, cf_rows = analyze_run(receipt, bank, results, counterfactuals, readout)
    np.savez_compressed(args.output_dir / "candidate_prediction_traces.npz", **traces)
    report = {"complete": True, "scope": "completed development diagnostic measurements; not frozen causal map, held-out confirmation, or steering efficacy",
              "decoder_validation": validation, "rows": rows, "timelines": timelines,
              "physical_counterfactuals": cf_rows, "sources": sources,
              "source_trajectories": [{k: row[k] for k in ("seed", "episode", "path", "sha256")} for row in trajectories],
              "source_causal_done_sha256": file_sha256(args.causal_dir / "DONE.json"),
              "source_bank_sha256": receipt["bank_sha256"], "script_sha256": file_sha256(Path(__file__)),
              "units": UNITS, "wall_geometry": {"center": WALL_CENTER.tolist(), "half_size": WALL_HALF_SIZE.tolist()},
              "trace_artifact": "candidate_prediction_traces.npz",
              "trace_layout": "arm__name keys; future arrays [6 imagined steps,300 same initial candidates,...]; XYZ ends in dimension 3",
              "limitations": [
                  "All physical futures inferred from latents are probe-decoded predictions, not native XYZ output or validated ground truth.",
                  "Discovery leave-one-seed-out decoding does not validate six-step autoregressive, edited, or on-policy candidate futures.",
                  "Initial 300 candidates were not physically executed; they have no observed path or candidate-specific ground truth.",
                  "Only the first three model steps of final selected plans were executed (15 raw steps); later futures are unobserved.",
                  "Only initial CEM candidate costs/ranks were saved; changes over all 15 CEM iterations cannot be reconstructed.",
                  "Sampled hand-center polylines omit motion between simulator observations and are not full hand/body collision checks.",
                  "Margin 0.03 is an inflated-wall geometric proxy, distinct from margin-zero point crossing and from body collision.",
                  "Hand-to-goal straight-line proxy is not the candidate path or actual realized path.",
                  "Pre/post edit block activations were not saved; edit magnitude and standardized latent differences are not manifold distance.",
                  "Proprio latent norm changes do not validate physical non-target specificity.",
                  "Single development state; unrelated-site/time controls, held-out confirmation, and full-episode paired efficacy remain absent."]}
    write_json_atomic(args.output_dir / "causal_measurements.json", report)
    write_json_atomic(args.output_dir / "DONE.json", {"complete": True, "scope": report["scope"],
        "outputs": [{"path": path.name, "sha256": file_sha256(path)} for path in sorted(args.output_dir.iterdir()) if path.is_file()]})
    print(json.dumps({"stage": "causal_measurements_complete", "rows": len(rows), "timelines": len(timelines), "output": str(args.output_dir)}), flush=True)


if __name__ == "__main__":
    main()
