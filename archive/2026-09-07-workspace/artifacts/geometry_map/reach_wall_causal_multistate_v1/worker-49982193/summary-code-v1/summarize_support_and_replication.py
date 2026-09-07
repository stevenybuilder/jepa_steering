#!/usr/bin/env python3
"""CPU-only descriptive support proxies and frozen five-start causal replication.

No fitting to edited outcomes, independent-candidate inference, or heldout loading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors

EPISODES = (0, 1, 4, 7, 10)
ARMS = ("unsteered", "identity", "subspace_up", "sham_up", "subspace_down", "sham_down")
CANDIDATE_SHA = "f4c295a148a22a9534cedd7d04b396f537bc8049842020886ba4f7fb0f1ca166"


def sha(path):
    with Path(path).open("rb") as source:
        digest = hashlib.file_digest(source, "sha256")
    return digest.hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def array(value):
    return value.detach().cpu().double().numpy() if torch.is_tensor(value) else np.asarray(value, dtype=float)


def distribution(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected nonempty finite measurements")
    return {"count": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
            **{f"q{q:02d}": float(np.quantile(values, q / 100)) for q in (5, 25, 75, 95)},
            "min": float(values.min()), "max": float(values.max())}


def paired_distance_summary(before, after):
    before, after = np.asarray(before), np.asarray(after)
    if before.shape != after.shape or np.any(before <= 0):
        raise ValueError("Paired ratios require matching arrays and positive baseline distances")
    return {"before": distribution(before), "after": distribution(after),
            "paired_change": distribution(after - before),
            "paired_after_over_before": distribution(after / before),
            "ratio_of_means": float(after.mean() / before.mean())}


def fit_support_metric(reference, mean, scale):
    """Fit once to saved discovery reference in the frozen candidate coordinates."""
    reference, mean, scale = map(array, (reference, mean, scale))
    if np.any(scale <= 0):
        raise ValueError("Positive saved scale required")
    natural = (reference - mean) / scale
    covariance = LedoitWolf().fit(natural)  # frozen sklearn defaults, discovery only
    nearest = NearestNeighbors(n_neighbors=1, algorithm="brute", metric="euclidean").fit(natural)

    def measure(points):
        standardized = (array(points) - mean) / scale
        return {"nearest_reference_euclidean": nearest.kneighbors(standardized)[0][:, 0],
                "shrinkage_mahalanobis": np.sqrt(np.maximum(covariance.mahalanobis(standardized), 0))}

    return measure, float(covariance.shrinkage_)


def checked_tensor(directory, receipt, name):
    entries = [x for x in receipt["outputs"] if x["path"] == name]
    if not receipt.get("complete") or len(entries) != 1 or sha(directory / name) != entries[0]["sha256"]:
        raise RuntimeError(f"Incomplete or hash-mismatched tensor: {name}")
    return torch.load(directory / name, map_location="cpu", weights_only=False)


def physical_metrics(start, goal, path, baseline_end, direction):
    start, goal, path, baseline_end = map(array, (start, goal, path, baseline_end))
    end = path[-1, :3]
    effect = end - baseline_end
    initial_distance, end_distance = np.linalg.norm(start - goal), np.linalg.norm(end - goal)
    return {"start_hand_xyz_m": start.tolist(), "task_goal_xyz_m": goal.tolist(),
            "end_hand_xyz_m": end.tolist(), "displacement_from_observed_start_xyz_m": (end - start).tolist(),
            "endpoint_effect_vs_unsteered_xyz_m": effect.tolist(),
            "endpoint_effect_l2_m": float(np.linalg.norm(effect)),
            "signed_target_z_effect_m": float(direction * effect[2]),
            "off_target_xy_effect_l2_m": float(np.linalg.norm(effect[:2])),
            "initial_goal_distance_m": float(initial_distance), "end_goal_distance_m": float(end_distance),
            "goal_progress_from_start_m": float(initial_distance - end_distance),
            "goal_improvement_vs_unsteered_m": float(np.linalg.norm(baseline_end - goal) - end_distance),
            "observed_path_xyz_m": np.vstack([start, path[:, :3]]).tolist()}


def extract_episode(episode_dir, bank_path, output_dir):
    receipt = json.loads((episode_dir / "DONE.json").read_text())
    # Both receipts are checked before opening any tensors, including sealed banks.
    episode = receipt["episode"]
    bank_receipt = json.loads((bank_path.parent / "DONE.json").read_text())
    entries = [x for x in bank_receipt["outputs"] if x["path"] == bank_path.name]
    if (episode not in EPISODES or len(entries) != 1 or entries[0]["episode"] != episode
            or not bank_receipt.get("complete") or not receipt.get("complete")
            or receipt["candidate_sha256"] != CANDIDATE_SHA or receipt["beta"] != .5
            or receipt["replan"] != 0 or tuple(receipt["arms"]) != ARMS):
        raise RuntimeError("Expected frozen completed development protocol and matching bank receipt")
    if sha(bank_path) != entries[0]["sha256"] or entries[0]["sha256"] != receipt["bank_sha256"]:
        raise RuntimeError("Development bank hash mismatch")
    bank = torch.load(bank_path, map_location="cpu", weights_only=False)
    if bank["episode"] != episode:
        raise RuntimeError("Bank episode mismatch")
    start = array(bank["replans"][0]["observation_proprio"]).reshape(-1, 4)[-1, :3]
    baseline = checked_tensor(episode_dir, receipt, "unsteered.pt")
    base_states = array(baseline["fork"]["states"])
    goal, baseline_end = base_states[-1, -3:], base_states[-1, :3]
    rows, edit_norms, checks = [], {}, {}
    for arm in ARMS:
        result = baseline if arm == "unsteered" else checked_tensor(episode_dir, receipt, arm + ".pt")
        if not torch.equal(result["first_candidates"]["actions"], baseline["first_candidates"]["actions"]):
            raise RuntimeError("Initial candidates differ")
        states = array(result["fork"]["states"])
        if not np.array_equal(states[:, -3:], np.broadcast_to(goal, states[:, -3:].shape)):
            raise RuntimeError("Physical task goal differs across fork")
        if arm == "identity":
            checks = {"plan": torch.equal(result["plan"], baseline["plan"]),
                      "costs": torch.equal(result["first_candidates"]["costs"], baseline["first_candidates"]["costs"]),
                      "states": np.array_equal(states, base_states),
                      "pixels": torch.equal(result["fork"]["frames"], baseline["fork"]["frames"])}
            if not all(checks.values()):
                raise RuntimeError("Exact identity failed")
        direction = -1 if arm.endswith("down") else 1
        row = next(x.copy() for x in receipt["rows"] if x["arm"] == arm)
        row.update(physical_metrics(start, goal, states, baseline_end, direction))
        if result.get("edit") is not None:
            edit_norms[arm] = np.linalg.norm(array(result["edit"]), axis=-1)
            row["requested_raw_edit_l2"] = distribution(edit_norms[arm])
        rows.append(row)
    dose_checks = {}
    for direction in ("up", "down"):
        left, right = edit_norms["subspace_" + direction], edit_norms["sham_" + direction]
        maximum = float(np.abs(left - right).max())
        if maximum > 1e-6:
            raise RuntimeError("Matched raw dose differs beyond float32 rounding")
        dose_checks[direction] = {"max_absolute_raw_l2_difference": maximum,
                                  "semantic_mean": float(left.mean()), "sham_mean": float(right.mean())}
    write(output_dir / f"episode-{episode:03d}.json", {
        "complete": True, "episode": episode, "replan": 0, "rows": rows,
        "identity_exact": checks, "paired_raw_dose": dose_checks,
        "goal_replay": receipt.get("goal_replay", {"mode": "original_regenerated_exact"}),
        "candidate_sha256": CANDIDATE_SHA, "bank_sha256": receipt["bank_sha256"],
        "diagnostic_receipt_sha256": sha(episode_dir / "DONE.json"),
        "source_outputs": receipt["outputs"],
        "start_source": "replan0 observation_proprio xyz, after reset warmup; not stale snapshot state",
        "script_sha256": sha(__file__), "held_confirmation_run": False,
    })


def bootstrap_mean(values, draws=10000, seed=905):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("One scalar per independent episode is required")
    indices = np.random.default_rng(seed).integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(1)
    return {"mean": float(values.mean()), "descriptive_bootstrap_ci95": np.quantile(means, [.025, .975]).tolist(),
            "positive_episodes": int((values > 0).sum()), "negative_episodes": int((values < 0).sum()),
            "N_episodes": len(values), "episode_values": values.tolist()}


def summarize_replication(inputs, output_dir):
    episodes = sorted([json.loads(path.read_text()) for path in inputs], key=lambda x: x["episode"])
    if tuple(x["episode"] for x in episodes) != EPISODES or not all(x["complete"] for x in episodes):
        raise RuntimeError("All five predetermined development starts are required, without exclusion")
    if any(x["candidate_sha256"] != CANDIDATE_SHA for x in episodes):
        raise RuntimeError("Candidate changed across states")
    fields = ("candidate_rank_changes", "candidate_cost_l2_change", "chosen_raw_action_l2_change",
              "predicted_visual_pooled_l2_change", "endpoint_effect_l2_m", "signed_target_z_effect_m",
              "off_target_xy_effect_l2_m", "goal_improvement_vs_unsteered_m")
    arms = {arm: {field: bootstrap_mean([next(r[field] for r in ep["rows"] if r["arm"] == arm)
                                        for ep in episodes]) for field in fields} for arm in ARMS}
    comparisons = {}
    for direction in ("up", "down"):
        comparisons[direction] = {}
        for field in ("signed_target_z_effect_m", "goal_improvement_vs_unsteered_m", "endpoint_effect_l2_m", "off_target_xy_effect_l2_m"):
            semantic = np.array(arms["subspace_" + direction][field]["episode_values"])
            sham = np.array(arms["sham_" + direction][field]["episode_values"])
            comparisons[direction][field] = bootstrap_mean(semantic - sham)
    # Average the two prescribed directions WITHIN each episode, then bootstrap episodes.
    combined = {field: bootstrap_mean(np.mean([comparisons[d][field]["episode_values"] for d in ("up", "down")], axis=0))
                for field in comparisons["up"]}
    write(output_dir / "replication_summary.json", {
        "complete": True, "status": "development_descriptive_short_forks_not_confirmatory_specificity",
        "episode_ids": list(EPISODES), "N_independent_development_starts": 5,
        "candidate_count_per_start": 300, "candidate_count_is_not_independent_N": True,
        "arms": arms, "semantic_minus_norm_matched_sham": comparisons,
        "combined_directions_semantic_minus_sham": combined, "episodes": episodes,
        "bootstrap": {"unit": "episode", "draws": 10000, "seed": 905, "interpretation": "descriptive percentile interval, not a confirmatory test"},
        "sign_conventions": {"signed_target_z_effect_m": "up: positive z; down: negative z, each relative to unsteered",
                             "goal_improvement_vs_unsteered_m": "positive means closer to physical task goal",
                             "endpoint_effect_l2_m": "magnitude only, larger is not better or more specific"},
        "held_confirmation_run": False,
        "remaining_controls": ["unrelated-site", "time-shift", "held-out confirmation", "full-episode efficacy"],
        "script_sha256": sha(__file__),
    })


def summarize_support(support_dir, output_dir):
    receipt = json.loads((support_dir / "DONE.json").read_text())
    if receipt["candidate_sha256"] != CANDIDATE_SHA or receipt["episode"] != 0 or receipt["beta"] != .5:
        raise RuntimeError("Unexpected fixed support protocol")
    reference = checked_tensor(support_dir, receipt, "natural_reference.pt")
    if reference["split"] != "discovery" or reference["block_pooled"].shape != (2850, 400):
        raise RuntimeError("Only the saved discovery reference may fit support metrics")
    measure, shrinkage = fit_support_metric(reference["block_pooled"], reference["mean"], reference["scale"])
    baseline = checked_tensor(support_dir, receipt, "unsteered.pt")
    before = baseline["activation_support"]["before_pooled"]
    before_metrics = measure(before)
    baseline_proprio = array(baseline["prediction"]["proprio"][-1]).reshape(300, -1)
    traces, rows, doses = {}, [], {}
    for arm in ARMS:
        result = baseline if arm == "unsteered" else checked_tensor(support_dir, receipt, arm + ".pt")
        support = result["activation_support"]
        if not torch.equal(before, support["before_pooled"]):
            raise RuntimeError("Support baselines are not the identical candidate inputs")
        after_metrics = before_metrics if torch.equal(before, support["after_pooled"]) else measure(support["after_pooled"])
        row = {"arm": arm, "distances": {key: paired_distance_summary(before_metrics[key], after_metrics[key]) for key in before_metrics},
               "activation_norms": {key: distribution(array(value)) for key, value in support.items() if key.endswith("_l2")}}
        requested, realized = array(support["requested_delta"]), array(support["realized_delta"])
        residual = realized - requested
        row["rounding_residual_raw_l2"] = distribution(np.linalg.norm(residual, axis=-1))
        row["rounding_residual_standardized_l2"] = distribution(np.linalg.norm(residual / array(reference["scale"]), axis=-1))
        proprio_delta = array(result["prediction"]["proprio"][-1]).reshape(300, -1) - baseline_proprio
        row["final_predicted_proprio_change_l2"] = distribution(np.linalg.norm(proprio_delta, axis=-1))
        row["final_predicted_proprio_change_rms"] = distribution(np.sqrt(np.mean(proprio_delta ** 2, axis=-1)))
        row["predicted_proprio_interpretation"] = "model latent proprio trace difference, not decoded physical specificity"
        doses[arm] = np.linalg.norm(requested, axis=-1)
        for metric in before_metrics:
            traces[f"{arm}__{metric}__before"] = before_metrics[metric]
            traces[f"{arm}__{metric}__after"] = after_metrics[metric]
        rows.append(row)
        print(json.dumps({"event": "support_summary_arm_complete", "arm": arm}), flush=True)
        if arm != "unsteered":
            del result
    paired_doses = {}
    for direction in ("up", "down"):
        delta = doses["subspace_" + direction] - doses["sham_" + direction]
        if np.abs(delta).max() > 1e-6:
            raise RuntimeError("Semantic-sham raw dose mismatch")
        paired_doses[direction] = {"max_absolute_difference": float(np.abs(delta).max()), "differences": distribution(delta)}
    np.savez_compressed(output_dir / "support_candidate_distances.npz", **traces)
    write(output_dir / "support_summary.json", {
        "complete": True, "status": "support_proxy_measured_not_true_manifold_membership",
        "episode": 0, "replan": 0, "independent_development_states": 1, "paired_candidates": 300,
        "candidate_count_is_not_independent_N": True, "rows": rows, "paired_raw_dose": paired_doses,
        "metric_fit": {"reference_states": 2850, "reference_trajectories": 150, "split": "discovery",
                       "coordinates": "same saved candidate mean/std, no refit", "covariance": "LedoitWolf default parameters",
                       "shrinkage": shrinkage, "mahalanobis": "square root of sklearn squared Mahalanobis", "natural_reference_sha256": sha(support_dir / "natural_reference.pt")},
        "identity_checks": receipt["identity_checks"], "cross_process_numerical_policy": receipt["cross_process_numerical_policy"],
        "cross_process_observed_checks": receipt["rows"], "remaining_controls": receipt["remaining_controls"],
        "support_receipt_sha256": sha(support_dir / "DONE.json"), "script_sha256": sha(__file__),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("extract", "support", "replication"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-dir", type=Path)
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--support-dir", type=Path)
    parser.add_argument("--inputs", type=Path, nargs="+")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    try:
        if args.mode == "extract":
            extract_episode(args.episode_dir, args.bank, args.output_dir)
        elif args.mode == "support":
            summarize_support(args.support_dir, args.output_dir)
        else:
            summarize_replication(args.inputs, args.output_dir)
        write(args.output_dir / "DONE.json", {"complete": True, "mode": args.mode, "script_sha256": sha(__file__),
              "outputs": [{"path": p.name, "sha256": sha(p)} for p in sorted(args.output_dir.iterdir()) if p.is_file()]})
    except Exception as exc:
        write(args.output_dir / "FAILED.json", {"complete": False, "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
