#!/usr/bin/env python3
"""Outcome-free Othello-style coordinate audit for Panel-P MetaWorld reach tasks.

The Othello lesson is used narrowly: a predictor state may make a state variable
linear in a context-relative ontology even when the corresponding absolute label
is not linear. For Reach and Reach-Wall, the primary matched comparison is the same
three-dimensional hand state expressed in world coordinates and relative to the
episode's changing goal. A fixed-wall translation is an affine-equivalent
negative control, and an episode-shuffled goal tests whether any advantage binds
the current goal rather than merely using an easier target scale.

Simulator state is used only to score offline readouts. It is never supplied to
the online intervention. A passing result nominates a coordinate hypothesis; it
does not establish causal use and does not automatically change the Sonar arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from public_panel_factor_gate import episode_folds, load_site_sequences
from public_panel_sonar_fit import transform


# Verified against the exact MetaWorld and vendor files used by the frozen cell.
# In this task, observation slots 4:7 are the randomized ``obj`` cylinder, not
# the wall. The wall is a separate fixed body in the XML and must not be inferred
# from those slots.
ADAPTER_IDS = {
    "mw-reach": "metaworld-reach-v3-2026-09-04",
    "mw-reach-wall": "metaworld-reach-wall-v3-2026-09-04",
}
REACH_WALL_CENTER = np.asarray([0.1, 0.75, 0.06], dtype=np.float64)
REACH_WALL_HALF_EXTENTS = np.asarray([0.12, 0.01, 0.06], dtype=np.float64)
EXPECTED_SOURCE_SHA256 = {
    "metaworld_base": "4282580b80befc557f67d25acffffec737ad0dae2c440601be76fc09139f351d",
    "reach_wall_env": "2540167409ccaf93c0a01f3333d96e50ddfdf90721b30c23e09e6c4bc2c2a279",
    "reach_wall_xml": "1281cf59d414a24522f7333f995dc4e712c3059538d9ab6434938da294710ecb",
    "vendor_evaluator": "7a81a47af6eeea6b024b10d65115cdef96650621b0d60339aaad0a161d9b2ee0",
}
EXPECTED_REACH_SOURCE_SHA256 = {
    "metaworld_base": "4282580b80befc557f67d25acffffec737ad0dae2c440601be76fc09139f351d",
    "reach_env": "6403bb039f0627c376be11f2157ff48b8747154024effd8c16f19904762c7c47",
    "vendor_evaluator": "7a81a47af6eeea6b024b10d65115cdef96650621b0d60339aaad0a161d9b2ee0",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_adapter_sources(vendor_repo: Path, task: str = "mw-reach-wall") -> dict[str, str]:
    """Fail closed if the code that gives the 39-vector its meaning changed."""
    import metaworld  # Imported only for the executable audit, not unit tests.

    package = Path(metaworld.__file__).resolve().parent
    if task == "mw-reach":
        expected = EXPECTED_REACH_SOURCE_SHA256
        paths = {
            "metaworld_base": package / "sawyer_xyz_env.py",
            "reach_env": package / "envs" / "sawyer_reach_v3.py",
            "vendor_evaluator": vendor_repo / "evals" / "simu_env_planning" / "planning" / "plan_evaluator.py",
        }
    elif task == "mw-reach-wall":
        expected = EXPECTED_SOURCE_SHA256
        paths = {
            "metaworld_base": package / "sawyer_xyz_env.py",
            "reach_wall_env": package / "envs" / "sawyer_reach_wall_v3.py",
            "reach_wall_xml": package / "assets" / "sawyer_xyz" / "sawyer_reach_wall_v3.xml",
            "vendor_evaluator": vendor_repo / "evals" / "simu_env_planning" / "planning" / "plan_evaluator.py",
        }
    else:
        raise ValueError(f"no audited coordinate adapter for {task!r}")
    actual = {name: sha256_file(path) for name, path in paths.items()}
    mismatches = {
        name: {"expected": expected[name], "actual": value, "path": str(paths[name])}
        for name, value in actual.items()
        if value != expected[name]
    }
    if mismatches:
        raise ValueError(f"reach-wall coordinate adapter source mismatch: {mismatches}")
    return actual


def signed_box_distance(points: np.ndarray) -> np.ndarray:
    """Signed Euclidean distance to the frozen reach-wall axis-aligned box."""
    q = np.abs(np.asarray(points, dtype=np.float64) - REACH_WALL_CENTER) - REACH_WALL_HALF_EXTENTS
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return outside + inside


def straight_path_wall_clearance(hand: np.ndarray, goal: np.ndarray, n_samples: int = 65) -> np.ndarray:
    """Minimum signed wall clearance along the current straight hand-to-goal path."""
    fraction = np.linspace(0.0, 1.0, n_samples, dtype=np.float64)
    path = hand[:, None, :] + fraction[None, :, None] * (goal - hand)[:, None, :]
    distance = signed_box_distance(path.reshape(-1, 3)).reshape(len(hand), n_samples)
    return distance.min(axis=1, keepdims=True)


def metaworld_coordinate_targets(states: np.ndarray, goal_state: np.ndarray) -> dict[str, np.ndarray]:
    """Create audited reach-wall targets from correctly aligned post-reset states."""
    states = np.asarray(states, dtype=np.float64)
    goal_state = np.asarray(goal_state, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 39 or goal_state.shape != (39,):
        raise ValueError("MetaWorld coordinate adapter requires states [T,39] and goal_state [39]")
    hand = states[:, :3]
    active_goal = states[:, -3:]
    expert_goal = np.broadcast_to(goal_state[-3:], active_goal.shape)
    if not np.allclose(active_goal, expert_goal, atol=2e-5, rtol=0.0):
        raise ValueError("aligned rollout target does not match the expert goal target")
    wall = np.broadcast_to(REACH_WALL_CENTER, hand.shape)
    hand_to_goal = active_goal - hand
    hand_to_wall = wall - hand
    return {
        # Primary Othello-style matched pair: same hand state, changing goal context.
        "absolute_hand_world": hand,
        "relative_hand_to_goal": hand_to_goal,
        # A fixed affine translation/sign must have the same linear accessibility.
        "relative_hand_to_fixed_wall": hand_to_wall,
        # Context and route diagnostics; they are not pooled into the matched verdict.
        "absolute_goal_world": active_goal,
        "distance_hand_to_goal": np.linalg.norm(hand_to_goal, axis=1, keepdims=True),
        "signed_hand_wall_clearance": signed_box_distance(hand)[:, None],
        "straight_path_wall_clearance": straight_path_wall_clearance(hand, active_goal),
    }


def metaworld_reach_coordinate_targets(states: np.ndarray, goal_state: np.ndarray) -> dict[str, np.ndarray]:
    """Create model-native candidates for obstacle-free MetaWorld Reach."""
    states = np.asarray(states, dtype=np.float64)
    goal_state = np.asarray(goal_state, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 39 or goal_state.shape != (39,):
        raise ValueError("MetaWorld Reach adapter requires states [T,39] and goal_state [39]")
    hand = states[:, :3]
    active_goal = states[:, -3:]
    expert_goal = np.broadcast_to(goal_state[-3:], active_goal.shape)
    if not np.allclose(active_goal, expert_goal, atol=2e-5, rtol=0.0):
        raise ValueError("aligned rollout target does not match the expert goal target")
    relative = active_goal - hand
    return {
        "absolute_hand_world": hand,
        "relative_hand_to_goal": relative,
        "absolute_goal_world": active_goal,
        "distance_hand_to_goal": np.linalg.norm(relative, axis=1, keepdims=True),
    }


def selected_replan_indices(window: tuple[int, int]) -> np.ndarray:
    """Select plan rows whose post-action endpoints are inside the common window.

    Capture row ``t`` is the final world-model prediction after selected chunk
    ``t``. It must therefore be compared with simulator state at boundary
    ``t+1``, not the pre-action state at boundary ``t``. Replan zero is valid
    under endpoint alignment even though the reset ``info`` at boundary zero is
    stale in the released evaluator.
    """
    indices = np.arange(window[0], window[1], dtype=np.int64)
    if len(indices) < 3:
        raise ValueError(f"fewer than three aligned replan endpoints remain: {window}")
    return indices


def load_aligned_targets(
    capture: Path,
    index: dict,
    replan_indices: np.ndarray,
    *,
    task: str = "mw-reach-wall",
) -> list[dict[str, np.ndarray]]:
    logs = {
        int(row["ep"]): row
        for row in (
            json.loads(line) for line in (capture / "episodes.jsonl").read_text().splitlines() if line.strip()
        )
    }
    out = []
    for row in index["episodes"]:
        episode = logs.get(int(row["ep"]), {})
        trace_rel = episode.get("behavior_trace")
        if not trace_rel:
            raise ValueError(f"episode {row['ep']} has no behavior trace")
        with np.load(capture / trace_rel) as trace:
            required = {"simulator_states", "goal_state", "plan_boundaries"}
            if not required.issubset(trace.files):
                raise ValueError(f"episode {row['ep']} trace lacks {sorted(required - set(trace.files))}")
            boundaries = np.asarray(trace["plan_boundaries"], dtype=np.int64)
            states = np.asarray(trace["simulator_states"], dtype=np.float64)
            if np.max(replan_indices) >= len(boundaries) - 1:
                raise ValueError(f"episode {row['ep']} has insufficient plan boundaries")
            endpoints = boundaries[replan_indices + 1]
            if np.any(endpoints >= len(states)):
                raise ValueError(f"episode {row['ep']} has insufficient simulator states")
            if task == "mw-reach":
                targets = metaworld_reach_coordinate_targets(states[endpoints], trace["goal_state"])
            elif task == "mw-reach-wall":
                targets = metaworld_coordinate_targets(states[endpoints], trace["goal_state"])
            else:
                raise ValueError(f"no audited target builder for {task!r}")
        out.append(targets)
    return out


def ridge_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, ridge: float) -> np.ndarray:
    x_mean, y_mean = train_x.mean(0), train_y.mean(0)
    x_scale = np.maximum(train_x.std(0), 1e-8)
    xs = (train_x - x_mean) / x_scale
    weights = np.linalg.solve(xs.T @ xs + ridge * np.eye(xs.shape[1]), xs.T @ (train_y - y_mean))
    return y_mean + ((test_x - x_mean) / x_scale) @ weights


def fit_native_readout(factors: list[np.ndarray], targets: list[dict[str, np.ndarray]], ridge: float) -> dict:
    """Fit and factor the final goal-relative linear readout in factor coordinates."""
    x = np.concatenate(factors)
    y = np.concatenate([episode["relative_hand_to_goal"] for episode in targets])
    x_mean, y_mean = x.mean(0), y.mean(0)
    x_scale = np.maximum(x.std(0), 1e-8)
    standardized = (x - x_mean) / x_scale
    standardized_weights = np.linalg.solve(
        standardized.T @ standardized + ridge * np.eye(x.shape[1]),
        standardized.T @ (y - y_mean),
    )
    weights = standardized_weights / x_scale[:, None]
    u, singular, _vt = np.linalg.svd(weights, full_matrices=False)
    tolerance = max(float(singular.max()) if len(singular) else 0.0, 1e-12) * 1e-6
    rank = max(1, int(np.sum(singular > tolerance)))
    basis = u[:, :rank]
    return {
        "factor_mean": x_mean,
        "target_mean": y_mean,
        "weights": weights,
        "rowspace_basis": basis,
        "singular_values": singular,
        "rank": rank,
    }


def bootstrap_mean(values: np.ndarray, seed: int, n_bootstrap: int = 2000) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = np.asarray([rng.choice(values, len(values), replace=True).mean() for _ in range(n_bootstrap)])
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def coordinate_cv(
    factors: list[np.ndarray], targets: list[dict[str, np.ndarray]], *, n_folds: int, ridge: float, seed: int
) -> dict:
    """Whole-episode readouts plus matched and context-binding controls."""
    folds = episode_folds(len(factors), n_folds, seed)
    episode_ids = np.concatenate([np.full(len(x), i, dtype=np.int32) for i, x in enumerate(factors)])
    x = np.concatenate(factors)
    progress = np.concatenate([
        np.linspace(0.0, 1.0, len(sequence), dtype=np.float64)[:, None] for sequence in factors
    ])
    progress_basis = np.c_[np.ones(len(progress)), progress, 2.0 * np.square(progress) - 1.0]

    # Preserve goal/trajectory marginals while breaking the binding between a
    # hidden state and its own goal.
    if len(targets) < 3:
        raise ValueError("coordinate audit requires at least three episodes")
    rng = np.random.default_rng(seed + 17)
    shift = int(rng.integers(1, len(targets)))
    shuffled_goals = [targets[(i + shift) % len(targets)]["absolute_goal_world"][0] for i in range(len(targets))]
    targets = [dict(episode) for episode in targets]
    for i, episode in enumerate(targets):
        episode["relative_hand_to_shuffled_goal"] = (
            np.broadcast_to(shuffled_goals[i], episode["absolute_hand_world"].shape)
            - episode["absolute_hand_world"]
        )

    target_names = list(targets[0])
    reports = {}
    for name in target_names:
        y = np.concatenate([episode[name] for episode in targets])
        episode_error = {"latent": np.full(len(factors), np.nan), "progress": np.full(len(factors), np.nan)}
        for held_episodes in folds:
            test = np.isin(episode_ids, held_episodes)
            train = ~test
            scale = np.maximum(y[train].std(0), 1e-8)
            for kind, features in (("latent", x), ("progress", progress_basis)):
                prediction = ridge_predict(features[train], y[train], features[test], ridge)
                squared = np.mean(np.square((y[test] - prediction) / scale), axis=1)
                for episode in held_episodes:
                    mask = episode_ids[test] == episode
                    episode_error[kind][episode] = float(squared[mask].mean())
        if not all(np.all(np.isfinite(value)) for value in episode_error.values()):
            raise ValueError(f"incomplete cross-validation coverage for {name}")
        gain = episode_error["progress"] - episode_error["latent"]
        reports[name] = {
            "latent_normalized_mse": float(episode_error["latent"].mean()),
            "progress_normalized_mse": float(episode_error["progress"].mean()),
            "latent_gain_over_progress": float(gain.mean()),
            "gain_episode_bootstrap_95_ci": bootstrap_mean(gain, seed + 101 + len(reports)),
            "per_episode_latent_normalized_mse": episode_error["latent"].tolist(),
            "per_episode_progress_normalized_mse": episode_error["progress"].tolist(),
        }

    def matched_comparison(reference: str, candidate: str, comparison_seed: int) -> dict:
        reference_error = np.asarray(reports[reference]["per_episode_latent_normalized_mse"])
        candidate_error = np.asarray(reports[candidate]["per_episode_latent_normalized_mse"])
        difference = candidate_error - reference_error
        interval = bootstrap_mean(difference, comparison_seed)
        advantage = float((reference_error.mean() - candidate_error.mean()) / max(reference_error.mean(), 1e-12))
        return {
            "reference": reference,
            "candidate": candidate,
            "candidate_minus_reference_normalized_mse": float(difference.mean()),
            "candidate_minus_reference_episode_bootstrap_95_ci": interval,
            "candidate_advantage_fraction": advantage,
            "candidate_better": bool(advantage >= 0.05 and interval[1] < 0.0),
        }

    goal_relative = matched_comparison("absolute_hand_world", "relative_hand_to_goal", seed + 401)
    shuffled_binding = matched_comparison("relative_hand_to_shuffled_goal", "relative_hand_to_goal", seed + 402)
    fixed_wall = (
        matched_comparison("absolute_hand_world", "relative_hand_to_fixed_wall", seed + 403)
        if "relative_hand_to_fixed_wall" in reports else None
    )
    goal_beats_progress = reports["relative_hand_to_goal"]["gain_episode_bootstrap_95_ci"][0] > 0.0
    full_native = fit_native_readout(factors, targets, ridge)
    fold_native = []
    all_episode_ids = np.arange(len(factors))
    for held_episodes in folds:
        train_episodes = all_episode_ids[~np.isin(all_episode_ids, held_episodes)]
        fold_native.append(fit_native_readout(
            [factors[int(episode)] for episode in train_episodes],
            [targets[int(episode)] for episode in train_episodes],
            ridge,
        ))
    fold_ranks = [int(value["rank"]) for value in fold_native]
    rowspace_angles = []
    for left in range(len(fold_native)):
        for right in range(left + 1, len(fold_native)):
            rank = min(fold_ranks[left], fold_ranks[right], int(full_native["rank"]))
            singular = np.linalg.svd(
                fold_native[left]["rowspace_basis"][:, :rank].T
                @ fold_native[right]["rowspace_basis"][:, :rank],
                compute_uv=False,
            )
            angles = np.degrees(np.arccos(np.clip(singular, -1.0, 1.0)))
            rowspace_angles.append(float(np.sqrt(np.mean(np.square(angles)))))
    rank_consistent = bool(all(value == int(full_native["rank"]) for value in fold_ranks))
    max_rowspace_angle = float(max(rowspace_angles, default=0.0))
    rowspace_stable = bool(rank_consistent and max_rowspace_angle <= 35.0)
    candidate = bool(
        goal_relative["candidate_better"]
        and shuffled_binding["candidate_better"]
        and goal_beats_progress
        and rowspace_stable
    )
    return {
        "targets": reports,
        "primary_matched_comparison": goal_relative,
        "goal_binding_control": shuffled_binding,
        "fixed_wall_affine_equivalence_control": (
            {
                **fixed_wall,
                "expected_result": "approximately zero: a fixed translation/sign cannot reveal a new linear ontology",
            }
            if fixed_wall is not None
            else {"applicable": False, "reason": "obstacle-free Reach has no wall"}
        ),
        "goal_relative_beats_progress": bool(goal_beats_progress),
        "native_readout_rowspace_stability": {
            "full_rank": int(full_native["rank"]),
            "fold_ranks": fold_ranks,
            "pairwise_rms_principal_angles_deg": rowspace_angles,
            "max_pairwise_rms_principal_angle_deg": max_rowspace_angle,
            "threshold_deg": 35.0,
            "rank_consistent": rank_consistent,
            "stable": rowspace_stable,
        },
        "model_native_relational_candidate": candidate,
        "interpretation": (
            "candidate means the changing goal-relative hand coordinate is a simpler held-out linear readout "
            "than the matched world-coordinate label, survives a shuffled-goal binding control, and beats "
            "progress, with a stable readout row space across episode folds. It remains a nomination, not "
            "evidence of causal use."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--coordinates", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--pool", default="mean", choices=["mean", "mean_all"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()

    with np.load(args.coordinates) as raw:
        coordinate = {key: np.asarray(raw[key]) for key in raw.files}
    site = str(coordinate["site"].item())
    hidden, _actions, index = load_site_sequences(args.capture, site, args.pool)
    task = str(index.get("meta", {}).get("task") or index.get("meta", {}).get("task_name") or "")
    task = {"reach": "mw-reach", "reach-wall": "mw-reach-wall", "": "mw-reach-wall"}.get(task, task)
    if task not in ADAPTER_IDS:
        raise SystemExit(f"coordinate adapters cannot interpret task {task!r}")
    vendor_repo_raw = index.get("meta", {}).get("repo")
    if not vendor_repo_raw:
        raise SystemExit("capture index does not identify the vendor repository")
    vendor_repo = Path(vendor_repo_raw)
    source_hashes = verify_adapter_sources(vendor_repo, task)

    window_raw = index.get("pre_outcome_window") or [0, min(len(x) for x in hidden)]
    window = (int(window_raw[0]), int(window_raw[1]))
    replan_indices = selected_replan_indices(window)
    factors = [transform(sequence[replan_indices], coordinate) for sequence in hidden]
    targets = load_aligned_targets(args.capture, index, replan_indices, task=task)
    report = {
        "status": "outcome_free_diagnostic_only",
        "capture": str(args.capture.resolve()),
        "capture_index_sha256": hashlib.sha256(
            (args.capture / "activations" / "index.json").read_bytes()
        ).hexdigest(),
        "coordinates": str(args.coordinates.resolve()),
        "site": site,
        "pool": args.pool,
        "pre_outcome_window": list(window),
        "aligned_replan_indices": replan_indices.tolist(),
        "simulator_alignment": "capture row t matched to post-action simulator state at plan boundary t+1",
        "n_episodes": len(factors),
        "cross_validation_unit": "whole episode / unseen goal context",
        "privileged_simulator_state_used_online": False,
        "adapter": {
            "id": ADAPTER_IDS[task],
            "task": task,
            "source_sha256": source_hashes,
            "observation_layout": {"hand_xyz": [0, 3], "active_goal_xyz": [36, 39]},
            "excluded_observation_slots": {
                "object_xyz": [4, 7],
                "reason": "not part of the hand-to-goal coordinate hypothesis",
            },
            **(
                {
                    "wall_center": REACH_WALL_CENTER.tolist(),
                    "wall_half_extents": REACH_WALL_HALF_EXTENTS.tolist(),
                }
                if task == "mw-reach-wall" else {}
            ),
        },
        **coordinate_cv(
            factors, targets, n_folds=min(args.folds, len(factors)), ridge=args.ridge, seed=args.seed
        ),
    }
    native_readout = fit_native_readout(factors, targets, args.ridge)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "model_native_coordinates.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        args.out / "model_native_coordinates.npz",
        site=np.asarray(site),
        factor_mean=native_readout["factor_mean"].astype(np.float32),
        target_mean=native_readout["target_mean"].astype(np.float32),
        readout_weights=native_readout["weights"].astype(np.float32),
        rowspace_basis=native_readout["rowspace_basis"].astype(np.float32),
        singular_values=native_readout["singular_values"].astype(np.float32),
        rank=np.asarray(native_readout["rank"], dtype=np.int32),
        eligible=np.asarray(report["model_native_relational_candidate"]),
        report_sha256=np.asarray(hashlib.sha256(path.read_bytes()).hexdigest()),
    )
    print(json.dumps({
        "site": report["site"],
        "goal_relative_advantage_fraction": report["primary_matched_comparison"]["candidate_advantage_fraction"],
        "model_native_relational_candidate": report["model_native_relational_candidate"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
