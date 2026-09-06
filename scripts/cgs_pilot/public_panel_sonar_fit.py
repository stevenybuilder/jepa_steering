#!/usr/bin/env python3
"""Fit the frozen Gaussian-energy / Gaussian-OT Sonar operator for Panel P.

This stage consumes one outcome-agnostic coordinate artifact from
``public_panel_factor_gate.py`` and native episode outcomes from a fitting capture.
Regimes are discovered without outcome labels.  Labels enter only when fitting the
regularized ``P(z | regime, local transition quality)`` densities that define
the intervention. Terminal episode outcomes remain evaluation endpoints and are
never copied onto earlier replans.

The output contains no selected dose or winning operator.  Energy-vs-OT,
angular-vs-radial-vs-joint, and dose are development-arm decisions that must be
frozen separately before a protected manifest is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from public_panel_factor_gate import (
    bootstrap_mean_ci,
    fit_diag_gmm,
    hmm_filter,
    load_site_sequences,
    log_gaussian_diag,
)
from public_panel_sonar_math import (
    contrastive_conceptor,
    fit_shrinkage_gaussian,
    gaussian_ot_affine,
    generalized_eigen_contrast,
    log_gaussian,
    matched_spectrum,
    outcome_energy_contrast_gradient,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transform(sequence: np.ndarray, coordinate: dict[str, np.ndarray]) -> np.ndarray:
    x = (np.asarray(sequence, dtype=np.float64) - coordinate["mean"]) @ coordinate["basis"]
    x = x / coordinate["scale"]
    if str(coordinate["representation"].item()) == "predictive_factors":
        x = x @ coordinate["factor_rotation"]
    return x


def coordinate_maps(coordinate: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    encoder = coordinate["basis"] / coordinate["scale"][None, :]
    if str(coordinate["representation"].item()) == "predictive_factors":
        encoder = encoder @ coordinate["factor_rotation"]
    decoder = np.linalg.pinv(encoder)
    if not np.allclose(decoder @ encoder, np.eye(encoder.shape[1]), atol=2e-5):
        raise RuntimeError("coordinate encoder/decoder failed the round-trip contract")
    return encoder, decoder


def static_responsibilities(rows: np.ndarray, model: dict[str, np.ndarray]) -> np.ndarray:
    joint = log_gaussian_diag(rows, model["means"], model["var"]) + np.log(model["weights"] + 1e-30)
    maximum = joint.max(axis=1, keepdims=True)
    return np.exp(joint - maximum) / np.exp(joint - maximum).sum(axis=1, keepdims=True)


def local_reach_progress_labels(
    capture: Path,
    index: dict,
    window: tuple[int, int],
    *,
    minimum_progress: float,
) -> tuple[list[np.ndarray], list[np.ndarray], dict]:
    """Label each physical replan by realized hand-to-goal progress.

    Hidden row ``t`` summarizes the world-model prediction for the selected
    action chunk at physical replan ``t``.  Its fitting label is therefore the
    simulator-space reduction in hand-to-goal distance from boundary ``t`` to
    boundary ``t+1``.  Terminal episode success is reported but is not copied
    backward onto earlier replans.
    """
    if minimum_progress < 0:
        raise ValueError("minimum_progress must be nonnegative")
    episode_logs = {
        int(row["ep"]): row
        for row in (
            json.loads(line)
            for line in (capture / "episodes.jsonl").read_text().splitlines()
            if line.strip()
        )
    }
    replan_indices = np.arange(window[0], window[1], dtype=np.int64)
    labels_by_episode: list[np.ndarray] = []
    progress_by_episode: list[np.ndarray] = []
    terminal_successes = 0
    for row in index["episodes"]:
        episode = episode_logs.get(int(row["ep"]))
        if episode is None or not episode.get("behavior_trace"):
            raise ValueError(f"episode {row['ep']} has no behavior trace for local labels")
        terminal_successes += int(bool(episode.get("success", row.get("success", False))))
        with np.load(capture / episode["behavior_trace"]) as trace:
            required = {"simulator_states", "goal_state", "plan_boundaries"}
            if not required.issubset(trace.files):
                raise ValueError(
                    f"episode {row['ep']} trace lacks {sorted(required - set(trace.files))}"
                )
            states = np.asarray(trace["simulator_states"], dtype=np.float64)
            boundaries = np.asarray(trace["plan_boundaries"], dtype=np.int64)
            goal_state = np.asarray(trace["goal_state"], dtype=np.float64)
        if states.ndim != 2 or states.shape[1] != 39 or goal_state.shape != (39,):
            raise ValueError("MetaWorld Reach progress labels require state [T,39] and goal [39]")
        if len(boundaries) <= window[1]:
            raise ValueError(f"episode {row['ep']} has insufficient plan boundaries")
        starts, stops = boundaries[replan_indices], boundaries[replan_indices + 1]
        if np.max(stops, initial=0) >= len(states):
            raise ValueError(f"episode {row['ep']} has insufficient simulator states")
        active_goal = states[:, -3:]
        if not np.allclose(active_goal[np.r_[starts, stops]], goal_state[-3:], atol=2e-5, rtol=0.0):
            raise ValueError(f"episode {row['ep']} simulator goal does not match expert goal")
        start_distance = np.linalg.norm(active_goal[starts] - states[starts, :3], axis=1)
        stop_distance = np.linalg.norm(active_goal[stops] - states[stops, :3], axis=1)
        progress = start_distance - stop_distance
        labels_by_episode.append((progress > minimum_progress).astype(np.int8))
        progress_by_episode.append(progress)
    label_rows = np.concatenate(labels_by_episode)
    progress_rows = np.concatenate(progress_by_episode)
    episode_support = {
        str(value): int(sum(np.any(labels == value) for labels in labels_by_episode))
        for value in (0, 1)
    }
    report = {
        "mode": "realized_replan_progress",
        "adapter": "metaworld_reach_hand_to_goal_v1",
        "minimum_progress_meters": float(minimum_progress),
        "positive_semantics": "executed action chunk reduced hand-to-goal distance",
        "negative_semantics": "executed action chunk made insufficient or negative progress",
        "terminal_episode_outcome_used_for_fit": False,
        "terminal_successes_descriptive_only": int(terminal_successes),
        "terminal_failures_descriptive_only": int(len(index["episodes"]) - terminal_successes),
        "transition_counts": {str(value): int(np.sum(label_rows == value)) for value in (0, 1)},
        "contributing_episode_counts": episode_support,
        "progress_meters": {
            "median": float(np.median(progress_rows)),
            "p10": float(np.quantile(progress_rows, 0.10)),
            "p90": float(np.quantile(progress_rows, 0.90)),
        },
    }
    return labels_by_episode, progress_by_episode, report


def class_conditioned_rows(
    sequences: list[np.ndarray],
    label_sequences: list[np.ndarray],
    regime_by_episode: list[np.ndarray],
    outcome: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool one local class while retaining unit total mass per episode."""
    rows, weights, regimes = [], [], []
    for sequence, labels, regime in zip(sequences, label_sequences, regime_by_episode):
        labels = np.asarray(labels, dtype=np.int8)
        if labels.shape != (len(sequence),) or regime.shape[0] != len(sequence):
            raise ValueError("local labels/regimes must align with every sequence row")
        keep = labels == outcome
        if np.any(keep):
            rows.append(sequence[keep])
            weights.append(np.full(int(keep.sum()), 1.0 / len(sequence), dtype=np.float64))
            regimes.append(regime[keep])
    if not rows:
        raise ValueError(f"local fitting class {outcome} has no rows")
    return np.concatenate(rows), np.concatenate(weights), np.concatenate(regimes)


def action_metric(
    action_sequences: list[np.ndarray] | None,
    sequences: list[np.ndarray],
    window: tuple[int, int],
    ridge: float,
    *,
    action_source: str,
    min_cv_explained: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """Estimate ``J_A^T J_A`` from aligned executed-control summaries."""
    if action_source != "exact_executed_control_chunk_summary":
        return np.zeros((sequences[0].shape[1], sequences[0].shape[1])), {
            "available": False,
            "reason": f"requires exact executed controls, got {action_source}",
        }
    if action_sequences is None or len(action_sequences) != len(sequences):
        return np.zeros((sequences[0].shape[1], sequences[0].shape[1])), {
            "available": False,
            "reason": "capture has no aligned executed-control chunks",
        }
    actions = []
    for action, sequence in zip(action_sequences, sequences):
        selected = np.asarray(action[window[0] : window[1]], dtype=np.float64)
        if len(selected) != len(sequence):
            return np.zeros((sequence.shape[1], sequence.shape[1])), {
                "available": False,
                "reason": "executed-control chunks do not align with the frozen window",
            }
        actions.append(np.nan_to_num(selected.reshape(len(selected), -1)))
    z = np.concatenate(sequences)
    a = np.concatenate(actions)
    cv_squared_error = 0.0
    cv_denominator = 0.0
    for held in range(len(sequences)):
        train_z = np.concatenate([value for i, value in enumerate(sequences) if i != held])
        train_a = np.concatenate([value for i, value in enumerate(actions) if i != held])
        z_mean, a_mean = train_z.mean(0), train_a.mean(0)
        zc_train, ac_train = train_z - z_mean, train_a - a_mean
        fold_weights = np.linalg.solve(
            zc_train.T @ zc_train + ridge * np.eye(z.shape[1]), zc_train.T @ ac_train
        )
        prediction = a_mean + (sequences[held] - z_mean) @ fold_weights
        cv_squared_error += float(np.square(actions[held] - prediction).sum())
        cv_denominator += float(np.square(actions[held] - a_mean).sum())
    cv_explained = 1.0 - cv_squared_error / cv_denominator if cv_denominator > 0 else 0.0
    if cv_explained < min_cv_explained:
        return np.zeros((z.shape[1], z.shape[1])), {
            "available": False,
            "reason": (
                f"action readout failed whole-episode CV: explained={cv_explained:.6f} "
                f"< {min_cv_explained:.6f}"
            ),
            "whole_episode_cv_variance_explained": float(cv_explained),
        }
    zc, ac = z - z.mean(0), a - a.mean(0)
    weights = np.linalg.solve(zc.T @ zc + ridge * np.eye(z.shape[1]), zc.T @ ac)
    prediction = zc @ weights
    denominator = float(np.square(ac).sum())
    explained = 1.0 - float(np.square(ac - prediction).sum()) / denominator if denominator > 0 else 0.0
    return weights @ weights.T, {
        "available": True,
        "source": "exact_executed_control_chunk_summary",
        "action_dim_flat": int(a.shape[1]),
        "ridge": float(ridge),
        "whole_episode_cv_variance_explained": float(cv_explained),
        "in_sample_variance_explained_descriptive_only": float(explained),
    }


def operator_cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.sum(left * right) / denominator) if denominator > 0 else float("nan")


def bootstrap_conceptor_stability(
    sequences: list[np.ndarray],
    label_sequences: list[np.ndarray],
    regime_responsibilities: list[np.ndarray],
    reference_global: np.ndarray,
    reference_local: np.ndarray,
    *,
    aperture: float,
    rcond: float,
    draws: int,
    seed: int,
) -> dict:
    """Whole-episode bootstrap stability for local-quality conceptors."""
    rng = np.random.default_rng(seed)
    global_cosines, local_cosines = [], []
    failures = 0
    failure_reasons: dict[str, int] = {}
    for _ in range(draws):
        sampled = rng.choice(len(sequences), len(sequences), replace=True)
        boot_sequences = [sequences[int(i)] for i in sampled]
        boot_labels = [label_sequences[int(i)] for i in sampled]
        boot_regime = [regime_responsibilities[int(i)] for i in sampled]
        class_rows, class_weights, class_regime = {}, {}, {}
        for outcome in (0, 1):
            (
                class_rows[outcome],
                class_weights[outcome],
                class_regime[outcome],
            ) = class_conditioned_rows(
                boot_sequences, boot_labels, boot_regime, outcome
            )
        try:
            fitted_global = contrastive_conceptor(
                class_rows[1], class_rows[0], aperture,
                success_weights=class_weights[1], failure_weights=class_weights[0], rcond=rcond,
            )["contrastive"]
            global_cosines.append(operator_cosine(reference_global, fitted_global))
            one_draw = []
            for state in range(reference_local.shape[0]):
                fitted = contrastive_conceptor(
                    class_rows[1], class_rows[0], aperture,
                    success_weights=class_weights[1] * class_regime[1][:, state],
                    failure_weights=class_weights[0] * class_regime[0][:, state],
                    rcond=rcond,
                )["contrastive"]
                one_draw.append(operator_cosine(reference_local[state], fitted))
            local_cosines.append(one_draw)
        except (ValueError, np.linalg.LinAlgError) as exc:
            failures += 1
            reason = f"{type(exc).__name__}: {exc}"
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    global_values = np.asarray(global_cosines, dtype=np.float64)
    local_values = np.asarray(local_cosines, dtype=np.float64)
    return {
        "draws_requested": int(draws),
        "draws_valid": int(len(global_values)),
        "draws_failed_numerically": int(failures),
        "failure_reasons": failure_reasons,
        "global_operator_cosine_median": float(np.nanmedian(global_values)) if len(global_values) else None,
        "global_operator_cosine_p05": float(np.nanquantile(global_values, 0.05)) if len(global_values) else None,
        "local_operator_cosine_median": (
            np.nanmedian(local_values, axis=0).tolist() if len(local_values) else []
        ),
        "local_operator_cosine_p05": (
            np.nanquantile(local_values, 0.05, axis=0).tolist() if len(local_values) else []
        ),
    }


def fit_outcome_models_from_episodes(
    sequences: list[np.ndarray],
    label_sequences: list[np.ndarray],
    regime_by_episode: list[np.ndarray],
    *,
    shrinkage: float,
    covariance_floor: float,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, list[float]]]:
    """Fit episode-balanced local-quality/regime Gaussians."""
    n_regimes = regime_by_episode[0].shape[1]
    outcome_models: dict[str, dict[str, np.ndarray]] = {}
    effective_n: dict[str, list[float]] = {}
    for outcome, name in ((1, "success"), (0, "failure")):
        class_rows, class_episode_weights, class_resp = class_conditioned_rows(
            sequences, label_sequences, regime_by_episode, outcome
        )
        weighted_resp = class_resp * class_episode_weights[:, None]
        mixture_weights = (weighted_resp.sum(0) + 1.0) / (weighted_resp.sum() + n_regimes)
        means, covariances, neff = [], [], []
        for state in range(n_regimes):
            mean, covariance, n_eff = fit_shrinkage_gaussian(
                class_rows,
                weights=weighted_resp[:, state],
                shrinkage=shrinkage,
                floor=covariance_floor,
            )
            means.append(mean)
            covariances.append(covariance)
            neff.append(n_eff)
        outcome_models[name] = {
            "weights": np.asarray(mixture_weights),
            "means": np.stack(means),
            "covariances": np.stack(covariances),
        }
        effective_n[name] = neff
    return outcome_models, effective_n


def ot_displacement_field(
    rows: np.ndarray,
    regime_responsibilities: np.ndarray,
    outcome_models: dict[str, dict[str, np.ndarray]],
    *,
    covariance_floor: float,
) -> np.ndarray:
    component = []
    for state in range(regime_responsibilities.shape[1]):
        affine, offset = gaussian_ot_affine(
            outcome_models["failure"]["means"][state],
            outcome_models["failure"]["covariances"][state],
            outcome_models["success"]["means"][state],
            outcome_models["success"]["covariances"][state],
            floor=covariance_floor,
        )
        transported = rows @ affine.T + offset
        component.append(transported - rows)
    return np.einsum("nk,knd->nd", regime_responsibilities, np.stack(component))


def median_row_cosine(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref_norm = np.linalg.norm(reference, axis=1)
    candidate_norm = np.linalg.norm(candidate, axis=1)
    live = (ref_norm > 1e-8) & (candidate_norm > 1e-8)
    if live.sum() < max(3, int(0.2 * len(reference))):
        raise ValueError("too few nonzero rows for a direction-stability estimate")
    cosine = np.einsum("nd,nd->n", reference[live], candidate[live]) / (
        ref_norm[live] * candidate_norm[live]
    )
    return float(np.median(cosine))


def bootstrap_density_stability(
    sequences: list[np.ndarray],
    label_sequences: list[np.ndarray],
    regime_by_episode: list[np.ndarray],
    reference_models: dict[str, dict[str, np.ndarray]],
    *,
    shrinkage: float,
    covariance_floor: float,
    draws: int,
    seed: int,
    minimum_cosine: float,
) -> dict:
    """Episode-bootstrap stability of energy gradients, OT fields, and precisions."""
    rows = np.concatenate(sequences)
    regime_rows = np.concatenate(regime_by_episode)
    _energy, reference_gradient, _responsibility = outcome_energy_contrast_gradient(
        rows, reference_models["success"], reference_models["failure"]
    )
    reference_ot = ot_displacement_field(
        rows, regime_rows, reference_models, covariance_floor=covariance_floor
    )
    rng = np.random.default_rng(seed)
    gradient_cosine, ot_cosine, precision_cosine = [], [], []
    failures = 0
    for _draw in range(draws):
        sampled = rng.choice(len(sequences), len(sequences), replace=True)
        boot_sequences = [sequences[int(i)] for i in sampled]
        boot_labels = [label_sequences[int(i)] for i in sampled]
        boot_regime = [regime_by_episode[int(i)] for i in sampled]
        try:
            fitted, _neff = fit_outcome_models_from_episodes(
                boot_sequences,
                boot_labels,
                boot_regime,
                shrinkage=shrinkage,
                covariance_floor=covariance_floor,
            )
            _value, gradient, _resp = outcome_energy_contrast_gradient(
                rows, fitted["success"], fitted["failure"]
            )
            ot_field = ot_displacement_field(
                rows, regime_rows, fitted, covariance_floor=covariance_floor
            )
            gradient_cosine.append(median_row_cosine(reference_gradient, gradient))
            ot_cosine.append(median_row_cosine(reference_ot, ot_field))
            one_precision = []
            for outcome in ("success", "failure"):
                for state in range(regime_rows.shape[1]):
                    reference_precision = np.linalg.inv(reference_models[outcome]["covariances"][state])
                    candidate_precision = np.linalg.inv(fitted[outcome]["covariances"][state])
                    one_precision.append(operator_cosine(reference_precision, candidate_precision))
            if not np.isfinite(one_precision).all():
                raise ValueError("non-finite precision cosine")
            precision_cosine.append(float(np.median(one_precision)))
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            failures += 1

    def summary(values: list[float]) -> dict[str, float | None]:
        return {
            "median": float(np.median(values)) if values else None,
            "p05": float(np.quantile(values, 0.05)) if values else None,
        }

    valid = len(gradient_cosine)
    energy_p05 = summary(gradient_cosine)["p05"]
    ot_p05 = summary(ot_cosine)["p05"]
    enough = valid >= max(20, int(0.8 * draws))
    return {
        "draws_requested": int(draws),
        "draws_valid": int(valid),
        "draws_failed_numerically": int(failures),
        "energy_gradient_median_row_cosine": summary(gradient_cosine),
        "ot_displacement_median_row_cosine": summary(ot_cosine),
        "precision_operator_cosine": summary(precision_cosine),
        "minimum_p05_cosine": float(minimum_cosine),
        "energy_eligible": bool(enough and energy_p05 is not None and energy_p05 >= minimum_cosine),
        "ot_eligible": bool(enough and ot_p05 is not None and ot_p05 >= minimum_cosine),
    }


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = np.asarray(scores)[np.asarray(labels) == 1]
    negative = np.asarray(scores)[np.asarray(labels) == 0]
    if not len(positive) or not len(negative):
        return float("nan")
    comparisons = positive[:, None] - negative[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


def routed_local_quality_cv(
    sequences: list[np.ndarray],
    label_sequences: list[np.ndarray],
    regime: dict[str, np.ndarray],
    *,
    actions: list[np.ndarray] | None,
    action_hmm: dict | None,
    cross_fitted_hmm_route: np.ndarray | None,
    shrinkage: float,
    covariance_floor: float,
    seed: int,
) -> dict:
    """Compare local transition-quality prediction under static versus HMM routing.

    The class/regime densities are identical within each fold. Only the routing
    weights differ. This is leave-one-episode-out and never uses future
    activations or terminal episode labels.
    """
    if len(regime["weights"]) < 2:
        return {"decidable": False, "reason": "one-regime model has no routing ablation"}
    if actions is None or action_hmm is None or cross_fitted_hmm_route is None:
        return {"decidable": False, "reason": "no aligned action-conditioned router"}
    if len(actions) != len(sequences) or any(len(action) != len(sequence) for action, sequence in zip(actions, sequences)):
        return {"decidable": False, "reason": "action-conditioned routing rows do not align by episode"}
    from public_panel_action_hmm import hmm_filter_conditioned, model_features

    hmm_model = {key: np.asarray(value) if key != "feature_mode" else str(value) for key, value in action_hmm.items()}
    if not (
        np.allclose(hmm_model["means"], regime["means"], atol=1e-7, rtol=1e-7)
        and np.allclose(hmm_model["var"], regime["var"], atol=1e-7, rtol=1e-7)
    ):
        return {"decidable": False, "reason": "action-HMM emissions do not match shared operator regimes"}
    static_route = [static_responsibilities(sequence[:, : regime["means"].shape[1]], regime) for sequence in sequences]
    hmm_route = [np.asarray(value, dtype=np.float64) for value in cross_fitted_hmm_route]
    if len(hmm_route) != len(sequences) or any(
        value.shape != (len(sequence), len(regime["weights"]))
        for value, sequence in zip(hmm_route, sequences)
    ):
        return {"decidable": False, "reason": "cross-fitted HMM beliefs do not align by episode"}
    losses = {"static": np.full(len(sequences), np.nan), "hmm": np.full(len(sequences), np.nan)}
    probabilities = {"static": [], "hmm": []}
    for held in range(len(sequences)):
        train_ids = np.asarray([i for i in range(len(sequences)) if i != held], dtype=np.int64)
        contributing = {
            value: sum(np.any(label_sequences[int(i)] == value) for i in train_ids)
            for value in (0, 1)
        }
        if min(contributing.values()) < 2:
            return {
                "decidable": False,
                "reason": "local-quality class episode support fails inside leave-one-episode-out CV",
            }
        train_sequences = [sequences[int(i)] for i in train_ids]
        train_labels = [label_sequences[int(i)] for i in train_ids]
        train_regime = [static_route[int(i)] for i in train_ids]
        fitted, _effective = fit_outcome_models_from_episodes(
            train_sequences,
            train_labels,
            train_regime,
            shrinkage=shrinkage,
            covariance_floor=covariance_floor,
        )
        x = sequences[held]
        per_regime_llr = []
        for state in range(len(regime["weights"])):
            # The route already supplies P(regime | x_1:t). Multiplying by a
            # second class-specific regime prior here would double-count regime
            # occupancy and would no longer isolate static versus HMM routing.
            success_log = log_gaussian(
                x,
                fitted["success"]["means"][state],
                fitted["success"]["covariances"][state],
            )
            failure_log = log_gaussian(
                x,
                fitted["failure"]["means"][state],
                fitted["failure"]["covariances"][state],
            )
            per_regime_llr.append(success_log - failure_log)
        per_regime_llr = np.stack(per_regime_llr, axis=1)
        positive_episode_mass = sum(float(np.mean(value)) for value in train_labels)
        prior = (positive_episode_mass + 1.0) / (len(train_labels) + 2.0)
        prior_logit = np.log(prior / (1.0 - prior))
        for name, routes in (("static", static_route), ("hmm", hmm_route)):
            row_logit = prior_logit + np.sum(routes[held] * per_regime_llr, axis=1)
            probability = np.clip(
                1.0 / (1.0 + np.exp(-np.clip(row_logit, -40.0, 40.0))),
                1e-8,
                1.0 - 1e-8,
            )
            probabilities[name].append(probability)
            target = np.asarray(label_sequences[held], dtype=np.float64)
            losses[name][held] = float(np.mean(
                -(target * np.log(probability) + (1.0 - target) * np.log(1.0 - probability))
            ))
    improvement = losses["static"] - losses["hmm"]
    interval = bootstrap_mean_ci(improvement, seed + 701)
    return {
        "decidable": True,
        "causal_filter_no_future": True,
        "router": "action_conditioned",
        "transition_router_cross_fitted_without_heldout_episode": True,
        "shared_emissions_with_static_operator_bank": True,
        "same_fold_local_densities_for_both_routes": True,
        "terminal_episode_outcome_used": False,
        "mean_log_loss": {name: float(value.mean()) for name, value in losses.items()},
        "per_episode_log_loss": {name: value.tolist() for name, value in losses.items()},
        "per_episode_mean_probability": {
            name: [float(np.mean(value)) for value in rows]
            for name, rows in probabilities.items()
        },
        "transition_auc_descriptive": {
            name: binary_auc(np.concatenate(label_sequences), np.concatenate(rows))
            for name, rows in probabilities.items()
        },
        "static_minus_hmm_log_loss": {
            "mean": float(improvement.mean()), "episode_bootstrap_95_ci": list(interval)
        },
        "hmm_routing_improves_local_quality_prediction": bool(
            improvement.mean() >= 0.01 and interval[0] > 0.0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--coordinates", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--pool", default="mean", choices=["mean", "mean_all"])
    parser.add_argument(
        "--label-adapter",
        default="metaworld_reach_progress",
        choices=["metaworld_reach_progress"],
        help="task-native per-replan fitting label; terminal episode labels are not eligible",
    )
    parser.add_argument("--minimum-progress-meters", type=float, default=1e-4)
    parser.add_argument("--regimes", type=int, default=2)
    parser.add_argument("--shrinkage", type=float, default=0.25)
    parser.add_argument("--covariance-floor", type=float, default=1e-3)
    parser.add_argument("--min-per-class", type=int, default=3)
    parser.add_argument("--action-ridge", type=float, default=1.0)
    parser.add_argument("--lambda-action", type=float, default=0.25)
    parser.add_argument("--min-action-cv-variance-explained", type=float, default=0.0)
    parser.add_argument("--lambda-support", type=float, default=0.10)
    parser.add_argument("--support-quantile", type=float, default=0.99)
    parser.add_argument("--conceptor-aperture", type=float, default=1.0)
    parser.add_argument("--conceptor-rcond", type=float, default=1e-12)
    parser.add_argument("--bootstrap-draws", type=int, default=200)
    parser.add_argument("--min-conceptor-bootstrap-cosine", type=float, default=0.75)
    parser.add_argument("--min-density-bootstrap-cosine", type=float, default=0.60)
    parser.add_argument(
        "--frankenstein-modules",
        type=Path,
        default=None,
        help="optional outcome-free frankenstein_modules.npz to bind into the runtime operator",
    )
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()

    frankenstein_arrays: dict[str, np.ndarray] = {}
    frankenstein_source = None
    if args.frankenstein_modules is not None:
        with np.load(args.frankenstein_modules) as raw:
            frankenstein_arrays = {
                f"frankenstein_{key}": np.asarray(raw[key]) for key in raw.files if key != "site"
            }
            frankenstein_site = str(raw["site"].item())
        frankenstein_source = {
            "path": str(args.frankenstein_modules.resolve()),
            "sha256": sha256_file(args.frankenstein_modules),
            "site": frankenstein_site,
        }

    args.out.mkdir(parents=True, exist_ok=True)
    with np.load(args.coordinates) as raw:
        coordinate = {key: np.asarray(raw[key]) for key in raw.files}
    site = str(coordinate["site"].item())
    if frankenstein_source is not None and frankenstein_source["site"] != site:
        raise SystemExit(
            f"Frankenstein artifact site {frankenstein_source['site']} does not match coordinate site {site}"
        )
    sequences_h, action_sequences, index = load_site_sequences(args.capture, site, args.pool)
    terminal_labels = np.asarray([int(row["success"]) for row in index["episodes"]], dtype=np.int8)
    window_raw = index.get("pre_outcome_window") or [0, min(len(x) for x in sequences_h)]
    window = (int(window_raw[0]), int(window_raw[1]))
    if window[1] - window[0] < 2:
        raise SystemExit(f"pre-outcome window too short: {window}")
    sequences = [transform(sequence[window[0] : window[1]], coordinate) for sequence in sequences_h]
    label_sequences, progress_sequences, label_report = local_reach_progress_labels(
        args.capture,
        index,
        window,
        minimum_progress=args.minimum_progress_meters,
    )
    contributing = label_report["contributing_episode_counts"]
    if min(contributing.values()) < args.min_per_class:
        raise SystemExit(
            f"local-quality episode support failed: {contributing}, need {args.min_per_class} per class"
        )
    rows = np.concatenate(sequences)
    row_labels = np.concatenate(label_sequences)
    episode_weights = np.concatenate(
        [np.full(len(sequence), 1.0 / len(sequence), dtype=np.float64) for sequence in sequences]
    )
    regime_keys = {"regime_weights", "regime_means", "regime_var"}
    if regime_keys.issubset(coordinate):
        regime = {key.removeprefix("regime_"): coordinate[key].astype(np.float64) for key in regime_keys}
        regime_source = str(coordinate.get("regime_source", np.asarray("factor_gate")).item())
    else:
        regime = fit_diag_gmm(rows, args.regimes, args.seed)
        regime_source = "sonar_fit_static_gmm_legacy_fallback"
    n_regimes = int(len(regime["weights"]))
    emission_dim = int(regime["means"].shape[1])
    if emission_dim > rows.shape[1]:
        raise SystemExit(f"regime emission width {emission_dim} exceeds coordinate width {rows.shape[1]}")
    regime_rows = rows[:, :emission_dim]
    regime_resp = static_responsibilities(regime_rows, regime)
    encoder, decoder = coordinate_maps(coordinate)
    cursor = 0
    regime_by_episode = []
    for sequence in sequences:
        regime_by_episode.append(regime_resp[cursor : cursor + len(sequence)])
        cursor += len(sequence)
    outcome_models, effective_n = fit_outcome_models_from_episodes(
        sequences,
        label_sequences,
        regime_by_episode,
        shrinkage=args.shrinkage,
        covariance_floor=args.covariance_floor,
    )

    ot_affine, ot_offset, ge_values, ge_vectors = [], [], [], []
    metrics, support_thresholds = [], []
    action_gram, action_report = action_metric(
        action_sequences,
        sequences,
        window,
        args.action_ridge,
        action_source=index.get("action_source", "mixed_or_unavailable"),
        min_cv_explained=args.min_action_cv_variance_explained,
    )
    if args.lambda_action > 0 and not action_report.get("available"):
        raise SystemExit(
            f"action-sensitivity metric required when lambda_action>0: {action_report.get('reason')}"
        )
    episode_regime_support = {"success": [], "failure": []}
    for outcome, name in ((1, "success"), (0, "failure")):
        for state in range(n_regimes):
            mass = np.asarray([
                np.mean((labels == outcome) * responsibility[:, state])
                for labels, responsibility in zip(label_sequences, regime_by_episode)
            ])
            effective = float(np.square(mass.sum()) / max(np.square(mass).sum(), 1e-30))
            episode_regime_support[name].append(effective)
            if effective < args.min_per_class:
                raise SystemExit(
                    f"regime {state} {name} episode-support floor failed: "
                    f"effective_n={effective:.3f}, need {args.min_per_class}"
                )

    for state in range(n_regimes):
        failure_cov = outcome_models["failure"]["covariances"][state]
        success_cov = outcome_models["success"]["covariances"][state]
        affine, offset = gaussian_ot_affine(
            outcome_models["failure"]["means"][state], failure_cov,
            outcome_models["success"]["means"][state], success_cov,
            floor=args.covariance_floor,
        )
        values, vectors = generalized_eigen_contrast(success_cov, failure_cov, floor=args.covariance_floor)
        support_precision = np.ones(rows.shape[1], dtype=np.float64)
        support_precision[:emission_dim] = 1.0 / np.maximum(
            regime["var"][state], args.covariance_floor
        )
        metric = (
            np.eye(rows.shape[1])
            + args.lambda_action * action_gram
            + args.lambda_support * np.diag(support_precision)
        )
        mahal = np.sum(
            np.square(
                (regime_rows - regime["means"][state]) / np.sqrt(regime["var"][state])
            ),
            axis=1,
        )
        # Responsibility weighting keeps the threshold tied to this regime without hard labels.
        order = np.argsort(mahal)
        support_weight = regime_resp[:, state] * episode_weights
        cumulative = np.cumsum(support_weight[order]); cumulative /= cumulative[-1]
        threshold = mahal[order[min(int(np.searchsorted(cumulative, args.support_quantile)), len(order) - 1)]]
        ot_affine.append(affine); ot_offset.append(offset)
        ge_values.append(values); ge_vectors.append(vectors)
        metrics.append(metric); support_thresholds.append(threshold)

    success_mask, failure_mask = row_labels == 1, row_labels == 0
    global_conceptor = contrastive_conceptor(
        rows[success_mask], rows[failure_mask], args.conceptor_aperture,
        success_weights=episode_weights[success_mask],
        failure_weights=episode_weights[failure_mask],
        rcond=args.conceptor_rcond,
    )
    local_conceptors = []
    local_success_conceptors, local_failure_conceptors = [], []
    local_success_centers, local_failure_centers = [], []
    for state in range(n_regimes):
        fitted = contrastive_conceptor(
            rows[success_mask], rows[failure_mask], args.conceptor_aperture,
            success_weights=episode_weights[success_mask] * regime_resp[success_mask, state],
            failure_weights=episode_weights[failure_mask] * regime_resp[failure_mask, state],
            rcond=args.conceptor_rcond,
        )
        local_conceptors.append(fitted["contrastive"])
        local_success_conceptors.append(fitted["success"])
        local_failure_conceptors.append(fitted["failure"])
        local_success_centers.append(fitted["success_center"])
        local_failure_centers.append(fitted["failure_center"])
    local_conceptors = np.stack(local_conceptors)
    conceptor_stability = bootstrap_conceptor_stability(
        sequences,
        label_sequences,
        regime_by_episode,
        global_conceptor["contrastive"],
        local_conceptors,
        aperture=args.conceptor_aperture,
        rcond=args.conceptor_rcond,
        draws=args.bootstrap_draws,
        seed=args.seed + 127,
    )
    stability_values = [conceptor_stability["global_operator_cosine_p05"]] + list(
        conceptor_stability["local_operator_cosine_p05"]
    )
    if (
        conceptor_stability["draws_valid"] < max(20, int(0.8 * args.bootstrap_draws))
        or any(
            value is None or not np.isfinite(value) or value < args.min_conceptor_bootstrap_cosine
            for value in stability_values
        )
    ):
        raise SystemExit(f"conceptor episode-bootstrap stability gate failed: {conceptor_stability}")
    density_stability = bootstrap_density_stability(
        sequences,
        label_sequences,
        regime_by_episode,
        outcome_models,
        shrinkage=args.shrinkage,
        covariance_floor=args.covariance_floor,
        draws=args.bootstrap_draws,
        seed=args.seed + 131,
        minimum_cosine=args.min_density_bootstrap_cosine,
    )
    action_window = (
        [value[window[0] : window[1]] for value in action_sequences]
        if action_sequences is not None else None
    )
    action_hmm_keys = {
        "action_hmm_initial", "action_hmm_transition_coef", "action_hmm_action_mean",
        "action_hmm_action_scale", "action_hmm_means", "action_hmm_var",
        "action_hmm_feature_mode", "action_hmm_cross_fitted_beliefs",
    }
    action_hmm = None
    if regime_source == "hmm_emissions" and action_hmm_keys.issubset(coordinate):
        action_hmm = {
            "initial": coordinate["action_hmm_initial"],
            "transition_coef": coordinate["action_hmm_transition_coef"],
            "action_mean": coordinate["action_hmm_action_mean"],
            "action_scale": coordinate["action_hmm_action_scale"],
            "means": coordinate["action_hmm_means"],
            "var": coordinate["action_hmm_var"],
            "feature_mode": str(coordinate["action_hmm_feature_mode"].item()),
        }
    local_quality_routing = routed_local_quality_cv(
        sequences,
        label_sequences,
        regime,
        actions=action_window,
        action_hmm=action_hmm,
        cross_fitted_hmm_route=(
            coordinate["action_hmm_cross_fitted_beliefs"] if action_hmm is not None else None
        ),
        shrinkage=args.shrinkage,
        covariance_floor=args.covariance_floor,
        seed=args.seed + 137,
    )
    local_sham_conceptors = np.stack(
        [matched_spectrum(value, seed=args.seed + 91 + state) for state, value in enumerate(local_conceptors)]
    )
    label_matrix = np.stack(label_sequences)
    shuffled_matrix = np.empty_like(label_matrix)
    shuffle_rng = np.random.default_rng(args.seed + 113)
    for physical_replan in range(label_matrix.shape[1]):
        shuffled_matrix[:, physical_replan] = label_matrix[
            shuffle_rng.permutation(len(label_matrix)), physical_replan
        ]
    shuffled_row_labels = shuffled_matrix.reshape(-1)
    shuffled_global = contrastive_conceptor(
        rows[shuffled_row_labels == 1], rows[shuffled_row_labels == 0], args.conceptor_aperture,
        success_weights=episode_weights[shuffled_row_labels == 1],
        failure_weights=episode_weights[shuffled_row_labels == 0],
        rcond=args.conceptor_rcond,
    )["contrastive"]
    shuffled_local = []
    for state in range(n_regimes):
        shuffled_local.append(
            contrastive_conceptor(
                rows[shuffled_row_labels == 1], rows[shuffled_row_labels == 0], args.conceptor_aperture,
                success_weights=(
                    episode_weights[shuffled_row_labels == 1]
                    * regime_resp[shuffled_row_labels == 1, state]
                ),
                failure_weights=(
                    episode_weights[shuffled_row_labels == 0]
                    * regime_resp[shuffled_row_labels == 0, state]
                ),
                rcond=args.conceptor_rcond,
            )["contrastive"]
        )
    soft_conceptor = global_conceptor["success"]
    sham_conceptor = matched_spectrum(global_conceptor["contrastive"], seed=args.seed + 81)
    sham_rotation, _ = np.linalg.qr(np.random.default_rng(args.seed + 17).normal(size=(rows.shape[1], rows.shape[1])))

    artifact = args.out / "sonar_operator.npz"
    np.savez_compressed(
        artifact,
        site=np.asarray(site),
        pool=np.asarray(args.pool),
        fit_label_adapter=np.asarray(args.label_adapter),
        fit_label_positive_semantics=np.asarray("realized productive transition"),
        minimum_progress_meters=np.asarray(args.minimum_progress_meters, dtype=np.float64),
        terminal_episode_outcome_used_for_fit=np.asarray(False),
        representation=coordinate["representation"],
        hidden_mean=coordinate["mean"].astype(np.float32),
        encoder=encoder.astype(np.float32),
        decoder=decoder.astype(np.float32),
        regime_weights=regime["weights"].astype(np.float32),
        regime_means=regime["means"].astype(np.float32),
        regime_var=regime["var"].astype(np.float32),
        success_weights=outcome_models["success"]["weights"].astype(np.float32),
        success_means=outcome_models["success"]["means"].astype(np.float32),
        success_covariances=outcome_models["success"]["covariances"].astype(np.float32),
        failure_weights=outcome_models["failure"]["weights"].astype(np.float32),
        failure_means=outcome_models["failure"]["means"].astype(np.float32),
        failure_covariances=outcome_models["failure"]["covariances"].astype(np.float32),
        trust_metrics=np.stack(metrics).astype(np.float32),
        action_gram=action_gram.astype(np.float32),
        support_thresholds=np.asarray(support_thresholds, dtype=np.float32),
        ot_affine=np.stack(ot_affine).astype(np.float32),
        ot_offset=np.stack(ot_offset).astype(np.float32),
        generalized_eigenvalues=np.stack(ge_values).astype(np.float32),
        generalized_eigenvectors=np.stack(ge_vectors).astype(np.float32),
        success_conceptor=soft_conceptor.astype(np.float32),
        matched_spectrum_conceptor=sham_conceptor.astype(np.float32),
        global_contrastive_conceptor=global_conceptor["contrastive"].astype(np.float32),
        global_success_center=global_conceptor["success_center"].astype(np.float32),
        global_failure_center=global_conceptor["failure_center"].astype(np.float32),
        local_contrastive_conceptors=local_conceptors.astype(np.float32),
        local_success_conceptors=np.stack(local_success_conceptors).astype(np.float32),
        local_failure_conceptors=np.stack(local_failure_conceptors).astype(np.float32),
        local_success_centers=np.stack(local_success_centers).astype(np.float32),
        local_failure_centers=np.stack(local_failure_centers).astype(np.float32),
        local_matched_spectrum_conceptors=local_sham_conceptors.astype(np.float32),
        global_label_shuffled_conceptor=shuffled_global.astype(np.float32),
        local_label_shuffled_conceptors=np.stack(shuffled_local).astype(np.float32),
        conceptor_aperture=np.asarray(args.conceptor_aperture, dtype=np.float32),
        conceptor_rcond=np.asarray(args.conceptor_rcond, dtype=np.float64),
        shrinkage=np.asarray(args.shrinkage, dtype=np.float32),
        covariance_floor=np.asarray(args.covariance_floor, dtype=np.float32),
        support_quantile=np.asarray(args.support_quantile, dtype=np.float32),
        lambda_action=np.asarray(args.lambda_action, dtype=np.float32),
        lambda_support=np.asarray(args.lambda_support, dtype=np.float32),
        coordinate_rank=np.asarray(rows.shape[1], dtype=np.int32),
        regime_source=np.asarray(regime_source),
        regime_emission_dim=np.asarray(emission_dim, dtype=np.int32),
        hmm_initial=np.asarray(coordinate.get("hmm_initial", regime["weights"]), dtype=np.float32),
        hmm_transition=np.asarray(
            coordinate.get("hmm_transition", np.eye(n_regimes)), dtype=np.float32
        ),
        action_hmm_initial=np.asarray(
            coordinate.get("action_hmm_initial", np.empty(0)), dtype=np.float32
        ),
        action_hmm_transition_coef=np.asarray(
            coordinate.get("action_hmm_transition_coef", np.empty(0)), dtype=np.float32
        ),
        action_hmm_means=np.asarray(
            coordinate.get("action_hmm_means", np.empty(0)), dtype=np.float32
        ),
        action_hmm_var=np.asarray(
            coordinate.get("action_hmm_var", np.empty(0)), dtype=np.float32
        ),
        action_hmm_cross_fitted_beliefs=np.asarray(
            coordinate.get("action_hmm_cross_fitted_beliefs", np.empty(0)), dtype=np.float32
        ),
        action_hmm_action_mean=np.asarray(
            coordinate.get("action_hmm_action_mean", np.empty(0)), dtype=np.float32
        ),
        action_hmm_action_scale=np.asarray(
            coordinate.get("action_hmm_action_scale", np.empty(0)), dtype=np.float32
        ),
        action_hmm_feature_mode=np.asarray(
            coordinate.get("action_hmm_feature_mode", np.asarray("unavailable"))
        ),
        action_hmm_sequence_length=np.asarray(window[1] - window[0], dtype=np.int32),
        offline_hmm_gates_pass=np.asarray(
            bool(coordinate.get("offline_hmm_gates_pass", np.asarray(False)).item())
        ),
        hmm_deployment_eligible=np.asarray(
            bool(coordinate.get("hmm_deployment_eligible", np.asarray(False)).item())
        ),
        hmm_local_quality_routing_eligible=np.asarray(
            bool(local_quality_routing.get("hmm_routing_improves_local_quality_prediction", False))
        ),
        # Retained as a runtime compatibility key; its semantics are now local
        # transition quality, never copied terminal episode outcome.
        hmm_outcome_routing_eligible=np.asarray(
            bool(local_quality_routing.get("hmm_routing_improves_local_quality_prediction", False))
        ),
        conceptor_deployment_eligible=np.asarray(True),
        energy_deployment_eligible=np.asarray(density_stability["energy_eligible"]),
        ot_deployment_eligible=np.asarray(density_stability["ot_eligible"]),
        sham_rotation=sham_rotation.astype(np.float32),
        **frankenstein_arrays,
    )
    report = {
        "status": "fit_only_not_behaviorally_selected",
        "script_sha256": sha256_file(Path(__file__)),
        "capture": str(args.capture.resolve()),
        "capture_index_sha256": sha256_file(args.capture / "activations" / "index.json"),
        "coordinates": str(args.coordinates.resolve()),
        "coordinates_sha256": sha256_file(args.coordinates),
        "frankenstein_modules": frankenstein_source,
        "operator": str(artifact.resolve()),
        "operator_sha256": sha256_file(artifact),
        "site": site,
        "pool": args.pool,
        "representation": str(coordinate["representation"].item()),
        "pre_outcome_window": list(window),
        "n_episodes": len(terminal_labels),
        "fit_label": label_report,
        "fit_hyperparameters": {
            "coordinate_rank": int(rows.shape[1]),
            "requested_regimes": int(args.regimes),
            "selected_regimes": int(n_regimes),
            "shrinkage": float(args.shrinkage),
            "covariance_floor": float(args.covariance_floor),
            "minimum_contributing_episodes_per_local_class_regime": int(args.min_per_class),
            "minimum_progress_meters": float(args.minimum_progress_meters),
            "action_ridge": float(args.action_ridge),
            "lambda_action": float(args.lambda_action),
            "lambda_support": float(args.lambda_support),
            "support_quantile": float(args.support_quantile),
            "conceptor_aperture": float(args.conceptor_aperture),
            "conceptor_rcond": float(args.conceptor_rcond),
            "bootstrap_draws": int(args.bootstrap_draws),
            "minimum_conceptor_bootstrap_cosine": float(args.min_conceptor_bootstrap_cosine),
            "minimum_density_bootstrap_cosine": float(args.min_density_bootstrap_cosine),
            "seed": int(args.seed),
        },
        "regime_discovery_used_outcomes": False,
        "regime_source": regime_source,
        "regime_emission_dim": emission_dim,
        "episode_weighting": "each episode has unit mass before local-quality/regime weighting",
        "regime_occupancy": regime_resp.mean(0).tolist(),
        "episode_regime_effective_n": episode_regime_support,
        "local_quality_regime_effective_rows": effective_n,
        "action_sensitivity_metric": action_report,
        "conceptor": {
            "form": "C_productive_transition AND NOT C_unproductive_transition",
            "class_centering": "separate productive/unproductive transition centers",
            "runtime_centering": "none; multiplicative gate is applied about the origin",
            "aperture": float(args.conceptor_aperture),
            "rcond": float(args.conceptor_rcond),
            "coordinate_span": int(rows.shape[1]),
            "global_quota": float(np.trace(global_conceptor["contrastive"]) / rows.shape[1]),
            "local_quotas": [float(np.trace(value) / rows.shape[1]) for value in local_conceptors],
            "episode_bootstrap_stability": conceptor_stability,
            "minimum_p05_cosine": float(args.min_conceptor_bootstrap_cosine),
        },
        "density_operator_stability": density_stability,
        "hmm_vs_static_local_quality_routing": local_quality_routing,
        "operator_eligibility": {
            "factor_space_conceptor": True,
            "mixture_energy_capped_metric": density_stability["energy_eligible"],
            "gaussian_ot_factor_transport": density_stability["ot_eligible"],
            "hmm_local_routing": bool(
                coordinate.get("hmm_deployment_eligible", np.asarray(False)).item()
            ),
        },
        "operator_candidates": (
            ["factor_space_coast_global"]
            + (["factor_space_coast_static_local"] if n_regimes > 1 else [])
            + (["factor_space_coast_hmm_local"] if bool(
                coordinate.get("hmm_deployment_eligible", np.asarray(False)).item()
            ) else [])
            + (["mixture_energy_capped_metric"] if density_stability["energy_eligible"] else [])
            + (["gaussian_ot_factor_transport"] if density_stability["ot_eligible"] else [])
        ),
        "displacement_modes": ["angular", "radial", "joint"],
        "controls": [
            "identity", "reverse", "wrong_factor", "metric_norm_matched_sham",
            "regime_conditioned_matched_spectrum", "time_stratified_shuffled_progress_geometry",
        ],
        "not_claimed": [
            "the regimes are semantic concepts",
            "the JEPA is an HMM",
            "the fitted operator improves behavior",
            "the conceptor is a novel operator",
            "terminal success has been causally assigned to every earlier replan",
        ],
    }
    (args.out / "fit_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
