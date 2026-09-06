#!/usr/bin/env python3
"""Action-conditioned latent-regime dynamics for Panel P.

This module keeps the JEPA claim ladder explicit.  It does not assume that a
first-order predictor exposes a sufficient Markov state.  It first reuses the
outcome-free Gaussian emissions from ``public_panel_factor_gate``, then fits

    A_t[i,j] = softmax_j(W[i] phi(u_{t+1}, progress_t))

with Baum--Welch joint transition posteriors.  Whole episodes are the fitting,
cross-validation, bootstrap, and permutation units.  The online filter uses
only the current endpoint emission, past belief, and the control chunk that led
into that endpoint. The first captured endpoint uses the initial state prior.
"""

from __future__ import annotations

import numpy as np

from public_panel_factor_gate import (
    bootstrap_mean_ci,
    episode_folds,
    fit_diag_hmm,
    hmm_forward_backward,
    log_gaussian_diag,
    logsumexp,
)


def _softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=axis, keepdims=True), 1e-300)


def fit_control_scaler(actions: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Episode-equal action mean/scale for endpoint transitions t -> t+1."""
    width = actions[0].shape[1]
    episode_means = []
    episode_seconds = []
    for action in actions:
        rows = np.asarray(action[1:], dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != width:
            raise ValueError("action sequences must have a common [T,A] shape")
        episode_means.append(rows.mean(0))
        episode_seconds.append(np.square(rows).mean(0))
    mean = np.mean(episode_means, axis=0)
    variance = np.maximum(np.mean(episode_seconds, axis=0) - np.square(mean), 1e-8)
    return mean, np.sqrt(variance)


def control_features(
    action: np.ndarray,
    sequence_length: int,
    mean: np.ndarray,
    scale: np.ndarray,
    *,
    mode: str,
) -> np.ndarray:
    """Build causal endpoint-transition features; progress is an explicit control.

    Capture row ``t`` is the final prediction after plan ``t``. Consequently,
    the transition from stored row ``t`` to row ``t+1`` uses action chunk
    ``t+1``. Action zero led into the initial captured row and is represented by
    the initial state distribution rather than a learned stored-row transition.
    """
    if mode not in {"fixed", "progress", "action_progress"}:
        raise ValueError(mode)
    n = sequence_length - 1
    if n < 1 or len(action) < n:
        raise ValueError("actions must cover every latent transition")
    progress = np.arange(n, dtype=np.float64) / max(n, 1)
    pieces = [np.ones((n, 1), dtype=np.float64)]
    if mode == "action_progress":
        pieces.append((np.asarray(action[1 : n + 1], dtype=np.float64) - mean) / scale)
    if mode in {"progress", "action_progress"}:
        pieces.extend([progress[:, None], np.square(progress)[:, None]])
    return np.concatenate(pieces, axis=1)


def transition_matrices(features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Return row-stochastic ``[time, source, destination]`` matrices."""
    logits = np.einsum("tp,ijp->tij", np.asarray(features), np.asarray(coefficients))
    return _softmax(logits, axis=2)


def hmm_forward_backward_conditioned(
    sequence: np.ndarray,
    features: np.ndarray,
    model: dict,
) -> tuple[float, np.ndarray, np.ndarray]:
    emission = log_gaussian_diag(sequence, model["means"], model["var"])
    transition = transition_matrices(features, model["transition_coef"])
    if len(transition) != len(sequence) - 1:
        raise ValueError("one action-conditioned transition is required per state interval")
    log_transition = np.log(transition + 1e-300)
    alpha = np.empty_like(emission)
    alpha[0] = np.log(model["initial"] + 1e-300) + emission[0]
    for time in range(1, len(sequence)):
        alpha[time] = emission[time] + logsumexp(
            alpha[time - 1][:, None] + log_transition[time - 1], axis=0
        )
    likelihood = float(logsumexp(alpha[-1], axis=0))
    beta = np.zeros_like(emission)
    for time in range(len(sequence) - 2, -1, -1):
        beta[time] = logsumexp(
            log_transition[time] + emission[time + 1][None, :] + beta[time + 1][None, :],
            axis=1,
        )
    gamma = np.exp(alpha + beta - likelihood)
    gamma /= np.maximum(gamma.sum(1, keepdims=True), 1e-300)
    xi = np.empty((len(sequence) - 1, emission.shape[1], emission.shape[1]))
    for time in range(len(sequence) - 1):
        log_xi = (
            alpha[time][:, None]
            + log_transition[time]
            + emission[time + 1][None, :]
            + beta[time + 1][None, :]
            - likelihood
        )
        xi[time] = np.exp(log_xi)
        xi[time] /= max(float(xi[time].sum()), 1e-300)
    return likelihood, gamma, xi


def hmm_filter_conditioned(sequence: np.ndarray, features: np.ndarray, model: dict) -> np.ndarray:
    """Causal beliefs P(z_t | x_0:t, u_0:t-1); future edits cannot affect prefixes."""
    emission = log_gaussian_diag(sequence, model["means"], model["var"])
    transition = transition_matrices(features, model["transition_coef"])
    beliefs = np.empty_like(emission)
    log_belief = np.log(model["initial"] + 1e-300) + emission[0]
    log_belief -= logsumexp(log_belief, axis=0)
    beliefs[0] = np.exp(log_belief)
    for time in range(1, len(sequence)):
        prediction = logsumexp(
            log_belief[:, None] + np.log(transition[time - 1] + 1e-300), axis=0
        )
        log_belief = prediction + emission[time]
        log_belief -= logsumexp(log_belief, axis=0)
        beliefs[time] = np.exp(log_belief)
    return beliefs


def _fit_transition_coefficients(
    features: list[np.ndarray],
    xis: list[np.ndarray],
    initial: np.ndarray,
    *,
    l2: float,
    iterations: int = 250,
) -> np.ndarray:
    """Weighted multinomial M-step with episode-equal transition mass."""
    coef = np.asarray(initial, dtype=np.float64).copy()
    first = np.zeros_like(coef)
    second = np.zeros_like(coef)
    for step in range(1, iterations + 1):
        gradient = -l2 * coef
        for feature, xi in zip(features, xis):
            probability = transition_matrices(feature, coef)
            mass = xi.sum(axis=2)
            residual = xi - mass[:, :, None] * probability
            gradient += np.einsum("tij,tp->ijp", residual, feature) / max(len(feature), 1)
        gradient /= max(len(features), 1)
        # Adam is stable for very differently scaled soft transition counts.
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * np.square(gradient)
        first_hat = first / (1.0 - 0.9**step)
        second_hat = second / (1.0 - 0.999**step)
        coef += 0.05 * first_hat / (np.sqrt(second_hat) + 1e-8)
        # Remove the softmax's unidentifiable common destination offset.
        coef -= coef.mean(axis=1, keepdims=True)
    return coef


def fit_action_conditioned_hmm(
    sequences: list[np.ndarray],
    actions: list[np.ndarray],
    k: int,
    seed: int,
    *,
    mode: str = "action_progress",
    l2: float = 1e-2,
    iterations: int = 30,
    fixed_state_model: dict | None = None,
) -> dict:
    if len(sequences) != len(actions) or len(sequences) < 2:
        raise ValueError("matching latent/action episodes are required")
    action_mean, action_scale = fit_control_scaler(actions)
    features = [
        control_features(action, len(sequence), action_mean, action_scale, mode=mode)
        for sequence, action in zip(sequences, actions)
    ]
    fixed = (
        fit_diag_hmm(sequences, k, seed, iterations=min(iterations, 20))
        if fixed_state_model is None else fixed_state_model
    )
    if np.asarray(fixed["means"]).shape[0] != k or np.asarray(fixed["var"]).shape != np.asarray(fixed["means"]).shape:
        raise ValueError("fixed HMM state model does not match requested state count/dimension")
    width = features[0].shape[1]
    coef = np.zeros((k, k, width), dtype=np.float64)
    coef[:, :, 0] = np.log(np.maximum(fixed["transition"], 1e-8))
    coef -= coef.mean(axis=1, keepdims=True)
    model = {
        "initial": fixed["initial"].copy(),
        "means": fixed["means"].copy(),
        "var": fixed["var"].copy(),
        "transition_coef": coef,
        "action_mean": action_mean,
        "action_scale": action_scale,
        "feature_mode": mode,
    }
    previous = -np.inf
    for _ in range(iterations):
        initial_sum = np.zeros(k)
        gamma_sum = np.zeros(k)
        mean_sum = np.zeros_like(model["means"])
        second_sum = np.zeros_like(model["var"])
        xis = []
        total_likelihood = 0.0
        for sequence, feature in zip(sequences, features):
            likelihood, gamma, xi = hmm_forward_backward_conditioned(sequence, feature, model)
            initial_sum += gamma[0]
            weight = 1.0 / len(sequence)
            gamma_sum += weight * gamma.sum(0)
            mean_sum += weight * gamma.T @ sequence
            second_sum += weight * gamma.T @ np.square(sequence)
            xis.append(xi)
            total_likelihood += likelihood / len(sequence)
        if fixed_state_model is None:
            model["initial"] = (initial_sum + 1e-2) / (initial_sum.sum() + k * 1e-2)
            model["means"] = mean_sum / np.maximum(gamma_sum[:, None], 1e-8)
            model["var"] = np.maximum(
                second_sum / np.maximum(gamma_sum[:, None], 1e-8) - np.square(model["means"]),
                1e-3,
            )
        model["transition_coef"] = _fit_transition_coefficients(
            features, xis, model["transition_coef"], l2=l2
        )
        if total_likelihood - previous < 1e-5:
            break
        previous = total_likelihood
    model["state_model_frozen"] = fixed_state_model is not None
    return model


def model_features(sequence: np.ndarray, action: np.ndarray, model: dict) -> np.ndarray:
    return control_features(
        action,
        len(sequence),
        model["action_mean"],
        model["action_scale"],
        mode=str(model["feature_mode"]),
    )


def align_action_hmm(reference: dict, candidate: dict) -> dict:
    """Permute a fold-fitted action HMM into the frozen full-model state order."""
    from scipy.optimize import linear_sum_assignment

    reference_mean = np.asarray(reference["means"])
    candidate_mean = np.asarray(candidate["means"])
    cost = np.linalg.norm(reference_mean[:, None, :] - candidate_mean[None, :, :], axis=2)
    reference_index, candidate_index = linear_sum_assignment(cost)
    order = np.empty(len(reference_index), dtype=np.int64)
    order[reference_index] = candidate_index
    aligned = dict(candidate)
    for key in ("initial", "means", "var"):
        aligned[key] = np.asarray(candidate[key])[order]
    coef = np.asarray(candidate["transition_coef"])
    aligned["transition_coef"] = coef[order][:, order, :]
    aligned["state_alignment_mean_distance"] = float(cost[reference_index, candidate_index].mean())
    return aligned


def action_conditioned_dynamics_gate(
    sequences: list[np.ndarray],
    actions: list[np.ndarray] | None,
    *,
    n_folds: int,
    k: int,
    seed: int,
    l2: float = 1e-2,
    full_state_model: dict | None = None,
) -> dict:
    """Episode-held-out action/progress test with a within-time permutation null."""
    if actions is None:
        return {"decidable": False, "eligible": False, "reason": "aligned executed controls are absent"}
    if len(actions) != len(sequences) or min(len(value) for value in sequences) < 4:
        return {"decidable": False, "eligible": False, "reason": "latent/action episode alignment is invalid"}
    folds = episode_folds(len(sequences), n_folds, seed)
    all_ids = np.arange(len(sequences))
    scores = {name: np.full(len(sequences), np.nan) for name in ("fixed", "progress", "action_progress")}
    permuted = np.full(len(sequences), np.nan)
    fold_action_models = []
    for fold_index, test_ids in enumerate(folds):
        train_ids = all_ids[~np.isin(all_ids, test_ids)]
        train_sequences = [sequences[int(i)] for i in train_ids]
        train_actions = [actions[int(i)] for i in train_ids]
        fixed = fit_diag_hmm(train_sequences, k, seed + 101 * fold_index)
        progress = fit_action_conditioned_hmm(
            train_sequences, train_actions, k, seed + 101 * fold_index + 1, mode="progress",
            l2=l2, fixed_state_model=fixed,
        )
        action_model = fit_action_conditioned_hmm(
            train_sequences, train_actions, k, seed + 101 * fold_index + 2,
            mode="action_progress", l2=l2, fixed_state_model=fixed,
        )
        fold_action_models.append((np.asarray(test_ids, dtype=np.int64), action_model))
        for episode in test_ids:
            sequence = sequences[int(episode)]
            action = actions[int(episode)]
            scores["fixed"][episode] = hmm_forward_backward(sequence, fixed)[0] / len(sequence)
            for name, model in (("progress", progress), ("action_progress", action_model)):
                feature = model_features(sequence, action, model)
                scores[name][episode] = (
                    hmm_forward_backward_conditioned(sequence, feature, model)[0] / len(sequence)
                )
            # Circularly shift controls within the same episode.  This preserves
            # its action marginal and progress grid while breaking state/action binding.
            shift = 1 + (fold_index % max(len(action) - 1, 1))
            shuffled = np.roll(action, shift=shift, axis=0)
            permuted[episode] = hmm_forward_backward_conditioned(
                sequence, model_features(sequence, shuffled, action_model), action_model
            )[0] / len(sequence)
    action_delta = scores["action_progress"] - np.maximum(scores["fixed"], scores["progress"])
    binding_delta = scores["action_progress"] - permuted
    action_ci = bootstrap_mean_ci(action_delta, seed + 7001)
    binding_ci = bootstrap_mean_ci(binding_delta, seed + 7002)
    if full_state_model is None:
        full_state_model = fit_diag_hmm(sequences, k, seed + 9000)
    full = fit_action_conditioned_hmm(
        sequences, actions, k, seed + 9001, mode="action_progress", l2=l2,
        fixed_state_model=full_state_model,
    )
    cross_fitted_beliefs: list[np.ndarray | None] = [None] * len(sequences)
    alignment_distances = []
    for test_ids, fold_model in fold_action_models:
        aligned = align_action_hmm(full, fold_model)
        alignment_distances.append(aligned["state_alignment_mean_distance"])
        for episode in test_ids:
            cross_fitted_beliefs[int(episode)] = hmm_filter_conditioned(
                sequences[int(episode)],
                model_features(sequences[int(episode)], actions[int(episode)], aligned),
                aligned,
            )
    if any(value is None for value in cross_fitted_beliefs):
        raise RuntimeError("action-HMM cross-fitting did not cover every episode")
    full["cross_fitted_beliefs"] = np.stack(cross_fitted_beliefs)
    filters = [hmm_filter_conditioned(s, model_features(s, a, full), full) for s, a in zip(sequences, actions)]
    # An empirical two-step check compares causal endpoint beliefs with the
    # transition-only Chapman--Kolmogorov composition.  It is diagnostic, not a
    # theorem: emissions between the endpoints can legitimately update belief.
    ck_errors = []
    for sequence, action, belief in zip(sequences, actions, filters):
        matrices = transition_matrices(model_features(sequence, action, full), full["transition_coef"])
        for time in range(len(sequence) - 2):
            composed = belief[time] @ matrices[time] @ matrices[time + 1]
            ck_errors.append(float(np.abs(composed - belief[time + 2]).sum() / 2.0))
    ck_median = float(np.median(ck_errors)) if ck_errors else float("inf")
    eligible = bool(action_ci[0] > 0.0 and binding_ci[0] > 0.0 and ck_median <= 0.25)
    return {
        "decidable": True,
        "heldout_log_likelihood_nats_per_step": {
            name: float(value.mean()) for name, value in scores.items()
        },
        "action_minus_strongest_fixed_or_progress": {
            "mean": float(action_delta.mean()), "bootstrap_95_ci": list(action_ci)
        },
        "correct_minus_within_episode_shifted_action": {
            "mean": float(binding_delta.mean()), "bootstrap_95_ci": list(binding_ci)
        },
        "chapman_kolmogorov": {
            "two_step_endpoint_total_variation_median": ck_median,
            "tolerance": 0.25,
            "passes": bool(ck_median <= 0.25),
            "caution": "endpoint beliefs include intervening emissions; this is an empirical adequacy check",
        },
        "causal_filter_no_future": True,
        "episode_equal_fit": True,
        "shared_emissions_and_initial_across_transition_comparators": True,
        "full_state_model_frozen": True,
        "cross_fitted_router_available": True,
        "fold_state_alignment_mean_distance": alignment_distances,
        "eligible": eligible,
        "full_model": full,
    }
