#!/usr/bin/env python3
"""Development-only predictive-factor and latent-regime gate for Panel P.

This script deliberately uses ordered *physical replans* (``__mean`` or
``__mean_all``), never CEM candidate-row order.  It performs whole-episode
cross-validation and keeps three questions separate:

1. Does a structured predictive factor basis improve blockwise next-state
   prediction over PCA and random orthogonal partitions?
2. Does sequence order improve held-out activation likelihood beyond a static
   Gaussian mixture or frozen time bins?
3. Does older history improve next-state prediction after conditioning on the
   current state and exact executed control chunk (the Markov-sufficiency diagnostic)?

The current legacy capture has no selected actions, so question 3 and any HMM
deployment verdict remain provisional for that capture.  A result here can
select or reject development components; it is not behavioral evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def logsumexp(x: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    out = m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True) + 1e-300)
    return out if keepdims else np.squeeze(out, axis=axis)


def episode_folds(n: int, n_folds: int, seed: int) -> list[np.ndarray]:
    order = np.random.default_rng(seed).permutation(n)
    return [order[i::n_folds] for i in range(n_folds)]


def summarize_action_chunks(actions: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """Summarize exact variable-length control chunks without using future state.

    The uncompressed controls remain in the behavior trace.  For each interval
    this returns per-control sum, mean, and endpoint change together with total
    control length, total variation, and command count.
    """
    actions = np.asarray(actions, dtype=np.float64)
    boundaries = np.asarray(boundaries, dtype=np.int64)
    if actions.ndim != 2 or boundaries.ndim != 1 or len(boundaries) < 2:
        raise ValueError("executed actions must be [N,A] with 1-D plan boundaries")
    if boundaries[0] != 0 or boundaries[-1] != len(actions) or np.any(np.diff(boundaries) <= 0):
        raise ValueError("plan boundaries must strictly partition the executed actions")
    features = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        chunk = actions[int(start) : int(stop)]
        endpoint_change = chunk[-1] - chunk[0]
        path_length = np.linalg.norm(chunk, axis=1).sum()
        total_variation = np.linalg.norm(np.diff(chunk, axis=0), axis=1).sum() if len(chunk) > 1 else 0.0
        features.append(
            np.r_[chunk.sum(0), chunk.mean(0), endpoint_change, path_length, total_variation, len(chunk)]
        )
    return np.asarray(features, dtype=np.float64)


def load_site_sequences(capture: Path, site: str, pool: str) -> tuple[list[np.ndarray], list[np.ndarray] | None, dict]:
    index = json.loads((capture / "activations" / "index.json").read_text())
    episode_log: dict[int, dict] = {}
    episode_path = capture / "episodes.jsonl"
    if episode_path.exists():
        for line in episode_path.read_text().splitlines():
            if line.strip():
                episode = json.loads(line)
                episode_log[int(episode["ep"])] = episode
    sequences: list[np.ndarray] = []
    action_sequences: list[np.ndarray] = []
    action_complete = True
    action_sources: list[str] = []
    for row in index["episodes"]:
        z = np.load(capture / "activations" / row["file"])
        key = f"{site}__{pool}"
        if key not in z.files:
            raise KeyError(f"{row['file']} has no {key}")
        x = np.asarray(z[key], dtype=np.float64)
        if x.ndim != 2 or len(x) < 3:
            raise ValueError(f"{row['file']}:{key} must be [T,D] with T>=3, got {x.shape}")
        sequences.append(x)
        episode = episode_log.get(int(row["ep"]), {})
        trace_rel = episode.get("behavior_trace")
        if trace_rel:
            trace = np.load(capture / trace_rel)
            if "executed_actions" in trace.files and "plan_boundaries" in trace.files:
                actions = summarize_action_chunks(trace["executed_actions"], trace["plan_boundaries"])
                if len(actions) == len(x):
                    action_sequences.append(actions)
                    action_sources.append("exact_executed_control_chunk_summary")
                    continue
        if "chosen_actions" not in z.files:
            action_complete = False
            continue
        actions = np.asarray(z["chosen_actions"], dtype=np.float64)
        if actions.ndim < 2 or len(actions) != len(x):
            action_complete = False
            continue
        action_sequences.append(np.nan_to_num(actions.reshape(len(actions), -1)))
        action_sources.append("selected_planner_action_fallback")
    index = dict(index)
    index["action_source"] = (
        action_sources[0] if action_sources and len(set(action_sources)) == 1 else "mixed_or_unavailable"
    )
    return sequences, action_sequences if action_complete and len(action_sequences) == len(sequences) else None, index


def available_sites(capture: Path, pool: str) -> list[str]:
    index = json.loads((capture / "activations" / "index.json").read_text())
    if not index.get("episodes"):
        return []
    z = np.load(capture / "activations" / index["episodes"][0]["file"])
    return sorted(key[: -len(f"__{pool}")] for key in z.files if key.endswith(f"__{pool}"))


def fit_pca(rows: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    mean = rows.mean(axis=0)
    centered = rows - mean
    _u, s, vt = np.linalg.svd(centered, full_matrices=False)
    rank = min(rank, len(s), max(1, len(rows) - 1))
    basis = vt[:rank].T
    scale = np.maximum(s[:rank] / np.sqrt(max(len(rows) - 1, 1)), 1e-8)
    eig = np.square(s) / max(len(rows) - 1, 1)
    retained = float(eig[:rank].sum() / max(eig.sum(), 1e-30))
    positive = eig[eig > max(eig.max(), 1e-30) * 1e-8]
    effective_rank = float(np.square(positive.sum()) / max(np.square(positive).sum(), 1e-30))
    diag = {
        "rank": int(rank),
        "retained_variance": retained,
        "ambient_effective_rank": effective_rank,
        "condition_number": float(scale.max() / scale.min()),
    }
    return mean, basis, scale, diag


def transform_sequences(sequences: list[np.ndarray], mean: np.ndarray, basis: np.ndarray, scale: np.ndarray) -> list[np.ndarray]:
    return [((sequence - mean) @ basis) / scale for sequence in sequences]


def transition_rows(sequences: list[np.ndarray], ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    current = np.concatenate([sequences[int(i)][:-1] for i in ids])
    future = np.concatenate([sequences[int(i)][1:] for i in ids])
    return current, future


def ridge_fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, ridge: float) -> np.ndarray:
    x_mean, y_mean = train_x.mean(0), train_y.mean(0)
    xc, yc = train_x - x_mean, train_y - y_mean
    gram = xc.T @ xc + ridge * np.eye(xc.shape[1])
    weights = np.linalg.solve(gram, xc.T @ yc)
    return y_mean + (test_x - x_mean) @ weights


def block_prediction(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    q: np.ndarray,
    block_size: int,
    ridge: float,
) -> np.ndarray:
    xq, yq, tq = train_x @ q, train_y @ q, test_x @ q
    pred = np.zeros_like(tq)
    for start in range(0, q.shape[1], block_size):
        stop = min(start + block_size, q.shape[1])
        pred[:, start:stop] = ridge_fit_predict(
            xq[:, start:stop], yq[:, start:stop], tq[:, start:stop], ridge
        )
    return pred @ q.T


def predictive_basis(current: np.ndarray, future: np.ndarray) -> np.ndarray:
    cross = (current - current.mean(0)).T @ (future - future.mean(0))
    left, _s, _right = np.linalg.svd(cross, full_matrices=False)
    return left


def subspace_angle_deg(a: np.ndarray, b: np.ndarray, width: int) -> float:
    singular = np.linalg.svd(a[:, :width].T @ b[:, :width], compute_uv=False)
    angles = np.arccos(np.clip(singular, -1.0, 1.0))
    return float(np.sqrt(np.mean(np.square(angles))) * 180.0 / np.pi)


def factor_cv(
    sequences: list[np.ndarray],
    *,
    rank: int,
    block_size: int,
    n_folds: int,
    n_random: int,
    ridge: float,
    seed: int,
) -> dict:
    folds = episode_folds(len(sequences), n_folds, seed)
    errors = {"global": [], "pca_blocks": [], "predictive_factors": []}
    random_errors: list[list[float]] = [[] for _ in range(n_random)]
    all_ids = np.arange(len(sequences))
    fold_diagnostics = []

    for fold_idx, test_ids in enumerate(folds):
        train_ids = all_ids[~np.isin(all_ids, test_ids)]
        state_rows = np.concatenate([sequences[int(i)] for i in train_ids])
        mean, basis, scale, diag = fit_pca(state_rows, rank)
        z = transform_sequences(sequences, mean, basis, scale)
        train_x, train_y = transition_rows(z, train_ids)
        test_x, test_y = transition_rows(z, test_ids)
        denom = float(np.mean(np.square(test_y - train_y.mean(0))) + 1e-12)

        pred = ridge_fit_predict(train_x, train_y, test_x, ridge)
        errors["global"].append(float(np.mean(np.square(test_y - pred)) / denom))
        eye = np.eye(train_x.shape[1])
        pred = block_prediction(train_x, train_y, test_x, eye, block_size, ridge)
        errors["pca_blocks"].append(float(np.mean(np.square(test_y - pred)) / denom))
        qp = predictive_basis(train_x, train_y)
        pred = block_prediction(train_x, train_y, test_x, qp, block_size, ridge)
        errors["predictive_factors"].append(float(np.mean(np.square(test_y - pred)) / denom))

        for random_idx in range(n_random):
            qr, _ = np.linalg.qr(
                np.random.default_rng(seed + 1009 * fold_idx + 7919 * random_idx).standard_normal(qp.shape)
            )
            pred = block_prediction(train_x, train_y, test_x, qr, block_size, ridge)
            random_errors[random_idx].append(float(np.mean(np.square(test_y - pred)) / denom))
        fold_diagnostics.append(diag)

    means = {name: float(np.mean(value)) for name, value in errors.items()}
    random_means = np.asarray([np.mean(value) for value in random_errors])
    means["random_orthogonal_median"] = float(np.median(random_means))
    predictive = means["predictive_factors"]
    p_random = float((1 + np.sum(random_means <= predictive)) / (1 + len(random_means)))
    gain_pca = float((means["pca_blocks"] - predictive) / max(means["pca_blocks"], 1e-12))
    gain_global = float((means["global"] - predictive) / max(means["global"], 1e-12))

    # Stability is computed in one fixed, full-data whitened coordinate system.
    full_rows = np.concatenate(sequences)
    mean, basis, scale, full_diag = fit_pca(full_rows, rank)
    z = transform_sequences(sequences, mean, basis, scale)
    current, future = transition_rows(z, np.arange(len(z)))
    q_full = predictive_basis(current, future)
    boot_angles = []
    rng = np.random.default_rng(seed + 991)
    for _ in range(50):
        ids = rng.integers(0, len(z), size=len(z))
        bx, by = transition_rows(z, ids)
        boot_angles.append(subspace_angle_deg(q_full, predictive_basis(bx, by), min(block_size, q_full.shape[1])))
    stability = float(np.median(boot_angles))
    eligible = bool(gain_pca >= 0.02 and gain_global >= 0.02 and p_random <= 0.10 and stability <= 35.0)
    fallback = "global" if means["global"] <= means["pca_blocks"] else "pca_blocks"
    return {
        "cv_normalized_mse": means,
        "fold_values": errors,
        "random_null": {
            "n": n_random,
            "median": float(np.median(random_means)),
            "p_predictive_no_better": p_random,
        },
        "relative_gain_predictive_vs_pca_blocks": gain_pca,
        "relative_gain_predictive_vs_global": gain_global,
        "bootstrap_top_block_angle_deg_median": stability,
        "predictive_factor_eligible": eligible,
        "selected_representation": "predictive_factors" if eligible else fallback,
        "pca_diagnostics": full_diag,
        "fold_pca_diagnostics": fold_diagnostics,
        "fit": {"mean": mean, "basis": basis, "scale": scale, "factor_rotation": q_full},
    }


def log_gaussian_diag(x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    var = np.maximum(var, 1e-5)
    squared_mahalanobis = (np.square(x[:, None, :] - mean[None, :, :]) / var[None, :, :]).sum(-1)
    return -0.5 * (
        x.shape[-1] * np.log(2 * np.pi) + np.log(var).sum(-1) + squared_mahalanobis
    )


def fit_diag_gmm(rows: np.ndarray, k: int, seed: int, iterations: int = 80) -> dict:
    rng = np.random.default_rng(seed)
    means = rows[rng.choice(len(rows), size=k, replace=False)].copy()
    var = np.tile(np.var(rows, axis=0, ddof=1) + 1e-3, (k, 1))
    weights = np.full(k, 1.0 / k)
    previous = -np.inf
    for _ in range(iterations):
        log_joint = log_gaussian_diag(rows, means, var) + np.log(weights + 1e-30)
        norm = logsumexp(log_joint, axis=1, keepdims=True)
        responsibility = np.exp(log_joint - norm)
        nk = responsibility.sum(0) + 1e-6
        weights = nk / nk.sum()
        means = responsibility.T @ rows / nk[:, None]
        var = np.stack(
            [
                (responsibility[:, state, None] * np.square(rows - means[state])).sum(0) / nk[state]
                for state in range(k)
            ]
        ) + 1e-3
        likelihood = float(norm.sum())
        if likelihood - previous < 1e-6:
            break
        previous = likelihood
    return {"weights": weights, "means": means, "var": var}


def gmm_log_prob(rows: np.ndarray, model: dict) -> np.ndarray:
    joint = log_gaussian_diag(rows, model["means"], model["var"]) + np.log(model["weights"] + 1e-30)
    return logsumexp(joint, axis=1)


def hmm_forward_backward(sequence: np.ndarray, model: dict) -> tuple[float, np.ndarray, np.ndarray]:
    emit = log_gaussian_diag(sequence, model["means"], model["var"])
    log_a = np.log(model["transition"] + 1e-30)
    alpha = np.empty_like(emit)
    alpha[0] = np.log(model["initial"] + 1e-30) + emit[0]
    for t in range(1, len(sequence)):
        alpha[t] = emit[t] + logsumexp(alpha[t - 1][:, None] + log_a, axis=0)
    ll = float(logsumexp(alpha[-1], axis=0))
    beta = np.zeros_like(emit)
    for t in range(len(sequence) - 2, -1, -1):
        beta[t] = logsumexp(log_a + emit[t + 1][None, :] + beta[t + 1][None, :], axis=1)
    gamma = np.exp(alpha + beta - ll)
    xi = np.empty((max(len(sequence) - 1, 0), len(model["initial"]), len(model["initial"])))
    for t in range(len(sequence) - 1):
        xi[t] = np.exp(alpha[t][:, None] + log_a + emit[t + 1][None, :] + beta[t + 1][None, :] - ll)
    return ll, gamma, xi


def hmm_filter(sequence: np.ndarray, model: dict) -> np.ndarray:
    """Causal filtered beliefs ``P(z_t | x_1:t)`` with no future emissions."""
    emission = log_gaussian_diag(sequence, model["means"], model["var"])
    beliefs = np.empty_like(emission)
    log_belief = np.log(model["initial"] + 1e-30) + emission[0]
    log_belief -= logsumexp(log_belief, axis=0)
    beliefs[0] = np.exp(log_belief)
    log_transition = np.log(model["transition"] + 1e-30)
    for time in range(1, len(sequence)):
        prediction = logsumexp(log_belief[:, None] + log_transition, axis=0)
        log_belief = prediction + emission[time]
        log_belief -= logsumexp(log_belief, axis=0)
        beliefs[time] = np.exp(log_belief)
    return beliefs


def fit_diag_hmm(
    sequences: list[np.ndarray], k: int, seed: int, iterations: int = 60,
    *, episode_equal: bool = True,
) -> dict:
    rows = np.concatenate(sequences)
    gmm = fit_diag_gmm(rows, k, seed)
    responsibility = np.exp(
        log_gaussian_diag(rows, gmm["means"], gmm["var"])
        + np.log(gmm["weights"] + 1e-30)
        - gmm_log_prob(rows, gmm)[:, None]
    )
    cursor = 0
    counts = np.ones((k, k))
    initial = np.ones(k)
    for sequence in sequences:
        hard = responsibility[cursor : cursor + len(sequence)].argmax(1)
        initial[hard[0]] += 1
        for before, after in zip(hard[:-1], hard[1:]):
            counts[before, after] += 1
        cursor += len(sequence)
    model = {
        "initial": initial / initial.sum(),
        "transition": counts / counts.sum(1, keepdims=True),
        "means": gmm["means"].copy(),
        "var": gmm["var"].copy(),
    }
    previous = -np.inf
    for _ in range(iterations):
        initial_sum = np.zeros(k)
        transition_sum = np.ones((k, k)) * 1e-2
        gamma_sum = np.zeros(k)
        mean_sum = np.zeros_like(model["means"])
        second_sum = np.zeros_like(model["var"])
        total_ll = 0.0
        for sequence in sequences:
            ll, gamma, xi = hmm_forward_backward(sequence, model)
            # A rollout is the independent sampling unit. Equal weighting is
            # immaterial in the common-length window, but prevents a long episode
            # from silently determining the model if a future environment varies.
            emission_scale = 1.0 / len(sequence) if episode_equal else 1.0
            transition_scale = 1.0 / max(len(sequence) - 1, 1) if episode_equal else 1.0
            total_ll += ll / len(sequence) if episode_equal else ll
            initial_sum += gamma[0]
            transition_sum += transition_scale * xi.sum(0)
            gamma_sum += emission_scale * gamma.sum(0)
            mean_sum += emission_scale * (gamma.T @ sequence)
            second_sum += emission_scale * (gamma.T @ np.square(sequence))
        model["initial"] = (initial_sum + 1e-2) / (initial_sum.sum() + 1e-2 * k)
        model["transition"] = transition_sum / transition_sum.sum(1, keepdims=True)
        model["means"] = mean_sum / np.maximum(gamma_sum[:, None], 1e-8)
        model["var"] = np.maximum(
            second_sum / np.maximum(gamma_sum[:, None], 1e-8) - np.square(model["means"]), 1e-3
        )
        if total_ll - previous < 1e-5:
            break
        previous = total_ll
    return model


def fit_time_bins(sequences: list[np.ndarray], k: int) -> dict:
    buckets = [[] for _ in range(k)]
    for sequence in sequences:
        labels = np.minimum((np.arange(len(sequence)) * k) // len(sequence), k - 1)
        for state in range(k):
            buckets[state].append(sequence[labels == state])
    means = np.stack([np.concatenate(bucket).mean(0) for bucket in buckets])
    var = np.stack([np.concatenate(bucket).var(0) + 1e-3 for bucket in buckets])
    return {"means": means, "var": var}


def time_bin_loglik(sequence: np.ndarray, model: dict) -> float:
    k = len(model["means"])
    labels = np.minimum((np.arange(len(sequence)) * k) // len(sequence), k - 1)
    emission = log_gaussian_diag(sequence, model["means"], model["var"])
    return float(emission[np.arange(len(sequence)), labels].sum())


def bootstrap_mean_ci(values: np.ndarray, seed: int, draws: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.asarray([rng.choice(values, len(values), replace=True).mean() for _ in range(draws)])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def static_posterior(sequence: np.ndarray, model: dict) -> np.ndarray:
    joint = log_gaussian_diag(sequence, model["means"], model["var"]) + np.log(model["weights"] + 1e-30)
    return np.exp(joint - logsumexp(joint, axis=1, keepdims=True))


def align_regime_model(reference: dict, candidate: dict) -> dict:
    """Align candidate component labels to the reference by emission geometry."""
    k = len(reference["means"])
    if len(candidate["means"]) != k:
        raise ValueError("reference and candidate must have the same regime count")
    scale = np.maximum(np.asarray(reference["var"], dtype=np.float64), 1e-6)
    best = min(
        itertools.permutations(range(k)),
        key=lambda permutation: float(
            np.sum(np.square(np.asarray(candidate["means"])[list(permutation)] - reference["means"]) / scale)
        ),
    )
    permutation = np.asarray(best, dtype=np.int64)
    aligned = {
        "means": np.asarray(candidate["means"])[permutation],
        "var": np.asarray(candidate["var"])[permutation],
    }
    for key in ("weights", "initial"):
        if key in candidate:
            aligned[key] = np.asarray(candidate[key])[permutation]
    if "transition" in candidate:
        aligned["transition"] = np.asarray(candidate["transition"])[np.ix_(permutation, permutation)]
    return aligned


def regime_bootstrap_stability(
    sequences: list[np.ndarray],
    reference: dict,
    *,
    kind: str,
    k: int,
    seed: int,
    draws: int = 50,
) -> dict:
    """Episode-bootstrap emission and causal-belief stability after label alignment."""
    if kind not in {"static_gmm", "hmm"}:
        raise ValueError(kind)
    rows = np.concatenate(sequences)
    if kind == "hmm":
        reference_weights = np.concatenate([
            hmm_forward_backward(sequence, reference)[1] for sequence in sequences
        ]).mean(0)
        reference_static = {**reference, "weights": reference_weights}
        reference_filtered = np.concatenate([hmm_filter(sequence, reference) for sequence in sequences])
    else:
        reference_static = reference
        reference_filtered = None
    reference_q = static_posterior(rows, reference_static)
    rng = np.random.default_rng(seed)
    agreement, soft_l1, filtered_agreement, transition_l1, emission_rmse = [], [], [], [], []
    failures = 0
    for draw in range(draws):
        ids = rng.integers(0, len(sequences), size=len(sequences))
        sampled = [sequences[int(i)] for i in ids]
        try:
            if kind == "hmm":
                candidate = fit_diag_hmm(sampled, k, seed + 1009 * (draw + 1))
                candidate = align_regime_model(reference, candidate)
                candidate_weights = np.concatenate([
                    hmm_forward_backward(sequence, candidate)[1] for sequence in sampled
                ]).mean(0)
                candidate_static = {**candidate, "weights": candidate_weights}
                candidate_filtered = np.concatenate([hmm_filter(sequence, candidate) for sequence in sequences])
                filtered_agreement.append(float(
                    np.mean(reference_filtered.argmax(1) == candidate_filtered.argmax(1))
                ))
                transition_l1.append(float(np.mean(np.abs(reference["transition"] - candidate["transition"]))))
            else:
                candidate = fit_diag_gmm(np.concatenate(sampled), k, seed + 1009 * (draw + 1))
                candidate = align_regime_model(reference, candidate)
                candidate_static = candidate
            candidate_q = static_posterior(rows, candidate_static)
            agreement.append(float(np.mean(reference_q.argmax(1) == candidate_q.argmax(1))))
            soft_l1.append(float(np.mean(np.abs(reference_q - candidate_q))))
            emission_rmse.append(float(np.sqrt(np.mean(
                np.square(candidate["means"] - reference["means"]) / np.maximum(reference["var"], 1e-6)
            ))))
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            failures += 1

    def quantile(values: list[float], q: float) -> float | None:
        return float(np.quantile(values, q)) if values else None

    valid = len(agreement)
    stable = bool(
        valid >= max(20, int(0.8 * draws))
        and quantile(agreement, 0.05) is not None
        and quantile(agreement, 0.05) >= 0.60
        and quantile(soft_l1, 0.95) <= 0.25
        and (kind != "hmm" or quantile(filtered_agreement, 0.05) >= 0.60)
    )
    return {
        "kind": kind,
        "draws_requested": int(draws),
        "draws_valid": int(valid),
        "draws_failed": int(failures),
        "hard_assignment_agreement_median": quantile(agreement, 0.50),
        "hard_assignment_agreement_p05": quantile(agreement, 0.05),
        "soft_responsibility_l1_median": quantile(soft_l1, 0.50),
        "soft_responsibility_l1_p95": quantile(soft_l1, 0.95),
        "emission_mean_standardized_rmse_median": quantile(emission_rmse, 0.50),
        "filtered_assignment_agreement_p05": quantile(filtered_agreement, 0.05),
        "transition_mean_absolute_error_p95": quantile(transition_l1, 0.95),
        "stable": stable,
    }


def regime_cv(
    sequences: list[np.ndarray], *, n_folds: int, k: int, seed: int, stability_draws: int = 50
) -> dict:
    folds = episode_folds(len(sequences), n_folds, seed)
    all_ids = np.arange(len(sequences))
    progress_bins = min(10, min(len(sequence) for sequence in sequences))
    scores = {
        name: np.full(len(sequences), np.nan)
        for name in ("one_gaussian", "static_gmm", "progress_bins", "hmm")
    }
    for fold_idx, test_ids in enumerate(folds):
        train_ids = all_ids[~np.isin(all_ids, test_ids)]
        train = [sequences[int(i)] for i in train_ids]
        rows = np.concatenate(train)
        one = {"weights": np.ones(1), "means": rows.mean(0, keepdims=True), "var": rows.var(0, keepdims=True) + 1e-3}
        gmm = fit_diag_gmm(rows, k, seed + fold_idx)
        time_model = fit_time_bins(train, progress_bins)
        hmm = fit_diag_hmm(train, k, seed + fold_idx)
        for episode in test_ids:
            sequence = sequences[int(episode)]
            length = len(sequence)
            scores["one_gaussian"][episode] = float(gmm_log_prob(sequence, one).sum() / length)
            scores["static_gmm"][episode] = float(gmm_log_prob(sequence, gmm).sum() / length)
            scores["progress_bins"][episode] = time_bin_loglik(sequence, time_model) / length
            scores["hmm"][episode] = hmm_forward_backward(sequence, hmm)[0] / length
    strongest_static = np.maximum.reduce([
        scores["one_gaussian"], scores["static_gmm"], scores["progress_bins"]
    ])
    delta = scores["hmm"] - strongest_static
    ci = bootstrap_mean_ci(delta, seed + 404)
    static_delta = scores["static_gmm"] - scores["one_gaussian"]
    static_ci = bootstrap_mean_ci(static_delta, seed + 403)
    full_hmm = fit_diag_hmm(sequences, k, seed + 808)
    full_gmm = fit_diag_gmm(np.concatenate(sequences), k, seed + 809)
    full_rows = np.concatenate(sequences)
    full_one = {
        "weights": np.ones(1),
        "means": full_rows.mean(0, keepdims=True),
        "var": full_rows.var(0, keepdims=True) + 1e-3,
    }
    posteriors = [hmm_forward_backward(sequence, full_hmm)[1] for sequence in sequences]
    occupancy = np.concatenate(posteriors).mean(0)
    entropy = np.concatenate([-p * np.log(p + 1e-30) for p in posteriors]).sum(1).mean()
    static_stability = regime_bootstrap_stability(
        sequences, full_gmm, kind="static_gmm", k=k, seed=seed + 1201, draws=stability_draws
    )
    hmm_stability = regime_bootstrap_stability(
        sequences, full_hmm, kind="hmm", k=k, seed=seed + 1202, draws=stability_draws
    )
    static_eligible = bool(static_delta.mean() >= 0.01 and static_ci[0] > 0.0 and static_stability["stable"])
    hmm_eligible = bool(delta.mean() >= 0.01 and ci[0] > 0.0 and hmm_stability["stable"])
    selected_source = "hmm_emissions" if hmm_eligible else ("static_gmm" if static_eligible else "one_gaussian")
    return {
        "heldout_log_likelihood_nats_per_step": {name: float(value.mean()) for name, value in scores.items()},
        "per_episode": {name: value.tolist() for name, value in scores.items()},
        "progress_control_n_bins": int(progress_bins),
        "static_gmm_minus_one_gaussian": {
            "mean": float(static_delta.mean()), "bootstrap_95_ci": list(static_ci)
        },
        "hmm_minus_strongest_static": {"mean": float(delta.mean()), "bootstrap_95_ci": list(ci)},
        # Compatibility field retained for old report readers. The eligibility
        # decision above is deliberately stricter than this pairwise contrast.
        "hmm_minus_static_gmm": {
            "mean": float((scores["hmm"] - scores["static_gmm"]).mean()),
            "bootstrap_95_ci": list(bootstrap_mean_ci(scores["hmm"] - scores["static_gmm"], seed + 405)),
        },
        "static_regime_eligible": static_eligible,
        "hmm_eligible_by_sequence_evidence": hmm_eligible,
        "selected_regime_source": selected_source,
        "regime_locality_eligible": selected_source != "one_gaussian",
        "regime_stability": {"static_gmm": static_stability, "hmm": hmm_stability},
        "full_one_gaussian": {key: value.tolist() for key, value in full_one.items()},
        "full_hmm": {
            "initial": full_hmm["initial"].tolist(),
            "transition": full_hmm["transition"].tolist(),
            "means": full_hmm["means"].tolist(),
            "var": full_hmm["var"].tolist(),
            "occupancy": occupancy.tolist(),
            "posterior_entropy_descriptive_only": float(entropy),
            "min_emission_variance": float(full_hmm["var"].min()),
        },
        "full_static_gmm": {key: value.tolist() for key, value in full_gmm.items()},
    }


def rbf_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, ridge: float) -> np.ndarray:
    scale = float(np.median(np.sum(np.square(train_x[:, None] - train_x[None, :]), axis=2)))
    scale = max(scale, 1e-6)
    kernel = np.exp(-np.sum(np.square(train_x[:, None] - train_x[None, :]), axis=2) / scale)
    alpha = np.linalg.solve(kernel + ridge * np.eye(len(kernel)), train_y)
    test_kernel = np.exp(-np.sum(np.square(test_x[:, None] - train_x[None, :]), axis=2) / scale)
    return test_kernel @ alpha


def markov_history_gate(
    sequences: list[np.ndarray], actions: list[np.ndarray] | None, *, n_folds: int, ridge: float, seed: int,
    action_source: str = "selected_planner_action_fallback",
) -> dict:
    if actions is None:
        return {
            "decidable": False,
            "reason": "capture has no aligned action chunks; history gain without action would be confounded",
        }
    rows = []
    for episode, (sequence, action) in enumerate(zip(sequences, actions)):
        for t in range(1, min(len(sequence), len(action)) - 1):
            # Each captured row is the endpoint of that plan's selected-action
            # unroll.  The chunk at t+1 therefore leads from row t to row t+1.
            rows.append((episode, sequence[t - 1], sequence[t], action[t + 1], sequence[t + 1]))
    episode_id = np.asarray([row[0] for row in rows])
    previous = np.stack([row[1] for row in rows])
    current = np.stack([row[2] for row in rows])
    action = np.stack([row[3] for row in rows])
    target = np.stack([row[4] for row in rows])
    folds = episode_folds(len(sequences), n_folds, seed)
    per_episode = {name: np.full(len(sequences), np.nan) for name in ("ridge_current", "ridge_history", "rbf_current", "rbf_history")}
    for test_episodes in folds:
        test = np.isin(episode_id, test_episodes)
        train = ~test
        action_mean = action[train].mean(0)
        action_scale = np.maximum(action[train].std(0), 1e-8)
        scaled_action = (action - action_mean) / action_scale
        xc = np.concatenate([current, scaled_action], axis=1)
        xh = np.concatenate([current, scaled_action, previous], axis=1)
        predictions = {
            "ridge_current": ridge_fit_predict(xc[train], target[train], xc[test], ridge),
            "ridge_history": ridge_fit_predict(xh[train], target[train], xh[test], ridge),
            "rbf_current": rbf_predict(xc[train], target[train], xc[test], ridge),
            "rbf_history": rbf_predict(xh[train], target[train], xh[test], ridge),
        }
        for name, prediction in predictions.items():
            row_ids = episode_id[test]
            row_targets = target[test]
            for episode in test_episodes:
                mask = row_ids == episode
                per_episode[name][episode] = float(np.mean(np.square(row_targets[mask] - prediction[mask])))
    ridge_gain = per_episode["ridge_current"] - per_episode["ridge_history"]
    rbf_gain = per_episode["rbf_current"] - per_episode["rbf_history"]
    ridge_ci = bootstrap_mean_ci(ridge_gain, seed + 17)
    rbf_ci = bootstrap_mean_ci(rbf_gain, seed + 19)
    return {
        "decidable": True,
        "conditioned_on": ["current_factor_state", action_source],
        "action_features_standardized_on_training_fold_only": True,
        "exact_controls_archived_separately": action_source == "exact_executed_control_chunk_summary",
        "mse": {name: float(value.mean()) for name, value in per_episode.items()},
        "history_gain_ridge": {"mean": float(ridge_gain.mean()), "bootstrap_95_ci": list(ridge_ci)},
        "history_gain_rbf": {"mean": float(rbf_gain.mean()), "bootstrap_95_ci": list(rbf_ci)},
        "reject_current_state_markov_sufficiency": bool(ridge_ci[0] > 0 and rbf_ci[0] > 0),
        "caution": "failure to reject is not proof of Markov sufficiency",
    }


def json_ready(report: dict) -> dict:
    return {key: value for key, value in report.items() if key != "fit"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--pool", default="mean", choices=["mean", "mean_all"])
    parser.add_argument("--site", action="append", default=[])
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=2)
    parser.add_argument(
        "--hmm-dim", type=int, default=0,
        help="emission width; 0 uses the full frozen coordinate span (default)",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-partitions", type=int, default=20)
    parser.add_argument("--regime-bootstrap-draws", type=int, default=50)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=240904)
    args = parser.parse_args()

    sites = args.site or available_sites(args.capture, args.pool)
    if not sites:
        raise SystemExit("no requested physical-replan sites found")
    args.out.mkdir(parents=True, exist_ok=True)
    site_reports = []
    fits = {}
    indices = {}
    for site in sites:
        sequences, actions, index = load_site_sequences(args.capture, site, args.pool)
        window_raw = index.get("pre_outcome_window") or [0, min(len(x) for x in sequences)]
        window = (int(window_raw[0]), int(window_raw[1]))
        if window[1] - window[0] < 3:
            raise SystemExit(f"pre-outcome physical-replan window too short: {window}")
        sequences = [sequence[window[0] : window[1]] for sequence in sequences]
        if actions is not None:
            actions = [action[window[0] : window[1]] for action in actions]
        indices[site] = index
        gate = factor_cv(
            sequences,
            rank=args.rank,
            block_size=args.block_size,
            n_folds=min(args.folds, len(sequences)),
            n_random=args.random_partitions,
            ridge=args.ridge,
            seed=args.seed,
        )
        fits[site] = gate.pop("fit")
        site_reports.append({"site": site, "n_episodes": len(sequences), "sequence_lengths": [len(x) for x in sequences], **gate})

    # Outcome labels never select this site; lower held-out next-state error does.
    selected = min(site_reports, key=lambda row: row["cv_normalized_mse"][row["selected_representation"]])
    site = selected["site"]
    sequences, actions, index = load_site_sequences(args.capture, site, args.pool)
    window_raw = index.get("pre_outcome_window") or [0, min(len(x) for x in sequences)]
    window = (int(window_raw[0]), int(window_raw[1]))
    sequences = [sequence[window[0] : window[1]] for sequence in sequences]
    if actions is not None:
        actions = [action[window[0] : window[1]] for action in actions]
    fit = fits[site]
    factors = transform_sequences(sequences, fit["mean"], fit["basis"], fit["scale"])
    if selected["selected_representation"] == "predictive_factors":
        factors = [sequence @ fit["factor_rotation"] for sequence in factors]
    hmm_width = factors[0].shape[1] if args.hmm_dim <= 0 else min(args.hmm_dim, factors[0].shape[1])
    factors_hmm = [sequence[:, :hmm_width] for sequence in factors]

    regime = regime_cv(
        factors_hmm,
        n_folds=min(args.folds, len(factors_hmm)),
        k=2,
        seed=args.seed,
        stability_draws=args.regime_bootstrap_draws,
    )
    history = markov_history_gate(
        factors_hmm, actions, n_folds=min(args.folds, len(factors_hmm)), ridge=args.ridge, seed=args.seed,
        action_source=index.get("action_source", "mixed_or_unavailable"),
    )
    # Imported here to keep the low-level HMM primitives in this file usable by
    # the action-conditioned implementation without an import cycle.
    from public_panel_action_hmm import action_conditioned_dynamics_gate

    action_dynamics = action_conditioned_dynamics_gate(
        factors_hmm,
        actions,
        n_folds=min(args.folds, len(factors_hmm)),
        k=2,
        seed=args.seed + 1701,
        full_state_model={
            "initial": np.asarray(regime["full_hmm"]["initial"]),
            "transition": np.asarray(regime["full_hmm"]["transition"]),
            "means": np.asarray(regime["full_hmm"]["means"]),
            "var": np.asarray(regime["full_hmm"]["var"]),
        },
    )
    action_hmm = action_dynamics.pop("full_model", None)
    # Sequence evidence plus history gain can nominate a temporal belief.  The
    # action-conditioned and Chapman--Kolmogorov tests are now executable here;
    # physical-replan belief lifecycle and downstream treatment separation remain
    # fail-closed in the intervention driver.
    hmm_candidate = bool(
        regime["hmm_eligible_by_sequence_evidence"]
        and regime["regime_stability"]["hmm"]["stable"]
        and history.get("decidable")
        and history.get("reject_current_state_markov_sufficiency")
    )
    offline_hmm_gates_pass = bool(hmm_candidate and action_dynamics.get("eligible", False))
    # This licenses serialization and the tested causal replan filter, not a
    # behavioral arm.  Sonar fitting and the arm registry still require outcome
    # routing and treatment-separation evidence before launch.
    hmm_deploy = offline_hmm_gates_pass
    capture_hash = hashlib.sha256((args.capture / "activations" / "index.json").read_bytes()).hexdigest()
    report = {
        "status": "development_only",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "capture": str(args.capture.resolve()),
        "capture_index_sha256": capture_hash,
        "pool": args.pool,
        "axis": "ordered physical replans in the common pre-outcome window",
        "pre_outcome_window": list(window),
        "outcome_labels_used_for_factor_or_regime_selection": False,
        "factor_sites": site_reports,
        "selected_site": site,
        "selected_representation": selected["selected_representation"],
        "regime_comparison": regime,
        "markov_history_diagnostic": history,
        "action_conditioned_dynamics": action_dynamics,
        "filtered_hmm_candidate": hmm_candidate,
        "offline_hmm_gates_pass": offline_hmm_gates_pass,
        "remaining_hmm_deployment_gates": [
            "HMM routing must improve held-out outcome routing over the same static local operator bank",
            "HMM and static treatments must differ materially in latent and executed-action space",
        ],
        "hmm_deployment_eligible": hmm_deploy,
        "decision": (
            "use filtered HMM belief" if hmm_deploy else "use static responsibilities until all HMM deployment gates pass"
        ),
        "claims_not_licensed": [
            "the frozen JEPA is an HMM",
            "the inferred regime is the model's explicit belief",
            "representation likelihood improves task performance",
        ],
    }
    (args.out / "factor_regime_gate.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        args.out / "selected_coordinates.npz",
        site=np.asarray(site),
        mean=fit["mean"].astype(np.float32),
        basis=fit["basis"].astype(np.float32),
        scale=fit["scale"].astype(np.float32),
        factor_rotation=fit["factor_rotation"].astype(np.float32),
        representation=np.asarray(selected["selected_representation"]),
        regime_source=np.asarray(regime["selected_regime_source"]),
        regime_weights=np.asarray(
            regime["full_hmm"]["occupancy"] if regime["selected_regime_source"] == "hmm_emissions"
            else regime["full_static_gmm"]["weights"] if regime["selected_regime_source"] == "static_gmm"
            else regime["full_one_gaussian"]["weights"], dtype=np.float32
        ),
        regime_means=np.asarray(
            regime["full_hmm"]["means"] if regime["selected_regime_source"] == "hmm_emissions"
            else regime["full_static_gmm"]["means"] if regime["selected_regime_source"] == "static_gmm"
            else regime["full_one_gaussian"]["means"], dtype=np.float32
        ),
        regime_var=np.asarray(
            regime["full_hmm"]["var"] if regime["selected_regime_source"] == "hmm_emissions"
            else regime["full_static_gmm"]["var"] if regime["selected_regime_source"] == "static_gmm"
            else regime["full_one_gaussian"]["var"], dtype=np.float32
        ),
        hmm_initial=np.asarray(regime["full_hmm"]["initial"], dtype=np.float32),
        hmm_transition=np.asarray(regime["full_hmm"]["transition"], dtype=np.float32),
        action_hmm_initial=np.asarray(
            action_hmm["initial"] if action_hmm is not None else [], dtype=np.float32
        ),
        action_hmm_transition_coef=np.asarray(
            action_hmm["transition_coef"] if action_hmm is not None else [], dtype=np.float32
        ),
        action_hmm_means=np.asarray(
            action_hmm["means"] if action_hmm is not None else [], dtype=np.float32
        ),
        action_hmm_var=np.asarray(
            action_hmm["var"] if action_hmm is not None else [], dtype=np.float32
        ),
        action_hmm_cross_fitted_beliefs=np.asarray(
            action_hmm["cross_fitted_beliefs"] if action_hmm is not None else [], dtype=np.float32
        ),
        action_hmm_action_mean=np.asarray(
            action_hmm["action_mean"] if action_hmm is not None else [], dtype=np.float32
        ),
        action_hmm_action_scale=np.asarray(
            action_hmm["action_scale"] if action_hmm is not None else [], dtype=np.float32
        ),
        action_hmm_feature_mode=np.asarray(
            action_hmm["feature_mode"] if action_hmm is not None else "unavailable"
        ),
        offline_hmm_gates_pass=np.asarray(offline_hmm_gates_pass),
        hmm_deployment_eligible=np.asarray(hmm_deploy),
    )
    print(json.dumps({
        "selected_site": site,
        "selected_representation": selected["selected_representation"],
        "hmm_sequence_delta": regime["hmm_minus_static_gmm"],
        "history_decidable": history.get("decidable"),
        "hmm_deployment_eligible": hmm_deploy,
        "out": str(args.out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
