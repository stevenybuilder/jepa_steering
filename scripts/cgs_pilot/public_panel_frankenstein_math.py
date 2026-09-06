#!/usr/bin/env python3
"""Reusable mathematics for the non-core Panel-P Frankenstein modules.

The functions are deliberately architecture independent.  They operate on
episode-grouped latent rows and executed controls, never on protected outcomes.
They implement five distinct objects which must not be conflated:

* calibrated probability geometry by factor block and latent regime;
* an outcome-agnostic overcomplete sparse dictionary and support statistics;
* matched-lure Fisher pattern separation and a runtime-preservation projection;
* orthogonal relational transport with an explicit row-vector convention;
* Q/K attention-retrieval diagnostics and attention-output restoration.
"""

from __future__ import annotations

import itertools

import numpy as np

from public_panel_factor_gate import bootstrap_mean_ci, episode_folds
from public_panel_sonar_math import fit_shrinkage_gaussian, log_gaussian, psd_power, symmetrize


SPARSE_ACTIVE_EPSILON = 1e-6
SPARSE_ISTA_ITERATIONS = 100


def weighted_quantile(values: np.ndarray, quantile: float, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    cumulative /= max(float(cumulative[-1]), 1e-300)
    return float(values[order[np.searchsorted(cumulative, quantile, side="left")]])


def fit_gaussian_family(
    rows: np.ndarray,
    family: str,
    *,
    weights: np.ndarray | None = None,
    shrinkage: float = 0.25,
    floor: float = 1e-3,
) -> dict[str, np.ndarray | float | str]:
    """Fit isotropic, diagonal, or shrinkage-full Gaussian geometry."""
    x = np.asarray(rows, dtype=np.float64)
    w = np.ones(len(x), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if x.ndim != 2 or w.shape != (len(x),) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("invalid rows or weights")
    w = w / w.sum()
    mean = w @ x
    centered = x - mean
    covariance = (centered * w[:, None]).T @ centered
    if family == "isotropic":
        variance = max(float(np.trace(covariance) / x.shape[1]), floor)
        covariance = variance * np.eye(x.shape[1])
    elif family == "diagonal":
        covariance = np.diag(np.maximum(np.diag(covariance), floor))
    elif family == "shrinkage_full":
        mean, covariance, _ = fit_shrinkage_gaussian(
            x, weights=w, shrinkage=shrinkage, floor=floor
        )
    else:
        raise ValueError(family)
    effective_n = float(1.0 / np.square(w).sum())
    return {"family": family, "mean": mean, "covariance": covariance, "effective_n": effective_n}


def mahalanobis_squared(rows: np.ndarray, model: dict) -> np.ndarray:
    centered = np.asarray(rows, dtype=np.float64) - np.asarray(model["mean"])
    precision = np.linalg.inv(symmetrize(np.asarray(model["covariance"])))
    return np.einsum("ni,ij,nj->n", centered, precision, centered)


def _episode_equal_rows(
    sequences: list[np.ndarray],
    responsibilities: list[np.ndarray],
    ids: np.ndarray,
    regime: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.concatenate([sequences[int(i)] for i in ids])
    weights = np.concatenate([
        responsibilities[int(i)][:, regime] / len(sequences[int(i)]) for i in ids
    ])
    return rows, weights


def probability_geometry_gate(
    sequences: list[np.ndarray],
    responsibilities: list[np.ndarray],
    *,
    blocks: list[tuple[int, int]] | None = None,
    n_folds: int = 5,
    seed: int = 0,
    shrinkage: float = 0.25,
    floor: float = 1e-3,
) -> dict:
    """Select and calibrate P(block activation | regime) out of episode."""
    if len(sequences) != len(responsibilities) or len(sequences) < 3:
        raise ValueError("matching activation/responsibility episodes are required")
    dimension = sequences[0].shape[1]
    if blocks is None:
        blocks = [(start, min(start + 2, dimension)) for start in range(0, dimension, 2)]
    if any(stop <= start or start < 0 or stop > dimension for start, stop in blocks):
        raise ValueError("invalid factor block")
    n_regimes = responsibilities[0].shape[1]
    folds = episode_folds(len(sequences), min(n_folds, len(sequences)), seed)
    all_ids = np.arange(len(sequences))
    families = ("isotropic", "diagonal", "shrinkage_full")
    block_reports = []
    selected_models = {}
    for block_index, (start, stop) in enumerate(blocks):
        scores = {family: np.zeros(len(sequences)) for family in families}
        coverages = {family: {0.90: [], 0.95: []} for family in families}
        for test_ids in folds:
            train_ids = all_ids[~np.isin(all_ids, test_ids)]
            models = {family: [] for family in families}
            thresholds = {family: [] for family in families}
            for family in families:
                for regime in range(n_regimes):
                    train_rows, train_weights = _episode_equal_rows(
                        [value[:, start:stop] for value in sequences], responsibilities, train_ids, regime
                    )
                    model = fit_gaussian_family(
                        train_rows, family, weights=train_weights, shrinkage=shrinkage, floor=floor
                    )
                    distance = mahalanobis_squared(train_rows, model)
                    models[family].append(model)
                    thresholds[family].append({
                        q: weighted_quantile(distance, q, train_weights) for q in (0.90, 0.95)
                    })
            for episode in test_ids:
                x = sequences[int(episode)][:, start:stop]
                q = responsibilities[int(episode)]
                for family in families:
                    component_log = np.stack([
                        log_gaussian(x, model["mean"], model["covariance"])
                        for model in models[family]
                    ], axis=1)
                    scores[family][episode] = float(np.mean(np.sum(q * component_log, axis=1)))
                    top = q.argmax(1)
                    distances = np.asarray([
                        mahalanobis_squared(x[row : row + 1], models[family][int(regime)])[0]
                        for row, regime in enumerate(top)
                    ])
                    for quantile in (0.90, 0.95):
                        inside = [
                            distances[row] <= thresholds[family][int(regime)][quantile]
                            for row, regime in enumerate(top)
                        ]
                        coverages[family][quantile].append(float(np.mean(inside)))
        mean_score = {family: float(values.mean()) for family, values in scores.items()}
        selected = max(families, key=lambda family: mean_score[family])
        selected_coverage = {
            str(quantile): float(np.mean(coverages[selected][quantile])) for quantile in (0.90, 0.95)
        }
        calibration_error = max(
            abs(selected_coverage[str(quantile)] - quantile) for quantile in (0.90, 0.95)
        )
        full_models = []
        for regime in range(n_regimes):
            rows, weights = _episode_equal_rows(
                [value[:, start:stop] for value in sequences], responsibilities, all_ids, regime
            )
            fitted = fit_gaussian_family(
                rows, selected, weights=weights, shrinkage=shrinkage, floor=floor
            )
            fitted["support_q99"] = weighted_quantile(
                mahalanobis_squared(rows, fitted), 0.99, weights
            )
            full_models.append(fitted)
        # Symmetric KL is a scale-aware statement that regime distributions differ;
        # it is not treated as an intervention probability.
        regime_divergence = []
        for left, right in itertools.combinations(range(n_regimes), 2):
            regime_divergence.append(symmetric_gaussian_kl(full_models[left], full_models[right]))
        selected_models[str(block_index)] = full_models
        block_reports.append({
            "block": [int(start), int(stop)],
            "heldout_log_score": mean_score,
            "selected_family": selected,
            "empirical_tail_coverage": selected_coverage,
            "maximum_calibration_error": float(calibration_error),
            "regime_symmetric_kl": regime_divergence,
            "calibrated": bool(calibration_error <= 0.10),
        })
    return {
        "blocks": block_reports,
        "all_blocks_calibrated": all(row["calibrated"] for row in block_reports),
        "at_least_one_regime_distribution_difference": any(
            max(row["regime_symmetric_kl"], default=0.0) >= 0.1 for row in block_reports
        ),
        "models": selected_models,
        "outcome_labels_used": False,
    }


def symmetric_gaussian_kl(left: dict, right: dict) -> float:
    def one(a: dict, b: dict) -> float:
        ma, mb = np.asarray(a["mean"]), np.asarray(b["mean"])
        ca, cb = np.asarray(a["covariance"]), np.asarray(b["covariance"])
        precision = np.linalg.inv(symmetrize(cb))
        delta = mb - ma
        sign_a, log_a = np.linalg.slogdet(ca)
        sign_b, log_b = np.linalg.slogdet(cb)
        if sign_a <= 0 or sign_b <= 0:
            raise ValueError("covariance must be positive definite")
        return 0.5 * (
            np.trace(precision @ ca) + delta @ precision @ delta - len(delta) + log_b - log_a
        )
    return float(0.5 * (one(left, right) + one(right, left)))


def encode_sparse(
    rows: np.ndarray, dictionary: np.ndarray, alpha: float, iterations: int = SPARSE_ISTA_ITERATIONS
) -> np.ndarray:
    """Deterministic ISTA encoder used identically offline and at runtime."""
    x = np.asarray(rows, dtype=np.float64)
    atoms = np.asarray(dictionary, dtype=np.float64)
    gram = atoms @ atoms.T
    lipschitz = max(float(np.linalg.eigvalsh(symmetrize(gram)).max()), 1e-8)
    step = 1.0 / lipschitz
    code = np.zeros((len(x), len(atoms)), dtype=np.float64)
    threshold = float(alpha) * step
    for _ in range(int(iterations)):
        gradient = (code @ atoms - x) @ atoms.T
        proposal = code - step * gradient
        code = np.sign(proposal) * np.maximum(np.abs(proposal) - threshold, 0.0)
    return code


def fit_sparse_dictionary(
    rows: np.ndarray,
    *,
    n_components: int,
    alpha: float,
    seed: int,
    max_iter: int = 250,
) -> dict[str, np.ndarray | float]:
    """Fit an outcome-agnostic overcomplete sparse dictionary."""
    from sklearn.decomposition import MiniBatchDictionaryLearning

    x = np.asarray(rows, dtype=np.float64)
    mean = x.mean(0)
    centered = x - mean
    estimator = MiniBatchDictionaryLearning(
        n_components=int(n_components),
        alpha=float(alpha),
        max_iter=int(max_iter),
        batch_size=min(128, len(x)),
        random_state=int(seed),
        transform_algorithm="lasso_lars",
        transform_alpha=float(alpha),
        fit_algorithm="cd",
    )
    estimator.fit(centered)
    dictionary = np.asarray(estimator.components_, dtype=np.float64)
    norm = np.maximum(np.linalg.norm(dictionary, axis=1, keepdims=True), 1e-12)
    dictionary /= norm
    # Recompute with the exact normalized-dictionary ISTA path used by both the
    # held-out gate and runtime.  Estimator.fit_transform uses a different solver
    # and its support must not define deployment probabilities.
    code = encode_sparse(centered, dictionary, alpha)
    return {"mean": mean, "dictionary": dictionary, "alpha": float(alpha), "code": code}


def sparse_reconstruct(rows: np.ndarray, model: dict) -> tuple[np.ndarray, np.ndarray]:
    centered = np.asarray(rows, dtype=np.float64) - np.asarray(model["mean"])
    code = encode_sparse(centered, np.asarray(model["dictionary"]), float(model["alpha"]))
    return np.asarray(model["mean"]) + code @ np.asarray(model["dictionary"]), code


def dictionary_atom_stability(left: np.ndarray, right: np.ndarray) -> float:
    from scipy.optimize import linear_sum_assignment

    similarity = np.abs(np.asarray(left) @ np.asarray(right).T)
    rows, columns = linear_sum_assignment(-similarity)
    return float(np.median(similarity[rows, columns]))


def _ridge_weights(x: np.ndarray, y: np.ndarray, ridge: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xm, ym = x.mean(0), y.mean(0)
    weights = np.linalg.solve(
        (x - xm).T @ (x - xm) + ridge * np.eye(x.shape[1]), (x - xm).T @ (y - ym)
    )
    return xm, ym, weights


def sparse_superposition_gate(
    sequences: list[np.ndarray],
    actions: list[np.ndarray] | None,
    *,
    responsibilities: list[np.ndarray] | None = None,
    n_components: int,
    alpha: float,
    n_folds: int = 5,
    seed: int = 0,
    max_iter: int = 250,
) -> dict:
    """Whole-episode sparse reconstruction, prediction, action, and stability gate."""
    if len(sequences) < 4:
        raise ValueError("at least four episodes are required")
    folds = episode_folds(len(sequences), min(n_folds, len(sequences)), seed)
    all_ids = np.arange(len(sequences))
    metrics = {name: [] for name in ("sparse_nmse", "pca_nmse", "random_nmse", "next_state_fidelity", "action_fidelity")}
    activity = []
    for fold_index, test_ids in enumerate(folds):
        train_ids = all_ids[~np.isin(all_ids, test_ids)]
        train = np.concatenate([sequences[int(i)] for i in train_ids])
        test = np.concatenate([sequences[int(i)] for i in test_ids])
        model = fit_sparse_dictionary(
            train, n_components=n_components, alpha=alpha, seed=seed + fold_index, max_iter=max_iter
        )
        reconstructed, code = sparse_reconstruct(test, model)
        denominator = float(np.square(test - train.mean(0)).mean() + 1e-12)
        metrics["sparse_nmse"].append(float(np.square(test - reconstructed).mean() / denominator))
        active = np.abs(code) > SPARSE_ACTIVE_EPSILON
        activity.append(active.mean(1))
        matched_rank = max(1, min(test.shape[1], int(np.ceil(active.sum(1).mean()))))
        _u, _s, vt = np.linalg.svd(train - train.mean(0), full_matrices=False)
        pca_basis = vt[:matched_rank]
        pca_reconstructed = train.mean(0) + (test - train.mean(0)) @ pca_basis.T @ pca_basis
        metrics["pca_nmse"].append(float(np.square(test - pca_reconstructed).mean() / denominator))
        random_dictionary = np.random.default_rng(seed + 991 * (fold_index + 1)).normal(
            size=np.asarray(model["dictionary"]).shape
        )
        random_dictionary /= np.maximum(np.linalg.norm(random_dictionary, axis=1, keepdims=True), 1e-12)
        random_code = encode_sparse(test - train.mean(0), random_dictionary, alpha)
        random_reconstructed = train.mean(0) + random_code @ random_dictionary
        metrics["random_nmse"].append(float(np.square(test - random_reconstructed).mean() / denominator))

        train_current = np.concatenate([sequences[int(i)][:-1] for i in train_ids])
        train_future = np.concatenate([sequences[int(i)][1:] for i in train_ids])
        test_current = np.concatenate([sequences[int(i)][:-1] for i in test_ids])
        reconstructed_current, _ = sparse_reconstruct(test_current, model)
        xm, ym, weights = _ridge_weights(train_current, train_future)
        direct = ym + (test_current - xm) @ weights
        reconstructed_prediction = ym + (reconstructed_current - xm) @ weights
        pred_variance = float(np.square(direct - direct.mean(0)).mean() + 1e-12)
        metrics["next_state_fidelity"].append(
            float(1.0 - np.square(direct - reconstructed_prediction).mean() / pred_variance)
        )
        if actions is not None:
            train_action = np.concatenate([actions[int(i)] for i in train_ids])
            test_action_rows = np.concatenate([actions[int(i)] for i in test_ids])
            if len(train_action) == len(train) and len(test_action_rows) == len(test):
                xm, ym, weights = _ridge_weights(train, train_action)
                direct = ym + (test - xm) @ weights
                reconstructed_action = ym + (reconstructed - xm) @ weights
                action_variance = float(np.square(direct - direct.mean(0)).mean() + 1e-12)
                metrics["action_fidelity"].append(
                    float(1.0 - np.square(direct - reconstructed_action).mean() / action_variance)
                )
    full = fit_sparse_dictionary(
        np.concatenate(sequences), n_components=n_components, alpha=alpha, seed=seed + 5001, max_iter=max_iter
    )
    split = len(sequences) // 2
    left = fit_sparse_dictionary(
        np.concatenate(sequences[:split]), n_components=n_components, alpha=alpha,
        seed=seed + 5002, max_iter=max_iter,
    )
    right = fit_sparse_dictionary(
        np.concatenate(sequences[split:]), n_components=n_components, alpha=alpha,
        seed=seed + 5003, max_iter=max_iter,
    )
    stability = dictionary_atom_stability(left["dictionary"], right["dictionary"])
    full_code = np.asarray(full.pop("code"))
    support = np.abs(full_code) > SPARSE_ACTIVE_EPSILON
    active_counts = support.sum(1)
    activation_probability = support.mean(0)
    support_entropy = -(
        activation_probability * np.log(activation_probability + 1e-30)
        + (1.0 - activation_probability) * np.log(1.0 - activation_probability + 1e-30)
    )
    if responsibilities is None:
        responsibilities = [np.ones((len(value), 1), dtype=np.float64) for value in sequences]
    if len(responsibilities) != len(sequences):
        raise ValueError("sparse support responsibilities must align by episode")
    regime_probability = np.concatenate(responsibilities)
    if len(regime_probability) != len(support):
        raise ValueError("sparse support responsibilities must align by row")
    regime_support_probability = []
    regime_magnitude = []
    for regime in range(regime_probability.shape[1]):
        weight = regime_probability[:, regime]
        mass = float(weight.sum())
        # Jeffreys smoothing prevents zero-probability support events online.
        regime_support_probability.append(
            ((weight[:, None] * support).sum(0) + 0.5) / (mass + 1.0)
        )
        regime_magnitude.append(
            (weight[:, None] * np.abs(full_code)).sum(0) / max(mass, 1e-12)
        )
    regime_support_probability = np.asarray(regime_support_probability)
    regime_magnitude = np.asarray(regime_magnitude)
    hard_regime = regime_probability.argmax(1)
    support_nll = np.empty(len(support))
    support_nll_threshold = np.empty(regime_probability.shape[1])
    for regime in range(regime_probability.shape[1]):
        p = np.clip(regime_support_probability[regime], 1e-6, 1.0 - 1e-6)
        rows = hard_regime == regime
        support_nll[rows] = -np.sum(
            support[rows] * np.log(p) + (~support[rows]) * np.log(1.0 - p), axis=1
        )
        support_nll_threshold[regime] = (
            float(np.quantile(support_nll[rows], 0.99)) if np.any(rows) else float("inf")
        )
    # Mutual information I(1[f_i!=0]; regime), reported per atom.
    regime_prior = regime_probability.mean(0)
    mutual_information = []
    for feature in range(support.shape[1]):
        value = 0.0
        for regime in range(regime_probability.shape[1]):
            for active in (0, 1):
                indicator = support[:, feature] if active else ~support[:, feature]
                joint = float(np.mean(regime_probability[:, regime] * indicator))
                marginal = float(np.mean(indicator))
                if joint > 0 and regime_prior[regime] > 0 and marginal > 0:
                    value += joint * np.log(joint / (regime_prior[regime] * marginal))
        mutual_information.append(value)
    frequent_sets = [
        set(np.flatnonzero(probability >= max(0.10, float(np.quantile(probability, 0.75)))))
        for probability in regime_support_probability
    ]
    support_jaccard = []
    for left, right in itertools.combinations(range(len(frequent_sets)), 2):
        union = frequent_sets[left] | frequent_sets[right]
        support_jaccard.append(
            float(len(frequent_sets[left] & frequent_sets[right]) / len(union)) if union else 1.0
        )
    mean_metrics = {name: (float(np.mean(value)) if value else None) for name, value in metrics.items()}
    action_ok = mean_metrics["action_fidelity"] is not None and mean_metrics["action_fidelity"] >= 0.80
    eligible = bool(
        mean_metrics["sparse_nmse"] < mean_metrics["random_nmse"]
        and mean_metrics["sparse_nmse"] <= 1.10 * mean_metrics["pca_nmse"]
        and mean_metrics["next_state_fidelity"] >= 0.80
        and action_ok
        and stability >= 0.60
        and float(np.mean(activity)) <= 0.25
        and float(np.mean(activation_probability < 1e-3)) <= 0.50
    )
    return {
        "cross_validated": mean_metrics,
        "mean_active_fraction": float(np.mean(activity)),
        "feature_activation_probability": activation_probability.tolist(),
        "feature_activation_probability_by_regime": regime_support_probability.tolist(),
        "feature_absolute_magnitude_by_regime": regime_magnitude.tolist(),
        "feature_regime_mutual_information_nats": mutual_information,
        "frequent_support_jaccard_between_regimes": support_jaccard,
        "support_negative_log_likelihood_q99_by_regime": support_nll_threshold.tolist(),
        "mean_support_entropy": float(support_entropy.mean()),
        "dead_feature_fraction": float(np.mean(activation_probability < 1e-3)),
        "active_count_empirical_q01_q99": [
            float(np.quantile(active_counts, 0.01)), float(np.quantile(active_counts, 0.99))
        ],
        "active_epsilon": SPARSE_ACTIVE_EPSILON,
        "ista_iterations": SPARSE_ISTA_ITERATIONS,
        "split_half_atom_cosine_median": stability,
        "eligible": eligible,
        "outcome_labels_used": False,
        "model": full,
    }


def fisher_pattern_basis(
    rows: np.ndarray,
    actions: np.ndarray,
    *,
    n_action_clusters: int = 3,
    rank: int = 2,
    ridge: float = 1e-2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Regularized Fisher directions for control-distinct latent states."""
    from sklearn.cluster import KMeans

    x = np.asarray(rows, dtype=np.float64)
    u = np.asarray(actions, dtype=np.float64)
    labels = KMeans(n_clusters=n_action_clusters, random_state=seed, n_init=10).fit_predict(u)
    global_mean = x.mean(0)
    between = np.zeros((x.shape[1], x.shape[1]))
    within = np.zeros_like(between)
    for label in range(n_action_clusters):
        group = x[labels == label]
        if len(group) < 2:
            continue
        delta = group.mean(0) - global_mean
        between += len(group) * np.outer(delta, delta)
        centered = group - group.mean(0)
        within += centered.T @ centered
    metric = within / max(len(x), 1) + ridge * np.eye(x.shape[1])
    whiten = psd_power(metric, -0.5, floor=1e-8)
    values, vectors = np.linalg.eigh(symmetrize(whiten @ (between / max(len(x), 1)) @ whiten))
    order = np.argsort(values)[::-1][: min(rank, x.shape[1])]
    basis = whiten @ vectors[:, order]
    basis, _ = np.linalg.qr(basis)
    return basis, labels, values[order]


def matched_lure_pairs(
    nuisance: np.ndarray,
    action_labels: np.ndarray,
    episode_ids: np.ndarray,
    *,
    max_pairs: int = 512,
) -> np.ndarray:
    """Nearest nuisance matches from other episodes requiring different controls."""
    nuisance = np.asarray(nuisance, dtype=np.float64)
    labels = np.asarray(action_labels)
    episodes = np.asarray(episode_ids)
    scale = np.maximum(nuisance.std(0), 1e-8)
    normalized = (nuisance - nuisance.mean(0)) / scale
    pairs = []
    for row in range(len(normalized)):
        allowed = (episodes != episodes[row]) & (labels != labels[row])
        candidates = np.flatnonzero(allowed)
        if len(candidates) == 0:
            continue
        distance = np.square(normalized[candidates] - normalized[row]).sum(1)
        partner = int(candidates[np.argmin(distance)])
        if row < partner:
            pairs.append((row, partner))
    pairs = sorted(set(pairs), key=lambda pair: float(np.linalg.norm(normalized[pair[0]] - normalized[pair[1]])))
    return np.asarray(pairs[:max_pairs], dtype=np.int64).reshape(-1, 2)


def pattern_preserving_delta(displacement: np.ndarray, basis: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Remove a registered fraction of edits along a stable lure-separation subspace."""
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must lie in [0,1]")
    delta = np.asarray(displacement, dtype=np.float64)
    q = np.asarray(basis, dtype=np.float64)
    return delta - float(strength) * (delta @ q) @ q.T


def pattern_separation_gate(
    sequences: list[np.ndarray],
    actions: list[np.ndarray] | None,
    *,
    nuisance: list[np.ndarray] | None = None,
    rank: int = 2,
    seed: int = 0,
) -> dict:
    if actions is None:
        return {"decidable": False, "eligible": False, "reason": "aligned executed controls are absent"}
    rows = np.concatenate(sequences)
    controls = np.concatenate(actions)
    if len(rows) != len(controls):
        return {"decidable": False, "eligible": False, "reason": "latent/action rows do not align"}
    episode_ids = np.concatenate([np.full(len(value), index) for index, value in enumerate(sequences)])
    progress = np.concatenate([
        np.linspace(0.0, 1.0, len(value), dtype=np.float64)[:, None] for value in sequences
    ])
    nuisance_rows = np.concatenate(nuisance) if nuisance is not None else progress
    from sklearn.cluster import KMeans

    folds = episode_folds(len(sequences), min(5, len(sequences)), seed + 71)
    per_episode_fisher = np.full(len(sequences), np.nan)
    per_episode_nuisance = np.full(len(sequences), np.nan)
    for fold_index, test_episodes in enumerate(folds):
        test = np.isin(episode_ids, test_episodes)
        train = ~test
        action_cluster = KMeans(
            n_clusters=3, random_state=seed + 100 + fold_index, n_init=10
        ).fit(controls[train])
        train_labels = action_cluster.labels_
        test_labels = action_cluster.predict(controls[test])
        fold_basis, _labels, _values = fisher_pattern_basis(
            rows[train], controls[train], rank=rank, seed=seed + 100 + fold_index
        )
        train_projected = rows[train] @ fold_basis
        test_projected = rows[test] @ fold_basis
        fisher_centers = np.stack([
            train_projected[train_labels == label].mean(0) for label in range(3)
        ])
        nuisance_centers = np.stack([
            nuisance_rows[train][train_labels == label].mean(0) for label in range(3)
        ])
        fisher_prediction = np.square(
            test_projected[:, None, :] - fisher_centers[None, :, :]
        ).sum(2).argmin(1)
        nuisance_scale = np.maximum(nuisance_rows[train].std(0), 1e-8)
        nuisance_prediction = np.square(
            (nuisance_rows[test, None, :] - nuisance_centers[None, :, :]) / nuisance_scale
        ).sum(2).argmin(1)
        test_episode_rows = episode_ids[test]
        for episode in test_episodes:
            mask = test_episode_rows == episode
            per_episode_fisher[episode] = float(np.mean(fisher_prediction[mask] == test_labels[mask]))
            per_episode_nuisance[episode] = float(np.mean(nuisance_prediction[mask] == test_labels[mask]))
    accuracy_gain = per_episode_fisher - per_episode_nuisance
    accuracy_ci = bootstrap_mean_ci(accuracy_gain, seed + 1701)
    basis, labels, eigenvalues = fisher_pattern_basis(rows, controls, rank=rank, seed=seed)
    pairs = matched_lure_pairs(nuisance_rows, labels, episode_ids)
    if len(pairs) < max(8, len(sequences) // 2):
        return {"decidable": False, "eligible": False, "reason": "too few cross-episode matched lures"}
    projected = rows @ basis
    margin = np.linalg.norm(projected[pairs[:, 0]] - projected[pairs[:, 1]], axis=1)
    full = np.linalg.norm(rows[pairs[:, 0]] - rows[pairs[:, 1]], axis=1)
    split = len(sequences) // 2
    left_mask = episode_ids < split
    right_mask = ~left_mask
    left_basis, _, _ = fisher_pattern_basis(rows[left_mask], controls[left_mask], rank=rank, seed=seed + 1)
    right_basis, _, _ = fisher_pattern_basis(rows[right_mask], controls[right_mask], rank=rank, seed=seed + 2)
    singular = np.linalg.svd(left_basis.T @ right_basis, compute_uv=False)
    angle = float(np.degrees(np.sqrt(np.mean(np.square(np.arccos(np.clip(singular, -1.0, 1.0)))))))
    # Permuting action requirements is the matched null for the observed lure margin.
    null = []
    rng = np.random.default_rng(seed + 3)
    for _ in range(100):
        shuffled = rng.permutation(labels)
        null_pairs = matched_lure_pairs(nuisance_rows, shuffled, episode_ids, max_pairs=len(pairs))
        if len(null_pairs):
            null.append(float(np.median(np.linalg.norm(
                projected[null_pairs[:, 0]] - projected[null_pairs[:, 1]], axis=1
            ))))
    observed = float(np.median(margin))
    p_value = float((1 + np.sum(np.asarray(null) >= observed)) / (1 + len(null)))
    eligible = bool(
        angle <= 35.0 and observed > 0.0 and p_value <= 0.10 and accuracy_ci[0] > 0.0
    )
    return {
        "decidable": True,
        "n_lure_pairs": int(len(pairs)),
        "median_fisher_margin": observed,
        "median_full_space_margin": float(np.median(full)),
        "action_label_permutation_p": p_value,
        "heldout_action_cluster_accuracy": {
            "fisher": float(per_episode_fisher.mean()),
            "nuisance": float(per_episode_nuisance.mean()),
            "fisher_minus_nuisance": float(accuracy_gain.mean()),
            "episode_bootstrap_95_ci": list(accuracy_ci),
        },
        "split_half_subspace_angle_deg": angle,
        "generalized_eigenvalues": eigenvalues.tolist(),
        "eligible": eligible,
        "basis": basis,
        "pairs": pairs,
        "outcome_labels_used": False,
    }


def fit_orthogonal_transport(source: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray | float]:
    """Fit row-vector map ``target ~= (source-mu_s) @ Q + mu_t``."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2:
        raise ValueError("matched source and target rows must have the same [N,D] shape")
    mean_source, mean_target = source.mean(0), target.mean(0)
    u, _s, vt = np.linalg.svd((source - mean_source).T @ (target - mean_target))
    rotation = u @ vt
    prediction = (source - mean_source) @ rotation + mean_target
    residual = float(
        np.linalg.norm(prediction - target, "fro") /
        max(np.linalg.norm(target - mean_target, "fro"), 1e-12)
    )
    return {"mean_source": mean_source, "mean_target": mean_target, "rotation": rotation, "relative_residual": residual}


def apply_orthogonal_transport(rows: np.ndarray, model: dict) -> np.ndarray:
    return (
        (np.asarray(rows) - np.asarray(model["mean_source"])) @ np.asarray(model["rotation"])
        + np.asarray(model["mean_target"])
    )


def transport_direction(direction: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return np.asarray(direction) @ np.asarray(rotation)


def transport_operator(operator: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Conjugate an operator used as row edit ``x' = x @ operator.T``."""
    q = np.asarray(rotation)
    return q.T @ np.asarray(operator) @ q


def energy_distance(left: np.ndarray, right: np.ndarray) -> float:
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    cross = np.linalg.norm(left[:, None] - right[None, :], axis=2).mean()
    within_left = np.linalg.norm(left[:, None] - left[None, :], axis=2).mean()
    within_right = np.linalg.norm(right[:, None] - right[None, :], axis=2).mean()
    return float(max(2.0 * cross - within_left - within_right, 0.0))


def attention_probabilities(query: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Exact softmax(QK^T/sqrt(d)) for arrays [N,H,T,D]."""
    q, k = np.asarray(query, dtype=np.float64), np.asarray(key, dtype=np.float64)
    if q.shape != k.shape or q.ndim != 4:
        raise ValueError("query/key must share [N,H,T,D]")
    scores = np.einsum("nhtd,nhsd->nhts", q, k) / np.sqrt(q.shape[-1])
    scores -= scores.max(axis=-1, keepdims=True)
    exp = np.exp(scores)
    return exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-300)


def attention_retrieval_gate(
    residual: np.ndarray,
    query: np.ndarray,
    key: np.ndarray,
    regime: np.ndarray,
    *,
    episode_ids: np.ndarray | None = None,
    seed: int = 0,
) -> dict:
    """Test whether Q/K projections amplify regime separation beyond residuals."""
    labels = np.asarray(regime).argmax(1) if np.asarray(regime).ndim == 2 else np.asarray(regime)
    if len(np.unique(labels)) < 2:
        return {"decidable": False, "eligible": False, "reason": "only one occupied regime"}
    q = np.asarray(query).mean(axis=2).reshape(len(labels), -1)
    k = np.asarray(key).mean(axis=2).reshape(len(labels), -1)
    h = np.asarray(residual).reshape(len(labels), -1)
    groups = [np.flatnonzero(labels == value) for value in np.unique(labels)]
    if min(len(group) for group in groups) < 4:
        return {"decidable": False, "eligible": False, "reason": "insufficient rows per regime"}
    def separation(rows: np.ndarray, group_rows: list[np.ndarray] = groups) -> float:
        # Per-coordinate standardization plus sqrt(width) normalization makes
        # residual/Q/K distances comparable despite different projection widths.
        scale = np.maximum(rows.std(0), 1e-8)
        normalized = (rows - rows.mean(0)) / scale
        between = np.mean([
            energy_distance(normalized[a], normalized[b])
            for a, b in itertools.combinations(group_rows, 2)
        ])
        return float(between / np.sqrt(rows.shape[1]))
    values = {"residual": separation(h), "query": separation(q), "key": separation(k)}
    probability = attention_probabilities(query, key)
    entropy = -np.sum(probability * np.log(probability + 1e-30), axis=-1).mean(axis=(1, 2))
    gain = max(values["query"], values["key"]) / max(values["residual"], 1e-12) - 1.0
    gain_interval = None
    permutation_p = None
    if episode_ids is not None:
        episode_ids = np.asarray(episode_ids)
        if episode_ids.shape != labels.shape:
            raise ValueError("attention episode IDs must align with rows")
        unique_episodes = np.unique(episode_ids)
        rng = np.random.default_rng(seed)
        gains = []
        for _ in range(500):
            sampled = rng.choice(unique_episodes, len(unique_episodes), replace=True)
            rows = np.concatenate([np.flatnonzero(episode_ids == episode) for episode in sampled])
            sampled_labels = labels[rows]
            sampled_groups = [np.flatnonzero(sampled_labels == value) for value in np.unique(labels)]
            if min(len(group) for group in sampled_groups) < 2:
                continue
            base = separation(h[rows], sampled_groups)
            projected = max(separation(q[rows], sampled_groups), separation(k[rows], sampled_groups))
            gains.append(projected / max(base, 1e-12) - 1.0)
        if gains:
            gain_interval = [float(np.quantile(gains, 0.025)), float(np.quantile(gains, 0.975))]
        null = []
        for _ in range(100):
            shuffled = labels.copy()
            for episode in unique_episodes:
                mask = episode_ids == episode
                shuffled[mask] = rng.permutation(shuffled[mask])
            shuffled_groups = [np.flatnonzero(shuffled == value) for value in np.unique(labels)]
            null.append(max(separation(q, shuffled_groups), separation(k, shuffled_groups)))
        observed_projected = max(values["query"], values["key"])
        permutation_p = float((1 + np.sum(np.asarray(null) >= observed_projected)) / (1 + len(null)))
    eligible = bool(
        gain >= 0.05
        and (gain_interval is None or gain_interval[0] > 0.0)
        and (permutation_p is None or permutation_p <= 0.10)
    )
    return {
        "decidable": True,
        "normalized_regime_energy_distance": values,
        "q_or_k_gain_over_residual_fraction": float(gain),
        "episode_bootstrap_95_ci": gain_interval,
        "within_episode_regime_permutation_p": permutation_p,
        "qk_array_attention_entropy_mean": (
            float(entropy.mean()) if np.asarray(query).shape[2] > 1 else None
        ),
        "qk_array_attention_entropy_descriptive_only": True,
        "eligible": eligible,
    }


def restore_attention_output(steered_output: np.ndarray, clean_output: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Causal-mediation primitive: restore a fraction of the paired clean attention output."""
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must lie in [0,1]")
    steered, clean = np.asarray(steered_output), np.asarray(clean_output)
    if steered.shape != clean.shape:
        raise ValueError("paired attention outputs must share a shape")
    return steered + float(strength) * (clean - steered)
