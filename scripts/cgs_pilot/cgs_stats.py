#!/usr/bin/env python3
"""Scene-level inference for the CGS pilot (no model dependencies).

Decision statistic: leave-one-scene-out (LOSO) cross-validated projection.  For
per-scene effect vectors ``I_1..I_n`` (one per scene/cluster, tokens already
pooled), scene ``i`` is scored by ``I_i . u_{-i}`` with ``u_{-i}`` the unit mean of
the other scenes.  Its expectation is 0 under a sign-symmetric null, unlike the
RMS of the mean vector, which is positively biased.  Everything is computed from
the Gram matrix so sign-flip permutations are O(n^2).

Significance: Fisher sign-flip randomization over scenes (exact enumeration when
``2**n <= n_perm``); max-T across sites at a step for family-wise control
(Westfall-Young); Benjamini-Yekutieli FDR for the descriptive map.  The clustered
bootstrap is kept for CIs only.

References: arXiv:2309.16042 (activation-patching best practices),
arXiv:2510.00845 (mediation-score variance motivates band rather than index
claims), arXiv:2606.27510 (INT: interaction of patching two sites),
arXiv:2511.04638 (off-manifold divergence of edits), arXiv:2602.07050 (physics
emergence zone / circular geometry in world models).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


SD_FLOOR = 1e-8


# --------------------------------------------------------------------------- #
# LOSO cross-validated projection
# --------------------------------------------------------------------------- #


def loso_projection_scores(vectors: np.ndarray, signs: np.ndarray | None = None) -> np.ndarray:
    """Per-scene scores ``s_i I_i . unit(sum_{j!=i} s_j I_j)`` from the Gram matrix."""

    x = np.asarray(vectors, dtype=np.float64)
    n = x.shape[0]
    if n < 2:
        return np.full(n, np.nan)
    s = np.ones(n) if signs is None else np.asarray(signs, dtype=np.float64)
    gram = x @ x.T
    sg = gram * s[None, :]  # G_ij s_j
    row = sg.sum(axis=1)  # sum_j G_ij s_j
    dots = s * row - np.diag(gram)  # s_i I_i . (sum_j s_j I_j - s_i I_i)
    total = float(s @ gram @ s)
    norms_sq = total - 2.0 * s * row + np.diag(gram)  # ||sum_{j!=i} s_j I_j||^2
    norms = np.sqrt(np.clip(norms_sq, 0.0, None))
    return dots / np.where(norms > 1e-12, norms, np.nan)


def cv_projection(vectors: np.ndarray) -> dict[str, Any]:
    scores = loso_projection_scores(vectors)
    valid = scores[~np.isnan(scores)]
    n = len(valid)
    mean = float(valid.mean()) if n else float("nan")
    sd = float(valid.std(ddof=1)) if n > 1 else float("nan")
    # sd below SD_FLOOR is degenerate (e.g. AdaLN vectors identical across scenes): no t, no d_z
    t = mean / (sd / np.sqrt(n)) if n > 1 and sd >= SD_FLOOR else float("nan")
    d_z = mean / sd if n > 1 and sd >= SD_FLOOR else float("nan")
    return {"mean": mean, "sd": sd, "t": t, "d_z": d_z, "n_scenes": int(n), "per_scene": [float(v) for v in scores]}


def _t_stat(values: np.ndarray) -> float:
    v = values[~np.isnan(values)]
    if len(v) < 2:
        return float("nan")
    sd = v.std(ddof=1)
    # sd below SD_FLOOR is degenerate (e.g. identically-zero effects); report nan rather than +-inf
    return float(v.mean() / (sd / np.sqrt(len(v)))) if sd >= SD_FLOOR else float("nan")


def sign_flip_matrix(n: int, n_perm: int, rng: np.random.Generator) -> np.ndarray:
    """[n_perm, n] of +-1; exact enumeration when 2**n <= n_perm."""

    if n <= 0:
        return np.zeros((0, 0))
    if 2**n <= n_perm:
        grid = ((np.arange(2**n)[:, None] >> np.arange(n)[None, :]) & 1) * 2 - 1
        return grid.astype(np.float64)
    return rng.choice([-1.0, 1.0], size=(n_perm, n))


def sign_flip_maxt(
    per_site_vectors: dict[str, np.ndarray],
    n_perm: int,
    rng: np.random.Generator,
    statistic: Callable[[np.ndarray, np.ndarray | None], np.ndarray] = loso_projection_scores,
) -> dict[str, Any]:
    """Sign-flip test for the LOSO projection at every site sharing the same scenes.

    ``per_site_vectors[site] = [n_scenes, d]`` (same scene order everywhere).
    Returns per-site observed t, raw p (one-sided, signal>0), max-T FWER p, and
    the max-T null quantiles.  Sign flips are shared across sites per permutation.
    """

    sites = list(per_site_vectors)
    if not sites:
        return {"sites": {}, "n_scenes": 0, "n_perm": 0}
    n = next(iter(per_site_vectors.values())).shape[0]
    flips = sign_flip_matrix(n, n_perm, rng)
    observed = {s: _t_stat(statistic(per_site_vectors[s], None)) for s in sites}
    null = np.full((flips.shape[0], len(sites)), np.nan)
    for b, signs in enumerate(flips):
        for k, s in enumerate(sites):
            null[b, k] = _t_stat(statistic(per_site_vectors[s], signs))
    max_null = np.nanmax(null, axis=1) if null.size else np.zeros(0)
    out: dict[str, Any] = {"n_scenes": int(n), "n_perm": int(flips.shape[0]), "exact": bool(2**n <= n_perm), "sites": {}}
    for k, s in enumerate(sites):
        obs = observed[s]
        col = null[:, k]
        if np.isnan(obs):
            raw = fwer = float("nan")
        else:
            raw = float((np.sum(col[~np.isnan(col)] >= obs - 1e-12) + 1) / (np.sum(~np.isnan(col)) + 1))
            fwer = float((np.sum(max_null[~np.isnan(max_null)] >= obs - 1e-12) + 1) / (np.sum(~np.isnan(max_null)) + 1))
        out["sites"][s] = {"t": obs, "p_raw": raw, "p_maxt_fwer": fwer}
    finite = max_null[np.isfinite(max_null)]
    out["maxt_null_quantiles"] = (
        {q: float(np.quantile(finite, q)) for q in (0.5, 0.9, 0.95, 0.99)} if len(finite) else {}
    )
    out["n_sites"] = len(sites)
    return out


def benjamini_yekutieli(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, Any]:
    keys = [k for k, p in pvalues.items() if p is not None and not np.isnan(p)]
    if not keys:
        return {"alpha": alpha, "n_tests": 0, "rejected": [], "q": {}}
    p = np.asarray([pvalues[k] for k in keys])
    m = len(p)
    c_m = float(np.sum(1.0 / np.arange(1, m + 1)))
    order = np.argsort(p)
    ranked = p[order]
    q = np.minimum.accumulate((ranked * m * c_m / np.arange(1, m + 1))[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    qmap = {keys[i]: float(q[r]) for r, i in enumerate(order)}
    return {"alpha": alpha, "n_tests": int(m), "rejected": [k for k in keys if qmap[k] <= alpha], "q": qmap}


# --------------------------------------------------------------------------- #
# Clustered bootstrap (CIs only)
# --------------------------------------------------------------------------- #


def cluster_bootstrap(
    values: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
    statistic: Callable[[np.ndarray], float] = np.mean,
    alpha: float = 0.05,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    n = len(values)
    point = float(statistic(values)) if n else float("nan")
    if n < 2 or n_boot <= 0:
        return {"point": point, "ci_low": None, "ci_high": None, "n_clusters": int(n)}
    idx = rng.integers(0, n, size=(n_boot, n))
    draws = np.asarray([statistic(values[i]) for i in idx])
    return {
        "point": point,
        "ci_low": float(np.quantile(draws, alpha / 2)),
        "ci_high": float(np.quantile(draws, 1 - alpha / 2)),
        "n_clusters": int(n),
    }


def json_safe(value: Any) -> Any:
    """Replace NaN/inf with None recursively so canonical_json (allow_nan=False) accepts it."""

    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    return value
