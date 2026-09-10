"""Scene-level inference helpers shared by the pilot analyses.

The unit of inference is the independently reset simulator scene (pair_id);
cells, tokens and imagined steps are repeated measures inside it.
"""

from __future__ import annotations

import itertools

import numpy as np


def sign_flip_p(values: np.ndarray, n_perm: int = 5000, seed: int = 0) -> float:
    """Two-sided sign-flip (permutation) test that the per-scene mean is 0.

    Exact enumeration for n <= 12 scenes, Monte Carlo otherwise. With n scenes
    the smallest attainable p is 2 / 2**n (n = 2 -> 0.5, n = 5 -> 0.0625).
    """

    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return float("nan")
    obs = abs(x.mean())
    if n <= 12:
        codes = np.arange(2**n)[:, None] >> np.arange(n)[None, :]
        signs = np.where(codes & 1, 1.0, -1.0)
        stats = np.abs((signs * x[None, :]).mean(axis=1))
        return float(np.mean(stats >= obs - 1e-15))
    rng = np.random.RandomState(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    stats = np.abs((signs * x[None, :]).mean(axis=1))
    return float((np.sum(stats >= obs - 1e-15) + 1) / (n_perm + 1))


def cluster_bootstrap_mean(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return {"mean": float("nan"), "median": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    rng = np.random.RandomState(seed)
    boots = np.array([x[rng.randint(0, n, n)].mean() for _ in range(n_boot)])
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
        "n": int(n),
    }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(a @ b / (na * nb))


def nmse(pred: np.ndarray, true: np.ndarray) -> float:
    p = np.asarray(pred, np.float64).ravel()
    t = np.asarray(true, np.float64).ravel()
    den = float(t @ t)
    if den == 0.0:
        return float("nan")
    return float(((p - t) @ (p - t)) / den)


def finite(obj):
    """Recursively convert numpy scalars and replace non-finite floats with None for JSON."""

    if isinstance(obj, dict):
        return {str(k): finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return finite(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
