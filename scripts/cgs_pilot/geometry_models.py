#!/usr/bin/env python3
"""Coordinate models, metrics and edit-direction export for the CGS geometry tournament.

numpy-only so it runs on the CPU-only laptop and on the remote box alike.

Every coordinate model exposes the same interface::

    model = MODELS[name](budget=k, **kwargs)
    model.fit(X, y, meta)             # X [n, d] float, y [n], meta: dict of per-cell arrays
    s = model.score(X)                # [n] scalar readout used for AUROC / R^2
    model.geometry_summary()          # dict: what chart was fit (rank, directions, circularity...)
    delta = model.edit_direction(x, target, norm)   # [d] near-manifold edit for one cell
    sim = model.similarity(other)     # 0..1 chart agreement with a refit (stability)

``budget`` is the number of representation dimensions the model may use, so
entrants are compared at equal capacity (design section "Phase 2").

The ``TranscoderPlaceholder`` entrant is deliberately unimplemented: per the
design it may only enter after causal localization (Phase 3) and only as a delta
edit ``MLP(z) + W_dec[i] * dF_i`` on the native MLP output.

Edits are additive and norm-controlled. The reference entrant is
``counterfactual_pair_direction``: the mean of matched within-scene H0->H1
activation deltas, i.e. a direction taken from real counterfactual pairs rather
than a fitted axis (design 1.4: state-conditioned donors before static
steering). Every other model's edit is compared to it on cosine and on an
on-manifold score (diagonal-covariance Mahalanobis distance of ``z + delta`` to
the site's clean distribution), following the manifold-steering argument of
arXiv:2605.05115 that off-manifold edits produce large but uninterpretable
effects. The ``circular`` chart follows the circular population-geometry
analysis of arXiv:2602.07050 (phase decoded from a fitted plane, readout as
harmonics of the phase).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np


# --------------------------------------------------------------------------- #
# Basic linear algebra helpers
# --------------------------------------------------------------------------- #


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    """Ridge regression with intercept. Uses the dual form when d > n."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xm = X.mean(axis=0)
    ym = float(y.mean())
    Xc = X - xm
    yc = y - ym
    n, d = Xc.shape
    if d <= n:
        A = Xc.T @ Xc + lam * np.eye(d)
        w = np.linalg.solve(A, Xc.T @ yc)
    else:
        K = Xc @ Xc.T + lam * np.eye(n)
        alpha = np.linalg.solve(K, yc)
        w = Xc.T @ alpha
    b = ym - float(xm @ w)
    return w, b


def ridge_predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return np.asarray(X, dtype=np.float64) @ w + b


def pca_basis(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-k principal directions of X (rows are samples). Returns (mean, basis [k,d], explained variance ratio [k])."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    Xc = X - mu
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    var = s**2
    ratio = var / max(var.sum(), 1e-12)
    k = int(min(k, vt.shape[0]))
    return mu, vt[:k], ratio[:k]


def effective_rank(X: np.ndarray, threshold: float = 0.90) -> int:
    """Number of principal components needed to explain ``threshold`` of the variance."""
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0)
    s = np.linalg.svd(Xc, compute_uv=False)
    var = s**2
    if var.sum() <= 0:
        return 0
    cum = np.cumsum(var) / var.sum()
    return int(np.searchsorted(cum, threshold) + 1)


def subspace_similarity(A: np.ndarray, B: np.ndarray) -> float:
    """Mean squared cosine of principal angles between row-spaces of A and B (1 = identical)."""
    if A.size == 0 or B.size == 0:
        return float("nan")
    qa, _ = np.linalg.qr(np.asarray(A, dtype=np.float64).T)
    qb, _ = np.linalg.qr(np.asarray(B, dtype=np.float64).T)
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return float(np.mean(s**2))


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC with tie handling; nan if one class is absent."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def r_squared(pred: np.ndarray, y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    sst = float(((y - y.mean()) ** 2).sum())
    if sst <= 0:
        return float("nan")
    return float(1.0 - ((y - pred) ** 2).sum() / sst)


def cluster_bootstrap(
    stat: Callable[[np.ndarray], float],
    clusters: np.ndarray,
    n_boot: int = 500,
    seed: int = 0,
) -> dict[str, float]:
    """Scene-clustered bootstrap of ``stat(index_array)``. Clusters (pair_ids) are resampled with replacement."""
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    members = {c: np.flatnonzero(clusters == c) for c in uniq}
    rng = np.random.default_rng(seed)
    point = stat(np.arange(len(clusters)))
    draws = []
    for _ in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([members[c] for c in picked])
        v = stat(idx)
        if np.isfinite(v):
            draws.append(v)
    if not draws:
        return {"point": float(point), "ci_low": float("nan"), "ci_high": float("nan"), "n_boot": 0}
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"point": float(point), "ci_low": float(lo), "ci_high": float(hi), "n_boot": len(draws)}


def scene_folds(clusters: np.ndarray, n_folds: int, seed: int = 0) -> list[np.ndarray]:
    """Split by cluster (pair_id): returns held-out index arrays, one per fold."""
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    n_folds = int(max(1, min(n_folds, len(uniq))))
    folds = []
    for f in range(n_folds):
        held = perm[f::n_folds]
        folds.append(np.flatnonzero(np.isin(clusters, held)))
    return folds


# --------------------------------------------------------------------------- #
# Coordinate models
# --------------------------------------------------------------------------- #


@dataclass
class CoordinateModel:
    budget: int = 1
    lam: float = 1.0
    name: str = "base"
    fitted: bool = field(default=False, init=False)

    def fit(self, X: np.ndarray, y: np.ndarray, meta: dict[str, np.ndarray]) -> "CoordinateModel":
        raise NotImplementedError

    def score(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def geometry_summary(self) -> dict[str, Any]:
        return {}

    def edit_direction(self, x: np.ndarray, target: float, norm: float) -> np.ndarray:
        raise NotImplementedError

    def similarity(self, other: "CoordinateModel") -> float:
        return float("nan")


@dataclass
class LinearMeanDiff(CoordinateModel):
    """One-dimensional direction: class-mean difference (binary y) or covariance direction (continuous y)."""

    name: str = "linear_meandiff"
    direction: np.ndarray | None = field(default=None, init=False)
    mu: np.ndarray | None = field(default=None, init=False)
    readout: tuple[float, float] = field(default=(1.0, 0.0), init=False)

    def fit(self, X, y, meta):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.mu = X.mean(axis=0)
        uniq = np.unique(y)
        if len(uniq) == 2:
            d = X[y == uniq.max()].mean(axis=0) - X[y == uniq.min()].mean(axis=0)
        else:
            d = ((X - self.mu) * (y - y.mean())[:, None]).mean(axis=0)
        self.direction = unit(d)
        proj = (X - self.mu) @ self.direction
        # 1-d calibration so score is on the label scale
        w, b = ridge_fit(proj[:, None], y, 1e-6)
        self.readout = (float(w[0]), float(b))
        self.fitted = True
        return self

    def score(self, X):
        proj = (np.asarray(X, dtype=np.float64) - self.mu) @ self.direction
        return proj * self.readout[0] + self.readout[1]

    def geometry_summary(self):
        return {"rank": 1, "kind": "direction"}

    def edit_direction(self, x, target, norm):
        sign = 1.0 if (target - float(self.score(x[None])[0])) * self.readout[0] >= 0 else -1.0
        return sign * norm * self.direction

    def similarity(self, other):
        return float(abs(self.direction @ other.direction))


@dataclass
class LinearRidge(CoordinateModel):
    """Ridge probe (continuous y) / ridge on +-1 labels (binary y). One readout direction."""

    name: str = "linear_ridge"
    w: np.ndarray | None = field(default=None, init=False)
    b: float = field(default=0.0, init=False)

    def fit(self, X, y, meta):
        self.w, self.b = ridge_fit(X, y, self.lam)
        self.fitted = True
        return self

    def score(self, X):
        return ridge_predict(X, self.w, self.b)

    def geometry_summary(self):
        return {"rank": 1, "kind": "direction", "weight_norm": float(np.linalg.norm(self.w))}

    def edit_direction(self, x, target, norm):
        cur = float(self.score(x[None])[0])
        sign = 1.0 if target >= cur else -1.0
        return sign * norm * unit(self.w)

    def similarity(self, other):
        return float(abs(unit(self.w) @ unit(other.w)))


@dataclass
class LowRankSubspace(CoordinateModel):
    """Reduced-rank regression: PCA of the *supervised residual* structure, then ridge inside the k-dim subspace.

    The subspace is the top-k PCA of ``X`` re-weighted by centred labels (a
    PLS-style first pass), so it is driven by label-relevant variance, not the
    dominant nuisance axes.
    """

    name: str = "lowrank_subspace"
    mu: np.ndarray | None = field(default=None, init=False)
    basis: np.ndarray | None = field(default=None, init=False)
    w: np.ndarray | None = field(default=None, init=False)
    b: float = field(default=0.0, init=False)
    explained: list[float] = field(default_factory=list, init=False)

    def fit(self, X, y, meta):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.mu = X.mean(axis=0)
        Xc = X - self.mu
        yc = y - y.mean()
        # PLS-style: weight rows by |yc| so label-relevant variance dominates
        weights = np.abs(yc) / max(np.abs(yc).max(), 1e-12)
        _, basis, ratio = pca_basis(Xc * (0.25 + weights)[:, None], self.budget)
        self.basis = basis
        self.explained = [float(r) for r in ratio]
        Z = Xc @ self.basis.T
        self.w, self.b = ridge_fit(Z, y, self.lam)
        self.fitted = True
        return self

    def _coords(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mu) @ self.basis.T

    def score(self, X):
        return ridge_predict(self._coords(X), self.w, self.b)

    def geometry_summary(self):
        return {"rank": int(self.basis.shape[0]), "kind": "subspace", "explained_ratio": self.explained}

    def edit_direction(self, x, target, norm):
        cur = float(self.score(x[None])[0])
        sign = 1.0 if target >= cur else -1.0
        d = self.basis.T @ self.w  # readout direction inside the subspace
        return sign * norm * unit(d)

    def similarity(self, other):
        return subspace_similarity(self.basis, other.basis)


@dataclass
class CircularFactorized(CoordinateModel):
    """Sine/cosine chart.

    Fit: ``X ~ mu + A [cos th, sin th]`` with the supervising angle
    ``meta['angle']`` (hazard bearing relative to the candidate approach path).
    ``A`` (2 x d) is the circular plane. At readout time the phase is *decoded*
    from the activation, ``th_hat = atan2(z1, z0)`` with ``z = (x - mu) A^+``,
    and the label is a ridge over harmonics ``[cos k th_hat, sin k th_hat]`` up to
    ``budget`` coordinates. That makes the chart genuinely circular: a readout
    that is nonlinear on the plane (e.g. ``cos 2 th``) is representable here but
    not by a linear direction.
    """

    name: str = "circular"
    mu: np.ndarray | None = field(default=None, init=False)
    A: np.ndarray | None = field(default=None, init=False)  # [2, d] plane loadings
    w: np.ndarray | None = field(default=None, init=False)
    b: float = field(default=0.0, init=False)
    circularity_r2: float = field(default=float("nan"), init=False)
    phase_r2: float = field(default=float("nan"), init=False)
    f_mean: np.ndarray | None = field(default=None, init=False)

    def _harmonics(self, angle: np.ndarray) -> np.ndarray:
        k = max(2, self.budget)
        cols = []
        h = 1
        while len(cols) < k:
            cols.append(np.cos(h * angle))
            if len(cols) < k:
                cols.append(np.sin(h * angle))
            h += 1
        return np.stack(cols, axis=1)

    def fit(self, X, y, meta):
        if "angle" not in meta:
            raise ValueError("CircularFactorized needs meta['angle']")
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        angle = np.asarray(meta["angle"], dtype=np.float64)
        F = np.stack([np.cos(angle), np.sin(angle)], axis=1)
        self.mu = X.mean(axis=0)
        Xc = X - self.mu
        self.f_mean = F.mean(axis=0)
        Fc = F - self.f_mean
        # least-squares plane loadings: Xc ~ Fc @ A
        self.A = np.linalg.lstsq(Fc, Xc, rcond=None)[0]
        recon = Fc @ self.A
        sst = float((Xc**2).sum())
        self.circularity_r2 = float(1.0 - ((Xc - recon) ** 2).sum() / sst) if sst > 0 else float("nan")
        # how well the decoded phase reproduces the supervising angle (circular R^2)
        th = self._phase(X)
        self.phase_r2 = float(np.mean(np.cos(th - angle)))
        self.w, self.b = ridge_fit(self._harmonics(th), y, self.lam)
        self.fitted = True
        return self

    def _phase(self, X) -> np.ndarray:
        Xc = np.asarray(X, dtype=np.float64) - self.mu
        z = Xc @ np.linalg.pinv(self.A) + self.f_mean
        return np.arctan2(z[:, 1], z[:, 0])

    def score(self, X):
        return ridge_predict(self._harmonics(self._phase(X)), self.w, self.b)

    def geometry_summary(self):
        return {
            "rank": 2,
            "kind": "circular",
            "harmonics": int(max(2, self.budget)),
            "circularity_r2": self.circularity_r2,
            "phase_agreement": self.phase_r2,
        }

    def edit_direction(self, x, target, norm):
        """Move along the circle tangent (on the fitted plane) in the phase direction that moves the readout toward target."""
        theta = float(self._phase(x[None])[0])
        cur = float(self.score(x[None])[0])
        eps = 1e-3
        up = float(ridge_predict(self._harmonics(np.array([theta + eps])), self.w, self.b)[0])
        sign = 1.0 if (target - cur) * (up - cur) >= 0 else -1.0
        tangent = np.array([-np.sin(theta), np.cos(theta)]) @ self.A
        return sign * norm * unit(tangent)

    def similarity(self, other):
        return subspace_similarity(self.A, other.A)


def quartet_residualize(X: np.ndarray, meta: dict[str, np.ndarray], keep_hazard: bool = True) -> np.ndarray:
    """Remove the scene mean and the action main effect from each factorial cell.

    Leaves ``(h-1/2) H_s +/- I_s/4`` (``keep_hazard``) or ``+/- I_s/4`` only. Needed
    whenever the target is the within-scene interaction contrast ``(2h-1)(2a-1)``:
    that label is an XOR of the two main effects, so any nonlinear readout can
    decode it from the main effects alone; after residualization, cells that
    share ``h`` differ only through the interaction component. Cells whose scene
    lacks a complete quartet are returned as nan rows.
    """
    X = np.asarray(X, dtype=np.float64)
    pair = np.asarray(meta["pair_id"])
    hz = np.asarray(meta["hazard"]).astype(int)
    ac = np.asarray(meta["action"]).astype(int)
    R = np.full_like(X, np.nan)
    for p in np.unique(pair):
        idx = np.flatnonzero(pair == p)
        cell = {(int(hz[i]), int(ac[i])): i for i in idx}
        if not all(k in cell for k in ((0, 0), (0, 1), (1, 0), (1, 1))):
            continue
        a00, a01, a10, a11 = (X[cell[k]] for k in ((0, 0), (0, 1), (1, 0), (1, 1)))
        mean = 0.25 * (a00 + a01 + a10 + a11)
        act = 0.5 * ((a01 - a00) + (a11 - a10))
        haz = 0.5 * ((a10 - a00) + (a11 - a01))
        for (h, a), i in cell.items():
            if (h, a) not in ((0, 0), (0, 1), (1, 0), (1, 1)):
                continue
            r = X[i] - mean - (a - 0.5) * act
            if not keep_hazard:
                r = r - (h - 0.5) * haz
            R[i] = r
    return R


def scene_center(X: np.ndarray, pair_id: np.ndarray) -> np.ndarray:
    """Subtract each scene's mean over its factorial cells (within-scene coordinates).

    All cells of a scene are always available together, so this uses no labels
    and is legal on held-out scenes. It removes between-scene nuisance (layout,
    lighting, absolute pose) and leaves the hazard / action / interaction
    structure the design cares about.
    """
    X = np.asarray(X, dtype=np.float64).copy()
    pair_id = np.asarray(pair_id)
    for p in np.unique(pair_id):
        m = pair_id == p
        X[m] -= X[m].mean(axis=0)
    return X


@dataclass
class JacobianLocal(CoordinateModel):
    """Local tangent chart from within-scene finite differences.

    For each scene the four factorial cells give difference vectors
    (H1-H0 at each action, A1-A0 at each hazard, and the interaction). The
    tangent basis is the top-k PCA of those *within-scene* differences, which
    cancels between-scene nuisance. Readout is ridge on tangent coordinates.
    """

    name: str = "jacobian_local"
    mu: np.ndarray | None = field(default=None, init=False)
    basis: np.ndarray | None = field(default=None, init=False)
    w: np.ndarray | None = field(default=None, init=False)
    b: float = field(default=0.0, init=False)
    n_differences: int = field(default=0, init=False)

    @staticmethod
    def within_scene_differences(X: np.ndarray, meta: dict[str, np.ndarray]) -> np.ndarray:
        pair = np.asarray(meta["pair_id"])
        hz = np.asarray(meta["hazard"]).astype(int)
        ac = np.asarray(meta["action"]).astype(int)
        diffs = []
        for p in np.unique(pair):
            idx = np.flatnonzero(pair == p)
            cell = {(int(hz[i]), int(ac[i])): X[i] for i in idx}
            for a in (0, 1):
                if (1, a) in cell and (0, a) in cell:
                    diffs.append(cell[(1, a)] - cell[(0, a)])
            for h in (0, 1):
                if (h, 1) in cell and (h, 0) in cell:
                    diffs.append(cell[(h, 1)] - cell[(h, 0)])
            if all(k in cell for k in ((0, 0), (0, 1), (1, 0), (1, 1))):
                diffs.append(cell[(1, 1)] - cell[(1, 0)] - cell[(0, 1)] + cell[(0, 0)])
        return np.asarray(diffs, dtype=np.float64) if diffs else np.zeros((0, X.shape[1]))

    def fit(self, X, y, meta):
        X = np.asarray(X, dtype=np.float64)
        D = self.within_scene_differences(X, meta)
        self.n_differences = int(D.shape[0])
        if D.shape[0] == 0:
            raise ValueError("JacobianLocal needs factorial cells sharing pair_id")
        # no centring: differences are already nuisance-cancelled
        _, s, vt = np.linalg.svd(D, full_matrices=False)
        self.basis = vt[: self.budget]
        self.mu = X.mean(axis=0)
        Z = (X - self.mu) @ self.basis.T
        self.w, self.b = ridge_fit(Z, y, self.lam)
        self.fitted = True
        return self

    def score(self, X):
        Z = (np.asarray(X, dtype=np.float64) - self.mu) @ self.basis.T
        return ridge_predict(Z, self.w, self.b)

    def geometry_summary(self):
        return {"rank": int(self.basis.shape[0]), "kind": "tangent", "n_differences": self.n_differences}

    def edit_direction(self, x, target, norm):
        cur = float(self.score(x[None])[0])
        sign = 1.0 if target >= cur else -1.0
        return sign * norm * unit(self.basis.T @ self.w)

    def similarity(self, other):
        return subspace_similarity(self.basis, other.basis)


@dataclass
class ManifoldKNN(CoordinateModel):
    """Local-neighbour chart: distance-weighted kNN readout with LLE-style reconstruction weights.

    ``budget`` is used as the neighbourhood size k (equal-capacity convention:
    k coordinates ~ k neighbours). Edit direction = move toward the local
    barycentre of neighbours that carry the target label, i.e. along the data
    manifold rather than along a global axis.
    """

    name: str = "manifold_knn"
    Xtr: np.ndarray | None = field(default=None, init=False)
    ytr: np.ndarray | None = field(default=None, init=False)
    scale: float = field(default=1.0, init=False)

    def _k(self) -> int:
        return int(max(2, min(self.budget, len(self.ytr) - 1)))

    def fit(self, X, y, meta):
        self.Xtr = np.asarray(X, dtype=np.float64)
        self.ytr = np.asarray(y, dtype=np.float64)
        d = self._pairwise(self.Xtr, self.Xtr)
        self.scale = float(np.median(d[d > 0])) if np.any(d > 0) else 1.0
        self.fitted = True
        return self

    @staticmethod
    def _pairwise(A, B):
        aa = (A**2).sum(1)[:, None]
        bb = (B**2).sum(1)[None, :]
        return np.sqrt(np.maximum(aa + bb - 2 * A @ B.T, 0.0))

    def _lle_weights(self, x: np.ndarray, nbrs: np.ndarray) -> np.ndarray:
        # solve min ||x - sum w_j n_j||^2 s.t. sum w = 1 (regularised Gram)
        G = (nbrs - x) @ (nbrs - x).T
        G = G + np.eye(len(nbrs)) * (1e-3 * np.trace(G) / max(len(nbrs), 1) + 1e-9)
        w = np.linalg.solve(G, np.ones(len(nbrs)))
        return w / w.sum()

    def score(self, X):
        X = np.asarray(X, dtype=np.float64)
        d = self._pairwise(X, self.Xtr)
        k = self._k()
        out = np.empty(len(X))
        for i in range(len(X)):
            order = np.argsort(d[i])
            # exclude exact self-matches (in-sample scoring)
            order = order[d[i][order] > 1e-12][:k] if np.any(d[i] > 1e-12) else order[:k]
            w = self._lle_weights(X[i], self.Xtr[order])
            out[i] = float(w @ self.ytr[order])
        return out

    def geometry_summary(self):
        return {"rank": None, "kind": "local_manifold", "k": self._k(), "scale": self.scale}

    def edit_direction(self, x, target, norm):
        d = self._pairwise(x[None], self.Xtr)[0]
        order = np.argsort(d)
        order = order[d[order] > 1e-12]
        closer = order[np.abs(self.ytr[order] - target) <= np.abs(self.ytr[order] - target).min() + 1e-9][: self._k()]
        if len(closer) == 0:
            return np.zeros_like(x)
        bary = self.Xtr[closer].mean(axis=0)
        return norm * unit(bary - x)

    def similarity(self, other):
        """Jaccard overlap of kNN graphs over the shared training rows (by exact row identity)."""
        if self.Xtr is None or other.Xtr is None:
            return float("nan")
        # match rows across the two fits
        key_self = {tuple(np.round(r, 6)): i for i, r in enumerate(self.Xtr)}
        key_other = {tuple(np.round(r, 6)): i for i, r in enumerate(other.Xtr)}
        shared = [k for k in key_self if k in key_other]
        if len(shared) < 3:
            return float("nan")
        S = np.asarray([self.Xtr[key_self[k]] for k in shared])
        k = max(2, min(self._k(), len(shared) - 1))
        dm = self._pairwise(S, S)
        np.fill_diagonal(dm, np.inf)
        nn = np.argsort(dm, axis=1)[:, :k]
        # both models see the same rows here; graph identity depends only on k and rows,
        # so compare against the other model's k
        k2 = max(2, min(other._k(), len(shared) - 1))
        nn2 = np.argsort(dm, axis=1)[:, :k2]
        jac = []
        for i in range(len(shared)):
            a, b = set(nn[i].tolist()), set(nn2[i].tolist())
            jac.append(len(a & b) / len(a | b))
        return float(np.mean(jac))


@dataclass
class CounterfactualPairDirection(CoordinateModel):
    """Reference entrant: direction = mean of matched within-scene H0->H1 deltas (both actions).

    Nothing is fitted to the label; the direction comes from real counterfactual
    pairs, so an edit along it is by construction a step other real cells
    actually take. Readout = calibrated projection (like ``linear_meandiff``).
    """

    name: str = "counterfactual_pair_direction"
    direction: np.ndarray | None = field(default=None, init=False)
    mu: np.ndarray | None = field(default=None, init=False)
    readout: tuple[float, float] = field(default=(1.0, 0.0), init=False)
    n_pairs: int = field(default=0, init=False)
    delta_norm: float = field(default=float("nan"), init=False)

    def fit(self, X, y, meta):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        pair = np.asarray(meta["pair_id"])
        hz = np.asarray(meta["hazard"]).astype(int)
        ac = np.asarray(meta["action"]).astype(int)
        deltas = []
        for p in np.unique(pair):
            idx = np.flatnonzero(pair == p)
            cell = {(int(hz[i]), int(ac[i])): X[i] for i in idx}
            for a in (0, 1):
                if (1, a) in cell and (0, a) in cell:
                    deltas.append(cell[(1, a)] - cell[(0, a)])
        if not deltas:
            raise ValueError("CounterfactualPairDirection needs matched H0/H1 cells sharing pair_id")
        D = np.asarray(deltas)
        self.n_pairs = int(len(D))
        self.delta_norm = float(np.linalg.norm(D, axis=1).mean())
        self.direction = unit(D.mean(axis=0))
        self.mu = X.mean(axis=0)
        proj = (X - self.mu) @ self.direction
        w, b = ridge_fit(proj[:, None], y, 1e-6)
        self.readout = (float(w[0]), float(b))
        self.fitted = True
        return self

    def score(self, X):
        proj = (np.asarray(X, dtype=np.float64) - self.mu) @ self.direction
        return proj * self.readout[0] + self.readout[1]

    def geometry_summary(self):
        return {"rank": 1, "kind": "counterfactual_direction", "n_pairs": self.n_pairs, "mean_pair_delta_norm": self.delta_norm}

    def edit_direction(self, x, target, norm):
        sign = 1.0 if (target - float(self.score(x[None])[0])) * self.readout[0] >= 0 else -1.0
        return sign * norm * self.direction

    def similarity(self, other):
        return float(abs(self.direction @ other.direction))


# --------------------------------------------------------------------------- #
# Soft conceptors (COAST, Miao et al. arXiv:2605.17144) in factored form
# --------------------------------------------------------------------------- #
#
# For centred rows X [N, d], R = X^T X / N and C = R (R + alpha^-2 I)^-1.  With
# R = U diag(lam) U^T this is C = U diag(mu) U^T, mu_i = lam_i / (lam_i + alpha^-2):
# a soft, variance-adaptive PCA projector.  We keep conceptors factored as
# (eigvecs [r, d], mu [r]) because rank r <= N << d; the full matrix is
# eigvecs.T @ diag(mu) @ eigvecs.  Boolean algebra (COAST Sec. 3): NOT C = I - C,
# A AND B = (A^+ + B^+ - I)^+ (pseudoinverses).  Quota q = tr(C) / d, similarity
# tr(C_A C_B) / sqrt(tr C_A^2 tr C_B^2).


def conceptor_factored(X: np.ndarray, alpha: float, center: bool = True) -> dict[str, np.ndarray]:
    """Soft conceptor of the rows of X via SVD (float64). Returns eigvecs [r,d], mu [r], lam [r], mean [d]."""
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0) if center else np.zeros(X.shape[1])
    Xc = X - mean
    if Xc.shape[0] == 0:
        return {"eigvecs": np.zeros((0, X.shape[1])), "mu": np.zeros(0), "lam": np.zeros(0), "mean": mean, "alpha": float(alpha)}
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    lam = s**2 / Xc.shape[0]
    keep = lam > 1e-12 * max(lam.max(), 1e-300)
    lam, vt = lam[keep], vt[keep]
    mu = lam / (lam + alpha ** -2)
    return {"eigvecs": vt, "mu": mu, "lam": lam, "mean": mean, "alpha": float(alpha)}


def conceptor_matrix(c: dict[str, np.ndarray]) -> np.ndarray:
    V, mu = np.asarray(c["eigvecs"], dtype=np.float64), np.asarray(c["mu"], dtype=np.float64)
    return (V.T * mu) @ V


def conceptor_apply(H: np.ndarray, c: dict[str, np.ndarray]) -> np.ndarray:
    """Rows H [n, d] -> H C (no centring; callers centre by c['mean'] if they want to preserve the mean)."""
    V, mu = np.asarray(c["eigvecs"], dtype=np.float64), np.asarray(c["mu"], dtype=np.float64)
    return ((np.asarray(H, dtype=np.float64) @ V.T) * mu) @ V


def conceptor_quota(c: dict[str, np.ndarray], d: int) -> float:
    return float(np.sum(c["mu"]) / d)


def conceptor_similarity(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> float:
    """tr(C_A C_B) / sqrt(tr C_A^2 tr C_B^2) computed in factored form."""
    Va, ma = np.asarray(a["eigvecs"]), np.asarray(a["mu"])
    Vb, mb = np.asarray(b["eigvecs"]), np.asarray(b["mu"])
    if Va.size == 0 or Vb.size == 0:
        return float("nan")
    G = Va @ Vb.T  # [ra, rb]
    num = float(np.sum((G**2) * np.outer(ma, mb)))
    den = float(np.sqrt(np.sum(ma**2) * np.sum(mb**2)))
    return num / den if den > 0 else float("nan")


def alpha_for_quota(lam: np.ndarray, d: int, target_quota: float, lo: float = 1e-3, hi: float = 1e5) -> float:
    """Aperture giving quota ~= target (bisection on log alpha); clipped to [lo, hi]."""
    lam = np.asarray(lam, dtype=np.float64)
    if lam.size == 0:
        return float("nan")

    def q(alpha):
        return float(np.sum(lam / (lam + alpha ** -2)) / d)

    if q(hi) < target_quota:
        return hi
    if q(lo) > target_quota:
        return lo
    a, b = np.log(lo), np.log(hi)
    for _ in range(60):
        m = 0.5 * (a + b)
        if q(np.exp(m)) < target_quota:
            a = m
        else:
            b = m
    return float(np.exp(0.5 * (a + b)))


def conceptor_and_not(c_int: dict[str, np.ndarray], c_sub: dict[str, np.ndarray], tol: float = 1e-8) -> dict[str, np.ndarray]:
    """C_int AND NOT C_sub = (C_int^-1 + (I - C_sub)^-1 - I)^-1, computed inside span(V_int u V_sub).

    Outside that span C_int = 0 and NOT C_sub = I, so the AND is 0 there, which
    keeps the operation O(r^3) instead of O(d^3).  Inverses are eps-regularised
    (A + tol I) rather than Moore-Penrose: a direction where C_int ~ 0 but
    NOT C_sub < 1 must give M -> infinity and C -> 0, whereas a pseudoinverse
    treats A^+ = 0 there and returns (B^+ - I)^-1, which explodes.  With
    A, B <= I the regularised M is >= I, so every eigenvalue of the result lies
    in [0, 1] even when the two conceptors do not commute.
    """
    Vi, mi = np.asarray(c_int["eigvecs"], dtype=np.float64), np.asarray(c_int["mu"], dtype=np.float64)
    Vs, ms = np.asarray(c_sub["eigvecs"], dtype=np.float64), np.asarray(c_sub["mu"], dtype=np.float64)
    d = Vi.shape[1]
    stack = np.concatenate([Vi, Vs], axis=0) if Vs.size else Vi
    if stack.shape[0] == 0:
        return {"eigvecs": np.zeros((0, d)), "mu": np.zeros(0), "lam": np.zeros(0), "mean": c_int.get("mean", np.zeros(d)), "alpha": c_int.get("alpha")}
    # exact orthonormal basis of the joint span (SVD; an unpivoted QR would mis-order dependent columns)
    _, sv, vt = np.linalg.svd(stack, full_matrices=False)
    Q = vt[sv > 1e-9 * max(sv.max(), 1e-300)].T  # [d, k]
    k = Q.shape[1]
    A = (Q.T @ Vi.T) * mi @ (Vi @ Q)  # C_int in the joint basis
    B = np.eye(k) - (Q.T @ Vs.T) * ms @ (Vs @ Q) if Vs.size else np.eye(k)  # NOT C_sub in the joint basis
    I = np.eye(k)
    M = np.linalg.solve(A + tol * I, I) + np.linalg.solve(B + tol * I, I) - I
    Cs = np.linalg.solve(0.5 * (M + M.T), I)
    Cs = 0.5 * (Cs + Cs.T)
    w, E = np.linalg.eigh(Cs)
    keep = w > 1e-10
    w, E = w[keep], E[:, keep]
    w = np.clip(w, 0.0, 1.0)
    eigvecs = (Q @ E).T  # [r, d]
    return {"eigvecs": eigvecs, "mu": w, "lam": w, "mean": c_int.get("mean", np.zeros(d)), "alpha": c_int.get("alpha")}


def matched_spectrum_random(c: dict[str, np.ndarray], seed: int = 0) -> dict[str, np.ndarray]:
    """Same mu spectrum, random orthonormal eigenvectors (COAST's matched-spectrum control)."""
    V, mu = np.asarray(c["eigvecs"]), np.asarray(c["mu"])
    r, d = V.shape
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(d, max(r, 1))))
    return {"eigvecs": Q[:, :r].T, "mu": mu.copy(), "lam": np.asarray(c.get("lam", mu)).copy(), "mean": c.get("mean"), "alpha": c.get("alpha")}


def rank_one_projector(X: np.ndarray) -> dict[str, np.ndarray]:
    """Hard projector onto the unit mean of the rows (the rank-one mean-difference control)."""
    X = np.asarray(X, dtype=np.float64)
    u = unit(X.mean(axis=0))
    return {"eigvecs": u[None, :], "mu": np.ones(1), "lam": np.ones(1), "mean": np.zeros(X.shape[1]), "alpha": float("inf")}


def within_scene_did_rows(X: np.ndarray, meta: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-scene interaction (DiD) and H-averaged action vectors. Returns (delta [S,d], action [S,d], pair_ids)."""
    pair = np.asarray(meta["pair_id"])
    hz = np.asarray(meta["hazard"]).astype(int)
    ac = np.asarray(meta["action"]).astype(int)
    D, A, ids = [], [], []
    for p in np.unique(pair):
        idx = np.flatnonzero(pair == p)
        cell = {(int(hz[i]), int(ac[i])): X[i] for i in idx}
        if all(k in cell for k in ((0, 0), (0, 1), (1, 0), (1, 1))):
            D.append(cell[(1, 1)] - cell[(1, 0)] - cell[(0, 1)] + cell[(0, 0)])
            A.append(0.5 * ((cell[(0, 1)] - cell[(0, 0)]) + (cell[(1, 1)] - cell[(1, 0)])))
            ids.append(p)
    d = X.shape[1]
    return (np.asarray(D) if D else np.zeros((0, d))), (np.asarray(A) if A else np.zeros((0, d))), np.asarray(ids)


@dataclass
class ConceptorSubspace(CoordinateModel):
    """Soft-conceptor entrant: C_safety = C_int AND NOT C_action fitted on within-scene DiD vectors.

    ``budget`` is read as the target quota in effective dimensions (aperture chosen
    so that tr(C_int)/d = budget/d); readout = ridge on h C_safety.  Rows are
    scaled to unit mean squared norm before the aperture is applied so alpha is
    comparable across sites.
    """

    name: str = "conceptor"
    target_dims: float = 4.0
    c_int: dict | None = field(default=None, init=False)
    c_action: dict | None = field(default=None, init=False)
    c_safety: dict | None = field(default=None, init=False)
    scale: float = field(default=1.0, init=False)
    w: np.ndarray | None = field(default=None, init=False)
    b: float = field(default=0.0, init=False)

    def fit(self, X, y, meta):
        X = np.asarray(X, dtype=np.float64)
        D, A, _ = within_scene_did_rows(X, meta)
        if D.shape[0] < 2:
            raise ValueError("ConceptorSubspace needs >= 2 complete quartets")
        self.scale = float(np.sqrt(np.mean(np.sum(D**2, axis=1)))) or 1.0
        d = X.shape[1]
        c0 = conceptor_factored(D / self.scale, 1.0, center=False)
        alpha = alpha_for_quota(c0["lam"], d, self.target_dims / d)
        self.c_int = conceptor_factored(D / self.scale, alpha, center=False)
        self.c_action = conceptor_factored(A / self.scale, alpha, center=False)
        self.c_safety = conceptor_and_not(self.c_int, self.c_action)
        Z = self._coords(X)
        self.w, self.b = ridge_fit(Z, np.asarray(y, dtype=np.float64), self.lam)
        self.fitted = True
        return self

    def _coords(self, X):
        return conceptor_apply(np.asarray(X, dtype=np.float64), self.c_safety)

    def score(self, X):
        return ridge_predict(self._coords(X), self.w, self.b)

    def geometry_summary(self):
        d = self.c_int["eigvecs"].shape[1]
        return {
            "rank": int(np.sum(self.c_safety["mu"] > 0.5)),
            "kind": "conceptor",
            "alpha": self.c_int["alpha"],
            "quota_int": conceptor_quota(self.c_int, d),
            "quota_action": conceptor_quota(self.c_action, d),
            "quota_safety": conceptor_quota(self.c_safety, d),
            "overlap_int_action": conceptor_similarity(self.c_int, self.c_action),
        }

    def edit_direction(self, x, target, norm):
        """Additive surrogate of the multiplicative steer: direction of x C_safety (or its readout gradient)."""
        cur = float(self.score(x[None])[0])
        g = conceptor_apply(self.w[None, :], self.c_safety)[0]  # readout gradient inside the subspace
        sign = 1.0 if target >= cur else -1.0
        return sign * norm * unit(g)

    def similarity(self, other):
        return conceptor_similarity(self.c_safety, other.c_safety)


# --------------------------------------------------------------------------- #
# Bilinear relational probe (design doc 2.4 / E4): y = z_g^T W z_e, W = U V^T of rank r
# --------------------------------------------------------------------------- #


def bilinear_fit(Zg: np.ndarray, Ze: np.ndarray, y: np.ndarray, rank: int, lam: float, n_iter: int = 8, seed: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
    """Alternating ridge for y ~ sum_j (z_g . u_j)(z_e . v_j) + b.

    For fixed V the model is linear in U with features [a_1 z_g, ..., a_r z_g], a_j = z_e . v_j (and
    symmetrically for V), so each half-step is a closed-form ridge (dual form when r*d > n).  V is
    initialised label-free from the top right singular vectors of Ze.  Returns (U [d,r], V [d,r], b).
    """
    Zg = np.asarray(Zg, dtype=np.float64)
    Ze = np.asarray(Ze, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = Zg.shape
    # label-informed initialisation on the training rows: rank-r SVD of the cross-covariance sum_i (y_i - ybar) z_g z_e^T
    yc = y - y.mean()
    W0 = (Zg * yc[:, None]).T @ Ze / max(n, 1)
    uu, ss, vvt = np.linalg.svd(W0, full_matrices=False)
    r_eff = min(rank, len(ss))
    V = vvt[:r_eff].T * np.sqrt(ss[:r_eff] + 1e-12)
    if r_eff < rank:
        V = np.concatenate([V, np.random.default_rng(seed).normal(size=(d, rank - r_eff)) * 1e-3], axis=1)
    U = np.zeros((d, rank))
    b = float(y.mean())
    for _ in range(n_iter):
        A = Ze @ V  # [n, r]
        F = np.concatenate([A[:, [j]] * Zg for j in range(rank)], axis=1)  # [n, r*d]
        w, b = ridge_fit(F, y, lam)
        U = w.reshape(rank, d).T
        Bm = Zg @ U
        F = np.concatenate([Bm[:, [j]] * Ze for j in range(rank)], axis=1)
        w, b = ridge_fit(F, y, lam)
        V = w.reshape(rank, d).T
        # keep the factorisation balanced
        for j in range(rank):
            su, sv = np.linalg.norm(U[:, j]) + 1e-12, np.linalg.norm(V[:, j]) + 1e-12
            s = np.sqrt(su * sv)
            U[:, j] *= s / su
            V[:, j] *= s / sv
    return U, V, float(b)


def bilinear_predict(Zg: np.ndarray, Ze: np.ndarray, U: np.ndarray, V: np.ndarray, b: float) -> np.ndarray:
    return np.sum((np.asarray(Zg, dtype=np.float64) @ U) * (np.asarray(Ze, dtype=np.float64) @ V), axis=1) + b


@dataclass
class BilinearRelational(CoordinateModel):
    """Rank-r bilinear readout on a concatenated feature row [z_g | z_e] (split in the middle).

    ``budget`` = rank r (1 or 2).  A linear readout on the concatenation cannot express
    "route content x egg content"; this is the minimal relational probe (arXiv:2606.09646
    shows probe capacity can dominate on physics readouts).
    """

    name: str = "bilinear"
    U: np.ndarray | None = field(default=None, init=False)
    V: np.ndarray | None = field(default=None, init=False)
    b: float = field(default=0.0, init=False)

    def _split(self, X):
        X = np.asarray(X, dtype=np.float64)
        h = X.shape[1] // 2
        return X[:, :h], X[:, h:]

    def fit(self, X, y, meta):
        Zg, Ze = self._split(X)
        self.U, self.V, self.b = bilinear_fit(Zg, Ze, y, max(1, self.budget), self.lam)
        self.fitted = True
        return self

    def score(self, X):
        Zg, Ze = self._split(X)
        return bilinear_predict(Zg, Ze, self.U, self.V, self.b)

    def geometry_summary(self):
        return {"rank": int(self.U.shape[1]), "kind": "bilinear", "u_norms": np.linalg.norm(self.U, axis=0).tolist(), "v_norms": np.linalg.norm(self.V, axis=0).tolist()}

    def edit_direction(self, x, target, norm):
        """Gradient of the bilinear readout w.r.t. the route half (holding the egg half fixed)."""
        Zg, Ze = self._split(x[None])
        g = self.U @ (Ze @ self.V)[0]
        cur = float(self.score(x[None])[0])
        sign = 1.0 if target >= cur else -1.0
        full = np.concatenate([unit(g), np.zeros_like(g)])
        return sign * norm * full

    def similarity(self, other):
        return subspace_similarity(self.U.T, other.U.T)


class TranscoderPlaceholder(CoordinateModel):
    """Phase-3 entrant. Not implemented on purpose.

    Interface contract (fill in only after the localization gate passes and the
    frozen site is an MLP output):

    - fit(X_mlp_out, y, meta): train an archetypal sparse transcoder on the
      native MLP output at the frozen site; report held-out explained variance
      (>= 0.45) and reconstruction cosine (>= 0.70) or refuse.
    - score(X): readout on sparse feature activations.
    - edit_direction(x, target, norm): returns ``W_dec[i] * dF_i`` for the
      selected feature i, scaled to ``norm``; the caller ADDS it to the
      untouched native MLP output (never replaces the output with the
      reconstruction).
    """

    name = "transcoder"

    def __init__(self, budget: int = 1, lam: float = 1.0):
        super().__init__(budget=budget, lam=lam, name="transcoder")

    def fit(self, X, y, meta):
        raise NotImplementedError("transcoder entrant is gated behind causal localization (design Phase 3)")


MODELS: dict[str, Callable[..., CoordinateModel]] = {
    "linear_meandiff": lambda budget, lam: LinearMeanDiff(budget=1, lam=lam),
    "linear_ridge": lambda budget, lam: LinearRidge(budget=1, lam=lam),
    "lowrank_subspace": lambda budget, lam: LowRankSubspace(budget=budget, lam=lam),
    "circular": lambda budget, lam: CircularFactorized(budget=budget, lam=lam),
    "jacobian_local": lambda budget, lam: JacobianLocal(budget=budget, lam=lam),
    "manifold_knn": lambda budget, lam: ManifoldKNN(budget=budget, lam=lam),
    "counterfactual_pair_direction": lambda budget, lam: CounterfactualPairDirection(budget=1, lam=lam),
    "conceptor": lambda budget, lam: ConceptorSubspace(budget=budget, lam=lam, target_dims=float(budget)),
    # bilinear needs a concatenated [z_g | z_e] feature row (tournament --token-pool pair:<g>,<e>); rank = min(budget, 2)
    "bilinear": lambda budget, lam: BilinearRelational(budget=min(int(budget), 2), lam=lam),
}


# --------------------------------------------------------------------------- #
# On-manifold scoring of edits
# --------------------------------------------------------------------------- #


def clean_distribution(X: np.ndarray) -> dict[str, np.ndarray]:
    """Diagonal-covariance summary of the site's clean activations (discovery cells)."""
    X = np.asarray(X, dtype=np.float64)
    return {"mean": X.mean(axis=0), "var": X.var(axis=0) + 1e-8}


def mahalanobis_diag(Z: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    """Per-row diagonal Mahalanobis distance normalised by sqrt(d), so 1.0 ~ a typical clean cell."""
    Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
    return np.sqrt(np.mean((Z - stats["mean"]) ** 2 / stats["var"], axis=1))


def on_manifold_scores(X_cells: np.ndarray, deltas: np.ndarray, stats: dict[str, np.ndarray]) -> dict[str, float]:
    """Mahalanobis before/after the edit; ``increase`` is what a random equal-norm direction inflates most."""
    before = mahalanobis_diag(X_cells, stats)
    after = mahalanobis_diag(X_cells + deltas, stats)
    return {
        "mahalanobis_before_mean": float(before.mean()),
        "mahalanobis_after_mean": float(after.mean()),
        "increase_mean": float((after - before).mean()),
        "on_manifold_fraction": float(np.mean(after <= np.percentile(before, 97.5))),
    }


def random_equal_norm_edits(deltas: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    R = rng.normal(size=deltas.shape)
    R /= np.linalg.norm(R, axis=1, keepdims=True) + 1e-12
    return R * np.linalg.norm(deltas, axis=1, keepdims=True)


def edit_cosine(A: np.ndarray, B: np.ndarray) -> float:
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    num = (A * B).sum(axis=1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12
    return float(np.mean(num / den))


# --------------------------------------------------------------------------- #
# Edit-vector artifact
# --------------------------------------------------------------------------- #


def write_edit_artifact(
    path: Path,
    *,
    model_name: str,
    site_id: str,
    step: int,
    token_pool: str,
    cell_ids: list[str],
    deltas: np.ndarray,
    norm: float,
    target: float,
    target_name: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write ``<path>.npz`` + ``<path>.json``.

    npz: ``cell_ids`` (str array [n]), ``delta`` float32 [n, d], ``norm`` (scalar),
    ``target`` (scalar).
    json: how to inject: ``site_id``, ``step`` (imagined step index the features
    were taken from), ``token_pool`` ("mean" => broadcast the same delta to every
    token of that group at that step; "token:<i>" => add only at token i;
    "flatten" => delta is [n_tokens*d] and must be reshaped), ``model``,
    ``target_name``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    npz_path, json_path = path.parent / (path.name + ".npz"), path.parent / (path.name + ".json")  # site ids contain dots
    np.savez_compressed(
        npz_path,
        cell_ids=np.asarray(cell_ids),
        delta=np.asarray(deltas, dtype=np.float32),
        norm=np.float32(norm),
        target=np.float32(target),
    )
    meta = {
        "model": model_name,
        "site_id": site_id,
        "step": int(step),
        "token_pool": token_pool,
        "target_name": target_name,
        "target": float(target),
        "norm": float(norm),
        "n_cells": int(len(cell_ids)),
        "inject": "add delta to the site activation at `step`; broadcast per `token_pool`; never replace",
    }
    if extra:
        meta.update(extra)
    json_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return npz_path
