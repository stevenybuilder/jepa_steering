#!/usr/bin/env python3
"""Curved / dense-manifold localization pass over a predictor activation dump.

Why: ``localize_interaction.py`` scores each site by the leave-one-scene-out
(LOSO) projection of a scene's interaction vector onto the *global* unit mean of
the other scenes.  That statistic is blind to a relational code whose direction
rotates with scene geometry (a "curved" code): if the interaction direction is
``f(bearing)`` with ``f`` covering the circle, the global mean is ~0 although
every scene carries a strong, geometry-locked interaction.  This pass asks four
questions per site x imagined step x token group, with LOSO scene folds and
scene-clustered bootstrap CIs:

1. Readout comparison on relational targets: linear ridge vs manifold kNN vs
   Jacobian-local tangent vs RBF kernel ridge at equal budgets; the nonlinear
   minus linear advantage with a CI, flagged when the CI excludes 0.
   Targets: within-scene interaction contrast ``(2h-1)(2a-1)`` signed by the
   scene's contact/force DiD; per-cell force with the same residualization;
   and the H1-vs-H0' null contrast on scenes that carry hazard-2 cells.
   The XOR contrast is trivially decodable by ANY nonlinear model from the two
   main effects alone, so readouts run on quartet-residualized features: each
   cell minus its scene mean minus the scene's action main effect ``(a-1/2) A_s``,
   which leaves ``(h-1/2) H_s +/- I_s/4``.  Cells that share ``h`` then differ only
   through the interaction, so separating them requires the interaction code.
   The fully residualized variant (``+/- I_s/4`` only) is reported as
   ``strict``.
2. Local-linear consistency: the LOSO projection recomputed against the unit
   mean of only the ``k`` scenes nearest in path-frame covariates (bearing,
   lateral offset, along-path distance, gripper pose) instead of the global
   mean.  ``local / global`` well above 1 means the direction rotates with
   geometry.
3. Direction-geometry coupling: RSA (Spearman, scene-permutation p) between
   pairwise cosine distances of scene interaction vectors and pairwise
   bearing / covariate distances; and a circular fit of the interaction
   direction as a function of bearing (LOSO predicted-direction cosine vs the
   global-mean baseline).
4. Density / curvature: TwoNN intrinsic dimension and local-PCA participation
   ratio of the pooled token activations and of the scene interaction vectors.

The linear map's ``t`` / ``p_maxt_fwer`` per site x step x group is joined in so
low-t / high-nonlinear sites stand out.

References: arXiv:2605.05115 (manifold steering: edits and readouts must
respect the local data manifold, motivating the kNN / local-tangent entrants),
arXiv:2602.07050 (circular population geometry in world models: a variable can
be encoded on a rotating plane rather than a fixed axis).

Descriptive only: no causal claim; site selection still requires patch gates.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cgs_stats import loso_projection_scores  # noqa: E402
from geometry_frames import frame_coordinates, load_cell_geometry  # noqa: E402
from geometry_models import (  # noqa: E402
    JacobianLocal,
    LinearRidge,
    ManifoldKNN,
    auroc,
    cluster_bootstrap,
    r_squared,
    ridge_fit,
    ridge_predict,
    scene_folds,
    unit,
)
from token_groups import frame_for_step, load_cell_groups, union  # noqa: E402

CELL_ORDER = ((0, 0), (0, 1), (1, 0), (1, 1))
DEFAULT_GROUPS = ("gripper_corridor", "egg")
READOUTS = ("linear_ridge", "manifold_knn", "jacobian_local", "rbf_kernel_ridge")


# --------------------------------------------------------------------------- #
# RBF kernel ridge (nonlinear entrant)
# --------------------------------------------------------------------------- #


class KernelRidgeRBF:
    """Kernel ridge with an RBF kernel; gamma = 1 / median squared distance of the training set."""

    name = "rbf_kernel_ridge"

    def __init__(self, budget: int = 4, lam: float = 1.0):
        self.budget = budget
        self.lam = lam

    def _k(self, A, B):
        aa = (A**2).sum(1)[:, None]
        bb = (B**2).sum(1)[None, :]
        return np.exp(-self.gamma * np.maximum(aa + bb - 2 * A @ B.T, 0.0))

    def fit(self, X, y, meta):
        self.Xtr = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        aa = (self.Xtr**2).sum(1)
        d2 = np.maximum(aa[:, None] + aa[None, :] - 2 * self.Xtr @ self.Xtr.T, 0.0)
        med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
        self.gamma = 1.0 / med
        self.ym = float(y.mean())
        K = self._k(self.Xtr, self.Xtr)
        self.alpha = np.linalg.solve(K + self.lam * np.eye(len(y)), y - self.ym)
        return self

    def score(self, X):
        return self._k(np.asarray(X, dtype=np.float64), self.Xtr) @ self.alpha + self.ym


def make_readout(name: str, budget: int, lam: float):
    if name == "linear_ridge":
        return LinearRidge(budget=1, lam=lam)
    if name == "manifold_knn":
        return ManifoldKNN(budget=budget, lam=lam)
    if name == "jacobian_local":
        return JacobianLocal(budget=budget, lam=lam)
    if name == "rbf_kernel_ridge":
        return KernelRidgeRBF(budget=budget, lam=lam)
    raise KeyError(name)


# --------------------------------------------------------------------------- #
# Loading the dump
# --------------------------------------------------------------------------- #


def load_index(dump: Path) -> dict[str, Any]:
    idx = json.loads((dump / "activations" / "index.json").read_text())
    if isinstance(idx, list):
        idx = {"cells": idx, "n_steps": None, "sites": {}, "group_frame_offset": 0}
    return idx


def cell_artifact(stimulus: Path, cell: dict[str, Any], manifest_row: dict[str, Any] | None) -> Path:
    p = Path(cell.get("artifact", ""))
    if p.is_absolute() and p.exists():
        return p
    if manifest_row is not None:
        q = stimulus / manifest_row["artifact"]
        if q.exists():
            return q
    return stimulus / cell.get("artifact", "")


def group_token_sets(stimulus: Path, cells: list[dict[str, Any]], groups: tuple[str, ...], n_steps: int, offset: int) -> dict[tuple[str, int, str], np.ndarray]:
    """Per (pair_id, step, group): union over the quartet of that group's spatial token ids (as the dump did)."""
    per_cell = {c["cell_id"]: load_cell_groups(stimulus, {"cell_id": c["cell_id"], "target_bbox_xyxy": c.get("target_bbox_xyxy")}) for c in cells}
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for c in cells:
        by_pair.setdefault(c["pair_id"], []).append(c)
    out = {}
    for pid, members in by_pair.items():
        for step in range(n_steps):
            frame = frame_for_step(step, offset)
            for g in groups:
                if g == "all":
                    out[(pid, step, g)] = np.arange(256)
                else:
                    out[(pid, step, g)] = union(*(per_cell[m["cell_id"]].group(frame, g) for m in members))
    return out


def pool_site(acts: np.ndarray, token_index: np.ndarray, cells: list[dict[str, Any]], token_sets, step: int, group: str) -> tuple[np.ndarray, np.ndarray]:
    """Mean over the cell's tokens that belong to the (pair, step, group) set. Returns (pooled [n, d], tokens [list])."""
    n, _, _, d = acts.shape
    pooled = np.full((n, d), np.nan)
    tok_rows = []
    for c in cells:
        r = c["row"]
        ti = token_index[r, step]
        want = token_sets[(c["pair_id"], step, group)]
        mask = np.isin(ti, want) & (ti >= 0)
        if acts.shape[2] == 1:  # AdaLN-style single vector
            mask = np.array([True])
        if mask.any():
            pooled[r] = acts[r, step, mask].astype(np.float32).mean(axis=0)
            tok_rows.append(acts[r, step, mask].astype(np.float32))
    return pooled, tok_rows


# --------------------------------------------------------------------------- #
# Scene structure: quartet effects, residualization, targets
# --------------------------------------------------------------------------- #


def quartets(cells: list[dict[str, Any]]) -> dict[str, dict[tuple[int, int], int]]:
    q: dict[str, dict[tuple[int, int], int]] = {}
    for c in cells:
        q.setdefault(c["pair_id"], {})[(int(c["hazard"]), int(c["candidate_action"]))] = int(c["row"])
    return q


def scene_effects(P: np.ndarray, q: dict[tuple[int, int], int]) -> dict[str, np.ndarray]:
    a00, a01, a10, a11 = (P[q[k]] for k in CELL_ORDER)
    return {
        "hazard": 0.5 * ((a10 - a00) + (a11 - a01)),
        "action": 0.5 * ((a01 - a00) + (a11 - a10)),
        "interaction": a11 - a10 - a01 + a00,
        "mean": 0.25 * (a00 + a01 + a10 + a11),
    }


def residualize(P: np.ndarray, Q: dict[str, dict[tuple[int, int], int]], mode: str) -> np.ndarray:
    """mode 'keep_hazard': x - mean - (a-1/2) A_s  ->  (h-1/2) H_s +/- I_s/4 ; 'strict': +/- I_s/4 only."""
    R = np.full_like(P, np.nan)
    for pid, q in Q.items():
        if not all(k in q for k in CELL_ORDER):
            continue
        e = scene_effects(P, q)
        for (h, a), r in q.items():
            if (h, a) not in CELL_ORDER:
                continue
            x = P[r] - e["mean"] - (a - 0.5) * e["action"]
            if mode == "strict":
                x = x - (h - 0.5) * e["hazard"]
            R[r] = x
    return R


def residualize_null(P: np.ndarray, Q: dict[str, dict[tuple[int, int], int]]) -> tuple[np.ndarray, np.ndarray]:
    """Null contrast set {h1a0, h1a1, h2a0, h2a1}: remove the set mean and action effect; label h==1."""
    R = np.full_like(P, np.nan)
    y = np.full(P.shape[0], np.nan)
    for pid, q in Q.items():
        keys = [(1, 0), (1, 1), (2, 0), (2, 1)]
        if not all(k in q for k in keys):
            continue
        b10, b11, b20, b21 = (P[q[k]] for k in keys)
        mean = 0.25 * (b10 + b11 + b20 + b21)
        act = 0.5 * ((b11 - b10) + (b21 - b20))
        for (h, a) in keys:
            r = q[(h, a)]
            R[r] = P[r] - mean - (a - 0.5) * act
            y[r] = float(h == 1)
    return R, y


def interaction_labels(cells: list[dict[str, Any]], manifest: dict[str, dict[str, Any]], Q) -> dict[str, np.ndarray]:
    n = max(c["row"] for c in cells) + 1
    contact = np.full(n, np.nan)
    force = np.full(n, np.nan)
    for c in cells:
        m = manifest.get(c["cell_id"], {})
        contact[c["row"]] = float(m.get("egg_robot_contact_count", np.nan))
        force[c["row"]] = float(m.get("max_normal_force_n", np.nan))
    y_int = np.full(n, np.nan)
    force_did = {}
    for pid, q in Q.items():
        if not all(k in q for k in CELL_ORDER):
            continue

        def did(v):
            return v[q[(1, 1)]] - v[q[(1, 0)]] - v[q[(0, 1)]] + v[q[(0, 0)]]

        s = did(contact)
        if not np.isfinite(s) or abs(s) < 1e-12:
            s = did(force)
        force_did[pid] = float(did(force))
        if not np.isfinite(s) or abs(s) < 1e-12:
            continue
        for (h, a), r in q.items():
            if (h, a) in CELL_ORDER:
                y_int[r] = float(np.sign(s) * (2 * h - 1) * (2 * a - 1) > 0)
    return {"interaction": y_int, "force": force, "force_did_by_scene": force_did}


# --------------------------------------------------------------------------- #
# Scene covariates from the path frame
# --------------------------------------------------------------------------- #


def scene_covariates(stimulus: Path, cells: list[dict[str, Any]], manifest: dict[str, dict[str, Any]], Q) -> dict[str, Any]:
    """Per scene: bearing of the off-path (H0) egg, lateral offsets, along-path distance, gripper pose."""
    rows = {c["cell_id"]: c for c in cells}
    cov, bearing, ids = [], [], []
    for pid in sorted(Q):
        q = Q[pid]
        if (0, 0) not in q:
            continue
        c0 = next(c for c in cells if c["row"] == q[(0, 0)])
        # load_cell_geometry joins stimulus / artifact; an absolute artifact path (as the dump index
        # stores it) survives the join, and masks are always read from stimulus/masks/<cell_id>.npz
        row = {"cell_id": c0["cell_id"], "artifact": str(cell_artifact(stimulus, c0, manifest.get(c0["cell_id"])))}
        g = load_cell_geometry(stimulus, row)
        frames, extras, _ = frame_coordinates(g)
        yaw = float(g.eef_euler[2])
        cov.append(np.concatenate([
            [np.cos(extras["bearing"]), np.sin(extras["bearing"])],
            frames["path"][:3],
            g.eef_pos, g.eef_euler[:2], [np.cos(yaw), np.sin(yaw)],
        ]))
        bearing.append(extras["bearing"])
        ids.append(pid)
    C = np.asarray(cov)
    Cz = (C - C.mean(0)) / (C.std(0) + 1e-9)
    return {"pair_ids": ids, "covariates": Cz, "raw": C, "bearing": np.asarray(bearing),
            "columns": ["cos_bearing", "sin_bearing", "along", "lat_u1", "lat_u2", "eef_x", "eef_y", "eef_z", "eef_roll", "eef_pitch", "cos_yaw", "sin_yaw"]}


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #


def readout_comparison(R: np.ndarray, y: np.ndarray, meta: dict[str, np.ndarray], kind: str, budget: int, lam: float, n_boot: int, seed: int) -> dict[str, Any]:
    ok = np.isfinite(y) & np.all(np.isfinite(R), axis=1)
    X, yy = R[ok], y[ok]
    m = {k: v[ok] for k, v in meta.items()}
    pairs = m["pair_id"]
    if len(np.unique(pairs)) < 3 or len(np.unique(yy)) < 2:
        return {"status": "insufficient", "n_cells": int(ok.sum())}
    folds = scene_folds(pairs, len(np.unique(pairs)), seed)  # LOSO
    oof = {}
    for name in READOUTS:
        s = np.full(len(yy), np.nan)
        for held in folds:
            train = np.setdiff1d(np.arange(len(yy)), held)
            try:
                mdl = make_readout(name, budget, lam).fit(X[train], yy[train], {k: v[train] for k, v in m.items()})
                s[held] = mdl.score(X[held])
            except Exception:
                pass
        oof[name] = s

    def metric(scores, idx):
        sc, yt = scores[idx], yy[idx]
        f = np.isfinite(sc)
        return auroc(sc[f], yt[f]) if kind == "binary" else r_squared(sc[f], yt[f])

    out: dict[str, Any] = {"status": "ok", "n_cells": int(ok.sum()), "n_scenes": int(len(np.unique(pairs))), "readouts": {}, "advantage": {}}
    for name in READOUTS:
        out["readouts"][name] = cluster_bootstrap(lambda idx, s=oof[name]: metric(s, idx), pairs, n_boot=n_boot, seed=seed)
    lin = oof["linear_ridge"]
    for name in READOUTS[1:]:
        boot = cluster_bootstrap(lambda idx, s=oof[name]: metric(s, idx) - metric(lin, idx), pairs, n_boot=n_boot, seed=seed)
        boot["ci_excludes_zero"] = bool(np.isfinite(boot["ci_low"]) and (boot["ci_low"] > 0 or boot["ci_high"] < 0))
        boot["nonlinear_wins"] = bool(np.isfinite(boot["ci_low"]) and boot["ci_low"] > 0)
        out["advantage"][name] = boot
    best = max(READOUTS[1:], key=lambda n: out["advantage"][n]["point"] if np.isfinite(out["advantage"][n]["point"]) else -np.inf)
    out["best_nonlinear"] = best
    out["nonlinear_advantage"] = out["advantage"][best]["point"]
    out["nonlinear_wins"] = out["advantage"][best]["nonlinear_wins"]
    return out


def local_vs_global_projection(I: np.ndarray, cov: np.ndarray, k: int, n_boot: int, seed: int) -> dict[str, Any]:
    """LOSO projection against the global unit mean vs the unit mean of the k covariate-nearest scenes."""
    n = I.shape[0]
    glob = loso_projection_scores(I)
    d2 = ((cov[:, None, :] - cov[None, :, :]) ** 2).sum(-1)
    loc = np.full(n, np.nan)
    for i in range(n):
        order = np.argsort(d2[i])
        nb = [j for j in order if j != i][: min(k, n - 1)]
        u = unit(I[nb].mean(axis=0))
        loc[i] = float(I[i] @ u)

    def tstat(v):
        v = v[np.isfinite(v)]
        return float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))) if len(v) > 1 and v.std(ddof=1) > 1e-12 else float("nan")

    def ratio(idx):
        g, l = glob[idx].mean(), loc[idx].mean()
        return float(l / g) if abs(g) > 1e-12 else float("nan")

    clusters = np.arange(n)
    return {
        "k": int(min(k, n - 1)),
        "global": {"mean": float(np.nanmean(glob)), "t": tstat(glob), "per_scene": glob.tolist()},
        "local": {"mean": float(np.nanmean(loc)), "t": tstat(loc), "per_scene": loc.tolist()},
        "local_minus_global": cluster_bootstrap(lambda idx: float(np.nanmean(loc[idx] - glob[idx])), clusters, n_boot=n_boot, seed=seed),
        "ratio_local_over_global": cluster_bootstrap(ratio, clusters, n_boot=n_boot, seed=seed),
        "norm_mean": float(np.linalg.norm(I, axis=1).mean()),
    }


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    def ranks(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v))
        r[order] = np.arange(len(v))
        return r

    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def rsa_direction_geometry(I: np.ndarray, bearing: np.ndarray, cov: np.ndarray, n_perm: int, seed: int) -> dict[str, Any]:
    n = I.shape[0]
    U = I / (np.linalg.norm(I, axis=1, keepdims=True) + 1e-12)
    cosd = 1.0 - U @ U.T
    bd = np.abs(np.angle(np.exp(1j * (bearing[:, None] - bearing[None, :]))))
    cd = np.sqrt(((cov[:, None, :] - cov[None, :, :]) ** 2).sum(-1))
    iu = np.triu_indices(n, 1)
    rng = np.random.default_rng(seed)

    def test(G):
        obs = spearman(cosd[iu], G[iu])
        null = []
        for _ in range(n_perm):
            p = rng.permutation(n)
            null.append(spearman(cosd[iu], G[np.ix_(p, p)][iu]))
        null = np.asarray(null)
        pval = float((np.sum(null >= obs) + 1) / (len(null) + 1)) if np.isfinite(obs) else float("nan")
        return {"spearman": obs, "p_perm": pval, "n_perm": n_perm}

    # circular fit: direction ~ [cos b, sin b]; LOSO predicted direction cosine vs global-mean baseline
    F = np.stack([np.cos(bearing), np.sin(bearing), np.ones(n)], axis=1)
    cos_circ, cos_glob = [], []
    for i in range(n):
        tr = np.setdiff1d(np.arange(n), [i])
        A = np.linalg.lstsq(F[tr], U[tr], rcond=None)[0]
        pred = F[i] @ A
        cos_circ.append(float(U[i] @ unit(pred)))
        cos_glob.append(float(U[i] @ unit(U[tr].mean(0))))
    return {
        "rsa_bearing": test(bd),
        "rsa_covariates": test(cd),
        "circular_fit": {
            "loso_cosine_circular_mean": float(np.mean(cos_circ)),
            "loso_cosine_global_mean": float(np.mean(cos_glob)),
            "advantage": float(np.mean(cos_circ) - np.mean(cos_glob)),
            "per_scene_circular": cos_circ,
            "per_scene_global": cos_glob,
        },
    }


def dedupe_near_duplicates(X: np.ndarray, rel_tol: float = 0.01) -> tuple[np.ndarray, float]:
    """Drop points that sit within ``rel_tol`` x the median pairwise distance of an earlier kept point.

    The four cells of a scene share every spatial token except those touched by
    the egg, so pooled token sets contain many near-identical rows; TwoNN and
    local PCA are meaningless on those (ratios collapse to 1, PR to ~1).
    Returns the kept rows and the fraction removed.
    """
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    if n < 3:
        return X, 0.0
    aa = (X**2).sum(1)
    d2 = np.maximum(aa[:, None] + aa[None, :] - 2 * X @ X.T, 0.0)
    np.fill_diagonal(d2, np.inf)
    d = np.sqrt(d2)
    med = float(np.median(d[np.triu_indices(n, 1)]))  # global scale, robust to the duplicates themselves
    keep = np.ones(n, dtype=bool)
    thr = rel_tol * med
    # greedy: walk rows, drop a row if it is within thr of an earlier kept row
    for i in range(n):
        if not keep[i]:
            continue
        close = np.flatnonzero(d[i] < thr)
        keep[close[close > i]] = False
    return X[keep], float(1.0 - keep.mean())


def twonn_dimension(X: np.ndarray) -> float:
    """Facco et al. 2017 TwoNN estimator (discarding the top 10% ratios). Callers should dedupe first."""
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    if n < 5:
        return float("nan")
    aa = (X**2).sum(1)
    d2 = np.maximum(aa[:, None] + aa[None, :] - 2 * X @ X.T, 0.0)
    np.fill_diagonal(d2, np.inf)
    s = np.sort(np.sqrt(d2), axis=1)[:, :2]
    ok = s[:, 0] > 1e-12
    mu = s[ok, 1] / s[ok, 0]
    mu = np.sort(mu)
    N = len(mu)
    F = np.arange(1, N + 1) / N
    keep = max(3, int(np.floor(0.9 * N)))  # discard the top 10% ratios (Facco et al.)
    x = np.log(mu[:keep])
    y = -np.log(1.0 - F[:keep])
    return float((x @ y) / (x @ x)) if x @ x > 0 else float("nan")


def local_pca_participation_ratio(X: np.ndarray, k: int = 10) -> float:
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    if n < 4:
        return float("nan")
    k = min(k, n - 1)
    aa = (X**2).sum(1)
    d2 = np.maximum(aa[:, None] + aa[None, :] - 2 * X @ X.T, 0.0)
    np.fill_diagonal(d2, np.inf)
    prs = []
    for i in range(n):
        nb = np.argsort(d2[i])[:k]
        Z = X[nb] - X[nb].mean(0)
        lam = np.linalg.svd(Z, compute_uv=False) ** 2
        if lam.sum() > 0:
            prs.append(lam.sum() ** 2 / (lam**2).sum())
    return float(np.median(prs)) if prs else float("nan")


# --------------------------------------------------------------------------- #
# Linear-map join
# --------------------------------------------------------------------------- #


def collect_linear_entries(obj: Any, acc: dict[tuple[str, int, str], dict[str, Any]]) -> None:
    if isinstance(obj, dict):
        if {"site_id", "group", "step"} <= set(obj) and ("t" in obj or "cv_projection" in obj):
            t = obj.get("t")
            p = obj.get("p_maxt_fwer")
            if t is None and isinstance(obj.get("cv_projection"), dict):
                inter = obj["cv_projection"].get("interaction", {})
                t, p = inter.get("t"), inter.get("p_maxt_fwer")
            key = (str(obj["site_id"]), int(obj["step"]), str(obj["group"]))
            acc.setdefault(key, {"t": t, "p_maxt_fwer": p, "labels": obj.get("labels"), "magnitude_ok": obj.get("magnitude_ok")})
        for v in obj.values():
            collect_linear_entries(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            collect_linear_entries(v, acc)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def analyze_site(site: str, acts, token_index, cells, token_sets, Q, labels, meta, cov, args) -> list[dict[str, Any]]:
    results = []
    for step in args.steps:
        for group in args.groups:
            P, tok_rows = pool_site(acts, token_index, cells, token_sets, step, group)
            entry: dict[str, Any] = {"site_id": site, "step": step, "group": group}
            valid_rows = np.flatnonzero(np.all(np.isfinite(P), axis=1))
            if len(valid_rows) < 8:
                entry["status"] = "no_tokens"
                results.append(entry)
                continue
            # scene interaction vectors (same scene order as cov["pair_ids"])
            I, scene_ids = [], []
            for pid in cov["pair_ids"]:
                q = Q[pid]
                if all(k in q for k in CELL_ORDER) and all(np.all(np.isfinite(P[q[k]])) for k in CELL_ORDER):
                    I.append(scene_effects(P, q)["interaction"])
                    scene_ids.append(pid)
            I = np.asarray(I)
            sel = np.asarray([cov["pair_ids"].index(p) for p in scene_ids])
            if len(I) >= 4:
                entry["local_vs_global"] = local_vs_global_projection(I, cov["covariates"][sel], args.k_local, args.n_boot, args.seed)
                entry["direction_geometry"] = rsa_direction_geometry(I, cov["bearing"][sel], cov["covariates"][sel], args.n_perm, args.seed)
                entry["density"] = {
                    "interaction_vectors": {"n": int(len(I)), "twonn": twonn_dimension(I), "local_pca_pr": local_pca_participation_ratio(I, k=min(5, len(I) - 1)),
                                            "note": "n = number of scenes; TwoNN needs n >> d and is unreliable here, reported for completeness"},
                }
            # Density on the tokens of ONE cell per scene (the h1a1 cell): the other three cells of a
            # scene are near-copies token-for-token (same frame, egg moved or action changed), which makes
            # every token's nearest neighbour its own sibling and collapses TwoNN toward 0.
            one_per_scene = [c for c in cells if (int(c["hazard"]), int(c["candidate_action"])) == (1, 1)]
            _, rows_one = pool_site(acts, token_index, one_per_scene, token_sets, step, group)
            toks = np.concatenate(rows_one) if rows_one else np.zeros((0, P.shape[1]))
            if len(toks) > 20:
                sub = toks if len(toks) <= args.max_density_points else toks[np.random.default_rng(args.seed).choice(len(toks), args.max_density_points, replace=False)]
                sub, dropped = dedupe_near_duplicates(sub)
                entry.setdefault("density", {})["pooled_tokens"] = {
                    "n": int(len(toks)), "n_scenes": int(len(one_per_scene)), "cells": "h1a1 only",
                    "n_after_dedupe": int(len(sub)), "near_duplicate_fraction": dropped,
                    "twonn": twonn_dimension(sub) if len(sub) >= 20 else float("nan"),
                    "local_pca_pr": local_pca_participation_ratio(sub, k=10) if len(sub) >= 12 else float("nan"),
                }
            if args.density_only:
                entry["status"] = "density_only"
                results.append(entry)
                continue
            entry["readouts"] = {}
            for mode in ("keep_hazard", "strict"):
                R = residualize(P, Q, mode)
                entry["readouts"][mode] = {
                    "interaction": readout_comparison(R, labels["interaction"], meta, "binary", args.budget, args.lam, args.n_boot, args.seed),
                    "force": readout_comparison(R, labels["force"], meta, "continuous", args.budget, args.lam, args.n_boot, args.seed),
                }
            Rn, yn = residualize_null(P, Q)
            if np.isfinite(yn).sum() >= 8:
                entry["readouts"]["null_contrast"] = readout_comparison(Rn, yn, meta, "binary", args.budget, args.lam, args.n_boot, args.seed)
            else:
                entry["readouts"]["null_contrast"] = {"status": "hazard2_cells_not_in_dump", "n_cells": int(np.isfinite(yn).sum())}
            entry["status"] = "ok"
            results.append(entry)
    return results


def curved_code(lvg: dict[str, Any]) -> bool:
    """Direction rotates with geometry: the covariate-local projection beats the global one (CI excludes 0),
    is itself positive (t > 2), and the global projection is weak (t < 2) or much smaller (ratio > 1.5).
    A perfectly rotating code makes the global LOSO projection negative (the other scenes sum to -I_i),
    so the ratio alone is not a usable criterion."""
    if not lvg:
        return False
    lt = lvg.get("local", {}).get("t")
    gt = lvg.get("global", {}).get("t")
    dlo = lvg.get("local_minus_global", {}).get("ci_low")
    ratio = lvg.get("ratio_local_over_global", {}).get("point")
    if lt is None or dlo is None or not np.isfinite(lt) or not np.isfinite(dlo):
        return False
    weak_global = (gt is None) or (not np.isfinite(gt)) or gt < 2.0 or (ratio is not None and np.isfinite(ratio) and ratio > 1.5)
    return bool(lt > 2.0 and dlo > 0.0 and weak_global)


def rank_table(entries: list[dict[str, Any]], linear: dict, steps=(0, 1), groups=("gripper_corridor", "egg"), top: int = 10) -> dict[str, Any]:
    tables = {}
    for step in steps:
        for group in groups:
            rows = []
            for e in entries:
                if e.get("status") != "ok" or e["step"] != step or e["group"] != group:
                    continue
                rc = e["readouts"]["keep_hazard"]["interaction"]
                if rc.get("status") != "ok":
                    continue
                lin = linear.get((e["site_id"], step, group), {})
                lvg = e.get("local_vs_global", {})
                dg = e.get("direction_geometry", {})
                rows.append({
                    "site_id": e["site_id"],
                    "nonlinear_advantage_auroc": rc["nonlinear_advantage"],
                    "best_nonlinear": rc["best_nonlinear"],
                    "nonlinear_wins_ci": rc["nonlinear_wins"],
                    "linear_auroc": rc["readouts"]["linear_ridge"]["point"],
                    "best_nonlinear_auroc": rc["readouts"][rc["best_nonlinear"]]["point"],
                    "strict_nonlinear_advantage": e["readouts"]["strict"]["interaction"].get("nonlinear_advantage"),
                    "local_over_global": lvg.get("ratio_local_over_global", {}).get("point"),
                    "local_t": lvg.get("local", {}).get("t"),
                    "global_t_recomputed": lvg.get("global", {}).get("t"),
                    "rsa_bearing_rho": dg.get("rsa_bearing", {}).get("spearman"),
                    "rsa_bearing_p": dg.get("rsa_bearing", {}).get("p_perm"),
                    "circular_fit_advantage": dg.get("circular_fit", {}).get("advantage"),
                    "linear_map_t": lin.get("t"),
                    "linear_map_p_maxt": lin.get("p_maxt_fwer"),
                    "local_minus_global_ci_low": lvg.get("local_minus_global", {}).get("ci_low"),
                    "curved_code_flag": bool(
                        curved_code(lvg) and (dg.get("rsa_bearing", {}).get("p_perm") or 1.0) < 0.05
                    ),
                })
            rows.sort(key=lambda r: -(r["nonlinear_advantage_auroc"] if np.isfinite(r["nonlinear_advantage_auroc"]) else -np.inf))
            tables[f"step{step}|{group}"] = rows[:top]
            tables[f"step{step}|{group}|curved_flags"] = [r for r in rows if r["curved_code_flag"]]
    return tables


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--stimulus", type=Path, required=True)
    ap.add_argument("--linear-map", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sites", nargs="*", default=None)
    ap.add_argument("--steps", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--groups", nargs="*", default=list(DEFAULT_GROUPS))
    ap.add_argument("--budget", type=int, default=4)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--k-local", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--n-perm", type=int, default=999)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-density-points", type=int, default=800)
    ap.add_argument("--report-site", default="L07.attn_out")
    ap.add_argument("--density-only", action="store_true", help="recompute only the density block and merge it into an existing nonlinear_map.json")
    args = ap.parse_args(argv)

    t0 = time.time()
    idx = load_index(args.dump)
    cells = idx["cells"]
    n_steps = int(idx.get("n_steps") or (max(args.steps) + 1))
    offset = int(idx.get("group_frame_offset", 0))
    manifest = {}
    mp = args.stimulus / "manifest.jsonl"
    if mp.exists():
        for line in mp.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                manifest[r["cell_id"]] = r
    for c in cells:
        c.setdefault("target_bbox_xyxy", manifest.get(c["cell_id"], {}).get("target_bbox_xyxy"))
    Q = quartets(cells)
    labels = interaction_labels(cells, manifest, Q)
    n = max(c["row"] for c in cells) + 1
    meta = {
        "pair_id": np.asarray([""] * n, dtype=object),
        "hazard": np.zeros(n, dtype=int),
        "action": np.zeros(n, dtype=int),
    }
    for c in cells:
        meta["pair_id"][c["row"]] = c["pair_id"]
        meta["hazard"][c["row"]] = int(c["hazard"])
        meta["action"][c["row"]] = int(c["candidate_action"])
    meta["pair_id"] = meta["pair_id"].astype(str)
    cov = scene_covariates(args.stimulus, cells, manifest, Q)
    token_sets = group_token_sets(args.stimulus, cells, tuple(args.groups), n_steps, offset)

    linear: dict[tuple[str, int, str], dict[str, Any]] = {}
    if args.linear_map and args.linear_map.exists():
        collect_linear_entries(json.loads(args.linear_map.read_text()), linear)

    sites = args.sites or sorted(p.stem for p in (args.dump / "activations").glob("*.npz"))
    entries: list[dict[str, Any]] = []
    for si, site in enumerate(sites):
        with np.load(args.dump / "activations" / f"{site}.npz") as z:
            acts = np.asarray(z["acts"])
            token_index = np.asarray(z["token_index"]) if "token_index" in z else np.broadcast_to(np.arange(acts.shape[2]), acts.shape[:3])
        entries.extend(analyze_site(site, acts, token_index, cells, token_sets, Q, labels, meta, cov, args))
        print(f"[{si + 1}/{len(sites)}] {site} done ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)

    if args.density_only:
        path = args.out / "nonlinear_map.json"
        if path.exists():
            existing = json.loads(path.read_text())
            by_key = {(e["site_id"], e["step"], e["group"]): e for e in entries}
            n_merged = 0
            for e in existing["entries"]:
                d = by_key.get((e["site_id"], e["step"], e["group"]))
                if d is not None and "density" in d:
                    e["density"] = d["density"]
                    n_merged += 1
            existing["density_recomputed"] = {"cells": "h1a1 only", "n_merged": n_merged}
            path.write_text(json.dumps(existing, indent=1, default=_json_default) + "\n")
            print(f"merged density into {path} ({n_merged} entries)")
            return existing
        (args.out / "density_only.json").write_text(json.dumps(entries, indent=1, default=_json_default) + "\n")
        return {"entries": entries}
    tables = rank_table(entries, linear, steps=tuple(s for s in (0, 1) if s in args.steps), groups=tuple(g for g in ("gripper_corridor", "egg") if g in args.groups))
    # sites the linear map ranked low but that show a curved code or a nonlinear win
    curved_low_linear = []
    for e in entries:
        if e.get("status") != "ok":
            continue
        lin = linear.get((e["site_id"], e["step"], e["group"]), {})
        rc = e["readouts"]["keep_hazard"]["interaction"]
        lvg = e.get("local_vs_global", {})
        curved = curved_code(lvg)
        if (curved or rc.get("nonlinear_wins")) and (lin.get("t") is None or not np.isfinite(lin.get("t") or np.nan) or lin.get("t") < 2.0):
            curved_low_linear.append({"site_id": e["site_id"], "step": e["step"], "group": e["group"], "linear_map_t": lin.get("t"),
                                      "local_over_global": lvg.get("ratio_local_over_global", {}).get("point"), "local_t": lvg.get("local", {}).get("t"),
                                      "nonlinear_advantage": rc.get("nonlinear_advantage"), "nonlinear_wins": rc.get("nonlinear_wins")})
    focus = [e for e in entries if e["site_id"] == args.report_site and e["step"] == 0]
    report = {
        "protocol": "cgs-geometry-localize-v0.1",
        "dump": str(args.dump), "stimulus": str(args.stimulus), "linear_map": str(args.linear_map),
        "n_cells": int(n), "n_scenes": len(Q), "scenes_with_hazard2": int(sum(all(k in q for k in ((2, 0), (2, 1))) for q in Q.values())),
        "steps": args.steps, "groups": args.groups, "budget": args.budget, "lam": args.lam, "k_local": args.k_local,
        "residualization": "keep_hazard: x - scene_mean - (a-1/2)*A_s = (h-1/2)*H_s +/- I_s/4 ; strict: +/- I_s/4",
        "covariate_columns": cov["columns"], "scene_order": cov["pair_ids"], "scene_bearing": cov["bearing"].tolist(),
        "group_token_counts": {g: {"min": int(min(len(v) for (pid, s, gg), v in token_sets.items() if gg == g)),
                                   "max": int(max(len(v) for (pid, s, gg), v in token_sets.items() if gg == g))} for g in args.groups},
        "force_did_by_scene": labels["force_did_by_scene"],
        "ranked_tables": tables,
        "curved_or_nonlinear_but_low_linear_t": curved_low_linear,
        "focus_site_step0": focus,
        "entries": entries,
        "runtime_s": time.time() - t0,
        "interpretation_scope": "Descriptive geometry of the interaction code; no causal claim. Nonlinear readouts run on quartet-residualized features so the XOR contrast cannot be decoded from main effects alone.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "nonlinear_map.json").write_text(json.dumps(report, indent=1, default=_json_default) + "\n")
    for key, rows in tables.items():
        if key.endswith("curved_flags"):
            continue
        print(f"== {key} ==")
        for r in rows:
            print(f"{r['site_id']:>14s} adv={r['nonlinear_advantage_auroc']:+.3f} ({r['best_nonlinear']}, win={r['nonlinear_wins_ci']}) lin={r['linear_auroc']:.3f} "
                  f"loc/glob={r['local_over_global'] if r['local_over_global'] is None else round(r['local_over_global'], 2)} local_t={r['local_t'] if r['local_t'] is None else round(r['local_t'], 2)} "
                  f"rsa_b={r['rsa_bearing_rho'] if r['rsa_bearing_rho'] is None else round(r['rsa_bearing_rho'], 2)}(p={r['rsa_bearing_p']}) map_t={r['linear_map_t']} p={r['linear_map_p_maxt']} curved={r['curved_code_flag']}")
    return report


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


if __name__ == "__main__":
    main()
