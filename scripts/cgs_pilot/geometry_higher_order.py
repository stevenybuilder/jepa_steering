#!/usr/bin/env python3
"""Higher-order distributional sonar + geometry-that-moves arm over a predictor activation dump.

Implements "Step 4 - Geometry that moves through the network" and "Step 5 -
Higher-order distributional sonar" of the EBM representational-geometry plan
(mechinterp-vla/EBM representational geometry.md), per site x imagined step x
token group, on nuisance-residualized pooled and per-token rows:

1. Cramer-Wold sliced-W2 arm.  Projection set chosen without held-out outcomes:
   covariance eigenvectors of the pooled rows, 256 random unit vectors (frozen
   seed), and leave-one-scene-out discriminant directions.  For each projection
   c the one-dimensional samples of the action-effect distributions
   ``Delta_A H | H1`` vs ``Delta_A H | H0`` (and ``| H0'`` when hazard-2 cells
   exist) are compared by sliced 2-Wasserstein, BEFORE and AFTER matching the
   first two moments of each projected sample (train-fold mean / sd).  Frozen
   top-k mean and median aggregates; scene-level sign-flip (H1<->H0 within
   scene) permutation p, and a max-stat p over the sites sharing the flips.
   Separation that survives moment matching is geometry beyond the conceptor
   ellipsoid (COAST, arXiv:2605.17144, is second-order only; Rectified LpJEPA,
   arXiv:2602.01456, motivates the Cramer-Wold sliced comparison).
2. nHSIC between residualized rows and relational labels (interaction contrast,
   force, hazard-under-path distance) with a median-heuristic bandwidth fixed on
   the train fold and scene-permutation p; linear CKA alongside.
3. Rotating-subspace tracking: rank-k (k=4) bases U_{s,t} of the interaction
   rows; principal angles / Grassmann distance across adjacent layers and across
   imagined steps; orthogonal Procrustes Q* fit on discovery scenes; LOSO test of
   whether the interaction readout transfers after transport but not raw.
4. Two-sided (token x hidden) operator check on the gripper+egg grid: token-only,
   hidden-only and two-sided (Tucker-2 / HOSVD) low-rank fits of the interaction
   tensor by held-out reconstruction; token participation ratio as spatial
   spread.
5. Descriptives: spectral entropy, participation ratio, coordinate concentration
   ||z||_1^2 / ||z||_2^2 of the interaction rows.

Joined with the linear map (localize_v2) t / p_maxT and the conceptor readouts;
flags sites where sliced-W2-after-matching or kernel HSIC is significant while
the linear and conceptor readouts are not.  Descriptive only (HSIC is an
association diagnostic, not causal evidence; Tao arXiv:2608.16753: proxies can
separate from the objective).  Othello (arXiv:2309.00941) is the ontology
lesson behind the projection-set design.
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
from geometry_conceptor import nuisance_matrix, residualize_nuisance, standardize_nuisance  # noqa: E402
from geometry_localize import collect_linear_entries, group_token_sets, interaction_labels, pool_site, quartets  # noqa: E402
from geometry_models import auroc, quartet_residualize, r_squared, ridge_fit, ridge_predict, scene_center, unit  # noqa: E402

CELL_ORDER = ((0, 0), (0, 1), (1, 0), (1, 1))
GRID = 16


# --------------------------------------------------------------------------- #
# 1. Sliced W2 (Cramer-Wold arm)
# --------------------------------------------------------------------------- #


def sliced_w2(a: np.ndarray, b: np.ndarray, match_moments: bool) -> np.ndarray:
    """Per-projection W2^2 between 1-d samples a, b of shape [n, P] (rows = samples, columns = projections).

    Unequal sample sizes are handled by quantile interpolation.  With
    ``match_moments`` each projected sample is standardized to zero mean / unit
    sd (first two moments matched), so only shape differences remain.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if match_moments:
        a = (a - a.mean(0)) / (a.std(0) + 1e-12)
        b = (b - b.mean(0)) / (b.std(0) + 1e-12)
    n = max(a.shape[0], b.shape[0], 2)
    q = (np.arange(n) + 0.5) / n
    A = np.sort(a, axis=0)
    B = np.sort(b, axis=0)
    if A.shape[0] != n:
        A = np.stack([np.interp(q, (np.arange(A.shape[0]) + 0.5) / A.shape[0], A[:, j]) for j in range(A.shape[1])], axis=1)
    if B.shape[0] != n:
        B = np.stack([np.interp(q, (np.arange(B.shape[0]) + 0.5) / B.shape[0], B[:, j]) for j in range(B.shape[1])], axis=1)
    return np.mean((A - B) ** 2, axis=0)


def projection_set(rows_pool: np.ndarray, n_random: int, seed: int, n_eig: int = 16) -> dict[str, np.ndarray]:
    """Covariance eigenvectors of the pooled rows + frozen-seed random unit vectors. [P, d] each."""
    X = np.asarray(rows_pool, dtype=np.float64)
    Xc = X - X.mean(0)
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    eig = vt[: min(n_eig, (s > 1e-9 * max(s.max(), 1e-300)).sum())]
    rng = np.random.default_rng(seed)
    R = rng.normal(size=(n_random, X.shape[1]))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    return {"eig": eig, "random": R}


def loso_discriminant_projection(rows: np.ndarray, hz: np.ndarray, scene: np.ndarray) -> np.ndarray:
    """Out-of-fold mean-difference (H1 - H0) projection: each scene's rows are projected on the direction fit
    on the other scenes.  Returns the 1-d projected values [n]."""
    out = np.full(len(rows), np.nan)
    for s in np.unique(scene):
        tr = scene != s
        d = unit(rows[tr & (hz == 1)].mean(0) - rows[tr & (hz == 0)].mean(0))
        out[scene == s] = rows[scene == s] @ d
    return out


def aggregate(w: np.ndarray, top_k: int) -> dict[str, float]:
    w = np.asarray(w, dtype=np.float64)
    w = w[np.isfinite(w)]
    if w.size == 0:
        return {"topk_mean": float("nan"), "median": float("nan")}
    k = min(top_k, w.size)
    return {"topk_mean": float(np.sort(w)[::-1][:k].mean()), "median": float(np.median(w))}


def cramer_wold_arm(
    rows: np.ndarray, hz: np.ndarray, scene: np.ndarray, proj: dict[str, np.ndarray], flips: np.ndarray, top_k: int = 8
) -> dict[str, Any]:
    """rows [n, d] are action-effect rows (Delta_A H) labelled by hazard level (0/1) and scene.

    ``flips`` [n_perm, n_scenes] of +-1: -1 swaps the H0/H1 label of that scene's rows.
    Returns observed and null aggregates (before/after moment matching) for each projection family and a
    combined family, plus raw p-values.  Max-stat over sites is applied by the caller.
    """
    scenes = np.unique(scene)
    sidx = {s: i for i, s in enumerate(scenes)}
    scene_i = np.asarray([sidx[s] for s in scene])
    Pfix = np.concatenate([proj["eig"], proj["random"]], axis=0)  # [P, d]
    Z = rows @ Pfix.T  # [n, P]
    fam_slices = {"eig": slice(0, len(proj["eig"])), "random": slice(len(proj["eig"]), Z.shape[1])}

    def stats_for(hz_lab: np.ndarray) -> dict[str, dict[str, float]]:
        a, b = Z[hz_lab == 1], Z[hz_lab == 0]
        out = {}
        for fam, sl in fam_slices.items():
            for match in (False, True):
                out[f"{fam}|{'after' if match else 'before'}"] = aggregate(sliced_w2(a[:, sl], b[:, sl], match), top_k)
        disc = loso_discriminant_projection(rows, hz_lab, scene)
        for match in (False, True):
            out[f"loso_disc|{'after' if match else 'before'}"] = {"topk_mean": float(sliced_w2(disc[hz_lab == 1][:, None], disc[hz_lab == 0][:, None], match)[0])}
            out[f"loso_disc|{'after' if match else 'before'}"]["median"] = out[f"loso_disc|{'after' if match else 'before'}"]["topk_mean"]
        return out

    obs = stats_for(hz)
    null: dict[str, dict[str, list[float]]] = {k: {"topk_mean": [], "median": []} for k in obs}
    for f in flips:
        hz_p = np.where(f[scene_i] < 0, 1 - hz, hz)
        st = stats_for(hz_p)
        for k in obs:
            for agg in ("topk_mean", "median"):
                null[k][agg].append(st[k][agg])
    res = {}
    for k in obs:
        res[k] = {}
        for agg in ("topk_mean", "median"):
            nv = np.asarray(null[k][agg])
            o = obs[k][agg]
            res[k][agg] = {"observed": o, "p_raw": float((np.sum(nv >= o) + 1) / (len(nv) + 1)) if np.isfinite(o) else float("nan"),
                           "null_mean": float(np.nanmean(nv)) if nv.size else float("nan"), "null": nv}
    return res


# --------------------------------------------------------------------------- #
# 2. nHSIC / CKA
# --------------------------------------------------------------------------- #


def rbf_kernel(X: np.ndarray, bandwidth: float) -> np.ndarray:
    aa = (X**2).sum(1)
    d2 = np.maximum(aa[:, None] + aa[None, :] - 2 * X @ X.T, 0.0)
    return np.exp(-d2 / (2 * bandwidth**2))


def median_bandwidth(X: np.ndarray) -> float:
    aa = (X**2).sum(1)
    d2 = np.maximum(aa[:, None] + aa[None, :] - 2 * X @ X.T, 0.0)
    iu = np.triu_indices(len(X), 1)
    v = np.sqrt(d2[iu])
    v = v[v > 0]
    return float(np.median(v)) if v.size else 1.0


def center_kernel(K: np.ndarray) -> np.ndarray:
    n = len(K)
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def nhsic(Kx: np.ndarray, Ky: np.ndarray) -> float:
    Kx, Ky = center_kernel(Kx), center_kernel(Ky)
    num = float(np.sum(Kx * Ky))
    den = float(np.sqrt(np.sum(Kx * Kx) * np.sum(Ky * Ky)))
    return num / den if den > 0 else float("nan")


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    return nhsic(X @ X.T, Y @ Y.T)


def label_kernel(y: np.ndarray, kind: str) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if kind == "binary":
        return (y[:, None] == y[None, :]).astype(np.float64)
    bw = median_bandwidth(y[:, None])
    return rbf_kernel(y[:, None], bw if bw > 0 else 1.0)


def hsic_arm(X: np.ndarray, labels: dict[str, tuple[np.ndarray, str]], scene: np.ndarray, n_perm: int, seed: int) -> dict[str, Any]:
    """nHSIC (RBF on rows, bandwidth = median heuristic, label-free hence fold-invariant) and linear CKA, with a
    within-scene (quartet) permutation of the labels.

    The factorial labels (interaction contrast, hazard-under-path, contact) have the SAME pattern in every
    scene, so permuting whole scene blocks is a no-op; the exchangeable null here permutes the labels among the
    cells of each scene, keeping scene identity and the scene-level nuisance fixed."""
    ok_rows = np.all(np.isfinite(X), axis=1)
    X = X[ok_rows]
    scene = scene[ok_rows]
    bw = median_bandwidth(X)
    Kx = rbf_kernel(X, bw)
    rng = np.random.default_rng(seed)
    scenes = np.unique(scene)
    out = {"bandwidth": bw, "n_rows": int(len(X))}
    for name, (y, kind) in labels.items():
        y = np.asarray(y, dtype=np.float64)[ok_rows]
        ok = np.isfinite(y)
        if ok.sum() < 8 or len(np.unique(y[ok])) < 2:
            out[name] = {"status": "insufficient"}
            continue
        Kxo = Kx[np.ix_(ok, ok)]
        Xo = X[ok]
        so = scene[ok]
        Ky = label_kernel(y[ok], kind)
        obs_h = nhsic(Kxo, Ky)
        Yl = y[ok][:, None] - y[ok].mean()
        obs_c = linear_cka(Xo - Xo.mean(0), Yl)
        null_h, null_c = [], []
        blocks = [np.flatnonzero(so == s) for s in np.unique(so)]
        yo = y[ok]
        for _ in range(n_perm):
            new = yo.copy()
            for blk in blocks:
                new[blk] = yo[blk][rng.permutation(len(blk))]
            Kyp = label_kernel(new, kind)
            null_h.append(nhsic(Kxo, Kyp))
            null_c.append(linear_cka(Xo - Xo.mean(0), new[:, None] - new.mean()))
        null_h, null_c = np.asarray(null_h), np.asarray(null_c)
        out[name] = {
            "nhsic": obs_h, "p_nhsic": float((np.sum(null_h >= obs_h) + 1) / (len(null_h) + 1)),
            "cka": obs_c, "p_cka": float((np.sum(null_c >= obs_c) + 1) / (len(null_c) + 1)),
            "nhsic_minus_cka_z": float(((obs_h - null_h.mean()) / (null_h.std() + 1e-12)) - ((obs_c - null_c.mean()) / (null_c.std() + 1e-12))),
            "status": "ok",
        }
    return out


# --------------------------------------------------------------------------- #
# 3. Rotating subspaces: principal angles, Grassmann distance, Procrustes transport
# --------------------------------------------------------------------------- #


def rank_k_basis(rows: np.ndarray, k: int) -> np.ndarray:
    X = np.asarray(rows, dtype=np.float64)
    _, _, vt = np.linalg.svd(X - X.mean(0) if len(X) > k + 1 else X, full_matrices=False)
    return vt[:k].T  # [d, k]


def principal_angles(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    s = np.linalg.svd(U.T @ V, compute_uv=False)
    return np.arccos(np.clip(s, -1.0, 1.0))


def grassmann_distance(U: np.ndarray, V: np.ndarray) -> float:
    return float(np.sqrt(np.sum(principal_angles(U, V) ** 2)))


def procrustes(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Q* = argmin_{Q orthogonal} ||U Q - V||_F  (k x k)."""
    A, _, Bt = np.linalg.svd(U.T @ V)
    return A @ Bt


def procrustes_residual(U: np.ndarray, V: np.ndarray) -> float:
    Q = procrustes(U, V)
    return float(np.linalg.norm(U @ Q - V) / max(np.linalg.norm(V), 1e-12))


def transfer_test(
    Xa: np.ndarray, Xb: np.ndarray, Da: np.ndarray, Db: np.ndarray, scene_rows: np.ndarray, scene_D: np.ndarray, y: np.ndarray, k: int, lam: float
) -> dict[str, Any]:
    """LOSO: bases U_a, U_b from the training scenes' interaction rows; Q* = argmin ||Z_a Q - Z_b||_F on the
    matched training scenes' coordinates Z = D U (the doc's "matched training counterfactuals": it fixes the
    internal rotation / sign convention of the two SVD bases, which basis-level Procrustes cannot when the
    subspaces differ).  Readout trained on source coordinates X_a U_a; scored on the held-out scene's TARGET
    activations either raw (X_b U_a) or transported ((X_b U_b) Q*^T).  Within-source held-out = ceiling."""
    scenes = np.unique(scene_rows)
    oof = {"raw": np.full(len(y), np.nan), "transported": np.full(len(y), np.nan), "source": np.full(len(y), np.nan)}
    for s in scenes:
        tr_rows, te_rows = scene_rows != s, scene_rows == s
        trD = scene_D != s
        if trD.sum() <= k:
            continue
        Ua, Ub = rank_k_basis(Da[trD], k), rank_k_basis(Db[trD], k)
        Q = procrustes(Da[trD] @ Ua, Db[trD] @ Ub)
        ok = tr_rows & np.isfinite(y)
        Za = Xa @ Ua
        w, b = ridge_fit(Za[ok], y[ok], lam)
        oof["source"][te_rows] = ridge_predict(Za[te_rows], w, b)
        oof["raw"][te_rows] = ridge_predict(Xb[te_rows] @ Ua, w, b)
        oof["transported"][te_rows] = ridge_predict((Xb[te_rows] @ Ub) @ Q.T, w, b)
    ok = np.isfinite(y) & np.isfinite(oof["raw"])
    if ok.sum() < 8 or len(np.unique(y[ok])) < 2:
        return {"status": "insufficient"}
    return {"status": "ok", **{k2: auroc(v[ok], y[ok]) for k2, v in oof.items()}, "n_cells": int(ok.sum())}


# --------------------------------------------------------------------------- #
# 4. Two-sided token x hidden operator (Tucker-2 / HOSVD) on the 16x16 grid
# --------------------------------------------------------------------------- #


def interaction_grid_tensors(acts, token_index, cells, token_sets, Q, step, groups: tuple[str, ...]) -> tuple[np.ndarray, list[str]]:
    """Per scene: [256, d] interaction matrix on the spatial grid (zeros off-group), from matched tokens."""
    tensors, ids = [], []
    d = acts.shape[-1]
    for pid, q in Q.items():
        if not all(k in q for k in CELL_ORDER):
            continue
        want = np.unique(np.concatenate([token_sets[(pid, step, g)] for g in groups]))
        per = {}
        for k in CELL_ORDER:
            r = q[k]
            ti = token_index[r, step]
            m = np.isin(ti, want) & (ti >= 0)
            per[k] = dict(zip(ti[m].tolist(), acts[r, step, m].astype(np.float64)))
        common = sorted(set(per[(0, 0)]) & set(per[(0, 1)]) & set(per[(1, 0)]) & set(per[(1, 1)]))
        if len(common) < 2:
            continue
        T = np.zeros((GRID * GRID, d))
        for tok in common:
            T[tok] = per[(1, 1)][tok] - per[(1, 0)][tok] - per[(0, 1)][tok] + per[(0, 0)][tok]
        tensors.append(T)
        ids.append(pid)
    return (np.asarray(tensors) if tensors else np.zeros((0, GRID * GRID, d))), ids


def two_sided_check(tensors: np.ndarray, r_tok: int = 4, r_hid: int = 4) -> dict[str, Any]:
    """Held-out (LOSO over scenes) reconstruction of the interaction tensor with token-only, hidden-only and
    two-sided (Tucker-2) projectors fit on the training scenes; equal-budget comparison r_tok = r_hid = 4."""
    S = tensors.shape[0]
    if S < 3:
        return {"status": "insufficient"}
    err = {"token_only": [], "hidden_only": [], "two_sided": [], "two_sided_2x2": []}
    energy_tok = []
    for i in range(S):
        tr = np.setdiff1d(np.arange(S), [i])
        Tr = tensors[tr]
        # token modes: left singular vectors of the [256, S*d] unfolding; hidden modes: right of the [S*256, d] unfolding
        W = np.linalg.svd(np.concatenate(list(Tr), axis=1), full_matrices=False)[0]
        V = np.linalg.svd(np.concatenate(list(Tr), axis=0), full_matrices=False)[2].T
        X = tensors[i]
        tot = float(np.sum(X**2)) + 1e-12
        PT = W[:, :r_tok] @ W[:, :r_tok].T
        PD = V[:, :r_hid] @ V[:, :r_hid].T
        err["token_only"].append(float(np.sum((X - PT @ X) ** 2)) / tot)
        err["hidden_only"].append(float(np.sum((X - X @ PD) ** 2)) / tot)
        err["two_sided"].append(float(np.sum((X - PT @ X @ PD) ** 2)) / tot)
        PT2, PD2 = W[:, :2] @ W[:, :2].T, V[:, :2] @ V[:, :2].T
        err["two_sided_2x2"].append(float(np.sum((X - PT2 @ X @ PD2) ** 2)) / tot)
        e = np.sum(X**2, axis=1)
        energy_tok.append(float(e.sum() ** 2 / (np.sum(e**2) + 1e-12)))
    out = {k: {"heldout_relative_error_mean": float(np.mean(v)), "explained": float(1 - np.mean(v))} for k, v in err.items()}
    out["token_participation_ratio"] = float(np.mean(energy_tok))
    out["status"] = "ok"
    out["n_scenes"] = int(S)
    out["spatially_distributed"] = bool(np.mean(energy_tok) > 2.0 and out["hidden_only"]["explained"] < out["two_sided"]["explained"] + 0.05)
    return out


# --------------------------------------------------------------------------- #
# 5. Descriptives
# --------------------------------------------------------------------------- #


def descriptives(rows: np.ndarray) -> dict[str, float]:
    X = np.asarray(rows, dtype=np.float64)
    if len(X) < 2:
        return {}
    s = np.linalg.svd(X - X.mean(0), compute_uv=False) ** 2
    p = s / max(s.sum(), 1e-12)
    p = p[p > 0]
    ent = float(-(p * np.log(p)).sum())
    pr = float(s.sum() ** 2 / max((s**2).sum(), 1e-12))
    conc = np.sum(np.abs(X), axis=1) ** 2 / (np.sum(X**2, axis=1) + 1e-12)
    return {"spectral_entropy": ent, "spectral_entropy_exp": float(np.exp(ent)), "participation_ratio": pr,
            "coordinate_concentration_mean": float(conc.mean()), "coordinate_concentration_over_d": float(conc.mean() / X.shape[1])}


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #


def per_token_cells(acts, token_index, cells, token_sets, Q, step, group):
    """Per scene: dict cell-key -> {token id: vector} restricted to tokens common to the quartet."""
    out = {}
    for pid, q in Q.items():
        if not all(k in q for k in CELL_ORDER):
            continue
        want = token_sets[(pid, step, group)]
        per = {}
        for k in CELL_ORDER:
            r = q[k]
            ti = token_index[r, step]
            m = np.isin(ti, want) & (ti >= 0)
            per[k] = dict(zip(ti[m].tolist(), acts[r, step, m].astype(np.float64)))
        common = sorted(set.intersection(*(set(per[k]) for k in CELL_ORDER)))
        if common:
            out[pid] = {k: np.stack([per[k][t] for t in common]) for k in CELL_ORDER}
            out[pid]["tokens"] = np.asarray(common)
    return out


def effect_rows(per_scene: dict, nuisance_B: np.ndarray | None, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Action-effect rows Delta_A H per scene/token with hazard labels; interaction rows.
    Returns (effect_rows, hazard, scene, interaction_rows_per_token)."""
    E, hz, sc, I = [], [], [], []
    for pid, cell in per_scene.items():
        e1 = cell[(1, 1)] - cell[(1, 0)]
        e0 = cell[(0, 1)] - cell[(0, 0)]
        E.append(e1); hz += [1] * len(e1); sc += [pid] * len(e1)
        E.append(e0); hz += [0] * len(e0); sc += [pid] * len(e0)
        I.append(e1 - e0)
    if not E:
        return np.zeros((0, 1)), np.zeros(0), np.zeros(0), np.zeros((0, 1))
    return np.concatenate(E), np.asarray(hz), np.asarray(sc), np.concatenate(I)


# --------------------------------------------------------------------------- #
# Main per-site analysis
# --------------------------------------------------------------------------- #


def analyze_site(site, acts, token_index, cells, token_sets, Q, labels, meta, N, lab_hup, args) -> list[dict[str, Any]]:
    results = []
    n = len(cells)
    train_all = np.arange(n)
    for step in args.steps:
        for group in args.groups:
            entry: dict[str, Any] = {"site_id": site, "step": step, "group": group}
            P, _ = pool_site(acts, token_index, cells, token_sets, step, group)
            valid = np.all(np.isfinite(P), axis=1)
            if valid.sum() < 8:
                entry["status"] = "no_tokens"
                results.append(entry)
                continue
            P = np.where(valid[:, None], P, np.nan_to_num(P[valid].mean(0)))
            Pr = residualize_nuisance(P, N, train_all)
            Pc = scene_center(Pr, meta["pair_id"])
            # pooled effect rows
            pooled_scene = {}
            for pid, q in Q.items():
                if all(k in q for k in CELL_ORDER):
                    pooled_scene[pid] = {k: Pr[q[k]][None, :] for k in CELL_ORDER}
            Ep, hzp, scp, Ip = effect_rows(pooled_scene, None, "pooled")
            # per-token rows (nuisance component removed with the pooled ridge fit, applied per token)
            per_scene = per_token_cells(acts, token_index, cells, token_sets, Q, step, group)
            Et, hzt, sct, It = effect_rows(per_scene, None, "token")
            entry["n_pooled_rows"] = int(len(Ep))
            entry["n_token_rows"] = int(len(Et))
            entry["n_scenes"] = int(len(pooled_scene))
            if len(pooled_scene) < 4:
                entry["status"] = "too_few_quartets"
                results.append(entry)
                continue
            scenes = sorted(pooled_scene)
            rng = np.random.default_rng(args.seed)
            flips = rng.choice([-1.0, 1.0], size=(args.n_perm, len(scenes)))
            entry["_flips_key"] = tuple(scenes)
            # 1. Cramer-Wold arm on pooled and per-token effect rows (projections from the pooled residualized cells)
            proj = projection_set(Pc, args.n_random, args.seed)
            cw = {}
            for name, (E, hz, sc) in {"pooled": (Ep, hzp, scp), "token": (Et, hzt, sct)}.items():
                if len(E) < 8:
                    cw[name] = {"status": "insufficient"}
                    continue
                sc_i = np.asarray([scenes.index(s) for s in sc])
                res = cramer_wold_arm(E, hz, sc_i, proj, flips, top_k=args.top_k)
                cw[name] = {k: {agg: {kk: vv for kk, vv in v[agg].items() if kk != "null"} for agg in v} for k, v in res.items()}
                cw[name]["_null"] = {k: {agg: v[agg]["null"] for agg in v} for k, v in res.items()}
            entry["cramer_wold"] = cw
            # 2. nHSIC on pooled residualized cells (scene-centred) and on per-token cell rows
            y_int = labels["interaction"]
            y_force = labels["force"]
            lab = {"interaction": (y_int, "binary"), "force": (y_force, "continuous"), "hazard_under_path": (lab_hup, "continuous")}
            R_int = np.nan_to_num(quartet_residualize(Pr, meta, keep_hazard=True))
            entry["hsic"] = {"pooled_scene_centred": hsic_arm(Pc, lab, meta["pair_id"], args.n_perm_hsic, args.seed),
                             "pooled_interaction_residual": hsic_arm(R_int, {"interaction": (y_int, "binary")}, meta["pair_id"], args.n_perm_hsic, args.seed)}
            # 3. bases of the interaction rows (per-token if available) kept for cross-site tracking
            I_rows = It if len(It) >= args.k + 2 else Ip
            entry["_U"] = rank_k_basis(I_rows, args.k)
            entry["_Ip"] = Ip
            entry["_Pc"] = Pc
            entry["_R_int"] = R_int
            entry["_scene_D"] = np.asarray(scenes)
            entry["basis_source"] = "token" if len(It) >= args.k + 2 else "pooled"
            # 4. two-sided operator check on the gripper+egg grid (once per site/step; stored under the first group)
            if group == args.groups[0]:
                tens, _ = interaction_grid_tensors(acts, token_index, cells, token_sets, Q, step, tuple(g for g in ("gripper_corridor", "egg") if (next(iter(Q)), step, g) in token_sets))
                entry["two_sided"] = two_sided_check(tens, args.k, args.k)
            # 5. descriptives
            entry["descriptives"] = {"pooled_interaction": descriptives(Ip), "token_interaction": descriptives(It) if len(It) >= 2 else {}}
            entry["status"] = "ok"
            results.append(entry)
    return results


def maxstat_correct(entries: list[dict[str, Any]]) -> None:
    """Max-stat p over sites sharing a (step, group, rows-kind, projection family, moment state, aggregate)."""
    groups: dict[tuple, list[tuple[dict, str, str, str]]] = {}
    for e in entries:
        if e.get("status") != "ok":
            continue
        for kind, cw in e["cramer_wold"].items():
            if cw.get("status") == "insufficient":
                continue
            for fam in cw["_null"]:
                for agg in cw["_null"][fam]:
                    groups.setdefault((e["step"], e["group"], kind, fam, agg, e["_flips_key"]), []).append((e, kind, fam, agg))
    for key, members in groups.items():
        nulls = np.stack([m[0]["cramer_wold"][m[1]]["_null"][m[2]][m[3]] for m in members])  # [sites, n_perm]
        # standardize each site's null and observed so the max is comparable across sites
        mu, sd = nulls.mean(1, keepdims=True), nulls.std(1, keepdims=True) + 1e-12
        zmax = ((nulls - mu) / sd).max(axis=0)
        for i, (e, kind, fam, agg) in enumerate(members):
            o = e["cramer_wold"][kind][fam][agg]["observed"]
            z = (o - mu[i, 0]) / sd[i, 0]
            e["cramer_wold"][kind][fam][agg]["z"] = float(z)
            e["cramer_wold"][kind][fam][agg]["p_maxstat"] = float((np.sum(zmax >= z) + 1) / (len(zmax) + 1))
            e["cramer_wold"][kind][fam][agg]["n_sites_in_family"] = len(members)
    for e in entries:
        if e.get("status") == "ok":
            for cw in e["cramer_wold"].values():
                cw.pop("_null", None)


def tracking_and_transfer(entries: list[dict[str, Any]], labels, meta, args) -> dict[str, Any]:
    by = {(e["site_id"], e["step"], e["group"]): e for e in entries if e.get("status") == "ok"}
    y = labels["interaction"]
    out: dict[str, Any] = {"adjacent_layers": {}, "adjacent_steps": {}}

    def pair_stats(ea, eb):
        ang = principal_angles(ea["_U"], eb["_U"])
        res = {"principal_angles_deg": np.degrees(ang).tolist(), "grassmann_distance": float(np.sqrt(np.sum(ang**2))), "mean_cos": float(np.mean(np.cos(ang))),
               "basis_procrustes_residual": procrustes_residual(ea["_U"], eb["_U"])}
        tr = transfer_test(ea["_R_int"], eb["_R_int"], ea["_Ip"], eb["_Ip"], meta["pair_id"], ea["_scene_D"], y, args.k, args.lam) if np.array_equal(ea["_scene_D"], eb["_scene_D"]) else {"status": "scene_mismatch"}
        res["transfer"] = tr
        if tr.get("status") == "ok":
            res["transport_helps"] = bool(tr["transported"] - tr["raw"] > 0.1 and tr["transported"] > 0.75)
        return res

    for (site, step, group), e in by.items():
        try:
            L = int(site.split(".")[0][1:])
        except ValueError:
            continue
        hook = site.split(".", 1)[1]
        nxt = by.get((f"L{L + 1:02d}.{hook}", step, group))
        if nxt is not None:
            out["adjacent_layers"][f"{site}->L{L + 1:02d}.{hook}|s{step}|{group}"] = pair_stats(e, nxt)
        nstep = by.get((site, step + 1, group))
        if nstep is not None:
            out["adjacent_steps"][f"{site}|{group}|s{step}->s{step + 1}"] = pair_stats(e, nstep)
    return out


def hazard_under_path_labels(stimulus: Path, cells, manifest) -> np.ndarray:
    try:
        from geometry_frames import frame_coordinates, load_cell_geometry
        from geometry_localize import cell_artifact
    except Exception:
        return np.full(len(cells), np.nan)
    y = np.full(max(c["row"] for c in cells) + 1, np.nan)
    for c in cells:
        try:
            row = {"cell_id": c["cell_id"], "artifact": str(cell_artifact(stimulus, c, manifest.get(c["cell_id"])))}
            fr, _, _ = frame_coordinates(load_cell_geometry(stimulus, row))
            y[c["row"]] = float(fr["hazard_under_path"][0])
        except Exception:
            pass
    return y


def rank_tables(entries, linear, conceptor, steps=(0, 1), groups=("gripper_corridor", "egg")) -> dict[str, Any]:
    tables = {}
    flags = []
    for step in steps:
        for group in groups:
            rows = []
            for e in entries:
                if e.get("status") != "ok" or e["step"] != step or e["group"] != group:
                    continue
                cw = e["cramer_wold"].get("token") if e["cramer_wold"].get("token", {}).get("status") != "insufficient" else e["cramer_wold"].get("pooled")
                if not cw or cw.get("status") == "insufficient":
                    continue
                after = cw["random|after"]["topk_mean"]
                before = cw["random|before"]["topk_mean"]
                disc_after = cw["loso_disc|after"]["topk_mean"]
                h = e["hsic"]["pooled_interaction_residual"].get("interaction", {})
                hs = e["hsic"]["pooled_scene_centred"]
                lin = linear.get((e["site_id"], step, group), {})
                con = conceptor.get((e["site_id"], step, group), {})
                row = {
                    "site_id": e["site_id"],
                    "w2_after_z": after.get("z"), "w2_after_p_raw": after["p_raw"], "w2_after_p_maxstat": after.get("p_maxstat"),
                    "w2_before_z": before.get("z"), "w2_before_p_raw": before["p_raw"],
                    "disc_after_p_raw": disc_after["p_raw"], "disc_after_p_maxstat": disc_after.get("p_maxstat"),
                    "nhsic_interaction": h.get("nhsic"), "p_nhsic_interaction": h.get("p_nhsic"), "cka_interaction": h.get("cka"), "p_cka_interaction": h.get("p_cka"),
                    "nhsic_force": hs.get("force", {}).get("nhsic"), "p_nhsic_force": hs.get("force", {}).get("p_nhsic"),
                    "nhsic_hazard_under_path": hs.get("hazard_under_path", {}).get("nhsic"), "p_nhsic_hup": hs.get("hazard_under_path", {}).get("p_nhsic"),
                    "linear_map_t": lin.get("t"), "linear_map_p_maxt": lin.get("p_maxt_fwer"),
                    "conceptor_interaction_auroc": con.get("safety"), "conceptor_beats_rank_one": con.get("beats_rank_one_ci"),
                    "two_sided": (e.get("two_sided") or {}).get("spatially_distributed"),
                    "participation_ratio": e["descriptives"]["pooled_interaction"].get("participation_ratio"),
                }
                w2_sig = any(cw[f"{fam}|after"]["topk_mean"].get("p_maxstat", 1.0) < 0.05 for fam in ("eig", "random", "loso_disc"))
                sig_higher = w2_sig or (h.get("p_nhsic", 1.0) < 0.05 and h.get("p_cka", 1.0) >= 0.05)
                row["w2_after_maxstat_any_family"] = bool(w2_sig)
                lin_weak = (lin.get("t") is None) or (not np.isfinite(lin.get("t") or np.nan)) or (lin.get("t") < 2.0)
                con_weak = not con.get("beats_rank_one_ci", False)
                row["higher_order_only"] = bool(sig_higher and lin_weak and con_weak)
                rows.append(row)
                if row["higher_order_only"]:
                    flags.append({"step": step, "group": group, **row})
            rows.sort(key=lambda r: -(r["w2_after_z"] if r["w2_after_z"] is not None and np.isfinite(r["w2_after_z"]) else -np.inf))
            tables[f"step{step}|{group}"] = rows
    return {"tables": tables, "higher_order_only_flags": flags}


def _jd(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--stimulus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--discovery-seeds", type=Path, default=None)
    ap.add_argument("--linear-map", type=Path, default=None)
    ap.add_argument("--conceptor-map", type=Path, default=None)
    ap.add_argument("--sites", nargs="*", default=None)
    ap.add_argument("--steps", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--groups", nargs="*", default=["gripper_corridor", "egg"])
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--n-random", type=int, default=256)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--n-perm", type=int, default=499)
    ap.add_argument("--n-perm-hsic", type=int, default=499)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hsic-only", action="store_true", help="recompute only the HSIC block and merge it into an existing higher_order_map.json (tables rebuilt)")
    args = ap.parse_args(argv)

    t0 = time.time()
    idx = json.loads((args.dump / "activations" / "index.json").read_text())
    cells = idx["cells"] if isinstance(idx, dict) else idx
    if args.discovery_seeds is not None:
        allowed = {int(v) for v in args.discovery_seeds.read_text().split() if v.strip()}
        cells = [c for c in cells if int(c["seed"]) in allowed]
    cells = sorted(cells, key=lambda c: int(c["row"]))
    dump_rows = np.asarray([int(c["row"]) for c in cells])
    for i, c in enumerate(cells):
        c["row"] = i
    n_steps = int(idx.get("n_steps", 3)) if isinstance(idx, dict) else 3
    offset = int(idx.get("group_frame_offset", 0)) if isinstance(idx, dict) else 0
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
    n = len(cells)
    meta = {"pair_id": np.asarray([c["pair_id"] for c in cells]).astype(str), "hazard": np.asarray([int(c["hazard"]) for c in cells]), "action": np.asarray([int(c["candidate_action"]) for c in cells])}
    N, ncols = nuisance_matrix(args.stimulus, cells, manifest)
    lab_hup = hazard_under_path_labels(args.stimulus, cells, manifest)
    token_sets = group_token_sets(args.stimulus, cells, tuple(dict.fromkeys(list(args.groups) + ["gripper_corridor", "egg"])), n_steps, offset)
    linear: dict = {}
    if args.linear_map and args.linear_map.exists():
        collect_linear_entries(json.loads(args.linear_map.read_text()), linear)
    conceptor: dict = {}
    if args.conceptor_map and args.conceptor_map.exists():
        cm = json.loads(args.conceptor_map.read_text())
        for r in cm.get("ranked_by_target", {}).get("interaction", []):
            conceptor[(r["site_id"], r["step"], r["group"])] = r
    sites = args.sites or sorted(p.stem for p in (args.dump / "activations").glob("*.npz"))
    entries: list[dict[str, Any]] = []
    if args.hsic_only:
        existing = json.loads((args.out / "higher_order_map.json").read_text())
        by_key = {(e["site_id"], e["step"], e["group"]): e for e in existing["entries"]}
        train_all = np.arange(n)
        for i, site in enumerate(sites):
            with np.load(args.dump / "activations" / f"{site}.npz") as z:
                acts = np.asarray(z["acts"])[dump_rows]
                tix = np.asarray(z["token_index"])[dump_rows] if "token_index" in z else np.broadcast_to(np.arange(acts.shape[2]), acts.shape[:3])
            for step in args.steps:
                for group in args.groups:
                    e = by_key.get((site, step, group))
                    if e is None or e.get("status") != "ok":
                        continue
                    P, _ = pool_site(acts, tix, cells, token_sets, step, group)
                    valid = np.all(np.isfinite(P), axis=1)
                    P = np.where(valid[:, None], P, np.nan_to_num(P[valid].mean(0)))
                    Pr = residualize_nuisance(P, N, train_all)
                    Pc = scene_center(Pr, meta["pair_id"])
                    R_int = np.nan_to_num(quartet_residualize(Pr, meta, keep_hazard=True))
                    lab = {"interaction": (labels["interaction"], "binary"), "force": (labels["force"], "continuous"), "hazard_under_path": (lab_hup, "continuous")}
                    e["hsic"] = {"pooled_scene_centred": hsic_arm(Pc, lab, meta["pair_id"], args.n_perm_hsic, args.seed),
                                 "pooled_interaction_residual": hsic_arm(R_int, {"interaction": (labels["interaction"], "binary")}, meta["pair_id"], args.n_perm_hsic, args.seed)}
            print(f"[hsic {i + 1}/{len(sites)}] {site} ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
        entries = existing["entries"]
        tracking = existing["tracking"]
        existing["hsic_null"] = "within-scene (quartet) label permutation"
    else:
        for i, site in enumerate(sites):
            with np.load(args.dump / "activations" / f"{site}.npz") as z:
                acts = np.asarray(z["acts"])[dump_rows]
                tix = np.asarray(z["token_index"])[dump_rows] if "token_index" in z else np.broadcast_to(np.arange(acts.shape[2]), acts.shape[:3])
            entries.extend(analyze_site(site, acts, tix, cells, token_sets, Q, labels, meta, N, lab_hup, args))
            print(f"[{i + 1}/{len(sites)}] {site} ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
        maxstat_correct(entries)
        tracking = tracking_and_transfer(entries, labels, meta, args)
        for e in entries:
            for k in list(e):
                if k.startswith("_"):
                    e.pop(k)
    tabs = rank_tables(entries, linear, conceptor, steps=tuple(s for s in (0, 1) if s in args.steps), groups=tuple(g for g in ("gripper_corridor", "egg") if g in args.groups))
    focus = [e for e in entries if e["site_id"] == "L07.attn_out" and e["step"] == 0]
    report = {
        "protocol": "cgs-geometry-higher-order-v0.1", "dump": str(args.dump), "stimulus": str(args.stimulus), "n_cells": n, "n_scenes": len(Q),
        "scenes_with_hazard2": int(sum(all(k in q for k in ((2, 0), (2, 1))) for q in Q.values())),
        "steps": args.steps, "groups": args.groups, "k": args.k, "n_random_projections": args.n_random, "top_k": args.top_k, "n_perm": args.n_perm,
        "nuisance_columns": ncols, "moment_matching": "per projection: each 1-d sample standardized to zero mean / unit sd (first two moments matched)",
        "ranked_tables": tabs["tables"], "higher_order_only_flags": tabs["higher_order_only_flags"],
        "tracking": tracking, "focus_L07_attn_out_step0": focus, "entries": entries, "runtime_s": time.time() - t0,
        "hsic_null": "within-scene (quartet) label permutation", "w2_flag_rule": "max-stat p < 0.05 in any projection family after moment matching",
        "interpretation_scope": "Descriptive; HSIC and sliced-W2 are association/separation diagnostics, not causal evidence.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "higher_order_map.json").write_text(json.dumps(report, indent=1, default=_jd) + "\n")
    for key, rows in tabs["tables"].items():
        print(f"== {key} ==")
        for r in rows[:8]:
            print(f"  {r['site_id']:>14s} W2after z={r['w2_after_z']:+.2f} p={r['w2_after_p_raw']:.3f}/{r['w2_after_p_maxstat']:.3f} | W2before z={r['w2_before_z']:+.2f} p={r['w2_before_p_raw']:.3f} "
                  f"| disc_after p={r['disc_after_p_raw']:.3f} | nHSIC int={r['nhsic_interaction']} p={r['p_nhsic_interaction']} cka p={r['p_cka_interaction']} | map_t={r['linear_map_t']} | conc={r['conceptor_interaction_auroc']} | flag={r['higher_order_only']}")
    return report


if __name__ == "__main__":
    main()
