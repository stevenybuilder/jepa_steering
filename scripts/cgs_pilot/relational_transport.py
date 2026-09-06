#!/usr/bin/env python3
"""Relational transport and residual scoring of the per-scene hazard x action fields (LRH.md section 2; CPU, numpy only).

Descriptive analysis added 2026-09-03 (experiment_design.md "Added descriptive analyses"): it characterises the
per-scene interaction fields, rescores the models against truth with the discovery-field template removed, and
tests whether the RELATIONAL organisation of the field (which scenes move alike) transports across arms / models.
It is not localisation and changes no gate threshold.

Inputs: two latent caches (``latent_cache.py`` npz, or the slim last-frame variant written by ``--slim``) computed on
the SAME discovery cells for arm A and arm B (one seed), plus the eval dirs (manifest.jsonl, masks/, discovery_seeds.txt).

Per scene w (pair_id) and token group g (``hazard_corridor`` = the gate's primary group, and ``hazard``; union of the
group over the scene's 8 cells at the last predicted frame, further unioned across the two arms so both arms use the
same token set), on the last predicted frame, with P(h, a) the latent grid of cell (hazard level h, action a) flattened
over the group tokens, the fields are

    ped_did              (P(1,1) - P(1,0)) - (P(0,1) - P(0,0))       pedestrian x action DiD
    cone_did             (P(3,1) - P(3,0)) - (P(0,1) - P(0,0))       cone x action DiD (level 3)
    null_did             (P(2,1) - P(2,0)) - (P(0,1) - P(0,0))       matched-displacement (H0') x action DiD
    ped_did_relational   ped_did - null_did
    cone_did_relational  cone_did - null_did
    action               P(0,1) - P(0,0)                             plain action field (hazard off path)

computed from ``prediction`` (MODEL field) and from ``target`` (TRUE field).

1. Field characterisation: PC1 (and top-5) variance fraction of the centred field, participation ratio, spherical
   variance 1 - ||mean unit vector|| versus uniformly random directions in the same dimension (Monte Carlo), and the
   mean-direction consistency (mean cosine of each scene to the field mean; in-sample and leave-one-scene-out).
2. Residual cosine (model vs truth): raw cos(M_w, T_w) and cos(M_w - g_-w, T_w - g_-w) with g_-w the mean of the TRUE
   field over the OTHER discovery scenes (never the scored scene). Baselines scored identically: copy-delta (the true
   field of the nearest other scene by pooled context latent; the retrieval_baseline.py donor rule) and the field mean
   g_-w itself (residual cosine 0 by construction). Scene-clustered bootstrap CIs; sign-flip p for model - baseline.
3. Relational transport (source field S, target field T, same scenes), leave-one-scene-out: for held-out w the weights
   are s_j = cos(S_w, S_j) over j != w and the synthesised target vector is T_hat_w = sum_j s_j T_j / sum_j |s_j|
   (signed kernel regression; no rotation, no shared coordinates). Scored with residual cosine against T_w (LOSO mean
   of T), raw cosine, and the Spearman correlation of ||T_hat_w|| with ||T_w|| across scenes. Configurations: TRUE
   A<->B, MODEL A<->B, MODEL->TRUE within arm and across arms, for every field; identity swap: A's pedestrian DiD
   relations onto B's cone DiD (consequence-matched, identity-swapped) vs onto B's pedestrian DiD (identity-matched).
   Nulls (all exact in Gram space): (a) scene permutation of the weights (marginals kept), (b) isotropic Gaussian
   vector at the norm of T_hat_w, (c) covariance-matched Gaussian (LOSO mean / covariance of the target field),
   (d) the real target vector T_w randomly rotated within the span of the target field (norm preserved).
   Reported: score - null with scene-clustered CI and sign-flip p, plus the draw-level p.
4. Covariate heterogeneity (descriptive): Spearman correlations (bootstrap CI, permutation p) of the DiD magnitude
   and of the directional deviation from the LOSO field mean with hazard pixel size, lane offset and ego speed from the
   manifest; a covariate with fewer than 5 distinct values is flagged degenerate.

``--synthetic`` runs the planted self-test (a field = shared template + scene code that transports across arms, and a
null variant whose scene codes are independent across arms) through the same code path.
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
from protocol import NULL_CONTROL_HAZARD, OBJECT_HAZARD, canonical_json, sha256_file  # noqa: E402
from stats_utils import finite, sign_flip_p  # noqa: E402
from token_groups import load_cell_groups, set_domain, union  # noqa: E402

GRID = 16
GROUPS = ("hazard_corridor", "hazard")
INTERPRETATION_SCOPE = "descriptive; relational organisation and residual scoring; not localisation"
FIELD_DEFS: dict[str, dict[tuple[int, int], float]] = {
    "ped_did": {(1, 1): 1.0, (1, 0): -1.0, (0, 1): -1.0, (0, 0): 1.0},
    "cone_did": {(OBJECT_HAZARD, 1): 1.0, (OBJECT_HAZARD, 0): -1.0, (0, 1): -1.0, (0, 0): 1.0},
    "null_did": {(NULL_CONTROL_HAZARD, 1): 1.0, (NULL_CONTROL_HAZARD, 0): -1.0, (0, 1): -1.0, (0, 0): 1.0},
    "ped_did_relational": {(1, 1): 1.0, (1, 0): -1.0, (NULL_CONTROL_HAZARD, 1): -1.0, (NULL_CONTROL_HAZARD, 0): 1.0},
    "cone_did_relational": {(OBJECT_HAZARD, 1): 1.0, (OBJECT_HAZARD, 0): -1.0, (NULL_CONTROL_HAZARD, 1): -1.0, (NULL_CONTROL_HAZARD, 0): 1.0},
    "action": {(0, 1): 1.0, (0, 0): -1.0},
}
FIELD_TEXT = {
    "ped_did": "(P(1,1)-P(1,0))-(P(0,1)-P(0,0))",
    "cone_did": "(P(3,1)-P(3,0))-(P(0,1)-P(0,0))",
    "null_did": "(P(2,1)-P(2,0))-(P(0,1)-P(0,0))",
    "ped_did_relational": "ped_did - null_did",
    "cone_did_relational": "cone_did - null_did",
    "action": "P(0,1)-P(0,0)",
}
# (source_kind, source_arm) -> (target_kind, target_arm), same field
TRANSPORT_BLOCKS = {
    "true_cross_arm": [(("true", "A"), ("true", "B")), (("true", "B"), ("true", "A"))],
    "model_cross_arm": [(("model", "A"), ("model", "B")), (("model", "B"), ("model", "A"))],
    "model_to_truth_within_arm": [(("model", "A"), ("true", "A")), (("model", "B"), ("true", "B"))],
    "model_to_truth_cross_arm": [(("model", "A"), ("true", "B")), (("model", "B"), ("true", "A"))],
}
# identity swap: source (arm A pedestrian) -> target arm B cone (swapped) vs arm B pedestrian (matched), and the reverse
IDENTITY_SWAPS = [
    ("A", "ped_did", "B", "cone_did", "ped_did"),
    ("B", "cone_did", "A", "ped_did", "cone_did"),
    ("A", "ped_did_relational", "B", "cone_did_relational", "ped_did_relational"),
    ("B", "cone_did_relational", "A", "ped_did_relational", "cone_did_relational"),
]
COVARIATES = {
    "hazard_pixels": ("target_visible_pixels", "visible hazard pixels in the context frame (in-lane pedestrian cell)"),
    "hazard_patches": ("silhouette_patches", "DINOv3 patches covered by the hazard silhouette"),
    "hazard_bbox_area_px": (None, "bbox_wh[0] * bbox_wh[1]"),
    "lane_offset_m": ("context_lateral_m", "hazard lateral offset from the ego-lane centreline at the context frame"),
    "ego_speed_mps": ("context_speed_mps", "ego speed at the context frame"),
    "prefix_travel_m": ("prefix_travel_m", "ego travel during the action prefix"),
}
MIN_DISTINCT_COVARIATE = 5


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_rows(eval_dir: Path, seeds_file: Path | None = None) -> list[dict[str, Any]]:
    rows = [json.loads(ln) for ln in (eval_dir / "manifest.jsonl").read_text().splitlines() if ln.strip()]
    if seeds_file is not None and seeds_file.exists():
        keep = {int(s) for s in seeds_file.read_text().split()}
        rows = [r for r in rows if int(r["seed"]) in keep]
    return rows


def slim_cache(src: Path, dst: Path) -> dict[str, Any]:
    """Keep only what this analysis reads: last-frame target / prediction / token mask, pooled context, cell ids."""

    with np.load(src, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        cell_id = z["cell_id"]
        target = z["target"]
        n_steps = int(target.shape[1])
        target_last = np.ascontiguousarray(target[:, -1]).astype(np.float16)
        del target
        prediction_last = np.ascontiguousarray(z["prediction"][:, -1]).astype(np.float16)
        token_mask_last = np.ascontiguousarray(z["token_mask"][:, -1]).astype(bool)
        has_mask = z["has_mask"]
        ctx = z["context"]
        context_pooled = ctx.reshape(len(ctx), -1, ctx.shape[-1]).astype(np.float32).mean(1)
    meta = dict(meta, slim="last predicted frame only (relational_transport.py --slim)", n_steps=n_steps, source_cache_sha256=sha256_file(src))
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, meta=json.dumps(meta), cell_id=cell_id, target_last=target_last, prediction_last=prediction_last,
                        token_mask_last=token_mask_last, has_mask=has_mask, context_pooled=context_pooled, n_steps=np.asarray(n_steps))
    return meta


def load_latents(path: Path) -> dict[str, Any]:
    """Full ``latent_cache.py`` npz or the slim variant -> last-frame arrays (float16 grids) + pooled context."""

    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        if "target_last" in z.files:
            out = {k: z[k] for k in ("cell_id", "target_last", "prediction_last", "token_mask_last", "has_mask", "context_pooled")}
            out["n_steps"] = int(z["n_steps"])
        else:
            target = z["target"]
            out = {"cell_id": z["cell_id"], "n_steps": int(target.shape[1]), "target_last": np.ascontiguousarray(target[:, -1])}
            del target
            out["prediction_last"] = np.ascontiguousarray(z["prediction"][:, -1])
            out["token_mask_last"] = np.ascontiguousarray(z["token_mask"][:, -1]).astype(bool)
            out["has_mask"] = z["has_mask"]
            ctx = z["context"]
            out["context_pooled"] = ctx.reshape(len(ctx), -1, ctx.shape[-1]).astype(np.float32).mean(1)
    out["meta"] = meta
    return out


def restrict(latents: dict[str, Any], rows_all: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Cache rows are in manifest order; keep the rows in ``rows`` (a subset, same order)."""

    if list(latents["cell_id"]) != [r["cell_id"] for r in rows_all]:
        raise SystemExit("cache cell order does not match manifest")
    keep = {r["cell_id"] for r in rows}
    idx = np.asarray([i for i, r in enumerate(rows_all) if r["cell_id"] in keep], dtype=int)
    out = {k: (v[idx] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(rows_all) else v) for k, v in latents.items()}
    return out


# --------------------------------------------------------------------------- #
# Scenes, tokens, fields
# --------------------------------------------------------------------------- #


def scene_cells(rows: list[dict[str, Any]]) -> dict[str, dict[tuple[int, int], int]]:
    grouped: dict[str, dict[tuple[int, int], int]] = {}
    for i, r in enumerate(rows):
        grouped.setdefault(r["pair_id"], {})[(int(r["hazard"]), int(r["candidate_action"]))] = i
    return grouped


def scene_tokens(rows: list[dict[str, Any]], cells: dict[str, dict[tuple[int, int], int]], eval_dir: Path | None, mask_last: np.ndarray, n_steps: int, group: str) -> tuple[dict[str, np.ndarray], str]:
    """Union of the group's token indices over the scene's cells at the endpoint frame (frame n_steps)."""

    out: dict[str, np.ndarray] = {}
    if eval_dir is not None and (Path(eval_dir) / "masks").exists():
        for pair, cc in cells.items():
            cgs = [load_cell_groups(Path(eval_dir), rows[i], n_frames=n_steps + 1, domain="driving") for i in cc.values()]
            out[pair] = union(*[cg.group(n_steps, group) for cg in cgs])
        return out, "masks"
    if group != "hazard_corridor":
        return {}, "unavailable_without_masks"
    for pair, cc in cells.items():
        m = np.any(np.stack([mask_last[i] for i in cc.values()]), axis=0)
        out[pair] = np.flatnonzero(m.reshape(-1)).astype(np.int64)
    return out, "cache_token_mask"


def field_vector(y: np.ndarray, cc: dict[tuple[int, int], int], tokens: np.ndarray, coef: dict[tuple[int, int], float], pool: bool = True) -> np.ndarray | None:
    """Field contrast of one scene on ``tokens``: pooled (mean over the tokens -> [D], the representation used for every
    cross-scene comparison, as the sonar 'pooled' feature and geometry_cross_arm.py's pooling) or per token ([n_tok * D],
    only comparable within the scene)."""

    if any(k not in cc for k in coef):
        return None
    grid = y.reshape(len(y), GRID * GRID, y.shape[-1])
    acc = np.zeros((len(tokens), y.shape[-1]), dtype=np.float64)
    for k, c in coef.items():
        acc += c * grid[cc[k]][tokens].astype(np.float64)
    return acc.mean(0) if pool else acc.reshape(-1)


def build_fields(y: np.ndarray, cells: dict[str, dict[tuple[int, int], int]], tokens: dict[str, np.ndarray], scenes: list[str], pool: bool = True) -> dict[str, np.ndarray]:
    """{field: [n_scenes, d]} (float64, pooled d = D); scenes lacking a level for a field get NaN rows."""

    out: dict[str, np.ndarray] = {}
    for name, coef in FIELD_DEFS.items():
        vecs = [field_vector(y, cells[p], tokens[p], coef, pool) for p in scenes]
        if all(v is None for v in vecs):
            continue
        d = next(len(v) for v in vecs if v is not None)
        mat = np.full((len(scenes), d), np.nan)
        for i, v in enumerate(vecs):
            if v is not None:
                mat[i] = v
        out[name] = mat
    return out


# --------------------------------------------------------------------------- #
# Small numerics
# --------------------------------------------------------------------------- #


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    den = na * nb
    num = np.einsum("ij,ij->i", a, b)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)


def average_ranks_rows(X: np.ndarray) -> np.ndarray:
    """Average (tie-aware) ranks 1..n along the last axis of [B, n] (vectorised)."""

    X = np.asarray(X, np.float64)
    if X.ndim == 1:
        return average_ranks_rows(X[None])[0]
    B, n = X.shape
    order = np.argsort(X, axis=1, kind="mergesort")
    sx = np.take_along_axis(X, order, axis=1)
    idx = np.broadcast_to(np.arange(n), (B, n))
    eq_prev = np.concatenate([np.zeros((B, 1), bool), sx[:, 1:] == sx[:, :-1]], axis=1)
    eq_next = np.concatenate([sx[:, 1:] == sx[:, :-1], np.zeros((B, 1), bool)], axis=1)
    first = np.maximum.accumulate(np.where(~eq_prev, idx, 0), axis=1)
    last = np.minimum.accumulate(np.where(~eq_next, idx, n - 1)[:, ::-1], axis=1)[:, ::-1]
    r_sorted = 0.5 * (first + last) + 1.0
    ranks = np.empty((B, n), np.float64)
    np.put_along_axis(ranks, order, r_sorted, axis=1)
    return ranks


def average_ranks(x: np.ndarray) -> np.ndarray:
    return average_ranks_rows(np.asarray(x, np.float64)[None])[0]


def _pearson_rows(RX: np.ndarray, RY: np.ndarray) -> np.ndarray:
    rx = RX - RX.mean(1, keepdims=True)
    ry = RY - RY.mean(1, keepdims=True)
    den = np.sqrt(np.einsum("ij,ij->i", rx, rx) * np.einsum("ij,ij->i", ry, ry))
    num = np.einsum("ij,ij->i", rx, ry)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    return float(_pearson_rows(average_ranks(x[ok])[None], average_ranks(y[ok])[None])[0])


def spearman_block(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_boot: int = 2000, n_perm: int = 2000) -> dict[str, Any]:
    """Spearman rho with a scene-resampling bootstrap CI and a permutation p (vectorised, tie-aware ranks)."""

    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    rho = spearman(x, y)
    if n < 3 or not np.isfinite(rho):
        return {"rho": rho, "ci_low": float("nan"), "ci_high": float("nan"), "p_perm": float("nan"), "n": int(n)}
    bi = rng.integers(0, n, size=(n_boot, n))
    boots = _pearson_rows(average_ranks_rows(x[bi]), average_ranks_rows(y[bi]))
    boots = boots[np.isfinite(boots)]
    ry = average_ranks(y)
    perm = np.stack([ry[rng.permutation(n)] for _ in range(n_perm)])
    perms = _pearson_rows(np.broadcast_to(average_ranks(x), (n_perm, n)), perm)
    perms = perms[np.isfinite(perms)]
    p = float((np.sum(np.abs(perms) >= abs(rho) - 1e-12) + 1) / (len(perms) + 1))
    return {"rho": rho, "ci_low": float(np.percentile(boots, 2.5)) if len(boots) else float("nan"), "ci_high": float(np.percentile(boots, 97.5)) if len(boots) else float("nan"), "p_perm": p, "n": int(n)}


def boot_mean(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Vectorised twin of stats_utils.cluster_bootstrap_mean (same RandomState stream, identical numbers)."""

    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return {"mean": float("nan"), "median": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    rng = np.random.RandomState(seed)
    boots = x[rng.randint(0, n, (n_boot, n))].mean(1)
    return {"mean": float(x.mean()), "median": float(np.median(x)), "ci_low": float(np.percentile(boots, 2.5)), "ci_high": float(np.percentile(boots, 97.5)), "n": int(n)}


def pooled(values: np.ndarray, seed: int = 0) -> dict[str, Any]:
    v = np.asarray(values, np.float64)
    return {**boot_mean(v, seed=seed), "sign_flip_p_vs_zero": sign_flip_p(v, seed=seed)}


def loso_mean(T: np.ndarray) -> np.ndarray:
    n = len(T)
    return (T.sum(0, keepdims=True) - T) / max(n - 1, 1)


# --------------------------------------------------------------------------- #
# 1. Field characterisation
# --------------------------------------------------------------------------- #


def characterise_field(X: np.ndarray, rng: np.random.Generator, n_mc: int = 50) -> dict[str, Any]:
    n, d = X.shape
    if n < 3:
        return {"status": "insufficient_scenes", "n_scenes": int(n)}
    Xc = X - X.mean(0, keepdims=True)
    ev = np.linalg.eigvalsh(Xc @ Xc.T)[::-1]
    ev = np.clip(ev, 0.0, None)
    tot = float(ev.sum())
    frac = ev / tot if tot > 0 else np.zeros_like(ev)
    norms = np.linalg.norm(X, axis=1)
    U = X / np.where(norms > 0, norms, 1.0)[:, None]
    m = U.mean(0)
    R = float(np.linalg.norm(m))
    mc = np.empty(n_mc)
    for b in range(n_mc):
        Z = rng.standard_normal((n, d), dtype=np.float32)
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        mc[b] = np.linalg.norm(Z.mean(0))
    mc_mean, mc_sd = float(mc.mean()), float(mc.std(ddof=1)) if n_mc > 1 else float("nan")
    loso = (U.sum(0, keepdims=True) - U) / (n - 1)
    return {
        "n_scenes": int(n),
        "dim": int(d),
        "pc1_variance_fraction": float(frac[0]),
        "top5_variance_fraction": [float(f) for f in frac[:5]],
        "participation_ratio": float(tot**2 / float((ev**2).sum())) if tot > 0 else float("nan"),
        "spherical_variance": 1.0 - R,
        "mean_resultant_length": R,
        "uniform_random_same_dim": {"mean_resultant_length_mean": mc_mean, "sd": mc_sd, "spherical_variance_mean": 1.0 - mc_mean, "n_mc": int(n_mc),
                                    "z": float((R - mc_mean) / mc_sd) if mc_sd and mc_sd > 0 else float("nan"),
                                    "p_mc_ge_observed": float((np.sum(mc >= R - 1e-12) + 1) / (n_mc + 1))},
        "mean_direction_consistency": {"in_sample": float(np.mean(row_cos(U, np.repeat(m[None], n, axis=0)))), "loso": float(np.mean(row_cos(U, loso)))},
        "norm": {"mean": float(norms.mean()), "sd": float(norms.std(ddof=1)), "cv": float(norms.std(ddof=1) / norms.mean()) if norms.mean() > 0 else float("nan")},
    }


# --------------------------------------------------------------------------- #
# 2. Residual cosine (model vs truth, baselines scored identically)
# --------------------------------------------------------------------------- #


def residual_scoring(M: np.ndarray, T: np.ndarray, donors: np.ndarray, scenes: list[str], seed: int = 0) -> dict[str, Any]:
    n = len(T)
    if n < 3:
        return {"status": "insufficient_scenes", "n_scenes": int(n)}
    G = loso_mean(T)
    Tr = T - G
    C = T[donors]
    per = {
        "model": {"raw": row_cos(M, T), "residual": row_cos(M - G, Tr)},
        "copy_delta": {"raw": row_cos(C, T), "residual": row_cos(C - G, Tr)},
        "field_mean": {"raw": row_cos(G, T), "residual": np.zeros(n)},
    }
    out: dict[str, Any] = {"n_scenes": int(n), "scenes": scenes, "donor_scene": [scenes[int(j)] for j in donors], "per_scene": {}, "pooled": {}, "model_minus_baseline": {}}
    for name, d in per.items():
        out["per_scene"][name] = {k: [float(v) for v in vals] for k, vals in d.items()}
        out["pooled"][name] = {k: pooled(vals, seed) for k, vals in d.items()}
    for base in ("copy_delta", "field_mean"):
        out["model_minus_baseline"][base] = {k: {**boot_mean(per["model"][k] - per[base][k], seed=seed), "sign_flip_p": sign_flip_p(per["model"][k] - per[base][k], seed=seed)} for k in ("raw", "residual")}
    out["template_share_of_truth"] = pooled(row_cos(T, G) ** 2, seed)
    out["nmse"] = {"model_raw": float(np.mean(np.sum((M - T) ** 2, 1) / np.sum(T**2, 1))), "model_residual": float(np.mean(np.sum((M - T) ** 2, 1) / np.sum(Tr**2, 1)))}
    out["note"] = "field_mean residual cosine is 0 by construction (the prediction equals the subtracted template)"
    return out


# --------------------------------------------------------------------------- #
# 3. Relational transport in Gram space
# --------------------------------------------------------------------------- #


def transport_weights(S: np.ndarray) -> np.ndarray:
    """W[w, j] = cos(S_w, S_j) / sum_j |cos| over j != w (diagonal 0)."""

    n = len(S)
    norms = np.linalg.norm(S, axis=1)
    Sn = S / np.where(norms > 0, norms, 1.0)[:, None]
    W = Sn @ Sn.T
    np.fill_diagonal(W, 0.0)
    den = np.abs(W).sum(1, keepdims=True)
    return W / np.where(den > 0, den, 1.0)


def centred_transport_weights(S: np.ndarray) -> np.ndarray:
    """Cosines of the LOSO-centred source: for held-out w, cos(S_w - m_w, S_j - m_w) with m_w the mean over j != w
    (Gram space); normalised like ``transport_weights``."""

    n = len(S)
    Gs = S @ S.T
    M = (np.ones((n, n)) - np.eye(n)) / max(n - 1, 1)
    GM = Gs @ M.T  # GM[i, w] = <S_i, m_w>
    mm = np.einsum("ij,ij->i", M @ Gs, M)  # <m_w, m_w>
    inner = Gs - GM.T - np.diag(GM)[:, None] + mm[:, None]  # [w, j] = <S_w - m_w, S_j - m_w>
    nj = np.diag(Gs)[None, :] - 2.0 * GM.T + mm[:, None]  # [w, j] = ||S_j - m_w||^2
    nw = np.diag(inner)  # ||S_w - m_w||^2
    den = np.sqrt(np.clip(nw[:, None] * nj, 0, None))
    W = np.where(den > 0, inner / np.where(den > 0, den, 1.0), 0.0)
    np.fill_diagonal(W, 0.0)
    d = np.abs(W).sum(1, keepdims=True)
    return W / np.where(d > 0, d, 1.0)


def centred_rows(W: np.ndarray) -> np.ndarray:
    """Prediction operator of the centred estimator: T_hat_w = g_w + sum_j W_wj (T_j - g_w) = (M + W - diag(W 1) M) T."""

    n = len(W)
    M = (np.ones((n, n)) - np.eye(n)) / max(n - 1, 1)
    return M + W - (W.sum(1)[:, None]) * M


def permute_offdiag(W: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(W)
    out = np.zeros_like(W)
    for w in range(n):
        idx = np.array([j for j in range(n) if j != w])
        out[w, idx] = W[w, idx][rng.permutation(n - 1)]
    return out


class GramScorer:
    """All transport scores from the target Gram matrix: a prediction row is A T (A [n, n]); residual cosine against
    T_w with the LOSO mean g_w = M T; Gaussian nulls are simulated in the eigen-basis of the Gram (exact)."""

    def __init__(self, T: np.ndarray, dim: int):
        self.n = n = len(T)
        self.dim = int(dim)
        self.G = T @ T.T
        self.M = (np.ones((n, n)) - np.eye(n)) / max(n - 1, 1)
        self.E = np.eye(n)
        self.norm_T = np.sqrt(np.clip(np.diag(self.G), 0, None))
        R = self.E - self.M
        self.rG = R @ self.G  # rows: <T_w - g_w, T_j>
        self.res_norm = np.sqrt(np.clip(np.einsum("ij,ij->i", self.rG, R), 0, None))  # ||T_w - g_w||
        lam, U = np.linalg.eigh(self.G)
        keep = lam > lam.max() * 1e-10 if lam.max() > 0 else np.zeros(n, bool)
        self.B = U[:, keep] * np.sqrt(lam[keep])[None, :]  # T_j coordinates in an orthonormal basis of span(T)
        self.rank = int(keep.sum())
        self.Bg = self.M @ self.B  # LOSO-mean coordinates

    def score_rows(self, A: np.ndarray) -> dict[str, np.ndarray]:
        AG = A @ self.G
        norm_pred = np.sqrt(np.clip(np.einsum("ij,ij->i", AG, A), 0, None))
        raw_num = np.einsum("ij,ij->i", AG, self.E)
        AM = A - self.M
        AMG = AM @ self.G
        res_num = np.einsum("ij,ij->i", AMG, self.E - self.M)
        res_pred = np.sqrt(np.clip(np.einsum("ij,ij->i", AMG, AM), 0, None))
        return {"raw": self._safe(raw_num, norm_pred * self.norm_T), "residual": self._safe(res_num, res_pred * self.res_norm), "norm_pred": norm_pred}

    def score_coords(self, coords: np.ndarray, norm_sq: np.ndarray) -> dict[str, np.ndarray]:
        """Prediction x_w given by its coordinates in the span basis (coords [n, r]) plus its full squared norm
        (may exceed the in-span part for isotropic vectors); returns raw / residual cosine against T_w."""

        dot_T = np.einsum("ir,ir->i", coords, self.B)  # <x_w, T_w>
        dot_g = np.einsum("ir,ir->i", coords, self.Bg)  # <x_w, g_w>
        g_sq = np.einsum("ir,ir->i", self.Bg, self.Bg)
        gT = np.einsum("ir,ir->i", self.Bg, self.B)
        res_num = dot_T - dot_g - gT + g_sq
        res_pred = np.sqrt(np.clip(norm_sq - 2 * dot_g + g_sq, 0, None))
        return {"raw": self._safe(dot_T, np.sqrt(norm_sq) * self.norm_T), "residual": self._safe(res_num, res_pred * self.res_norm), "norm_pred": np.sqrt(norm_sq)}

    @staticmethod
    def _safe(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)

    # nulls -------------------------------------------------------------- #
    def null_isotropic(self, norm_pred: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
        c = rng.standard_normal((self.n, self.rank))
        extra = rng.chisquare(max(self.dim - self.rank, 1), size=self.n) if self.dim > self.rank else np.zeros(self.n)
        z_sq = np.einsum("ir,ir->i", c, c) + extra
        scale = norm_pred / np.sqrt(z_sq)
        return self.score_coords(c * scale[:, None], norm_pred**2)

    def null_cov_matched(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        n = self.n
        c = rng.standard_normal((n, n))
        np.fill_diagonal(c, 0.0)
        s = np.sqrt(max(n - 2, 1))
        A = c / s
        off = (1.0 - c.sum(1) / s) / max(n - 1, 1)
        A += off[:, None] * (1.0 - np.eye(n))
        return self.score_rows(A)

    def null_rotated_target(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        u = rng.standard_normal((self.n, self.rank))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        b_norm = np.linalg.norm(self.B, axis=1)
        return self.score_coords(u * b_norm[:, None], b_norm**2)


def _transport_variant(scorer: GramScorer, W: np.ndarray, rows_of: Any, rng: np.random.Generator, n_perm: int, n_draws: int, seed: int) -> dict[str, Any]:
    n = scorer.n
    obs = scorer.score_rows(rows_of(W))
    mag = spearman(obs["norm_pred"], scorer.norm_T)
    out: dict[str, Any] = {
        "observed": {"raw_cosine": pooled(obs["raw"], seed), "residual_cosine": pooled(obs["residual"], seed),
                     "magnitude_spearman": spearman_block(obs["norm_pred"], scorer.norm_T, rng, n_boot=1000, n_perm=1000),
                     "per_scene": {"raw": obs["raw"].tolist(), "residual": obs["residual"].tolist(), "norm_pred": obs["norm_pred"].tolist(), "norm_true": scorer.norm_T.tolist()}},
        "weights": {"mean_abs": float(np.abs(W).sum(1).mean() / (n - 1)), "positive_fraction": float((W > 0).sum() / (n * (n - 1))), "max_abs_row_weight_mean": float(np.abs(W).max(1).mean()),
                    "signed_sum_mean": float(W.sum(1).mean())},
        "nulls": {},
    }
    draws = {"scene_permutation": n_perm, "isotropic_gaussian": n_draws, "covariance_matched_gaussian": n_draws, "rotated_real_target": n_draws}
    for name, k in draws.items():
        res = np.empty((k, n))
        raw = np.empty((k, n))
        mags = np.full(k, np.nan)
        for b in range(k):
            if name == "scene_permutation":
                sc = scorer.score_rows(rows_of(permute_offdiag(W, rng)))
                mags[b] = spearman(sc["norm_pred"], scorer.norm_T)
            elif name == "isotropic_gaussian":
                sc = scorer.null_isotropic(obs["norm_pred"], rng)
            elif name == "covariance_matched_gaussian":
                sc = scorer.null_cov_matched(rng)
                mags[b] = spearman(sc["norm_pred"], scorer.norm_T)
            else:
                sc = scorer.null_rotated_target(rng)
            res[b], raw[b] = sc["residual"], sc["raw"]
        entry: dict[str, Any] = {"n_draws": int(k)}
        for key, arr, o in (("residual_cosine", res, obs["residual"]), ("raw_cosine", raw, obs["raw"])):
            diff = o - arr.mean(0)
            entry[key] = {"null_mean": float(arr.mean()), "null_sd_of_draw_means": float(arr.mean(1).std(ddof=1)) if k > 1 else float("nan"),
                          "score_minus_null": {**boot_mean(diff, seed=seed), "sign_flip_p": sign_flip_p(diff, seed=seed)},
                          "p_draws_ge_observed": float((np.sum(arr.mean(1) >= o.mean() - 1e-12) + 1) / (k + 1))}
        if np.isfinite(mags).any():
            fm = mags[np.isfinite(mags)]
            entry["magnitude_spearman"] = {"null_mean": float(fm.mean()), "p_draws_ge_observed": float((np.sum(fm >= mag - 1e-12) + 1) / (len(fm) + 1)) if np.isfinite(mag) else float("nan")}
        else:
            entry["magnitude_spearman"] = {"note": "norm matched by construction; no magnitude null"}
        out["nulls"][name] = entry
    return out


def transport(S: np.ndarray, T: np.ndarray, rng: np.random.Generator, n_perm: int = 100, n_draws: int = 100, seed: int = 0) -> dict[str, Any]:
    """Preregistered signed-kernel transport (top-level ``observed`` / ``nulls``) plus the LOSO-centred variant
    (``centred_variant``: weights from the centred source, prediction g_w + sum_j s_j (T_j - g_w) / sum |s_j|), which
    does not lose the template when the signed weights sum to ~0."""

    n = len(S)
    if n < 4:
        return {"status": "insufficient_scenes", "n_scenes": int(n)}
    scorer = GramScorer(T, T.shape[1])
    out: dict[str, Any] = {"n_scenes": int(n), "dim": int(T.shape[1]), "target_span_rank": scorer.rank}
    out.update(_transport_variant(scorer, transport_weights(S), lambda W: W, rng, n_perm, n_draws, seed))
    out["centred_variant"] = _transport_variant(scorer, centred_transport_weights(S), centred_rows, rng, n_perm, n_draws, seed)
    return out


def valid_rows(*mats: np.ndarray) -> np.ndarray:
    ok = np.ones(len(mats[0]), bool)
    for m in mats:
        ok &= np.isfinite(m).all(1)
    return ok


# --------------------------------------------------------------------------- #
# 4. Covariates
# --------------------------------------------------------------------------- #


def scene_covariates(rows: list[dict[str, Any]], cells: dict[str, dict[tuple[int, int], int]], scenes: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, (key, desc) in COVARIATES.items():
        vals = []
        for p in scenes:
            cc = cells[p]
            i = cc.get((1, 1), cc.get((1, 0), next(iter(cc.values()))))
            r = rows[i]
            if name == "hazard_bbox_area_px":
                wh = r.get("bbox_wh")
                vals.append(float(wh[0]) * float(wh[1]) if wh else np.nan)
            else:
                v = r.get(key)
                vals.append(float(v) if v is not None else np.nan)
        arr = np.asarray(vals, np.float64)
        fin = arr[np.isfinite(arr)]
        n_distinct = int(len(np.unique(np.round(fin, 9)))) if len(fin) else 0
        out[name] = {"values": arr, "description": desc, "manifest_key": key or "bbox_wh", "n_distinct": n_distinct,
                     "range": [float(fin.min()), float(fin.max())] if len(fin) else None, "degenerate": n_distinct < MIN_DISTINCT_COVARIATE}
    return out


def heterogeneity(X: np.ndarray, covs: dict[str, dict[str, Any]], rng: np.random.Generator) -> dict[str, Any]:
    ok = np.isfinite(X).all(1)
    if ok.sum() < 5:
        return {"status": "insufficient_scenes"}
    Xv = X[ok]
    G = loso_mean(Xv)
    mag = np.linalg.norm(Xv, axis=1)
    dev = 1.0 - row_cos(Xv, G)
    out: dict[str, Any] = {"n_scenes": int(ok.sum()), "covariates": {}}
    for name, c in covs.items():
        v = c["values"][ok]
        out["covariates"][name] = {"degenerate": bool(c["degenerate"]), "n_distinct": c["n_distinct"], "range": c["range"], "description": c["description"],
                                   "magnitude": spearman_block(v, mag, rng, n_boot=1000, n_perm=1000), "directional_deviation": spearman_block(v, dev, rng, n_boot=1000, n_perm=1000)}
    return out


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def analyze(arms: dict[str, dict[str, Any]], groups: tuple[str, ...] = GROUPS, n_perm: int = 100, n_draws: int = 100, n_mc: int = 50, seed: int = 0, tokens_override: dict[str, dict[str, np.ndarray]] | None = None) -> dict[str, Any]:
    """``arms[arm] = {"rows": [...], "latents": load_latents(...), "eval_dir": Path | None}``."""

    set_domain("driving")
    rng = np.random.default_rng(seed)
    t0 = time.time()
    result: dict[str, Any] = {
        "interpretation_scope": INTERPRETATION_SCOPE,
        "definitions": {"fields": FIELD_TEXT, "frame": "last predicted frame (endpoint)", "token_groups": list(groups),
                        "representation": "per scene: mean over the group's tokens of the contrast -> one D-vector (sonar 'pooled' feature; geometry_cross_arm.py pooling); scenes have different token sets, so per-token vectors are not cross-scene comparable (per-token raw cosine kept as a diagnostic)",
                        "transport_centred_variant": "weights cos(S_w - m_w, S_j - m_w) on the LOSO-centred source; prediction g_w + sum_j s_j (T_j - g_w)/sum|s_j| (no template loss when signed weights cancel)",
                        "token_union": "group tokens unioned over the scene's cells and over both arms (identical token set per scene in A and B)",
                        "residual_cosine": "cos(x - g_-w, T_w - g_-w), g_-w = mean of the TRUE field over the other discovery scenes",
                        "transport": "T_hat_w = sum_{j!=w} cos(S_w,S_j) T_j / sum_j |cos(S_w,S_j)|; scored by residual cosine (LOSO mean of T), raw cosine and Spearman(|T_hat_w|, |T_w|)",
                        "nulls": {"scene_permutation": "per held-out scene the weight vector is permuted over the other scenes (marginals kept)",
                                  "isotropic_gaussian": "N(0, I_d) direction at the norm of T_hat_w (exact simulation in the target span + chi-square complement)",
                                  "covariance_matched_gaussian": "g_-w + sum_j c_j (T_j - g_-w)/sqrt(n-2), c ~ N(0, I): the LOSO mean and covariance of the target field",
                                  "rotated_real_target": "T_w rotated by a Haar-random orthogonal transform within span(T) (norm preserved)"},
                        "copy_delta_baseline": "true field of the nearest OTHER scene by pooled context latent (L2), LOSO-mean-subtracted like the model",
                        "field_mean_baseline": "the LOSO mean of the true field itself"},
        "n_perm": int(n_perm), "n_draws": int(n_draws), "n_mc_spherical": int(n_mc), "seed": int(seed),
        "arms": {}, "groups": {},
    }
    cells = {a: scene_cells(v["rows"]) for a, v in arms.items()}
    common = sorted(set.intersection(*[set(c) for c in cells.values()]))
    common = [p for p in common if all((0, 0) in cells[a][p] and (0, 1) in cells[a][p] for a in arms)]
    for a, v in arms.items():
        lat = v["latents"]
        result["arms"][a] = {"n_cells": int(len(v["rows"])), "n_scenes": int(len(cells[a])), "cache_meta": lat.get("meta"), "eval_dir": str(v.get("eval_dir")) if v.get("eval_dir") else None,
                             "levels_present": sorted({int(r["hazard"]) for r in v["rows"]})}
    ctx = {a: v["latents"]["context_pooled"].astype(np.float64) for a, v in arms.items()}
    for g in groups:
        tok_by_arm = {}
        sources = {}
        for a, v in arms.items():
            if tokens_override and g in tokens_override:
                tok_by_arm[a], sources[a] = tokens_override[g], "override"
            else:
                tok_by_arm[a], sources[a] = scene_tokens(v["rows"], cells[a], v.get("eval_dir"), v["latents"]["token_mask_last"], v["latents"]["n_steps"], g)
        if any(len(t) == 0 for t in tok_by_arm.values()):
            result["groups"][g] = {"status": "unavailable", "token_source": sources}
            continue
        tokens = {p: union(*[tok_by_arm[a][p] for a in arms]) for p in common}
        scenes = [p for p in common if len(tokens[p]) > 0]
        n = len(scenes)
        block: dict[str, Any] = {"n_scenes": n, "scenes": scenes, "token_source": sources, "n_tokens_per_scene": {p: int(len(tokens[p])) for p in scenes},
                                 "field_characterisation": {}, "residual_cosine": {}, "transport": {}, "identity_swap": {}, "covariate_heterogeneity": {}}
        if n < 4:
            block["status"] = "insufficient_scenes"
            result["groups"][g] = block
            continue
        # fields per (kind, arm) -> {field: [n, d]}; tokens unioned across arms so d matches per scene
        F: dict[tuple[str, str], dict[str, np.ndarray]] = {}
        for a, v in arms.items():
            F[("model", a)] = build_fields(v["latents"]["prediction_last"], cells[a], tokens, scenes)
            F[("true", a)] = build_fields(v["latents"]["target_last"], cells[a], tokens, scenes)
        block["fields_available"] = {f"{k}_{a}": sorted(d) for (k, a), d in F.items()}
        # donors (copy-delta): nearest other scene by pooled context latent, per arm
        donors = {}
        for a in arms:
            sc_ctx = np.stack([np.mean([ctx[a][i] for i in cells[a][p].values()], axis=0) for p in scenes])
            d2 = ((sc_ctx[:, None, :] - sc_ctx[None, :, :]) ** 2).sum(-1)
            np.fill_diagonal(d2, np.inf)
            donors[a] = d2.argmin(1)
        covs = {a: scene_covariates(v["rows"], cells[a], scenes) for a, v in arms.items()}
        # 1 + 2 + 4 per arm / field
        for a in arms:
            v_lat = arms[a]["latents"]
            block["field_characterisation"][a] = {}
            block["residual_cosine"][a] = {}
            block["covariate_heterogeneity"][a] = {}
            for f in FIELD_DEFS:
                if f not in F[("true", a)] or f not in F[("model", a)]:
                    continue
                Tm, Mm = F[("true", a)][f], F[("model", a)][f]
                ok = valid_rows(Tm, Mm)
                if ok.sum() < 4:
                    continue
                idx = np.flatnonzero(ok)
                remap = {int(i): k for k, i in enumerate(idx)}
                dn = np.asarray([remap.get(int(donors[a][i]), -1) for i in idx])
                if (dn < 0).any():  # donor dropped for this field: fall back to nearest valid scene
                    sc_ctx = np.stack([np.mean([ctx[a][i] for i in cells[a][scenes[j]].values()], axis=0) for j in idx])
                    d2 = ((sc_ctx[:, None, :] - sc_ctx[None, :, :]) ** 2).sum(-1)
                    np.fill_diagonal(d2, np.inf)
                    dn = d2.argmin(1)
                sub = [scenes[i] for i in idx]
                block["field_characterisation"][a][f] = {"true": characterise_field(Tm[ok], rng, n_mc), "model": characterise_field(Mm[ok], rng, n_mc)}
                block["residual_cosine"][a][f] = residual_scoring(Mm[ok], Tm[ok], dn, sub, seed)
                ptc = np.asarray([row_cos(field_vector(v_lat["prediction_last"], cells[a][scenes[i]], tokens[scenes[i]], FIELD_DEFS[f], pool=False)[None],
                                          field_vector(v_lat["target_last"], cells[a][scenes[i]], tokens[scenes[i]], FIELD_DEFS[f], pool=False)[None])[0] for i in idx])
                block["residual_cosine"][a][f]["per_token_raw_cosine"] = {**pooled(ptc, seed), "note": "cos over the scene's own [n_tok x D] contrast (the gate's interaction_cosine currency); not cross-scene comparable"}
                block["covariate_heterogeneity"][a][f] = {"true": heterogeneity(Tm, covs[a], rng), "model": heterogeneity(Mm, covs[a], rng)}
        # 3 transport, same field
        for name, pairs in TRANSPORT_BLOCKS.items():
            block["transport"][name] = {}
            for (sk, sa), (tk, ta) in pairs:
                key = f"{sk}_{sa}->{tk}_{ta}"
                block["transport"][name][key] = {}
                for f in FIELD_DEFS:
                    if f not in F[(sk, sa)] or f not in F[(tk, ta)]:
                        continue
                    S, T = F[(sk, sa)][f], F[(tk, ta)][f]
                    ok = valid_rows(S, T)
                    if ok.sum() < 4:
                        continue
                    block["transport"][name][key][f] = transport(S[ok], T[ok], rng, n_perm, n_draws, seed)
        # identity swap
        for kind in ("true", "model"):
            for sa, sf, ta, tf_swapped, tf_matched in IDENTITY_SWAPS:
                if sf not in F[(kind, sa)] or tf_swapped not in F[(kind, ta)] or tf_matched not in F[(kind, ta)]:
                    continue
                S = F[(kind, sa)][sf]
                Tsw, Tma = F[(kind, ta)][tf_swapped], F[(kind, ta)][tf_matched]
                ok = valid_rows(S, Tsw, Tma)
                if ok.sum() < 4:
                    continue
                rs = transport(S[ok], Tsw[ok], rng, n_perm, n_draws, seed)
                rm = transport(S[ok], Tma[ok], rng, n_perm, n_draws, seed)
                diff = np.asarray(rs["observed"]["per_scene"]["residual"]) - np.asarray(rm["observed"]["per_scene"]["residual"])
                diff_c = np.asarray(rs["centred_variant"]["observed"]["per_scene"]["residual"]) - np.asarray(rm["centred_variant"]["observed"]["per_scene"]["residual"])
                block["identity_swap"][f"{kind}: {sa}.{sf} -> {ta}.{tf_swapped} (swapped) vs {ta}.{tf_matched} (matched)"] = {
                    "swapped": rs, "matched": rm,
                    "swapped_minus_matched_residual": {**boot_mean(diff, seed=seed), "sign_flip_p": sign_flip_p(diff, seed=seed)},
                    "swapped_minus_matched_residual_centred": {**boot_mean(diff_c, seed=seed), "sign_flip_p": sign_flip_p(diff_c, seed=seed)},
                }
        result["groups"][g] = block
    result["elapsed_s"] = time.time() - t0
    result["summary"] = summarize(result)
    return result


def _obs(tr: dict[str, Any] | None) -> dict[str, Any] | None:
    if not tr or "observed" not in tr:
        return None
    nulls = {k: {"score_minus_null": v["residual_cosine"]["score_minus_null"]["mean"], "ci": [v["residual_cosine"]["score_minus_null"]["ci_low"], v["residual_cosine"]["score_minus_null"]["ci_high"]],
                 "p": v["residual_cosine"]["score_minus_null"]["sign_flip_p"]} for k, v in tr["nulls"].items()}
    c = tr.get("centred_variant", {})
    cn = {k: {"score_minus_null": v["residual_cosine"]["score_minus_null"]["mean"], "p": v["residual_cosine"]["score_minus_null"]["sign_flip_p"]} for k, v in c.get("nulls", {}).items()}
    return {"residual_cosine": tr["observed"]["residual_cosine"]["mean"], "residual_ci": [tr["observed"]["residual_cosine"]["ci_low"], tr["observed"]["residual_cosine"]["ci_high"]],
            "raw_cosine": tr["observed"]["raw_cosine"]["mean"], "magnitude_spearman": tr["observed"]["magnitude_spearman"]["rho"], "nulls": nulls,
            "centred_residual_cosine": c.get("observed", {}).get("residual_cosine", {}).get("mean"), "centred_nulls": cn}


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for g, block in result["groups"].items():
        if "field_characterisation" not in block:
            out[g] = {"status": block.get("status")}
            continue
        s: dict[str, Any] = {"n_scenes": block["n_scenes"], "field": {}, "residual_cosine": {}, "transport": {}, "identity_swap": {}}
        for a, fields in block["field_characterisation"].items():
            for f, kinds in fields.items():
                s["field"][f"{a}.{f}"] = {k: {"pc1": v.get("pc1_variance_fraction"), "spherical_variance": v.get("spherical_variance"),
                                              "uniform_spherical_variance": v.get("uniform_random_same_dim", {}).get("spherical_variance_mean"), "z_vs_uniform": v.get("uniform_random_same_dim", {}).get("z"),
                                              "mean_direction_consistency_loso": v.get("mean_direction_consistency", {}).get("loso")} for k, v in kinds.items()}
        for a, fields in block["residual_cosine"].items():
            for f, r in fields.items():
                if "pooled" not in r:
                    continue
                s["residual_cosine"][f"{a}.{f}"] = {n: {"raw": r["pooled"][n]["raw"]["mean"], "residual": r["pooled"][n]["residual"]["mean"]} for n in ("model", "copy_delta", "field_mean")}
                s["residual_cosine"][f"{a}.{f}"]["model_minus_copy_delta_residual"] = {"mean": r["model_minus_baseline"]["copy_delta"]["residual"]["mean"], "p": r["model_minus_baseline"]["copy_delta"]["residual"]["sign_flip_p"]}
                s["residual_cosine"][f"{a}.{f}"]["model_minus_copy_delta_raw"] = {"mean": r["model_minus_baseline"]["copy_delta"]["raw"]["mean"], "p": r["model_minus_baseline"]["copy_delta"]["raw"]["sign_flip_p"]}
        for name, cfgs in block["transport"].items():
            s["transport"][name] = {f"{k}.{f}": _obs(tr) for k, fields in cfgs.items() for f, tr in fields.items()}
        for k, v in block["identity_swap"].items():
            s["identity_swap"][k] = {"swapped": _obs(v["swapped"]), "matched": _obs(v["matched"]), "swapped_minus_matched": v["swapped_minus_matched_residual"]}
        out[g] = s
    return out


def fmt(x: Any, nd: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.{nd}f}" if isinstance(x, (int, float)) else str(x)


def markdown_summary(result: dict[str, Any]) -> str:
    lines = ["# Relational transport / residual scoring (descriptive)", "", f"Scope: {result['interpretation_scope']}.", ""]
    for g, s in result["summary"].items():
        lines.append(f"## Token group `{g}`")
        if "n_scenes" not in s:
            lines += [f"status: {s.get('status')}", ""]
            continue
        lines.append(f"n = {s['n_scenes']} discovery scenes; nulls: {result['n_perm']} permutations / {result['n_draws']} draws.")
        lines += ["", "### Field characterisation (true | model)", "", "| arm.field | PC1 true | PC1 model | sph.var true (uniform) | z true | sph.var model | z model | mean-dir LOSO true/model |", "|---|---|---|---|---|---|---|---|"]
        for k, v in s["field"].items():
            t, m = v["true"], v["model"]
            lines.append(f"| {k} | {fmt(t['pc1'])} | {fmt(m['pc1'])} | {fmt(t['spherical_variance'])} ({fmt(t['uniform_spherical_variance'])}) | {fmt(t['z_vs_uniform'], 1)} | {fmt(m['spherical_variance'])} | {fmt(m['z_vs_uniform'], 1)} | {fmt(t['mean_direction_consistency_loso'])} / {fmt(m['mean_direction_consistency_loso'])} |")
        lines += ["", "### Residual cosine, model vs truth (LOSO true-field mean subtracted)", "", "| arm.field | model raw | model residual | copy-delta raw | copy-delta residual | field-mean raw | model - copy residual (p) | model - copy raw (p) |", "|---|---|---|---|---|---|---|---|"]
        for k, v in s["residual_cosine"].items():
            lines.append(f"| {k} | {fmt(v['model']['raw'])} | {fmt(v['model']['residual'])} | {fmt(v['copy_delta']['raw'])} | {fmt(v['copy_delta']['residual'])} | {fmt(v['field_mean']['raw'])} | {fmt(v['model_minus_copy_delta_residual']['mean'])} ({fmt(v['model_minus_copy_delta_residual']['p'])}) | {fmt(v['model_minus_copy_delta_raw']['mean'])} ({fmt(v['model_minus_copy_delta_raw']['p'])}) |")
        lines += ["", "### Relational transport (residual cosine; score - null, sign-flip p; 'centred' = LOSO-centred variant vs its permutation null)", "", "| block | config.field | residual [CI] | raw | |mag| rho | - perm (p) | - iso Gauss (p) | - cov Gauss (p) | - rotated (p) | centred residual, - perm (p) |", "|---|---|---|---|---|---|---|---|---|---|"]
        for name, cfgs in s["transport"].items():
            for k, o in cfgs.items():
                if o is None:
                    continue
                nl = o["nulls"]
                cells = [f"{fmt(nl[q]['score_minus_null'])} ({fmt(nl[q]['p'])})" if q in nl else "n/a" for q in ("scene_permutation", "isotropic_gaussian", "covariance_matched_gaussian", "rotated_real_target")]
                cp = o.get("centred_nulls", {}).get("scene_permutation", {})
                lines.append(f"| {name} | {k} | {fmt(o['residual_cosine'])} [{fmt(o['residual_ci'][0])}, {fmt(o['residual_ci'][1])}] | {fmt(o['raw_cosine'])} | {fmt(o['magnitude_spearman'])} | " + " | ".join(cells) + f" | {fmt(o.get('centred_residual_cosine'))}, {fmt(cp.get('score_minus_null'))} ({fmt(cp.get('p'))}) |")
        if s["identity_swap"]:
            lines += ["", "### Identity swap (A pedestrian relations -> B cone field vs B pedestrian field)", "", "| config | swapped residual | matched residual | swapped - matched [CI] (p) |", "|---|---|---|---|"]
            for k, v in s["identity_swap"].items():
                d = v["swapped_minus_matched"]
                lines.append(f"| {k} | {fmt(v['swapped']['residual_cosine'] if v['swapped'] else None)} | {fmt(v['matched']['residual_cosine'] if v['matched'] else None)} | {fmt(d['mean'])} [{fmt(d['ci_low'])}, {fmt(d['ci_high'])}] ({fmt(d['sign_flip_p'])}) |")
        cov = result["groups"][g].get("covariate_heterogeneity", {})
        rows_c = []
        for a, fields in cov.items():
            for f, kinds in fields.items():
                for kind, h in kinds.items():
                    for cname, c in h.get("covariates", {}).items():
                        rows_c.append(f"| {a}.{f} | {kind} | {cname}{' (degenerate)' if c['degenerate'] else ''} | {fmt(c['magnitude']['rho'])} [{fmt(c['magnitude']['ci_low'])}, {fmt(c['magnitude']['ci_high'])}] | {fmt(c['directional_deviation']['rho'])} [{fmt(c['directional_deviation']['ci_low'])}, {fmt(c['directional_deviation']['ci_high'])}] |")
        if rows_c:
            lines += ["", "### Covariate heterogeneity (Spearman; 'degenerate' = < 5 distinct values, stereotyped stimulus)", "", "| arm.field | source | covariate | rho magnitude [CI] | rho directional deviation [CI] |", "|---|---|---|---|---|"] + rows_c
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Synthetic planted self-test
# --------------------------------------------------------------------------- #


def make_synthetic(rng: np.random.Generator, n_scenes: int = 24, D: int = 8, n_tok: int = 12, k: int = 3, transports: bool = True, template: float = 2.0, model_gain: float = 0.7, noise: float = 0.3) -> dict[str, dict[str, Any]]:
    """Two arms of cells on a 16x16 grid with D channels. Scene code c_w is shared by both arms' consequence fields
    (arm A: pedestrian, arm B: cone) when ``transports`` is True, independent otherwise. Appearance / action fields are
    shared across arms (identical renders). Model fields = attenuated truth + noise + template mismatch."""

    d = n_tok * D
    codes = rng.standard_normal((n_scenes, k))
    codes_b = codes if transports else rng.standard_normal((n_scenes, k))
    size = 1.0 + 0.5 * rng.standard_normal(n_scenes)  # scene-specific magnitude, correlates with a manifest covariate
    L = {a: rng.standard_normal((k, d)) * 1.5 for a in "AB"}
    g = {a: rng.standard_normal(d) for a in "AB"}
    g = {a: v / np.linalg.norm(v) * template * 1.5 * np.sqrt(k * d) for a, v in g.items()}  # template = multiple of the scene-structure norm
    K = {name: rng.standard_normal((k, d)) for name in ("ped", "cone", "null", "act")}
    h = {name: rng.standard_normal(d) * 2.0 for name in ("ped", "cone", "null", "act")}
    g_model_bias = {a: rng.standard_normal(d) * 0.5 for a in "AB"}
    arms: dict[str, dict[str, Any]] = {}
    for a in "AB":
        rows, tgt, prd, msk, ctxp = [], [], [], [], []
        for w in range(n_scenes):
            pid = f"syn{w:03d}"
            r0 = 4 + (w % 6)
            c0 = 3 + (w % 5)
            grid = np.zeros((GRID, GRID), bool)
            grid[r0 : r0 + 3, c0 : c0 + 4] = True
            tok = np.flatnonzero(grid.reshape(-1))[:n_tok]
            base = rng.standard_normal((GRID * GRID, D)) * 0.5
            cw, cb = codes[w], codes_b[w]
            app = {n: h[n] + cw @ K[n] for n in ("ped", "cone", "null")}
            act = h["act"] + cw @ K["act"] + noise * rng.standard_normal(d)
            cons_A = g["A"] + size[w] * (cw @ L["A"]) + noise * rng.standard_normal(d)
            cons_B = g["B"] + size[w] * (cb @ L["B"]) + noise * rng.standard_normal(d)
            ghost = 0.3 * noise * rng.standard_normal(d)
            inter = {1: cons_A if a == "A" else ghost, OBJECT_HAZARD: cons_B if a == "B" else 0.3 * noise * rng.standard_normal(d), NULL_CONTROL_HAZARD: 0.3 * noise * rng.standard_normal(d)}
            appl = {0: np.zeros(d), 1: app["ped"], NULL_CONTROL_HAZARD: app["null"], OBJECT_HAZARD: app["cone"]}
            for hz in (0, 1, NULL_CONTROL_HAZARD, OBJECT_HAZARD):
                for ac in (0, 1):
                    true_vec = appl[hz] + ac * (act + (inter[hz] if hz != 0 else 0.0))
                    model_inter = model_gain * inter[hz] + g_model_bias[a] + noise * rng.standard_normal(d) if hz != 0 else 0.0
                    model_vec = appl[hz] + ac * (act + model_inter) + 0.2 * noise * rng.standard_normal(d)
                    t = base.copy()
                    p = base.copy()
                    t[tok] += true_vec.reshape(n_tok, D)
                    p[tok] += model_vec.reshape(n_tok, D)
                    rows.append({"pair_id": pid, "seed": w, "hazard": hz, "candidate_action": ac, "cell_id": f"{pid}__h{hz}a{ac}", "target_visible_pixels": float(1000 + 200 * size[w]),
                                 "silhouette_patches": 12, "bbox_wh": [23, 80], "context_lateral_m": float(rng.standard_normal() * 1e-5), "context_speed_mps": float(8.3 + 0.01 * rng.standard_normal()), "prefix_travel_m": 3.9})
                    tgt.append(t.reshape(GRID, GRID, D).astype(np.float16))
                    prd.append(p.reshape(GRID, GRID, D).astype(np.float16))
                    msk.append(grid)
                    ctxp.append(base.mean(0).astype(np.float32) + 0.1 * w)
        arms[a] = {"rows": rows, "eval_dir": None,
                   "latents": {"cell_id": np.asarray([r["cell_id"] for r in rows]), "target_last": np.stack(tgt), "prediction_last": np.stack(prd), "token_mask_last": np.stack(msk),
                               "has_mask": np.ones(len(rows), bool), "context_pooled": np.stack(ctxp), "n_steps": 3, "meta": {"synthetic": True, "transports": transports}}}
    return arms


def self_test(out_dir: Path | None, n_perm: int = 100, n_draws: int = 100, seed: int = 0) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    res = {}
    for variant, tr in (("planted", True), ("null", False)):
        arms = make_synthetic(np.random.default_rng(seed + (0 if tr else 1)), transports=tr)
        r = analyze(arms, groups=("hazard_corridor",), n_perm=n_perm, n_draws=n_draws, n_mc=20, seed=seed)
        res[variant] = r
        g = r["groups"]["hazard_corridor"]
        sw = g["identity_swap"]["true: A.ped_did -> B.cone_did (swapped) vs B.ped_did (matched)"]
        swapped = sw["swapped"]["nulls"]
        checks[variant] = {
            "swapped_beats_permutation_p": swapped["scene_permutation"]["residual_cosine"]["score_minus_null"]["sign_flip_p"],
            "swapped_beats_isotropic_p": swapped["isotropic_gaussian"]["residual_cosine"]["score_minus_null"]["sign_flip_p"],
            "swapped_beats_cov_p": swapped["covariance_matched_gaussian"]["residual_cosine"]["score_minus_null"]["sign_flip_p"],
            "swapped_beats_rotated_p": swapped["rotated_real_target"]["residual_cosine"]["score_minus_null"]["sign_flip_p"],
            "swapped_residual": sw["swapped"]["observed"]["residual_cosine"]["mean"],
            "matched_residual": sw["matched"]["observed"]["residual_cosine"]["mean"],
            "swapped_minus_matched_p": sw["swapped_minus_matched_residual"]["sign_flip_p"],
            "model_A_ped_residual_minus_copy_p": g["residual_cosine"]["A"]["ped_did"]["model_minus_baseline"]["copy_delta"]["residual"]["sign_flip_p"],
            "model_A_ped_residual_minus_copy": g["residual_cosine"]["A"]["ped_did"]["model_minus_baseline"]["copy_delta"]["residual"]["mean"],
            "copy_raw_A_ped": g["residual_cosine"]["A"]["ped_did"]["pooled"]["copy_delta"]["raw"]["mean"],
            "pc1_true_A_ped": g["field_characterisation"]["A"]["ped_did"]["true"]["pc1_variance_fraction"],
            "z_vs_uniform_true_A_ped": g["field_characterisation"]["A"]["ped_did"]["true"]["uniform_random_same_dim"]["z"],
            "magnitude_vs_pixels_rho": g["covariate_heterogeneity"]["A"]["ped_did"]["true"]["covariates"]["hazard_pixels"]["magnitude"]["rho"],
            "model_A_to_true_A_ped_p": r["groups"]["hazard_corridor"]["transport"]["model_to_truth_within_arm"]["model_A->true_A"]["ped_did"]["nulls"]["scene_permutation"]["residual_cosine"]["score_minus_null"]["sign_flip_p"],
        }
    p, q = checks["planted"], checks["null"]
    verdict = {
        "planted_swapped_transport_beats_all_nulls": all(p[k] < 0.05 for k in ("swapped_beats_permutation_p", "swapped_beats_isotropic_p", "swapped_beats_cov_p", "swapped_beats_rotated_p")),
        "planted_swapped_exceeds_matched": p["swapped_minus_matched_p"] < 0.05 and p["swapped_residual"] > p["matched_residual"],
        "planted_model_beats_copy_delta_on_residual": p["model_A_ped_residual_minus_copy_p"] < 0.05 and p["model_A_ped_residual_minus_copy"] > 0,
        "planted_copy_delta_raw_cosine_high": p["copy_raw_A_ped"] > 0.5,
        "planted_field_not_single_direction": p["pc1_true_A_ped"] < 0.9,
        "planted_field_below_uniform_spherical_variance": p["z_vs_uniform_true_A_ped"] > 3.0,
        "planted_magnitude_tracks_pixels": p["magnitude_vs_pixels_rho"] > 0.3,
        "planted_model_to_truth_transport": p["model_A_to_true_A_ped_p"] < 0.05,
        "null_swapped_transport_not_significant": q["swapped_beats_permutation_p"] > 0.05,
    }
    out = {"passed": all(verdict.values()), "verdict": verdict, "checks": checks, "interpretation_scope": INTERPRETATION_SCOPE}
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "relational_transport_selftest.json").write_text(json.dumps(json.loads(canonical_json(finite(out))), indent=2) + "\n")
        for variant, r in res.items():
            (out_dir / f"relational_transport_synthetic_{variant}.json").write_text(json.dumps(json.loads(canonical_json(finite(r))), indent=2) + "\n")
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--synthetic", action="store_true", help="planted self-test through the full code path")
    ap.add_argument("--slim", type=Path, default=None, help="latent_cache.py npz to slim to the last frame (--slim-out)")
    ap.add_argument("--slim-out", type=Path, default=None)
    ap.add_argument("--eval-a", type=Path, default=None, help="arm A eval dir (manifest.jsonl, masks/, discovery_seeds.txt)")
    ap.add_argument("--eval-b", type=Path, default=None)
    ap.add_argument("--cache-a", type=Path, default=None, help="arm A latent cache (full or slim)")
    ap.add_argument("--cache-b", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="output directory (relational_transport.json + .md)")
    ap.add_argument("--groups", nargs="+", default=list(GROUPS))
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--n-draws", type=int, default=100)
    ap.add_argument("--n-mc", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label-a", default="JEPA-WM driving arm A seed 0")
    ap.add_argument("--label-b", default="JEPA-WM driving arm B seed 0")
    args = ap.parse_args()

    if args.slim is not None:
        if args.slim_out is None:
            raise SystemExit("--slim needs --slim-out")
        meta = slim_cache(args.slim, args.slim_out)
        print(json.dumps({"slim": str(args.slim_out), "n_cells": meta.get("n_cells"), "n_steps": meta.get("n_steps")}))
        return
    if args.synthetic:
        out = self_test(args.out, n_perm=args.n_perm, n_draws=args.n_draws, seed=args.seed)
        print(json.dumps(finite({"passed": out["passed"], "verdict": out["verdict"], "checks": out["checks"]}), indent=2))
        sys.exit(0 if out["passed"] else 1)
    for k in ("eval_a", "eval_b", "cache_a", "cache_b", "out"):
        if getattr(args, k) is None:
            raise SystemExit(f"--{k.replace('_', '-')} is required")
    arms = {}
    for a, ev, ca, label in (("A", args.eval_a, args.cache_a, args.label_a), ("B", args.eval_b, args.cache_b, args.label_b)):
        rows_all = load_rows(ev)
        rows = load_rows(ev, ev / "discovery_seeds.txt")
        lat = restrict(load_latents(ca), rows_all, rows)
        lat["meta"] = dict(lat["meta"], label=label, cache_sha256=sha256_file(ca), manifest_sha256=sha256_file(ev / "manifest.jsonl"))
        arms[a] = {"rows": rows, "latents": lat, "eval_dir": ev}
    if [r["pair_id"] for r in arms["A"]["rows"]] != [r["pair_id"] for r in arms["B"]["rows"]]:
        raise SystemExit("arm A and arm B eval dirs do not hold the same discovery scenes in the same order")
    result = analyze(arms, groups=tuple(args.groups), n_perm=args.n_perm, n_draws=args.n_draws, n_mc=args.n_mc, seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "relational_transport.json").write_text(json.dumps(json.loads(canonical_json(finite(result))), indent=2) + "\n")
    (args.out / "relational_transport.md").write_text(markdown_summary(finite(result)))
    print(json.dumps(finite(result["summary"]), indent=1)[:6000])
    print(f"elapsed {result['elapsed_s']:.0f}s -> {args.out / 'relational_transport.json'}")


if __name__ == "__main__":
    main()
