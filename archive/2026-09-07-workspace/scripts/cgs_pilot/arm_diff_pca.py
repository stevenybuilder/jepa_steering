#!/usr/bin/env python3
"""Arm diffing: PCA of activation differences between the matched-consequence arms (CAFT-inspired, experiment 1).

Concept Ablation Fine-Tuning (Casademunt et al. 2025, arXiv:2507.16795; ``representational_geometry_paper_concepts.md``
section 5) finds candidate concept directions by *model diffing*: PCA of ``delta_h(x) = h_finetuned(x) - h_base(x)`` on
matched inputs, followed by inspection of the maximally / minimally projecting examples.  Our reversed-assignment arms
are the world-model analogue of a base / fine-tuned pair: two trainings from the SAME initialisation, seed and clip
order that differ only in which in-lane identity carries the collision consequence (arm A pedestrian solid / cone ghost,
arm B the reverse).  On the factorial scenes both arms see identical encoder tokens, so at every predictor site the
per-cell difference ``delta_h(x) = h_A(x) - h_B(x)`` is defined on identical frames and actions.

Per site x imagined step x token group (``hazard``, ``corridor``, ``hazard_corridor``), on the discovery scenes shared by
the two localization dumps (``geometry_cross_arm.py`` loaders, alignment, pooling and hazard-free Procrustes transport,
imported, not copied):

1. ``delta_h`` in two coordinate conventions: **transported** (arm B moved into arm A's coordinates with the LOSO
   hazard-free orthogonal Procrustes map of the cell's scene, ``h_B->A = (h_B - m_B) W^T + m_A``; the map is fitted on
   levels 0/2 only, so by construction it minimises the hazard-free part of ``delta_h`` and the interesting question is
   the excess at the in-lane cells) and **raw** (shared ambient coordinates; meaningful here because the arms share the
   initialisation, unlike two independently initialised models).
2. PCA (centred) of ``delta_h`` over scenes x cells for four cell subsets: ``all`` (8 cells per scene), ``in_lane``
   (levels 1 and 3), ``hazard_free`` (levels 0 and 2) and ``did`` (per scene the arm difference of the two identity DiD
   fields, ``DiD_ped^A - DiD_ped^B`` and ``DiD_cone^A - DiD_cone^B``; a shared template cancels there).  Spectrum:
   PC1 variance ratio, participation ratio ``(sum lam)^2 / sum lam^2``, 90 %-energy rank, and the split between the mean
   shift and the variance around it.
3. CAFT's inspection step: per PC the mean loading of every cell type (hazard level x action), the cells loading
   maximally / minimally, and the loading variance explained by cell type and by scene (eta^2).
4. Overlap of the top-k ``delta_h`` subspace (k in 1, 2, 4, 8, 16) with three registered references, all fitted on arm A
   in arm A coordinates at the same site / group / step: (a) the relational conceptor ``C_rel = C_DiD(ped) AND NOT
   C_action AND NOT C_null`` exactly as ``geometry_cross_arm.py`` fits it (trace overlap ``tr(P_Q C) / sqrt(tr P_Q^2 tr
   C^2)``, the fraction of the conceptor's soft mass inside the subspace, principal angles against the conceptor's top-k
   eigenvectors) plus arm B's cone conceptor transported into A; (b) the identity-contrast direction ``unit(mean_i
   (DiD_ped,i - DiD_cone,i))`` of ``identity_contrast_geometry.py``; (c) the donor-patch direction, the unit mean of the
   corridor-pooled donor delta ``h[1,a] - h[0,a]`` written by ``patch_site.py``'s safe->unsafe patch (action-averaged and
   throttle-cell forms), at the same site and, separately, the registered ``L03.mlp_out`` corridor donor direction
   compared at every residual-stream site.  Projected cosines are reported against the chance value ``sqrt(k / d)`` and a
   matched-rank random subspace; conceptor overlaps against a matched-spectrum random conceptor.
5. Inference: scene-clustered bootstrap CIs (scenes resampled with replacement, PCA and overlaps refitted) and a
   scene-permutation null (scene i of arm A paired with scene pi(i) of arm B, same cell type; PCA and overlaps
   refitted) for the spectrum statistics and every overlap; ``p`` = fraction of permutations at or above the observed
   value.  The permutation null answers "is the structure of delta_h scene-paired", the random-subspace / matched-
   spectrum controls answer "is the overlap above chance"; both are reported, neither is a causal test.

Outputs ``<out>/arm_diff_map.json`` (all entries), ``<out>/arm_diff_map.md`` (per-group tables) and
``<out>/arm_diff_subspaces.npz`` (per site / group / step: top-16 PCs of the transported all-cell delta_h, the mean shift,
the three reference objects) for ``denial_finetune.py``.

``--self-test`` builds the planted and null synthetic dump pairs of ``geometry_cross_arm.py`` (arm B randomly rotated;
planted: a rank-2 relational subspace attached to the solid identity of each arm) and checks that the transported
in-lane delta_h subspace captures the planted subspace, that the raw (untransported) one does not, that the
permutation null is exceeded, and that the null pair shows chance-level capture.

Descriptive only (design doc, "Registered additions" (e)): candidate directions for the denial experiment, never C3
evidence.  References: arXiv:2507.16795 (CAFT), arXiv:2605.17144 (conceptor similarity, matched-spectrum control).
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
from geometry_cross_arm import (  # noqa: E402
    ACTION_LEVELS,
    NULL,
    OBJ,
    PED,
    align_arms,
    boot_mean,
    load_arm,
    load_site,
    matched_token_rows,
    orthogonal_procrustes,
    relational_rows,
    scene_rows,
    synthetic_dumps,
    token_rows,
    token_sets_for,
)
from geometry_localize import pool_site  # noqa: E402
from geometry_models import alpha_for_quota, conceptor_factored, conceptor_similarity, matched_spectrum_random  # noqa: E402
from geometry_conceptor import fit_conceptor_set  # noqa: E402
from sonar_metrics import principal_angles_deg  # noqa: E402
from stats_utils import finite  # noqa: E402
from token_groups import alias_note, set_domain  # noqa: E402

PROTOCOL = "cgs-arm-diff-pca-v0.1"
KS = (1, 2, 4, 8, 16)
K_REGISTERED = 4
DEFAULT_GROUPS = ("hazard", "corridor", "hazard_corridor")
SUBSETS = ("all", "in_lane", "hazard_free", "did")
SUBSET_LEVELS = {"all": (0, PED, NULL, OBJ), "in_lane": (PED, OBJ), "hazard_free": (0, NULL)}
CONVENTIONS = ("transported", "raw")
DONOR_SITE_DEFAULT = "L03.mlp_out"
DONOR_GROUP_DEFAULT = "corridor"
INTERPRETATION_SCOPE = (
    "Descriptive / associational (CAFT model-diffing analogue on the matched-consequence arms). PCA of h_A - h_B on identical "
    "inputs proposes candidate directions and is compared with the relational conceptor, the identity-contrast direction and "
    "the donor-patch direction; scene-bootstrap CIs and a scene-permutation null quantify pairing and chance, not use. "
    "Nothing here is C3 evidence; the candidate subspaces feed denial_finetune.py (necessity / re-routing follow-up)."
)


# --------------------------------------------------------------------------- #
# Linear algebra helpers
# --------------------------------------------------------------------------- #


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def pca(X: np.ndarray, kmax: int) -> dict[str, Any]:
    """Centred PCA of the rows of X. Returns mean, top-kmax PCs [k, d] (rows), variance ratios, spectrum summaries."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(0)
    Xc = X - mu
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    lam = s**2 / max(len(X) - 1, 1)
    tot = float(lam.sum())
    ratio = lam / tot if tot > 0 else lam
    k = int(min(kmax, len(lam)))
    cum = np.cumsum(ratio)
    energy_mean = float(len(X) * (mu @ mu))
    energy_total = float((X**2).sum())
    return {
        "mean": mu, "pcs": vt[:k], "lam": lam[:k], "ratio": ratio[:k],
        "pc1_ratio": float(ratio[0]) if len(ratio) else float("nan"),
        "participation_ratio": float(tot**2 / max(float((lam**2).sum()), 1e-300)) if tot > 0 else float("nan"),
        "rank90": int(np.searchsorted(cum, 0.9) + 1) if tot > 0 else 0,
        "mean_energy_fraction": float(energy_mean / energy_total) if energy_total > 0 else float("nan"),
        "rms_norm": float(np.sqrt(np.mean(np.sum(X**2, axis=1)))),
        "n_rows": int(len(X)),
    }


def proj_cos(Q: np.ndarray, u: np.ndarray) -> float:
    """||Q^T u|| / ||u|| for orthonormal columns Q [d, k]."""
    n = float(np.linalg.norm(u))
    return float(np.linalg.norm(Q.T @ u) / n) if n > 0 and np.all(np.isfinite(u)) else float("nan")


def conceptor_top(c: dict[str, np.ndarray], k: int) -> np.ndarray:
    """Top-k eigenvectors (by mu) of a factored conceptor as columns [d, k]."""
    V, mu = np.asarray(c["eigvecs"], dtype=np.float64), np.asarray(c["mu"], dtype=np.float64)
    order = np.argsort(mu)[::-1]
    return V[order[:k]].T


def conceptor_overlap(Q: np.ndarray, c: dict[str, np.ndarray]) -> dict[str, Any]:
    """Trace overlap of the hard projector on span(Q) with the soft conceptor C, the fraction of C's soft mass inside
    span(Q), and principal angles between span(Q) and C's top-k eigenvectors."""
    V, mu = np.asarray(c["eigvecs"], dtype=np.float64), np.asarray(c["mu"], dtype=np.float64)
    if V.size == 0:
        return {"trace_overlap": float("nan"), "mass_inside": float("nan"), "principal_angles_deg": []}
    k = Q.shape[1]
    hard = {"eigvecs": Q.T, "mu": np.ones(k)}
    G = V @ Q  # [r, k]
    mass = float(np.sum(mu * np.sum(G**2, axis=1)) / max(float(mu.sum()), 1e-300))
    kk = int(min(k, V.shape[0]))
    return {"trace_overlap": conceptor_similarity(hard, c), "mass_inside": mass, "principal_angles_deg": principal_angles_deg(Q, conceptor_top(c, kk))}


def transport_maps(XA: np.ndarray, XB: np.ndarray, scene: np.ndarray, scenes: list[str]) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, float]]:
    """LOSO hazard-free Procrustes maps as ``geometry_cross_arm.transport_folds`` but returning the fold means as well:
    per scene s -> (W_s, m_A,s, m_B,s) with x_B ~= (x_A - m_A) W + m_B, fitted without scene s; plus the full-data map."""
    M_all = XA.T @ XB
    sA, sB = XA.sum(0), XB.sum(0)
    n = len(XA)
    W_full = orthogonal_procrustes(M_all - np.outer(sA, sB) / n)
    full = (W_full, sA / n, sB / n)
    folds: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    res_t, res_r = [], []
    for s in scenes:
        te = scene == s
        n_tr = n - int(te.sum())
        if n_tr < 2:
            folds[s] = full
            continue
        sA_tr, sB_tr = sA - XA[te].sum(0), sB - XB[te].sum(0)
        W = orthogonal_procrustes(M_all - XA[te].T @ XB[te] - np.outer(sA_tr, sB_tr) / n_tr)
        mA, mB = sA_tr / n_tr, sB_tr / n_tr
        folds[s] = (W, mA, mB)
        a, b = XA[te] - mA, XB[te] - mB
        den = max(float(np.linalg.norm(b)), 1e-12)
        res_t.append(float(np.linalg.norm(a @ W - b) / den))
        res_r.append(float(np.linalg.norm(a - b) / den))
    diag = {"n_rows": int(n), "heldout_relative_residual_transported": float(np.mean(res_t)) if res_t else float("nan"),
            "heldout_relative_residual_raw": float(np.mean(res_r)) if res_r else float("nan")}
    return folds, full, diag


def transport_b_to_a(PB: np.ndarray, cells: list[dict[str, Any]], folds: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
    """Move arm B's pooled cell vectors into arm A coordinates with the LOSO map of the cell's own scene."""
    out = np.full_like(PB, np.nan)
    for c in cells:
        W, mA, mB = folds[c["pair_id"]]
        out[c["row"]] = (PB[c["row"]] - mB) @ W.T + mA
    return out


# --------------------------------------------------------------------------- #
# References (arm A coordinates)
# --------------------------------------------------------------------------- #


def relational_conceptor(rows: dict[str, np.ndarray], name: str, d: int, k_action: int, target_dims: float, seed: int) -> dict[str, Any] | None:
    """Exactly the ``rel_<name>`` safety conceptor of ``geometry_cross_arm.analyze_group`` (full-data fit)."""
    S = rows["app_ped"].shape[0]
    rel, _ = relational_rows(rows, np.arange(S), k_action)
    X = rel[f"rel_{name}"]
    X = X[np.all(np.isfinite(X), axis=1)]
    if len(X) < 2:
        return None
    scale = float(np.sqrt(np.mean(np.sum(X**2, axis=1)))) or 1.0
    base = conceptor_factored(X / scale, 1.0, center=False)
    alpha = alpha_for_quota(base["lam"], d, target_dims / d)
    Draw, A, Dn = rows[f"did_{name}"], rows["action"], rows["did_null"]
    okr = np.all(np.isfinite(Draw), axis=1) & np.all(np.isfinite(A), axis=1)
    Dn = Dn[np.all(np.isfinite(Dn), axis=1)]
    cs = fit_conceptor_set(Draw[okr] / scale, A[okr] / scale, alpha, Dnull=(Dn / scale) if len(Dn) >= 2 else None, seed=seed)
    return {"conceptor": cs["safety"], "alpha": alpha, "form": "C_did AND NOT C_action" + (" AND NOT C_null" if len(Dn) >= 2 else ""), "rank": int(len(cs["safety"]["mu"])),
            "quota": float(np.sum(cs["safety"]["mu"]) / d), "rel_rows": rel[f"rel_{name}"]}


def reference_objects(rowsA: dict[str, np.ndarray], rowsB: dict[str, np.ndarray], W_full: np.ndarray, d: int, args, seed: int) -> dict[str, Any]:
    """Conceptors, identity-contrast direction and donor-patch directions of this site / group (arm A coordinates)."""
    refs: dict[str, Any] = {}
    ca = relational_conceptor(rowsA, "ped", d, args.k_action, args.target_dims, seed)
    cb = relational_conceptor(rowsB, "cone", d, args.k_action, args.target_dims, seed)
    if ca is not None:
        refs["conceptor_A_rel_ped"] = ca
    if cb is not None:
        cb = dict(cb)
        cb["conceptor"] = {**cb["conceptor"], "eigvecs": np.asarray(cb["conceptor"]["eigvecs"]) @ W_full.T}  # v_A = v_B W^T
        refs["conceptor_B_rel_cone_to_A"] = cb
    ident = rowsA["did_ped"] - rowsA["did_cone"]
    ident = ident[np.all(np.isfinite(ident), axis=1)]
    refs["identity_direction"] = unit(ident.mean(0)) if len(ident) else None
    refs["identity_rows"] = ident
    app = rowsA["app_ped"]
    app = app[np.all(np.isfinite(app), axis=1)]
    refs["donor_direction"] = unit(app.mean(0)) if len(app) else None
    a1 = rowsA["app_ped_a1"]
    a1 = a1[np.all(np.isfinite(a1), axis=1)]
    refs["donor_direction_a1"] = unit(a1.mean(0)) if len(a1) else None
    return refs


def add_a1_rows(rows: dict[str, np.ndarray], P: np.ndarray, cells: list[dict[str, Any]], scenes: list[str]) -> dict[str, np.ndarray]:
    """``app_ped_a1 = h[1,1] - h[0,1]`` (throttle-cell donor delta) next to the rows of ``scene_rows``."""
    by: dict[str, dict[tuple[int, int], np.ndarray]] = {}
    for c in cells:
        by.setdefault(c["pair_id"], {})[(int(c["hazard"]), int(c["candidate_action"]))] = P[c["row"]]
    out = np.full((len(scenes), P.shape[1]), np.nan)
    for i, s in enumerate(scenes):
        h = by[s]
        if (PED, 1) in h and (0, 1) in h:
            out[i] = h[(PED, 1)] - h[(0, 1)]
    rows = dict(rows)
    rows["app_ped_a1"] = out
    return rows


# --------------------------------------------------------------------------- #
# delta_h matrices
# --------------------------------------------------------------------------- #


def delta_rows(PA: np.ndarray, PBt: np.ndarray, cells: list[dict[str, Any]], scenes: list[str], subset: str, perm: np.ndarray | None = None) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """delta_h rows for a cell subset.  ``perm`` maps scene index i -> the arm-B scene paired with it (None = identity).
    Returns (rows [n, d], row descriptors with cell type / scene / cell_id)."""
    by_scene: dict[str, dict[tuple[int, int], int]] = {}
    for c in cells:
        by_scene.setdefault(c["pair_id"], {})[(int(c["hazard"]), int(c["candidate_action"]))] = c["row"]
    pos = {s: i for i, s in enumerate(scenes)}
    rows, desc = [], []
    for s in scenes:
        sb = scenes[perm[pos[s]]] if perm is not None else s
        ca, cb = by_scene[s], by_scene[sb]
        if subset == "did":
            need = [(lv, a) for lv in (0, PED, OBJ) for a in (0, 1)]
            if not all(k in ca and k in cb for k in need):
                continue
            base_a = PA[ca[(0, 1)]] - PA[ca[(0, 0)]]
            base_b = PBt[cb[(0, 1)]] - PBt[cb[(0, 0)]]
            for lv, name in ((PED, "did_ped"), (OBJ, "did_cone")):
                da = (PA[ca[(lv, 1)]] - PA[ca[(lv, 0)]]) - base_a
                db = (PBt[cb[(lv, 1)]] - PBt[cb[(lv, 0)]]) - base_b
                if np.all(np.isfinite(da)) and np.all(np.isfinite(db)):
                    rows.append(da - db)
                    desc.append({"scene": s, "scene_b": sb, "type": name, "level": lv, "action": None, "cell_id": f"{s}__{name}"})
            continue
        for (lv, a), ra in sorted(ca.items()):
            if lv not in SUBSET_LEVELS[subset] or (lv, a) not in cb:
                continue
            v = PA[ra] - PBt[cb[(lv, a)]]
            if np.all(np.isfinite(v)):
                rows.append(v)
                desc.append({"scene": s, "scene_b": sb, "type": f"h{lv}a{a}", "level": lv, "action": a, "cell_id": f"{s}__h{lv}a{a}"})
    return (np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, PA.shape[1]))), desc


def derangement(S: int, rng: np.random.Generator, max_tries: int = 200) -> np.ndarray:
    """Random permutation of range(S) without fixed points (rejection sampling; cyclic shift as the fallback)."""
    if S < 2:
        return np.arange(S)
    for _ in range(max_tries):
        perm = rng.permutation(S)
        if not np.any(perm == np.arange(S)):
            return perm
    return np.roll(np.arange(S), int(rng.integers(1, S)))


def inspect_loadings(X: np.ndarray, desc: list[dict[str, Any]], p: dict[str, Any], n_pcs: int = 4, n_extreme: int = 3) -> list[dict[str, Any]]:
    """CAFT inspection: per PC the cell-type mean loadings, extreme cells, and eta^2 of cell type / scene."""
    Xc = X - p["mean"]
    out = []
    types = np.asarray([d["type"] for d in desc])
    scenes = np.asarray([d["scene"] for d in desc])
    for j in range(min(n_pcs, len(p["pcs"]))):
        s = Xc @ p["pcs"][j]
        tot = float(((s - s.mean()) ** 2).sum())

        def eta2(groups: np.ndarray) -> float:
            if tot <= 0:
                return float("nan")
            between = 0.0
            for g in np.unique(groups):
                m = groups == g
                between += m.sum() * (s[m].mean() - s.mean()) ** 2
            return float(between / tot)

        order = np.argsort(s)
        out.append({
            "pc": j + 1, "variance_ratio": float(p["ratio"][j]),
            "by_cell_type": {str(t): {"mean": float(s[types == t].mean()), "sd": float(s[types == t].std()), "n": int((types == t).sum())} for t in np.unique(types)},
            "eta2_cell_type": eta2(types), "eta2_scene": eta2(scenes),
            "max_loading_cells": [{"cell_id": desc[i]["cell_id"], "loading": float(s[i])} for i in order[::-1][:n_extreme]],
            "min_loading_cells": [{"cell_id": desc[i]["cell_id"], "loading": float(s[i])} for i in order[:n_extreme]],
        })
    return out


# --------------------------------------------------------------------------- #
# Statistics of one delta_h matrix against the references
# --------------------------------------------------------------------------- #


def overlap_stats(p: dict[str, Any], refs: dict[str, Any], ks: tuple[int, ...], d: int, rng: np.random.Generator, extra_dirs: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
    """Overlap of the top-k PC subspaces (and the mean-shift direction) with the references."""
    out: dict[str, Any] = {"pc1_ratio": p["pc1_ratio"], "participation_ratio": p["participation_ratio"], "rank90": p["rank90"],
                           "mean_energy_fraction": p["mean_energy_fraction"], "rms_norm": p["rms_norm"], "ks": {}}
    dirs = {"identity": refs.get("identity_direction"), "donor": refs.get("donor_direction"), "donor_a1": refs.get("donor_direction_a1")}
    if extra_dirs:
        dirs.update(extra_dirs)
    m = unit(p["mean"])
    out["mean_shift"] = {name: float(abs(m @ u)) if u is not None and np.linalg.norm(m) > 0 else float("nan") for name, u in dirs.items()}
    for name in ("conceptor_A_rel_ped", "conceptor_B_rel_cone_to_A"):
        if name in refs and np.linalg.norm(m) > 0:
            out["mean_shift"][name] = conceptor_overlap(m[:, None], refs[name]["conceptor"])["mass_inside"]
    for k in ks:
        kk = int(min(k, len(p["pcs"])))
        if kk < 1:
            continue
        Q = p["pcs"][:kk].T
        Qr, _ = np.linalg.qr(rng.normal(size=(d, kk)))
        e: dict[str, Any] = {"k": kk, "chance_proj_cos": float(np.sqrt(kk / d)), "variance_captured": float(np.sum(p["ratio"][:kk]))}
        for name, u in dirs.items():
            e[f"{name}_proj_cos"] = proj_cos(Q, u) if u is not None else float("nan")
            e[f"{name}_random_proj_cos"] = proj_cos(Qr, u) if u is not None else float("nan")
        for name in ("conceptor_A_rel_ped", "conceptor_B_rel_cone_to_A"):
            if name not in refs:
                continue
            c = refs[name]["conceptor"]
            ov = conceptor_overlap(Q, c)
            e[name] = ov
            e[f"{name}_matched_random"] = conceptor_overlap(Q, matched_spectrum_random(c, seed=int(rng.integers(1 << 30))))
        out["ks"][f"k{kk}"] = e
    return out


def flat_stats(st: dict[str, Any]) -> dict[str, float]:
    """Scalar view of ``overlap_stats`` for bootstrap / permutation bookkeeping."""
    f: dict[str, float] = {"pc1_ratio": st["pc1_ratio"], "participation_ratio": st["participation_ratio"], "rank90": float(st["rank90"]), "mean_energy_fraction": st["mean_energy_fraction"]}
    for name, v in st["mean_shift"].items():
        f[f"mean_shift.{name}"] = v
    for kname, e in st["ks"].items():
        for key, v in e.items():
            if isinstance(v, dict):
                f[f"{kname}.{key}.trace_overlap"] = v["trace_overlap"]
                f[f"{kname}.{key}.mass_inside"] = v["mass_inside"]
            elif isinstance(v, (int, float, np.floating)):
                f[f"{kname}.{key}"] = float(v)
    return f


def analyze_matrix(PA: np.ndarray, PBt: np.ndarray, cells: list[dict[str, Any]], scenes: list[str], subset: str, refs: dict[str, Any], args, rng: np.random.Generator, n_boot: int, n_perm: int, extra_dirs: dict[str, np.ndarray] | None) -> dict[str, Any]:
    X, desc = delta_rows(PA, PBt, cells, scenes, subset)
    d = PA.shape[1]
    if len(X) < 4:
        return {"status": "too_few_rows", "n_rows": int(len(X))}
    kmax = max(args.ks)
    p = pca(X, kmax)
    obs = overlap_stats(p, refs, tuple(args.ks), d, rng, extra_dirs)
    fo = flat_stats(obs)
    entry: dict[str, Any] = {"status": "ok", "n_rows": int(len(X)), "n_scenes": len({dd["scene"] for dd in desc}), "observed": obs, "loadings": inspect_loadings(X, desc, p),
                             "energy_by_cell_type": {str(t): float(np.mean(np.sum(X[np.asarray([dd["type"] for dd in desc]) == t] ** 2, axis=1))) for t in sorted({dd["type"] for dd in desc})}}
    # scene-clustered bootstrap
    S = len(scenes)
    boots: dict[str, list[float]] = {k: [] for k in fo}
    for _ in range(n_boot):
        pick = rng.integers(0, S, size=S)
        sub = [scenes[i] for i in pick]
        Xb, _ = delta_rows(PA, PBt, cells, sub, subset)
        if len(Xb) < 4:
            continue
        fb = flat_stats(overlap_stats(pca(Xb, kmax), refs, tuple(args.ks), d, rng, extra_dirs))
        for k in boots:
            boots[k].append(fb.get(k, np.nan))
    entry["bootstrap"] = {k: {"ci_low": float(np.nanpercentile(v, 2.5)), "ci_high": float(np.nanpercentile(v, 97.5)), "n": int(np.sum(np.isfinite(v)))} if len(v) else None for k, v in boots.items()}
    # scene-permutation null (arm A scene i paired with arm B scene pi(i); pi has no fixed point)
    nulls: dict[str, list[float]] = {k: [] for k in fo}
    for _ in range(n_perm):
        perm = derangement(S, rng)
        Xp, _ = delta_rows(PA, PBt, cells, scenes, subset, perm=perm)
        if len(Xp) < 4:
            continue
        fp = flat_stats(overlap_stats(pca(Xp, kmax), refs, tuple(args.ks), d, rng, extra_dirs))
        for k in nulls:
            nulls[k].append(fp.get(k, np.nan))
    entry["permutation_null"] = {}
    for k, v in nulls.items():
        arr = np.asarray(v, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        o = fo.get(k, np.nan)
        entry["permutation_null"][k] = {"null_mean": float(arr.mean()) if len(arr) else float("nan"), "null_q95": float(np.percentile(arr, 95)) if len(arr) else float("nan"),
                                        "p_ge": float((np.sum(arr >= o - 1e-12) + 1) / (len(arr) + 1)) if len(arr) and np.isfinite(o) else float("nan"), "n_perm": int(len(arr))}
    entry["_pca"] = p
    return entry


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def run(args) -> dict[str, Any]:
    t0 = time.time()
    set_domain(args.domain)
    arm_a, arm_b = load_arm(args.dump_a, args.discovery_seeds), load_arm(args.dump_b, args.discovery_seeds)
    cells, scenes, status = align_arms(arm_a, arm_b)
    if len(scenes) < 4:
        raise SystemExit(f"only {len(scenes)} complete scenes shared by the two dumps (need >= 4)")
    n_steps = min(arm_a["n_steps"], arm_b["n_steps"])
    steps = [s for s in args.steps if s < n_steps]
    groups = tuple(args.groups)
    token_sets = token_sets_for(args.stimulus, cells, groups, n_steps, arm_a["offset"], args.domain)
    rows_a, rows_b = np.asarray([c["row_a"] for c in cells]), np.asarray([c["row_b"] for c in cells])
    if args.sites:
        sites = list(args.sites)
    else:
        sa = {p.stem for p in (args.dump_a / "activations").glob("*.npz")}
        sb = {p.stem for p in (args.dump_b / "activations").glob("*.npz")}
        sites = sorted(s for s in sa & sb if "adaln" not in s)
    # the registered donor direction (L03.mlp_out corridor by default) is computed first and compared at every site
    donor_site_dir: dict[int, np.ndarray | None] = {}
    if args.donor_site in sites:
        actsA, tixA = load_site(args.dump_a, args.donor_site, rows_a)
        for step in steps:
            PA, _ = pool_site(actsA, tixA, cells, token_sets, step, args.donor_group)
            ok = np.all(np.isfinite(PA), axis=1)
            keep = [s for s in scenes if all(ok[c["row"]] for c in cells if c["pair_id"] == s)]
            if len(keep) >= 2:
                r = scene_rows(PA, [c for c in cells if c["pair_id"] in set(keep)], keep)
                app = r["app_ped"]
                donor_site_dir[step] = unit(app[np.all(np.isfinite(app), axis=1)].mean(0))
            else:
                donor_site_dir[step] = None
    rng = np.random.default_rng(args.seed)
    entries: list[dict[str, Any]] = []
    export: dict[str, np.ndarray] = {}
    export_meta: list[dict[str, Any]] = []
    transport_log: dict[str, Any] = {}
    for si, site in enumerate(sites):
        actsA, tixA = load_site(args.dump_a, site, rows_a)
        actsB, tixB = load_site(args.dump_b, site, rows_b)
        if actsA.shape[2] == 1 or actsB.shape[2] == 1:
            entries.append({"site_id": site, "status": "adaln_skipped"})
            continue
        d = actsA.shape[-1]
        for step in steps:
            lv = tuple(args.transport_levels)
            XA, XB, scene_idx = matched_token_rows(token_rows(actsA, tixA, cells, step, lv), token_rows(actsB, tixB, cells, step, lv))
            if len(XA) < 8:
                for g in groups:
                    entries.append({"site_id": site, "step": step, "group": g, "status": "no_matched_tokens"})
                continue
            folds, (W_full, mA_full, mB_full), tlog = transport_maps(XA, XB, scene_idx, scenes)
            transport_log[f"{site}|s{step}"] = tlog
            for g in groups:
                PA, _ = pool_site(actsA, tixA, cells, token_sets, step, g)
                PB, _ = pool_site(actsB, tixB, cells, token_sets, step, g)
                ok = np.all(np.isfinite(PA), axis=1) & np.all(np.isfinite(PB), axis=1)
                bad = {c["pair_id"] for c in cells if not ok[c["row"]]}
                keep = [s for s in scenes if s not in bad]
                if len(keep) < 4:
                    entries.append({"site_id": site, "step": step, "group": g, "status": "no_tokens"})
                    continue
                sub_cells = [c for c in cells if c["pair_id"] in set(keep)]
                PBt = transport_b_to_a(PB, sub_cells, folds)
                rowsA = add_a1_rows(scene_rows(PA, sub_cells, keep), PA, sub_cells, keep)
                rowsB = add_a1_rows(scene_rows(PBt, sub_cells, keep), PBt, sub_cells, keep)
                refs = reference_objects(rowsA, rowsB, W_full, d, args, args.seed)
                extra = {"donor_registered_site": donor_site_dir.get(step)} if donor_site_dir.get(step) is not None else None
                entry: dict[str, Any] = {"site_id": site, "step": step, "group": g, "status": "ok", "n_scenes": len(keep), "n_scenes_dropped_no_tokens": len(scenes) - len(keep), "d": int(d),
                                         "references": {"conceptor_A_rel_ped": {k: v for k, v in refs.get("conceptor_A_rel_ped", {}).items() if k in ("alpha", "form", "rank", "quota")},
                                                        "conceptor_B_rel_cone_to_A": {k: v for k, v in refs.get("conceptor_B_rel_cone_to_A", {}).items() if k in ("alpha", "form", "rank", "quota")},
                                                        "identity_direction": "unit mean over scenes of DiD_ped - DiD_cone (arm A)", "donor_direction": "unit mean over scenes of mean_a (h[1,a] - h[0,a]) (arm A; patch_site safe->unsafe donor delta)",
                                                        "donor_direction_a1": "unit mean over scenes of h[1,1] - h[0,1] (arm A throttle cell)",
                                                        "donor_registered_site": f"{args.donor_site} {args.donor_group} donor direction compared here (cross-site; residual-stream coordinates)" if extra else None,
                                                        "cos_identity_donor": float(abs(refs["identity_direction"] @ refs["donor_direction"])) if refs.get("identity_direction") is not None and refs.get("donor_direction") is not None else None},
                                         "conventions": {}}
                for conv in CONVENTIONS:
                    B_here = PBt if conv == "transported" else PB
                    nb, npm = (args.n_boot, args.n_perm) if conv == "transported" else (max(10, args.n_boot // 4), max(10, args.n_perm // 4))
                    entry["conventions"][conv] = {}
                    for subset in args.subsets:
                        res = analyze_matrix(PA, B_here, sub_cells, keep, subset, refs, args, rng, nb, npm, extra)
                        if conv == "transported" and subset == "all" and res.get("status") == "ok" and args.export is not None:
                            p = res["_pca"]
                            pre = f"{site}__s{step}__{g}"
                            export[f"{pre}__pcs"] = p["pcs"].astype(np.float32)
                            export[f"{pre}__variance_ratio"] = np.asarray(p["ratio"], dtype=np.float32)
                            export[f"{pre}__mean_shift"] = p["mean"].astype(np.float32)
                            if refs.get("identity_direction") is not None:
                                export[f"{pre}__identity_direction"] = refs["identity_direction"].astype(np.float32)
                            if refs.get("donor_direction") is not None:
                                export[f"{pre}__donor_direction"] = refs["donor_direction"].astype(np.float32)
                            if "conceptor_A_rel_ped" in refs:
                                c = refs["conceptor_A_rel_ped"]["conceptor"]
                                order = np.argsort(np.asarray(c["mu"]))[::-1]
                                export[f"{pre}__conceptor_A_rel_ped__eigvecs"] = np.asarray(c["eigvecs"])[order].astype(np.float32)
                                export[f"{pre}__conceptor_A_rel_ped__mu"] = np.asarray(c["mu"])[order].astype(np.float32)
                            export[f"{pre}__W_full"] = W_full.astype(np.float32)
                            export_meta.append({"site": site, "step": step, "group": g, "n_scenes": len(keep), "pc1_ratio": p["pc1_ratio"], "participation_ratio": p["participation_ratio"],
                                                "conceptor_rank": refs.get("conceptor_A_rel_ped", {}).get("rank"), "conceptor_alpha": refs.get("conceptor_A_rel_ped", {}).get("alpha")})
                        res.pop("_pca", None)
                        entry["conventions"][conv][subset] = res
                entry["transport"] = tlog
                entries.append(entry)
        print(f"[{si + 1}/{len(sites)}] {site} ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    tables = ranked_tables(entries, args)
    report = {
        "protocol": PROTOCOL, "dump_a": str(args.dump_a), "dump_b": str(args.dump_b), "stimulus": str(args.stimulus), "domain": args.domain, "token_group_source": alias_note(args.domain),
        "arms": {"A": args.arm_a_name, "B": args.arm_b_name}, "n_scenes": len(scenes), "scenes": scenes, "n_cells": len(cells), "alignment": status, "steps": steps, "groups": list(groups), "subsets": list(args.subsets),
        "ks": list(args.ks), "k_registered": K_REGISTERED, "k_action": args.k_action, "target_dims": args.target_dims, "sites": sites, "transport_levels": list(args.transport_levels),
        "donor_registered": {"site": args.donor_site, "group": args.donor_group, "available_steps": [s for s, v in donor_site_dir.items() if v is not None]},
        "definitions": {"delta_h": "h_A(x) - h_B(x) per cell on group-pooled tokens; transported: h_B moved into A coordinates with the LOSO hazard-free Procrustes map of the cell's scene (x_B ~= (x_A - m_A) W + m_B); raw: shared ambient coordinates",
                        "subsets": {"all": "all 8 cells per scene", "in_lane": "levels 1 (pedestrian) and 3 (cone)", "hazard_free": "levels 0 and 2 (the transport was fitted on these; their delta_h is the Procrustes residual by construction)",
                                    "did": "per scene DiD_ped^A - DiD_ped^B and DiD_cone^A - DiD_cone^B (a template shared by both arms cancels)"},
                        "pca": "centred PCA over rows; pc1_ratio = lam_1 / sum lam; participation_ratio = (sum lam)^2 / sum lam^2; rank90 = PCs for 90 % variance; mean_energy_fraction = ||mean||^2 n / sum ||row||^2",
                        "overlaps": "proj_cos = ||Q^T u|| (chance sqrt(k/d), matched-rank random subspace reported); conceptor: trace overlap tr(P_Q C)/sqrt(tr P_Q^2 tr C^2), mass_inside = sum_j mu_j ||Q^T v_j||^2 / sum mu_j, principal angles vs the top-k conceptor eigenvectors; matched-spectrum random conceptor control",
                        "inference": "scene-clustered bootstrap (scenes resampled, everything refitted) for CIs; scene-permutation null (A scene i paired with B scene pi(i), same cell type, no fixed points) with p_ge = P(null >= observed)",
                        "references": "fitted on arm A (full data, in-sample; descriptive) at the same site/group/step in arm A coordinates; arm B's cone conceptor moved with the full-data W (v_A = v_B W^T)"},
        "ranked_tables": tables, "transport": transport_log, "entries": entries, "runtime_s": time.time() - t0, "interpretation_scope": INTERPRETATION_SCOPE,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    if args.export is not None and export:
        Path(args.export).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.export, meta=json.dumps(finite({"protocol": PROTOCOL, "entries": export_meta, "layout": "<site>__s<step>__<group>__{pcs [16,d] rows | variance_ratio | mean_shift | identity_direction | donor_direction | conceptor_A_rel_ped__eigvecs [r,d] (mu-descending) | conceptor_A_rel_ped__mu | W_full [d,d]}; transported all-cell delta_h, arm A coordinates"})), **export)
        report["export"] = str(args.export)
    (args.out / "arm_diff_map.json").write_text(json.dumps(finite(report), indent=1) + "\n")
    (args.out / "arm_diff_map.md").write_text(markdown(report))
    return report


def ranked_tables(entries: list[dict[str, Any]], args) -> dict[str, list[dict[str, Any]]]:
    k = f"k{K_REGISTERED}"
    tables: dict[str, list[dict[str, Any]]] = {}
    for e in entries:
        if e.get("status") != "ok":
            continue
        key = f"s{e['step']}|{e['group']}"
        row: dict[str, Any] = {"site_id": e["site_id"], "n_scenes": e["n_scenes"], "transport_residual": e["transport"]["heldout_relative_residual_transported"]}
        for conv in CONVENTIONS:
            for subset in args.subsets:
                r = e["conventions"][conv].get(subset, {})
                if r.get("status") != "ok":
                    continue
                o, b, pn = r["observed"], r["bootstrap"], r["permutation_null"]
                kk = o["ks"].get(k) or o["ks"].get(f"k{max(int(x[1:]) for x in o['ks'])}")
                pre = f"{conv}.{subset}"
                row[f"{pre}.pc1"] = o["pc1_ratio"]
                row[f"{pre}.pc1_ci"] = [b["pc1_ratio"]["ci_low"], b["pc1_ratio"]["ci_high"]] if b.get("pc1_ratio") else None
                row[f"{pre}.pc1_p_perm"] = pn["pc1_ratio"]["p_ge"]
                row[f"{pre}.pr"] = o["participation_ratio"]
                row[f"{pre}.rms"] = o["rms_norm"]
                if kk:
                    kn = f"k{kk['k']}"
                    row[f"{pre}.identity_cos"] = kk["identity_proj_cos"]
                    row[f"{pre}.identity_cos_ci"] = [b[f"{kn}.identity_proj_cos"]["ci_low"], b[f"{kn}.identity_proj_cos"]["ci_high"]] if b.get(f"{kn}.identity_proj_cos") else None
                    row[f"{pre}.identity_cos_p_perm"] = pn.get(f"{kn}.identity_proj_cos", {}).get("p_ge")
                    row[f"{pre}.donor_cos"] = kk["donor_proj_cos"]
                    row[f"{pre}.donor_cos_p_perm"] = pn.get(f"{kn}.donor_proj_cos", {}).get("p_ge")
                    row[f"{pre}.donor_registered_cos"] = kk.get("donor_registered_site_proj_cos")
                    row[f"{pre}.chance_cos"] = kk["chance_proj_cos"]
                    if "conceptor_A_rel_ped" in kk:
                        row[f"{pre}.conceptor_overlap"] = kk["conceptor_A_rel_ped"]["trace_overlap"]
                        row[f"{pre}.conceptor_mass"] = kk["conceptor_A_rel_ped"]["mass_inside"]
                        row[f"{pre}.conceptor_random"] = kk["conceptor_A_rel_ped_matched_random"]["trace_overlap"]
                        row[f"{pre}.conceptor_mass_ci"] = [b[f"{kn}.conceptor_A_rel_ped.mass_inside"]["ci_low"], b[f"{kn}.conceptor_A_rel_ped.mass_inside"]["ci_high"]] if b.get(f"{kn}.conceptor_A_rel_ped.mass_inside") else None
                        row[f"{pre}.conceptor_mass_p_perm"] = pn.get(f"{kn}.conceptor_A_rel_ped.mass_inside", {}).get("p_ge")
                        row[f"{pre}.conceptor_angle_min"] = min(kk["conceptor_A_rel_ped"]["principal_angles_deg"]) if kk["conceptor_A_rel_ped"]["principal_angles_deg"] else None
                    if "conceptor_B_rel_cone_to_A" in kk:
                        row[f"{pre}.conceptorB_mass"] = kk["conceptor_B_rel_cone_to_A"]["mass_inside"]
        tables.setdefault(key, []).append(row)
    for rows in tables.values():
        rows.sort(key=lambda r: -(r.get("transported.all.conceptor_mass") if isinstance(r.get("transported.all.conceptor_mass"), (int, float)) and np.isfinite(r.get("transported.all.conceptor_mass")) else -np.inf))
    return tables


def _f(v, nd=3) -> str:
    if v is None:
        return "nan"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_f(x, nd) for x in v) + "]"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def markdown(report: dict[str, Any]) -> str:
    lines = [f"# Arm diffing (CAFT-style PCA of h_A - h_B) — {report['protocol']}", "",
             f"Dumps: A `{report['dump_a']}`, B `{report['dump_b']}`; scenes {report['n_scenes']}, cells {report['n_cells']}; steps {report['steps']}; groups {report['groups']}; k registered {report['k_registered']}; runtime {report['runtime_s']:.0f} s.", "",
             f"Scope: {report['interpretation_scope']}", ""]
    for key, rows in report["ranked_tables"].items():
        lines += [f"## {key} (sorted by the conceptor mass inside the top-{report['k_registered']} transported all-cell delta_h subspace)", "",
                  "| site | transport res. | PC1 (all) [CI] p_perm | PR | PC1 in-lane | PC1 did | conceptor mass k4 [CI] p_perm (rand) | overlap | min angle | identity cos [CI] p_perm | donor cos | L03 donor cos | chance | raw PC1 | raw conceptor mass |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            g = lambda k: r.get(k)  # noqa: E731
            lines.append(f"| {r['site_id']} | {_f(g('transport_residual'))} | {_f(g('transported.all.pc1'))} {_f(g('transported.all.pc1_ci'))} {_f(g('transported.all.pc1_p_perm'))} | {_f(g('transported.all.pr'), 1)} | {_f(g('transported.in_lane.pc1'))} | {_f(g('transported.did.pc1'))} | "
                         f"{_f(g('transported.all.conceptor_mass'))} {_f(g('transported.all.conceptor_mass_ci'))} {_f(g('transported.all.conceptor_mass_p_perm'))} ({_f(g('transported.all.conceptor_random'))}) | {_f(g('transported.all.conceptor_overlap'))} | {_f(g('transported.all.conceptor_angle_min'), 1)} | "
                         f"{_f(g('transported.all.identity_cos'))} {_f(g('transported.all.identity_cos_ci'))} {_f(g('transported.all.identity_cos_p_perm'))} | {_f(g('transported.all.donor_cos'))} | {_f(g('transported.all.donor_registered_cos'))} | {_f(g('transported.all.chance_cos'))} | {_f(g('raw.all.pc1'))} | {_f(g('raw.all.conceptor_mass'))} |")
        lines.append("")
    lines += ["## Cell-type loadings of PC1 (transported, all cells) — the CAFT inspection step", "", "| site | group | eta2 cell type | eta2 scene | mean loading per cell type | max cells | min cells |", "|---|---|---|---|---|---|---|"]
    for e in report["entries"]:
        if e.get("status") != "ok":
            continue
        r = e["conventions"]["transported"].get("all", {})
        if r.get("status") != "ok" or not r.get("loadings"):
            continue
        l0 = r["loadings"][0]
        bt = ", ".join(f"{t}:{_f(v['mean'], 2)}" for t, v in l0["by_cell_type"].items())
        lines.append(f"| {e['site_id']} | {e['group']} | {_f(l0['eta2_cell_type'], 2)} | {_f(l0['eta2_scene'], 2)} | {bt} | {', '.join(c['cell_id'] for c in l0['max_loading_cells'])} | {', '.join(c['cell_id'] for c in l0['min_loading_cells'])} |")
    lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #


def self_test_verdict(report: dict[str, Any], planted: bool, S_rel: np.ndarray, S_ped_A: np.ndarray, site: str, group: str) -> dict[str, Any]:
    """Planted: the transported in-lane / did delta_h top-2 subspace captures the planted relational subspace (energy
    >= 0.5), the raw one does not (< 0.25; arm B is rotated), the PC1 permutation p < 0.05 (delta_h is scene-paired),
    the identity direction is inside the top-2 (proj cos > 0.7) and the conceptor mass exceeds the matched-spectrum
    random control by >= 0.2.  Null: planted-subspace energy <= 0.25 in every subset."""
    by = {(e["site_id"], e["group"]): e for e in report["entries"] if e.get("status") == "ok"}
    e = by[(site, group)]
    exp = np.load(report["export"]) if "export" in report else None
    pre = f"{site}__s0__{group}"
    Q = exp[f"{pre}__pcs"][:2].T.astype(np.float64) if exp is not None else None

    def energy(conv, subset, k=2):
        r = e["conventions"][conv][subset]
        if r.get("status") != "ok":
            return float("nan")
        return r["observed"]["ks"][f"k{k}"].get("planted_energy", float("nan"))

    checks: dict[str, Any] = {}
    en_t_in, en_t_did, en_r_in = energy("transported", "in_lane"), energy("transported", "did"), energy("raw", "in_lane")
    o = e["conventions"]["transported"]["in_lane"]
    k2 = o["observed"]["ks"]["k2"]
    ident = k2["identity_proj_cos"]
    cmass, crand = k2["conceptor_A_rel_ped"]["mass_inside"], k2["conceptor_A_rel_ped_matched_random"]["mass_inside"]
    p_perm = o["permutation_null"]["pc1_ratio"]["p_ge"]
    if planted:
        checks = {"planted_energy_in_lane_transported_ge_0.5": en_t_in >= 0.5, "planted_energy_did_transported_ge_0.5": en_t_did >= 0.5, "planted_energy_in_lane_raw_lt_0.25": en_r_in < 0.25,
                  "pc1_permutation_p_lt_0.05": p_perm < 0.05, "identity_proj_cos_gt_0.7": ident > 0.7, "conceptor_mass_beats_random_by_0.2": (cmass - crand) >= 0.2}
    else:
        checks = {"planted_energy_in_lane_transported_le_0.25": en_t_in <= 0.25, "planted_energy_did_transported_le_0.25": en_t_did <= 0.25, "planted_energy_all_transported_le_0.25": energy("transported", "all") <= 0.25}
    return {"planted": planted, "site": site, "group": group, "checks": checks, "passed": all(checks.values()),
            "values": {"planted_energy_in_lane_transported": en_t_in, "planted_energy_did_transported": en_t_did, "planted_energy_in_lane_raw": en_r_in, "pc1_permutation_p": p_perm, "identity_proj_cos": ident, "conceptor_mass": cmass, "conceptor_mass_random": crand}}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dump-a", type=Path, default=None)
    ap.add_argument("--dump-b", type=Path, default=None)
    ap.add_argument("--stimulus", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--export", type=Path, default=None, help="write arm_diff_subspaces.npz here (default <out>/arm_diff_subspaces.npz)")
    ap.add_argument("--discovery-seeds", type=Path, default=None)
    ap.add_argument("--domain", default="driving")
    ap.add_argument("--sites", nargs="*", default=None)
    ap.add_argument("--steps", type=int, nargs="*", default=[0])
    ap.add_argument("--groups", nargs="*", default=list(DEFAULT_GROUPS))
    ap.add_argument("--subsets", nargs="*", default=list(SUBSETS))
    ap.add_argument("--ks", type=int, nargs="*", default=list(KS))
    ap.add_argument("--k-action", type=int, default=4)
    ap.add_argument("--target-dims", type=float, default=4.0)
    ap.add_argument("--transport-levels", type=int, nargs="*", default=list(ACTION_LEVELS))
    ap.add_argument("--donor-site", default=DONOR_SITE_DEFAULT, help="registered donor-patch site whose corridor donor direction is compared at every site")
    ap.add_argument("--donor-group", default=DONOR_GROUP_DEFAULT)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm-a-name", default="A: PED-solid / OBJ-ghost")
    ap.add_argument("--arm-b-name", default="B: PED-ghost / OBJ-solid")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--synthetic-scenes", type=int, default=16)
    ap.add_argument("--synthetic-dim", type=int, default=64)
    return ap


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.self_test:
        verdicts = {}
        for null in (False, True):
            root = args.out / ("self_test_null" if null else "self_test_planted")
            stim, dA, dB = synthetic_dumps(root, null=null, n_scenes=args.synthetic_scenes, d=args.synthetic_dim, sites=("L03.attn_out", "L07.attn_out"), seed=args.seed, noise=0.3)
            idx = json.loads((dA / "activations" / "index.json").read_text())["synthetic"]
            S_rel, S_ped_A = np.asarray(idx["S_rel"]), np.asarray(idx["S_ped_A"])
            a2 = build_parser().parse_args(["--out", str(root / "out"), "--dump-a", str(dA), "--dump-b", str(dB), "--stimulus", str(stim), "--steps", "0", "--groups", "hazard_corridor", "corridor",
                                            "--donor-site", "L03.attn_out", "--donor-group", "hazard_corridor", "--n-boot", str(args.n_boot), "--n-perm", str(args.n_perm), "--seed", str(args.seed),
                                            "--export", str(root / "out" / "arm_diff_subspaces.npz")])
            # planted-subspace energy inside the top-k delta_h subspace: patched in through the extra-direction slot of
            # overlap_stats by wrapping analyze_matrix's reference dict (self-test only)
            global overlap_stats  # noqa: PLW0603
            _orig = overlap_stats

            def with_planted(p, refs, ks, d, rng, extra_dirs=None, _o=_orig, S=S_rel):
                out = _o(p, refs, ks, d, rng, extra_dirs)
                for kname, e in out["ks"].items():
                    Q = p["pcs"][: e["k"]].T
                    e["planted_energy"] = float(np.sum((S @ Q) ** 2) / S.shape[0])  # mean over planted directions of ||Q^T s||^2
                return out

            overlap_stats = with_planted
            try:
                rep = run(a2)
            finally:
                overlap_stats = _orig
            v = self_test_verdict(rep, planted=not null, S_rel=S_rel, S_ped_A=S_ped_A, site="L03.attn_out", group="hazard_corridor")
            verdicts["null" if null else "planted"] = v
            print(json.dumps(finite(v)))
        ok = all(v["passed"] for v in verdicts.values())
        (args.out / "self_test_verdict.json").write_text(json.dumps(finite({"passed": ok, **verdicts}), indent=1) + "\n")
        print("SELF_TEST_PASSED" if ok else "SELF_TEST_FAILED")
        if not ok:
            raise SystemExit(1)
        return verdicts
    if not (args.dump_a and args.dump_b and args.stimulus):
        raise SystemExit("--dump-a, --dump-b and --stimulus are required (or --self-test)")
    if args.export is None:
        args.export = args.out / "arm_diff_subspaces.npz"
    return run(args)


if __name__ == "__main__":
    main()
