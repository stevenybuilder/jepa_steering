#!/usr/bin/env python3
"""Cross-arm representational geometry for the reversed-assignment arms (protocol v0.7 amendment).

Two predictors trained on the identical clip multiset -- arm A (pedestrian solid / cone ghost) and
arm B (pedestrian ghost / cone solid) -- are evaluated on the SAME factorial scenes with the same
encoder tokens (``localize_interaction.py --domain driving`` dumps: hazard levels 0 = sidewalk,
1 = pedestrian in lane, 2 = H0' mirror pose, 3 = cone in lane; actions 0 = brake, 1 = throttle).
Per site x imagined step x token group, on discovery scenes, leave-one-scene-out (LOSO) wherever a
fit is involved, this module computes

1. Per-arm contrast subspaces (uncentred top-k singular subspaces, k in {1,2,4,8}, as in
   ``sonar_metrics.py``) of four per-scene contrast rows built from the group-pooled activations
   ``h[level, action]``:
     app_ped   = mean_a (h[1,a] - h[0,a])                    pedestrian-in-lane vs sidewalk main effect
     app_cone  = mean_a (h[3,a] - h[0,a])                    cone vs sidewalk main effect
     app_*_a0  = h[l,0] - h[0,0]                             brake-cell appearance = main effect - DiD(l)/2; the
                 action-averaged main effect of the SOLID identity carries half its relational DiD by
                 construction, so test (i) is registered on the brake-cell contrast (both are reported)
     rel_ped   = DiD(1) - DiD(2), then projected off the train-fold action main-effect subspace
                 (action rows = mean over the hazard-free levels 0/2 of h[l,1] - h[l,0])
     rel_cone  = DiD(3) - DiD(2), same residualisation
   with DiD(l) = (h[l,1] - h[l,0]) - (h[0,1] - h[0,0]).  Subtracting DiD(2) (the H0' null-factor
   DiD) and projecting off the action subspace is the hard-subspace analogue of
   ``C_int AND NOT C_action AND NOT C_null`` in ``geometry_conceptor.py``; the conceptor form is
   computed alongside (``fit_conceptor_set`` with ``Dnull``).  Scenes without level-2 cells use the
   unsubtracted DiD (flagged ``null_factor``).
2. Cross-arm comparison per subspace type: principal angles / Grassmann distance between the two
   arms' bases, raw (shared ambient coordinates) and after orthogonal Procrustes transport
   W = argmin_{W orthogonal} ||X_A W - X_B||_F fitted on the training scenes' matched per-token rows of the
   HAZARD-FREE cells only (levels 0 and 2; same cell, same step, same token id in both arms; train-fold
   centred) -- rows carrying the in-lane identity would let a d x d map absorb the contrast under test; the
   transported projected cosine ||(W^T Q_A)^T r_B|| / ||r_B|| of arm B's held-out row on arm A's
   transported subspace, against a matched-rank random subspace and against arm B's own within-arm
   LOSO cosine (and the mirror B -> A); a signed k=1 statistic cosine(r_B, W^T mean_A); and the
   COAST conceptor cross-similarity tr(C_A->B C_B) / sqrt(tr C_A->B^2 tr C_B^2) with C_A->B the
   transported conceptor (eigvecs @ W), against its matched-spectrum random control.
3. The registered prediction tests at ``--k-registered`` (default 4), unit = scene, scene bootstrap
   CIs (values are already one per scene, so the scene bootstrap is the cluster bootstrap) and
   two-sided sign-flip p:
   (i)   appearance subspaces shared: transported(A->B) - within(B) (report, CI, equivalence margin)
         and transported - random (should exceed 0); same B->A; at ``--appearance-group``.
   (ii)  relational subspace attaches to the solid identity: excess LOSO capture (cos^2 minus the
         random-subspace cos^2) of the pedestrian-DiD subspace minus that of the cone-DiD subspace,
         per arm; double-dissociation statistic DD_i = (ped - cone)_A,i - (ped - cone)_B,i per
         scene; registered p from a REFIT permutation null (ped<->cone relabelling of a scene in both arms
         with every LOSO subspace refitted -- sign-flipping precomputed LOSO scores is anti-conservative
         because scene i sits in every other fold's training set), one-sided, max-T over the sites of a
         (step, group) family with shared flips (Westfall-Young as in ``cgs_stats``).
   (iii) A's pedestrian-relational subspace transported into B scores B's pedestrian cells at chance
         and B's cone cells above chance after identity-token remapping (the pooled hazard tokens are
         the scene-level union shared by the level-1 and level-3 cells, which sit at the identical
         kind-paired pose); mirror for B's cone-relational subspace into A.  Reported, no threshold.
   (iv)  rank / quota: conceptor aperture at which the operator retains 80 % of the contrast energy
         (sum mu_j^2 lam_j / sum lam_j = 0.8) and the resulting quota tr(C)/d, plus the hard rank for
         80 % energy, per arm and type (in-sample descriptive).
4. ``<out>/cross_arm_map.json``: ranked tables per step / group (sorted by the double-dissociation
   t), ``registered_tests`` with the four blocks, ``interpretation_scope``.

``--synthetic`` writes two random dumps (arm B's hidden coordinates are a random orthogonal
rotation of arm A's, so raw ambient overlap is at chance and transport is required) with a planted
shared appearance subspace and a planted identity-attached relational subspace (same S_rel in
both arms, attached to level 1 in A and level 3 in B, plus an AdaLN-style hazard-scaled action
artifact shared by the H0' cells) and checks that the four tests come out as predicted;
``--synthetic --synthetic-null`` plants independent appearance subspaces and no relational
component and checks that they do not.

Descriptive / associational only; causal claims come from patching (``patch_site.py``,
``conceptor_patch.py``).  References: COAST arXiv:2605.17144 (conceptor similarity, matched-spectrum
control), Procrustes transport and principal angles as in ``geometry_higher_order.py``,
Westfall-Young max-T sign-flip as in ``cgs_stats.py``.
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
from cgs_stats import sign_flip_matrix, sign_flip_maxt  # noqa: E402
from geometry_conceptor import fit_conceptor_set  # noqa: E402
from geometry_localize import pool_site  # noqa: E402
from geometry_models import alpha_for_quota, conceptor_factored, conceptor_quota, conceptor_similarity, matched_spectrum_random  # noqa: E402
from protocol import NULL_CONTROL_HAZARD, OBJECT_HAZARD  # noqa: E402
from sonar_metrics import principal_angles_deg, random_subspace, svd_subspace  # noqa: E402
from stats_utils import cosine, finite, sign_flip_p  # noqa: E402
from token_groups import alias_note, frame_for_step, load_cell_groups, set_domain, union  # noqa: E402

PROTOCOL = "cgs-geometry-cross-arm-v0.1"
PED, NULL, OBJ = 1, NULL_CONTROL_HAZARD, OBJECT_HAZARD
ACTION_LEVELS = (0, NULL_CONTROL_HAZARD)  # hazard-free cells that define the action main effect and fit the transport
KS = (1, 2, 4, 8)
TYPES = ("app_ped", "app_cone", "app_ped_a0", "app_cone_a0", "rel_ped", "rel_cone")
APPEARANCE_TYPES = ("app_ped", "app_cone", "app_ped_a0", "app_cone_a0")
REGISTERED_APPEARANCE = {"app_ped": "app_ped_a0", "app_cone": "app_cone_a0"}  # test (i) is registered on the brake-cell (interaction-free) contrast
RELATIONAL_TYPES = ("rel_ped", "rel_cone")
DEFAULT_GROUPS = ("hazard", "corridor", "hazard_corridor")
NULL_DD_BAND, NULL_EXCESS_BAND = 0.3, 0.2  # synthetic-null magnitude bands (planted effects are ~5x larger)
INTERPRETATION_SCOPE = (
    "Descriptive/associational; causal claims come from patching. Subspaces, transports and conceptors are fitted on "
    "discovery scenes only (LOSO for every held-out score); the registered tests are reported with scene-level CIs and "
    "sign-flip p (max-T over sites for the double dissociation) and do not by themselves establish that either arm USES "
    "the subspace."
)


# --------------------------------------------------------------------------- #
# Small numerics
# --------------------------------------------------------------------------- #


def orthogonal_procrustes(M: np.ndarray) -> np.ndarray:
    """W = argmin_{W orthogonal} ||X_A W - X_B||_F from the cross-covariance M = X_A^T X_B (d x d)."""
    U, _, Vt = np.linalg.svd(M)
    return U @ Vt


def proj_cos(Q: np.ndarray, r: np.ndarray) -> float:
    """Cosine between r and its projection on span(Q) = ||Q^T r|| / ||r|| (in [0, 1]; k=1 is |cos|)."""
    n = float(np.linalg.norm(r))
    return float(np.linalg.norm(Q.T @ r) / n) if n > 0 and np.all(np.isfinite(r)) else float("nan")


def project_off(R: np.ndarray, Q: np.ndarray | None) -> np.ndarray:
    return R if Q is None or Q.size == 0 else R - (R @ Q) @ Q.T


def boot_mean(values: np.ndarray, n_boot: int, seed: int) -> dict[str, float]:
    """Scene bootstrap of the mean (vectorised twin of ``stats_utils.cluster_bootstrap_mean``; one value per scene,
    so the scene is the cluster)."""
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return {"mean": float("nan"), "median": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    rng = np.random.RandomState(seed)
    draws = x[rng.randint(0, n, size=(n_boot, n))].mean(axis=1)
    return {"mean": float(x.mean()), "median": float(np.median(x)), "ci_low": float(np.percentile(draws, 2.5)), "ci_high": float(np.percentile(draws, 97.5)), "n": int(n)}


def summ(values: np.ndarray, n_boot: int, seed: int, n_perm: int = 5000) -> dict[str, float]:
    return {**boot_mean(values, n_boot, seed), "sign_flip_p": sign_flip_p(np.asarray(values, dtype=np.float64), n_perm=n_perm, seed=seed)}


def ci_excludes_zero(s: dict[str, float], positive: bool = True) -> bool:
    lo, hi = s.get("ci_low"), s.get("ci_high")
    if lo is None or hi is None or not np.isfinite(lo) or not np.isfinite(hi):
        return False
    return bool(lo > 0) if positive else bool(hi < 0)


def ci_within(s: dict[str, float], margin: float) -> bool:
    lo, hi = s.get("ci_low"), s.get("ci_high")
    return bool(lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi) and lo > -margin and hi < margin)


def quota_for_capture(lam: np.ndarray, d: int, target: float = 0.8, lo: float = 1e-3, hi: float = 1e5) -> dict[str, float]:
    """Aperture at which the conceptor retains ``target`` of the row energy, sum mu_j^2 lam_j / sum lam_j
    (energy of x C), the quota tr(C)/d there, and the hard rank for the same energy fraction."""
    lam = np.sort(np.asarray(lam, dtype=np.float64))[::-1]
    tot = float(lam.sum())
    if lam.size == 0 or tot <= 0:
        return {"alpha": float("nan"), "quota": float("nan"), "hard_rank": 0, "target": target}

    def cap(alpha):
        mu = lam / (lam + alpha ** -2)
        return float(np.sum(mu**2 * lam) / tot)

    if cap(hi) < target:
        alpha = hi
    elif cap(lo) > target:
        alpha = lo
    else:
        a, b = np.log(lo), np.log(hi)
        for _ in range(60):
            m = 0.5 * (a + b)
            if cap(np.exp(m)) < target:
                a = m
            else:
                b = m
        alpha = float(np.exp(0.5 * (a + b)))
    mu = lam / (lam + alpha ** -2)
    hard = int(np.searchsorted(np.cumsum(lam) / tot, target) + 1)
    return {"alpha": float(alpha), "quota": float(np.sum(mu) / d), "effective_dims": float(np.sum(mu)), "hard_rank": hard, "target": target, "capture_at_alpha": cap(alpha)}


def transport_conceptor(c: dict[str, np.ndarray], W: np.ndarray) -> dict[str, np.ndarray]:
    """Row-vector directions map as v -> v W (x_B ~= x_A W)."""
    return {**c, "eigvecs": np.asarray(c["eigvecs"]) @ W}


# --------------------------------------------------------------------------- #
# Dump IO and alignment
# --------------------------------------------------------------------------- #


def load_arm(dump: Path, discovery_seeds: Path | None) -> dict[str, Any]:
    idx = json.loads((dump / "activations" / "index.json").read_text())
    cells = idx["cells"] if isinstance(idx, dict) else idx
    if discovery_seeds is not None:
        allowed = {int(v) for v in discovery_seeds.read_text().split() if v.strip()}
        cells = [c for c in cells if int(c["seed"]) in allowed]
    meta = idx if isinstance(idx, dict) else {}
    return {"dump": Path(dump), "cells": cells, "n_steps": int(meta.get("n_steps") or 3), "offset": int(meta.get("group_frame_offset", 0)), "domain": meta.get("domain", "driving"), "index": meta}


def align_arms(arm_a: dict[str, Any], arm_b: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str], dict[str, str]]:
    """Cells present in both dumps keyed by (pair_id, hazard, action); scenes complete for levels {0, 1, 3} x {0, 1}."""

    def key(c):
        return (str(c["pair_id"]), int(c["hazard"]), int(c["candidate_action"]))

    ka = {key(c): c for c in arm_a["cells"]}
    kb = {key(c): c for c in arm_b["cells"]}
    common = sorted(set(ka) & set(kb))
    by_scene: dict[str, set] = {}
    for k in common:
        by_scene.setdefault(k[0], set()).add((k[1], k[2]))
    need = {(lv, a) for lv in (0, PED, OBJ) for a in (0, 1)}
    scenes = sorted(s for s, ks in by_scene.items() if need <= ks)
    cells = []
    for k in common:
        if k[0] not in scenes:
            continue
        ca = ka[k]
        cells.append({"row": len(cells), "row_a": int(ca["row"]), "row_b": int(kb[k]["row"]), "pair_id": k[0], "hazard": k[1], "candidate_action": k[2],
                      "cell_id": ca["cell_id"], "seed": ca.get("seed"), "hazard_bbox_xyxy": ca.get("hazard_bbox_xyxy"), "target_bbox_xyxy": ca.get("target_bbox_xyxy")})
    n_null = sum(1 for s in scenes if {(NULL, 0), (NULL, 1)} <= by_scene[s])
    status = {"null_factor": "none" if n_null == 0 else ("full" if n_null == len(scenes) else "partial"), "n_null_scenes": n_null,
              "n_cells_a": len(arm_a["cells"]), "n_cells_b": len(arm_b["cells"]), "n_common_cells": len(common), "n_scenes_dropped_incomplete": len(by_scene) - len(scenes)}
    return cells, scenes, status


def token_sets_for(stimulus: Path, cells: list[dict[str, Any]], groups: tuple[str, ...], n_steps: int, offset: int, domain: str) -> dict[tuple[str, int, str], np.ndarray]:
    """Per (pair_id, step, group): union over the scene's cells of the group's token ids (domain-aware masks)."""
    per_cell = {c["cell_id"]: load_cell_groups(stimulus, {"cell_id": c["cell_id"], "hazard_bbox_xyxy": c.get("hazard_bbox_xyxy"), "target_bbox_xyxy": c.get("target_bbox_xyxy")}, domain=domain) for c in cells}
    by_pair: dict[str, list] = {}
    for c in cells:
        by_pair.setdefault(c["pair_id"], []).append(c)
    out = {}
    for pid, members in by_pair.items():
        for step in range(n_steps):
            frame = frame_for_step(step, offset)
            for g in groups:
                out[(pid, step, g)] = np.arange(256) if g == "all" else union(*(per_cell[m["cell_id"]].group(frame, g) for m in members))
    return out


def load_site(dump: Path, site: str, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.load(dump / "activations" / f"{site}.npz") as z:
        acts = np.asarray(z["acts"])[rows].astype(np.float32)
        tix = np.asarray(z["token_index"])[rows] if "token_index" in z else np.broadcast_to(np.arange(acts.shape[2]), acts.shape[:3]).copy()
    return acts, tix


def token_rows(acts: np.ndarray, tix: np.ndarray, cells: list[dict[str, Any]], step: int, levels: tuple[int, ...]) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray]:
    """Per-token rows (uncentred) of every cell whose hazard level is in ``levels`` at ``step``.
    Returns (X [N, d], keys (cell_row, token), scene [N]).  The transport is fitted on hazard-free cells only
    (default levels 0 = sidewalk and 2 = H0' mirror pose): rows that carry the in-lane identity would let a d x d
    orthogonal map absorb the very contrast under test (the synthetic null exposed this)."""
    keys, vecs, scene = [], [], []
    for c in cells:
        if int(c["hazard"]) not in levels:
            continue
        r = c["row"]
        ti = tix[r, step]
        m = ti >= 0
        for t, v in zip(ti[m].tolist(), acts[r, step, m]):
            keys.append((r, int(t)))
            vecs.append(v)
            scene.append(c["pair_id"])
    return np.asarray(vecs, dtype=np.float64), keys, np.asarray(scene)


def matched_token_rows(A: tuple, B: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intersect the two arms' per-token rows on (cell, token id)."""
    XA, keysA, sceneA = A
    XB, keysB, _ = B
    pos = {k: i for i, k in enumerate(keysA)}
    ia, ib = [], []
    for j, k in enumerate(keysB):
        i = pos.get(k)
        if i is not None:
            ia.append(i)
            ib.append(j)
    ia, ib = np.asarray(ia, dtype=int), np.asarray(ib, dtype=int)
    return XA[ia], XB[ib], sceneA[ia]


def transport_folds(XA: np.ndarray, XB: np.ndarray, scene: np.ndarray, scenes: list[str]) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    """LOSO orthogonal Procrustes maps on train-fold-centred rows (held-out scene removed from the cross-covariance
    and from the means) plus the full-data map; held-out relative residuals as the transport-quality diagnostic."""
    M_all = XA.T @ XB
    sA, sB = XA.sum(0), XB.sum(0)
    n = len(XA)

    def solve(M_tr, sA_tr, sB_tr, n_tr):
        return orthogonal_procrustes(M_tr - np.outer(sA_tr, sB_tr) / n_tr)

    W_full = solve(M_all, sA, sB, n)
    folds, res_t, res_r = {}, [], []
    for s in scenes:
        te = scene == s
        n_tr = n - int(te.sum())
        if n_tr < 2:
            folds[s] = W_full
            continue
        sA_tr, sB_tr = sA - XA[te].sum(0), sB - XB[te].sum(0)
        W = solve(M_all - XA[te].T @ XB[te], sA_tr, sB_tr, n_tr)
        folds[s] = W
        a, b = XA[te] - sA_tr / n_tr, XB[te] - sB_tr / n_tr
        den = max(float(np.linalg.norm(b)), 1e-12)
        res_t.append(float(np.linalg.norm(a @ W - b) / den))
        res_r.append(float(np.linalg.norm(a - b) / den))
    a, b = XA - sA / n, XB - sB / n
    return folds, W_full, {"n_rows": int(n), "relative_residual_transported_full": float(np.linalg.norm(a @ W_full - b) / max(np.linalg.norm(b), 1e-12)),
                           "relative_residual_raw_full": float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12)),
                           "heldout_relative_residual_transported": float(np.mean(res_t)) if res_t else float("nan"),
                           "heldout_relative_residual_raw": float(np.mean(res_r)) if res_r else float("nan")}


# --------------------------------------------------------------------------- #
# Contrast rows
# --------------------------------------------------------------------------- #


def scene_rows(P: np.ndarray, cells: list[dict[str, Any]], scenes: list[str]) -> dict[str, np.ndarray]:
    """Per-scene contrast rows from group-pooled cell vectors ``P[row]`` (nan where a level is absent)."""
    by: dict[str, dict[tuple[int, int], np.ndarray]] = {}
    for c in cells:
        by.setdefault(c["pair_id"], {})[(int(c["hazard"]), int(c["candidate_action"]))] = P[c["row"]]
    d = P.shape[1]
    out = {k: np.full((len(scenes), d), np.nan) for k in ("app_ped", "app_cone", "app_ped_a0", "app_cone_a0", "did_ped", "did_cone", "did_null", "action")}
    for i, s in enumerate(scenes):
        h = by[s]
        base = h[(0, 1)] - h[(0, 0)]
        out["app_ped"][i] = 0.5 * ((h[(PED, 0)] - h[(0, 0)]) + (h[(PED, 1)] - h[(0, 1)]))
        out["app_cone"][i] = 0.5 * ((h[(OBJ, 0)] - h[(0, 0)]) + (h[(OBJ, 1)] - h[(0, 1)]))
        # brake-cell appearance = main effect - DiD/2: the action-averaged main effect of the SOLID identity carries half
        # the relational DiD by construction, so the interaction-free appearance contrast is the one under braking
        out["app_ped_a0"][i] = h[(PED, 0)] - h[(0, 0)]
        out["app_cone_a0"][i] = h[(OBJ, 0)] - h[(0, 0)]
        out["did_ped"][i] = (h[(PED, 1)] - h[(PED, 0)]) - base
        out["did_cone"][i] = (h[(OBJ, 1)] - h[(OBJ, 0)]) - base
        # action main effect from hazard-free cells only (levels 0 and, when present, 2): averaging over levels 1/3 would
        # put the identity x action interaction itself into the "action" subspace that is projected off
        levels = [lv for lv in ACTION_LEVELS if (lv, 0) in h and (lv, 1) in h]
        out["action"][i] = np.mean([h[(lv, 1)] - h[(lv, 0)] for lv in levels], axis=0)
        if (NULL, 0) in h and (NULL, 1) in h:
            out["did_null"][i] = (h[(NULL, 1)] - h[(NULL, 0)]) - base
    return out


def relational_rows(rows: dict[str, np.ndarray], train: np.ndarray, k_action: int) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    """rel_x = DiD(x) - DiD(H0') (where H0' exists) projected off the train-fold action main-effect subspace.
    Returns the rows and the action subspace Q_act [d, k_action] (None when not fitted)."""
    act = rows["action"][train]
    act = act[np.all(np.isfinite(act), axis=1)]
    Q_act = svd_subspace(act, k_action) if len(act) >= 1 and k_action > 0 else None
    out = {}
    for name in ("ped", "cone"):
        R = rows[f"did_{name}"].copy()
        has_null = np.all(np.isfinite(rows["did_null"]), axis=1)
        R[has_null] = R[has_null] - rows["did_null"][has_null]
        out[f"rel_{name}"] = project_off(R, Q_act)
    return out, Q_act


def residual_random_subspace(Qr: np.ndarray, Q_act: np.ndarray | None) -> np.ndarray:
    """Random k-subspace inside the complement of Q_act (the space the relational rows live in): the chance reference
    must satisfy the same constraint as the rows, otherwise a subspace that is merely orthogonal to the shared action
    direction scores above 'chance' (synthetic null check)."""
    if Q_act is None or Q_act.size == 0:
        return Qr
    q, _ = np.linalg.qr(project_off(Qr.T, Q_act).T)
    return q


def capture_k(X: np.ndarray, r: np.ndarray, k: int) -> float:
    """||Q_k^T r||^2 / ||r||^2 with Q_k the top-k UNCENTRED right singular subspace of X [n, d] (n << d), via the n x n Gram
    matrix: Q_k^T r = diag(w_k)^-1/2 U_k^T (X r).  Same subspace as ``svd_subspace``."""
    n2 = float(r @ r)
    if n2 <= 0 or len(X) == 0:
        return float("nan")
    G = X @ X.T
    w, U = np.linalg.eigh(G)
    order = np.argsort(w)[::-1]
    w, U = w[order], U[:, order]
    keep = w > 1e-12 * max(w[0], 1e-300)
    w, U = w[keep][:k], U[:, keep][:, :k]
    z = (U.T @ (X @ r)) / np.sqrt(w)
    return float(z @ z / n2)


def dd_refit_statistic(fold_rows: dict[str, dict[str, np.ndarray]], rand_cap: dict[str, dict[str, np.ndarray]], k: int, signs: np.ndarray | None) -> np.ndarray:
    """Per-scene double-dissociation DD_i under a ped<->cone relabelling of the scenes with sign -1 (applied in BOTH
    arms, so DD_i flips sign under the null of no double dissociation), with every LOSO subspace REFITTED on the
    relabelled training rows.  ``fold_rows[arm][type]`` is [S_folds, S, d]: row j residualised with fold i's action
    subspace; ``rand_cap[arm][type]`` [S] is the residual-random-subspace cos^2 of held-out row i in fold i."""
    S = next(iter(fold_rows["A"].values())).shape[0]
    sg = np.ones(S) if signs is None else np.asarray(signs)
    out = np.full(S, np.nan)
    for i in range(S):
        tr = np.asarray([j for j in range(S) if j != i])
        swap_tr = (sg[tr] < 0)[:, None]
        val = {}
        for arm in ("A", "B"):
            Rp, Rc = fold_rows[arm]["rel_ped"][i], fold_rows[arm]["rel_cone"][i]
            ped_tr = np.where(swap_tr, Rc[tr], Rp[tr])
            cone_tr = np.where(swap_tr, Rp[tr], Rc[tr])
            okp, okc = np.all(np.isfinite(ped_tr), axis=1), np.all(np.isfinite(cone_tr), axis=1)
            if sg[i] > 0:
                rp, rc, qp, qc = Rp[i], Rc[i], rand_cap[arm]["rel_ped"][i], rand_cap[arm]["rel_cone"][i]
            else:
                rp, rc, qp, qc = Rc[i], Rp[i], rand_cap[arm]["rel_cone"][i], rand_cap[arm]["rel_ped"][i]
            val[arm] = (capture_k(ped_tr[okp], rp, k) - qp) - (capture_k(cone_tr[okc], rc, k) - qc)
        out[i] = val["A"] - val["B"]
    return out


def t_stat(v: np.ndarray) -> float:
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return float("nan")
    sd = v.std(ddof=1)
    return float(v.mean() / (sd / np.sqrt(len(v)))) if sd >= 1e-8 else float("nan")


def dd_refit_null(fold_rows, rand_cap, k: int, flips: np.ndarray) -> dict[str, Any]:
    obs = dd_refit_statistic(fold_rows, rand_cap, k, None)
    null = np.asarray([t_stat(dd_refit_statistic(fold_rows, rand_cap, k, f)) for f in flips])
    t_obs = t_stat(obs)
    ok = np.isfinite(null)
    p_raw = float((np.sum(null[ok] >= t_obs - 1e-12) + 1) / (ok.sum() + 1)) if np.isfinite(t_obs) else float("nan")
    return {"observed_per_scene": obs, "t": t_obs, "p_raw_one_sided": p_raw, "null_t": null, "n_perm": int(len(flips))}


# --------------------------------------------------------------------------- #
# Per (site, step, group) analysis
# --------------------------------------------------------------------------- #


def analyze_group(rowsA: dict[str, np.ndarray], rowsB: dict[str, np.ndarray], scenes: list[str], W_folds: dict[str, np.ndarray], W_full: np.ndarray, args, seed: int) -> dict[str, Any]:
    S = len(scenes)
    d = rowsA["app_ped"].shape[1]
    ks = tuple(k for k in args.ks if k <= S - 2) or (1,)
    kmax = max(ks)
    rng = np.random.RandomState(seed)
    stats = ("within_A", "within_B", "trans_AB", "trans_BA", "raw_AB", "raw_BA", "rand_A", "rand_B", "signed_within_A", "signed_within_B", "signed_trans_AB", "signed_trans_BA")
    per = {t: {k: {s: np.full(S, np.nan) for s in stats} for k in ks} for t in TYPES}
    angles = {t: {k: {"transported": [], "raw": []} for k in ks} for t in TYPES}
    cross = {k: {s: np.full(S, np.nan) for s in ("A_ped_to_B_ped", "A_ped_to_B_cone", "B_cone_to_A_cone", "B_cone_to_A_ped")} for k in ks}
    R_full = {"A": dict(rowsA), "B": dict(rowsB)}
    fold_rows = {arm: {t: np.full((S, S, d), np.nan) for t in RELATIONAL_TYPES} for arm in ("A", "B")}
    for i, s in enumerate(scenes):
        train = np.asarray([j for j in range(S) if j != i])
        W = W_folds[s]
        relA, QactA = relational_rows(rowsA, train, args.k_action)
        relB, QactB = relational_rows(rowsB, train, args.k_action)
        for t in RELATIONAL_TYPES:
            fold_rows["A"][t][i], fold_rows["B"][t][i] = relA[t], relB[t]
        RA, RB = {**rowsA, **relA}, {**rowsB, **relB}
        Qr_full = random_subspace(d, kmax, rng)
        Qr_res = {"A": residual_random_subspace(Qr_full, QactA), "B": residual_random_subspace(Qr_full, QactB)}
        Q = {}
        for t in TYPES:
            trA, trB = RA[t][train], RB[t][train]
            okA, okB = np.all(np.isfinite(trA), axis=1), np.all(np.isfinite(trB), axis=1)
            if okA.sum() < 2 or okB.sum() < 2:
                continue
            QA_full, QB_full = svd_subspace(trA[okA], kmax), svd_subspace(trB[okB], kmax)
            mA, mB = trA[okA].mean(0), trB[okB].mean(0)
            rA, rB = RA[t][i], RB[t][i]
            for k in ks:
                kk = min(k, QA_full.shape[1], QB_full.shape[1])
                QA, QB = QA_full[:, :kk], QB_full[:, :kk]
                relational = t in RELATIONAL_TYPES
                QrA = (Qr_res["A"] if relational else Qr_full)[:, :kk]
                QrB = (Qr_res["B"] if relational else Qr_full)[:, :kk]
                QA_B, QB_A = W.T @ QA, W @ QB  # A's basis expressed in B coordinates and vice versa
                Q[(t, k)] = (QA, QB, QA_B, QB_A)
                p = per[t][k]
                p["within_A"][i], p["within_B"][i] = proj_cos(QA, rA), proj_cos(QB, rB)
                p["trans_AB"][i], p["trans_BA"][i] = proj_cos(QA_B, rB), proj_cos(QB_A, rA)
                p["raw_AB"][i], p["raw_BA"][i] = proj_cos(QA, rB), proj_cos(QB, rA)
                p["rand_A"][i], p["rand_B"][i] = proj_cos(QrA, rA), proj_cos(QrB, rB)
                p["signed_within_A"][i], p["signed_within_B"][i] = cosine(rA, mA), cosine(rB, mB)
                p["signed_trans_AB"][i], p["signed_trans_BA"][i] = cosine(rB, W.T @ mA), cosine(rA, W @ mB)
                angles[t][k]["transported"].append(principal_angles_deg(QA_B, QB))
                angles[t][k]["raw"].append(principal_angles_deg(QA, QB))
        for k in ks:
            if ("rel_ped", k) in Q and ("rel_cone", k) in Q:
                QA_ped_B = Q[("rel_ped", k)][2]
                QB_cone_A = Q[("rel_cone", k)][3]
                cross[k]["A_ped_to_B_ped"][i] = proj_cos(QA_ped_B, RB["rel_ped"][i])
                cross[k]["A_ped_to_B_cone"][i] = proj_cos(QA_ped_B, RB["rel_cone"][i])
                cross[k]["B_cone_to_A_cone"][i] = proj_cos(QB_cone_A, RA["rel_cone"][i])
                cross[k]["B_cone_to_A_ped"][i] = proj_cos(QB_cone_A, RA["rel_ped"][i])
    # full-data relational rows (for conceptors / quotas / in-sample angles; descriptive)
    all_idx = np.arange(S)
    R_full["A"].update(relational_rows(rowsA, all_idx, args.k_action)[0])
    R_full["B"].update(relational_rows(rowsB, all_idx, args.k_action)[0])
    out: dict[str, Any] = {"ks": list(ks), "n_scenes": S, "subspaces": {}, "cross_identity": {}, "conceptor": {}, "quota80": {}}
    nb, sd = args.n_boot, seed
    # registered test (ii): refit permutation null (ped<->cone relabelling per scene, LOSO subspaces refitted); the flip
    # matrix depends only on (seed, S) so every site of a (step, group) family shares it -> max-T is valid
    kr = args.k_registered if args.k_registered in ks else max(ks)
    rand_cap = {arm: {t: per[t][kr][f"rand_{arm}"] ** 2 for t in RELATIONAL_TYPES} for arm in ("A", "B")}
    flips = sign_flip_matrix(S, args.n_perm_refit, np.random.default_rng(seed + 12345))
    out["dd_refit"] = {**dd_refit_null(fold_rows, rand_cap, kr, flips), "k": kr}
    for t in TYPES:
        out["subspaces"][t] = {}
        for k in ks:
            p = per[t][k]
            e = {s: summ(p[s], nb, sd) for s in stats}
            e["trans_AB_minus_within_B"] = summ(p["trans_AB"] - p["within_B"], nb, sd)
            e["trans_BA_minus_within_A"] = summ(p["trans_BA"] - p["within_A"], nb, sd)
            e["trans_AB_minus_rand_B"] = summ(p["trans_AB"] - p["rand_B"], nb, sd)
            e["trans_BA_minus_rand_A"] = summ(p["trans_BA"] - p["rand_A"], nb, sd)
            e["raw_AB_minus_rand_B"] = summ(p["raw_AB"] - p["rand_B"], nb, sd)
            e["within_A_minus_rand_A"] = summ(p["within_A"] - p["rand_A"], nb, sd)
            e["within_B_minus_rand_B"] = summ(p["within_B"] - p["rand_B"], nb, sd)
            e["excess_capture_A"] = summ(p["within_A"] ** 2 - p["rand_A"] ** 2, nb, sd)
            e["excess_capture_B"] = summ(p["within_B"] ** 2 - p["rand_B"] ** 2, nb, sd)
            e["_per_scene"] = {s: p[s] for s in stats}
            ang_t, ang_r = np.asarray(angles[t][k]["transported"]), np.asarray(angles[t][k]["raw"])
            fa, fb = R_full["A"][t], R_full["B"][t]
            fa, fb = fa[np.all(np.isfinite(fa), axis=1)], fb[np.all(np.isfinite(fb), axis=1)]
            kk = min(k, len(fa) - 1 if len(fa) > 1 else 1, len(fb) - 1 if len(fb) > 1 else 1)
            if len(fa) >= 2 and len(fb) >= 2:
                QA, QB = svd_subspace(fa, kk), svd_subspace(fb, kk)
                pa_t, pa_r = principal_angles_deg(W_full.T @ QA, QB), principal_angles_deg(QA, QB)
            else:
                pa_t = pa_r = []
            e["principal_angles_deg"] = {
                "full_fit_transported": pa_t, "full_fit_raw": pa_r,
                "grassmann_transported": float(np.sqrt(np.sum(np.radians(pa_t) ** 2))) if len(pa_t) else float("nan"),
                "grassmann_raw": float(np.sqrt(np.sum(np.radians(pa_r) ** 2))) if len(pa_r) else float("nan"),
                "loso_mean_transported": ang_t.mean(0).tolist() if ang_t.size else [], "loso_mean_raw": ang_r.mean(0).tolist() if ang_r.size else [],
                "chance_expected_cos": float(np.sqrt(k / (d - args.k_action))) if t in RELATIONAL_TYPES else float(np.sqrt(k / d)),
                "random_control": "matched-rank random subspace inside the complement of the action subspace" if t in RELATIONAL_TYPES else "matched-rank random subspace",
            }
            out["subspaces"][t][f"k{k}"] = e
    for k in ks:
        c = cross[k]
        rb_ped, rb_cone = per["rel_ped"][k]["rand_B"], per["rel_cone"][k]["rand_B"]
        ra_ped, ra_cone = per["rel_ped"][k]["rand_A"], per["rel_cone"][k]["rand_A"]
        out["cross_identity"][f"k{k}"] = {
            "A_ped_to_B_ped": summ(c["A_ped_to_B_ped"], nb, sd), "A_ped_to_B_cone": summ(c["A_ped_to_B_cone"], nb, sd),
            "A_ped_to_B_ped_minus_random": summ(c["A_ped_to_B_ped"] - rb_ped, nb, sd), "A_ped_to_B_cone_minus_random": summ(c["A_ped_to_B_cone"] - rb_cone, nb, sd),
            "A_ped_to_B_cone_minus_ped": summ(c["A_ped_to_B_cone"] - c["A_ped_to_B_ped"], nb, sd),
            "B_cone_to_A_cone": summ(c["B_cone_to_A_cone"], nb, sd), "B_cone_to_A_ped": summ(c["B_cone_to_A_ped"], nb, sd),
            "B_cone_to_A_cone_minus_random": summ(c["B_cone_to_A_cone"] - ra_cone, nb, sd), "B_cone_to_A_ped_minus_random": summ(c["B_cone_to_A_ped"] - ra_ped, nb, sd),
            "B_cone_to_A_ped_minus_cone": summ(c["B_cone_to_A_ped"] - c["B_cone_to_A_cone"], nb, sd),
            "_per_scene": {**c, "rand_B_ped": rb_ped, "rand_B_cone": rb_cone, "rand_A_ped": ra_ped, "rand_A_cone": ra_cone},
        }
    # conceptors (full fit, descriptive) + quotas
    cons = {}
    for arm in ("A", "B"):
        R = R_full[arm]
        cons[arm] = {}
        for t in TYPES:
            X = R[t]
            X = X[np.all(np.isfinite(X), axis=1)]
            if len(X) < 2:
                continue
            scale = float(np.sqrt(np.mean(np.sum(X**2, axis=1)))) or 1.0
            Xs = X / scale
            base = conceptor_factored(Xs, 1.0, center=False)
            alpha = alpha_for_quota(base["lam"], d, args.target_dims / d)
            if t in RELATIONAL_TYPES:
                Draw = R[f"did_{t.split('_')[1]}"]
                A = R["action"]
                Dn = R["did_null"]
                okr = np.all(np.isfinite(Draw), axis=1) & np.all(np.isfinite(A), axis=1)
                Dn = Dn[np.all(np.isfinite(Dn), axis=1)]
                cs = fit_conceptor_set(Draw[okr] / scale, A[okr] / scale, alpha, Dnull=(Dn / scale) if len(Dn) >= 2 else None, seed=seed)
                cons[arm][t] = cs["safety"]
                out["conceptor"].setdefault(arm, {})[t] = {"alpha": alpha, "quota_safety": conceptor_quota(cs["safety"], d), "quota_int": conceptor_quota(cs["int"], d),
                                                          "overlap_int_action": conceptor_similarity(cs["int"], cs["action"]), "form": "C_did AND NOT C_action" + (" AND NOT C_null" if len(Dn) >= 2 else "")}
            else:
                c = conceptor_factored(Xs, alpha, center=False)
                cons[arm][t] = c
                out["conceptor"].setdefault(arm, {})[t] = {"alpha": alpha, "quota": conceptor_quota(c, d), "form": "C_appearance"}
            out["quota80"].setdefault(arm, {})[t] = quota_for_capture(base["lam"], d, 0.8)
    xs = {}
    for t in TYPES:
        if t in cons["A"] and t in cons["B"]:
            cAB = transport_conceptor(cons["A"][t], W_full)
            xs[t] = {"transported": conceptor_similarity(cAB, cons["B"][t]), "raw": conceptor_similarity(cons["A"][t], cons["B"][t]),
                     "matched_spectrum_random": conceptor_similarity(matched_spectrum_random(cAB, seed=seed), cons["B"][t])}
    if "rel_ped" in cons["A"] and "rel_cone" in cons["B"] and "rel_ped" in cons["B"]:
        cAB = transport_conceptor(cons["A"]["rel_ped"], W_full)
        xs["A_rel_ped_vs_B"] = {"B_rel_cone": conceptor_similarity(cAB, cons["B"]["rel_cone"]), "B_rel_ped": conceptor_similarity(cAB, cons["B"]["rel_ped"]),
                                "matched_spectrum_random_vs_B_rel_cone": conceptor_similarity(matched_spectrum_random(cAB, seed=seed), cons["B"]["rel_cone"])}
    out["conceptor"]["cross_similarity"] = xs
    # kept in memory for ``--export-transport`` (stripped before the JSON is written): the factored conceptors of both
    # arms per type and the unit mean row (rank-one control direction) of the same rows
    out["_cons"] = cons
    out["_rank_one"] = {}
    for arm in ("A", "B"):
        out["_rank_one"][arm] = {}
        for t in TYPES:
            X = R_full[arm][t]
            X = X[np.all(np.isfinite(X), axis=1)]
            if len(X) >= 1:
                m = X.mean(0)
                n = float(np.linalg.norm(m))
                out["_rank_one"][arm][t] = m / n if n > 0 else m
    return out


# --------------------------------------------------------------------------- #
# Registered tests
# --------------------------------------------------------------------------- #


def registered_tests(entries: list[dict[str, Any]], scenes: list[str], args) -> dict[str, Any]:
    k = f"k{args.k_registered}"
    by = {(e["site_id"], e["step"], e["group"]): e for e in entries if e.get("status") == "ok"}
    steps = sorted({s for _, s, _ in by})
    groups = sorted({g for _, _, g in by})
    rng = np.random.default_rng(args.seed)
    res: dict[str, Any] = {
        "k_registered": args.k_registered, "appearance_group": args.appearance_group, "equivalence_margin": args.equiv_margin,
        "unit": "scene (pair_id); CIs = scene bootstrap of the mean; sign_flip_p = two-sided sign-flip of the per-scene LOSO scores (descriptive, approximate: scene i sits in every other fold); test (ii) uses a refit permutation null; max-T = Westfall-Young over the sites of a (step, group) family with shared flips",
        "i_appearance_shared": {}, "ii_identity_attachment": {}, "iii_transport_identity_remap": {}, "iv_rank_quota": {},
        "pattern_flags_note": "pattern_consistent booleans are descriptive summaries of the registered directions, not decision thresholds",
    }
    for step in steps:
        # (i) at the appearance group
        g = args.appearance_group
        blk = {}
        for site in sorted({s for s, st, gg in by if st == step and gg == g}):
            e = by[(site, step, g)]
            if k not in e["analysis"]["subspaces"]["app_ped"]:
                continue
            r = {}
            for t in APPEARANCE_TYPES:
                sub = e["analysis"]["subspaces"][t][k]
                ab = {"transported_minus_within": sub["trans_AB_minus_within_B"], "transported_minus_random": sub["trans_AB_minus_rand_B"], "raw_minus_random": sub["raw_AB_minus_rand_B"],
                      "within": sub["within_B"], "transported": sub["trans_AB"], "random": sub["rand_B"], "signed_transported_k1": sub["signed_trans_AB"], "signed_within_k1": sub["signed_within_B"]}
                ba = {"transported_minus_within": sub["trans_BA_minus_within_A"], "transported_minus_random": sub["trans_BA_minus_rand_A"],
                      "within": sub["within_A"], "transported": sub["trans_BA"], "random": sub["rand_A"], "signed_transported_k1": sub["signed_trans_BA"], "signed_within_k1": sub["signed_within_A"]}
                ok = all(ci_excludes_zero(x["transported_minus_random"]) and ci_within(x["transported_minus_within"], args.equiv_margin) for x in (ab, ba))
                r[t] = {"A_to_B": ab, "B_to_A": ba, "principal_angles_deg": sub["principal_angles_deg"], "shared_pattern": bool(ok),
                        "registered": t in REGISTERED_APPEARANCE.values(),
                        "note": "action-averaged main effect: contains DiD/2 of the solid identity by construction" if t in REGISTERED_APPEARANCE else "brake-cell contrast h[l,0] - h[0,0] (interaction-free by design)"}
            for main, a0 in REGISTERED_APPEARANCE.items():
                r[main]["pattern_consistent"] = r[a0]["shared_pattern"]
                r[a0]["pattern_consistent"] = r[a0]["shared_pattern"]
            blk[site] = r
        res["i_appearance_shared"][f"s{step}|{g}"] = blk
        for g in groups:
            sites = sorted({s for s, st, gg in by if st == step and gg == g})
            dd_vec, per_site, null_t = {}, {}, {}
            for site in sites:
                e = by[(site, step, g)]
                sp = e["analysis"]["subspaces"]
                if k not in sp["rel_ped"] or k not in sp["rel_cone"]:
                    continue
                pa, ca = sp["rel_ped"][k]["_per_scene"], sp["rel_cone"][k]["_per_scene"]
                exA = (pa["within_A"] ** 2 - pa["rand_A"] ** 2) - (ca["within_A"] ** 2 - ca["rand_A"] ** 2)
                exB = (pa["within_B"] ** 2 - pa["rand_B"] ** 2) - (ca["within_B"] ** 2 - ca["rand_B"] ** 2)
                dd = exA - exB
                rf = e["analysis"]["dd_refit"]
                dd_vec[site] = dd[:, None]
                null_t[site] = np.asarray(rf["null_t"], dtype=np.float64)
                per_site[site] = {"ped_minus_cone_A": summ(exA, args.n_boot, args.seed), "ped_minus_cone_B": summ(exB, args.n_boot, args.seed),
                                  "double_dissociation": {**summ(dd, args.n_boot, args.seed), "t": rf["t"], "p_raw_one_sided": rf["p_raw_one_sided"], "n_perm": rf["n_perm"],
                                                          "refit_matches_scores": bool(np.allclose(np.asarray(rf["observed_per_scene"]), dd, atol=1e-6, equal_nan=True))},
                                  "excess_capture": {t: {"A": sp[t][k]["excess_capture_A"], "B": sp[t][k]["excess_capture_B"]} for t in RELATIONAL_TYPES}}
            if dd_vec:
                # naive sign-flip of the precomputed LOSO scores (descriptive; anti-conservative because scene i sits in every other fold)
                mt_scores = sign_flip_maxt(dd_vec, args.n_perm, rng, statistic=lambda v, signs: v[:, 0] * (1.0 if signs is None else signs))
                # registered: refit permutation null with shared flips -> max-T over the family's sites
                names = sorted(null_t)
                N = np.stack([null_t[n] for n in names], axis=1)  # [n_perm, n_sites]
                with np.errstate(invalid="ignore"):
                    max_null = np.nanmax(N, axis=1)
                fin = max_null[np.isfinite(max_null)]
                for site, r in per_site.items():
                    t_obs = r["double_dissociation"]["t"]
                    r["double_dissociation"]["p_maxt_fwer"] = float((np.sum(fin >= t_obs - 1e-12) + 1) / (len(fin) + 1)) if np.isfinite(t_obs) else float("nan")
                    r["double_dissociation"]["p_scores_sign_flip_raw"] = mt_scores["sites"][site]["p_raw"]
                    r["double_dissociation"]["p_scores_sign_flip_maxt"] = mt_scores["sites"][site]["p_maxt_fwer"]
                    r["pattern_consistent"] = bool(np.isfinite(r["double_dissociation"]["p_maxt_fwer"]) and r["double_dissociation"]["p_maxt_fwer"] < 0.05 and r["ped_minus_cone_A"]["mean"] > 0 and r["ped_minus_cone_B"]["mean"] < 0)
                res["ii_identity_attachment"][f"s{step}|{g}"] = {
                    "statistic": "DD_i = (excess_capture_ped - excess_capture_cone)_A,i - (same)_B,i; excess capture = LOSO cos^2 minus residual-random-subspace cos^2",
                    "null": "ped<->cone relabelling per scene (same sign in both arms), every LOSO subspace refitted; one-sided (DD > 0); max-T over the family's sites with shared flips",
                    "maxt": {"n_scenes": len(scenes), "n_perm": int(N.shape[0]), "n_sites": len(names), "exact": bool(2 ** len(scenes) <= args.n_perm_refit),
                             "maxt_null_quantiles": {q: float(np.quantile(fin, q)) for q in (0.5, 0.9, 0.95, 0.99)} if len(fin) else {}},
                    "sites": per_site}
            blk3 = {}
            for site in sites:
                e = by[(site, step, g)]
                ci = e["analysis"]["cross_identity"].get(k)
                if ci is None:
                    continue
                r = {kk: v for kk, v in ci.items() if kk != "_per_scene"}
                band = args.equiv_margin
                r["pattern_consistent"] = bool(ci_excludes_zero(ci["A_ped_to_B_cone_minus_random"]) and ci_excludes_zero(ci["A_ped_to_B_cone_minus_ped"]) and abs(ci["A_ped_to_B_ped_minus_random"]["mean"]) < band
                                          and ci_excludes_zero(ci["B_cone_to_A_ped_minus_random"]) and ci_excludes_zero(ci["B_cone_to_A_ped_minus_cone"]) and abs(ci["B_cone_to_A_cone_minus_random"]["mean"]) < band)
                r["pattern_rule"] = f"solid-identity cells: transported-minus-random CI > 0 and (solid - ghost) CI > 0; ghost-identity cells: |transported - random| < {band} (chance band); both directions"
                r["conceptor_cross_similarity"] = e["analysis"]["conceptor"]["cross_similarity"].get("A_rel_ped_vs_B")
                blk3[site] = r
            res["iii_transport_identity_remap"][f"s{step}|{g}"] = {"identity_token_remapping": "pooled over the scene-level union of the group's tokens, shared by the level-1 (pedestrian) and level-3 (cone) cells at the identical kind-paired pose", "sites": blk3}
            blk4 = {}
            for site in sites:
                q = by[(site, step, g)]["analysis"]["quota80"]
                if not (q.get("A") and q.get("B")):
                    continue
                r = {"A": q["A"], "B": q["B"]}
                try:
                    r["pattern_consistent"] = bool(q["A"]["rel_ped"]["quota"] < q["A"]["rel_cone"]["quota"] and q["B"]["rel_cone"]["quota"] < q["B"]["rel_ped"]["quota"])
                except KeyError:
                    r["pattern_consistent"] = False
                blk4[site] = r
            res["iv_rank_quota"][f"s{step}|{g}"] = {"statistic": "conceptor aperture with 80 % retained energy (sum mu^2 lam / sum lam) -> quota tr(C)/d; hard_rank = singular directions for 80 % energy; in-sample", "sites": blk4}
    return res


def ranked_tables(entries: list[dict[str, Any]], reg: dict[str, Any], args) -> dict[str, list[dict[str, Any]]]:
    k = f"k{args.k_registered}"
    tables: dict[str, list[dict[str, Any]]] = {}
    for e in entries:
        if e.get("status") != "ok":
            continue
        key = f"s{e['step']}|{e['group']}"
        ii = reg["ii_identity_attachment"].get(key, {}).get("sites", {}).get(e["site_id"], {})
        iii = reg["iii_transport_identity_remap"].get(key, {}).get("sites", {}).get(e["site_id"], {})
        sp = e["analysis"]["subspaces"]
        row = {"site_id": e["site_id"], "dd_t": ii.get("double_dissociation", {}).get("t"), "dd_mean": ii.get("double_dissociation", {}).get("mean"),
               "dd_p_refit_raw": ii.get("double_dissociation", {}).get("p_raw_one_sided"), "dd_p_maxt": ii.get("double_dissociation", {}).get("p_maxt_fwer"),
               "dd_p_scores_sign_flip": ii.get("double_dissociation", {}).get("sign_flip_p"),
               "ped_minus_cone_A": ii.get("ped_minus_cone_A", {}).get("mean"), "ped_minus_cone_B": ii.get("ped_minus_cone_B", {}).get("mean"),
               "iii_A_ped_to_B_cone_excess": iii.get("A_ped_to_B_cone_minus_random", {}).get("mean"), "iii_A_ped_to_B_ped_excess": iii.get("A_ped_to_B_ped_minus_random", {}).get("mean"),
               "iii_B_cone_to_A_ped_excess": iii.get("B_cone_to_A_ped_minus_random", {}).get("mean"), "iii_B_cone_to_A_cone_excess": iii.get("B_cone_to_A_cone_minus_random", {}).get("mean")}
        for t in TYPES:
            s = sp[t].get(k)
            if s is None:
                continue
            row[f"{t}_trans_AB_minus_within_B"] = s["trans_AB_minus_within_B"]["mean"]
            row[f"{t}_trans_AB_minus_rand_B"] = s["trans_AB_minus_rand_B"]["mean"]
            row[f"{t}_grassmann_transported"] = s["principal_angles_deg"]["grassmann_transported"]
            row[f"{t}_grassmann_raw"] = s["principal_angles_deg"]["grassmann_raw"]
            row[f"{t}_conceptor_xsim"] = e["analysis"]["conceptor"]["cross_similarity"].get(t, {}).get("transported")
        row["quota80"] = {arm: {t: v.get("quota") for t, v in e["analysis"]["quota80"].get(arm, {}).items()} for arm in ("A", "B")}
        tables.setdefault(key, []).append(row)
    for rows in tables.values():
        rows.sort(key=lambda r: -(r["dd_t"] if r["dd_t"] is not None and np.isfinite(r["dd_t"]) else -np.inf))
    return tables


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
    token_sets = token_sets_for(args.stimulus, cells, tuple(args.groups), n_steps, arm_a["offset"], args.domain)
    rows_a, rows_b = np.asarray([c["row_a"] for c in cells]), np.asarray([c["row_b"] for c in cells])
    if args.sites:
        sites = list(args.sites)
    else:
        sa = {p.stem for p in (args.dump_a / "activations").glob("*.npz")}
        sb = {p.stem for p in (args.dump_b / "activations").glob("*.npz")}
        sites = sorted(s for s in sa & sb if "adaln" not in s)
    entries: list[dict[str, Any]] = []
    transport_log: dict[str, Any] = {}
    export: dict[str, np.ndarray] = {}
    export_meta: dict[str, Any] = {"transports": [], "conceptors": []}
    for si, site in enumerate(sites):
        actsA, tixA = load_site(args.dump_a, site, rows_a)
        actsB, tixB = load_site(args.dump_b, site, rows_b)
        if actsA.shape[2] == 1 or actsB.shape[2] == 1:
            entries.append({"site_id": site, "status": "adaln_skipped"})
            continue
        for step in steps:
            lv = tuple(args.transport_levels)
            XA, XB, scene_rows_idx = matched_token_rows(token_rows(actsA, tixA, cells, step, lv), token_rows(actsB, tixB, cells, step, lv))
            if len(XA) < 8:
                for g in args.groups:
                    entries.append({"site_id": site, "step": step, "group": g, "status": "no_matched_tokens"})
                continue
            W_folds, W_full, tlog = transport_folds(XA, XB, scene_rows_idx, scenes)
            transport_log[f"{site}|s{step}"] = tlog
            if args.export_transport is not None:
                export[f"{site}__s{step}__W"] = W_full.astype(np.float32)
                export_meta["transports"].append({"site": site, "step": step, **tlog})
            for g in args.groups:
                entry: dict[str, Any] = {"site_id": site, "step": step, "group": g}
                PA, _ = pool_site(actsA, tixA, cells, token_sets, step, g)
                PB, _ = pool_site(actsB, tixB, cells, token_sets, step, g)
                ok = np.all(np.isfinite(PA), axis=1) & np.all(np.isfinite(PB), axis=1)
                if ok.sum() < len(cells):
                    bad = {c["pair_id"] for c in cells if not ok[c["row"]]}
                    keep = [s for s in scenes if s not in bad]
                    if len(keep) < 4:
                        entry["status"] = "no_tokens"
                        entries.append(entry)
                        continue
                    sub_cells = [c for c in cells if c["pair_id"] in set(keep)]
                    rA, rB = scene_rows(PA, sub_cells, keep), scene_rows(PB, sub_cells, keep)
                    entry["n_scenes_dropped_no_tokens"] = len(scenes) - len(keep)
                    entry["status"] = "partial_scenes"
                    entries.append(entry)
                    continue
                rA, rB = scene_rows(PA, cells, scenes), scene_rows(PB, cells, scenes)
                entry["analysis"] = analyze_group(rA, rB, scenes, W_folds, W_full, args, args.seed)
                entry["status"] = "ok"
                entries.append(entry)
                if args.export_transport is not None:
                    pre = f"{site}__s{step}__{g}"
                    export[f"{pre}__mean_A"] = PA.mean(0).astype(np.float32)
                    export[f"{pre}__mean_B"] = PB.mean(0).astype(np.float32)
                    for arm in ("A", "B"):
                        for t, c in entry["analysis"]["_cons"][arm].items():
                            export[f"{pre}__{arm}__{t}__eigvecs"] = np.asarray(c["eigvecs"], dtype=np.float32)
                            export[f"{pre}__{arm}__{t}__mu"] = np.asarray(c["mu"], dtype=np.float32)
                            u = entry["analysis"]["_rank_one"][arm].get(t)
                            if u is not None:
                                export[f"{pre}__{arm}__{t}__rank_one"] = np.asarray(u, dtype=np.float32)
                    export_meta["conceptors"].append({"site": site, "step": step, "group": g, "types": sorted(entry["analysis"]["_cons"]["A"]), "d": int(PA.shape[1]),
                                                      "quota": {arm: {t: float(np.sum(c["mu"]) / PA.shape[1]) for t, c in entry["analysis"]["_cons"][arm].items()} for arm in ("A", "B")}})
        print(f"[{si + 1}/{len(sites)}] {site} ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    reg = registered_tests(entries, scenes, args)
    tables = ranked_tables(entries, reg, args)
    for e in entries:
        an = e.get("analysis")
        if an:
            an.pop("_cons", None)
            an.pop("_rank_one", None)
            for t in an["subspaces"].values():
                for kk in t.values():
                    kk.pop("_per_scene", None)
            for kk in an["cross_identity"].values():
                kk.pop("_per_scene", None)
            an["dd_refit"] = {"k": an["dd_refit"]["k"], "t": an["dd_refit"]["t"], "p_raw_one_sided": an["dd_refit"]["p_raw_one_sided"], "n_perm": an["dd_refit"]["n_perm"]}
    report = {
        "protocol": PROTOCOL, "dump_a": str(args.dump_a), "dump_b": str(args.dump_b), "stimulus": str(args.stimulus),
        "arms": {"A": args.arm_a_name, "B": args.arm_b_name}, "domain": args.domain, "token_group_source": alias_note(args.domain),
        "n_scenes": len(scenes), "scenes": scenes, "n_cells": len(cells), "alignment": status, "steps": steps, "groups": list(args.groups), "ks": list(args.ks),
        "k_action": args.k_action, "target_dims": args.target_dims, "sites": sites,
        "contrasts": {"app_ped": "mean_a (h[1,a] - h[0,a]) (action-averaged main effect; contains DiD(1)/2)", "app_cone": "mean_a (h[3,a] - h[0,a])",
                      "app_ped_a0": "h[1,0] - h[0,0] (brake-cell appearance = main effect - DiD/2; registered for test i)", "app_cone_a0": "h[3,0] - h[0,0]",
                      "action": "mean over hazard-free levels (0, 2) of (h[l,1] - h[l,0])",
                      "rel_ped": "DiD(1) - DiD(2), projected off the train-fold rank-k_action action main-effect subspace",
                      "rel_cone": "DiD(3) - DiD(2), same", "DiD(l)": "(h[l,1] - h[l,0]) - (h[0,1] - h[0,0])", "pooling": "mean over the scene-level union of the group's tokens"},
        "transport": {"method": "orthogonal Procrustes W = argmin ||X_A W - X_B||_F on matched per-token rows (same cell, step, token id) of the hazard-free cells (levels " + ",".join(str(v) for v in args.transport_levels) + "), centred on the train fold; LOSO: held-out scene removed from the cross-covariance and the means",
                      "why_hazard_free": "rows carrying the in-lane identity would let the d x d map absorb the contrast under test (synthetic null check)",
                      "direction_map": "column direction q in A -> W^T q in B; conceptor eigvecs V -> V W", "per_site_step": transport_log},
        "ranked_tables": tables, "registered_tests": reg, "entries": entries, "runtime_s": time.time() - t0, "interpretation_scope": INTERPRETATION_SCOPE,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    if args.export_transport is not None:
        report["transport_export"] = str(write_transport_export(args.export_transport, export, export_meta, report, args))
    (args.out / "cross_arm_map.json").write_text(json.dumps(finite(report), indent=1) + "\n")
    for key, rows in tables.items():
        print(f"== {key} (sorted by double-dissociation t; k={args.k_registered}) ==")
        for r in rows[:8]:
            fmt = lambda v: "nan" if v is None else f"{v:+.3f}"  # noqa: E731
            print(f"  {r['site_id']:>14s} DD t={fmt(r['dd_t'])} p_maxT={fmt(r['dd_p_maxt'])} (A ped-cone {fmt(r['ped_minus_cone_A'])}, B {fmt(r['ped_minus_cone_B'])}) | "
                  f"app_ped trans-within {fmt(r.get('app_ped_trans_AB_minus_within_B'))} trans-rand {fmt(r.get('app_ped_trans_AB_minus_rand_B'))} | "
                  f"iii A.ped->B.cone {fmt(r['iii_A_ped_to_B_cone_excess'])} ->B.ped {fmt(r['iii_A_ped_to_B_ped_excess'])}")
    return report


def write_transport_export(path: Path, arrays: dict[str, np.ndarray], meta: dict[str, Any], report: dict[str, Any], args) -> Path:
    """Transport export for ``steered_planner_ranking.py --transported-conceptor`` (donor-free steering of arm B with
    arm A's relational conceptor, and the mirror).  One ``.npz`` with, per site and imagined step,

      ``<site>__s<step>__W``                               float32 [d, d]  full-data orthogonal Procrustes map fitted on the
                                                           hazard-free per-token rows: x_B ~= (x_A - mean_A) W + mean_B;
                                                           row-vector directions map as v_B = v_A W (and v_A = v_B W^T)
      ``<site>__s<step>__<group>__mean_{A,B}``             float32 [d]     mean group-pooled activation of the discovery cells
      ``<site>__s<step>__<group>__<arm>__<type>__eigvecs`` float32 [r, d]  factored conceptor (C = V^T diag(mu) V) of that arm's
      ``<site>__s<step>__<group>__<arm>__<type>__mu``      float32 [r]     contrast rows; relational types are the C_did AND NOT
                                                           C_action (AND NOT C_null) safety conceptors, appearance types plain
      ``<site>__s<step>__<group>__<arm>__<type>__rank_one``float32 [d]     unit mean contrast row (rank-one control direction)

    plus ``meta`` (JSON: scenes, groups, types, transport diagnostics, quotas).  Everything is fitted on the discovery scenes
    given to this run; the steering script applies ``eigvecs @ W`` to move arm A's conceptor into arm B's coordinates."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    full_meta = {
        "protocol": PROTOCOL, "written_by": "geometry_cross_arm.py --export-transport", "dump_a": report["dump_a"], "dump_b": report["dump_b"],
        "arms": report["arms"], "scenes": report["scenes"], "n_scenes": report["n_scenes"], "steps": report["steps"], "groups": report["groups"],
        "types": list(TYPES), "relational_types": list(RELATIONAL_TYPES), "target_dims": args.target_dims, "k_action": args.k_action,
        "transport_levels": list(args.transport_levels), "direction_map": "v_B = v_A @ W ; v_A = v_B @ W.T ; x_B ~= (x_A - mean_A) W + mean_B",
        "conceptor_form": {"rel_*": "C_did AND NOT C_action (AND NOT C_null); rows scaled by their RMS before the aperture", "app_*": "C_appearance"},
        "fitted_on": "discovery scenes only (full-data fit; LOSO statistics live in cross_arm_map.json)",
        **meta,
    }
    np.savez_compressed(path, meta=json.dumps(finite(full_meta)), **arrays)
    return path


# --------------------------------------------------------------------------- #
# Synthetic self-test
# --------------------------------------------------------------------------- #


def _orthonormal(rng: np.random.Generator, d: int, k: int) -> np.ndarray:
    return np.linalg.qr(rng.normal(size=(d, k)))[0].T  # [k, d]


def synthetic_dumps(root: Path, *, null: bool, n_scenes: int = 16, d: int = 64, n_steps: int = 2, sites=("L03.attn_out", "L07.attn_out"), seed: int = 0, noise: float = 0.3) -> tuple[Path, Path, Path]:
    """Two random driving dumps on shared scenes.  Arm B's hidden coordinates = arm A's rotated by a random orthogonal
    W_true (raw ambient overlap at chance).  Planted: shared appearance subspaces (levels 1, 2, 3), a rank-2 relational
    subspace attached to level 1 in A and level 3 in B (same S_rel), an AdaLN-style hazard-scaled action artifact in all
    hazard levels (so DiD(H0') carries it too).  ``null``: independent appearance subspaces per arm, no relational term."""
    rng = np.random.default_rng(seed)
    stim, dA, dB = root / "stimulus", root / "arm_A", root / "arm_B"
    (stim / "masks").mkdir(parents=True, exist_ok=True)
    for p in (dA, dB):
        (p / "activations").mkdir(parents=True, exist_ok=True)
    Bs = _orthonormal(rng, d, 12)
    S_ped, S_cone, S_null, S_rel, a_dir = Bs[:3], Bs[3:6], Bs[6:9], Bs[9:11], Bs[11]
    S_ped_B, S_cone_B = (_orthonormal(rng, d, 3), _orthonormal(rng, d, 3)) if null else (S_ped, S_cone)
    W_true = np.linalg.qr(rng.normal(size=(d, d)))[0]
    E = rng.normal(size=(256, d)) * 2.0
    gamma = 0.5
    cells, rows = [], []
    acts = {"A": [], "B": []}
    tix = []
    yy, xx = np.mgrid[:256, :256]
    for s in range(n_scenes):
        seed_id = 500 + s
        pid = f"drive_seed{seed_id:06d}"
        content = rng.normal(size=(256, d)) * 0.5
        c_ped, c_cone, c_null = rng.normal(size=3) * [3.0, 2.0, 1.5], rng.normal(size=3) * [3.0, 2.0, 1.5], rng.normal(size=3) * [3.0, 2.0, 1.5]
        rho = rng.normal(size=2) * [3.0, 2.0]
        g = 2.0 + 0.3 * rng.normal()
        col = 7 + int(rng.integers(0, 2))
        lane = (xx >= col * 16) & (xx < (col + 2) * 16) & (yy >= 128) & (yy < 208)
        side = (xx >= 208) & (xx < 240) & (yy >= 128) & (yy < 208)
        mirror = (xx >= 16) & (xx < 48) & (yy >= 128) & (yy < 208)
        corridor = (yy >= 144) & (np.abs(xx - 128) < (yy - 120) * 0.35)
        hz_masks = {0: side, PED: lane, NULL: mirror, OBJ: lane}
        keys = [(lv, a) for lv in (0, PED, NULL, OBJ) for a in (0, 1)]
        masks = {}
        for lv, a in keys:
            cid = f"{pid}__h{lv}a{a}"
            np.savez(stim / "masks" / f"{cid}.npz", hazard_mask=np.repeat(hz_masks[lv][None], 4, axis=0), corridor_mask=np.repeat(corridor[None], 4, axis=0))
            masks[(lv, a)] = load_cell_groups(stim, {"cell_id": cid}, domain="driving")
        tok_union = union(*(masks[k].group(0, "hazard_corridor") for k in keys))
        hazard_tok = union(*(masks[k].group(0, "hazard") for k in keys))
        in_h = np.isin(tok_union, hazard_tok)
        for lv, a in keys:
            cid = f"{pid}__h{lv}a{a}"
            row = len(cells)
            cells.append({"row": row, "pair_id": pid, "seed": seed_id, "hazard": lv, "candidate_action": a, "cell_id": cid, "artifact": f"cells/{cid}.npz"})
            rows.append({"cell_id": cid, "pair_id": pid, "seed": seed_id, "hazard": lv, "candidate_action": a, "artifact": f"cells/{cid}.npz"})
            per_arm = {}
            for arm in ("A", "B"):
                Sp, Sc = (S_ped, S_cone) if arm == "A" else (S_ped_B, S_cone_B)
                x = E[tok_union] + content[tok_union]
                app = np.zeros(d)
                if lv == PED:
                    app = c_ped @ Sp
                elif lv == OBJ:
                    app = c_cone @ Sc
                elif lv == NULL:
                    app = c_null @ S_null
                x = x + in_h[:, None] * app[None, :]
                x = x + a * g * (1.0 + gamma * (lv > 0)) * a_dir[None, :]
                solid = PED if arm == "A" else OBJ
                if not null and lv == solid and a == 1:
                    x = x + (rho @ S_rel)[None, :]
                per_arm[arm] = x
            steps_A, steps_B = [], []
            for t in range(n_steps):
                nA = rng.normal(size=per_arm["A"].shape) * noise
                nB = rng.normal(size=per_arm["B"].shape) * noise
                steps_A.append(per_arm["A"] * (1 + 0.1 * t) + nA)
                steps_B.append((per_arm["B"] * (1 + 0.1 * t) + nB) @ W_true)
            acts["A"].append(np.stack(steps_A))
            acts["B"].append(np.stack(steps_B))
            tix.append(tok_union)
    n, T = len(cells), max(len(t) for t in tix)
    token_index = np.full((n, n_steps, T), -1, dtype=np.int64)
    arr = {"A": np.zeros((n, n_steps, T, d), np.float32), "B": np.zeros((n, n_steps, T, d), np.float32)}
    for r in range(n):
        token_index[r, :, : len(tix[r])] = tix[r][None, :]
        for arm in ("A", "B"):
            arr[arm][r, :, : len(tix[r])] = acts[arm][r]
    for arm, dp in (("A", dA), ("B", dB)):
        for i, site in enumerate(sites):
            jitter = np.random.default_rng(seed + 100 + i).normal(size=arr[arm].shape).astype(np.float32) * 0.05
            np.savez_compressed(dp / "activations" / f"{site}.npz", acts=(arr[arm] + jitter).astype(np.float16), token_index=token_index)
        (dp / "activations" / "index.json").write_text(json.dumps({"cells": cells, "n_steps": n_steps, "group_frame_offset": 0, "domain": "driving", "dump_groups": ["hazard", "corridor"],
                                                                     "dump_hazard_levels": [0, 1, 2, 3], "synthetic": {"null": null, "d": d, "noise": noise, "rotation": "random orthogonal W_true on arm B (x_B = x_A W_true)", "W_true": W_true.tolist(),
                                                                                                   "S_rel": S_rel.tolist(), "S_ped_A": S_ped.tolist(), "S_ped_B": S_ped_B.tolist()}}, indent=1))
    (stim / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return stim, dA, dB


def self_test_verdict(report: dict[str, Any], planted: bool) -> dict[str, Any]:
    """Planted: every site must show the registered pattern in all four tests.  Null: (i) must fail at every site, (ii)
    |DD mean| below NULL_DD_BAND at every site, (iii) cross-identity excess below NULL_EXCESS_BAND at every site,
    (iv) the quota order must not hold at every site (a two-inequality coin flip under the null, ~25 % per site).
    Counts of pattern flags and of the one-sided significance events are reported alongside."""
    reg = report["registered_tests"]
    steps0 = [k for k in reg["ii_identity_attachment"]]
    i_flags = [r["app_ped_a0"]["pattern_consistent"] and r["app_cone_a0"]["pattern_consistent"] for blk in reg["i_appearance_shared"].values() for r in blk.values()]
    ii_flags = [r["pattern_consistent"] for blk in reg["ii_identity_attachment"].values() for r in blk["sites"].values()]
    ii_sig = [r["double_dissociation"]["p_maxt_fwer"] is not None and r["double_dissociation"]["p_maxt_fwer"] < 0.05 for blk in reg["ii_identity_attachment"].values() for r in blk["sites"].values()]
    iii_flags = [r["pattern_consistent"] for blk in reg["iii_transport_identity_remap"].values() for r in blk["sites"].values()]
    iii_cone = [ci_excludes_zero(r["A_ped_to_B_cone_minus_random"]) or ci_excludes_zero(r["B_cone_to_A_ped_minus_random"]) for blk in reg["iii_transport_identity_remap"].values() for r in blk["sites"].values()]
    iv_flags = [r["pattern_consistent"] for blk in reg["iv_rank_quota"].values() for r in blk["sites"].values()]
    dd_means = [abs(r["double_dissociation"]["mean"]) for blk in reg["ii_identity_attachment"].values() for r in blk["sites"].values()]
    iii_means = [max(abs(r["A_ped_to_B_cone_minus_random"]["mean"]), abs(r["B_cone_to_A_ped_minus_random"]["mean"])) for blk in reg["iii_transport_identity_remap"].values() for r in blk["sites"].values()]
    if planted:
        checks = {"i": bool(i_flags) and all(i_flags), "ii": bool(ii_flags) and all(ii_flags), "iii": bool(iii_flags) and all(iii_flags), "iv": bool(iv_flags) and all(iv_flags)}
    else:
        # magnitude bands for the null: the pipeline must not manufacture an effect of the planted size (planted DD ~ 1.5,
        # planted cone excess ~ 0.6); p-value calibration is a separate property (see the module notes) and a single
        # family at p < 0.05 is expected in ~1 of 4 null runs with six families
        checks = {"i": bool(i_flags) and not any(i_flags), "ii": bool(dd_means) and max(dd_means) < NULL_DD_BAND, "iii": bool(iii_means) and max(iii_means) < NULL_EXCESS_BAND, "iv": bool(iv_flags) and not all(iv_flags)}
    return {"planted": planted, "checks": checks, "passed": all(checks.values()), "n_families": len(steps0),
            "max_abs_dd_mean": max(dd_means) if dd_means else None, "max_cross_identity_excess": max(iii_means) if iii_means else None,
            "counts": {"i": [sum(i_flags), len(i_flags)], "ii": [sum(ii_flags), len(ii_flags)], "ii_maxt_sig": [sum(ii_sig), len(ii_sig)], "iii": [sum(iii_flags), len(iii_flags)], "iii_cone_excess": [sum(iii_cone), len(iii_cone)], "iv": [sum(iv_flags), len(iv_flags)]}}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dump-a", type=Path, default=None, help="localization dump of arm A (pedestrian solid / cone ghost)")
    ap.add_argument("--dump-b", type=Path, default=None, help="localization dump of arm B (pedestrian ghost / cone solid)")
    ap.add_argument("--stimulus", type=Path, default=None, help="shared stimulus dir (masks/<cell_id>.npz)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--discovery-seeds", type=Path, default=None)
    ap.add_argument("--domain", default="driving")
    ap.add_argument("--sites", nargs="*", default=None)
    ap.add_argument("--steps", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--groups", nargs="*", default=list(DEFAULT_GROUPS))
    ap.add_argument("--appearance-group", default="hazard", help="group at which test (i) is registered")
    ap.add_argument("--ks", type=int, nargs="*", default=list(KS))
    ap.add_argument("--k-registered", type=int, default=4)
    ap.add_argument("--k-action", type=int, default=4, help="rank of the action main-effect subspace (from hazard-free cells) projected off the relational rows")
    ap.add_argument("--transport-levels", type=int, nargs="*", default=list(ACTION_LEVELS), help="hazard levels whose per-token rows fit the Procrustes transport (hazard-free by default)")
    ap.add_argument("--target-dims", type=float, default=4.0, help="conceptor aperture rule: quota * d")
    ap.add_argument("--equiv-margin", type=float, default=0.1, help="equivalence margin for transported - within (test i)")
    ap.add_argument("--export-transport", type=Path, default=None, help="write W per site/step + both arms' factored conceptors/means to this .npz (steered_planner_ranking.py --transported-conceptor)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=2000, help="sign flips for the descriptive score-level tests")
    ap.add_argument("--n-perm-refit", type=int, default=999, help="refit permutations for the registered double-dissociation test (ii)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm-a-name", default="A: PED-solid / OBJ-ghost")
    ap.add_argument("--arm-b-name", default="B: PED-ghost / OBJ-solid")
    ap.add_argument("--synthetic", action="store_true", help="write planted random dumps under <out>/synthetic and verify the four predictions")
    ap.add_argument("--synthetic-null", action="store_true", help="with --synthetic: plant independent appearance subspaces and no relational subspace; predictions must NOT hold")
    ap.add_argument("--synthetic-scenes", type=int, default=16)
    ap.add_argument("--synthetic-dim", type=int, default=64)
    ap.add_argument("--synthetic-noise", type=float, default=0.3)
    ap.add_argument("--synthetic-sites", nargs="*", default=["L03.attn_out", "L07.attn_out"])
    return ap


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.synthetic:
        root = args.out / ("synthetic_null" if args.synthetic_null else "synthetic")
        stim, dA, dB = synthetic_dumps(root, null=args.synthetic_null, n_scenes=args.synthetic_scenes, d=args.synthetic_dim, sites=tuple(args.synthetic_sites), seed=args.seed, noise=args.synthetic_noise)
        args.dump_a, args.dump_b, args.stimulus = dA, dB, stim
        args.out = root / "out"
        args.groups = [g for g in args.groups if g in ("hazard", "corridor", "hazard_corridor")]
        report = run(args)
        verdict = self_test_verdict(report, planted=not args.synthetic_null)
        report["self_test"] = verdict
        (args.out / "cross_arm_map.json").write_text(json.dumps(finite(report), indent=1) + "\n")
        print(json.dumps(verdict))
        if not verdict["passed"]:
            raise SystemExit(f"synthetic self-test FAILED: {verdict}")
        return report
    if not (args.dump_a and args.dump_b and args.stimulus):
        raise SystemExit("--dump-a, --dump-b and --stimulus are required (or --synthetic)")
    return run(args)


if __name__ == "__main__":
    main()
