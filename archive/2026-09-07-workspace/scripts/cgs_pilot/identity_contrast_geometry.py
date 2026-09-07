#!/usr/bin/env python3
"""Identity-contrast geometry under the shared consequence template (METHOD fix for the seed-0 cross-arm null, F17).

``geometry_cross_arm.py`` (registered tests i-iv) came out null on the seed-0 pair because the hazard x action DiD field
at every site is dominated by a rank-1 "consequence template" shared by both arms AND both identities (PC1 ~ 0.6, hard
rank 1), so every LOSO contrast subspace captures the template and nothing else.  The identity attachment is nonetheless
real behaviourally (cross-truth 12/12) and causally (identity-swapped donor 0.02 vs real donor 0.72 at L03.mlp_out): it
must be a SMALL component hidden under the template.  This module looks for it directly, per site x token group at
imagined step 0, on the same two localization dumps and with the same loaders, alignment, pooling and hazard-free
Procrustes transport as ``geometry_cross_arm.py`` (imported, not copied):

1. Per-scene DiD fields for the pedestrian identity (level 1 vs 0, x action) and the cone identity (level 3 vs 0) per arm,
   D^X_ped, D^X_cone (X in {A, B}), DiD(l) = (h[l,1] - h[l,0]) - (h[0,1] - h[0,0]) on the group-pooled activations.
2. Template removal, leave-one-scene-out: the SHARED template tau_{-i} = mean over the other scenes of all four
   (arm x identity) fields, arm B's fields first moved into arm A's coordinates with the fold's transport (v_A = v_B W^T);
   residuals r^X_x,i = D^X_x,i - tau_{-i}.  A PER-ARM variant (tau^A_{-i} = mean of A's two fields, tau^B_{-i} of B's) is
   reported alongside.
3. Identity-contrast vector per scene per arm, I^A_i = r^A_ped,i - r^A_cone,i and I^B_i = r^B_ped,i - r^B_cone,i.  Because
   the same template vector is subtracted from both identities of an arm, I is ALGEBRAICALLY template-invariant
   (I = D_ped - D_cone under either variant); the template matters for its magnitude (c), for the residual rank (d) and for
   the template-orthogonal variant I_perp = I - (I . unit(tau_{-i})) unit(tau_{-i}) which is also tested (the part of the
   identity contrast that is not "more of the same template" -- the template-parallel part is reported as the
   template-amplitude asymmetry coef_ped - coef_cone per arm, expected > 0 in A and < 0 in B).
   Tests, unit = scene, on every variant (I, I_perp shared template, I_perp per-arm template):
   (a) within-arm LOSO direction consistency: s_i = I_i . unit(mean_{j != i} I_j) (``cgs_stats.loso_projection_scores``),
       t, sign-flip p with the LOSO mean refitted under every flip, max-T over the sites of a (group, arm) family.
   (b) cross-arm anti-alignment predicted by the reversed assignment (consequence attached to the pedestrian in A, to the
       cone in B): after the hazard-free Procrustes transport, I^A_i and T_i(I^B_i) should be ANTI-aligned.
       (b1) per-scene cosine c_i = cos(I^A_i, I^B_i W_i^T): mean, scene-bootstrap CI, two-sided sign-flip p and one-sided
            (negative) p; scene-permutation null (I^A_i paired with I^B_pi(i), transported with fold i's map) which asks
            whether the alignment is scene-specific beyond a shared direction (the null's mean IS the shared-direction
            alignment).  The identity-relabel null leaves c_i invariant (both vectors flip), so it cannot be used here.
       (b2) registered cross-arm statistic: held-out projection onto the OTHER arm's LOSO identity direction,
            q_i = 0.5 [cos(I^A_i, T_i mean_{j != i} I^B_j) + cos(T_i I^B_i, mean_{j != i} I^A_j)], against the
            identity-relabel REFIT null (ped <-> cone relabelling of a scene applied in BOTH arms, every LOSO mean
            refitted; the per-scene statistic flips sign and the LOSO means change) -- one-sided (q < 0), max-T over the
            sites of a (group) family with shared flips (Westfall-Young as in ``cgs_stats``).
   (c) magnitude: ||I_i|| / ||tau_{-i}|| (median over scenes) per arm and variant, held-out and in-sample fraction of the
       DiD-field energy explained by the template (mean-direction and PC1), the residual relative norms ||r|| / ||D||, the
       template-amplitude asymmetry, and cos(DiD(H0'), tau) where the level-2 mirror-pose cells exist.
   (d) rank / quota (``geometry_cross_arm.quota_for_capture``, 80 % energy, in-sample) of the raw fields, of the
       residual fields under both templates and of I per arm, with the uncentred PC1 fraction.
4. ``<out>/identity_geometry.json`` (+ ``identity_geometry.md``): per site x group blocks, per-(group) families with the
   max-T results, ranked tables (by the anti-alignment t of variant I), ``interpretation_scope``.

``--self-test`` writes two small synthetic dump pairs in the exact layout the loaders expect (arm B's coordinates rotated
by a random orthogonal map, so the transport is required): PLANTED = shared template (in-lane x throttle, both identities,
both arms, amplitude 3) + a small identity component (amplitude 0.6, one fixed direction orthogonal to the template,
attached to the SOLID identity of each arm, so I^A ~ +u and I^B ~ -u) + a zero-mean shared appearance x action term;
NULL = the same without the identity component.  The planted case must be detected at every site (variants I and
I_perp), the null case at none; the planted template must dominate (median ||I|| / ||tau|| < 0.5) so that the test runs
in the regime the real data are in.

Descriptive / associational only; the causal claims stay with the patching stage (``patch_site.py``, ``conceptor_patch.py``).
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
from cgs_stats import loso_projection_scores, sign_flip_maxt  # noqa: E402
from geometry_cross_arm import (  # noqa: E402
    ACTION_LEVELS,
    NULL,
    OBJ,
    PED,
    _orthonormal,
    align_arms,
    ci_excludes_zero,
    load_arm,
    load_site,
    matched_token_rows,
    quota_for_capture,
    scene_rows,
    summ,
    t_stat,
    token_rows,
    token_sets_for,
    transport_folds,
)
from geometry_localize import pool_site  # noqa: E402
from stats_utils import cosine, finite  # noqa: E402
from token_groups import alias_note, load_cell_groups, set_domain, union  # noqa: E402

PROTOCOL = "cgs-identity-contrast-geometry-v0.1"
DEFAULT_GROUPS = ("corridor", "hazard", "hazard_corridor")
VARIANTS = ("I", "I_perp_shared", "I_perp_perarm")
ARMS = ("A", "B")
FIELDS = ("A_ped", "A_cone", "B_ped", "B_cone")
NULL_COS_BAND = 0.3  # synthetic-null magnitude band on |mean cosine| and |mean within-arm cosine| (planted effects ~0.5-0.9)
INTERPRETATION_SCOPE = (
    "descriptive/associational; template-projected; causal claims remain with patching. Everything is fitted on discovery "
    "scenes only (LOSO for every held-out score, the hazard-free Procrustes transport included); the identity-contrast vector "
    "I = DiD(ped) - DiD(cone) is algebraically invariant to subtracting one template vector from both identities, so the "
    "template removal changes its magnitude bookkeeping and the I_perp variant, not I itself; anti-alignment of I^A with the "
    "transported I^B is a geometric consequence of the reversed assignment and does not by itself show that either arm USES "
    "that direction."
)


# --------------------------------------------------------------------------- #
# Small numerics
# --------------------------------------------------------------------------- #


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def row_norms(X: np.ndarray) -> np.ndarray:
    return np.linalg.norm(X, axis=1)


def project_off_dir(X: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Rows of X with their component along the unit vector u removed."""
    return X - np.outer(X @ u, u)


def sign_flip_p_onesided(values: np.ndarray, negative: bool = True, n_perm: int = 5000, seed: int = 0) -> float:
    """One-sided sign-flip p that the per-scene mean is < 0 (``negative``) or > 0; exact for n <= 12 (as ``stats_utils.sign_flip_p``)."""
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return float("nan")
    obs = x.mean()
    if n <= 12:
        codes = np.arange(2**n)[:, None] >> np.arange(n)[None, :]
        signs = np.where(codes & 1, 1.0, -1.0)
        stats = (signs * x[None, :]).mean(axis=1)
        return float(np.mean(stats <= obs + 1e-15) if negative else np.mean(stats >= obs - 1e-15))
    rng = np.random.RandomState(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    stats = (signs * x[None, :]).mean(axis=1)
    hits = np.sum(stats <= obs + 1e-15) if negative else np.sum(stats >= obs - 1e-15)
    return float((hits + 1) / (n_perm + 1))


def rank_summary(X: np.ndarray, d: int) -> dict[str, Any]:
    """Uncentred spectrum of the rows of X [n, d]: PC1 energy fraction and the 80 %-energy quota / hard rank."""
    X = np.asarray(X, dtype=np.float64)
    X = X[np.all(np.isfinite(X), axis=1)]
    if len(X) < 2:
        return {"n": int(len(X)), "pc1_fraction": float("nan"), "hard_rank": 0, "quota": float("nan"), "effective_dims": float("nan")}
    lam = np.clip(np.linalg.eigvalsh(X @ X.T), 0.0, None)
    q = quota_for_capture(lam, d, 0.8)
    return {"n": int(len(X)), "pc1_fraction": float(lam.max() / lam.sum()) if lam.sum() > 0 else float("nan"), "hard_rank": int(q["hard_rank"]),
            "quota": q["quota"], "effective_dims": q.get("effective_dims", float("nan")), "alpha": q["alpha"]}


def energy_fraction_along(X: np.ndarray, u: np.ndarray) -> float:
    """sum_i (x_i . u)^2 / sum_i ||x_i||^2 for a unit vector u."""
    den = float(np.sum(X**2))
    return float(np.sum((X @ u) ** 2) / den) if den > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Cross-arm held-out projection statistic from packed Gram blocks (for the max-T machinery)
# --------------------------------------------------------------------------- #


def pack_grams(IA: np.ndarray, IB: np.ndarray, W_list: list[np.ndarray]) -> np.ndarray:
    """[S, 4S] = [G_AB | G_BB | G_BA | G_AA] with G_AB[i, j] = (I^A_i W_i) . I^B_j (fold i's transport moves A into B
    coordinates; norms are transport-invariant), G_BA[i, j] = (I^B_i W_i^T) . I^A_j = I^B_i . (I^A_j W_i),
    G_AA = I^A I^A^T, G_BB = I^B I^B^T.  Everything the held-out projection statistic needs under any relabelling."""
    S = IA.shape[0]
    GAB, GBA = np.zeros((S, S)), np.zeros((S, S))
    for i in range(S):
        AW = IA @ W_list[i]  # [S, d] in B coordinates
        GAB[i] = AW[i] @ IB.T
        GBA[i] = IB[i] @ AW.T
    return np.concatenate([GAB, IB @ IB.T, GBA, IA @ IA.T], axis=1)


def heldout_cross_projection(packed: np.ndarray, signs: np.ndarray | None = None) -> np.ndarray:
    """Per-scene q_i = 0.5 [cos(f_i I^A_i, T_i sum_{j != i} f_j I^B_j) + cos(f_i T_i I^B_i, sum_{j != i} f_j I^A_j)] under the
    relabelling signs f (None = observed).  Relabelling ped <-> cone in scene j in both arms maps I^X_j -> -I^X_j."""
    S = packed.shape[0]
    GAB, GBB, GBA, GAA = packed[:, :S], packed[:, S : 2 * S], packed[:, 2 * S : 3 * S], packed[:, 3 * S :]
    f = np.ones(S) if signs is None else np.asarray(signs, dtype=np.float64)
    dAA, dBB = np.diag(GAA), np.diag(GBB)

    def loso_cos(G_cross, G_other, d_self, d_other):
        num = f * (G_cross @ f) - np.diag(G_cross)  # f_i I_i . sum_{j != i} f_j J_j
        tot = float(f @ G_other @ f)
        n2 = tot - 2.0 * f * (G_other @ f) + d_other  # ||sum_{j != i} f_j J_j||^2
        den = np.sqrt(np.clip(d_self, 0, None)) * np.sqrt(np.clip(n2, 0, None))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 1e-12, num / den, np.nan)

    return 0.5 * (loso_cos(GAB, GBB, dAA, dBB) + loso_cos(GBA, GAA, dBB, dAA))


def neg_heldout_cross_projection(packed: np.ndarray, signs: np.ndarray | None = None) -> np.ndarray:
    """Negated so that anti-alignment is the ``signal > 0`` direction ``sign_flip_maxt`` tests."""
    return -heldout_cross_projection(packed, signs)


# --------------------------------------------------------------------------- #
# Per (site, group) analysis
# --------------------------------------------------------------------------- #


def identity_contrast_analysis(DA: dict[str, np.ndarray], DB: dict[str, np.ndarray], scenes: list[str], W_folds: dict[str, np.ndarray], W_full: np.ndarray,
                               n_boot: int, n_perm: int, seed: int, DnullA: np.ndarray | None = None, DnullB: np.ndarray | None = None) -> dict[str, Any]:
    """DA / DB: {"ped": [S, d], "cone": [S, d]} per-scene DiD fields of arm A (A coordinates) and arm B (B coordinates).
    Returns the per-site block (JSON-ready statistics) plus ``_family`` arrays consumed by the family-level max-T tests."""
    S, d = DA["ped"].shape
    W_list = [W_folds[s] for s in scenes]
    IA = DA["ped"] - DA["cone"]  # A coords; identical under both template variants (see module notes)
    IB = DB["ped"] - DB["cone"]  # B coords
    # ---- LOSO templates and residuals
    sumA = DA["ped"].sum(0) + DA["cone"].sum(0)
    sumB = DB["ped"].sum(0) + DB["cone"].sum(0)
    tau_shared, tau_A, tau_B = np.zeros((S, d)), np.zeros((S, d)), np.zeros((S, d))
    IB_A = np.zeros((S, d))  # transported I^B (A coords), fold-wise
    for i in range(S):
        W = W_list[i]
        sA = sumA - DA["ped"][i] - DA["cone"][i]
        sB = sumB - DB["ped"][i] - DB["cone"][i]
        tau_A[i] = sA / (2 * (S - 1))
        tau_B[i] = sB / (2 * (S - 1))
        tau_shared[i] = (sA + sB @ W.T) / (4 * (S - 1))
        IB_A[i] = IB[i] @ W.T
    uS, uA, uB = np.array([unit(v) for v in tau_shared]), np.array([unit(v) for v in tau_A]), np.array([unit(v) for v in tau_B])
    # held-out residuals (norm bookkeeping; coordinates do not matter for norms)
    res = {}
    for x in ("ped", "cone"):
        res[("shared", "A", x)] = DA[x] - tau_shared
        res[("shared", "B", x)] = np.array([DB[x][i] @ W_list[i].T for i in range(S)]) - tau_shared
        res[("perarm", "A", x)] = DA[x] - tau_A
        res[("perarm", "B", x)] = DB[x] - tau_B
    # ---- variants of the identity contrast
    var: dict[str, dict[str, np.ndarray]] = {
        "I": {"A": IA, "B": IB, "B_in_A": IB_A},
        "I_perp_shared": {"A": np.array([project_off_dir(IA[i : i + 1], uS[i])[0] for i in range(S)]),
                          "B_in_A": np.array([project_off_dir(IB_A[i : i + 1], uS[i])[0] for i in range(S)])},
        "I_perp_perarm": {"A": np.array([project_off_dir(IA[i : i + 1], uA[i])[0] for i in range(S)]),
                          "B": np.array([project_off_dir(IB[i : i + 1], uB[i])[0] for i in range(S)])},
    }
    var["I_perp_shared"]["B"] = np.array([var["I_perp_shared"]["B_in_A"][i] @ W_list[i] for i in range(S)])  # back in B coords for the within-arm test
    var["I_perp_perarm"]["B_in_A"] = np.array([var["I_perp_perarm"]["B"][i] @ W_list[i].T for i in range(S)])
    rng = np.random.RandomState(seed)
    out: dict[str, Any] = {"n_scenes": S, "d": d, "variants": {}, "template": {}, "rank": {}}
    fam: dict[str, Any] = {}
    for name, v in var.items():
        A, B, B_A = v["A"], v["B"], v["B_in_A"]
        # (a) within-arm LOSO direction consistency (per-site descriptive; the sign-flip / max-T live at the family level)
        sA, sB = loso_projection_scores(A), loso_projection_scores(B)
        cosA, cosB = sA / np.where(row_norms(A) > 0, row_norms(A), np.nan), sB / np.where(row_norms(B) > 0, row_norms(B), np.nan)
        # (b1) per-scene cosine and the scene-permutation null (fold i's transport, scene pi(i) of arm B)
        nA, nB = row_norms(A), row_norms(B_A)
        C = cosine_matrix_fold(A, B, W_list)
        c_obs = np.diag(C)
        perm_means = np.array([np.mean(C[np.arange(S), rng.permutation(S)]) for _ in range(n_perm)])
        off = C[~np.eye(S, dtype=bool)]
        c_mean = float(np.nanmean(c_obs))
        # (b2) held-out cross projection (observed; null at the family level)
        packed = pack_grams(A, B, W_list)
        q_obs = heldout_cross_projection(packed)
        e = {
            "within_A": {**summ(sA, n_boot, seed), "t": t_stat(sA), "mean_cosine_to_loso_mean": float(np.nanmean(cosA)), "p_onesided_pos": sign_flip_p_onesided(sA, negative=False, seed=seed)},
            "within_B": {**summ(sB, n_boot, seed), "t": t_stat(sB), "mean_cosine_to_loso_mean": float(np.nanmean(cosB)), "p_onesided_pos": sign_flip_p_onesided(sB, negative=False, seed=seed)},
            "cross_cosine": {**summ(c_obs, n_boot, seed), "t": t_stat(c_obs), "p_onesided_neg": sign_flip_p_onesided(c_obs, negative=True, seed=seed),
                             "scene_permutation": {"n_perm": int(n_perm), "null_mean": float(perm_means.mean()), "null_sd": float(perm_means.std()),
                                                   "p_onesided_neg": float((np.sum(perm_means <= c_mean + 1e-15) + 1) / (n_perm + 1)),
                                                   "p_twosided": float((np.sum(np.abs(perm_means - perm_means.mean()) >= abs(c_mean - perm_means.mean()) - 1e-15) + 1) / (n_perm + 1)),
                                                   "observed_minus_null_mean": float(c_mean - perm_means.mean()),
                                                   "shared_direction_alignment_offdiag_mean": float(off.mean()),
                                                   "note": "the null's mean is the alignment a shared direction alone produces; observed - null_mean is the scene-specific part"}},
            "cross_heldout_projection": {**summ(q_obs, n_boot, seed), "t": t_stat(q_obs), "p_onesided_neg_naive": sign_flip_p_onesided(q_obs, negative=True, seed=seed),
                                         "note": "registered p (identity-relabel refit null, max-T) is filled in from the family level"},
            "norm_A_median": float(np.median(nA)), "norm_B_median": float(np.median(nB)),
        }
        out["variants"][name] = e
        fam[name] = {"within_A_vectors": A, "within_B_vectors": B, "cross_packed": packed, "cross_cosine": c_obs, "cross_q": q_obs}
    # ---- (c) template magnitude bookkeeping
    nT, nTA, nTB = row_norms(tau_shared), row_norms(tau_A), row_norms(tau_B)
    IAn, IBn = row_norms(IA), row_norms(IB)
    IpS_A, IpS_B = row_norms(var["I_perp_shared"]["A"]), row_norms(var["I_perp_shared"]["B_in_A"])
    coef = {"shared": {}, "perarm": {}}
    for x in ("ped", "cone"):
        coef["shared"][f"A_{x}"] = np.einsum("ij,ij->i", DA[x], uS)
        coef["shared"][f"B_{x}"] = np.einsum("ij,ij->i", np.array([DB[x][i] @ W_list[i].T for i in range(S)]), uS)
        coef["perarm"][f"A_{x}"] = np.einsum("ij,ij->i", DA[x], uA)
        coef["perarm"][f"B_{x}"] = np.einsum("ij,ij->i", DB[x], uB)
    heldout_frac = {}
    for x in ("ped", "cone"):
        heldout_frac[f"A_{x}"] = float(np.mean(coef["shared"][f"A_{x}"] ** 2 / np.maximum(row_norms(DA[x]) ** 2, 1e-300)))
        heldout_frac[f"B_{x}"] = float(np.mean(coef["shared"][f"B_{x}"] ** 2 / np.maximum(row_norms(DB[x]) ** 2, 1e-300)))
    # in-sample: the four fields stacked in A coordinates with the full-data transport
    F = np.concatenate([DA["ped"], DA["cone"], DB["ped"] @ W_full.T, DB["cone"] @ W_full.T], axis=0)
    u_full = unit(F.mean(0))
    lamF = np.clip(np.linalg.eigvalsh(F @ F.T), 0, None)
    insample = {"mean_direction_fraction_all_fields": energy_fraction_along(F, u_full), "pc1_fraction_all_fields": float(lamF.max() / lamF.sum()) if lamF.sum() > 0 else float("nan"),
                "mean_direction_fraction_per_field": {"A_ped": energy_fraction_along(DA["ped"], u_full), "A_cone": energy_fraction_along(DA["cone"], u_full),
                                                      "B_ped": energy_fraction_along(DB["ped"] @ W_full.T, u_full), "B_cone": energy_fraction_along(DB["cone"] @ W_full.T, u_full)},
                "template_norm_full": float(np.linalg.norm(F.mean(0)))}
    def rel(a, b):
        with np.errstate(invalid="ignore", divide="ignore"):
            return a / np.where(b > 0, b, np.nan)
    tmpl: dict[str, Any] = {
        "definition": {"shared": "LOSO mean over the other scenes of all four (arm x identity) DiD fields, arm B moved into arm A coordinates with the fold's hazard-free transport",
                       "perarm": "LOSO mean over the other scenes of the arm's two identity DiD fields (own coordinates)"},
        "identity_over_template": {
            "shared": {"A": {"median": float(np.median(rel(IAn, nT))), "mean": float(np.mean(rel(IAn, nT)))}, "B": {"median": float(np.median(rel(IBn, nT))), "mean": float(np.mean(rel(IBn, nT)))},
                       "A_perp": {"median": float(np.median(rel(IpS_A, nT)))}, "B_perp": {"median": float(np.median(rel(IpS_B, nT)))}},
            "perarm": {"A": {"median": float(np.median(rel(IAn, nTA)))}, "B": {"median": float(np.median(rel(IBn, nTB)))}},
            "note": "||I_i|| / ||tau_{-i}|| per scene; I = DiD(ped) - DiD(cone); _perp = template direction projected off"},
        "residual_over_field_norm": {v: {f"{arm}_{x}": float(np.median(rel(row_norms(res[(v, arm, x)]), row_norms(DA[x] if arm == "A" else DB[x])))) for arm in ARMS for x in ("ped", "cone")} for v in ("shared", "perarm")},
        "residual_over_template_norm": {v: {f"{arm}_{x}": float(np.median(rel(row_norms(res[(v, arm, x)]), nT if v == "shared" else (nTA if arm == "A" else nTB)))) for arm in ARMS for x in ("ped", "cone")} for v in ("shared", "perarm")},
        "heldout_energy_fraction_along_shared_template": heldout_frac,
        "insample": insample,
        "template_amplitude": {v: {k: summ(rel(c, nT if v == "shared" else (nTA if k.startswith("A") else nTB)), n_boot, seed) for k, c in coef[v].items()} for v in coef},
        "template_amplitude_asymmetry": {v: {"A_ped_minus_cone": summ(rel(coef[v]["A_ped"] - coef[v]["A_cone"], nT if v == "shared" else nTA), n_boot, seed),
                                            "B_ped_minus_cone": summ(rel(coef[v]["B_ped"] - coef[v]["B_cone"], nT if v == "shared" else nTB), n_boot, seed),
                                            "prediction": "A > 0 (solid pedestrian carries more template), B < 0 (solid cone)"} for v in coef},
        "template_norm_loso_median": {"shared": float(np.median(nT)), "A": float(np.median(nTA)), "B": float(np.median(nTB))},
    }
    if DnullA is not None and DnullB is not None:
        okA, okB = np.all(np.isfinite(DnullA), axis=1), np.all(np.isfinite(DnullB), axis=1)
        if okA.sum() >= 2 and okB.sum() >= 2:
            cnA = np.array([cosine(DnullA[i], uS[i]) if okA[i] else np.nan for i in range(S)])
            cnB = np.array([cosine(DnullB[i] @ W_list[i].T, uS[i]) if okB[i] else np.nan for i in range(S)])
            tmpl["h0prime_did_cosine_to_template"] = {"A": summ(cnA, n_boot, seed), "B": summ(cnB, n_boot, seed), "note": "DiD of the level-2 mirror-pose (off-lane) cells against the shared template; ~0 if the template is in-lane x action"}
    for v in ("shared", "perarm"):
        tmpl["template_amplitude_asymmetry"][v]["pattern_consistent"] = bool(ci_excludes_zero(tmpl["template_amplitude_asymmetry"][v]["A_ped_minus_cone"], True) and ci_excludes_zero(tmpl["template_amplitude_asymmetry"][v]["B_ped_minus_cone"], False))
    out["template"] = tmpl
    # ---- (d) rank / quota (in-sample)
    rk: dict[str, Any] = {"raw": {"A_ped": rank_summary(DA["ped"], d), "A_cone": rank_summary(DA["cone"], d), "B_ped": rank_summary(DB["ped"], d), "B_cone": rank_summary(DB["cone"], d), "all_four_fields": rank_summary(F, d)}}
    m4 = F.mean(0)
    rk["residual_shared"] = {"A_ped": rank_summary(DA["ped"] - m4, d), "A_cone": rank_summary(DA["cone"] - m4, d), "B_ped": rank_summary(DB["ped"] @ W_full.T - m4, d), "B_cone": rank_summary(DB["cone"] @ W_full.T - m4, d),
                             "all_four_fields": rank_summary(F - m4, d)}
    mA, mB = 0.5 * (DA["ped"].mean(0) + DA["cone"].mean(0)), 0.5 * (DB["ped"].mean(0) + DB["cone"].mean(0))
    rk["residual_perarm"] = {"A_ped": rank_summary(DA["ped"] - mA, d), "A_cone": rank_summary(DA["cone"] - mA, d), "B_ped": rank_summary(DB["ped"] - mB, d), "B_cone": rank_summary(DB["cone"] - mB, d)}
    rk["identity_contrast"] = {"A": rank_summary(IA, d), "B": rank_summary(IB, d), "A_perp_shared": rank_summary(var["I_perp_shared"]["A"], d), "B_perp_shared": rank_summary(var["I_perp_shared"]["B"], d)}
    rk["note"] = "uncentred spectrum of the per-scene rows; hard_rank = singular directions for 80 % energy; quota = conceptor tr(C)/d at 80 % retained energy (geometry_cross_arm.quota_for_capture); in-sample"
    out["rank"] = rk
    out["_family"] = fam
    return out


def cosine_matrix_fold(A: np.ndarray, B: np.ndarray, W_list: list[np.ndarray]) -> np.ndarray:
    """C[i, j] = cos(A_i, B_j W_i^T) = cos(A_i W_i, B_j): fold i's transport applied to every arm-B row."""
    S = A.shape[0]
    nA, nB = row_norms(A), row_norms(B)
    C = np.zeros((S, S))
    for i in range(S):
        num = (A[i] @ W_list[i]) @ B.T
        with np.errstate(invalid="ignore", divide="ignore"):
            C[i] = np.where((nA[i] > 0) & (nB > 0), num / (nA[i] * nB), np.nan)
    return C


# --------------------------------------------------------------------------- #
# Family-level tests (max-T over the sites of a group)
# --------------------------------------------------------------------------- #


def family_tests(entries: list[dict[str, Any]], args) -> dict[str, Any]:
    """Per group x variant: within-arm max-T (both arms), cross-arm held-out projection max-T against the identity-relabel
    refit null (shared flips across the family's sites); writes the per-site p-values back into the entries."""
    fams: dict[str, Any] = {}
    by_group: dict[str, list[dict[str, Any]]] = {}
    for e in entries:
        if e.get("status") == "ok":
            by_group.setdefault(e["group"], []).append(e)
    for g, es in by_group.items():
        scene_sets = {tuple(e["scenes"]) for e in es}
        if len(scene_sets) > 1:
            common = sorted(set.intersection(*(set(s) for s in scene_sets)))
            idx_of = {tuple(e["scenes"]): [e["scenes"].index(s) for s in common] for e in es}
            note = f"sites differ in complete scenes; family restricted to the {len(common)} common scenes"
        else:
            common = list(next(iter(scene_sets)))
            idx_of = {tuple(common): list(range(len(common)))}
            note = "all sites share the same scenes"
        S = len(common)
        fams[g] = {"n_scenes": S, "n_sites": len(es), "scenes_note": note, "variants": {}}
        for v in VARIANTS:
            rng_w = np.random.default_rng(args.seed)
            vecA = {}
            vecB = {}
            packed = {}
            for e in es:
                ix = np.asarray(idx_of[tuple(e["scenes"])])
                f = e["analysis"]["_family"][v]
                vecA[e["site_id"]] = f["within_A_vectors"][ix]
                vecB[e["site_id"]] = f["within_B_vectors"][ix]
                P = f["cross_packed"]
                S0 = P.shape[0]
                blocks = [P[:, k * S0 : (k + 1) * S0][np.ix_(ix, ix)] for k in range(4)]
                packed[e["site_id"]] = np.concatenate(blocks, axis=1)
            wa = sign_flip_maxt(vecA, args.n_perm, np.random.default_rng(args.seed))
            wb = sign_flip_maxt(vecB, args.n_perm, np.random.default_rng(args.seed))
            xc = sign_flip_maxt(packed, args.n_perm, rng_w, statistic=neg_heldout_cross_projection)
            fams[g]["variants"][v] = {
                "within_A": {"statistic": "s_i = I^A_i . unit(mean_{j != i} I^A_j); one-sided (> 0); LOSO mean refitted under every flip", "maxt": {k: wa[k] for k in ("n_scenes", "n_perm", "exact", "n_sites", "maxt_null_quantiles")}, "sites": wa["sites"]},
                "within_B": {"statistic": "same for I^B (own coordinates)", "maxt": {k: wb[k] for k in ("n_scenes", "n_perm", "exact", "n_sites", "maxt_null_quantiles")}, "sites": wb["sites"]},
                "cross_anti_alignment": {"statistic": "q_i = 0.5 [cos(I^A_i, T_i mean_{j != i} I^B_j) + cos(T_i I^B_i, mean_{j != i} I^A_j)]; tested as -q > 0 (anti-alignment)",
                                         "null": "identity-relabel refit: ped <-> cone relabelling per scene, same flips in both arms, every LOSO mean refitted; max-T over the family's sites with shared flips",
                                         "maxt": {k: xc[k] for k in ("n_scenes", "n_perm", "exact", "n_sites", "maxt_null_quantiles")}, "sites": xc["sites"]},
            }
            for e in es:
                sid = e["site_id"]
                ev = e["analysis"]["variants"][v]
                ev["within_A"].update({"p_raw_refit": wa["sites"][sid]["p_raw"], "p_maxt_fwer": wa["sites"][sid]["p_maxt_fwer"], "t_family": wa["sites"][sid]["t"]})
                ev["within_B"].update({"p_raw_refit": wb["sites"][sid]["p_raw"], "p_maxt_fwer": wb["sites"][sid]["p_maxt_fwer"], "t_family": wb["sites"][sid]["t"]})
                ev["cross_heldout_projection"].update({"p_raw_relabel_refit": xc["sites"][sid]["p_raw"], "p_maxt_fwer": xc["sites"][sid]["p_maxt_fwer"], "t_neg_family": xc["sites"][sid]["t"]})
                ev["detected"] = {
                    "within_A": bool(np.isfinite(wa["sites"][sid]["p_maxt_fwer"]) and wa["sites"][sid]["p_maxt_fwer"] < 0.05),
                    "within_B": bool(np.isfinite(wb["sites"][sid]["p_maxt_fwer"]) and wb["sites"][sid]["p_maxt_fwer"] < 0.05),
                    "anti_alignment": bool(np.isfinite(xc["sites"][sid]["p_maxt_fwer"]) and xc["sites"][sid]["p_maxt_fwer"] < 0.05 and ci_excludes_zero(ev["cross_cosine"], positive=False)),
                    "rule": "within: max-T p < 0.05; anti_alignment: relabel-refit max-T p < 0.05 AND the per-scene cosine CI lies below 0 (descriptive flags, not decision thresholds)",
                }
    return fams


def ranked_tables(entries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    tables: dict[str, list[dict[str, Any]]] = {}
    for e in entries:
        if e.get("status") != "ok":
            continue
        an = e["analysis"]
        row = {"site_id": e["site_id"], "n_scenes": an["n_scenes"]}
        for v in VARIANTS:
            ev = an["variants"][v]
            row[v] = {"cross_cos_mean": ev["cross_cosine"]["mean"], "cross_cos_ci": [ev["cross_cosine"]["ci_low"], ev["cross_cosine"]["ci_high"]],
                      "cross_cos_p_signflip_neg": ev["cross_cosine"]["p_onesided_neg"], "cross_cos_p_sceneperm_neg": ev["cross_cosine"]["scene_permutation"]["p_onesided_neg"],
                      "cross_cos_sceneperm_null_mean": ev["cross_cosine"]["scene_permutation"]["null_mean"],
                      "cross_q_mean": ev["cross_heldout_projection"]["mean"], "cross_q_t": ev["cross_heldout_projection"]["t"], "cross_q_p_maxt": ev["cross_heldout_projection"].get("p_maxt_fwer"),
                      "within_A_t": ev["within_A"]["t"], "within_A_p_maxt": ev["within_A"].get("p_maxt_fwer"), "within_A_cos": ev["within_A"]["mean_cosine_to_loso_mean"],
                      "within_B_t": ev["within_B"]["t"], "within_B_p_maxt": ev["within_B"].get("p_maxt_fwer"), "within_B_cos": ev["within_B"]["mean_cosine_to_loso_mean"],
                      "detected": ev.get("detected")}
        t = an["template"]
        row["template"] = {"insample_mean_dir_fraction": t["insample"]["mean_direction_fraction_all_fields"], "insample_pc1_fraction": t["insample"]["pc1_fraction_all_fields"],
                           "I_over_template_median_A": t["identity_over_template"]["shared"]["A"]["median"], "I_over_template_median_B": t["identity_over_template"]["shared"]["B"]["median"],
                           "asym_A": t["template_amplitude_asymmetry"]["shared"]["A_ped_minus_cone"]["mean"], "asym_B": t["template_amplitude_asymmetry"]["shared"]["B_ped_minus_cone"]["mean"],
                           "asym_pattern": t["template_amplitude_asymmetry"]["shared"]["pattern_consistent"]}
        row["rank"] = {"raw_hard_rank_all": an["rank"]["raw"]["all_four_fields"]["hard_rank"], "residual_shared_hard_rank_all": an["rank"]["residual_shared"]["all_four_fields"]["hard_rank"],
                       "raw_pc1_all": an["rank"]["raw"]["all_four_fields"]["pc1_fraction"], "residual_shared_pc1_all": an["rank"]["residual_shared"]["all_four_fields"]["pc1_fraction"],
                       "I_hard_rank_A": an["rank"]["identity_contrast"]["A"]["hard_rank"], "I_hard_rank_B": an["rank"]["identity_contrast"]["B"]["hard_rank"]}
        tables.setdefault(e["group"], []).append(row)
    for rows in tables.values():  # most anti-aligned (most negative held-out projection t) first
        rows.sort(key=lambda r: r["I"]["cross_q_t"] if r["I"]["cross_q_t"] is not None and np.isfinite(r["I"]["cross_q_t"]) else np.inf)
    return tables


def summarize(entries: list[dict[str, Any]], fams: dict[str, Any]) -> dict[str, Any]:
    s: dict[str, Any] = {}
    for g, f in fams.items():
        es = [e for e in entries if e.get("status") == "ok" and e["group"] == g]
        s[g] = {"n_sites": len(es), "n_scenes": f["n_scenes"], "variants": {}}
        for v in VARIANTS:
            det = [e["analysis"]["variants"][v]["detected"] for e in es]
            s[g]["variants"][v] = {"n_anti_alignment_detected": sum(d["anti_alignment"] for d in det), "n_within_A_detected": sum(d["within_A"] for d in det), "n_within_B_detected": sum(d["within_B"] for d in det),
                                   "cross_cos_mean_range": [min(e["analysis"]["variants"][v]["cross_cosine"]["mean"] for e in es), max(e["analysis"]["variants"][v]["cross_cosine"]["mean"] for e in es)] if es else None,
                                   "sites_anti_aligned": [e["site_id"] for e, d in zip(es, det) if d["anti_alignment"]]}
        s[g]["template_insample_mean_dir_fraction_range"] = [min(e["analysis"]["template"]["insample"]["mean_direction_fraction_all_fields"] for e in es), max(e["analysis"]["template"]["insample"]["mean_direction_fraction_all_fields"] for e in es)] if es else None
        s[g]["I_over_template_median_A_range"] = [min(e["analysis"]["template"]["identity_over_template"]["shared"]["A"]["median"] for e in es), max(e["analysis"]["template"]["identity_over_template"]["shared"]["A"]["median"] for e in es)] if es else None
        s[g]["n_template_asymmetry_pattern"] = sum(e["analysis"]["template"]["template_amplitude_asymmetry"]["shared"]["pattern_consistent"] for e in es)
    return s


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
    step = args.step
    if step >= n_steps:
        raise SystemExit(f"step {step} not in the dumps (n_steps {n_steps})")
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
    for si, site in enumerate(sites):
        actsA, tixA = load_site(args.dump_a, site, rows_a)
        actsB, tixB = load_site(args.dump_b, site, rows_b)
        if actsA.shape[2] == 1 or actsB.shape[2] == 1:
            entries.append({"site_id": site, "status": "adaln_skipped"})
            continue
        lv = tuple(args.transport_levels)
        XA, XB, scene_idx = matched_token_rows(token_rows(actsA, tixA, cells, step, lv), token_rows(actsB, tixB, cells, step, lv))
        if len(XA) < 8:
            for g in args.groups:
                entries.append({"site_id": site, "step": step, "group": g, "status": "no_matched_tokens"})
            continue
        W_folds, W_full, tlog = transport_folds(XA, XB, scene_idx, scenes)
        transport_log[site] = tlog
        for g in args.groups:
            entry: dict[str, Any] = {"site_id": site, "step": step, "group": g}
            PA, _ = pool_site(actsA, tixA, cells, token_sets, step, g)
            PB, _ = pool_site(actsB, tixB, cells, token_sets, step, g)
            ok = np.all(np.isfinite(PA), axis=1) & np.all(np.isfinite(PB), axis=1)
            keep, sub_cells = scenes, cells
            if ok.sum() < len(cells):
                bad = {c["pair_id"] for c in cells if not ok[c["row"]]}
                keep = [s for s in scenes if s not in bad]
                if len(keep) < 4:
                    entry["status"] = "no_tokens"
                    entries.append(entry)
                    continue
                sub_cells = [c for c in cells if c["pair_id"] in set(keep)]
                entry["n_scenes_dropped_no_tokens"] = len(scenes) - len(keep)
            rA, rB = scene_rows(PA, sub_cells, keep), scene_rows(PB, sub_cells, keep)
            DA = {"ped": rA["did_ped"], "cone": rA["did_cone"]}
            DB = {"ped": rB["did_ped"], "cone": rB["did_cone"]}
            entry["scenes"] = list(keep)
            entry["analysis"] = identity_contrast_analysis(DA, DB, keep, W_folds, W_full, args.n_boot, args.n_perm, args.seed, rA["did_null"], rB["did_null"])
            entry["status"] = "ok"
            entries.append(entry)
        print(f"[{si + 1}/{len(sites)}] {site} ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    fams = family_tests(entries, args)
    tables = ranked_tables(entries)
    summary = summarize(entries, fams)
    for e in entries:
        if e.get("analysis"):
            e["analysis"].pop("_family", None)
    report = {
        "protocol": PROTOCOL, "dump_a": str(args.dump_a), "dump_b": str(args.dump_b), "stimulus": str(args.stimulus), "arms": {"A": args.arm_a_name, "B": args.arm_b_name},
        "domain": args.domain, "token_group_source": alias_note(args.domain), "n_scenes": len(scenes), "scenes": scenes, "n_cells": len(cells), "alignment": status, "step": step,
        "groups": list(args.groups), "sites": sites, "transport_levels": list(args.transport_levels),
        "definitions": {
            "DiD(l)": "(h[l,1] - h[l,0]) - (h[0,1] - h[0,0]) on the group-pooled activations (mean over the scene-level union of the group's tokens); levels 0 sidewalk, 1 pedestrian in lane, 2 H0' mirror pose, 3 cone in lane; actions 0 brake, 1 throttle",
            "fields": "D^A_ped = DiD(1) in arm A, D^A_cone = DiD(3) in arm A, D^B_ped, D^B_cone likewise (arm B coordinates)",
            "template_shared": "tau_{-i} = mean over the other scenes of the four fields, arm B moved into arm A coordinates with fold i's transport (v_A = v_B W_i^T)",
            "template_perarm": "tau^X_{-i} = mean over the other scenes of arm X's two fields",
            "I": "I^X_i = r^X_ped,i - r^X_cone,i = D^X_ped,i - D^X_cone,i (template-invariant by construction)",
            "I_perp_shared": "I with its component along unit(tau_{-i}) removed (arm B after transport into A coordinates)",
            "I_perp_perarm": "I^X with its component along unit(tau^X_{-i}) removed, in the arm's own coordinates",
            "transport": "orthogonal Procrustes W fitted LOSO on matched per-token rows of the hazard-free cells (levels " + ",".join(str(v) for v in args.transport_levels) + ") as in geometry_cross_arm.py; x_B ~= x_A W, row vectors map v_B = v_A W",
            "prediction": "reversed assignment (consequence attached to the pedestrian in A and to the cone in B) -> I^A anti-aligned with T(I^B); within each arm I has a consistent direction",
            "unit": "scene (pair_id); CIs = scene bootstrap of the mean; sign-flip p as in stats_utils; max-T = Westfall-Young over the sites of a group family with shared flips (cgs_stats.sign_flip_maxt)",
        },
        "transport_per_site": transport_log, "families": fams, "ranked_tables": tables, "summary": summary, "entries": entries,
        "runtime_s": time.time() - t0, "interpretation_scope": INTERPRETATION_SCOPE,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "identity_geometry.json").write_text(json.dumps(finite(report), indent=1) + "\n")
    (args.out / "identity_geometry.md").write_text(markdown_summary(report))
    for g, rows in tables.items():
        print(f"== group {g} (sorted by the anti-alignment t of I, most anti-aligned first) ==")
        for r in rows[:8]:
            v = r["I"]
            fmt = lambda x: "nan" if x is None or not np.isfinite(x) else f"{x:+.3f}"  # noqa: E731
            print(f"  {r['site_id']:>14s} cos(I^A,T I^B) {fmt(v['cross_cos_mean'])} [{fmt(v['cross_cos_ci'][0])},{fmt(v['cross_cos_ci'][1])}] q_t {fmt(v['cross_q_t'])} p_maxT {fmt(v['cross_q_p_maxt'])} | "
                  f"within A t {fmt(v['within_A_t'])} p {fmt(v['within_A_p_maxt'])} B t {fmt(v['within_B_t'])} p {fmt(v['within_B_p_maxt'])} | "
                  f"template frac {fmt(r['template']['insample_mean_dir_fraction'])} |I|/|tau| {fmt(r['template']['I_over_template_median_A'])}/{fmt(r['template']['I_over_template_median_B'])}")
    return report


def markdown_summary(report: dict[str, Any]) -> str:
    def f(x, nd=3):
        return "nan" if x is None or (isinstance(x, float) and not np.isfinite(x)) else (f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x))

    lines = [f"# Identity-contrast geometry under the shared template ({report['protocol']})", "",
             f"Scope: {report['interpretation_scope']}", "",
             f"Dumps: A = `{report['dump_a']}`, B = `{report['dump_b']}`; stimulus `{report['stimulus']}`; {report['n_scenes']} discovery scenes, {len(report['sites'])} sites, step {report['step']}, groups {', '.join(report['groups'])}; runtime {report['runtime_s']:.0f} s.", "",
             "Prediction (reversed assignment): I^A = DiD(ped) - DiD(cone) in arm A and the transported I^B are ANTI-aligned; each arm's I has a consistent direction across scenes. "
             "`cos` = per-scene cosine of I^A with T(I^B) (mean [scene-bootstrap 95 % CI]); `p_perm` = scene-permutation one-sided p (null mean in brackets = shared-direction alignment); "
             "`q` = held-out projection onto the other arm's LOSO identity direction with the identity-relabel refit max-T p (registered); `within A/B` = LOSO direction-consistency t with max-T p; "
             "`template` = in-sample fraction of the four DiD fields' energy along the shared mean direction; `|I|/|tau|` = median identity-contrast norm over template norm (A / B); "
             "`asym` = template-amplitude asymmetry ped - cone in units of the template norm (A / B; predicted + / -); `rank` = hard rank for 80 % energy of the four raw fields -> of the shared-template residuals.", ""]
    for g, rows in report["ranked_tables"].items():
        s = report["summary"][g]
        lines += [f"## group `{g}` -- {s['n_sites']} sites, {s['n_scenes']} scenes", ""]
        for v in VARIANTS:
            sv = s["variants"][v]
            lines.append(f"- {v}: anti-alignment detected at {sv['n_anti_alignment_detected']}/{s['n_sites']} sites {sv['sites_anti_aligned'] if sv['sites_anti_aligned'] else ''}; within-arm consistency A {sv['n_within_A_detected']}/{s['n_sites']}, B {sv['n_within_B_detected']}/{s['n_sites']}; cos range {f(sv['cross_cos_mean_range'][0])} .. {f(sv['cross_cos_mean_range'][1])}")
        lines.append(f"- template energy fraction {f(s['template_insample_mean_dir_fraction_range'][0])} .. {f(s['template_insample_mean_dir_fraction_range'][1])}; |I|/|tau| (A) {f(s['I_over_template_median_A_range'][0])} .. {f(s['I_over_template_median_A_range'][1])}; template-amplitude asymmetry pattern (A > 0, B < 0) at {s['n_template_asymmetry_pattern']}/{s['n_sites']} sites")
        lines += ["", "| site | cos(I^A, T I^B) [CI] | p sign-flip | p_perm [null mean] | q mean, t, p_maxT | within A t (p_maxT) | within B t (p_maxT) | I_perp cos [CI], p_maxT | template | \\|I\\|/\\|tau\\| | asym A / B | rank raw -> resid |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            v, vp, t, rk = r["I"], r["I_perp_shared"], r["template"], r["rank"]
            lines.append(f"| {r['site_id']} | {f(v['cross_cos_mean'])} [{f(v['cross_cos_ci'][0])}, {f(v['cross_cos_ci'][1])}] | {f(v['cross_cos_p_signflip_neg'], 4)} | {f(v['cross_cos_p_sceneperm_neg'], 4)} [{f(v['cross_cos_sceneperm_null_mean'])}] | "
                         f"{f(v['cross_q_mean'])}, {f(v['cross_q_t'], 2)}, {f(v['cross_q_p_maxt'], 4)} | {f(v['within_A_t'], 2)} ({f(v['within_A_p_maxt'], 4)}) | {f(v['within_B_t'], 2)} ({f(v['within_B_p_maxt'], 4)}) | "
                         f"{f(vp['cross_cos_mean'])} [{f(vp['cross_cos_ci'][0])}, {f(vp['cross_cos_ci'][1])}], {f(vp['cross_q_p_maxt'], 4)} | {f(t['insample_mean_dir_fraction'])} | {f(t['I_over_template_median_A'], 2)} / {f(t['I_over_template_median_B'], 2)} | "
                         f"{f(t['asym_A'], 2)} / {f(t['asym_B'], 2)} | {rk['raw_hard_rank_all']} -> {rk['residual_shared_hard_rank_all']} (I: {rk['I_hard_rank_A']}/{rk['I_hard_rank_B']}) |")
        lines.append("")
    lines += ["## Transport diagnostics (held-out relative residual, transported vs raw)", ""]
    for site, tl in report["transport_per_site"].items():
        lines.append(f"- {site}: {f(tl['heldout_relative_residual_transported'])} vs {f(tl['heldout_relative_residual_raw'])} ({tl['n_rows']} matched hazard-free token rows)")
    if "self_test" in report:
        lines += ["", "## Self-test", "", "```", json.dumps(report["self_test"], indent=1), "```"]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Synthetic self-test
# --------------------------------------------------------------------------- #


def synthetic_dumps(root: Path, *, null: bool, n_scenes: int = 16, d: int = 64, sites=("L03.mlp_out", "L05.resid_post"), seed: int = 0, noise: float = 0.3,
                    template_amp: float = 3.0, identity_amp: float = 0.6, appearance_action_amp: float = 0.2) -> tuple[Path, Path, Path]:
    """Two random driving dumps (layout of ``localize_interaction.py --dump-groups``) on shared scenes, arm B's coordinates
    rotated by a random orthogonal W_true.  Planted in BOTH arms and BOTH in-lane identities: the consequence template
    (in-lane x throttle, direction tau, amplitude ~template_amp) and a zero-mean shared appearance x action term; in the
    planted case additionally a small identity component (direction u_id orthogonal to tau, amplitude ~identity_amp,
    positive sign) on the throttle cell of each arm's SOLID identity (pedestrian in A, cone in B) so that I^A ~ +u_id
    and I^B ~ -u_id.  ``null``: no identity component."""
    rng = np.random.default_rng(seed)
    stim, dA, dB = root / "stimulus", root / "arm_A", root / "arm_B"
    (stim / "masks").mkdir(parents=True, exist_ok=True)
    for p in (dA, dB):
        (p / "activations").mkdir(parents=True, exist_ok=True)
    Bs = _orthonormal(rng, d, 12)
    S_ped, S_cone, S_null, tau, u_id, a_dir = Bs[:3], Bs[3:6], Bs[6:9], Bs[9], Bs[10], Bs[11]
    W_true = np.linalg.qr(rng.normal(size=(d, d)))[0]
    E = rng.normal(size=(256, d)) * 2.0
    gamma = 0.5
    cells, rows, tix = [], [], []
    acts = {"A": [], "B": []}
    yy, xx = np.mgrid[:256, :256]
    for s in range(n_scenes):
        seed_id = 500 + s
        pid = f"drive_seed{seed_id:06d}"
        content = rng.normal(size=(256, d)) * 0.5
        c_ped, c_cone, c_null = rng.normal(size=3) * [3.0, 2.0, 1.5], rng.normal(size=3) * [3.0, 2.0, 1.5], rng.normal(size=3) * [3.0, 2.0, 1.5]
        kappa = template_amp * (1.0 + 0.2 * rng.normal())
        rho = 0.0 if null else identity_amp * (1.0 + 0.3 * rng.normal())
        eta_ped, eta_cone = rng.normal(size=2) * appearance_action_amp
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
            cells.append({"row": len(cells), "pair_id": pid, "seed": seed_id, "hazard": lv, "candidate_action": a, "cell_id": cid, "artifact": f"cells/{cid}.npz"})
            rows.append({"cell_id": cid, "pair_id": pid, "seed": seed_id, "hazard": lv, "candidate_action": a, "artifact": f"cells/{cid}.npz"})
            per_arm = {}
            for arm in ("A", "B"):
                x = E[tok_union] + content[tok_union]
                app = np.zeros(d)
                if lv == PED:
                    app = c_ped @ S_ped
                elif lv == OBJ:
                    app = c_cone @ S_cone
                elif lv == NULL:
                    app = c_null @ S_null
                x = x + in_h[:, None] * app[None, :]
                x = x + a * g * (1.0 + gamma * (lv > 0)) * a_dir[None, :]  # AdaLN-style hazard-scaled action main effect
                if a == 1 and lv in (PED, OBJ):
                    x = x + kappa * tau[None, :]  # shared consequence template: in-lane x throttle, both identities, both arms
                    x = x + (eta_ped if lv == PED else eta_cone) * (S_ped[0] if lv == PED else S_cone[0])[None, :]  # shared appearance x action (zero-mean)
                solid = PED if arm == "A" else OBJ
                if a == 1 and lv == solid and rho != 0.0:
                    x = x + rho * u_id[None, :]  # identity-attached component on the solid identity only
                per_arm[arm] = x
            nA = rng.normal(size=per_arm["A"].shape) * noise
            nB = rng.normal(size=per_arm["B"].shape) * noise
            acts["A"].append((per_arm["A"] + nA)[None])
            acts["B"].append(((per_arm["B"] + nB) @ W_true)[None])
            tix.append(tok_union)
    n, T = len(cells), max(len(t) for t in tix)
    token_index = np.full((n, 1, T), -1, dtype=np.int64)
    arr = {"A": np.zeros((n, 1, T, d), np.float32), "B": np.zeros((n, 1, T, d), np.float32)}
    for r in range(n):
        token_index[r, :, : len(tix[r])] = tix[r][None, :]
        for arm in ("A", "B"):
            arr[arm][r, :, : len(tix[r])] = acts[arm][r]
    for arm, dp in (("A", dA), ("B", dB)):
        for i, site in enumerate(sites):
            jitter = np.random.default_rng(seed + 100 + i).normal(size=arr[arm].shape).astype(np.float32) * 0.05
            np.savez_compressed(dp / "activations" / f"{site}.npz", acts=(arr[arm] + jitter).astype(np.float16), token_index=token_index)
        (dp / "activations" / "index.json").write_text(json.dumps({"cells": cells, "n_steps": 1, "group_frame_offset": 0, "domain": "driving", "dump_groups": ["hazard", "corridor"], "dump_hazard_levels": [0, 1, 2, 3],
                                                                     "synthetic": {"null": null, "d": d, "noise": noise, "template_amp": template_amp, "identity_amp": 0.0 if null else identity_amp,
                                                                                   "appearance_action_amp": appearance_action_amp, "rotation": "random orthogonal W_true on arm B (x_B = x_A W_true)"}}, indent=1))
    (stim / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return stim, dA, dB


def self_test_verdict(report: dict[str, Any], planted: bool) -> dict[str, Any]:
    """Planted: every site must be detected (anti-alignment with relabel-refit max-T p < 0.05 and cosine CI < 0; within-arm
    consistency max-T p < 0.05 in both arms) on variants I and I_perp_shared, and the template must dominate (median
    ||I|| / ||tau|| < 0.5 at every site).  Null: no site detected on any variant, |mean cosine| and |mean within-arm cosine|
    below NULL_COS_BAND everywhere (magnitude bands, as in geometry_cross_arm.py; significance counts are reported)."""
    es = [e for e in report["entries"] if e.get("status") == "ok"]
    det = {v: [e["analysis"]["variants"][v]["detected"] for e in es] for v in VARIANTS}
    cos_means = {v: [abs(e["analysis"]["variants"][v]["cross_cosine"]["mean"]) for e in es] for v in VARIANTS}
    within_cos = [max(abs(e["analysis"]["variants"]["I"]["within_A"]["mean_cosine_to_loso_mean"]), abs(e["analysis"]["variants"]["I"]["within_B"]["mean_cosine_to_loso_mean"])) for e in es]
    ratio = [max(e["analysis"]["template"]["identity_over_template"]["shared"]["A"]["median"], e["analysis"]["template"]["identity_over_template"]["shared"]["B"]["median"]) for e in es]
    tfrac = [e["analysis"]["template"]["insample"]["mean_direction_fraction_all_fields"] for e in es]
    if planted:
        checks = {"anti_alignment_I": bool(es) and all(d["anti_alignment"] for d in det["I"]),
                  "anti_alignment_I_perp": bool(es) and all(d["anti_alignment"] for d in det["I_perp_shared"]),
                  "within_arm_I": bool(es) and all(d["within_A"] and d["within_B"] for d in det["I"]),
                  "template_dominates": bool(ratio) and max(ratio) < 0.5 and min(tfrac) > 0.5}
    else:
        checks = {"no_anti_alignment": bool(es) and not any(d["anti_alignment"] for v in VARIANTS for d in det[v]),
                  "cosine_band": bool(es) and max(max(cos_means[v]) for v in VARIANTS) < NULL_COS_BAND,
                  "within_band": bool(within_cos) and max(within_cos) < NULL_COS_BAND}
    return {"planted": planted, "checks": checks, "passed": all(checks.values()), "n_sites": len(es), "n_groups": len(report["ranked_tables"]),
            "counts": {v: {"anti_alignment": [sum(d["anti_alignment"] for d in det[v]), len(det[v])], "within_A": [sum(d["within_A"] for d in det[v]), len(det[v])], "within_B": [sum(d["within_B"] for d in det[v]), len(det[v])]} for v in VARIANTS},
            "max_abs_cross_cos_mean": {v: max(cos_means[v]) if cos_means[v] else None for v in VARIANTS}, "max_abs_within_cos_mean_I": max(within_cos) if within_cos else None,
            "identity_over_template_max_median": max(ratio) if ratio else None, "template_fraction_range": [min(tfrac), max(tfrac)] if tfrac else None}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dump-a", type=Path, default=None, help="localization dump of arm A (pedestrian solid / cone ghost)")
    ap.add_argument("--dump-b", type=Path, default=None, help="localization dump of arm B (pedestrian ghost / cone solid)")
    ap.add_argument("--stimulus", type=Path, default=None, help="shared stimulus dir (masks/<cell_id>.npz), e.g. drive_factorial_merged/armA")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--discovery-seeds", type=Path, default=None)
    ap.add_argument("--domain", default="driving")
    ap.add_argument("--sites", nargs="*", default=None)
    ap.add_argument("--step", type=int, default=0, help="imagined step (the slimmed dumps hold step 0 only)")
    ap.add_argument("--groups", nargs="*", default=list(DEFAULT_GROUPS))
    ap.add_argument("--transport-levels", type=int, nargs="*", default=list(ACTION_LEVELS), help="hazard levels whose per-token rows fit the Procrustes transport (hazard-free by default, as geometry_cross_arm.py)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=2000, help="sign flips / relabellings for the max-T tests and scene permutations for the permutation null")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm-a-name", default="A: PED-solid / OBJ-ghost")
    ap.add_argument("--arm-b-name", default="B: PED-ghost / OBJ-solid")
    ap.add_argument("--self-test", action="store_true", help="write planted and null synthetic dump pairs under <out>/self_test and check detection / non-detection")
    ap.add_argument("--synthetic-scenes", type=int, default=16)
    ap.add_argument("--synthetic-dim", type=int, default=64)
    ap.add_argument("--synthetic-noise", type=float, default=0.3)
    ap.add_argument("--synthetic-sites", nargs="*", default=["L03.mlp_out", "L05.resid_post"])
    return ap


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.self_test:
        base = args.out
        verdicts = {}
        reports = {}
        for label, null in (("planted", False), ("null", True)):
            root = base / "self_test" / label
            stim, dA, dB = synthetic_dumps(root, null=null, n_scenes=args.synthetic_scenes, d=args.synthetic_dim, sites=tuple(args.synthetic_sites), seed=args.seed, noise=args.synthetic_noise)
            a = argparse.Namespace(**vars(args))
            a.dump_a, a.dump_b, a.stimulus, a.out, a.discovery_seeds = dA, dB, stim, root / "out", None
            a.groups = [g for g in args.groups if g in ("hazard", "corridor", "hazard_corridor")]
            rep = run(a)
            verdicts[label] = self_test_verdict(rep, planted=not null)
            rep["self_test"] = verdicts[label]
            (a.out / "identity_geometry.json").write_text(json.dumps(finite(rep), indent=1) + "\n")
            (a.out / "identity_geometry.md").write_text(markdown_summary(rep))
            reports[label] = rep
            print(json.dumps(finite(verdicts[label])))
        passed = all(v["passed"] for v in verdicts.values())
        (base / "self_test" / "verdict.json").write_text(json.dumps(finite({"passed": passed, **verdicts}), indent=1) + "\n")
        if not passed:
            raise SystemExit(f"self-test FAILED: {json.dumps(finite(verdicts))}")
        print("SELF_TEST_PASSED")
        return {"self_test": verdicts, "reports": reports}
    if not (args.dump_a and args.dump_b and args.stimulus):
        raise SystemExit("--dump-a, --dump-b and --stimulus are required (or --self-test)")
    return run(args)


if __name__ == "__main__":
    main()
