"""Representation geometry of the fitted rank-4 intervention subspace.

Question. Is the fitted intervention subspace a compact, model-native coordinate system
(few directions, reused across tasks, constant edit: the Othello-GPT / linear-representation
picture), or a distributed, task-specific object (broad spatial support, all four directions in
use, orthogonal across tasks: the "Interpreting Physics in Video World Models" picture, where
steering needs tens of directions)?

Inputs (read-only): the fitted refined operator banks (torch; Reach and Reach-Wall only, other
tasks are skipped with a logged reason), the fresh confirmation results tree (delivered
coefficients of the refined arms), the frozen analysis report (pre-registered arm-vs-native
contrasts, used only to name the behavioural regime the geometry is read against).

Everything computed here is EXPLORATORY / descriptive. Only the arm-vs-native contrasts quoted
from the frozen report are pre-registered. Never calls the model, simulator or GPU; never
modifies src/offline_study or analysis/mechanism/common.py.

Outputs: OUT/representation_geometry.json, OUT/representation_geometry.md,
OUT/representation_geometry_{spatial_loadings,principal_angles,coefficient_spectrum}.png
"""
from __future__ import annotations

import logging
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT = "representation_geometry"
SEED = 20260913
RANDOM_DRAWS = 1000
PERMUTATIONS = 5000
PATCHES, FEATURES = 256, 400
AMBIENT = PATCHES * FEATURES
LEARNED, CONTROL = "fixed_rank4", "matched_random_fixed_rank4"
GRID = 16

# Explicit rubric thresholds (stated verbatim in the .md).
RUBRIC = dict(
    spatial_concentrated_eff_patches=64.0, spatial_broad_eff_patches=192.0,
    feature_concentrated_eff=100.0, feature_broad_eff=300.0,
    pr_single_direction=1.5, pr_full_rank=3.0,
    intercept_constant_share=0.8, intercept_state_share=0.2,
    cross_task_p=0.01, cross_task_shared_angle_deg=45.0, cross_task_orthogonal_overlap=0.01,
)

log = logging.getLogger(SCRIPT)
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200, "savefig.dpi": 200,
    "font.family": "sans-serif"})


# =============================================================================== helpers
def entropy_bits(p):
    p = np.asarray(p, float)
    p = p[p > 0]
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def gini(p):
    x = np.sort(np.asarray(p, float))
    n = len(x)
    if x.sum() <= 0:
        return float("nan")
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def top_mass(p, k):
    return float(np.sort(np.asarray(p, float))[::-1][:k].sum())


def loading_stats(p, k_top, n):
    h = entropy_bits(p)
    return dict(entropy_bits=h, max_entropy_bits=float(np.log2(n)), effective_count=float(2 ** h),
                gini=gini(p), top16_mass=top_mass(p, k_top), uniform_top16_mass=float(k_top / n),
                max_loading=float(np.max(p)))


def principal_angles_deg(qa, qb):
    """Rows of qa, qb orthonormal. Returns ascending angles (deg) and the overlap
    ||qa qb^T||_F^2 / rank(qb) (fraction of B captured by A)."""
    s = np.clip(np.linalg.svd(qa @ qb.T, compute_uv=False), -1.0, 1.0)
    return np.degrees(np.arccos(s)), float((s ** 2).sum() / qb.shape[0])


def random_subspace(rng, d, k):
    q, _ = np.linalg.qr(rng.standard_normal((d, k), dtype=np.float32))
    return q.T.astype(np.float64)


def participation_ratio(eig):
    eig = np.clip(np.asarray(eig, float), 0, None)
    s = eig.sum()
    return float(s ** 2 / (eig ** 2).sum()) if s > 0 and (eig ** 2).sum() > 0 else float("nan")


def cosine(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def fmt(x, nd=3):
    if x is None:
        return "n/a"
    try:
        if not np.isfinite(x):
            return "n/a"
    except TypeError:
        return str(x)
    return f"{x:.{nd}f}"


def deg(x):
    s = fmt(x, 1)
    return s if s == "n/a" else s + "°"


# ============================================================================ regime
def behavioural_regime(report, df, tasks):
    """Pre-registered fixed_rank4-native contrasts (quoted) + EXPLORATORY learned-vs-control
    paired bootstrap (family = number of tasks) -> common.classify_regime per task."""
    per_task, family = {}, 0
    avail = [t for t in tasks if t in report.get("results", {})]
    family = max(1, len(avail))
    for task in avail:
        r = report["results"][task]
        c = r["contrasts"].get(f"{LEARNED}-native")
        if c is None:
            continue
        vs_native = dict(difference=float(c["difference_pp"]), lower=float(c["simultaneous_95_interval_pp"][0]),
                         upper=float(c["simultaneous_95_interval_pp"][1]), pre_registered=True)
        vs_control = None
        if len(df) and task in set(df.task):
            m = common.success_matrix(df, task, arms=(LEARNED, CONTROL))
            if LEARNED in m.columns and CONTROL in m.columns and len(m) >= 2:
                b = common.paired_bootstrap(m[LEARNED].values, m[CONTROL].values, family=family)
                vs_control = dict(difference=b["difference"], lower=b["lower"], upper=b["upper"], n=b["n"],
                                  family=b["family"], exploratory=True)
        if vs_control is None:
            vs_control = dict(difference=None, lower=None, upper=None, exploratory=True, missing=True,
                              reason="learned or control arm absent from the results tree for this task")
            cls = "harmful" if vs_native["upper"] < 0 else ("positive_vs_native_only" if vs_native["lower"] > 0
                                                             else "inconclusive")
        else:
            cls = common.classify_regime(vs_native, vs_control)
        per_task[task] = dict(vs_native=vs_native, vs_control=vs_control, classification=cls,
                              success_percent=r.get("success_percent", {}))
    classes = [v["classification"] for v in per_task.values()]
    positive_ish = [t for t, v in per_task.items() if v["classification"] in
                    ("confirmed", "positive_vs_native_only", "positive_vs_control_only")]
    confirmed = [t for t, v in per_task.items() if v["classification"] == "confirmed"]
    harmful = [t for t, v in per_task.items() if v["classification"] == "harmful"]
    if not classes:
        overall = "inconclusive"
    elif confirmed and not harmful:
        overall = "positive"
    elif harmful and not positive_ish:
        overall = "negative"
    elif positive_ish and harmful:
        overall = "mixed"
    elif positive_ish:
        overall = "mixed"
    else:
        overall = "inconclusive"
    lean = {t: ("up" if v["vs_native"]["difference"] > 0 else "down" if v["vs_native"]["difference"] < 0 else "flat")
            for t, v in per_task.items()}
    return dict(overall=overall, per_task=per_task, confirmed=confirmed, positive_ish=positive_ish, harmful=harmful,
                point_estimate_lean=lean, exploratory_control_family=family,
                note="arm-vs-native intervals are the pre-registered Bonferroni-48 simultaneous intervals from the "
                     "frozen report; learned-vs-control is EXPLORATORY (paired scenario bootstrap, Bonferroni over "
                     f"{family} tasks)")


# ============================================================================= banks
def basis_diagnostics(bank):
    ops = bank["bank"]["operators"]
    out = {}
    for arm in (LEARNED, CONTROL):
        b = ops[arm]["basis"].detach().cpu().double().numpy()  # [4, 256, 400]
        flat = b.reshape(4, -1)
        gram = flat @ flat.T
        orth_err = float(np.abs(gram - np.eye(4)).max())
        patch = (b ** 2).sum(2)  # [4, 256]
        feat = (b ** 2).sum(1)  # [4, 400]
        dirs = []
        for d in range(4):
            dirs.append(dict(spatial=loading_stats(patch[d], 16, PATCHES), feature=loading_stats(feat[d], 16, FEATURES)))
        sub_patch, sub_feat = patch.mean(0), feat.mean(0)
        out[arm] = dict(orthonormality_max_abs_error=orth_err, directions=dirs,
                        subspace_mean=dict(spatial=loading_stats(sub_patch, 16, PATCHES),
                                           feature=loading_stats(sub_feat, 16, FEATURES)),
                        patch_loadings=patch.tolist(), subspace_mean_patch_loading=sub_patch.tolist(),
                        subspace_mean_feature_loading=sub_feat.tolist(),
                        note="per-direction loadings depend on the basis chosen inside the subspace; the subspace "
                             "mean (diagonal of the rank-4 projector / 4) is basis-invariant")
    return out


def coefficient_map(bank):
    ops = bank["bank"]["operators"]
    out = {}
    for arm in (LEARNED, CONTROL):
        a = ops[arm]["map"].detach().cpu().double().numpy()  # rows: 4 directions; cols: [1, s1, s2, s3]
        sv = np.linalg.svd(a, compute_uv=False)
        intercept = a[:, 0]
        scores = a[:, 1:]
        i2, s2 = float(intercept @ intercept), float((scores ** 2).sum())
        out[arm] = dict(map=a.tolist(), singular_values=sv.tolist(),
                        condition_number=float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf"),
                        intercept_column_norm=float(np.sqrt(i2)), score_columns_frobenius=float(np.sqrt(s2)),
                        intercept_share=float(i2 / (i2 + s2)) if i2 + s2 > 0 else float("nan"),
                        per_score_column_norm=np.linalg.norm(scores, axis=0).tolist(),
                        constant_direction=(intercept / np.linalg.norm(intercept)).tolist() if i2 > 0 else None,
                        note="c = A [1, s1, s2, s3]; scores are standardised by the bank's `scale`, so the share "
                             "||A[:,0]||^2 / ||A||_F^2 approximates the constant fraction of the pre-rescaling edit "
                             "energy only under unit-variance scores")
    return out


def readout_alignment(bank):
    """Principal angles between the learned rank-4 basis and the 3-score readout projection."""
    proj = bank["bank"]["projection"].detach().cpu().double().numpy()
    q, _ = np.linalg.qr(proj.T)
    q = q.T
    b = bank["bank"]["operators"][LEARNED]["basis"].detach().cpu().double().numpy().reshape(4, -1)
    ang, ov = principal_angles_deg(b, q)
    return dict(angles_deg=ang.tolist(), overlap_of_readout_by_basis=ov)


def checkpoint_of(bank):
    return str(bank["bank"].get("binding", {}).get("checkpoint_sha256", "unknown"))


def readout_rows(bank):
    """Orthonormal rows spanning the 3-score readout projection."""
    proj = bank["bank"]["projection"].detach().cpu().double().numpy()
    q, _ = np.linalg.qr(proj.T)
    return q.T


def residual_rows(basis, readout, rel_tol=0.1):
    """Rows of `basis` with the readout span projected out, as orthonormal rows.

    Returns (rows [r, d], singular values [4]); r = number of singular values above rel_tol * max.
    When the readout lies exactly inside the rank-4 basis this is the single genuinely fitted
    direction (singular values ~ [1, 0, 0, 0])."""
    res = basis - (basis @ readout.T) @ readout
    _, s, vt = np.linalg.svd(res, full_matrices=False)
    keep = s > rel_tol * max(float(s[0]), 1e-12)
    return vt[keep], s


def readout_decomposition(ba, bb, qa, qb, ra, rb):
    """Split the overlap tr(P_a P_b) = ||Ba Bb^T||_F^2 of two same-checkpoint learned bases into the
    part carried by the two 3-score readouts, the cross terms, and the residual-vs-residual part.
    The split is exact when each readout lies inside its basis (then P = P_readout + P_residual)."""
    total = float((np.linalg.svd(ba @ bb.T, compute_uv=False) ** 2).sum())
    ang_rr, _ = principal_angles_deg(qa, qb)
    readout_part = float((np.linalg.svd(qa @ qb.T, compute_uv=False) ** 2).sum())
    resid_part = float((np.linalg.svd(ra @ rb.T, compute_uv=False) ** 2).sum()) if len(ra) and len(rb) else 0.0
    cross = total - readout_part - resid_part
    ang_ra_bb, _ = principal_angles_deg(ra, bb) if len(ra) else (np.array([]), 0.0)
    ang_rb_ba, _ = principal_angles_deg(rb, ba) if len(rb) else (np.array([]), 0.0)
    ang_rr_res, _ = principal_angles_deg(ra, rb) if len(ra) and len(rb) else (np.array([]), 0.0)
    return dict(total_sum_cos2=total, readout_vs_readout_sum_cos2=readout_part, cross_terms_sum_cos2=cross,
                residual_vs_residual_sum_cos2=resid_part,
                readout_share_of_overlap=float(readout_part / total) if total > 0 else float("nan"),
                readout_vs_readout_angles_deg=ang_rr.tolist(),
                readout_vs_readout_overlap=float(readout_part / qb.shape[0]),
                residual_rank_a=int(len(ra)), residual_rank_b=int(len(rb)),
                residual_a_vs_basis_b_angles_deg=ang_ra_bb.tolist(),
                residual_b_vs_basis_a_angles_deg=ang_rb_ba.tolist(),
                residual_a_vs_residual_b_angles_deg=ang_rr_res.tolist(),
                residual_min_angle_to_other_basis_deg=(float(min(ang_ra_bb.min() if len(ang_ra_bb) else 90.0,
                                                                 ang_rb_ba.min() if len(ang_rb_ba) else 90.0))))


def principal_angle_block(banks, rng):
    """Learned-vs-random within task, learned-vs-learned across tasks, plus a 1000-draw random null.

    Cross-task pairs are tagged `same_checkpoint`; principal angles between bases fitted on
    different predictors compare unrelated coordinate systems and are reported but not scored.
    For same-checkpoint pairs the learned-vs-learned overlap is decomposed into the part carried
    by the 3-score PCA readout (shared by construction: both fits read the same predictor's
    activation PCs) and the part carried by the genuinely fitted residual direction; the rubric
    scores the residual."""
    d, k = AMBIENT, 4
    analytic = dict(expected_overlap_random=k / d, expected_cos2_per_angle=k / d,
                    expected_angle_deg=float(np.degrees(np.arccos(np.sqrt(k / d)))),
                    note="for independent k-dim subspaces in d dims, E||Qa Qb^T||_F^2 / k = k/d (Vershynin 2018)")
    flat = {t: {arm: banks[t]["bank"]["operators"][arm]["basis"].detach().cpu().double().numpy().reshape(4, -1)
                for arm in (LEARNED, CONTROL)} for t in banks}
    readout = {t: readout_rows(banks[t]) for t in banks}
    resid = {t: residual_rows(flat[t][LEARNED], readout[t]) for t in banks}
    ckpt = {t: checkpoint_of(banks[t]) for t in banks}
    # Is the matched-random control basis shared (identical) across task banks?
    tasks_sorted = sorted(banks)
    identical_control = {}
    for a, b in combinations(tasks_sorted, 2):
        identical_control[f"{a}_vs_{b}"] = bool(np.array_equal(flat[a][CONTROL], flat[b][CONTROL]))
    pairs = {}
    for t in banks:
        ang, ov = principal_angles_deg(flat[t][LEARNED], flat[t][CONTROL])
        pairs[f"{t}:learned_vs_random"] = dict(angles_deg=ang.tolist(), overlap=ov, kind="within_task_learned_vs_control",
                                               same_checkpoint=True)
    for a, b in combinations(tasks_sorted, 2):
        same = ckpt[a] == ckpt[b]
        ang, ov = principal_angles_deg(flat[a][LEARNED], flat[b][LEARNED])
        entry = dict(angles_deg=ang.tolist(), overlap=ov, kind="cross_task_learned", same_checkpoint=same)
        if same:
            entry["readout_decomposition"] = readout_decomposition(flat[a][LEARNED], flat[b][LEARNED], readout[a],
                                                                   readout[b], resid[a][0], resid[b][0])
        pairs[f"{a}_vs_{b}:learned_vs_learned"] = entry
        if not identical_control[f"{a}_vs_{b}"]:
            ang, ov = principal_angles_deg(flat[a][LEARNED], flat[b][CONTROL])
            pairs[f"{a}_vs_{b}:learned_vs_other_random"] = dict(angles_deg=ang.tolist(), overlap=ov,
                                                                kind="cross_task_learned_vs_control", same_checkpoint=same)
            ang, ov = principal_angles_deg(flat[a][CONTROL], flat[b][CONTROL])
            pairs[f"{a}_vs_{b}:random_vs_random"] = dict(angles_deg=ang.tolist(), overlap=ov, kind="cross_task_controls",
                                                         same_checkpoint=same)
    # Monte-Carlo null: random-vs-random pairs and learned-vs-fresh-random for each task.
    null = dict(random_vs_random_overlap=[], random_vs_random_min_angle_deg=[],
                learned_vs_random={t: dict(overlap=[], min_angle_deg=[]) for t in banks},
                unit_vs_learned={t: [] for t in banks})
    for _ in range(RANDOM_DRAWS):
        r1, r2 = random_subspace(rng, d, k), random_subspace(rng, d, k)
        ang, ov = principal_angles_deg(r1, r2)
        null["random_vs_random_overlap"].append(ov)
        null["random_vs_random_min_angle_deg"].append(float(ang[0]))
        for t in banks:
            ang, ov = principal_angles_deg(flat[t][LEARNED], r1)
            null["learned_vs_random"][t]["overlap"].append(ov)
            null["learned_vs_random"][t]["min_angle_deg"].append(float(ang[0]))
            # a single random unit direction (first row of r2) against the learned basis: the
            # reference for the 1-dim fitted residual direction of another task
            null["unit_vs_learned"][t].append(float(np.degrees(np.arccos(min(1.0, np.linalg.norm(flat[t][LEARNED] @ r2[0]))))))
    rr = np.asarray(null["random_vs_random_overlap"])
    rmin = np.asarray(null["random_vs_random_min_angle_deg"])
    unit = {t: np.asarray(v) for t, v in null["unit_vs_learned"].items()}
    summary = dict(draws=RANDOM_DRAWS, seed=SEED,
                   random_vs_random=dict(overlap_mean=float(rr.mean()), overlap_p99=float(np.quantile(rr, .99)),
                                         overlap_max=float(rr.max()), min_angle_mean_deg=float(rmin.mean()),
                                         min_angle_p01_deg=float(np.quantile(rmin, .01)),
                                         min_angle_min_deg=float(rmin.min())),
                   learned_vs_random={t: dict(overlap_mean=float(np.mean(v["overlap"])),
                                              overlap_p99=float(np.quantile(v["overlap"], .99)),
                                              min_angle_mean_deg=float(np.mean(v["min_angle_deg"])),
                                              min_angle_p01_deg=float(np.quantile(v["min_angle_deg"], .01)))
                                      for t, v in null["learned_vs_random"].items()},
                   unit_vs_learned={t: dict(angle_mean_deg=float(u.mean()), angle_p01_deg=float(np.quantile(u, .01)),
                                            angle_min_deg=float(u.min())) for t, u in unit.items()})
    for key, p in pairs.items():
        p["p_overlap_ge_random_null"] = float((rr >= p["overlap"]).mean())
        p["p_min_angle_le_random_null"] = float((rmin <= p["angles_deg"][0]).mean())
        p["n_angles_below_45"] = int(sum(1 for a in p["angles_deg"] if a < RUBRIC["cross_task_shared_angle_deg"]))
        rd = p.get("readout_decomposition")
        if rd is not None:
            a, b = key.split(":")[0].split("_vs_")
            # residual of a against basis of b is judged against random unit vectors vs basis b, and vice versa
            pa = float((unit[b] <= (min(rd["residual_a_vs_basis_b_angles_deg"]) if rd["residual_a_vs_basis_b_angles_deg"] else 90.0)).mean())
            pb = float((unit[a] <= (min(rd["residual_b_vs_basis_a_angles_deg"]) if rd["residual_b_vs_basis_a_angles_deg"] else 90.0)).mean())
            rd["p_residual_min_angle_le_unit_null"] = max(pa, pb)
            rd["residual_verdict"] = residual_verdict(rd)
    return dict(analytic_baseline=analytic, pairs=pairs, monte_carlo=summary, checkpoint_sha256=ckpt,
                identical_control_basis_across_tasks=identical_control,
                note="p-values are the fraction of 1000 random 4-dim pairs in R^102400 whose overlap is >= the observed "
                     "overlap (min angle <= observed); for the fitted residual direction the null is a random unit vector "
                     "against the other task's learned basis. EXPLORATORY, not part of the pre-registered family. Pairs "
                     "with same_checkpoint=false compare bases of different predictors and are not scored.")


def residual_verdict(rd):
    """shared / essentially orthogonal / partial, for the genuinely fitted residual direction(s) of
    two same-checkpoint learned bases (readout span projected out of each)."""
    if rd["residual_rank_a"] == 0 or rd["residual_rank_b"] == 0:
        return "no residual direction (basis = readout span)"
    ang = rd["residual_min_angle_to_other_basis_deg"]
    cos2 = float(np.cos(np.radians(ang)) ** 2)
    if rd["p_residual_min_angle_le_unit_null"] < RUBRIC["cross_task_p"] and ang < RUBRIC["cross_task_shared_angle_deg"]:
        return "shared subspace component"
    if cos2 < RUBRIC["cross_task_orthogonal_overlap"]:
        return "essentially orthogonal"
    return "partial overlap"


def cross_task_verdict(p):
    """Verdict for a same-checkpoint learned-vs-learned pair. Scored on the fitted RESIDUAL direction
    (readout span projected out), because the 3-score PCA readout is shared by construction."""
    if not p.get("same_checkpoint", False):
        return "not comparable (different predictors)"
    rd = p.get("readout_decomposition")
    if rd is None or "residual_verdict" not in rd:
        return "not comparable (no readout decomposition)"
    return rd["residual_verdict"]


# ================================================================ applied coefficients
def candidate_moments(results, tasks):
    """Single streaming pass over the refined arms: pooled second moments of the dose-scaled
    coefficients actually applied to every candidate in every H6 population call."""
    out = {}
    for task in tasks:
        acc = {arm: dict(n=0, s1=np.zeros(4), s2=np.zeros((4, 4)), norm_sum=0.0, norm_sq=0.0, active=0, calls=0)
               for arm in (LEARNED, CONTROL)}
        found = False
        for _, arm, rec in common.iter_records(results, task, arms=(LEARNED, CONTROL)):
            for c in rec.get("calls", []):
                if c.get("horizon") != 6 or c.get("candidates", 0) <= 1:
                    continue
                e = c.get("energy", {})
                if "coefficients" not in e:
                    continue
                found = True
                coef = np.asarray(e["coefficients"], float)
                act = np.asarray(e.get("active", [True] * len(coef)), bool)
                coef = coef[act] if act.size == len(coef) else coef
                a = acc[arm]
                a["n"] += len(coef)
                a["s1"] += coef.sum(0)
                a["s2"] += coef.T @ coef
                nn = np.linalg.norm(coef, axis=1)
                a["norm_sum"] += float(nn.sum())
                a["norm_sq"] += float((nn ** 2).sum())
                a["active"] += int(act.sum())
                a["calls"] += 1
        if not found:
            out[task] = dict(skipped="no H6 population calls with recorded coefficients for the refined arms")
            log.warning("%s: %s", task, out[task]["skipped"])
            continue
        res = {}
        for arm, a in acc.items():
            if a["n"] == 0:
                res[arm] = dict(skipped="no applied coefficients")
                continue
            m = a["s1"] / a["n"]
            second = a["s2"] / a["n"]
            cov = second - np.outer(m, m)
            ev_c = np.sort(np.linalg.eigvalsh(cov))[::-1]
            ev_u = np.sort(np.linalg.eigvalsh(second))[::-1]
            mean_norm = a["norm_sum"] / a["n"]
            res[arm] = dict(
                n_candidates=int(a["n"]), n_calls=int(a["calls"]), mean=m.tolist(),
                mean_direction=(m / np.linalg.norm(m)).tolist() if np.linalg.norm(m) > 0 else None,
                mean_norm=float(mean_norm), norm_std=float(np.sqrt(max(a["norm_sq"] / a["n"] - mean_norm ** 2, 0))),
                resultant_length=float(np.linalg.norm(m) / mean_norm) if mean_norm > 0 else float("nan"),
                covariance=cov.tolist(), covariance_eigenvalues=ev_c.tolist(),
                covariance_eigenfraction=(ev_c / ev_c.sum()).tolist() if ev_c.sum() > 0 else None,
                participation_ratio_centered=participation_ratio(ev_c),
                second_moment_eigenvalues=ev_u.tolist(),
                second_moment_eigenfraction=(ev_u / ev_u.sum()).tolist() if ev_u.sum() > 0 else None,
                participation_ratio_uncentered=participation_ratio(ev_u))
        out[task] = res
    return out


def call_level(ef, tasks):
    """From common.energy_frame: between-call covariance of the per-call mean coefficient, and the
    within-call directional concentration already summarised by common."""
    out = {}
    for task in tasks:
        sub = ef[(ef.task == task)] if len(ef) else ef
        if not len(sub) or "coef_mean" not in sub.columns:
            out[task] = dict(skipped="no refined-arm energy rows")
            continue
        res = {}
        for arm in (LEARNED, CONTROL):
            s = sub[(sub.arm == arm) & sub.coef_mean.notna()]
            if not len(s):
                res[arm] = dict(skipped="no rows")
                continue
            cm = np.stack(s.coef_mean.values)
            cov = np.cov(cm.T) if len(cm) > 1 else np.zeros((4, 4))
            ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
            res[arm] = dict(n_calls=int(len(s)), between_call_participation_ratio=participation_ratio(ev),
                            between_call_eigenvalues=ev.tolist(),
                            within_call_mean_cosine_to_call_mean=float(np.nanmean(s.coef_cosine_to_call_mean)),
                            active_fraction_mean=float(np.nanmean(s.active_fraction)),
                            rounding_loss_mean=float(np.nanmean(s.rounding_loss)),
                            coef_norm_mean=float(np.nanmean(s.coef_norm_mean)))
        out[task] = res
    return out


def scenario_mean_coefficients(ef, task, arm):
    """Per-scenario candidate-weighted mean of the per-call mean coefficient (common.coef_mean).
    Note: common.coef_mean averages over ALL candidates; inactive candidates carry exactly zero
    coefficients on real records, so this agrees with the active-only mean up to the active fraction."""
    if not len(ef) or "coef_mean" not in ef.columns:
        return {}
    s = ef[(ef.task == task) & (ef.arm == arm) & ef.coef_mean.notna()]
    if not len(s):
        return {}
    res = {}
    for ep, g in s.groupby("episode"):
        w = g.candidates.values.astype(float)
        cm = np.stack(g.coef_mean.values)
        res[int(ep)] = (cm * w[:, None]).sum(0) / w.sum()
    return res


def rescued_vs_regressed(df, ef, tasks, rng):
    """EXPLORATORY: does the mean applied coefficient direction differ between scenarios the arm
    rescued (arm success, native failure) and those it regressed (native success, arm failure)?
    Statistic: cosine between the two group-mean vectors; permutation of group labels."""
    out, pvals, keys = {}, [], []
    for task in tasks:
        for arm in (LEARNED, CONTROL):
            key = f"{task}:{arm}"
            if not len(df) or task not in set(df.task):
                out[key] = dict(skipped="no episodes for task")
                continue
            m = common.success_matrix(df, task, arms=("native", arm))
            if arm not in m.columns or "native" not in m.columns:
                out[key] = dict(skipped="arm or native missing")
                continue
            if not len(ef) or "coef_mean" not in ef.columns:
                out[key] = dict(skipped="only 0 scenarios with coefficients (no refined-arm record carries energy.coefficients)")
                continue
            vec = scenario_mean_coefficients(ef, task, arm)
            eps = [e for e in m.index if e in vec]
            if len(eps) < 4:
                out[key] = dict(skipped=f"only {len(eps)} scenarios with coefficients")
                continue
            a, n = m.loc[eps, arm].values, m.loc[eps, "native"].values
            lab = np.where(a & ~n, 1, np.where(~a & n, -1, 0))
            v = np.stack([vec[e] for e in eps])
            n_res, n_reg = int((lab == 1).sum()), int((lab == -1).sum())
            if n_res < 2 or n_reg < 2:
                out[key] = dict(skipped=f"too few discordant scenarios (rescued={n_res}, regressed={n_reg})",
                                rescued=n_res, regressed=n_reg)
                continue
            sel = lab != 0
            vs, ls = v[sel], lab[sel]
            obs = cosine(vs[ls == 1].mean(0), vs[ls == -1].mean(0))
            perm = np.empty(PERMUTATIONS)
            for i in range(PERMUTATIONS):
                p = rng.permutation(ls)
                perm[i] = cosine(vs[p == 1].mean(0), vs[p == -1].mean(0))
            pv = float((1 + int((perm <= obs).sum())) / (PERMUTATIONS + 1))  # +1 pseudo-count (Phipson & Smyth 2010)
            unchanged = v[lab == 0].mean(0) if (lab == 0).any() else None
            out[key] = dict(rescued=n_res, regressed=n_reg, unchanged=int((lab == 0).sum()),
                            cosine_rescued_vs_regressed=obs, permutation_p=pv, permutations=PERMUTATIONS,
                            p_note="p = (1 + #{perm <= obs}) / (permutations + 1); lower tail (a smaller cosine than "
                                   "chance means the two groups were pushed in different directions)",
                            null_cosine_mean=float(perm.mean()), null_cosine_p05=float(np.quantile(perm, .05)),
                            rescued_mean_direction=(vs[ls == 1].mean(0) / np.linalg.norm(vs[ls == 1].mean(0))).tolist(),
                            regressed_mean_direction=(vs[ls == -1].mean(0) / np.linalg.norm(vs[ls == -1].mean(0))).tolist(),
                            cosine_rescued_vs_unchanged=cosine(vs[ls == 1].mean(0), unchanged) if unchanged is not None else None,
                            cosine_regressed_vs_unchanged=cosine(vs[ls == -1].mean(0), unchanged) if unchanged is not None else None)
            pvals.append(pv)
            keys.append(key)
    if pvals:
        adj = common.holm(pvals)
        for k, p in zip(keys, adj):
            out[k]["holm_p"] = float(p)
    return dict(tests=out, family=len(pvals), correction="Holm over all (task, arm) tests with data",
                exploratory=True, seed=SEED)


# ================================================================================ rubric
def apply_rubric(banks_diag, cmap, angles, applied, bank_tasks, result_tasks):
    R = RUBRIC
    crit = {}
    # spatial / feature support (subspace-mean, learned arm), per task with a bank
    for t in bank_tasks:
        s = banks_diag[t][LEARNED]["subspace_mean"]["spatial"]["effective_count"]
        f = banks_diag[t][LEARNED]["subspace_mean"]["feature"]["effective_count"]
        sr = banks_diag[t][CONTROL]["subspace_mean"]["spatial"]["effective_count"]
        crit[f"spatial:{t}"] = dict(value=s, random_control=sr,
                                    verdict="concentrated" if s < R["spatial_concentrated_eff_patches"] else
                                    "broad" if s > R["spatial_broad_eff_patches"] else "intermediate")
        crit[f"feature:{t}"] = dict(value=f, verdict="concentrated" if f < R["feature_concentrated_eff"] else
                                    "broad" if f > R["feature_broad_eff"] else "intermediate")
        share = cmap[t][LEARNED]["intercept_share"]
        crit[f"map:{t}"] = dict(value=share, verdict="constant-dominated" if share >= R["intercept_constant_share"] else
                                "state-dominated" if share <= R["intercept_state_share"] else "mixed")
    for t in result_tasks:
        a = applied.get(t, {})
        if isinstance(a, dict) and LEARNED in a and "participation_ratio_uncentered" in a[LEARNED]:
            pr = a[LEARNED]["participation_ratio_uncentered"]
            crit[f"applied_pr:{t}"] = dict(value=pr, random_control=a.get(CONTROL, {}).get("participation_ratio_uncentered"),
                                           verdict="single fixed direction" if pr <= R["pr_single_direction"] else
                                           "full rank-4 in use" if pr >= R["pr_full_rank"] else "intermediate")
    for key, p in angles.get("pairs", {}).items():
        if p["kind"] == "cross_task_learned":
            rd = p.get("readout_decomposition") or {}
            crit[f"cross_task:{key.split(':')[0]}"] = dict(
                value=p["overlap"], min_angle_deg=p["angles_deg"][0], p=p["p_overlap_ge_random_null"],
                same_checkpoint=p.get("same_checkpoint", False),
                readout_share_of_overlap=rd.get("readout_share_of_overlap"),
                residual_min_angle_to_other_basis_deg=rd.get("residual_min_angle_to_other_basis_deg"),
                residual_p=rd.get("p_residual_min_angle_le_unit_null"),
                scored_on="fitted residual direction (readout span projected out)", verdict=cross_task_verdict(p))
    # Each axis awards one point to the side holding a strict majority (> n/2) of its per-task verdicts.
    COMPACT_V = {"applied_pr": "single fixed direction", "map": "constant-dominated", "spatial": "concentrated",
                 "cross_task": "shared subspace component"}
    DISTRIB_V = {"applied_pr": "full rank-4 in use", "map": "state-dominated", "spatial": "broad",
                 "cross_task": "essentially orthogonal"}
    axes = {}
    for ax in COMPACT_V:
        vs = [v["verdict"] for k, v in crit.items() if k.startswith(ax + ":")
              and not v["verdict"].startswith("not comparable")]
        if not vs:
            continue
        nc, nd = vs.count(COMPACT_V[ax]), vs.count(DISTRIB_V[ax])
        axes[ax] = dict(n=len(vs), compact=nc, distributed=nd,
                        point="compact" if nc > len(vs) / 2 else "distributed" if nd > len(vs) / 2 else "none")
    compact = sum(1 for a in axes.values() if a["point"] == "compact")
    distributed = sum(1 for a in axes.values() if a["point"] == "distributed")
    n_scored = len(axes)
    if n_scored == 0:
        verdict = "not assessable (no bank and no applied coefficients)"
    elif compact >= 3 and distributed <= 1:
        verdict = "compact / model-native-like"
    elif distributed >= 3 and compact <= 1:
        verdict = "distributed / task-specific"
    else:
        verdict = "mixed: neither picture cleanly"
    return dict(criteria=crit, axes=axes, compact_points=int(compact), distributed_points=int(distributed),
                axes_scored=int(n_scored), verdict=verdict, thresholds=R,
                scoring="each axis gives one point to a side only if that side holds a strict majority (more than half) "
                        "of the axis's per-task verdicts; intermediate verdicts count against both sides, so an axis "
                        "with no strict majority gives no point")


# =============================================================================== figures
def fig_spatial(banks_diag, bank_tasks, out):
    if not bank_tasks:
        return None
    rows = len(bank_tasks)
    cols = 6
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.25, rows * 1.2 + 0.3), squeeze=False)
    panels = []
    for t in bank_tasks:
        pl = np.asarray(banks_diag[t][LEARNED]["patch_loadings"])
        panels.append([pl[d] for d in range(4)] + [np.asarray(banks_diag[t][LEARNED]["subspace_mean_patch_loading"]),
                                                   np.asarray(banks_diag[t][CONTROL]["subspace_mean_patch_loading"])])
    vmax = max(float(np.max(p)) for row in panels for p in row)
    labels = ["dir 1", "dir 2", "dir 3", "dir 4", "subspace mean", "random ctrl mean"]
    im = None
    for i, t in enumerate(bank_tasks):
        for j in range(cols):
            ax = axes[i, j]
            im = ax.imshow(panels[i][j].reshape(GRID, GRID), vmin=0, vmax=vmax, cmap="magma", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(labels[j], fontsize=7, pad=2)
            if j == 0:
                ax.set_ylabel(common.TASK_LABEL.get(t, t), fontsize=7)
    fig.subplots_adjust(left=0.06, right=0.9, top=0.93, bottom=0.03, wspace=0.08, hspace=0.08)
    cax = fig.add_axes([0.915, 0.15, 0.012, 0.7])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("squared loading per patch", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_visible(False)
    path = out / f"{SCRIPT}_spatial_loadings.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def fig_angles(angles, out):
    pairs = angles.get("pairs", {})
    if not pairs:
        return None
    order = [k for k in pairs if pairs[k]["kind"] == "within_task_learned_vs_control"] + \
            [k for k in pairs if pairs[k]["kind"] == "cross_task_learned" and pairs[k].get("same_checkpoint")]
    if not order:
        return None
    fig, ax = plt.subplots(figsize=(max(3.6, 1.3 * len(order) + 1.2), 2.4))
    w = 0.18
    colors = ["#1F5FA8", "#4C86C6", "#8FB3DD", "#C9DBEE"]
    for i, k in enumerate(order):
        a = pairs[k]["angles_deg"]
        for j, v in enumerate(a):
            ax.bar(i + (j - 1.5) * w, v, width=w, color=colors[j], edgecolor="none")
    mc = angles["monte_carlo"]["random_vs_random"]
    ax.axhline(mc["min_angle_mean_deg"], color="#555555", lw=0.8, ls="--")
    ax.text(-0.45, mc["min_angle_mean_deg"] + 1.2,
            f"random-subspace null: smallest angle {mc['min_angle_mean_deg']:.1f}° (1st pct {mc['min_angle_p01_deg']:.1f}°)",
            ha="left", va="bottom", fontsize=6, color="#555555")
    ax.set_xlabel("four principal angles per pair, ascending; 'vs ctrl' = learned vs matched-random control basis, "
                  "'a vs b' = learned vs learned (same checkpoint)", fontsize=6.5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([k.replace(":learned_vs_random", "\nvs ctrl")
                        .replace(":learned_vs_learned", "\nlearned vs learned").replace("_vs_", " vs ") for k in order],
                       fontsize=6.5)
    ax.set_ylabel("principal angle (deg)")
    ax.set_ylim(0, 95)
    ax.set_yticks([0, 30, 60, 90])
    fig.tight_layout()
    path = out / f"{SCRIPT}_principal_angles.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def fig_spectrum(applied, result_tasks, out):
    tasks = [t for t in result_tasks if isinstance(applied.get(t), dict) and LEARNED in applied[t]
             and "second_moment_eigenfraction" in applied[t][LEARNED]]
    if not tasks:
        return None
    fig, axes = plt.subplots(1, len(tasks), figsize=(2.1 * len(tasks) + 0.6, 2.3), squeeze=False, sharey=True)
    for ax, t in zip(axes[0], tasks):
        for arm, color, lab in ((LEARNED, "#1A8F7A", "learned"), (CONTROL, "#9E9E9E", "random ctrl")):
            a = applied[t].get(arm, {})
            if "second_moment_eigenfraction" not in a:
                continue
            ev = a["second_moment_eigenfraction"]
            ax.plot(range(1, 5), ev, marker="o", ms=3, lw=1, color=color,
                    label=f"{lab}, 2nd moment (PR {a['participation_ratio_uncentered']:.2f})")
            if a.get("covariance_eigenfraction"):
                ax.plot(range(1, 5), a["covariance_eigenfraction"], marker="s", ms=2.5, lw=0.9, ls="--", color=color,
                        label=f"{lab}, covariance (PR {a['participation_ratio_centered']:.2f})")
        ax.set_title(common.TASK_LABEL.get(t, t), pad=2)
        ax.set_xticks(range(1, 5))
        ax.set_xlabel("eigen-component")
        ax.set_ylim(0, 1.02)
        ax.legend(loc="upper right", fontsize=5.5)
    axes[0][0].set_ylabel("eigenvalue fraction")
    fig.text(0.5, 0.005, "solid: second-moment matrix of the applied 4-vectors; dashed: their covariance (centered)",
             ha="center", va="bottom", fontsize=6, color="#555555")
    fig.tight_layout()
    path = out / f"{SCRIPT}_coefficient_spectrum.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


# ============================================================================== markdown
def write_md(path, J):
    L = []
    prov = J["provenance"]
    reg = J["behavioural_regime"]
    R = J["rubric"]
    bank_tasks, result_tasks = J["bank_tasks"], J["result_tasks"]
    L.append("# Representation geometry of the rank-4 intervention subspace\n")
    L.append("**Status: EXPLORATORY / descriptive.** Only the arm-vs-native success contrasts quoted from the frozen "
             "report are pre-registered; every geometric number, every learned-vs-control contrast and every "
             "permutation test below is post hoc. Corrections: learned-vs-control paired bootstrap is Bonferroni over "
             f"{reg['exploratory_control_family']} tasks; rescued-vs-regressed permutation tests are Holm over "
             f"{J['rescued_vs_regressed']['family']} (task, arm) tests; principal-angle p-values are against a "
             f"{RANDOM_DRAWS}-draw random-subspace null and are uncorrected.\n")
    if prov["synthetic"]:
        L.append("> **SYNTHETIC FIXTURE.** `provenance.synthetic` is true: the results tree was generated by "
                 "`make_fixture.py`. Its coefficient records are i.i.d. Gaussian noise, so every number derived from "
                 "the fresh results (sections 4-5, the applied-coefficient rubric axes, the behavioural regime) is "
                 "meaningless and only exercises the code path. The bank-derived numbers (sections 1-3) come from the "
                 "real fitted banks on disk and are real, but their reading against the fixture regime is not.\n")
    L.append(f"Provenance: results `{prov['results']}`, freeze sha256 `{prov['freeze_sha256'][:12]}…`, analysis sha256 "
             f"`{prov['analysis_sha256'][:12]}…`, method `{prov['method']}`. Seed {SEED}.\n")
    L.append("Banks: " + "; ".join(f"{t}: `{J['banks'][t]['path']}` (sha256 `{J['banks'][t]['sha256'][:12]}…`)"
                                   for t in bank_tasks) + ("." if bank_tasks else "none found."))
    for t, why in J["skipped_banks"].items():
        L.append(f"Bank skipped for {t}: {why}.")
    L.append("")

    # ---- regime
    L.append("## 0. Behavioural regime the geometry is read against\n")
    L.append(f"Refined rank-4 edit (`{LEARNED}`), overall reading: **{reg['overall']}**.\n")
    rows = []
    for t, v in reg["per_task"].items():
        n, c = v["vs_native"], v["vs_control"]
        missing = bool(c.get("missing"))
        rows.append(dict(task=t, vs_native_pp=n["difference"], native_interval=f"[{n['lower']:.1f}, {n['upper']:.1f}]",
                         vs_control_pp=("n/a" if missing else c.get("difference")),
                         control_interval=("n/a" if missing else f"[{c['lower']:.1f}, {c['upper']:.1f}]"),
                         classification=v["classification"]))
    if rows:
        L.append(common.md_table(pd.DataFrame(rows)))
    L.append("\n`vs_native` is pre-registered (Bonferroni-48 simultaneous 95%); `vs_control` is exploratory "
             f"(Bonferroni over {reg['exploratory_control_family']} tasks). Classification uses `common.classify_regime` "
             "with the 5 pp useful-gain rule.\n")
    if reg["overall"] == "positive":
        L.append(f"A confirmed success gain exists on {', '.join(reg['confirmed'])}; the geometry below describes the "
                 "operator that produced it, which raises the stakes of the compact-vs-distributed question.")
    elif reg["overall"] == "negative":
        L.append(f"The refined edit is harmful on {', '.join(reg['harmful'])} and positive nowhere; the geometry below "
                 "describes an operator that reliably degraded success, so any compactness found is compactness of a "
                 "harmful direction, not of a useful one.")
    elif reg["overall"] == "mixed":
        parts = []
        cls = {t: v["classification"] for t, v in reg["per_task"].items()}
        pn = [t for t, c in cls.items() if c == "positive_vs_native_only"]
        pc = [t for t, c in cls.items() if c == "positive_vs_control_only"]
        if reg["confirmed"]:
            parts.append(f"confirmed on {', '.join(reg['confirmed'])}")
        if pn:
            parts.append(f"positive against the unsteered model but not its random control on {', '.join(pn)}")
        if pc:
            parts.append(f"positive against its random control but not the unsteered model on {', '.join(pc)}")
        if reg["harmful"]:
            parts.append(f"harmful on {', '.join(reg['harmful'])}")
        L.append(f"The behavioural reading is mixed ({'; '.join(parts)}): the useful-effect criterion carried over from "
                 "the development panels (both references, >= 5 pp; only the vs-native half is pre-registered for this "
                 "confirmation, the vs-control half is the exploratory bootstrap above) is not met"
                 + (" anywhere" if not reg["confirmed"] else " on every task") +
                 ". Because each task has its own fitted basis, the geometry is compared across tasks below to ask "
                 "whether the same construction produced geometrically similar or different objects where its "
                 "outcomes differed.")
    else:
        L.append("No success effect is resolved on any task, so the geometry below describes what was applied, not "
                 "what it achieved; the compact-vs-distributed reading is about the fitted object, and no claim that "
                 "either geometry is decision-relevant follows from this panel.")
    lean = reg["point_estimate_lean"]
    if lean:
        L.append(" Point-estimate signs (descriptive only, not inferential): " +
                 ", ".join(f"{t} {s}" for t, s in lean.items()) + ".\n")

    # ---- 1 basis diagnostics
    L.append("## 1. Basis diagnostics (per task, learned vs random-subspace control)\n")
    if not bank_tasks:
        L.append("No operator bank was found for any task; this section is empty.\n")
    else:
        rows = []

        def _row(task, arm, direction, s):
            return dict(task=task, arm=arm, direction=direction,
                        spatial_H_bits=s["spatial"]["entropy_bits"], spatial_eff_patches=s["spatial"]["effective_count"],
                        spatial_gini=s["spatial"]["gini"], spatial_top16=s["spatial"]["top16_mass"],
                        feature_H_bits=s["feature"]["entropy_bits"], feature_eff=s["feature"]["effective_count"],
                        feature_gini=s["feature"]["gini"], feature_top16=s["feature"]["top16_mass"])
        for t in bank_tasks:
            d = J["banks"][t]["diagnostics"]
            for i, dd in enumerate(d[LEARNED]["directions"]):
                rows.append(_row(t, "learned", i + 1, dd))
            rows.append(_row(t, "learned", "mean", d[LEARNED]["subspace_mean"]))
            rows.append(_row(t, "random", "mean", d[CONTROL]["subspace_mean"]))
        L.append(common.md_table(pd.DataFrame(rows)))
        ident = J["principal_angles"].get("identical_control_basis_across_tasks", {})
        if ident and all(ident.values()):
            L.append("\nThe matched-random control basis is byte-identical across all task banks (one shared random "
                     "subspace, calibrated per task through its own map A); its rows above therefore repeat, and "
                     "cross-task control-vs-control angles are degenerate and omitted.")
        elif ident and any(ident.values()):
            L.append("\nThe matched-random control basis is identical for some task pairs: " +
                     ", ".join(k for k, v in ident.items() if v) + ".")
        L.append(f"\nMaxima: spatial entropy log2(256) = 8.000 bits (uniform top-16 mass 0.0625); feature entropy "
                 f"log2(400) = {np.log2(400):.3f} bits (uniform top-16 mass 0.040). Effective count = 2^H.\n")
        for t in bank_tasks:
            d = J["banks"][t]["diagnostics"]
            le, ra = d[LEARNED], d[CONTROL]
            L.append(f"- {common.TASK_LABEL.get(t, t)}: orthonormality error {le['orthonormality_max_abs_error']:.1e} "
                     f"(learned) / {ra['orthonormality_max_abs_error']:.1e} (random). Subspace-mean spatial support: "
                     f"{le['subspace_mean']['spatial']['effective_count']:.0f} effective patches of 256 (learned) vs "
                     f"{ra['subspace_mean']['spatial']['effective_count']:.0f} (random); top-16 mass "
                     f"{le['subspace_mean']['spatial']['top16_mass']:.3f} vs {ra['subspace_mean']['spatial']['top16_mass']:.3f}. "
                     f"Feature support: {le['subspace_mean']['feature']['effective_count']:.0f} effective features of 400 "
                     f"(learned) vs {ra['subspace_mean']['feature']['effective_count']:.0f} (random).")
            sv = R["criteria"].get(f"spatial:{t}", {}).get("verdict")
            fv = R["criteria"].get(f"feature:{t}", {}).get("verdict")
            per = [dd["spatial"]["effective_count"] for dd in le["directions"]]
            L.append(f"  Reading: the learned subspace is spatially **{sv}** (rubric: concentrated < "
                     f"{RUBRIC['spatial_concentrated_eff_patches']:.0f}, broad > {RUBRIC['spatial_broad_eff_patches']:.0f} effective "
                     f"patches) and **{fv}** over features; per-direction effective patch counts range "
                     f"{min(per):.0f}-{max(per):.0f}, so " +
                     ("the four directions have similar support." if max(per) - min(per) < 32 else
                      "the directions differ substantially in how localised they are."))
            ra_info = J["banks"][t]["readout_alignment"]
            L.append(f"  Readout alignment: principal angles between the rank-4 edit basis and the 3-score readout "
                     f"projection are {', '.join(deg(a) for a in ra_info['angles_deg'])}; "
                     + ("the readout scores live inside the edit subspace (the edit writes along the axes it reads)."
                        if max(ra_info["angles_deg"]) < 5 else
                        "the readout scores are partly outside the edit subspace." if max(ra_info["angles_deg"]) < 60 else
                        "the readout and edit subspaces are close to orthogonal."))
        L.append("")

    # ---- 2 principal angles
    L.append("## 2. Principal angles: learned vs random, and across tasks\n")
    A = J["principal_angles"]
    if not A.get("pairs"):
        L.append("Fewer than one bank available; no subspace comparison possible.\n")
    else:
        an = A["analytic_baseline"]
        mc = A["monte_carlo"]
        L.append(f"Analytic random baseline for two independent 4-dim subspaces of R^{AMBIENT}: expected overlap "
                 f"k/d = {an['expected_overlap_random']:.2e}, expected per-angle cos^2 = {an['expected_cos2_per_angle']:.2e} "
                 f"(angle {an['expected_angle_deg']:.2f}°). Monte-Carlo ({mc['draws']} random pairs): mean overlap "
                 f"{mc['random_vs_random']['overlap_mean']:.2e} (99th pct {mc['random_vs_random']['overlap_p99']:.2e}), "
                 f"mean smallest angle {mc['random_vs_random']['min_angle_mean_deg']:.2f}° "
                 f"(1st pct {mc['random_vs_random']['min_angle_p01_deg']:.2f}°).\n")
        ck = A.get("checkpoint_sha256", {})
        groups = {}
        for t, c in ck.items():
            groups.setdefault(c, []).append(t)
        L.append("Checkpoints: " + "; ".join(f"`{c[:12]}…` -> {', '.join(ts)}" for c, ts in groups.items()) +
                 ". Only pairs fitted on the same checkpoint are scored; bases of different predictors live in unrelated "
                 "coordinate systems.\n")
        rows = []
        for k, p in A["pairs"].items():
            rows.append(dict(pair=k, kind=p["kind"], same_ckpt=p.get("same_checkpoint"),
                             angles_deg=", ".join(f"{a:.1f}" for a in p["angles_deg"]),
                             overlap=p["overlap"], p_vs_random_null=p["p_overlap_ge_random_null"]))
        L.append(common.md_table(pd.DataFrame(rows), floatfmt="{:.3g}"))
        L.append("")
        diff_ck = [(k, p) for k, p in A["pairs"].items() if p["kind"] != "within_task_learned_vs_control"
                   and not p.get("same_checkpoint")]
        if diff_ck:
            above = [(k, p) for k, p in diff_ck if p["p_overlap_ge_random_null"] < RUBRIC["cross_task_p"]]
            ovs = [p["overlap"] for _, p in diff_ck]
            kd = an["expected_overlap_random"]
            if above:
                L.append(f"Different-checkpoint pairs ({len(diff_ck)}): overlaps {min(ovs):.1e}-{max(ovs):.1e}, i.e. "
                         f"{min(ovs) / kd:.0f}-{max(ovs) / kd:.0f}x the k/d baseline, and {len(above)} of {len(diff_ck)} are "
                         f"above the random-subspace null at p < {RUBRIC['cross_task_p']}. This is not interpreted: bases "
                         "fitted on different predictors share the architecture's ambient coordinates (patch grid, feature "
                         "channels), so an overlap above the isotropic-random null says only that neither basis is "
                         "isotropic in those coordinates, not that the two fits found related directions.\n")
            else:
                L.append(f"Different-checkpoint pairs ({len(diff_ck)}): overlaps {min(ovs):.1e}-{max(ovs):.1e}, none above "
                         f"the random-subspace null at p < {RUBRIC['cross_task_p']}; consistent with unrelated coordinate "
                         "systems and not interpreted further.\n")
        for k, p in A["pairs"].items():
            t = k.split(":")[0]
            if p["kind"] == "within_task_learned_vs_control":
                L.append(f"- {t}: learned vs its matched-random basis, smallest angle {deg(p['angles_deg'][0])} "
                         f"(p = {p['p_overlap_ge_random_null']:.3f} vs null): " +
                         ("as expected for an independent random control." if p["p_overlap_ge_random_null"] >= 0.05 else
                          "the random control is NOT independent of the learned subspace, which would compromise it as a control."))
            elif p["kind"] == "cross_task_learned":
                v = R["criteria"].get(f"cross_task:{t}", {})
                verdict = v.get("verdict", "n/a")
                s = (f"- {t.replace('_vs_', ' vs ')}: learned vs learned, angles "
                     f"{', '.join(deg(a) for a in p['angles_deg'])}, overlap {p['overlap']:.3g} "
                     f"(random-null p = {p['p_overlap_ge_random_null']:.3f}); {p['n_angles_below_45']} of 4 angles below 45°. ")
                rd = p.get("readout_decomposition")
                if rd is None:
                    s += "Different predictors: not scored."
                    L.append(s)
                    continue
                a_t, b_t = t.split("_vs_")
                s += (f"Decomposition of the overlap (sum of cos^2 = {rd['total_sum_cos2']:.3f} of 4): "
                      f"{rd['readout_vs_readout_sum_cos2']:.3f} ({rd['readout_share_of_overlap']:.0%}) is the 3-score PCA "
                      f"readout that both fits read from the same predictor (readout-vs-readout angles "
                      f"{', '.join(deg(x) for x in rd['readout_vs_readout_angles_deg'])}), which is shared by construction, "
                      f"not found; {rd['cross_terms_sum_cos2']:.3f} is cross terms; {rd['residual_vs_residual_sum_cos2']:.3f} "
                      f"is residual-vs-residual. ")
                if rd["residual_rank_a"] and rd["residual_rank_b"]:
                    s += (f"The genuinely fitted residual direction (readout span projected out; residual rank "
                          f"{rd['residual_rank_a']}/{rd['residual_rank_b']}) is at "
                          f"{', '.join(deg(x) for x in rd['residual_a_vs_basis_b_angles_deg'])} ({a_t} -> {b_t} basis) and "
                          f"{', '.join(deg(x) for x in rd['residual_b_vs_basis_a_angles_deg'])} ({b_t} -> {a_t} basis); "
                          f"residual-vs-residual {', '.join(deg(x) for x in rd['residual_a_vs_residual_b_angles_deg'])}; "
                          f"random-unit-vector null p = {rd['p_residual_min_angle_le_unit_null']:.3f}. ")
                s += f"Verdict (scored on the residual): **{verdict}**. "
                if verdict == "shared subspace component":
                    s += ("Beyond the shared readout, the fitted residual direction of one task lies close to the other "
                          "task's subspace: within this checkpoint the construction found a reusable non-readout "
                          "component, which reflects shared activation statistics of one model, not transfer between models.")
                elif verdict == "essentially orthogonal":
                    s += ("Beyond the shared readout the two fits have nothing in common: the residual direction of each "
                          "task is essentially orthogonal to the other task's subspace, so what the fit added to the "
                          "readout is task-specific.")
                elif verdict == "partial overlap":
                    s += ("Beyond the shared readout, the fitted residual direction is neither inside nor orthogonal to the "
                          "other task's subspace: the part of the fit that is not the common PCA readout is mostly, but "
                          "not entirely, task-specific. The full-basis overlap above therefore overstates reuse: most of "
                          "it is the readout both fits inherit from the same predictor.")
                else:
                    s += "Not scored."
                L.append(s)
        L.append("")

    # ---- 3 coefficient map
    L.append("## 3. Coefficient map A (4 x 4): constant vs state-dependent edit\n")
    if not bank_tasks:
        L.append("No bank; skipped.\n")
    else:
        rows = []
        for t in bank_tasks:
            for arm in (LEARNED, CONTROL):
                c = J["banks"][t]["coefficient_map"][arm]
                rows.append(dict(task=t, arm=("learned" if arm == LEARNED else "random"),
                                 singular_values=", ".join(f"{s:.3g}" for s in c["singular_values"]),
                                 condition_number=c["condition_number"], intercept_norm=c["intercept_column_norm"],
                                 score_cols_frob=c["score_columns_frobenius"], intercept_share=c["intercept_share"]))
        L.append(common.md_table(pd.DataFrame(rows), floatfmt="{:.3g}"))
        L.append("")
        for t in bank_tasks:
            c = J["banks"][t]["coefficient_map"][LEARNED]
            v = R["criteria"].get(f"map:{t}", {}).get("verdict")
            L.append(f"- {common.TASK_LABEL.get(t, t)}: intercept share {c['intercept_share']:.3f} -> **{v}** (rubric: "
                     f">= {RUBRIC['intercept_constant_share']} constant-dominated, <= {RUBRIC['intercept_state_share']} "
                     f"state-dominated). Condition number {c['condition_number']:.3g}: " +
                     ("the map is well conditioned; all four output directions receive comparable weight."
                      if c["condition_number"] < 10 else
                      "the map is moderately ill-conditioned; one or two output directions dominate."
                      if c["condition_number"] < 100 else
                      "the map is strongly ill-conditioned; the edit is effectively lower-rank than 4 before rescaling.")
                     + " Because the applied edit is rescaled to a fixed dose, the intercept share governs the "
                       "*direction* of the applied edit: a constant-dominated map yields a nearly state-independent direction.")
        L.append("")

    # ---- 4 applied coefficients
    L.append("## 4. Applied coefficients in the fresh confirmation (refined arms) - EXPLORATORY\n")
    if prov["synthetic"]:
        L.append("(Synthetic fixture: these are statistics of i.i.d. noise and carry no information about the model.)\n")
    ap, cl = J["applied_coefficients"], J["call_level"]
    rows = []
    for t in result_tasks:
        a = ap.get(t, {})
        if "skipped" in a:
            L.append(f"- {t}: skipped ({a['skipped']}).")
            continue
        for arm in (LEARNED, CONTROL):
            x = a.get(arm, {})
            if "skipped" in x or not x:
                continue
            c = cl.get(t, {}).get(arm, {})
            rows.append(dict(task=t, arm=("learned" if arm == LEARNED else "random"), n_candidates=x["n_candidates"],
                             n_calls=x["n_calls"], mean_norm=x["mean_norm"], resultant=x["resultant_length"],
                             PR_uncentered=x["participation_ratio_uncentered"], PR_centered=x["participation_ratio_centered"],
                             top_eigenfrac=(x["second_moment_eigenfraction"] or [float("nan")])[0],
                             between_call_PR=c.get("between_call_participation_ratio"),
                             within_call_cos=c.get("within_call_mean_cosine_to_call_mean"),
                             active_frac=c.get("active_fraction_mean")))
    if rows:
        L.append(common.md_table(pd.DataFrame(rows), floatfmt="{:.3g}"))
    L.append("\nPR = participation ratio (tr M)^2 / tr(M^2); uncentered uses the second-moment matrix of the applied "
             "4-vectors (how many directions carry the delivered edit energy), centered uses their covariance (how many "
             "directions the edit *varies* along). `resultant` = |mean vector| / mean norm (1 = every candidate got the "
             "same direction). `between_call_PR` and `within_call_cos` come from `common.energy_frame`. The candidate-level "
             "moments above drop inactive candidates; the per-call/per-scenario means from `common.energy_frame` average "
             "over all candidates, and the two agree up to the active fraction because inactive candidates carry exactly "
             "zero coefficients on real records.\n")
    for t in result_tasks:
        a = ap.get(t, {})
        x = a.get(LEARNED, {}) if isinstance(a, dict) else {}
        if not x or "skipped" in x:
            continue
        v = R["criteria"].get(f"applied_pr:{t}", {}).get("verdict", "n/a")
        y = a.get(CONTROL, {})
        L.append(f"- {common.TASK_LABEL.get(t, t)}: PR(uncentered) {x['participation_ratio_uncentered']:.2f}, "
                 f"resultant {x['resultant_length']:.2f} -> **{v}** (rubric: <= {RUBRIC['pr_single_direction']} single "
                 f"fixed direction, >= {RUBRIC['pr_full_rank']} full rank-4). " +
                 (f"Random control: PR {y['participation_ratio_uncentered']:.2f}, resultant {y['resultant_length']:.2f}; "
                  + ("the learned edit is more concentrated than its control." if
                     x["participation_ratio_uncentered"] < y["participation_ratio_uncentered"] - 0.25 else
                     "the learned edit is less concentrated than its control." if
                     x["participation_ratio_uncentered"] > y["participation_ratio_uncentered"] + 0.25 else
                     "learned and control have similar effective dimensionality, consistent with the dimensionality being "
                     "set by the calibration procedure rather than by the learned directions.")
                  if y and "skipped" not in y else "no control coefficients."))
        if x["norm_std"] < 1e-3 * max(x["mean_norm"], 1e-12):
            L.append(f"  Applied norms are constant ({x['mean_norm']:.3f}), consistent with the fixed-dose rescaling"
                     + (f" (bank dose {J['banks'][t]['dose']:.3f})." if t in J["banks"] else "."))
        else:
            L.append(f"  Applied norms vary (mean {x['mean_norm']:.3f}, sd {x['norm_std']:.3f}); real records rescale every "
                     "active candidate's edit to the frozen dose (B is orthonormal, so |coefficients| = dose), so "
                     "variation here means the source did not apply that rescaling (a synthetic fixture) or the "
                     "operator differs from the one on disk.")
    ct = J["cross_task_mean_direction"]
    if ct:
        L.append("\nCross-task cosine of the mean applied direction (coefficient coordinates are per-task bases, so only "
                 "the ambient, basis-mapped cosine between fits on the same checkpoint is interpretable):")
        for k, v in ct.items():
            s = f"- {k}: coefficient-coordinate cosine {fmt(v['cosine_coefficient_coordinates'])}"
            if v.get("cosine_ambient") is not None:
                s += f"; ambient (R^{AMBIENT}, same checkpoint) cosine {fmt(v['cosine_ambient'])}"
                s += (" - the two tasks push the field the same way." if v["cosine_ambient"] > 0.7 else
                      " - the two tasks push the field in unrelated directions." if abs(v["cosine_ambient"]) < 0.3 else
                      " - partially aligned.")
            else:
                s += " (" + v.get("why_no_ambient", "no ambient comparison") + ")"
            L.append(s)
    cd = J["constant_direction_check"]
    if cd:
        L.append("\nConsistency of the applied mean direction with the map's intercept column A[:,0]:")
        for t, v in cd.items():
            L.append(f"- {t}: cosine {fmt(v['cosine'])}" +
                     (" - the delivered edits are, on average, the constant part of the map." if v["cosine"] > 0.9 else
                      " - state-dependent terms rotate the delivered edit away from the constant direction."
                      if v["cosine"] < 0.5 else " - mostly constant with a state-dependent component."))
    L.append("")

    # ---- 5 rescued vs regressed
    L.append("## 5. Rescued vs regressed scenarios: does the mean applied direction differ? - EXPLORATORY\n")
    rr = J["rescued_vs_regressed"]
    L.append(f"Statistic: cosine between the group-mean per-scenario coefficient vectors (rescued = arm success & "
             f"native failure; regressed = the reverse); {PERMUTATIONS} label permutations; Holm over {rr['family']} tests. "
             "A small cosine with small p would mean the edit pushed rescued and regressed scenarios in different "
             "directions of the fitted subspace.\n")
    rows = []
    for k, v in rr["tests"].items():
        if "skipped" in v:
            L.append(f"- {k}: skipped ({v['skipped']}).")
            continue
        rows.append(dict(test=k, rescued=v["rescued"], regressed=v["regressed"], unchanged=v["unchanged"],
                         cosine=v["cosine_rescued_vs_regressed"], perm_p=v["permutation_p"], holm_p=v.get("holm_p"),
                         null_mean=v["null_cosine_mean"]))
    if rows:
        L.append(common.md_table(pd.DataFrame(rows), floatfmt="{:.3f}"))
        sig = [r["test"] for r in rows if r["holm_p"] is not None and r["holm_p"] < 0.05]
        if sig:
            L.append(f"\nAfter Holm, {len(sig)} of {len(rows)} tests are below 0.05 ({', '.join(sig)}): in those cells the "
                     "arm's mean applied direction differs between the scenarios it helped and those it hurt. This is a "
                     "post-hoc subgroup contrast on the same outcomes used to define the groups; it is a hypothesis for a "
                     "future pre-registered test, not evidence of a mechanism.")
        else:
            L.append(f"\nAfter Holm, none of the {len(rows)} tests is below 0.05: within the resolution of this panel the "
                     "edit did not point in a systematically different direction on the scenarios it rescued than on "
                     "those it regressed. Flips are therefore not explained by a coefficient-direction difference "
                     "visible at the scenario-mean level.")
    L.append("")

    # ---- 6 rubric
    L.append("## 6. Rubric: mapping the numbers to the two pictures\n")
    L.append("| axis | compact / model-native reading | distributed / task-specific reading | observed |\n|---|---|---|---|")
    L.append(f"| applied dimensionality (PR, learned) | <= {RUBRIC['pr_single_direction']}: one fixed direction did the work | "
             f">= {RUBRIC['pr_full_rank']}: all four directions in use | " +
             "; ".join(f"{k.split(':')[1]} {fmt(v['value'], 2)} ({v['verdict']})" for k, v in R["criteria"].items() if k.startswith("applied_pr:")) + " |")
    L.append(f"| coefficient map | intercept share >= {RUBRIC['intercept_constant_share']}: constant edit | "
             f"<= {RUBRIC['intercept_state_share']}: state-dependent edit | " +
             "; ".join(f"{k.split(':')[1]} {fmt(v['value'], 2)} ({v['verdict']})" for k, v in R["criteria"].items() if k.startswith("map:")) + " |")
    L.append(f"| spatial support (effective patches) | < {RUBRIC['spatial_concentrated_eff_patches']:.0f}: localised | "
             f"> {RUBRIC['spatial_broad_eff_patches']:.0f}: broad | " +
             "; ".join(f"{k.split(':')[1]} {fmt(v['value'], 0)} ({v['verdict']})" for k, v in R["criteria"].items() if k.startswith("spatial:")) + " |")
    L.append(f"| cross-task fitted residual direction (same checkpoint only; the 3-score readout is shared by construction "
             f"and excluded) | residual angle to the other task's basis < {RUBRIC['cross_task_shared_angle_deg']:.0f}° and "
             f"above the random-unit-vector null (p < {RUBRIC['cross_task_p']}): reused | cos^2 of that angle < "
             f"{RUBRIC['cross_task_orthogonal_overlap']:.0%}: task-specific | " +
             ("; ".join(f"{k.split(':')[1]} full-basis overlap {fmt(v['value'], 3)} of which "
                        f"{fmt(v.get('readout_share_of_overlap'), 2)} is readout; residual min angle "
                        f"{deg(v.get('residual_min_angle_to_other_basis_deg'))} ({v['verdict']})"
                        for k, v in R["criteria"].items() if k.startswith("cross_task:") and v["same_checkpoint"]) or
              "no same-checkpoint pair") + " |")
    L.append(f"\nScoring: {R['scoring']}. Per axis: " +
             ("; ".join(f"{ax} {a['compact']} compact / {a['distributed']} distributed of {a['n']} -> {a['point']}"
                        for ax, a in R["axes"].items()) or "none scored") +
             f". Total {R['compact_points']} compact, {R['distributed_points']} distributed over {R['axes_scored']} scored "
             f"axes. **Verdict: {R['verdict']}.**\n")
    if R["verdict"].startswith("compact"):
        L.append("Reading: the edit that was actually delivered is close to a single fixed direction in a subspace that "
                 "is reused across tasks and/or localised; this is the linear-representation picture (Othello-GPT style) "
                 "at the level of the *edit*, which is the only level this analysis sees.")
    elif R["verdict"].startswith("distributed"):
        L.append("Reading: the delivered edit uses the full rank-4 budget, depends on the state, loads broadly over the "
                 "patch grid and shares no subspace with the other task beyond chance. That is the 'Interpreting "
                 "Physics in Video World Models' picture: steering-relevant structure is distributed and task-specific, "
                 "and a four-direction budget is small against it.")
    else:
        crit = R["criteria"]
        c_axes = [k for k, v in crit.items() if v["verdict"] in ("single fixed direction", "constant-dominated", "concentrated", "shared subspace component")]
        d_axes = [k for k, v in crit.items() if v["verdict"] in ("full rank-4 in use", "state-dominated", "broad", "essentially orthogonal")]
        L.append("Reading: the axes disagree. Compact-side evidence: " + (", ".join(c_axes) if c_axes else "none") +
                 ". Distributed-side evidence: " + (", ".join(d_axes) if d_axes else "none") +
                 ". A fixed direction that is spatially broad (or a state-dependent edit inside a localised subspace) is "
                 "not either archetype; the honest summary is 'a small number of directions with distributed support', "
                 "which neither picture predicts cleanly.")
    # tie to regime
    if reg["overall"] == "positive":
        L.append(" Combined with the confirmed success gain, this says what *kind* of object moved the planner, not what it "
                 "encodes.")
    elif reg["overall"] == "negative":
        L.append(" Combined with the harmful behavioural reading, the geometry characterises a direction the planner is "
                 "sensitive to in the wrong way; that is still informative about steerability, not about semantics.")
    elif reg["overall"] == "mixed":
        L.append(" Combined with the mixed behavioural reading, note that the per-task geometry is fitted independently; "
                 "whether the tasks where the edit helped share a geometric signature is a question for a larger panel.")
    else:
        L.append(" Combined with the inconclusive behavioural reading, no inference from geometry to decision-relevance is "
                 "available: compact or distributed, the delivered edit did not resolvably change success.")
    L.append("")

    # ---- caveats
    L.append("## What this does and does not establish\n")
    L.append("**Does establish (descriptively, on the artifacts read):**")
    if bank_tasks:
        L.append(f"- The fitted rank-4 bases for {', '.join(bank_tasks)} are orthonormal to "
                 f"{max(J['banks'][t]['diagnostics'][LEARNED]['orthonormality_max_abs_error'] for t in bank_tasks):.1e}, "
                 "and their spatial and feature loading concentrations, principal angles to their controls and to each "
                 "other, and coefficient-map structure are as tabulated above.")
    if any(isinstance(ap.get(t), dict) and LEARNED in ap[t] and "skipped" not in ap[t][LEARNED] for t in result_tasks):
        L.append("- The effective dimensionality and mean direction of the coefficients that were actually delivered to "
                 "the planner's candidates in the fresh confirmation, for the learned arm and its random control"
                 + (" (synthetic here)." if prov["synthetic"] else "."))
    L.append("- Which rubric cell those numbers fall in, under thresholds stated before the numbers were read.\n")
    L.append("**Does not establish:**")
    L.append("- Semantics. The basis directions are PCA axes of activation variance on the fit split, not decoded "
             "physical variables; nothing here says what a direction *means*, and the fresh panel has no probe that "
             "could identify it. Compactness of an edit is not evidence of a model-native variable.")
    L.append("- Decision relevance. The geometry is computed from the bank and from the coefficients applied; it does not "
             "measure whether any direction changes the planner's ranking (see the decision diagnostic for that).")
    ck_groups = {}
    for t, c in J["principal_angles"].get("checkpoint_sha256", {}).items():
        ck_groups.setdefault(c, []).append(t)
    shared_groups = [ts for ts in ck_groups.values() if len(ts) > 1]
    if shared_groups:
        ck_sentence = ("cross-task comparison is within one predictor: " +
                       "; ".join(" and ".join(common.TASK_LABEL.get(t, t) for t in ts) + " share one checkpoint"
                                 for ts in shared_groups) +
                       ". Nothing here is a claim about reuse across predictors.")
    elif len(ck_groups) > 1:
        ck_sentence = ("no two banks share a checkpoint, so no cross-task subspace comparison is scored; nothing here is "
                       "a claim about reuse across tasks.")
    else:
        ck_sentence = "with at most one bank, no cross-task comparison exists."
    L.append("- Generality. Only the tasks with a bank on disk are analysed geometrically"
             + (f" ({', '.join(bank_tasks)})" if bank_tasks else "") + "; " +
             (f"skipped: {', '.join(f'{t} ({w})' for t, w in J['skipped_banks'].items())}; " if J["skipped_banks"] else "") +
             ck_sentence)
    if any(p.get("readout_decomposition") for p in J["principal_angles"].get("pairs", {}).values()):
        L.append("- Reuse of the readout. Every learned basis contains the 3-score PCA readout of its predictor, so two "
                 "fits on one checkpoint share up to three directions by construction; only the fitted residual "
                 "direction (section 2 decomposition) can speak to reuse, and it is what the rubric scores.")
    L.append("- Anything pre-registered. Every statistic in sections 1-6 is exploratory; the only pre-registered inputs are "
             "the arm-vs-native intervals quoted in section 0. Permutation and bootstrap corrections are stated per "
             "section; no correction spans sections.")
    if prov["synthetic"]:
        L.append("- Anything about the fresh confirmation: this run used a synthetic fixture.")
    L.append("")
    L.append("Figures: " + ", ".join(f"`{Path(p).name}`" for p in J["figures"].values() if p) + ".")
    Path(path).write_text("\n".join(L) + "\n")


# ================================================================================== main
def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    p = common.standard_parser("Representation geometry of the rank-4 intervention subspace (exploratory).")
    p.add_argument("--fixture-regime", default=None, help="informational label only; does not change the analysis")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    tasks = list(args.tasks)

    prov = common.provenance(args.results, args.freeze, args.analysis)
    if prov["synthetic"]:
        log.warning("provenance.synthetic is true: fresh-result geometry numbers are meaningless (fixture noise)")
    report = common.load_analysis(args.analysis)

    # --- banks
    banks, skipped_banks = {}, {}
    for t in tasks:
        try:
            b = common.load_bank(t)
        except Exception as e:  # noqa: BLE001
            b = None
            skipped_banks[t] = f"load_bank raised {type(e).__name__}: {e}"
        if b is None:
            skipped_banks.setdefault(t, "no operator_bank.pt on disk (refined fits for this task may only be on Drive)")
            log.warning("bank skipped for %s: %s", t, skipped_banks[t])
            continue
        banks[t] = b
        log.info("bank %s: %s", t, b["path"])
    bank_tasks = sorted(banks)
    bank_out = {}
    for t in bank_tasks:
        bank_out[t] = dict(path=banks[t]["path"], sha256=banks[t]["sha256"], dose=float(banks[t]["bank"]["dose"]),
                           checkpoint_sha256=checkpoint_of(banks[t]),
                           binding_task=str(banks[t]["bank"].get("binding", {}).get("task", "unknown")),
                           diagnostics=basis_diagnostics(banks[t]), coefficient_map=coefficient_map(banks[t]),
                           readout_alignment=readout_alignment(banks[t]))
    angles = principal_angle_block(banks, rng) if banks else dict(pairs={}, monte_carlo={}, analytic_baseline={})

    # --- fresh results
    log.info("loading episodes_frame")
    df = common.episodes_frame(args.results, tasks=tasks)
    result_tasks = [t for t in tasks if len(df) and t in set(df.task)]
    for t in tasks:
        if t not in result_tasks:
            log.warning("no completed scenarios for %s in %s", t, args.results)
    log.info("loading energy_frame")
    ef = common.energy_frame(args.results, tasks=result_tasks) if result_tasks else pd.DataFrame()
    log.info("streaming candidate-level coefficient moments")
    applied = candidate_moments(args.results, result_tasks)
    cl = call_level(ef, result_tasks)
    regime = behavioural_regime(report, df, tasks)
    rr = rescued_vs_regressed(df, ef, result_tasks, rng)

    # --- cross-task mean direction and map consistency
    cross = {}
    for a, b in combinations(result_tasks, 2):
        xa, xb = applied.get(a, {}).get(LEARNED, {}), applied.get(b, {}).get(LEARNED, {})
        if not xa.get("mean") or not xb.get("mean"):
            continue
        entry = dict(cosine_coefficient_coordinates=cosine(xa["mean"], xb["mean"]), cosine_ambient=None)
        if a in banks and b in banks and checkpoint_of(banks[a]) == checkpoint_of(banks[b]):
            ba = banks[a]["bank"]["operators"][LEARNED]["basis"].detach().cpu().double().numpy().reshape(4, -1)
            bb = banks[b]["bank"]["operators"][LEARNED]["basis"].detach().cpu().double().numpy().reshape(4, -1)
            entry["cosine_ambient"] = cosine(np.asarray(xa["mean"]) @ ba, np.asarray(xb["mean"]) @ bb)
        elif a in banks and b in banks:
            entry["why_no_ambient"] = "different checkpoints: ambient coordinates are not comparable"
        else:
            entry["why_no_ambient"] = "no bank for at least one task"
        cross[f"{a}_vs_{b}"] = entry
    const_check = {}
    for t in bank_tasks:
        x = applied.get(t, {}).get(LEARNED, {})
        cdir = bank_out[t]["coefficient_map"][LEARNED]["constant_direction"]
        if x.get("mean") and cdir:
            const_check[t] = dict(cosine=cosine(x["mean"], cdir))

    rubric = apply_rubric({t: bank_out[t]["diagnostics"] for t in bank_tasks},
                          {t: bank_out[t]["coefficient_map"] for t in bank_tasks}, angles, applied, bank_tasks, result_tasks)

    figs = dict(spatial=fig_spatial({t: bank_out[t]["diagnostics"] for t in bank_tasks}, bank_tasks, out),
                angles=fig_angles(angles, out), spectrum=fig_spectrum(applied, result_tasks, out))

    J = dict(script=SCRIPT, exploratory=True,
             pre_registered_inputs=[f"{LEARNED}-native simultaneous intervals quoted from the frozen report (section 0)"],
             families=dict(learned_vs_control=f"Bonferroni over {regime['exploratory_control_family']} tasks",
                           rescued_vs_regressed=f"Holm over {rr['family']} (task, arm) permutation tests",
                           principal_angles=f"uncorrected, against a {RANDOM_DRAWS}-draw random-subspace null"),
             seed=SEED, fixture_regime_label=args.fixture_regime, provenance=prov, synthetic=prov["synthetic"],
             bank_tasks=bank_tasks, result_tasks=result_tasks, skipped_banks=skipped_banks, banks=bank_out,
             principal_angles=angles, behavioural_regime=regime, applied_coefficients=applied, call_level=cl,
             cross_task_mean_direction=cross, constant_direction_check=const_check, rescued_vs_regressed=rr,
             rubric=rubric, figures=figs,
             caveat="PCA axes are activation-variance coordinates, not semantic variables; the fresh panel cannot identify semantics.")
    # keep the JSON readable: drop the bulky per-patch arrays into a sidecar-free compact form
    for t in bank_tasks:
        for arm in (LEARNED, CONTROL):
            d = J["banks"][t]["diagnostics"][arm]
            d["patch_loadings"] = [[round(v, 7) for v in row] for row in d["patch_loadings"]]
            d["subspace_mean_patch_loading"] = [round(v, 7) for v in d["subspace_mean_patch_loading"]]
            d["subspace_mean_feature_loading"] = [round(v, 7) for v in d["subspace_mean_feature_loading"]]
    common.write_json(out / f"{SCRIPT}.json", J)
    write_md(out / f"{SCRIPT}.md", J)
    log.info("wrote %s", out / f"{SCRIPT}.md")
    print(f"{SCRIPT}: regime={regime['overall']} verdict={rubric['verdict']} synthetic={prov['synthetic']} out={out}")


if __name__ == "__main__":
    main()
