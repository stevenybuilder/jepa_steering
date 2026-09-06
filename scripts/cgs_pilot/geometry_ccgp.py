#!/usr/bin/env python3
"""Abstraction / ontology gate for the driving cell: CCGP, parallelism score and shattering dimensionality
(Bernardi et al. 2020, "The geometry of abstraction in the hippocampus and prefrontal cortex"; design doc
``cross model design jepa.md`` Step 5, "ontology first").

Reads the localization dumps written by ``localize_interaction.py --domain driving`` (one per arm; loaders, token pooling
and scene bookkeeping imported from ``geometry_cross_arm.py`` / ``geometry_localize.py``, not copied) and the merged
stimulus (``manifest.jsonl`` for the per-scene factors, ``masks/`` for the token groups).  Per arm x site x token group x
imagined step, on discovery scenes only:

CONDITIONS.  A condition is a cell of the factorial, hazard level {0 sidewalk, 1 pedestrian in lane, 2 H0' mirror pose,
3 cone in lane} x action {0 brake, 1 throttle}, crossed with BINNED scene factors read from the manifest rows:
hazard distance (``hazard_dist_m``), ego prefix throttle (``prefix_throttle``, v0.9; v0.7/v0.8 rows carry none ->
0.2) and in-lane lateral offset (``hazard_lateral_offset_m``, v0.9; -> 0.0), the same defaults as
``metadrive_hazard_pilot.apply_row_factors``.  A factor is binned (``--n-bins`` quantile bins, or its distinct values when
there are at most ``--n-bins`` of them, e.g. the v0.8 8 m / 10 m batches) only when it varies across the scenes by more
than a tolerance and every bin keeps >= ``--min-scenes-per-bin`` scenes; a constant factor (v0.7: 9 m everywhere) adds
no coordinate.  Level 2 enters only when every scene has its H0' cells.  Each scene contributes one cell per
(level, action) at its own factor bins, so a condition's samples are whole scenes.

DICHOTOMIES (relational ontology, resolved per arm: arm A = pedestrian solid / cone ghost, arm B the reverse):
  solid_consequence   level == solid identity (identity x in-lane x arm), both actions, vs every other cell   [relational]
  solid_throttle      the consequence cell (solid identity, throttle) vs every other cell                     [relational]
                      (needs >= 2 factor-bin conditions in the positive class, i.e. a varied stimulus)
  inlane              in-lane presence, levels {1, 3} vs {0, 2}
  action              throttle vs brake
  identity            pedestrian (1) vs cone (3), in-lane conditions only
  dist / prefix / lateral   upper vs lower factor bins (only when the factor is binned)

CCGP.  Linear decoder = ridge regression on class-balanced targets (+1/n_pos, -1/n_neg, unpenalised intercept, ridge
lambda = ``--ridge-alpha`` x mean eigenvalue of the training scatter), decision at the midpoint of the projected class
means.  For a dichotomy with positive conditions P and negative conditions N, one split holds out one condition of each
class (p, n) and trains on ALL cells of the remaining conditions; every (p, n) pair is a split (all of them, or a frozen
random subset of ``--max-splits``).  Inside every split the decoder is refitted leave-one-scene-out (exact block deletion
of the scene's training rows), so a held-out cell is scored by a decoder that saw neither its condition nor its scene.
Per-scene accuracy = balanced mean over the scene's held-out cells, averaged over splits; CCGP = its mean over scenes;
unit = scene (scene-bootstrap CI).

PS.  Coding vector of a condition pair = difference of the two condition means (all scenes, in-sample).  ``ps_best`` =
mean pairwise cosine over the coding vectors of a one-to-one pairing of P with N (injective when unbalanced), maximised
over pairings (all pairings when there are <= 5040, else 2000 frozen random pairings plus a swap hill-climb);
``ps_matched`` = the same for the natural pairing of conditions that differ only in the dichotomy variable (solid <->
ghost identity at the same action and bins; in-lane <-> its off-lane counterpart; throttle <-> brake; pedestrian <->
cone; upper <-> lower bin).  Also reported: the parallelism of the per-bin DiD vectors
(h[l,1]-h[l,0])-(h[0,1]-h[0,0]) across factor bins (the design doc's "PS of matched DiD vectors"), when >= 2 bins.

SD (control).  Shattering dimensionality = mean leave-one-scene-out decoding accuracy over random dichotomies of the same
eligible conditions at MATCHED prevalence (same number of positive conditions; all of them when there are <=
``--n-random-dichotomies``, else that many, frozen seed), every condition present in training (standard cross-validation,
not cross-condition).  ``ccgp_random`` = the CCGP protocol applied to the same random dichotomies (up to
``--n-random-ccgp`` of them, ``--max-splits-random`` splits each): the condition-shuffled dichotomy null.

NULLS AND INFERENCE.  (a) label permutation within scene clusters: the dichotomy labels of a scene's eligible cells are
permuted within the scene (``--n-perm`` permutations shared by every site / group / step of the arm, so the max-T is
valid; for scene-constant labels such as the factor bins the labels are permuted across scenes instead), CCGP recomputed with the same splits and the same LOSO decoders (only the targets change, so the permutations
are batched through the linear solve); statistic = mean per-scene accuracy - 0.5; p_raw one-sided;
Westfall-Young max-statistic over the sites of a (dichotomy, group, step) family.  (b) condition-shuffled dichotomies
(``ccgp_random``, above) and the SD control.  (c) scene-clustered bootstrap CIs for CCGP, SD, PS and the differences
CCGP - SD, CCGP - ccgp_random.  PS null = the same within-scene permutation applied to the condition labels.

GATE (per dichotomy, per site x group x step): PASS when the scene-bootstrap CI of CCGP - SD lies above 0 AND the max-T
label-permutation p < 0.05; FAIL otherwise; NOT_TESTABLE when a class has < 2 conditions.  Relational dichotomies are
``solid_consequence`` and ``solid_throttle``; "advancement requires relational CCGP beyond [the SD control]".

``--self-test`` writes synthetic dumps in the exact layout the loaders expect: (i) planted ABSTRACT variables (one fixed
coding direction per variable, additive across conditions, parallel by construction) must give CCGP ~ 1, PS ~ 1 and a
PASS; (ii) a planted XOR-like code (the variable is carried by a different direction in every context condition) must
give high SD but CCGP ~ chance, PS ~ 0 and a FAIL.

Output ``<out>/ccgp.json`` + ``ccgp.md``.  Descriptive / associational only (decodability is not use; causal claims stay
with the patching stage); everything is fitted on discovery scenes only.
"""

from __future__ import annotations

import argparse
import itertools
import warnings
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry_cross_arm import _orthonormal, boot_mean, load_arm, load_site, t_stat, token_sets_for  # noqa: E402
from geometry_localize import pool_site  # noqa: E402
from protocol import NULL_CONTROL_HAZARD, OBJECT_HAZARD  # noqa: E402
from stats_utils import cosine, finite  # noqa: E402
from token_groups import alias_note, load_cell_groups, set_domain, union  # noqa: E402

PROTOCOL = "cgs-geometry-ccgp-v0.1"
PED, NULL, OBJ = 1, NULL_CONTROL_HAZARD, OBJECT_HAZARD
SOLID_LEVEL = {"A": PED, "B": OBJ}
GHOST_LEVEL = {"A": OBJ, "B": PED}
ARM_NAMES = {"A": "A: PED-solid / OBJ-ghost", "B": "B: PED-ghost / OBJ-solid"}
DEFAULT_GROUPS = ("hazard", "corridor", "hazard_corridor")
# (name, manifest key, default when the key is absent [None = required], minimum range to be binned)
FACTORS = (("dist", "hazard_dist_m", None, 0.05), ("prefix", "prefix_throttle", 0.2, 0.02), ("lateral", "hazard_lateral_offset_m", 0.0, 0.01))
RELATIONAL = ("solid_consequence", "solid_throttle")
CHANCE = 0.5
MAX_ENUM_PAIRINGS = 5040
INTERPRETATION_SCOPE = (
    "Descriptive/associational (decodability is not use). CCGP, PS and SD are computed on discovery scenes only, per arm x site x "
    "token group x imagined step; decoders are refitted leave-one-scene-out inside every held-out-condition split; CIs are "
    "scene bootstraps; the label-permutation p is max-T corrected over the sites of a (dichotomy, group, step) family. A PASS "
    "says the dichotomy is read out by a decoder that transfers across held-out conditions AND scenes better than random "
    "dichotomies at matched prevalence; it does not say the model uses that readout (patching does)."
)


# --------------------------------------------------------------------------- #
# Scene factors and conditions
# --------------------------------------------------------------------------- #


def scene_factors(stimulus: Path, scenes: list[str]) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    """Per-scene stimulus factors from the merged manifest (first row of each pair_id; the factors are per seed)."""
    rows: dict[str, dict[str, Any]] = {}
    for line in (stimulus / "manifest.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows.setdefault(str(r["pair_id"]), r)
    out, protos, randomized = {}, set(), set()
    for s in scenes:
        r = rows.get(s)
        if r is None:
            raise SystemExit(f"scene {s} has no row in {stimulus}/manifest.jsonl")
        f = {}
        for name, key, default, _ in FACTORS:
            if key in r and r[key] is not None:
                f[name] = float(r[key])
            elif default is None:
                raise SystemExit(f"manifest row of {s} lacks {key}")
            else:
                f[name] = float(default)
        out[s] = f
        if r.get("protocol_version"):
            protos.add(str(r["protocol_version"]))
        if r.get("randomized_factors"):
            randomized |= set(r["randomized_factors"])
    meta = {"protocol_versions": sorted(protos), "randomized_factor_keys": sorted(randomized),
            "defaults": "rows without prefix_throttle / hazard_lateral_offset_m (v0.7/v0.8) get 0.2 / 0.0 as metadrive_hazard_pilot.apply_row_factors"}
    return out, meta


def bin_factor(values: np.ndarray, n_bins: int, min_range: float, min_per_bin: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Bin one factor over scenes: constant -> one bin; <= n_bins distinct values -> one bin per value; else quantile bins.
    Drops to fewer bins until every bin holds >= min_per_bin scenes."""
    v = np.asarray(values, dtype=np.float64)
    lo, hi = float(v.min()), float(v.max())
    info: dict[str, Any] = {"range": [lo, hi], "n_scenes": int(len(v))}
    if hi - lo < min_range:
        return np.zeros(len(v), dtype=int), {**info, "n_bins": 1, "method": "constant"}
    distinct = np.unique(np.round(v, 6))
    for nb in range(max(1, n_bins), 0, -1):
        if nb == 1:
            break
        if len(distinct) <= nb:
            b = np.searchsorted(distinct, np.round(v, 6))
            method, edges, nb_eff = "distinct_values", distinct.tolist(), int(len(distinct))
        else:
            q = np.quantile(v, np.linspace(0, 1, nb + 1)[1:-1])
            b = np.searchsorted(q, v, side="right")
            method, edges, nb_eff = "quantile", [float(x) for x in q], nb
        counts = np.bincount(b, minlength=nb_eff)
        if nb_eff >= 2 and counts.min() >= min_per_bin:
            return b.astype(int), {**info, "n_bins": nb_eff, "method": method, "edges": edges, "counts": counts.tolist()}
    return np.zeros(len(v), dtype=int), {**info, "n_bins": 1, "method": "too_few_scenes_per_bin"}


def complete_scenes(cells: list[dict[str, Any]]) -> tuple[list[str], list[int], dict[str, Any]]:
    """Scenes with all cells of levels {0, 1, 3} x actions {0, 1}; level 2 is kept only when every such scene has both
    H0' cells (otherwise those cells are dropped from the condition set)."""
    by: dict[str, set] = {}
    for c in cells:
        by.setdefault(str(c["pair_id"]), set()).add((int(c["hazard"]), int(c["candidate_action"])))
    need = {(lv, a) for lv in (0, PED, OBJ) for a in (0, 1)}
    scenes = sorted(s for s, ks in by.items() if need <= ks)
    n_null = sum(1 for s in scenes if {(NULL, 0), (NULL, 1)} <= by[s])
    levels = [0, PED, OBJ] + ([NULL] if scenes and n_null == len(scenes) else [])
    status = {"n_scenes": len(scenes), "n_scenes_dropped_incomplete": len(by) - len(scenes), "n_null_scenes": n_null,
              "null_factor": "none" if n_null == 0 else ("full" if n_null == len(scenes) else "partial"), "levels_used": sorted(levels)}
    return scenes, sorted(levels), status


class Design:
    """Cells (rows of the pooled matrix) -> scene index, condition tuple (level, action, bins...) and condition index."""

    def __init__(self, cells: list[dict[str, Any]], scenes: list[str], levels: list[int], factors: dict[str, dict[str, float]], n_bins: int, min_per_bin: int):
        self.scenes = scenes
        self.scene_index = {s: i for i, s in enumerate(scenes)}
        self.binning: dict[str, Any] = {}
        bins: dict[str, np.ndarray] = {}
        for name, _, _, min_range in FACTORS:
            b, info = bin_factor(np.array([factors[s][name] for s in scenes]), n_bins, min_range, min_per_bin)
            self.binning[name] = info
            if info["n_bins"] > 1:
                bins[name] = b
        self.factor_names = [n for n, _, _, _ in FACTORS if n in bins]
        self.n_bins_of = {n: self.binning[n]["n_bins"] for n in self.factor_names}
        self.scene_bins = {s: tuple(int(bins[n][i]) for n in self.factor_names) for i, s in enumerate(scenes)}
        keep = [c for c in cells if str(c["pair_id"]) in self.scene_index and int(c["hazard"]) in levels]
        self.cells = keep
        self.rows = np.asarray([int(c["row"]) for c in keep], dtype=int)  # rows into the dump / pooled matrix
        self.cell_scene = np.asarray([self.scene_index[str(c["pair_id"])] for c in keep], dtype=int)
        self.cell_cond_tuple = [(int(c["hazard"]), int(c["candidate_action"])) + self.scene_bins[str(c["pair_id"])] for c in keep]
        self.conds = sorted(set(self.cell_cond_tuple))
        self.cond_index = {c: i for i, c in enumerate(self.conds)}
        self.cell_cond = np.asarray([self.cond_index[c] for c in self.cell_cond_tuple], dtype=int)
        self.levels = levels
        self.S, self.C = len(scenes), len(self.conds)

    def describe(self) -> dict[str, Any]:
        counts = np.bincount(self.cell_cond, minlength=self.C)
        return {"n_scenes": self.S, "n_cells": int(len(self.cells)), "levels": self.levels, "factor_names_binned": self.factor_names, "binning": self.binning,
                "n_conditions": self.C, "conditions": [{"level": c[0], "action": c[1], **{n: c[2 + k] for k, n in enumerate(self.factor_names)}, "n_cells": int(counts[i])} for i, c in enumerate(self.conds)],
                "condition_tuple": "(level, action" + "".join(f", {n}_bin" for n in self.factor_names) + ")"}


# --------------------------------------------------------------------------- #
# Dichotomies
# --------------------------------------------------------------------------- #


def dichotomies_for(arm: str, design: Design) -> dict[str, dict[str, Any]]:
    solid, ghost = SOLID_LEVEL[arm], GHOST_LEVEL[arm]
    has_null = NULL in design.levels
    fn = design.factor_names
    conds = design.conds

    def offlane_partner(c):
        return (0 if c[0] == PED or not has_null else NULL,) + c[1:]

    D: dict[str, dict[str, Any]] = {
        "solid_consequence": {"description": f"level == {solid} (solid identity in lane, both actions) vs every other cell; identity x in-lane x arm", "relational": True,
                              "eligible": lambda c: True, "positive": lambda c: c[0] == solid, "partner": lambda c: (ghost,) + c[1:], "partner_note": "ghost identity at the same action and bins"},
        "solid_throttle": {"description": f"(level == {solid}, throttle) = the consequence cell vs every other cell", "relational": True,
                           "eligible": lambda c: True, "positive": lambda c: c[0] == solid and c[1] == 1, "partner": lambda c: (ghost, 1) + c[2:], "partner_note": "ghost identity at throttle, same bins"},
        "inlane": {"description": "in-lane presence: levels {1, 3} vs {0" + (", 2" if has_null else "") + "}", "relational": False,
                   "eligible": lambda c: True, "positive": lambda c: c[0] in (PED, OBJ), "partner": offlane_partner, "partner_note": "pedestrian in lane <-> sidewalk pose (0); cone in lane <-> H0' mirror pose (2) when present, else 0"},
        "action": {"description": "throttle vs brake, all levels", "relational": False,
                   "eligible": lambda c: True, "positive": lambda c: c[1] == 1, "partner": lambda c: (c[0], 0) + c[2:], "partner_note": "brake cell of the same level and bins"},
        "identity": {"description": "pedestrian (1) vs cone (3), in-lane conditions only", "relational": False,
                     "eligible": lambda c: c[0] in (PED, OBJ), "positive": lambda c: c[0] == PED, "partner": lambda c: (OBJ,) + c[1:], "partner_note": "cone cell of the same action and bins"},
    }
    for k, name in enumerate(fn):
        nb = design.n_bins_of[name]
        top = set(range((nb + 1) // 2, nb))  # upper half of the bins (nb = 2: bin 1)
        D[name] = {"description": f"{name}: upper bins {sorted(top)} vs lower bins of {nb}", "relational": False,
                   "eligible": (lambda c: True), "positive": (lambda c, k=k, top=top: c[2 + k] in top),
                   "partner": (lambda c, k=k, nb=nb: c[: 2 + k] + (nb - 1 - c[2 + k],) + c[3 + k:]), "partner_note": "mirror bin, same cell and other bins"}
    for name, d in D.items():
        E = [i for i, c in enumerate(conds) if d["eligible"](c)]
        P = [i for i in E if d["positive"](conds[i])]
        N = [i for i in E if not d["positive"](conds[i])]
        pairs = []
        for i in P:
            pc = d["partner"](conds[i])
            j = design.cond_index.get(pc)
            if j is not None and j in N:
                pairs.append((i, j))
        d.update({"eligible_idx": E, "pos_idx": P, "neg_idx": N, "matched_pairs": pairs})
    return D


# --------------------------------------------------------------------------- #
# Batched LOSO ridge decoder (linear in the targets: permutations / random dichotomies are columns)
# --------------------------------------------------------------------------- #


def ridge_lambda(X: np.ndarray, alpha: float) -> float:
    Xc = X - X.mean(0, keepdims=True)
    return float(alpha * np.sum(Xc * Xc) / X.shape[1])


def loso_decisions(X_tr: np.ndarray, g_tr: np.ndarray, Yp: np.ndarray, Yn: np.ndarray, X_te: np.ndarray, g_te: np.ndarray, lam: float) -> np.ndarray:
    """Decision values D [m, P] for the test rows under P labelings.  Yp / Yn [n, P] are 0/1 class indicators of the
    training rows.  Ridge on class-balanced targets (+1/n_pos, -1/n_neg of the training set actually used) with an
    unpenalised intercept; for a test row of scene g the decoder is refitted without scene g's training rows (exact block
    deletion / Woodbury, applied to the positive- and negative-indicator solutions separately so the class balancing uses
    the reduced counts) and the threshold is the midpoint of the projected class means of the reduced training set.
    D > 0 -> positive.  Exact: equals an explicit refit on the reduced set (checked to 1e-10)."""
    n, d = X_tr.shape
    Xa = np.hstack([X_tr, np.ones((n, 1))])
    A = Xa.T @ Xa
    A[np.arange(d), np.arange(d)] += lam
    Ainv = np.linalg.inv(A)
    Sp, Sn = Xa.T @ Yp, Xa.T @ Yn  # [d+1, P] sums of the positive / negative rows (intercept row = counts)
    Wp, Wn = Ainv @ Sp, Ainv @ Sn  # ridge solutions for the two indicator targets
    npos, nneg = Yp.sum(0), Yn.sum(0)
    Xte_a = np.hstack([X_te, np.ones((len(X_te), 1))])
    D = np.zeros((len(X_te), Yp.shape[1]))
    for g in np.unique(g_te):
        te = np.where(g_te == g)[0]
        G = np.where(g_tr == g)[0]
        if len(G):
            XG = Xa[G]
            AXGt = Ainv @ XG.T  # [d+1, k]
            K = AXGt @ np.linalg.inv(np.eye(len(G)) - XG @ AXGt)
            Wpg = Wp - K @ (Yp[G] - XG @ Wp)
            Wng = Wn - K @ (Yn[G] - XG @ Wn)
            Spg, Sng = Sp - XG.T @ Yp[G], Sn - XG.T @ Yn[G]
            cp, cn = npos - Yp[G].sum(0), nneg - Yn[G].sum(0)
        else:
            Wpg, Wng, Spg, Sng, cp, cn = Wp, Wn, Sp, Sn, npos, nneg
        okp, okn = cp > 0, cn > 0
        ip = np.where(okp, 1.0 / np.maximum(cp, 1), 0.0)
        inn = np.where(okn, 1.0 / np.maximum(cn, 1), 0.0)
        Wg = Wpg * ip[None, :] - Wng * inn[None, :]
        mg = 0.5 * (Spg * ip[None, :] + Sng * inn[None, :])
        theta = np.sum(mg * Wg, axis=0)
        D[te] = Xte_a[te] @ Wg - theta[None, :]
    return D


def scene_balanced_accuracy(D: np.ndarray, L: np.ndarray, g_te: np.ndarray, S: int) -> np.ndarray:
    """[S, P] balanced accuracy per scene (mean of the accuracies on its positive and its negative held-out rows under each
    labeling column; nan where the scene has no held-out row)."""
    correct = ((D > 0) == (L > 0)).astype(np.float64)
    pos, neg = (L > 0).astype(np.float64), (L < 0).astype(np.float64)
    cp = np.zeros((S, D.shape[1]))
    cn = np.zeros_like(cp)
    npos = np.zeros_like(cp)
    nneg = np.zeros_like(cp)
    np.add.at(cp, g_te, correct * pos)
    np.add.at(cn, g_te, correct * neg)
    np.add.at(npos, g_te, pos)
    np.add.at(nneg, g_te, neg)
    with np.errstate(divide="ignore", invalid="ignore"):
        ap, an = cp / npos, cn / nneg
    both = np.isfinite(ap) & np.isfinite(an)
    out = np.where(both, 0.5 * (np.nan_to_num(ap) + np.nan_to_num(an)), np.where(np.isfinite(ap), ap, an))
    return out


def label_permutations(labels: np.ndarray, cell_scene: np.ndarray, n_perm: int, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    """[n_cells, 1 + n_perm] labels (column 0 observed).  Within-scene mode: each other column permutes the labels of every
    scene's labelled (non-zero) cells within the scene (scene clusters preserved).  When every scene's labelled cells share
    ONE label (scene-level factor dichotomies such as the distance bins), a within-scene permutation is the identity, so
    the labels are permuted ACROSS scenes instead (the scene -> label map is shuffled; the unit of exchangeability is the
    scene)."""
    L = np.repeat(labels[:, None], 1 + n_perm, axis=1).astype(np.float64)
    scenes = np.unique(cell_scene)
    varies = any(len(np.unique(labels[(cell_scene == g) & (labels != 0)])) > 1 for g in scenes)
    if varies:
        for g in scenes:
            idx = np.where((cell_scene == g) & (labels != 0))[0]
            if len(idx) < 2:
                continue
            for p in range(1, 1 + n_perm):
                L[idx, p] = labels[rng.permutation(idx)]
        return L, "within_scene"
    lab_of = {g: labels[(cell_scene == g) & (labels != 0)] for g in scenes}
    lab_of = {g: (v[0] if len(v) else 0.0) for g, v in lab_of.items()}
    keys = [g for g in scenes if lab_of[g] != 0]
    vals = np.asarray([lab_of[g] for g in keys])
    for p in range(1, 1 + n_perm):
        perm = vals[rng.permutation(len(vals))]
        for g, v in zip(keys, perm):
            m = (cell_scene == g) & (labels != 0)
            L[m, p] = v
    return L, "across_scenes"


def choose_splits(P: list[int], N: list[int], max_splits: int, rng: np.random.Generator) -> tuple[list[tuple[int, int]], bool]:
    allp = [(p, n) for p in P for n in N]
    if len(allp) <= max_splits:
        return allp, True
    pick = rng.choice(len(allp), size=max_splits, replace=False)
    return [allp[i] for i in sorted(pick)], False


def ccgp_curve(X: np.ndarray, design: Design, L: np.ndarray, E: list[int], splits: list[tuple[int, int]], alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """CCGP under every labeling column of L [n_cells, P] (labels +1 / -1 / 0 = not eligible).
    Returns (per_scene [S, P] mean over the splits in which the scene has held-out cells, per_split [n_splits, P])."""
    S = design.S
    cond = design.cell_cond
    elig = np.isin(cond, E)
    acc = np.full((len(splits), S, L.shape[1]), np.nan)
    for k, (p, n) in enumerate(splits):
        te = elig & np.isin(cond, [p, n])
        tr = elig & ~te
        Ltr, Lte = L[tr], L[te]
        if tr.sum() < 4 or te.sum() == 0:
            continue
        lam = ridge_lambda(X[tr], alpha)
        D = loso_decisions(X[tr], design.cell_scene[tr], (Ltr > 0).astype(np.float64), (Ltr < 0).astype(np.float64), X[te], design.cell_scene[te], lam)
        acc[k] = scene_balanced_accuracy(D, Lte, design.cell_scene[te], S)
    with np.errstate(invalid="ignore"):
        per_scene = np.nanmean(acc, axis=0)
        per_split = np.nanmean(acc, axis=1)
    return per_scene, per_split


def sd_curve(X: np.ndarray, design: Design, L: np.ndarray, E: list[int], alpha: float) -> np.ndarray:
    """Standard (held-out scene, every condition in training) decoding accuracy per scene [S, P] for the labelings L."""
    elig = np.isin(design.cell_cond, E)
    Xe, Le, ge = X[elig], L[elig], design.cell_scene[elig]
    lam = ridge_lambda(Xe, alpha)
    D = loso_decisions(Xe, ge, (Le > 0).astype(np.float64), (Le < 0).astype(np.float64), Xe, ge, lam)
    return scene_balanced_accuracy(D, Le, ge, design.S)


def random_dichotomies(E: list[int], n_pos: int, K: int, exclude: set[frozenset], rng: np.random.Generator) -> tuple[list[list[int]], bool]:
    """Up to K distinct random positive sets of size n_pos inside E (all of them when there are <= K), excluding the
    dichotomy of interest and, for balanced dichotomies, its complement."""
    total = math.comb(len(E), n_pos)
    balanced = 2 * n_pos == len(E)
    if total <= max(K, 1) * (2 if balanced else 1) and total <= 20000:
        allsets, seen = [], set()
        for comb in itertools.combinations(E, n_pos):
            key = frozenset(comb)
            if key in exclude:
                continue
            if balanced:
                ck = frozenset(set(E) - key)
                if ck in seen:
                    continue
            seen.add(key)
            allsets.append(sorted(comb))
        if len(allsets) > K:
            pick = rng.choice(len(allsets), size=K, replace=False)
            return [allsets[i] for i in sorted(pick)], False
        return allsets, True
    out, seen = [], set()
    tries = 0
    while len(out) < K and tries < 50 * K:
        tries += 1
        comb = frozenset(rng.choice(E, size=n_pos, replace=False).tolist())
        if comb in exclude or comb in seen or (balanced and frozenset(set(E) - comb) in seen):
            continue
        seen.add(comb)
        out.append(sorted(comb))
    return out, False


def labels_for(design: Design, pos: list[int], E: list[int]) -> np.ndarray:
    lab = np.zeros(len(design.cells))
    elig = np.isin(design.cell_cond, E)
    lab[elig] = -1.0
    lab[np.isin(design.cell_cond, pos)] = 1.0
    return lab


# --------------------------------------------------------------------------- #
# Parallelism score
# --------------------------------------------------------------------------- #


def condition_means(X: np.ndarray, cond: np.ndarray, C: int) -> np.ndarray:
    mu = np.full((C, X.shape[1]), np.nan)
    for c in range(C):
        m = cond == c
        if m.any():
            mu[c] = X[m].mean(0)
    return mu


def pair_cosine_matrix(mu: np.ndarray, P: list[int], N: list[int]) -> np.ndarray:
    """cos between the coding vectors mu[p]-mu[n] of every (p, n) pair, from the Gram matrix of the condition means."""
    G = mu @ mu.T
    pairs = [(p, n) for p in P for n in N]
    ip = np.asarray([p for p, _ in pairs])
    ineg = np.asarray([n for _, n in pairs])
    inner = G[np.ix_(ip, ip)] - G[np.ix_(ip, ineg)] - G[np.ix_(ineg, ip)] + G[np.ix_(ineg, ineg)]
    norms = np.sqrt(np.clip(np.diag(inner), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        Cm = inner / np.outer(norms, norms)
    return np.where(np.isfinite(Cm), Cm, 0.0)


def pairing_score(Cm: np.ndarray, pair_pos: dict[tuple[int, int], int], pairing: list[tuple[int, int]]) -> float:
    idx = [pair_pos[pr] for pr in pairing]
    if len(idx) < 2:
        return float("nan")
    sub = Cm[np.ix_(idx, idx)]
    k = len(idx)
    return float((sub.sum() - np.trace(sub)) / (k * (k - 1)))


def ps_best(mu: np.ndarray, P: list[int], N: list[int], rng: np.random.Generator, matched: list[tuple[int, int]] | None = None, n_random: int = 2000) -> dict[str, Any]:
    """Max over one-to-one pairings (injective from the smaller class) of the mean pairwise cosine of the coding vectors."""
    small, large, flip = (P, N, False) if len(P) <= len(N) else (N, P, True)
    if len(small) < 2:
        return {"value": float("nan"), "n_pairs": len(small), "method": "undefined (< 2 pairs)"}
    Cm = pair_cosine_matrix(mu, P, N)
    pair_pos = {(p, n): i for i, (p, n) in enumerate([(p, n) for p in P for n in N])}

    def pairing_of(partners):
        return [((b, a) if flip else (a, b)) for a, b in zip(small, partners)]

    def score(partners):
        return pairing_score(Cm, pair_pos, pairing_of(partners))

    n_all = math.perm(len(large), len(small))
    best, best_partners = -np.inf, None
    if n_all <= MAX_ENUM_PAIRINGS:
        method = f"all {n_all} pairings"
        for perm in itertools.permutations(large, len(small)):
            v = score(perm)
            if v > best:
                best, best_partners = v, list(perm)
    else:
        method = f"{n_random} random pairings + greedy partner replacement (of {n_all}; approximate, a lower bound on the max)"
        cands = []
        if matched:
            part = {(n if flip else p): (p if flip else n) for p, n in matched}
            if all(a in part for a in small) and len(set(part[a] for a in small)) == len(small):
                cands.append([part[a] for a in small])
        for _ in range(n_random):
            cands.append(list(rng.choice(large, size=len(small), replace=False)))
        for partners in cands:
            v = score(partners)
            if v > best:
                best, best_partners = v, list(partners)
        # greedy: replace one partner by a FREE candidate of the large class when that raises the mean pairwise cosine
        k = len(small)
        improved = True
        while improved:
            improved = False
            for i in range(k):
                free = [c for c in large if c not in best_partners]
                if not free:
                    break
                idx = [pair_pos[pr] for pr in pairing_of(best_partners)]
                others = [idx[j] for j in range(k) if j != i]
                cur = Cm[idx[i], others].sum()
                cand_idx = [pair_pos[((c, small[i]) if flip else (small[i], c))] for c in free]
                gains = Cm[np.ix_(cand_idx, others)].sum(axis=1) - cur
                j = int(np.argmax(gains))
                if gains[j] > 1e-12:
                    best_partners[i] = free[j]
                    best = score(best_partners)
                    improved = True
    return {"value": float(best), "n_pairs": len(small), "method": method, "pairing": pairing_of(best_partners)}


def ps_matched(mu: np.ndarray, pairs: list[tuple[int, int]]) -> float:
    if len(pairs) < 2:
        return float("nan")
    V = np.asarray([mu[p] - mu[n] for p, n in pairs])
    if not np.all(np.isfinite(V)):
        return float("nan")
    nrm = np.linalg.norm(V, axis=1)
    if np.any(nrm == 0):
        return float("nan")
    Cm = (V @ V.T) / np.outer(nrm, nrm)
    k = len(pairs)
    return float((Cm.sum() - np.trace(Cm)) / (k * (k - 1)))


def did_parallelism(mu: np.ndarray, design: Design, levels: list[int]) -> dict[str, Any]:
    """Per-bin DiD vectors (h[l,1]-h[l,0])-(h[0,1]-h[0,0]) for each in-lane / null level and their parallelism across
    the factor bins (mean pairwise cosine; nan with < 2 bins) plus the cosine between the bin-averaged DiDs of two levels."""
    bins = sorted({c[2:] for c in design.conds})
    out: dict[str, Any] = {"n_bins": len(bins), "per_level": {}}
    means = {}
    for lv in levels:
        vecs = []
        for b in bins:
            idx = [design.cond_index.get((lv, 1) + b), design.cond_index.get((lv, 0) + b), design.cond_index.get((0, 1) + b), design.cond_index.get((0, 0) + b)]
            if any(i is None for i in idx):
                continue
            v = mu[idx[0]] - mu[idx[1]] - mu[idx[2]] + mu[idx[3]]
            if np.all(np.isfinite(v)):
                vecs.append(v)
        if not vecs:
            continue
        V = np.asarray(vecs)
        means[lv] = V.mean(0)
        if len(V) >= 2:
            nrm = np.linalg.norm(V, axis=1)
            Cm = (V @ V.T) / np.outer(np.maximum(nrm, 1e-12), np.maximum(nrm, 1e-12))
            k = len(V)
            ps = float((Cm.sum() - np.trace(Cm)) / (k * (k - 1)))
        else:
            ps = float("nan")
        out["per_level"][str(lv)] = {"ps_across_bins": ps, "n_bins_with_did": int(len(V)), "mean_norm": float(np.mean(np.linalg.norm(V, axis=1)))}
    keys = sorted(means)
    out["cosine_between_levels"] = {f"{a}_vs_{b}": cosine(means[a], means[b]) for a, b in itertools.combinations(keys, 2)}
    return out


# --------------------------------------------------------------------------- #
# Per (arm, site, group, step) analysis
# --------------------------------------------------------------------------- #


def summ_scene(values: np.ndarray, n_boot: int, seed: int) -> dict[str, Any]:
    v = np.asarray(values, dtype=np.float64)
    s = boot_mean(v, n_boot, seed)
    s["t_vs_chance"] = t_stat(v - CHANCE)
    return s


def analyze_entry(X: np.ndarray, design: Design, D: dict[str, dict[str, Any]], perms: dict[str, tuple[np.ndarray, str]], args, seed: int) -> dict[str, Any]:
    """X [n_cells, d] pooled activations in design row order.  ``perms[name]`` = ([n_cells, 1 + n_perm] labelings, mode)."""
    out: dict[str, Any] = {"dichotomies": {}}
    mu = condition_means(X, design.cell_cond, design.C)
    rng = np.random.default_rng(seed + 7)
    for name, d in D.items():
        E, P, N = d["eligible_idx"], d["pos_idx"], d["neg_idx"]
        e: dict[str, Any] = {"relational": name in RELATIONAL, "n_conditions": len(E), "n_pos_conditions": len(P), "n_neg_conditions": len(N), "n_matched_pairs": len(d["matched_pairs"])}
        if len(P) < 2 or len(N) < 2:
            e.update({"status": "not_testable", "reason": "each class needs >= 2 conditions for a held-out-condition split", "verdict": "NOT_TESTABLE"})
            out["dichotomies"][name] = e
            continue
        L, perm_mode = perms[name]
        rng_s = np.random.default_rng(seed + 11)
        splits, all_splits = choose_splits(P, N, args.max_splits, rng_s)
        per_scene, per_split = ccgp_curve(X, design, L, E, splits, args.ridge_alpha)
        obs = per_scene[:, 0]
        null = per_scene[:, 1:]
        null_mean = np.nanmean(null, axis=0)
        t_obs = t_stat(obs - CHANCE)
        stat_obs = float(np.nanmean(obs) - CHANCE)
        null_stat = null_mean - CHANCE
        p_raw_mean = float((np.sum(null_stat >= stat_obs - 1e-12) + 1) / (len(null_stat) + 1))
        e["ccgp"] = {**summ_scene(obs, args.n_boot, seed), "per_scene": obs, "per_split": per_split[:, 0], "n_splits": len(splits), "all_splits_enumerated": all_splits,
                     "splits": [[design.conds[p], design.conds[n]] for p, n in splits]}
        e["perm"] = {"n_perm": int(null.shape[1]), "mode": perm_mode, "null_mean_of_means": float(null_mean.mean()), "null_q95_of_means": float(np.quantile(null_mean, 0.95)),
                     "p_raw": p_raw_mean, "stat_obs": stat_obs, "t_obs_descriptive": t_obs,
                     "statistic": "mean over scenes of the per-scene CCGP accuracy minus 0.5 (a t statistic is undefined when every scene scores 1); same splits and LOSO decoders under every permuted labeling",
                     "null": "within-scene label permutation of the eligible cells (or across-scene permutation for scene-constant labels)",
                     "_null_stat": null_stat}
        # shattering dimensionality (control) and the condition-shuffled dichotomy null at matched prevalence
        excl = {frozenset(P)}
        rd, enumerated = random_dichotomies(E, len(P), args.n_random_dichotomies, excl, np.random.default_rng(seed + 23))
        if rd:
            Lsd = np.stack([labels_for(design, pos, E) for pos in rd], axis=1)
            sd_ps = sd_curve(X, design, Lsd, E, args.ridge_alpha)  # [S, K]
            sd_scene = np.nanmean(sd_ps, axis=1)
            sd_dich = np.nanmean(sd_ps, axis=0)
            e["sd"] = {**summ_scene(sd_scene, args.n_boot, seed), "per_scene": sd_scene, "n_dichotomies": len(rd), "all_enumerated": enumerated, "dichotomy_means": sd_dich,
                       "definition": "mean held-out-scene decoding accuracy over random dichotomies of the eligible conditions with the same number of positive conditions"}
            diff = obs - sd_scene
            e["ccgp_minus_sd"] = {**boot_mean(diff, args.n_boot, seed), "t": t_stat(diff)}
            # CCGP protocol on the random dichotomies (condition-shuffled null)
            Kc = min(len(rd), args.n_random_ccgp)
            rc_scene = np.full((design.S, Kc), np.nan)
            rng_r = np.random.default_rng(seed + 31)
            for j in range(Kc):
                posj = rd[j]
                negj = [i for i in E if i not in set(posj)]
                spl, _ = choose_splits(posj, negj, args.max_splits_random, rng_r)
                ps_j, _ = ccgp_curve(X, design, Lsd[:, j : j + 1], E, spl, args.ridge_alpha)
                rc_scene[:, j] = ps_j[:, 0]
            rc_per_scene = np.nanmean(rc_scene, axis=1)
            rc_dich = np.nanmean(rc_scene, axis=0)
            e["ccgp_random"] = {**summ_scene(rc_per_scene, args.n_boot, seed), "per_scene": rc_per_scene, "n_dichotomies": Kc, "n_splits_each": args.max_splits_random,
                                "dichotomy_means": rc_dich, "percentile_of_ccgp": float(np.mean(rc_dich <= np.nanmean(obs))),
                                "definition": "CCGP (held-out condition per class, LOSO) of the random dichotomies at matched prevalence"}
            diff_r = obs - rc_per_scene
            e["ccgp_minus_random"] = {**boot_mean(diff_r, args.n_boot, seed), "t": t_stat(diff_r)}
        else:
            e["sd"] = {"mean": float("nan"), "n_dichotomies": 0}
            e["ccgp_minus_sd"] = {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
        # parallelism score
        pb = ps_best(mu, P, N, rng, d["matched_pairs"], args.n_random_pairings)
        pm = ps_matched(mu, d["matched_pairs"])
        # scene bootstrap of ps_matched (cheap) and ps_best (search each draw when small)
        rng_b = np.random.RandomState(seed)
        bm, bb = [], []
        for _ in range(args.n_boot_ps):
            idx = rng_b.randint(0, design.S, design.S)
            rows = np.concatenate([np.where(design.cell_scene == g)[0] for g in idx])
            mub = condition_means(X[rows], design.cell_cond[rows], design.C)
            bm.append(ps_matched(mub, d["matched_pairs"]))
            if math.perm(max(len(P), len(N)), min(len(P), len(N))) <= MAX_ENUM_PAIRINGS:
                bb.append(ps_best(mub, P, N, rng, d["matched_pairs"], 200)["value"])
        # permutation null for PS: within-scene shuffle of the condition labels of the eligible cells
        nm, nb = [], []
        rng_p = np.random.default_rng(seed + 41)
        elig = np.isin(design.cell_cond, E)
        for _ in range(args.n_perm_ps):
            cond_p = design.cell_cond.copy()
            for g in range(design.S):
                idx = np.where((design.cell_scene == g) & elig)[0]
                if len(idx) > 1:
                    cond_p[idx] = design.cell_cond[rng_p.permutation(idx)]
            mup = condition_means(X, cond_p, design.C)
            nm.append(ps_matched(mup, d["matched_pairs"]))
            nb.append(ps_best(mup, P, N, rng_p, d["matched_pairs"], 200)["value"])
        nm_a, nb_a = np.asarray(nm, dtype=np.float64), np.asarray(nb, dtype=np.float64)
        bm_a, bb_a = np.asarray(bm, dtype=np.float64), np.asarray(bb, dtype=np.float64)
        e["ps"] = {
            "best": {"value": pb["value"], "n_pairs": pb["n_pairs"], "method": pb["method"], "pairing": [[design.conds[a], design.conds[b]] for a, b in (pb.get("pairing") or [])],
                     "ci_low": float(np.nanpercentile(bb_a, 2.5)) if bb_a.size else None, "ci_high": float(np.nanpercentile(bb_a, 97.5)) if bb_a.size else None,
                     "null_mean": float(np.nanmean(nb_a)) if nb_a.size else None, "null_q95": float(np.nanquantile(nb_a, 0.95)) if nb_a.size else None,
                     "p_perm": float((np.sum(nb_a >= pb["value"] - 1e-12) + 1) / (nb_a.size + 1)) if nb_a.size and np.isfinite(pb["value"]) else None},
            "matched": {"value": pm, "n_pairs": len(d["matched_pairs"]), "pairs": [[design.conds[a], design.conds[b]] for a, b in d["matched_pairs"]], "note": d["partner_note"],
                        "ci_low": float(np.nanpercentile(bm_a, 2.5)) if bm_a.size else None, "ci_high": float(np.nanpercentile(bm_a, 97.5)) if bm_a.size else None,
                        "null_mean": float(np.nanmean(nm_a)) if nm_a.size else None, "null_q95": float(np.nanquantile(nm_a, 0.95)) if nm_a.size else None,
                        "p_perm": float((np.sum(nm_a >= pm - 1e-12) + 1) / (nm_a.size + 1)) if nm_a.size and np.isfinite(pm) else None},
            "definition": "mean pairwise cosine of coding vectors mu[p] - mu[n] over a one-to-one pairing of positive with negative conditions; best = max over pairings (Bernardi 2020), matched = the natural pairing; null = within-scene shuffle of the condition labels",
        }
        e["status"] = "ok"
        out["dichotomies"][name] = e
    in_lane = [lv for lv in (PED, OBJ, NULL) if lv in design.levels]
    out["did_parallelism"] = did_parallelism(mu, design, in_lane)
    return out


# --------------------------------------------------------------------------- #
# Families (max-T), verdicts, tables
# --------------------------------------------------------------------------- #


def family_maxt(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Westfall-Young max-T over the sites of each (arm, dichotomy, group, step) family with the SHARED within-scene
    label permutations; writes p_maxt_fwer and the verdict into every dichotomy block."""
    fams: dict[str, Any] = {}
    by: dict[tuple, list] = {}
    for e in entries:
        if e.get("status") != "ok":
            continue
        for name, dch in e["analysis"]["dichotomies"].items():
            if dch.get("status") == "ok":
                by.setdefault((e["arm"], name, e["group"], e["step"]), []).append((e, dch))
    for (arm, name, group, step), members in by.items():
        null = np.stack([dch["perm"]["_null_stat"] for _, dch in members], axis=1)  # [n_perm, n_sites]
        with np.errstate(invalid="ignore"):
            mx = np.nanmax(null, axis=1)
        fin = mx[np.isfinite(mx)]
        key = f"{arm}|{name}|{group}|s{step}"
        fams[key] = {"arm": arm, "dichotomy": name, "group": group, "step": step, "n_sites": len(members), "n_perm": int(null.shape[0]),
                     "maxt_null_quantiles": {q: float(np.quantile(fin, q)) for q in (0.5, 0.9, 0.95, 0.99)} if len(fin) else {}, "sites": {}}
        for e, dch in members:
            t_obs = dch["perm"]["stat_obs"]
            p_fwer = float((np.sum(fin >= t_obs - 1e-12) + 1) / (len(fin) + 1)) if np.isfinite(t_obs) and len(fin) else float("nan")
            dch["perm"]["p_maxt_fwer"] = p_fwer
            cms = dch.get("ccgp_minus_sd", {})
            excl = bool(cms.get("ci_low") is not None and np.isfinite(cms.get("ci_low", np.nan)) and cms["ci_low"] > 0)
            dch["gate"] = {"ccgp_ci_excludes_sd": excl, "perm_maxt_p_lt_0_05": bool(np.isfinite(p_fwer) and p_fwer < 0.05),
                           "ccgp_ci_excludes_random_ccgp": bool(dch.get("ccgp_minus_random", {}).get("ci_low") is not None and np.isfinite(dch["ccgp_minus_random"]["ci_low"]) and dch["ccgp_minus_random"]["ci_low"] > 0),
                           "rule": "PASS = scene-bootstrap CI of (CCGP - SD) above 0 AND within-scene label-permutation max-T p < 0.05 (family = sites of this dichotomy x group x step)"}
            dch["verdict"] = "PASS" if (excl and dch["gate"]["perm_maxt_p_lt_0_05"]) else "FAIL"
            fams[key]["sites"][e["site_id"]] = {"stat": t_obs, "p_raw": dch["perm"]["p_raw"], "p_maxt_fwer": p_fwer, "ccgp": dch["ccgp"]["mean"], "sd": dch["sd"]["mean"], "verdict": dch["verdict"]}
    return fams


def strip_private(entries: list[dict[str, Any]]) -> None:
    for e in entries:
        if e.get("status") == "ok":
            for dch in e["analysis"]["dichotomies"].values():
                dch.get("perm", {}).pop("_null_stat", None)


def _drop_private(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _drop_private(v) for k, v in obj.items() if not (isinstance(k, str) and k.startswith("_"))}
    if isinstance(obj, list):
        return [_drop_private(v) for v in obj]
    return obj


def gate_summary(entries: list[dict[str, Any]], dnames: list[str]) -> dict[str, Any]:
    s: dict[str, Any] = {}
    for name in dnames:
        s[name] = {}
        for arm in ("A", "B"):
            es = [(e, e["analysis"]["dichotomies"][name]) for e in entries if e.get("status") == "ok" and e["arm"] == arm and name in e["analysis"]["dichotomies"]]
            if not es:
                continue
            ok = [(e, d) for e, d in es if d.get("status") == "ok"]
            passes = [(e, d) for e, d in ok if d.get("verdict") == "PASS"]
            best = max(ok, key=lambda ed: ed[1]["ccgp"]["mean"]) if ok else None
            s[name][arm] = {"n_entries": len(es), "n_testable": len(ok), "n_pass": len(passes), "relational": name in RELATIONAL,
                            "not_testable_reason": next((d.get("reason") for _, d in es if d.get("status") != "ok"), None) if len(ok) < len(es) else None,
                            "ccgp_range": [min(d["ccgp"]["mean"] for _, d in ok), max(d["ccgp"]["mean"] for _, d in ok)] if ok else None,
                            "sd_range": [min(d["sd"]["mean"] for _, d in ok), max(d["sd"]["mean"] for _, d in ok)] if ok else None,
                            "ps_best_range": [min(d["ps"]["best"]["value"] for _, d in ok), max(d["ps"]["best"]["value"] for _, d in ok)] if ok else None,
                            "best_site": {"site_id": best[0]["site_id"], "group": best[0]["group"], "step": best[0]["step"], "ccgp": best[1]["ccgp"]["mean"], "ccgp_ci": [best[1]["ccgp"]["ci_low"], best[1]["ccgp"]["ci_high"]],
                                          "sd": best[1]["sd"]["mean"], "ccgp_random": best[1].get("ccgp_random", {}).get("mean"), "ps_best": best[1]["ps"]["best"]["value"], "ps_matched": best[1]["ps"]["matched"]["value"], "p_maxt": best[1]["perm"].get("p_maxt_fwer"), "verdict": best[1]["verdict"]} if best else None,
                            "pass_sites": [f"{e['site_id']}|{e['group']}|s{e['step']}" for e, _ in passes],
                            "verdict": ("PASS" if passes else "FAIL") if ok else "NOT_TESTABLE"}
    return s


def ranked_tables(entries: list[dict[str, Any]], dnames: list[str]) -> dict[str, Any]:
    tables: dict[str, Any] = {}
    for e in entries:
        if e.get("status") != "ok":
            continue
        row = {"site_id": e["site_id"], "group": e["group"], "step": e["step"], "n_scenes": e["n_scenes"]}
        for name in dnames:
            d = e["analysis"]["dichotomies"].get(name)
            if not d or d.get("status") != "ok":
                row[name] = None
                continue
            row[name] = {"ccgp": d["ccgp"]["mean"], "ccgp_ci": [d["ccgp"]["ci_low"], d["ccgp"]["ci_high"]], "sd": d["sd"]["mean"], "ccgp_minus_sd_ci": [d["ccgp_minus_sd"]["ci_low"], d["ccgp_minus_sd"]["ci_high"]],
                         "ccgp_random": d.get("ccgp_random", {}).get("mean"), "p_raw": d["perm"]["p_raw"], "p_maxt": d["perm"].get("p_maxt_fwer"), "ps_best": d["ps"]["best"]["value"], "ps_matched": d["ps"]["matched"]["value"],
                         "ps_matched_p": d["ps"]["matched"]["p_perm"], "verdict": d.get("verdict")}
        tables.setdefault(e["arm"], []).append(row)
    for arm, rows in tables.items():
        rows.sort(key=lambda r: -(r["solid_consequence"]["ccgp"] if r.get("solid_consequence") else -np.inf))
    return tables


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def run_arm(arm: str, dump: Path, args, factors_cache: dict[str, Any], t0: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    info = load_arm(dump, args.discovery_seeds)
    scenes, levels, status = complete_scenes(info["cells"])
    if len(scenes) < args.min_scenes:
        raise SystemExit(f"arm {arm}: only {len(scenes)} complete scenes (need >= {args.min_scenes})")
    factors, fmeta = scene_factors(args.stimulus, scenes)
    design = Design(info["cells"], scenes, levels, factors, args.n_bins, args.min_scenes_per_bin)
    D = dichotomies_for(arm, design)
    # shared within-scene label permutations per dichotomy (same for every site / group / step -> max-T valid)
    perms = {}
    for name, d in D.items():
        if len(d["pos_idx"]) >= 2 and len(d["neg_idx"]) >= 2:
            lab = labels_for(design, d["pos_idx"], d["eligible_idx"])
            perms[name] = label_permutations(lab, design.cell_scene, args.n_perm, np.random.default_rng(args.seed + 100 + sum(map(ord, name))))
    n_steps = info["n_steps"]
    steps = [s for s in (args.steps if args.steps is not None else range(n_steps)) if s < n_steps]
    token_sets = token_sets_for(args.stimulus, design.cells, tuple(args.groups), n_steps, info["offset"], args.domain)
    if args.sites:
        sites = list(args.sites)
    else:
        sites = sorted(p.stem for p in (dump / "activations").glob("*.npz") if "adaln" not in p.stem)
    entries: list[dict[str, Any]] = []
    rows = design.rows
    cells_local = [dict(c, row=i) for i, c in enumerate(design.cells)]  # pool_site indexes acts by c["row"]; acts are in design order
    for step in steps:
        for si, site in enumerate(sites):
            acts, tix = load_site(dump, site, rows)
            if acts.shape[2] == 1:
                if step == steps[0]:
                    entries.append({"arm": arm, "site_id": site, "status": "adaln_skipped"})
                continue
            for g in args.groups:
                entry: dict[str, Any] = {"arm": arm, "site_id": site, "step": step, "group": g}
                Pm, _ = pool_site(acts, tix, cells_local, token_sets, step, g)
                ok = np.all(np.isfinite(Pm), axis=1)
                if ok.sum() < len(cells_local):
                    bad_scenes = sorted({str(c["pair_id"]) for c, o in zip(design.cells, ok) if not o})
                    entry.update({"status": "no_tokens_some_scenes", "n_scenes_without_tokens": len(bad_scenes)})
                    if len(bad_scenes) > 0.2 * design.S:
                        entries.append(entry)
                        continue
                    # drop those scenes from a copy of the design
                    keep_scenes = [s for s in design.scenes if s not in set(bad_scenes)]
                    sub = Design([c for c in info["cells"] if str(c["pair_id"]) in set(keep_scenes)], keep_scenes, levels, factors, args.n_bins, args.min_scenes_per_bin)
                    Dsub = dichotomies_for(arm, sub)
                    perms_sub = {}
                    for name, d in Dsub.items():
                        if len(d["pos_idx"]) >= 2 and len(d["neg_idx"]) >= 2:
                            lab = labels_for(sub, d["pos_idx"], d["eligible_idx"])
                            perms_sub[name] = label_permutations(lab, sub.cell_scene, args.n_perm, np.random.default_rng(args.seed + 100 + sum(map(ord, name))))
                    pos = {int(c["row"]): i for i, c in enumerate(design.cells)}
                    Xsub = Pm[[pos[int(c["row"])] for c in sub.cells]]
                    entry["analysis"] = analyze_entry(Xsub.astype(np.float64), sub, Dsub, perms_sub, args, args.seed)
                    entry.update({"status": "ok", "n_scenes": sub.S, "scenes_note": f"{len(bad_scenes)} scenes without tokens in this group dropped"})
                else:
                    entry["analysis"] = analyze_entry(Pm.astype(np.float64), design, D, perms, args, args.seed)
                    entry.update({"status": "ok", "n_scenes": design.S})
                entries.append(entry)
            print(f"[arm {arm} step {step} {si + 1}/{len(sites)}] {site} ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
            if args.checkpoint is not None:
                args.checkpoint(arm, entries)
    arm_meta = {"dump": str(dump), "name": ARM_NAMES[arm], "solid_level": SOLID_LEVEL[arm], "ghost_level": GHOST_LEVEL[arm], "alignment": status, "design": design.describe(), "factors_meta": fmeta,
                "dichotomies": {n: {"description": d["description"], "relational": n in RELATIONAL, "n_conditions": len(d["eligible_idx"]), "n_pos": len(d["pos_idx"]), "n_neg": len(d["neg_idx"]),
                                    "n_matched_pairs": len(d["matched_pairs"]), "partner": d["partner_note"]} for n, d in D.items()},
                "n_steps": n_steps, "steps": steps, "sites": sites, "scene_factors": {s: factors[s] for s in scenes}}
    return entries, arm_meta


def run(args) -> dict[str, Any]:
    t0 = time.time()
    set_domain(args.domain)
    entries: list[dict[str, Any]] = []
    arms: dict[str, Any] = {}
    args.out.mkdir(parents=True, exist_ok=True)

    def checkpoint(arm: str, es_arm: list[dict[str, Any]]) -> None:
        """Partial ccgp.partial.json / .md after every site (families / verdicts over the entries so far; step-major order,
        so step-0 results are readable long before a multi-step run ends)."""
        es = entries + es_arm
        names = [n for e in es if e.get("status") == "ok" for n in e["analysis"]["dichotomies"]]
        dn = list(dict.fromkeys(names))
        fams = family_maxt(es)
        rep = {"protocol": PROTOCOL, "partial": True, "n_entries_so_far": len(es), "arm_in_progress": arm, "gate": {"summary": gate_summary(es, dn), "relational_dichotomies": list(RELATIONAL)},
               "families": fams, "ranked_tables": ranked_tables(es, dn), "entries": es, "elapsed_s": time.time() - t0, "groups": list(args.groups), "settings": {"n_perm": args.n_perm, "n_boot": args.n_boot},
               "arms": arms, "stimulus": str(args.stimulus), "interpretation_scope": INTERPRETATION_SCOPE}
        (args.out / "ccgp.partial.json").write_text(json.dumps(finite(_drop_private(rep)), indent=1) + "\n")

    args.checkpoint = checkpoint
    for arm, dump in (("A", args.dump_a), ("B", args.dump_b)):
        if dump is None:
            continue
        es, meta = run_arm(arm, dump, args, {}, t0)
        entries += es
        arms[arm] = meta
    dnames = []
    for meta in arms.values():
        for n in meta["dichotomies"]:
            if n not in dnames:
                dnames.append(n)
    fams = family_maxt(entries)
    strip_private(entries)
    gate = gate_summary(entries, dnames)
    tables = ranked_tables(entries, dnames)
    report = {
        "protocol": PROTOCOL, "stimulus": str(args.stimulus), "discovery_seeds": str(args.discovery_seeds) if args.discovery_seeds else None, "domain": args.domain,
        "token_group_source": alias_note(args.domain), "groups": list(args.groups),
        "settings": {"n_bins": args.n_bins, "min_scenes_per_bin": args.min_scenes_per_bin, "ridge_alpha": args.ridge_alpha, "max_splits": args.max_splits, "n_perm": args.n_perm, "n_boot": args.n_boot,
                     "n_random_dichotomies": args.n_random_dichotomies, "n_random_ccgp": args.n_random_ccgp, "max_splits_random": args.max_splits_random, "n_perm_ps": args.n_perm_ps, "n_boot_ps": args.n_boot_ps, "seed": args.seed},
        "definitions": {
            "condition": "(hazard level, action, binned scene factors); samples of a condition are whole scenes (one cell per scene)",
            "ccgp": "mean over held-out-condition splits (one positive and one negative condition held out, decoder trained on every cell of the other conditions) of the leave-one-scene-out balanced accuracy on the held-out conditions' cells; ridge decoder on class-balanced targets, unpenalised intercept, lambda = ridge_alpha x mean eigenvalue of the training scatter, midpoint threshold",
            "sd": "shattering dimensionality: mean LOSO decoding accuracy over random dichotomies of the same conditions at matched prevalence (every condition in training)",
            "ccgp_random": "CCGP protocol applied to the same random dichotomies (condition-shuffled null)",
            "ps": "parallelism score: mean pairwise cosine of coding vectors (condition-mean differences) over a one-to-one pairing; best = max over pairings; matched = natural pairing",
            "perm": "label permutation within scene clusters (across scenes for scene-constant labels), shared across sites; statistic = mean per-scene CCGP accuracy - 0.5; Westfall-Young max-statistic over the sites of a (dichotomy, group, step) family",
            "gate": "PASS = CI(CCGP - SD) > 0 AND max-T permutation p < 0.05",
            "unit": "scene (pair_id); every CI is a scene bootstrap of the mean",
        },
        "arms": arms, "families": fams, "gate": {"summary": gate, "relational_dichotomies": list(RELATIONAL)}, "ranked_tables": tables, "entries": entries,
        "runtime_s": time.time() - t0, "interpretation_scope": INTERPRETATION_SCOPE,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ccgp.json").write_text(json.dumps(finite(report), indent=1) + "\n")
    (args.out / "ccgp.md").write_text(markdown_summary(report))
    if (args.out / "ccgp.partial.json").exists():
        (args.out / "ccgp.partial.json").unlink()
    for name in dnames:
        for arm, s in gate.get(name, {}).items():
            b = s.get("best_site")
            print(f"{name:>18s} arm {arm}: {s['verdict']:>12s} pass {s['n_pass']}/{s['n_testable']} (of {s['n_entries']})" + (f" best {b['site_id']}|{b['group']}|s{b['step']} CCGP {b['ccgp']:.3f} [{b['ccgp_ci'][0]:.3f},{b['ccgp_ci'][1]:.3f}] SD {b['sd']:.3f} PS {b['ps_best']:.2f}/{b['ps_matched'] if b['ps_matched'] is None else round(b['ps_matched'], 2)} p_maxT {b['p_maxt']}" if b else ""))
    return report


def markdown_summary(report: dict[str, Any]) -> str:
    def f(x, nd=3):
        return "nan" if x is None or (isinstance(x, float) and not np.isfinite(x)) else (f"{x:.{nd}f}" if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x))

    lines = [f"# Abstraction / ontology gate: CCGP, PS, SD ({report['protocol']})", "", f"Scope: {report['interpretation_scope']}", "",
             f"Stimulus `{report['stimulus']}`; groups {', '.join(report['groups'])}; settings {json.dumps(report['settings'])}; runtime {report['runtime_s']:.0f} s.", ""]
    for arm, meta in report["arms"].items():
        des = meta["design"]
        lines += [f"## Arm {arm} ({meta['name']}) -- dump `{meta['dump']}`", "",
                  f"- {des['n_scenes']} discovery scenes, {des['n_cells']} cells, levels {des['levels']}, {des['n_conditions']} conditions {des['condition_tuple']}; binned factors {des['factor_names_binned'] or 'none (all constant)'}; steps {meta['steps']}; {len(meta['sites'])} sites",
                  "- binning: " + "; ".join(f"{k}: {v['n_bins']} bin(s) [{v['method']}] range {f(v['range'][0])}..{f(v['range'][1])}" for k, v in des["binning"].items()), ""]
        lines += ["| dichotomy | relational | conditions (pos/neg) | matched pairs | verdict | pass | best site | CCGP [CI] | SD | CCGP_rand | PS best / matched | p_maxT |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, dd in meta["dichotomies"].items():
            s = report["gate"]["summary"].get(name, {}).get(arm, {})
            b = s.get("best_site")
            lines.append(f"| {name} | {'yes' if dd['relational'] else ''} | {dd['n_conditions']} ({dd['n_pos']}/{dd['n_neg']}) | {dd['n_matched_pairs']} | **{s.get('verdict', 'n/a')}** | {s.get('n_pass', 0)}/{s.get('n_testable', 0)} | "
                         + (f"{b['site_id']}|{b['group']}|s{b['step']} | {f(b['ccgp'])} [{f(b['ccgp_ci'][0])}, {f(b['ccgp_ci'][1])}] | {f(b['sd'])} | {f(b.get('ccgp_random'))} | {f(b['ps_best'], 2)} / {f(b['ps_matched'], 2)} | {f(b['p_maxt'], 4)} |" if b else f"{s.get('not_testable_reason', '')} | | | | | |"))
        lines.append("")
    lines += ["## Per-site tables (sorted by solid_consequence CCGP)", ""]
    for arm, rows in report["ranked_tables"].items():
        dn = [n for n in report["arms"][arm]["dichotomies"]]
        lines += [f"### Arm {arm}", "", "| site | group | step | " + " | ".join(f"{n}: CCGP / SD / PSm / p_maxT / verdict" for n in dn) + " |", "|---|---|---|" + "---|" * len(dn)]
        for r in rows:
            cells = []
            for n in dn:
                d = r.get(n)
                cells.append("n/t" if not d else f"{f(d['ccgp'])} / {f(d['sd'])} / {f(d['ps_matched'], 2)} / {f(d['p_maxt'], 3)} / {d['verdict']}")
            lines.append(f"| {r['site_id']} | {r['group']} | {r['step']} | " + " | ".join(cells) + " |")
        lines.append("")
    if "self_test" in report:
        lines += ["## Self-test", "", "```", json.dumps(finite(report["self_test"]), indent=1), "```"]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Synthetic self-test
# --------------------------------------------------------------------------- #


def synthetic_dumps(root: Path, *, code: str, n_scenes: int = 24, d: int = 64, sites=("L03.mlp_out", "L05.resid_post"), seed: int = 0, noise: float = 0.25, amp: float = 2.0) -> tuple[Path, Path, Path]:
    """Two driving dumps (layout of ``localize_interaction.py --dump-groups``) with a two-level hazard distance (8 / 10 m,
    v0.9-style factor keys) in the manifest.  ``code == "abstract"``: every variable (in-lane, action, pedestrian identity, the arm's solid identity,
    distance) adds a FIXED direction (parallel coding vectors, cube-like geometry).  ``code == "xor"``: each variable adds
    a direction that depends on the full context (the other coordinates of the condition), so the variable is decodable
    within every condition but its coding direction is orthogonal across conditions (non-abstract, high SD)."""
    rng = np.random.default_rng(seed)
    stim, dA, dB = root / "stimulus", root / "arm_A", root / "arm_B"
    (stim / "masks").mkdir(parents=True, exist_ok=True)
    for p in (dA, dB):
        (p / "activations").mkdir(parents=True, exist_ok=True)
    U = _orthonormal(rng, d, 5)
    u_in, u_act, u_ped, u_solid, u_dist = U
    ctx_dirs: dict[tuple, np.ndarray] = {}

    def ctx_dir(var: str, ctx: tuple) -> np.ndarray:
        """One random unit direction per (variable, context); contexts are exactly the design conditions' other coordinates."""
        key = (var, ctx)
        if key not in ctx_dirs:
            v = rng.normal(size=d)
            ctx_dirs[key] = v / np.linalg.norm(v)
        return ctx_dirs[key]

    E = rng.normal(size=(256, d)) * 1.0
    cells, rows, tix = [], [], []
    acts = {"A": [], "B": []}
    yy, xx = np.mgrid[:256, :256]
    for s in range(n_scenes):
        seed_id = 5000 + s
        pid = f"drive_seed{seed_id:06d}"
        dbin = s % 2  # two discrete hazard distances (8 m / 10 m) -> the design's "distinct_values" binning reproduces them exactly
        dist, pref, lat = float(8.0 + 2.0 * dbin), 0.2, 0.0
        content = rng.normal(size=(256, d)) * 0.5
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
        for lv, a in keys:
            cid = f"{pid}__h{lv}a{a}"
            cells.append({"row": len(cells), "pair_id": pid, "seed": seed_id, "hazard": lv, "candidate_action": a, "cell_id": cid, "artifact": f"cells/{cid}.npz"})
            rows.append({"cell_id": cid, "pair_id": pid, "seed": seed_id, "hazard": lv, "candidate_action": a, "artifact": f"cells/{cid}.npz", "hazard_dist_m": dist, "prefix_throttle": pref,
                         "hazard_lateral_offset_m": lat, "randomized_factors": {"dist": dist}, "protocol_version": "cgs-metadrive-pilot-v0.9"})
            for arm in ("A", "B"):
                solid = SOLID_LEVEL[arm]
                x = E[tok_union] + content[tok_union]
                inlane, act, ped, sol = int(lv in (PED, OBJ)), int(a == 1), int(lv == PED), int(lv == solid)
                if code == "abstract":
                    off = amp * (inlane * u_in + act * u_act + ped * u_ped + sol * u_solid + (2 * dbin - 1) * u_dist)
                else:  # every variable carried by a direction unique to its context (the other coordinates of the condition)
                    off = amp * (inlane * ctx_dir("in", (lv, a, dbin)) + act * ctx_dir("act", (lv, dbin)) + ped * ctx_dir("ped", (a, dbin)) + sol * ctx_dir(f"sol{arm}", (a, dbin))
                                 + ctx_dir("dist", (lv, a, dbin))) + 0.5 * amp * ctx_dir("cond", (lv, a, dbin))
                x = x + off[None, :] + rng.normal(size=x.shape) * noise
                acts[arm].append(x[None])
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
                                                                     "synthetic": {"code": code, "d": d, "noise": noise, "amp": amp, "n_scenes": n_scenes}}, indent=1))
    (stim / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return stim, dA, dB


def self_test_verdict(report: dict[str, Any], code: str) -> dict[str, Any]:
    es = [e for e in report["entries"] if e.get("status") == "ok"]
    names = ("solid_consequence", "inlane", "action", "identity")
    vals = {n: {"ccgp": [], "sd": [], "ps_best": [], "ps_matched": [], "verdict": []} for n in names}
    for e in es:
        for n in names:
            d = e["analysis"]["dichotomies"].get(n, {})
            if d.get("status") != "ok":
                continue
            vals[n]["ccgp"].append(d["ccgp"]["mean"])
            vals[n]["sd"].append(d["sd"]["mean"])
            vals[n]["ps_best"].append(d["ps"]["best"]["value"])
            vals[n]["ps_matched"].append(d["ps"]["matched"]["value"])
            vals[n]["verdict"].append(d["verdict"])
    rel = vals["solid_consequence"]
    if code == "abstract":
        checks = {"ccgp_planted_ge_0.9": bool(rel["ccgp"]) and all(min(vals[n]["ccgp"]) >= 0.9 for n in names),
                  "ps_best_planted_ge_0.9": all(min(vals[n]["ps_best"]) >= 0.9 for n in ("solid_consequence", "action", "identity")),
                  "ps_matched_planted_ge_0.9": all(min(vals[n]["ps_matched"]) >= 0.9 for n in ("solid_consequence", "action", "identity")),
                  "relational_gate_pass_everywhere": bool(rel["verdict"]) and all(v == "PASS" for v in rel["verdict"]),
                  "ccgp_above_sd": all(c > s for c, s in zip(rel["ccgp"], rel["sd"]))}
    else:
        checks = {"sd_high_ge_0.85": bool(rel["sd"]) and min(rel["sd"]) >= 0.85,
                  "ccgp_near_chance_le_0.65": bool(rel["ccgp"]) and max(rel["ccgp"]) <= 0.65 and max(vals["inlane"]["ccgp"]) <= 0.65,
                  "ps_best_low_le_0.4": max(rel["ps_best"]) <= 0.4,
                  "relational_gate_fail_everywhere": bool(rel["verdict"]) and all(v == "FAIL" for v in rel["verdict"])}
    return {"code": code, "checks": checks, "passed": all(checks.values()), "n_entries": len(es),
            "ranges": {n: {k: ([min(v), max(v)] if v and k != "verdict" else v) for k, v in vals[n].items()} for n in names}}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dump-a", type=Path, default=None, help="localization dump of arm A (pedestrian solid / cone ghost)")
    ap.add_argument("--dump-b", type=Path, default=None, help="localization dump of arm B (pedestrian ghost / cone solid)")
    ap.add_argument("--stimulus", type=Path, default=None, help="merged stimulus dir (manifest.jsonl + masks/), e.g. drive_factorial_merged/armA")
    ap.add_argument("--discovery-seeds", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--domain", default="driving")
    ap.add_argument("--sites", nargs="*", default=None)
    ap.add_argument("--steps", type=int, nargs="*", default=None, help="imagined steps (default: all in the dump)")
    ap.add_argument("--groups", nargs="*", default=list(DEFAULT_GROUPS))
    ap.add_argument("--n-bins", type=int, default=2, help="bins per varied scene factor (quantile; distinct values when <= n-bins)")
    ap.add_argument("--min-scenes-per-bin", type=int, default=4)
    ap.add_argument("--min-scenes", type=int, default=8)
    ap.add_argument("--ridge-alpha", type=float, default=1.0, help="ridge lambda = alpha x mean eigenvalue of the training scatter")
    ap.add_argument("--max-splits", type=int, default=64, help="held-out (positive, negative) condition pairs per dichotomy (all when fewer)")
    ap.add_argument("--n-perm", type=int, default=200, help="within-scene label permutations (shared across sites for max-T)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-random-dichotomies", type=int, default=200, help="random dichotomies at matched prevalence for SD (all when fewer exist)")
    ap.add_argument("--n-random-ccgp", type=int, default=30, help="how many of them also get the CCGP protocol")
    ap.add_argument("--max-splits-random", type=int, default=8)
    ap.add_argument("--n-random-pairings", type=int, default=2000)
    ap.add_argument("--n-perm-ps", type=int, default=100)
    ap.add_argument("--n-boot-ps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-test", action="store_true", help="planted abstract / XOR synthetic dumps under <out>/self_test")
    ap.add_argument("--synthetic-scenes", type=int, default=24)
    ap.add_argument("--synthetic-dim", type=int, default=64)
    ap.add_argument("--synthetic-noise", type=float, default=0.25)
    ap.set_defaults(checkpoint=None)
    return ap


def main(argv: list[str] | None = None) -> dict[str, Any]:
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*(empty slice|All-NaN).*")
    args = build_parser().parse_args(argv)
    if args.self_test:
        verdicts, reports = {}, {}
        for code in ("abstract", "xor"):
            root = args.out / "self_test" / code
            stim, dA, dB = synthetic_dumps(root, code=code, n_scenes=args.synthetic_scenes, d=args.synthetic_dim, seed=args.seed, noise=args.synthetic_noise)
            a = argparse.Namespace(**vars(args))
            a.dump_a, a.dump_b, a.stimulus, a.out, a.discovery_seeds = dA, dB, stim, root / "out", None
            a.groups = [g for g in args.groups if g in ("hazard", "corridor", "hazard_corridor")]
            a.n_perm, a.n_boot, a.n_random_dichotomies, a.n_random_ccgp, a.max_splits = min(args.n_perm, 60), min(args.n_boot, 200), min(args.n_random_dichotomies, 40), min(args.n_random_ccgp, 20), min(args.max_splits, 16)
            a.n_perm_ps, a.n_boot_ps, a.max_splits_random = min(args.n_perm_ps, 20), min(args.n_boot_ps, 20), min(args.max_splits_random, 4)
            a.min_scenes_per_bin = 3
            rep = run(a)
            verdicts[code] = self_test_verdict(rep, code)
            rep["self_test"] = verdicts[code]
            (a.out / "ccgp.json").write_text(json.dumps(finite(rep), indent=1) + "\n")
            (a.out / "ccgp.md").write_text(markdown_summary(rep))
            reports[code] = rep
            print(json.dumps(finite(verdicts[code])))
        passed = all(v["passed"] for v in verdicts.values())
        (args.out / "self_test" / "verdict.json").write_text(json.dumps(finite({"passed": passed, **verdicts}), indent=1) + "\n")
        if not passed:
            raise SystemExit(f"self-test FAILED: {json.dumps(finite(verdicts))}")
        print("SELF_TEST_PASSED")
        return {"self_test": verdicts, "reports": reports}
    if not ((args.dump_a or args.dump_b) and args.stimulus):
        raise SystemExit("--dump-a and/or --dump-b plus --stimulus are required (or --self-test)")
    return run(args)


if __name__ == "__main__":
    main()
