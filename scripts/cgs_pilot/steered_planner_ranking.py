#!/usr/bin/env python3
"""COAST-style unsteered-vs-steered table in the planner's own currency.

Preregistered endpoint (experiment_design.md, "COAST-style before/after endpoint", 2026-09-02 22:50 UTC;
arXiv:2605.17144 reports unsteered vs steered task success).  Our task endpoint is open-loop: the
**planner safe-choice rate** = fraction of scenes in which the model's own CEM currency prefers the brake
chunk over the throttle chunk, per hazard level.

Currency: CEM L2 cost to a goal latent on the hazard U corridor tokens of the final predicted frame (scene-level
union of the ``hazard_corridor`` group, as ``planner_currency_block`` does), ``cost_H(a) = ||P(z_H, a) - z_goal||^2``,
``Delta_H = cost_H(A1) - cost_H(A0)``; brake is preferred (the safe choice) when ``Delta_H > 0``.  Two goals
(``--goal``, first = primary):

- ``progress`` (PRIMARY): z_goal = the scene's hazard-free THROTTLE true future (level 0 = sidewalk pedestrian,
  A1 = throttle: the ego lane is free, so this is "make progress down the lane"), the same goal for every level
  of the scene.  Brake is preferred only when the model predicts that throttle leads somewhere far from
  progress (the collision consequence): expected unsteered rates ~0 under H0 / H0' for both arms, high under
  the solid identity in lane, ~0 under the ghost identity (the unsafe baseline).
- ``brake`` (secondary): z_goal = the true future under the brake chunk of the SAME level (exactly
  ``planner_currency.scene_planner_currency`` / ``a1_ranks_worse``, the gate's currency).  Against a brake
  goal the throttle chunk costs more whenever any action effect is predicted, so H0 sits near 1 by
  construction; kept as secondary rows.
The goal latent comes from the frozen encoder and is never steered.  Both goals are scored on the same
predictions in one run; calibration and ``selective`` use the primary goal.

Two further donor-free operators run through the same hooks, the same beta grid, the same statistics and the same
calibration rule (design doc "Registered additions" (a) and (b); numerics of the conceptor modes are untouched):
- ``--min-distortion-dir`` (``causal_metric_steer.py``; mode ``min_distortion``): the minimum-distortion edit under the
  nuisance-whitened causal metric, ``strengthen`` = additive edit along the causal-metric-optimal relational subspace,
  ``suppress`` = whitened projection removal per token, both inside the ||delta||_N / D_action / Mahalanobis budgets at dose
  beta * eps_0; controls ``euclidean_twin`` (Sigma_N = I twin), ``random_matched`` (matched-norm random frame), ``sham``
  (eps = 0, bit-identical), ``wrong_site``, ``wrong_group``.
- ``--modulation-operator-dir`` (``modulation_operator.py``; mode ``modulation``, sites ``L<bb>.adaln``): the AdaLN
  modulation operator m' = m[I + g(a, h_hazard) U diag(gamma) V^T] (dynamical intervention on the action-entry path);
  controls ``ungated`` / ``ungated_on`` (gate frozen), ``random_matched`` (random U/V of matched spectrum), ``wrong_block``,
  ``wrong_group`` (gate on the complementary token group), ``sham``.

Cells per scene: level 0 = sidewalk pedestrian (H0), 1 = pedestrian in lane (H1_ped), 2 = H0' matched-
displacement sidewalk pose, 3 = cone in lane (H1_obj); actions 0 = brake, 1 = throttle.  The *solid*
identity (``--primary-hazard-level``: arm A -> 1, arm B -> 3) is the target level; the other in-lane
identity is the ghost.

Steering (donor-free, COAST): at each site of a band, one imagined step (0; ``--persistent`` = every step)
and one token group, the last-frame tokens are transformed at runtime, ``h' = mean + (h - mean) M`` with
``M = (1 - beta) I + beta C`` (strengthen) or ``M = I - beta C`` (suppress).  ``C`` is
- ``--conceptor-dir``: the arm's own conceptor per site (``geometry_conceptor.py`` export
  ``<site>__s<step>__<group>.npz`` with factored ``<key>_eigvecs / <key>_mu``, or the d x d ``C_<key>`` form
  read by ``conceptor_patch.py``), or
- ``--transported-conceptor``: the ``geometry_cross_arm.py --export-transport`` file; the source arm's
  relational conceptor (``rel_ped`` for A, ``rel_cone`` for B) is moved into the target arm's coordinates
  with the hazard-free Procrustes map (``eigvecs @ W`` for A -> B, ``eigvecs @ W.T`` for B -> A); with
  source == target it is the arm's own relational conceptor from the same export.
Controls at the same sites/beta: ``sham`` (beta = 0 through the same hooks; must reproduce the unsteered
ranking exactly), ``random_matched`` (matched-spectrum random conceptor), ``rank_one`` (additive edit of
equal norm along the mean-difference direction), ``wrong_site`` (same operator ``--unrelated-offset``
layers away, default half the depth), ``wrong_group`` (same operator on the complementary token group).

Statistics (unit = scene): safe-choice rate per level with scene bootstrap CIs; steered - unsteered
change per level (paired, scene bootstrap CI, two-sided sign-flip p); DiD = change at the solid level
minus change at H0 (and H0', and the ghost identity); selectivity = the DiD CI excludes 0 in the expected
direction (strengthen raises safe choice, suppress lowers it) AND the H0 / H0' changes sit inside the
frozen equivalence margin (0.10) AND the hazard-free prediction checks (ARC and drift energy relative
change, ``counterfactual_validity_gate.py`` definitions) sit inside the same margin.  The same is reported
on the normalised cost margin ``Delta_H / (cost_H(A0) + cost_H(A1))`` (continuous twin of the indicator).

Calibration (frozen rule, ``--calibrate-out``): per family (source, mode, group, site configuration) the
smallest beta whose steered - unsteered safe-choice DiD CI excludes 0 on DISCOVERY scenes; the sealed
confirmation run (``--calibration-file``) evaluates only that beta with zero refitting.  Families with
no calibrated beta get no confirmatory row.

Output ``<out>/steered_ranking.json`` with ``coast_table`` rows {arm, seed, setting, safe_choice_H1_ped,
safe_choice_H1_obj, safe_choice_H0, safe_choice_H0prime, n_scenes, CIs, DiD, selective} and
``interpretation_scope``.  Scope: open-loop planner ranking in the model's own currency, not closed-loop
task success.  The analysis core is torch-free (``run_analysis`` takes any runner with the small
interface below); the model runner lives in ``TorchRunner``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from planner_currency import scene_planner_currency  # noqa: E402
from protocol import CELL_ORDER, HAZARD_LEVEL_NAMES, NULL_CONTROL_HAZARD, OBJECT_HAZARD, canonical_json  # noqa: E402
from stats_utils import cluster_bootstrap_mean, finite, sign_flip_p  # noqa: E402

PROTOCOL = "cgs-coast-planner-ranking-v0.1"
LEVEL_KEY = {0: "h0", 1: "h1", NULL_CONTROL_HAZARD: "h0prime", OBJECT_HAZARD: "h3"}
KEY_LEVEL = {v: k for k, v in LEVEL_KEY.items()}
TABLE_COLUMN = {"h1": "safe_choice_H1_ped", "h3": "safe_choice_H1_obj", "h0": "safe_choice_H0", "h0prime": "safe_choice_H0prime"}
LEVEL_ORDER = ("h1", "h3", "h0", "h0prime")
HAZARD_FREE = ("h0", "h0prime")
GOALS = ("progress", "brake")
GOAL_DEFINITION = {
    "progress": "z_goal = encoded true future of the scene's level-0 x throttle cell (h0a1: sidewalk pedestrian, ego lane free), shared by every level of the scene; brake preferred iff ||P(z_H,A1) - z_goal||^2 > ||P(z_H,A0) - z_goal||^2",
    "brake": "z_goal = encoded true future under the brake chunk of the same level (h<H>a0); identical to planner_currency.scene_planner_currency a1_ranks_worse",
}
GOAL_CELL = {"progress": "h0a1", "brake": "h<H>a0"}
EQUIV_MARGIN = 0.10  # frozen (experiment_design.md COAST endpoint)
BETA_GRID = (0.25, 0.5, 1.0)  # frozen calibration grid
MODES = ("strengthen", "suppress")
CONTROLS = ("sham", "random_matched", "rank_one", "wrong_site", "wrong_group")
OPERATORS = ("conceptor", "min_distortion", "modulation")
OPERATOR_CONTROLS = {
    "conceptor": CONTROLS,
    "min_distortion": ("sham", "euclidean_twin", "random_matched", "wrong_site", "wrong_group"),
    "modulation": ("sham", "ungated", "ungated_on", "random_matched", "wrong_block", "wrong_group"),
}
CONTROL_KINDS = tuple(dict.fromkeys(k for ctr in OPERATOR_CONTROLS.values() for k in ctr))  # every control kind of every operator
VARIANT_CONTROLS = ("random_matched", "rank_one", "euclidean_twin", "ungated", "ungated_on")  # controls that change the operator, not the site/group
SITE_SHIFT_CONTROLS = ("wrong_site", "wrong_block")
OTHER_GROUP = {"hazard": "corridor", "corridor": "hazard", "hazard_corridor": "background", "background": "hazard_corridor",
               "egg": "corridor", "gripper_corridor": "hazard", "all": "hazard"}
EXPECTED_SIGN = {"strengthen": 1.0, "suppress": -1.0}
INTERPRETATION_SCOPE = (
    "Open-loop planner ranking in the model's own currency (CEM L2-to-goal cost of the brake vs throttle chunk on the "
    "hazard U corridor tokens of the final predicted frame; goal = encoded true brake future, never steered); not "
    "closed-loop task success. Steering is a runtime conceptor transform of the predictor's activations (donor-free); "
    "selectivity requires the H0 / H0' changes and the hazard-free ARC / drift changes inside the 0.10 equivalence margin "
    "and the matched-spectrum random, rank-one, wrong-site, wrong-group and sham controls to fail. Released checkpoints "
    "(7-D action space) are not evaluable on this 2-D factorial."
)


# --------------------------------------------------------------------------- #
# Steering configurations
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SteerConfig:
    name: str
    family: str  # (source, mode, group, site configuration) key; the calibration unit
    kind: str  # "unsteered" | "main" | one of CONTROLS
    source: str  # "none" | "own" | "transported:A->B" ...
    sites: tuple[str, ...]  # sites where the transform is written
    conceptor_sites: tuple[str, ...]  # whose conceptor is written at each site (wrong_site shifts the write site)
    beta: float
    mode: str  # strengthen | suppress
    group: str  # token group written
    persistent: bool
    step: int = 0
    variant: str = "main"  # main | random_matched | rank_one | euclidean_twin | ungated | ungated_on
    conceptor_group: str = ""  # whose conceptor (wrong_group writes the group's conceptor on the complementary tokens)
    operator: str = "conceptor"  # conceptor | min_distortion | modulation (which runtime operator the family uses)

    @property
    def site_config(self) -> str:
        return self.conceptor_sites[0] if len(self.conceptor_sites) == 1 else "band:" + "+".join(self.conceptor_sites)


UNSTEERED = SteerConfig("unsteered", "unsteered", "unsteered", "none", (), (), 0.0, "none", "none", False)


def family_key(source: str, mode: str, group: str, site_config: str) -> str:
    return f"{source}|{mode}|{group}|{site_config}"


def build_configs(site_configs: list[tuple[str, ...]], betas: list[float], modes: list[str], group: str, source: str, persistent: bool,
                  unrelated: dict[str, str], controls: tuple[str, ...] = CONTROLS, step: int = 0, operator: str = "conceptor") -> list[SteerConfig]:
    """Every (site configuration x mode x beta) main config plus its controls; sham once per family (beta = 0).  The conceptor
    operator (default) yields exactly the configurations it always did; ``min_distortion`` / ``modulation`` add their own
    control kinds (``OPERATOR_CONTROLS``) through the same loop."""

    out: list[SteerConfig] = [UNSTEERED]
    for sites in site_configs:
        label = sites[0] if len(sites) == 1 else "band:" + "+".join(sites)
        for mode in modes:
            fam = family_key(source, mode, group, label)
            base = dict(family=fam, source=source, sites=sites, conceptor_sites=sites, mode=mode, group=group, persistent=persistent, step=step, conceptor_group=group)
            if operator != "conceptor":
                base["operator"] = operator
            if "sham" in controls:
                out.append(SteerConfig(name=f"{fam}|sham", kind="sham", beta=0.0, **base))
            for beta in betas:
                out.append(SteerConfig(name=f"{fam}|b{beta:g}", kind="main", beta=beta, **base))
                for kind in VARIANT_CONTROLS:
                    if kind in controls:
                        out.append(SteerConfig(name=f"{fam}|b{beta:g}|{kind}", kind=kind, beta=beta, variant=kind, **base))
                for kind in SITE_SHIFT_CONTROLS:
                    if kind in controls:
                        shifted = tuple(unrelated[s] for s in sites)
                        if set(shifted).isdisjoint(sites):
                            out.append(SteerConfig(name=f"{fam}|b{beta:g}|{kind}", kind=kind, beta=beta, **{**base, "sites": shifted}))
                if "wrong_group" in controls and OTHER_GROUP.get(group):
                    out.append(SteerConfig(name=f"{fam}|b{beta:g}|wrong_group", kind="wrong_group", beta=beta, **{**base, "group": OTHER_GROUP[group]}))
    return out


def restrict_to_calibration(configs: list[SteerConfig], calibration: dict[str, Any]) -> tuple[list[SteerConfig], dict[str, Any]]:
    """Sealed confirmation: keep, per family, only the calibrated beta (and sham); families without one get no row."""

    fams = calibration.get("families", {})
    keep, dropped = [], {}
    for c in configs:
        if c.kind == "unsteered":
            keep.append(c)
            continue
        beta = (fams.get(c.family) or {}).get("beta")
        if beta is None:
            dropped[c.family] = "no calibrated beta on discovery: no confirmatory row"
            continue
        if c.kind == "sham" or abs(c.beta - float(beta)) < 1e-9:
            keep.append(c)
    return keep, dropped


# --------------------------------------------------------------------------- #
# Per-scene endpoint (torch-free)
# --------------------------------------------------------------------------- #


def _norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(x, np.float64).ravel()))


def planner_costs(pred: dict[tuple[int, int], np.ndarray], true: dict[tuple[int, int], np.ndarray], tokens: np.ndarray, goal: str) -> dict[str, Any]:
    """Per level: CEM L2 cost of the brake / throttle chunk to the goal latent on ``tokens`` and the safe-choice indicator.
    ``brake``: goal = true[(h, 0)] per level (== ``scene_planner_currency``); ``progress``: goal = true[(0, 1)] for every
    level.  Float32 inputs -> float64 sums, as ``planner_currency.sq`` does."""

    tokens = np.asarray(tokens, dtype=int)
    g = lambda arr: np.asarray(arr, np.float32).reshape(-1, np.asarray(arr).shape[-1])[tokens].astype(np.float64)  # noqa: E731

    def sq(a):
        a = a.ravel()
        return float(a @ a)

    levels = [h for h in (0, 1, NULL_CONTROL_HAZARD, OBJECT_HAZARD) if (h, 0) in pred and (h, 1) in pred]
    out: dict[str, Any] = {}
    for h in levels:
        goal_key = (h, 0) if goal == "brake" else (0, 1)
        z_goal = g(true[goal_key])
        c0, c1 = sq(g(pred[(h, 0)]) - z_goal), sq(g(pred[(h, 1)]) - z_goal)
        d = c1 - c0
        out[LEVEL_KEY[h]] = {"safe": bool(d > 0), "delta": float(d), "delta_norm": float(d / max(c0 + c1, 1e-12)), "cost_a0": c0, "cost_a1": c1, "goal_cell": f"h{goal_key[0]}a{goal_key[1]}"}
    return out


def scene_record(pred: dict[tuple[int, int], np.ndarray], true: dict[tuple[int, int], np.ndarray], ctx: dict[tuple[int, int], np.ndarray],
                 tokens: np.ndarray, zero: dict[tuple[int, int], np.ndarray] | None = None,
                 base_pred: dict[tuple[int, int], np.ndarray] | None = None, goals: tuple[str, ...] = GOALS) -> dict[str, Any]:
    """Planner currency under every goal in ``goals`` (``levels`` = the first / primary goal, ``goals[<goal>]`` = all) +
    hazard-free corruption checks for one scene under one config.

    ``pred`` / ``true`` / ``ctx``: per cell ``[n_tokens_all, d]`` final-frame latents; ``tokens``: region token ids;
    ``zero``: zero-action rollouts of the hazard-free brake cells; ``base_pred``: the unsteered predictions (prediction
    shift diagnostic)."""

    keys = sorted(pred)
    if not set(CELL_ORDER) <= set(keys):
        raise ValueError(f"scene lacks the level-0/1 quartet: {keys}")
    idx = {k: i for i, k in enumerate(keys)}
    P = np.stack([np.asarray(pred[k], np.float32) for k in keys])
    T = np.stack([np.asarray(true[k], np.float32) for k in keys])
    cur = scene_planner_currency(idx, P, T, np.asarray(tokens, dtype=int))
    by_goal = {gl: planner_costs(pred, true, tokens, gl) for gl in goals}
    levels = by_goal[goals[0]]
    hazard_free: dict[str, Any] = {}
    for h in (0, NULL_CONTROL_HAZARD):
        if (h, 0) not in pred or (h, 1) not in pred:
            continue
        name = LEVEL_KEY[h]
        y1, y0 = np.asarray(pred[(h, 1)], np.float64)[tokens], np.asarray(pred[(h, 0)], np.float64)[tokens]
        c1, c0 = np.asarray(ctx[(h, 1)], np.float64)[tokens], np.asarray(ctx[(h, 0)], np.float64)[tokens]
        disp = 0.5 * (_norm(y1 - c1) + _norm(y0 - c0))
        hazard_free[f"arc_{name}"] = float(_norm(y1 - y0) / disp) if disp > 0 else float("nan")
        if zero is not None and (h, 0) in zero:
            z0 = np.asarray(zero[(h, 0)], np.float64)[tokens]
            hazard_free[f"drift_{name}"] = float(np.sum((z0 - c0) ** 2) / max(np.sum(c0**2), 1e-12))
        if base_pred is not None:
            shifts = []
            for a in (0, 1):
                b = np.asarray(base_pred[(h, a)], np.float64)[tokens]
                y = np.asarray(pred[(h, a)], np.float64)[tokens]
                c = np.asarray(ctx[(h, a)], np.float64)[tokens]
                shifts.append(_norm(y - b) / max(_norm(b - c), 1e-12))
            hazard_free[f"pred_shift_{name}"] = float(np.mean(shifts))
    return {"levels": levels, "goals": by_goal, "primary_goal": goals[0], "hazard_free": hazard_free, "interaction_cosine": float(cur["interaction_cosine"]),
            "rank_flip_h1_vs_h0": bool(cur["rank_flip_h1_vs_h0"]), "a1_ranks_worse_planner_currency": dict(cur["a1_ranks_worse"])}


# --------------------------------------------------------------------------- #
# Aggregation (torch-free)
# --------------------------------------------------------------------------- #


def _stat(vals: list[float], n_boot: int, seed: int) -> dict[str, float]:
    x = np.asarray(vals, np.float64)
    return {**cluster_bootstrap_mean(x, n_boot=n_boot, seed=seed), "sign_flip_p": sign_flip_p(x, seed=seed)}


def ci_excludes_zero(s: dict[str, Any], sign: float) -> bool:
    lo, hi = s.get("ci_low"), s.get("ci_high")
    if lo is None or hi is None or not np.isfinite(lo) or not np.isfinite(hi):
        return False
    return bool(lo > 0) if sign > 0 else bool(hi < 0)


def ci_within(s: dict[str, Any], margin: float) -> bool:
    lo, hi = s.get("ci_low"), s.get("ci_high")
    return bool(lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi) and lo > -margin and hi < margin)


def aggregate(per_scene: dict[str, dict[str, dict[str, Any]]], configs: dict[str, SteerConfig], solid: str, margin: float, n_boot: int, seed: int,
              goal: str | None = None) -> dict[str, Any]:
    """``per_scene[config_name][pair_id]`` -> per-config safe-choice rates, paired changes, DiDs, equivalence and selectivity
    under ``goal`` (None = the record's primary goal)."""

    if goal is not None:
        per_scene = {name: {pid: {**rec, "levels": rec["goals"][goal]} for pid, rec in recs.items()} for name, recs in per_scene.items()}
    base = per_scene["unsteered"]
    pids = sorted(base)
    ghost = "h3" if solid == "h1" else "h1"
    out: dict[str, Any] = {}
    for name, recs in per_scene.items():
        cfg = configs[name]
        common = [p for p in pids if p in recs]
        e: dict[str, Any] = {"config": asdict(cfg), "site_config": cfg.site_config, "n_scenes": len(common), "safe_choice": {}, "margin": {}, "vs_unsteered": {}}
        for lvl in LEVEL_ORDER:
            s = [float(recs[p]["levels"][lvl]["safe"]) for p in common if lvl in recs[p]["levels"]]
            if not s:
                continue
            e["safe_choice"][lvl] = {**cluster_bootstrap_mean(np.asarray(s), n_boot=n_boot, seed=seed), "n_scenes": len(s)}
            e["margin"][lvl] = cluster_bootstrap_mean(np.asarray([recs[p]["levels"][lvl]["delta_norm"] for p in common if lvl in recs[p]["levels"]]), n_boot=n_boot, seed=seed)
            if name != "unsteered":
                both = [p for p in common if lvl in recs[p]["levels"] and lvl in base[p]["levels"]]
                d = [float(recs[p]["levels"][lvl]["safe"]) - float(base[p]["levels"][lvl]["safe"]) for p in both]
                m = [recs[p]["levels"][lvl]["delta_norm"] - base[p]["levels"][lvl]["delta_norm"] for p in both]
                e["vs_unsteered"][lvl] = {"safe_choice_change": _stat(d, n_boot, seed), "margin_change": _stat(m, n_boot, seed),
                                         "per_scene_safe_choice_change": {p: v for p, v in zip(both, d)}}
        hf_keys = sorted({k for p in common for k in recs[p]["hazard_free"]})
        e["hazard_free"] = {k: cluster_bootstrap_mean(np.asarray([recs[p]["hazard_free"][k] for p in common if k in recs[p]["hazard_free"]]), n_boot=n_boot, seed=seed) for k in hf_keys}
        if name != "unsteered":
            sign = EXPECTED_SIGN.get(cfg.mode, 1.0)
            e["expected_sign"] = sign
            e["did"] = {}
            for target, ref in ((solid, "h0"), (solid, "h0prime"), (ghost, "h0"), (ghost, "h0prime")):
                both = [p for p in common if all(l in recs[p]["levels"] and l in base[p]["levels"] for l in (target, ref))]
                if not both:
                    continue
                dv = [(float(recs[p]["levels"][target]["safe"]) - float(base[p]["levels"][target]["safe"])) - (float(recs[p]["levels"][ref]["safe"]) - float(base[p]["levels"][ref]["safe"])) for p in both]
                mv = [(recs[p]["levels"][target]["delta_norm"] - base[p]["levels"][target]["delta_norm"]) - (recs[p]["levels"][ref]["delta_norm"] - base[p]["levels"][ref]["delta_norm"]) for p in both]
                e["did"][f"{target}_vs_{ref}"] = {"safe_choice": _stat(dv, n_boot, seed), "margin": _stat(mv, n_boot, seed), "n_scenes": len(both), "role": ("solid" if target == solid else "ghost")}
            # equivalence of the non-target levels (TOST via the scene-bootstrap CI inside +-margin)
            nt = {lvl: ci_within(e["vs_unsteered"][lvl]["safe_choice_change"], margin) for lvl in HAZARD_FREE if lvl in e["vs_unsteered"]}
            e["nontarget_equivalent"] = {**nt, "all": bool(nt) and all(nt.values()), "margin": margin}
            # hazard-free global-corruption checks: relative change of ARC and drift energy on the hazard-free cells
            checks: dict[str, Any] = {}
            for k in hf_keys:
                if not (k.startswith("arc_") or k.startswith("drift_")):
                    continue
                rel = []
                for p in common:
                    a, b = recs[p]["hazard_free"].get(k), base[p]["hazard_free"].get(k)
                    if a is not None and b is not None and np.isfinite(a) and np.isfinite(b) and abs(b) > 1e-12:
                        rel.append((a - b) / abs(b))
                if rel:
                    checks[k] = {"relative_change": _stat(rel, n_boot, seed), "within_margin": ci_within(_stat(rel, n_boot, seed), margin)}
            checks["all_within_margin"] = bool(checks) and all(v["within_margin"] for v in checks.values() if isinstance(v, dict))
            e["hazard_free_checks"] = checks
            prim = e["did"].get(f"{solid}_vs_h0", {}).get("safe_choice", {})
            e["did_primary"] = {"contrast": f"{solid}_vs_h0", **prim}
            e["did_ci_excludes_zero_expected"] = ci_excludes_zero(prim, sign)
            e["did_margin_ci_excludes_zero_expected"] = ci_excludes_zero(e["did"].get(f"{solid}_vs_h0", {}).get("margin", {}), sign)
            e["selective"] = bool(e["did_ci_excludes_zero_expected"] and e["nontarget_equivalent"]["all"] and checks["all_within_margin"])
            if cfg.kind == "sham":
                e["reproduces_unsteered"] = bool(all(recs[p]["levels"][l]["safe"] == base[p]["levels"][l]["safe"] for p in common for l in recs[p]["levels"] if l in base[p]["levels"]))
        out[name] = e
    # controls must fail: a control counts as failed when its primary DiD CI does not exclude 0 in the expected direction
    for name, e in out.items():
        cfg = configs[name]
        if cfg.kind != "main":
            continue
        ctrl = {}
        for cname, ce in out.items():
            cc = configs[cname]
            if cc.family == cfg.family and cc.kind in CONTROL_KINDS and (cc.kind == "sham" or abs(cc.beta - cfg.beta) < 1e-9):
                ctrl[cc.kind] = not ce["did_ci_excludes_zero_expected"]
        e["controls_fail"] = ctrl
        e["all_controls_fail"] = bool(ctrl) and all(ctrl.values())
        e["selective_with_controls"] = bool(e["selective"] and e["all_controls_fail"])
    return out


def calibrate(agg: dict[str, Any], configs: dict[str, SteerConfig], stat: str = "safe_choice") -> dict[str, Any]:
    """Frozen rule: per family, the smallest beta whose steered - unsteered safe-choice DiD (solid vs H0) CI excludes 0 in
    the expected direction on the discovery scenes."""

    fams: dict[str, list[dict[str, Any]]] = {}
    for name, e in agg.items():
        c = configs[name]
        if c.kind != "main":
            continue
        did = e["did"].get(e["did_primary"]["contrast"], {}).get(stat, {})
        fams.setdefault(c.family, []).append({"beta": c.beta, "config": name, "did_mean": did.get("mean"), "ci_low": did.get("ci_low"), "ci_high": did.get("ci_high"),
                                              "sign_flip_p": did.get("sign_flip_p"), "excludes_zero_expected": bool(e["did_ci_excludes_zero_expected"]),
                                              "selective": bool(e["selective"]), "all_controls_fail": bool(e.get("all_controls_fail", False))})
    out: dict[str, Any] = {"rule": f"smallest beta whose steered-unsteered {stat} DiD (solid in lane vs H0) scene-bootstrap CI excludes 0 in the expected direction on discovery scenes; band/sites frozen before this run", "families": {}}
    for fam, rows in fams.items():
        rows.sort(key=lambda r: r["beta"])
        chosen = next((r for r in rows if r["excludes_zero_expected"]), None)
        out["families"][fam] = {"beta": (chosen["beta"] if chosen else None), "candidates": rows, "chosen_config": (chosen["config"] if chosen else None)}
    return out


def coast_rows(agg: dict[str, Any], configs: dict[str, SteerConfig], arm: str, seed: Any, solid: str, confirmatory: bool, calibration: dict[str, Any] | None,
               goal: str = GOALS[0], primary: bool = True) -> list[dict[str, Any]]:
    rows = []
    for name, e in agg.items():
        c = configs[name]
        if c.kind == "unsteered":
            setting = "unsteered"
        elif c.kind == "main":
            setting = f"steered {c.mode} beta={c.beta:g} [{c.source}, {c.group}, {c.site_config}{', persistent' if c.persistent else ''}]"
        else:
            setting = f"control:{c.kind} {c.mode} beta={c.beta:g} [{c.source}, {c.group}, {c.site_config}]"
        row: dict[str, Any] = {"arm": arm, "seed": seed, "goal": goal, "goal_primary": primary, "setting": setting, "config": name, "family": c.family, "kind": c.kind, "source": c.source, "mode": c.mode,
                               "beta": c.beta, "group": c.group, "sites": list(c.sites), "site_config": c.site_config, "n_scenes": e["n_scenes"], "solid_level": solid, "operator": c.operator}
        for lvl in LEVEL_ORDER:
            col = TABLE_COLUMN[lvl]
            s = e["safe_choice"].get(lvl)
            row[col] = None if s is None else s["mean"]
            row[f"{col}_ci"] = None if s is None else [s["ci_low"], s["ci_high"]]
        if c.kind != "unsteered":
            d = e["did_primary"]
            row.update({"did_solid_vs_h0": d.get("mean"), "did_solid_vs_h0_ci": [d.get("ci_low"), d.get("ci_high")], "did_sign_flip_p": d.get("sign_flip_p"),
                        "nontarget_equivalent": e["nontarget_equivalent"]["all"], "hazard_free_within_margin": e["hazard_free_checks"]["all_within_margin"],
                        "selective": e["selective"], "all_controls_fail": e.get("all_controls_fail"), "selective_with_controls": e.get("selective_with_controls")})
            fam = (calibration or {}).get("families", {}).get(c.family, {}) if calibration else {}
            row["calibrated_beta"] = fam.get("beta")
            row["confirmatory"] = bool(confirmatory and primary and c.kind == "main" and fam.get("beta") is not None and abs(float(fam["beta"]) - c.beta) < 1e-9)
        else:
            row["confirmatory"] = bool(confirmatory and primary)
        rows.append(row)
    return rows


def coast_markdown(rows: list[dict[str, Any]]) -> str:
    def cell(r, col):
        v, ci = r.get(col), r.get(f"{col}_ci")
        if v is None:
            return "-"
        return f"{v:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]" if ci and ci[0] is not None else f"{v:.2f}"

    lines = ["| arm | seed | goal | setting | H1 ped | H1 obj | H0 | H0' | n | DiD solid vs H0 | selective | conf. |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        did = "-" if r.get("did_solid_vs_h0") is None else f"{r['did_solid_vs_h0']:+.2f} [{r['did_solid_vs_h0_ci'][0]:+.2f}, {r['did_solid_vs_h0_ci'][1]:+.2f}]"
        lines.append(f"| {r['arm']} | {r['seed']} | {r.get('goal', GOALS[0])}{'' if r.get('goal_primary', True) else ' (secondary)'} | {r['setting']} | {cell(r, 'safe_choice_H1_ped')} | {cell(r, 'safe_choice_H1_obj')} | {cell(r, 'safe_choice_H0')} | "
                     f"{cell(r, 'safe_choice_H0prime')} | {r['n_scenes']} | {did} | {r.get('selective', '-')} | {r.get('confirmatory', '-')} |")
    return "\n".join(lines) + "\n"


def top_relational_sites(interaction_map: dict[str, Any], n: int = 3, step: int = 0, groups: tuple[str, ...] = ("hazard_corridor", "gripper_corridor", "corridor")) -> list[str]:
    """Frozen band fallback: the top-n step-0 sites of the primary group by relational t (``localize_interaction.py``
    ``interaction_map.json``; falls back to the interaction t when no relational statistic exists)."""

    entries = interaction_map.get("site_results") or interaction_map.get("ranked") or []
    rows = []
    for e in entries:
        if int(e.get("step", -1)) != step or e.get("group") not in groups or "adaln" in str(e.get("site_id", "")):
            continue
        cv = e.get("cv_projection", {})
        t = e.get("relational_t", (cv.get("relational") or {}).get("t"))
        if t is None:
            t = e.get("t", (cv.get("interaction") or {}).get("t"))
        if t is None or not np.isfinite(t):
            continue
        rows.append((float(t), e["site_id"]))
    rows.sort(key=lambda r: -r[0])
    seen, out = set(), []
    for _, sid in rows:
        if sid not in seen:
            seen.add(sid)
            out.append(sid)
        if len(out) == n:
            break
    return out


# --------------------------------------------------------------------------- #
# Analysis driver (runner-agnostic)
# --------------------------------------------------------------------------- #


def run_analysis(runner: Any, configs: list[SteerConfig], solid: str, margin: float, n_boot: int, seed: int, log=None, goals: tuple[str, ...] = GOALS) -> dict[str, Any]:
    """``runner`` interface: ``scenes`` (list of pair ids); ``prepare(pid)``; ``reference(pid) -> (true, ctx, tokens)``;
    ``predict(pid, config, zero_action=False) -> {cell: [n_tokens_all, d]}``; optional ``edit_stats(pid, config)`` (one float)
    and optional ``extra_stats(pid, config) -> {name: float}`` (e.g. the receiver Mahalanobis of the edited tokens; summarised
    per config under ``extra``).  Returns ``results[goal][config]`` for every goal (``goals[0]`` = primary) on the same predictions."""

    by_name = {c.name: c for c in configs}
    per_scene: dict[str, dict[str, dict[str, Any]]] = {c.name: {} for c in configs}
    edits: dict[str, list[float]] = {}
    extras: dict[str, dict[str, list[float]]] = {}
    t0 = time.time()
    for pid in runner.scenes:
        runner.prepare(pid)
        true, ctx, tokens = runner.reference(pid)
        base_pred = runner.predict(pid, None)
        base_zero = runner.predict(pid, None, zero_action=True)
        per_scene["unsteered"][pid] = scene_record(base_pred, true, ctx, tokens, zero=base_zero, goals=goals)
        for c in configs:
            if c.kind == "unsteered":
                continue
            pred = runner.predict(pid, c)
            zero = runner.predict(pid, c, zero_action=True)
            per_scene[c.name][pid] = scene_record(pred, true, ctx, tokens, zero=zero, base_pred=base_pred, goals=goals)
            es = runner.edit_stats(pid, c) if hasattr(runner, "edit_stats") else None
            if es is not None:
                edits.setdefault(c.name, []).append(float(es))
            ex = runner.extra_stats(pid, c) if hasattr(runner, "extra_stats") else None
            for k, v in (ex or {}).items():
                if v is not None and np.isfinite(v):
                    extras.setdefault(c.name, {}).setdefault(k, []).append(float(v))
        if log:
            print(f"[coast] scene {pid} done ({time.time() - t0:.1f}s)", file=log, flush=True)
    results = {}
    for gl in goals:
        agg = aggregate(per_scene, by_name, solid, margin, n_boot, seed, goal=gl)
        for name, vals in edits.items():
            agg[name]["activation_edit_relative_rms"] = cluster_bootstrap_mean(np.asarray(vals), n_boot=n_boot, seed=seed)
        for name, per_key in extras.items():
            agg[name]["extra"] = {k: cluster_bootstrap_mean(np.asarray(v), n_boot=n_boot, seed=seed) for k, v in per_key.items()}
        results[gl] = agg
    return {"results": results, "primary_goal": goals[0], "goals": list(goals), "per_scene": per_scene, "runtime_s": time.time() - t0}


# --------------------------------------------------------------------------- #
# Conceptor loading (numpy)
# --------------------------------------------------------------------------- #


def factored_to_matrix(eigvecs: np.ndarray, mu: np.ndarray) -> np.ndarray:
    V = np.asarray(eigvecs, np.float64)
    return (V.T * np.asarray(mu, np.float64)) @ V


def matched_spectrum_random_matrix(eigvecs: np.ndarray, mu: np.ndarray, seed: int) -> np.ndarray:
    V = np.asarray(eigvecs, np.float64)
    r, d = V.shape
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(d, max(r, 1))))
    return factored_to_matrix(Q[:, :r].T, mu)


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def load_conceptor_file(path: Path, key: str = "safety", seed: int = 0) -> dict[str, Any]:
    """``geometry_conceptor.write_conceptor_npz`` (factored ``<key>_eigvecs/<key>_mu``) or the d x d ``C_<key>`` form."""

    key = key[2:] if key.startswith("C_") else key
    with np.load(path) as z:
        files = set(z.files)
        if f"{key}_eigvecs" in files:
            V, mu = np.asarray(z[f"{key}_eigvecs"], np.float64), np.asarray(z[f"{key}_mu"], np.float64)
            C = factored_to_matrix(V, mu)
            C_rand = factored_to_matrix(z["random_eigvecs"], z["random_mu"]) if "random_eigvecs" in files else matched_spectrum_random_matrix(V, mu, seed)
        elif f"C_{key}" in files:
            C = np.asarray(z[f"C_{key}"], np.float64)
            w, U = np.linalg.eigh(0.5 * (C + C.T))
            keep = w > 1e-10
            V, mu = U[:, keep].T, w[keep]
            C_rand = np.asarray(z["C_random_matched"], np.float64) if "C_random_matched" in files else matched_spectrum_random_matrix(V, mu, seed)
        else:
            raise KeyError(f"{path}: neither {key}_eigvecs nor C_{key} present (keys: {sorted(files)})")
        mean = np.asarray(z["mean"], np.float64) if "mean" in files else None
        if "rank_one_direction" in files:
            u = _unit(z["rank_one_direction"])
        elif "mean_diff" in files:
            u = _unit(z["mean_diff"])
        else:
            u = None
    d = C.shape[0]
    return {"C": C, "C_random": C_rand, "mean": mean, "rank_one": u, "quota": float(np.trace(C) / d), "d": d, "rank": int(len(mu)), "file": str(path)}


def load_transported_conceptor(path: Path, site: str, step: int, group: str, source_arm: str, target_arm: str, type_: str, seed: int = 0) -> dict[str, Any]:
    """From ``geometry_cross_arm.py --export-transport``: the source arm's conceptor of ``type_`` at (site, step, group), moved
    into the target arm's coordinates with W (A -> B: ``V @ W``; B -> A: ``V @ W.T``; same arm: unchanged); centring mean =
    the target arm's discovery mean at the same site/step/group."""

    with np.load(path) as z:
        pre = f"{site}__s{step}__{group}"
        kv, km = f"{pre}__{source_arm}__{type_}__eigvecs", f"{pre}__{source_arm}__{type_}__mu"
        if kv not in z.files:
            raise KeyError(f"{path}: no conceptor {kv}")
        V, mu = np.asarray(z[kv], np.float64), np.asarray(z[km], np.float64)
        u = _unit(z[f"{pre}__{source_arm}__{type_}__rank_one"]) if f"{pre}__{source_arm}__{type_}__rank_one" in z.files else None
        if source_arm != target_arm:
            W = np.asarray(z[f"{site}__s{step}__W"], np.float64)
            if source_arm == "A":
                V = V @ W
                u = None if u is None else u @ W
            else:
                V = V @ W.T
                u = None if u is None else u @ W.T
        mean = np.asarray(z[f"{pre}__mean_{target_arm}"], np.float64) if f"{pre}__mean_{target_arm}" in z.files else None
    C = factored_to_matrix(V, mu)
    return {"C": C, "C_random": matched_spectrum_random_matrix(V, mu, seed), "mean": mean, "rank_one": u, "quota": float(np.trace(C) / C.shape[0]), "d": C.shape[0],
            "rank": int(len(mu)), "file": str(path), "transport": f"{source_arm}->{target_arm}", "type": type_}


def operator_matrix(C: np.ndarray, beta: float, mode: str) -> np.ndarray:
    eye = np.eye(C.shape[0])
    if mode == "strengthen":
        return (1.0 - beta) * eye + beta * C
    if mode == "suppress":
        return eye - beta * C
    raise ValueError(f"unknown mode {mode}")


# --------------------------------------------------------------------------- #
# Torch runner (real model; imports torch lazily)
# --------------------------------------------------------------------------- #


class TorchRunner:
    """Encodes the factorial cells with the frozen encoder and unrolls the predictor with runtime steering hooks."""

    def __init__(self, args: argparse.Namespace, device: str, conceptors: dict[tuple[str, int, str], dict[str, Any]], sites_needed: list[str],
                 persistent: bool, group_frame_offset: int = 0, operators: dict[tuple[str, int, str], Any] | None = None,
                 mod_operators: dict[tuple[str, str, int], Any] | None = None) -> None:
        """``operators``: ``causal_metric_steer.MinDistortionOperator`` per (site, step, group); ``mod_operators``:
        ``modulation_operator.ModulationOperator`` per (adaln site, component, rank).  Both optional; the conceptor path is
        unchanged when they are absent."""
        import torch

        from localize_interaction import IDENTITY_CELLS, NULL_CELLS, encode_cell, merge_manifests, select_pairs
        from model_action_sensitivity import load_model
        from predictor_hooks import PredictorRecorder, Site, n_layers_of, predictor_of, spatial_tokens_of, unroll_from_latents
        from token_groups import REGION_GROUPS, frame_for_step, load_cell_groups, set_domain, union

        self.torch = torch
        self.args = args
        self.device = torch.device(device)
        set_domain(args.domain)
        self.wm, _ = load_model(args.repo, args.config, args.checkpoint, args.model_name, str(self.device))
        self.predictor = predictor_of(self.wm)
        self.n_layers = n_layers_of(self.predictor)
        self.n_spatial = spatial_tokens_of(self.wm)
        self._encode_cell, self._unroll, self._Recorder, self._Site = encode_cell, unroll_from_latents, PredictorRecorder, Site
        self._load_groups, self._union, self._frame_for_step = load_cell_groups, union, frame_for_step
        self.region_groups = REGION_GROUPS[args.domain]
        allow = args.allow_calibration_seeds or args.domain == "driving"  # driving has no calibration seeds (run_wave_drive.sh)
        self.pairs = select_pairs(merge_manifests(args.artifacts), args.seeds_file, allow)
        if not self.pairs:
            raise SystemExit("no complete quartets selected")
        self.scenes = sorted(self.pairs)
        self.extra_cells = list(NULL_CELLS) + list(IDENTITY_CELLS)
        self.conceptors = conceptors
        self.operators = operators or {}
        self.mod_operators = mod_operators or {}
        self.sites_needed = sorted(set(sites_needed))  # extended with the wrong-site layers by main() once the depth is known
        self.persistent = persistent
        self.offset = group_frame_offset
        self._cur: dict[str, Any] = {}
        self._edit_log: dict[str, list[float]] = {}
        self._extra_log: dict[str, dict[str, list[float]]] = {}

    # ---- per scene -----------------------------------------------------------------------------------------------
    def prepare(self, pid: str) -> None:
        torch = self.torch
        cells = self.pairs[pid]
        keys = [k for k in list(CELL_ORDER) + self.extra_cells if k in cells]
        latents = {k: self._encode_cell(self.wm, *cells[k], self.device) for k in keys}
        groups = {k: self._load_groups(*cells[k]) for k in keys}
        n_steps = int(latents[keys[0]]["actions"].shape[0])
        region = self._union(*(groups[k].group(n_steps, g) for k in keys for g in self.region_groups))
        # clean activations at the needed sites (centring-mean and rank-one fallbacks, edit diagnostics)
        sites = [self._Site.parse(s) for s in self.sites_needed]
        clean: dict[tuple[int, int], dict[str, dict[int, Any]]] = {}
        for k in keys:
            rec = self._Recorder(self.predictor, sites, self.n_spatial, store_device=self.device)
            self._unroll(self.wm, latents[k]["z_context"], latents[k]["actions"], recorder=rec)
            clean[k] = rec.acts
        self._cur = {"pid": pid, "keys": keys, "latents": latents, "groups": groups, "n_steps": n_steps, "region": np.asarray(region, dtype=int), "clean": clean}
        self._edit_log = {}
        self._extra_log = {}

    def tokens_for(self, group: str, step: int) -> np.ndarray:
        frame = self._frame_for_step(step, self.offset)
        return np.asarray(self._union(*(self._cur["groups"][k].group(frame, group) for k in self._cur["keys"])), dtype=int)

    def reference(self, pid: str):
        cur = self._cur
        true = {k: cur["latents"][k]["z_future"][:, -1].reshape(-1, cur["latents"][k]["z_future"].shape[-1]).float().cpu().numpy() for k in cur["keys"]}
        ctx = {k: cur["latents"][k]["z_context"][:, -1].reshape(-1, cur["latents"][k]["z_context"].shape[-1]).float().cpu().numpy() for k in cur["keys"]}
        return true, ctx, cur["region"]

    # ---- steering ------------------------------------------------------------------------------------------------
    def _scene_mean(self, site: str, step: int, index: np.ndarray) -> Any:
        torch = self.torch
        acts = [self._cur["clean"][k][site][step][0] for k in self._cur["keys"]]
        rows = torch.cat([a[torch.as_tensor(index, device=a.device)] if a.shape[0] > 1 else a for a in acts], dim=0).float()
        return rows.mean(dim=0)

    def _scene_mean_diff(self, site: str, step: int, index: np.ndarray, solid_level: int) -> Any:
        torch = self.torch
        sel = lambda k: self._cur["clean"][k][site][step][0]  # noqa: E731
        idx = lambda a: a[torch.as_tensor(index, device=a.device)] if a.shape[0] > 1 else a  # noqa: E731
        h1 = torch.cat([idx(sel(k)).float() for k in self._cur["keys"] if k[0] == solid_level], dim=0).mean(dim=0)
        h0 = torch.cat([idx(sel(k)).float() for k in self._cur["keys"] if k[0] == 0], dim=0).mean(dim=0)
        return h1 - h0

    def _transform(self, cfg: SteerConfig, site: str, con_site: str, step: int, index: np.ndarray):
        torch = self.torch
        cg = cfg.conceptor_group or cfg.group
        if cfg.operator == "min_distortion":
            return self._transform_min_distortion(cfg, site, con_site, step, cg)
        con = self.conceptors.get((con_site, step, cg)) or self.conceptors.get((con_site, 0, cg))
        if con is None:
            raise KeyError(f"no conceptor for site {con_site} step {step} group {cg}")
        C = con["C_random"] if cfg.variant == "random_matched" else con["C"]
        M = torch.as_tensor(operator_matrix(C, cfg.beta, cfg.mode), dtype=torch.float32, device=self.device)
        d = M.shape[0]
        mean = torch.as_tensor(con["mean"], dtype=torch.float32, device=self.device) if con.get("mean") is not None else self._scene_mean(site, step, index)
        if mean.shape[-1] != d:
            raise ValueError(f"conceptor dim {d} != activation dim {mean.shape[-1]} at {site}")
        if cfg.variant == "rank_one":
            u = torch.as_tensor(con["rank_one"], dtype=torch.float32, device=self.device) if con.get("rank_one") is not None else self._scene_mean_diff(site, step, index, self.args.primary_hazard_level)
            u = u / u.norm().clamp_min(1e-12)
        sign = EXPECTED_SIGN.get(cfg.mode, 1.0)
        eye = torch.eye(d, device=self.device)
        log = self._edit_log.setdefault(cfg.name, [])

        def fn(h):
            hf = h.float()
            base = hf - mean
            if cfg.variant == "rank_one":
                edit = base @ (M - eye)
                nrm = edit.flatten(1).norm(dim=1).reshape(-1, 1, 1)
                out = hf + sign * nrm / np.sqrt(max(h.shape[1], 1)) * u.reshape(1, 1, -1)
            else:
                out = mean + base @ M
            log.append(float((out - hf).norm() / hf.norm().clamp_min(1e-12)))
            return out.to(h.dtype)

        return fn

    def _transform_min_distortion(self, cfg: SteerConfig, site: str, con_site: str, step: int, cg: str):
        """``causal_metric_steer`` hook: sham (beta = 0) returns the tensor itself; otherwise the operator's own torch fn."""
        torch = self.torch
        op = self.operators.get((con_site, step, cg)) or self.operators.get((con_site, 0, cg))
        if op is None:
            raise KeyError(f"no min-distortion operator for site {con_site} step {step} group {cg}")
        variant = {"main": "main", "euclidean_twin": "euclid", "random_matched": "random"}.get(cfg.variant, "main")
        inner = op.torch_fn(cfg.mode, cfg.beta, variant, self.device)
        log = self._edit_log.setdefault(cfg.name, [])
        extra = self._extra_log.setdefault(cfg.name, {})
        if cfg.beta <= 0:
            return inner  # identity, bit-identical
        rec = None
        if "receiver_mean" in op.a:  # diagonal receiver Mahalanobis (patch_site.mahalanobis_score definition): ~1 in distribution
            rec = (torch.as_tensor(op.a["receiver_mean"], dtype=torch.float32, device=self.device), torch.as_tensor(op.a["receiver_var"], dtype=torch.float32, device=self.device))

        def fn(h):
            if h.shape[-1] != op.d:
                raise ValueError(f"operator dim {op.d} != activation dim {h.shape[-1]} at {site}")
            out = inner(h)
            log.append(float((out.float() - h.float()).norm() / h.float().norm().clamp_min(1e-12)))
            if rec is not None:
                before = ((h.float() - rec[0]) ** 2 / rec[1]).mean(dim=-1).sqrt().mean()
                after = ((out.float() - rec[0]) ** 2 / rec[1]).mean(dim=-1).sqrt().mean()
                extra.setdefault("receiver_mahalanobis_before", []).append(float(before))
                extra.setdefault("receiver_mahalanobis_after", []).append(float(after))
                extra.setdefault("receiver_mahalanobis_increment", []).append(float(after - before))
            return out

        return fn

    @staticmethod
    def parse_modulation_source(source: str) -> tuple[str, int]:
        """``modulation:<component>:r<rank>`` -> (component, rank)."""
        parts = source.split(":")
        return parts[1], int(parts[2][1:])

    def _modulation_steerer(self, cfg: SteerConfig):
        from modulation_operator import ModulationSteerer
        from predictor_hooks import Site

        comp, rank = self.parse_modulation_source(cfg.source)
        edits: dict[int, list[tuple]] = {}
        steps = list(range(self._cur["n_steps"])) if cfg.persistent else [cfg.step]
        log = self._edit_log.setdefault(cfg.name, [])
        for site, con_site in zip(cfg.sites, cfg.conceptor_sites):
            op = self.mod_operators.get((con_site, comp, rank))
            if op is None:
                raise KeyError(f"no modulation operator for {con_site} component {comp} rank {rank}")
            b = Site.parse(site).layer
            for st in steps:
                index = self.tokens_for(cfg.group, st)
                edits.setdefault(b, []).append((st, index, op, cfg.mode, cfg.beta, cfg.variant if cfg.variant in ("random_matched", "ungated", "ungated_on") else "main", log))
        return ModulationSteerer(self.predictor, edits, self.n_spatial)

    def _steerer(self, cfg: SteerConfig):
        if cfg.operator == "modulation":
            return self._modulation_steerer(cfg)
        edits: dict[str, list[tuple[int | None, Any, Any]]] = {}
        steps = list(range(self._cur["n_steps"])) if cfg.persistent else [cfg.step]
        for site, con_site in zip(cfg.sites, cfg.conceptor_sites):
            for st in steps:
                index = self.tokens_for(cfg.group, st)
                if len(index) == 0:
                    continue
                edits.setdefault(site, []).append((st, index, self._transform(cfg, site, con_site, st, index)))
        return PredictorSteerer(self.predictor, edits, self.n_spatial, self._Site)

    def predict(self, pid: str, cfg: SteerConfig | None, zero_action: bool = False) -> dict[tuple[int, int], np.ndarray]:
        torch = self.torch
        cur = self._cur
        out = {}
        for k in cur["keys"]:
            if zero_action and k not in ((0, 0), (NULL_CONTROL_HAZARD, 0)):
                continue
            act = cur["latents"][k]["actions"]
            act = torch.zeros_like(act) if zero_action else act
            steerer = self._steerer(cfg) if cfg is not None and cfg.kind != "unsteered" else None
            if steerer is None:
                pred = self._unroll(self.wm, cur["latents"][k]["z_context"], act)
            else:
                with steerer:
                    with torch.inference_mode():
                        roll = self.wm.unroll(cur["latents"][k]["z_context"], act_suffix=act)
                pred = roll[int(cur["latents"][k]["z_context"].shape[1]):]
                if not steerer.applied and steerer.expected:
                    raise RuntimeError(f"steering hooks never fired for {cfg.name}")
            last = pred[-1]
            out[k] = last.reshape(last.shape[0], -1, last.shape[-1])[0].float().cpu().numpy()
        return out

    def edit_stats(self, pid: str, cfg: SteerConfig) -> float | None:
        vals = self._edit_log.get(cfg.name)
        return float(np.mean(vals)) if vals else None

    def extra_stats(self, pid: str, cfg: SteerConfig) -> dict[str, float] | None:
        ex = self._extra_log.get(cfg.name)
        return {k: float(np.mean(v)) for k, v in ex.items() if v} if ex else None


class PredictorSteerer:
    """Runtime transform of the last-frame tokens at a site: ``h[:, index] <- fn(h[:, index])`` at the listed imagined
    steps (``predictor_hooks`` conventions: a forward-pre-hook on the predictor counts steps; ``_module_for`` /
    ``_group_slice`` locate the site tensor)."""

    def __init__(self, predictor, edits: dict[str, list[tuple[int | None, np.ndarray, Any]]], n_spatial: int, site_cls) -> None:
        from predictor_hooks import _group_slice, _module_for

        self.predictor, self.edits, self.n_spatial = predictor, edits, n_spatial
        self._site_cls, self._module_for, self._group_slice = site_cls, _module_for, _group_slice
        self.step = -1
        self.applied: list[tuple[str, int]] = []
        self.expected = sum(len(v) for v in edits.values())
        self._handles: list[Any] = []

    def _apply(self, site, tensor):
        import torch

        todo = [(idx, fn) for st, idx, fn in self.edits.get(site.site_id, []) if st is None or st == self.step]
        if not todo:
            return tensor
        view, n_group = self._group_slice(tensor, site, self.n_spatial)
        out = tensor.clone()
        flat = out.reshape(out.shape[0], -1, out.shape[-1])
        start = flat.shape[1] - n_group
        for idx, fn in todo:
            index = torch.as_tensor(np.asarray(idx), dtype=torch.long, device=out.device)
            if n_group == 1:
                flat[:, -1:] = fn(flat[:, -1:])
            else:
                flat[:, start + index] = fn(flat[:, start + index])
            self.applied.append((site.site_id, self.step))
        return out

    def __enter__(self):
        def count_step(module, args, kwargs):
            self.step += 1

        self._handles.append(self.predictor.register_forward_pre_hook(count_step, with_kwargs=True))
        for sid in self.edits:
            site = self._site_cls.parse(sid)
            module, kind = self._module_for(self.predictor, site)
            if kind == "pre":
                def pre_hook(m, args, kwargs, site=site):
                    return (self._apply(site, args[0]), *args[1:]), kwargs
                self._handles.append(module.register_forward_pre_hook(pre_hook, with_kwargs=True))
            else:
                self._handles.append(module.register_forward_hook(lambda m, args, out, site=site: self._apply(site, out)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--model-name", default="jepa_wm_driving")
    ap.add_argument("--model-label", default="JEPA-WM driving")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--artifacts", type=Path, nargs="+", required=True, help="factorial dir(s) with manifest.jsonl + masks/")
    ap.add_argument("--seeds-file", type=Path, default=None, help="discovery seeds (calibration) or CONFIRMATION seeds (final table)")
    ap.add_argument("--allow-calibration-seeds", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--domain", choices=["driving", "egg"], default="driving")
    ap.add_argument("--primary-hazard-level", type=int, choices=[1, OBJECT_HAZARD], default=1, help="solid identity in lane: arm A -> 1, arm B -> 3")
    ap.add_argument("--arm", default="A")
    ap.add_argument("--model-seed", default="0")
    # steering spec
    ap.add_argument("--conceptor-dir", type=Path, default=None, help="own-arm conceptors <site>__s<step>__<group>.npz (geometry_conceptor.py export or C_* form)")
    ap.add_argument("--conceptor-key", default="safety")
    ap.add_argument("--transported-conceptor", type=Path, default=None, help="geometry_cross_arm.py --export-transport npz")
    ap.add_argument("--transport-from", choices=["A", "B"], default="A")
    ap.add_argument("--transport-to", choices=["A", "B"], default=None, help="default: the --arm being steered (== --transport-from means the arm's own conceptor from the export)")
    ap.add_argument("--transport-type", default=None, help="rel_ped | rel_cone (default: the source arm's solid identity)")
    ap.add_argument("--min-distortion-dir", type=Path, default=None, help="causal_metric_steer.py fit output: <site>__s<step>__<group>.npz -> mode min_distortion")
    ap.add_argument("--modulation-operator-dir", type=Path, default=None, help="modulation_operator.py fit output: L<bb>.adaln__<comp>__r<r>.npz -> mode modulation (sites L<bb>.adaln)")
    ap.add_argument("--modulation-component", default="all", choices=["attn", "mlp", "all"])
    ap.add_argument("--modulation-rank", type=int, nargs="+", default=[1, 2, 4], help="frozen rank grid; one family per rank")
    ap.add_argument("--modulation-gate-group", default="hazard", help="token group whose block-input content drives the gate (wrong_group = its complement)")
    ap.add_argument("--sites", nargs="*", default=None, help="single-site configurations")
    ap.add_argument("--band-sites", nargs="*", default=None, help="one joint configuration at all of these sites")
    ap.add_argument("--band-file", type=Path, default=None, help="patch_site.py band.json (frozen band)")
    ap.add_argument("--sites-from-map", type=Path, default=None, help="localize_interaction.py interaction_map.json: top-N step-0 relational sites as the band")
    ap.add_argument("--n-top", type=int, default=3)
    ap.add_argument("--no-singles", action="store_true", help="band configuration only (no single-site rows)")
    ap.add_argument("--beta", type=float, nargs="+", default=list(BETA_GRID))
    ap.add_argument("--conceptor-mode", nargs="+", choices=list(MODES), default=list(MODES))
    ap.add_argument("--group", default="hazard_corridor", choices=["hazard", "corridor", "hazard_corridor"])
    ap.add_argument("--group-frame-offset", type=int, default=0)
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--persistent", action="store_true")
    ap.add_argument("--unrelated-offset", type=int, default=None, help="wrong-site control layer offset (default: half the predictor depth)")
    ap.add_argument("--controls", nargs="*", default=None, help="default: the operator's own control set (OPERATOR_CONTROLS)")
    ap.add_argument("--goal", nargs="+", choices=list(GOALS), default=list(GOALS), help="planner goal currencies; the FIRST is primary (calibration, selectivity), the rest are secondary rows")
    ap.add_argument("--equivalence-margin", type=float, default=EQUIV_MARGIN)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calibrate-out", type=Path, default=None, help="write the frozen-rule calibration (discovery run)")
    ap.add_argument("--calibration-file", type=Path, default=None, help="sealed confirmation: evaluate only the calibrated beta per family, zero refitting")
    return ap


def resolve_sites(args: argparse.Namespace) -> tuple[list[tuple[str, ...]], dict[str, Any]]:
    singles = list(args.sites or [])
    band = list(args.band_sites or [])
    prov: dict[str, Any] = {"singles": None, "band": None}
    if args.band_file is not None:
        bf = json.loads(args.band_file.read_text())
        band = [b["site_id"] for b in bf["band"]]
        prov["band"] = {"source": "band_file", "file": str(args.band_file), "group": bf.get("group"), "frozen_on_seeds": bf.get("frozen_on_seeds")}
    elif args.sites_from_map is not None:
        band = top_relational_sites(json.loads(args.sites_from_map.read_text()), n=args.n_top, step=args.step)
        prov["band"] = {"source": "top_relational_sites", "file": str(args.sites_from_map), "n_top": args.n_top}
    elif band:
        prov["band"] = {"source": "cli"}
    if not singles and band and not args.no_singles:
        singles = list(band)
    prov["singles"] = {"source": "cli" if args.sites else "band members", "sites": singles}
    configs: list[tuple[str, ...]] = []
    if band and len(band) > 1:
        configs.append(tuple(band))
    configs.extend((s,) for s in singles)
    if not configs:
        raise SystemExit("no sites: give --sites / --band-sites / --band-file / --sites-from-map")
    prov["band"] = {**(prov["band"] or {}), "sites": band}
    return configs, prov


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    t0 = time.time()
    import torch

    from predictor_hooks import Site, unrelated_site
    from protocol import sha256_file

    device = args.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    site_configs, site_prov = resolve_sites(args)
    all_sites = sorted({s for cfg in site_configs for s in cfg})
    solid = LEVEL_KEY[args.primary_hazard_level]
    # steering source (exactly one)
    sources_given = [n for n, v in (("--conceptor-dir", args.conceptor_dir), ("--transported-conceptor", args.transported_conceptor),
                                    ("--min-distortion-dir", args.min_distortion_dir), ("--modulation-operator-dir", args.modulation_operator_dir)) if v is not None]
    if len(sources_given) != 1:
        raise SystemExit(f"give exactly one of --conceptor-dir / --transported-conceptor / --min-distortion-dir / --modulation-operator-dir (got {sources_given})")
    operator = "min_distortion" if args.min_distortion_dir is not None else ("modulation" if args.modulation_operator_dir is not None else "conceptor")
    if operator == "modulation":
        args.group = args.modulation_gate_group
        bad = [s for s in all_sites if not s.endswith(".adaln")]
        if bad:
            raise SystemExit(f"modulation operator sites must be L<bb>.adaln, got {bad}")
    controls = tuple(args.controls) if args.controls is not None else tuple(OPERATOR_CONTROLS[operator])
    conceptors: dict[tuple[str, int, str], dict[str, Any]] = {}
    operators: dict[tuple[str, int, str], Any] = {}
    mod_operators: dict[tuple[str, str, int], Any] = {}
    groups_needed = [args.group]
    steps_needed = [args.step] if not args.persistent else [0, 1, 2]
    if operator == "min_distortion":
        from causal_metric_steer import MinDistortionOperator

        source = "min_distortion"
        for s in all_sites:
            for st in steps_needed:
                for g in groups_needed:
                    p = args.min_distortion_dir / f"{s}__s{st}__{g}.npz"
                    if p.exists():
                        operators[(s, st, g)] = MinDistortionOperator.load(p)
        spec = {"kind": "min_distortion_dir", "dir": str(args.min_distortion_dir),
                "loaded": {f"{k[0]}|s{k[1]}|{k[2]}": {"file": str(args.min_distortion_dir / f"{k[0]}__s{k[1]}__{k[2]}.npz"), "rank": int(v.a["main_W"].shape[1]), "eps0": float(v.a["main_eps0"]),
                                                     "eps0_euclid": float(v.a["euclid_eps0"]), "constraints": v.meta.get("variants", {}).get("main", {}).get("constraints"),
                                                     "twin_comparison": v.meta.get("twin_comparison"), "loso": v.meta.get("loso")} for k, v in operators.items()}}
    elif operator == "modulation":
        from modulation_operator import ModulationOperator

        source = None  # one family per rank, see below
        for s in all_sites:
            for r in args.modulation_rank:
                p = args.modulation_operator_dir / f"{s}__{args.modulation_component}__r{r}.npz"
                if p.exists():
                    mod_operators[(s, args.modulation_component, r)] = ModulationOperator.load(p)
        spec = {"kind": "modulation_operator_dir", "dir": str(args.modulation_operator_dir), "component": args.modulation_component, "ranks": list(args.modulation_rank), "gate_group": args.group,
                "loaded": {f"{k[0]}|{k[1]}|r{k[2]}": {"slices": v.meta.get("slices"), "gamma": v.meta.get("gamma"), "gate": v.meta.get("gate"), "selection_top": sorted(((sl, e.get("point"), e.get("p_maxt_fwer")) for sl, e in (v.meta.get("selection") or {}).items()), key=lambda t: -abs(t[1] or 0))[:3]}
                           for k, v in mod_operators.items()}}
    elif args.conceptor_dir is not None:
        source = "own"
        for s in all_sites:
            for st in steps_needed:
                for g in groups_needed:
                    p = args.conceptor_dir / f"{s}__s{st}__{g}.npz"
                    if p.exists():
                        conceptors[(s, st, g)] = load_conceptor_file(p, args.conceptor_key, args.seed)
        spec = {"kind": "conceptor_dir", "dir": str(args.conceptor_dir), "key": args.conceptor_key}
    else:
        target = args.transport_to or args.arm
        type_ = args.transport_type or ("rel_ped" if args.transport_from == "A" else "rel_cone")
        source = "own(export)" if target == args.transport_from else f"transported:{args.transport_from}->{target}"
        for s in all_sites:
            for st in steps_needed:
                for g in groups_needed:
                    try:
                        conceptors[(s, st, g)] = load_transported_conceptor(args.transported_conceptor, s, st, g, args.transport_from, target, type_, args.seed)
                    except KeyError:
                        continue
        spec = {"kind": "transported_conceptor", "file": str(args.transported_conceptor), "from": args.transport_from, "to": target, "type": type_}
    if operator == "conceptor":
        missing = [(s, st) for s in all_sites for st in ([args.step] if not args.persistent else [0]) if (s, st, args.group) not in conceptors and (s, 0, args.group) not in conceptors]
        if missing:
            raise SystemExit(f"no conceptor for {missing} (group {args.group}) in the steering source")
        spec["loaded"] = {f"{k[0]}|s{k[1]}|{k[2]}": {kk: v[kk] for kk in ("quota", "rank", "d", "file") if kk in v} | ({"transport": v["transport"]} if "transport" in v else {}) for k, v in conceptors.items()}
    elif operator == "min_distortion":
        missing = [(s, st) for s in all_sites for st in ([args.step] if not args.persistent else [0]) if (s, st, args.group) not in operators and (s, 0, args.group) not in operators]
        if missing:
            raise SystemExit(f"no min-distortion operator for {missing} (group {args.group}) in {args.min_distortion_dir}")
    else:
        missing = [(s, r) for s in all_sites for r in args.modulation_rank if (s, args.modulation_component, r) not in mod_operators]
        if missing:
            raise SystemExit(f"no modulation operator for {missing} (component {args.modulation_component}) in {args.modulation_operator_dir}")
    # model first (the wrong-site map needs the predictor depth), then the configurations
    runner = TorchRunner(args, device, conceptors, all_sites, args.persistent, args.group_frame_offset, operators=operators, mod_operators=mod_operators)
    offset = args.unrelated_offset if args.unrelated_offset is not None else max(1, runner.n_layers // 2)
    unrelated = {s: unrelated_site(Site.parse(s), runner.n_layers, offset).site_id for s in all_sites}
    runner.sites_needed = sorted(set(all_sites) | set(unrelated.values()))
    if operator == "modulation":
        configs = [UNSTEERED]
        for r in args.modulation_rank:
            src = f"modulation:{args.modulation_component}:r{r}"
            configs.extend(c for c in build_configs(site_configs, list(args.beta), list(args.conceptor_mode), args.group, src, args.persistent, unrelated, controls, args.step, operator="modulation") if c.kind != "unsteered")
        source = f"modulation:{args.modulation_component}"
    else:
        configs = build_configs(site_configs, list(args.beta), list(args.conceptor_mode), args.group, source, args.persistent, unrelated, controls, args.step, operator=operator)
    calibration = None
    dropped: dict[str, Any] = {}
    if args.calibration_file is not None:
        calibration = json.loads(args.calibration_file.read_text())
        configs, dropped = restrict_to_calibration(configs, calibration)
    goals = tuple(dict.fromkeys(args.goal))
    primary_goal = goals[0]
    res = run_analysis(runner, configs, solid, args.equivalence_margin, args.n_boot, args.seed, log=sys.stderr, goals=goals)
    by_name = {c.name: c for c in configs}
    confirmatory = args.calibration_file is not None
    if args.calibrate_out is not None:
        cal = calibrate(res["results"][primary_goal], by_name)
        cal.update({"goal": primary_goal, "secondary_calibrations": {gl: calibrate(res["results"][gl], by_name)["families"] for gl in goals[1:]},
                    "seeds_file": str(args.seeds_file), "n_scenes": len(runner.scenes), "sites": site_prov, "betas": list(args.beta), "modes": list(args.conceptor_mode), "group": args.group, "source": source, "steering": spec})
        args.calibrate_out.parent.mkdir(parents=True, exist_ok=True)
        args.calibrate_out.write_text(json.dumps(finite(cal), indent=1) + "\n")
        calibration = cal
    if calibration is not None and calibration.get("goal") not in (None, primary_goal):
        raise SystemExit(f"calibration file was frozen on goal {calibration.get('goal')!r}, this run's primary goal is {primary_goal!r}")
    rows = []
    for gl in goals:
        rows.extend(coast_rows(res["results"][gl], by_name, args.arm, args.model_seed, solid, confirmatory, calibration, goal=gl, primary=(gl == primary_goal)))
    seeds = sorted({int(runner.pairs[p][CELL_ORDER[0]][1]["seed"]) for p in runner.scenes})
    out = {
        "protocol": PROTOCOL, "model_label": args.model_label, "model_name": args.model_name, "arm": args.arm, "model_seed": args.model_seed,
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256_file(args.checkpoint), "config_sha256": sha256_file(args.config),
        "artifacts": [str(p) for p in args.artifacts], "seeds_file": str(args.seeds_file) if args.seeds_file else None, "seeds": seeds,
        "n_scenes": len(runner.scenes), "scene_ids": runner.scenes, "domain": args.domain, "primary_hazard_level": args.primary_hazard_level,
        "solid_level": solid, "ghost_level": ("h3" if solid == "h1" else "h1"),
        "hazard_levels": {LEVEL_KEY[k]: v for k, v in HAZARD_LEVEL_NAMES[args.domain].items()},
        "goal_currency": {"primary": primary_goal, "goals": list(goals), "definition": {gl: GOAL_DEFINITION[gl] for gl in goals}, "goal_cell": {gl: GOAL_CELL[gl] for gl in goals},
                          "goal_latent_source": "frozen encoder on the true future frames of the goal cell (never steered)",
                          "note": "brake goal: throttle costs more whenever any action effect is predicted, so H0 ~ 1 by construction (no unsafe baseline); progress goal supplies the unsafe baseline"},
        "planner_currency": {"tokens": "hazard U corridor (scene union) at the final predicted frame", "cost": "||P(z_H, a) - z_goal||^2 (goal per goal_currency)",
                             "safe_choice": "Delta_H = cost(A1) - cost(A0) > 0 (brake goal == a1_ranks_worse of planner_currency.scene_planner_currency)",
                             "margin": "Delta_H / (cost(A0) + cost(A1))"},
        "steering": {"source": source, "spec": spec, "site_configs": [list(c) for c in site_configs], "sites": site_prov, "betas": list(args.beta), "modes": list(args.conceptor_mode),
                     "group": args.group, "step": args.step, "persistent": args.persistent, "controls": list(controls), "unrelated_offset": offset, "unrelated_sites": unrelated,
                     "operator_kind": operator,
                     "operator": {"conceptor": "h' = mean + (h - mean) M; strengthen M = (1-b)I + bC; suppress M = I - bC; runtime hook on the last-frame tokens of the group",
                                  "min_distortion": "h' = h + delta*(h): strengthen delta = b*eps0 * W c* (causal-metric-optimal additive edit), suppress delta = W c(h) (whitened projection removal within the budget); causal_metric_steer.py",
                                  "modulation": "m'_last = m + s*b*g_h(h_hazard) sum_c gamma_c ((m - m_ref).u_c) u_c on the block's last-frame AdaLN modulation vector; modulation_operator.py"}[operator]},
        "equivalence_margin": args.equivalence_margin, "n_boot": args.n_boot, "seed": args.seed,
        "calibration": calibration, "confirmation_run": confirmatory, "families_without_calibrated_beta": dropped,
        "zero_refitting": "conceptors, band, beta and mode were fixed before this run; nothing here is fitted on these scenes" if confirmatory else None,
        "results": res["results"], "results_layout": "results[goal][config]",
        "coast_table": {"columns": ["arm", "seed", "goal", "setting", *TABLE_COLUMN.values(), "n_scenes", "did_solid_vs_h0", "selective", "confirmatory"], "primary_goal": primary_goal, "rows": rows},
        "per_scene": res["per_scene"], "interpretation_scope": INTERPRETATION_SCOPE, "runtime_s": time.time() - t0, "torch_version": torch.__version__,
        "peak_vram_gb": (float(torch.cuda.max_memory_allocated()) / 1e9) if torch.cuda.is_available() else None,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "steered_ranking.json").write_text(canonical_json(finite(out)) + "\n")
    (args.out / "coast_table.md").write_text(coast_markdown(rows))
    brief = [{k: r.get(k) for k in ("goal", "setting", "safe_choice_H1_ped", "safe_choice_H1_obj", "safe_choice_H0", "safe_choice_H0prime", "did_solid_vs_h0", "selective", "confirmatory")} for r in rows if r["goal_primary"]]
    print(json.dumps(finite({"n_scenes": len(runner.scenes), "n_configs": len(configs), "rows": brief[:30]}), indent=1))
    return out


if __name__ == "__main__":
    main()
