"""Planner dynamics: what the activation edit does to the CEM planner over an episode.

The "inner machinery" view. CEM is iterative energy minimisation over imagined futures:
each planning call draws 300 candidate action sequences, unrolls the frozen predictor for
H=6 imagined steps, scores every candidate by the latent distance between its imagined
final state and the encoded goal, refits the sampling distribution to the 10 elites and
repeats for 15 (MetaWorld) or 30 (navigation) iterations. The edit at (B3, H3) shifts the
imagined final latent of every candidate, so it can only act by moving the energy
landscape the planner descends. This script measures, from finished artifacts only:

  1. planning machinery per task and arm (calls, iterations, seconds, edit overhead);
  2. when the edited planner's chosen actions first diverge from the unsteered planner;
  3. whether divergence timing differs between rescued, regressed and unchanged scenarios;
  4. 'commitment' (fraction of divergent planning calls) and its relation to net effect;
  5. dose over time (delivered edit magnitude as a function of planning call / iteration);
  6. optional CEM convergence traces (--cem-traces DIR, JSONL) when a run exported them.

Everything except the arm-vs-native success contrasts read from ANALYSIS/report.json is
EXPLORATORY; every JSON block carries 'exploratory': true and names its correction family.
No model, simulator or GPU calls. Deterministic (fixed seeds).

Usage:
  .venv/bin/python analysis/mechanism/planner_dynamics.py --results R --freeze F --analysis A --out OUT
      [--cem-traces DIR] [--fixture-regime NAME]
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import (ARM_LABEL, ARMS, BOOTSTRAP_SEED, LEARNED, TASK_LABEL, USEFUL_GAIN_PP,  # noqa: E402
                    action_divergence, energy_frame, episodes_frame, holm, iter_records,
                    load_analysis, md_table, paired_bootstrap, provenance, spearman, standard_parser,
                    success_matrix, write_json)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT = "planner_dynamics"
LOG = logging.getLogger(SCRIPT)
EXPECTED_ITERATIONS = {"reach": 15, "reach-wall": 15, "pointmaze": 30, "wall": 30}
EDIT_ARMS = tuple(a for a in ARMS if a != "native")
LEARNED_ARMS = tuple(LEARNED.keys())
CONTROL_OF = dict(LEARNED)
DOSE_ARMS = ("fixed_rank4", "matched_random_fixed_rank4", "coupling_only", "matched_random_coupling")
DOSE_KEY = {"fixed_rank4": "coef_norm_mean", "matched_random_fixed_rank4": "coef_norm_mean",
            "coupling_only": "realized_sq_l2_sum", "matched_random_coupling": "realized_sq_l2_sum"}
DOSE_UNIT = {"coef_norm_mean": "mean coefficient L2 norm over candidates",
             "realized_sq_l2_sum": "realized squared-L2 sum over candidates"}
DRIFT_FLOOR = 1e-3  # |mean relative slope| below 0.1% of the scenario-mean dose per step is not called drift
CEM_TRACE_FIELDS = ("task", "episode", "arm", "planning_call", "iteration", "best_loss", "elite_mean", "elite_std")
SEED = BOOTSTRAP_SEED  # overridden by --seed in main(); every bootstrap and jitter rng in this script reads it
# Same overall rule as analysis/mechanism/regime_report.py OVERALL_RULE (re-implemented, not imported).
OVERALL_RULE = ("positive: >=1 confirmed cell and no harmful cell; negative: >=1 harmful cell, no confirmed and "
                "no partial-positive cell; mixed: any other combination with at least one non-inconclusive cell; "
                "inconclusive: every cell inconclusive. A cell's class comes from common.classify_regime on the "
                "registered arm-vs-native interval and the exploratory arm-vs-control interval.")

# Reference categorical palette (dataviz skill, light mode, fixed slot order): blue, orange, aqua, yellow.
COLOR = {"native": "#6b6a66", "fixed_rank4": "#2a78d6", "matched_random_fixed_rank4": "#eb6834",
         "coupling_only": "#1baf7a", "matched_random_coupling": "#eda100", "joint": "#4a3aa7",
         "visual_only": "#e87ba4", "action_condition_only": "#e34948"}
SHORT = {"native": "Unsteered", "fixed_rank4": "Refined", "matched_random_fixed_rank4": "Rand. subspace",
         "coupling_only": "Coupling", "matched_random_coupling": "Rand. direction", "joint": "Joint",
         "visual_only": "Visual-only", "action_condition_only": "Action-only"}
CLASS_COLOR = {"rescued": "#2a78d6", "regressed": "#eb6834", "unchanged": "#6b6a66"}


# ----------------------------------------------------------------------------- helpers
def _style():
    plt.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 200, "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#9a9891", "axes.linewidth": 0.6, "xtick.color": "#52514e",
        "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b", "grid.color": "#e6e5e0", "grid.linewidth": 0.5,
        "axes.grid": True, "axes.axisbelow": True, "legend.frameon": False, "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})


def fmt_ci(d, unit="", nd=2):
    return f"{d['difference']:+.{nd}f}{unit} [{d['lower']:+.{nd}f}, {d['upper']:+.{nd}f}]"


def pb(a, b, family):
    """common.paired_bootstrap with this script's seed (SEED, set from --seed)."""
    return paired_bootstrap(np.asarray(a, float), np.asarray(b, float), family=family, seed=SEED)


def pb_fraction(a, b, family):
    """paired_bootstrap for a fraction-valued quantity; undo the automatic pp scaling when the
    per-scenario values happen to be all 0/1 (e.g. commitment on single-call tasks)."""
    d = pb(a, b, family)
    if d["scale"] != 1.0:
        for k in ("difference", "lower", "upper"):
            d[k] = d[k] / d["scale"]
        d["scale"] = 1.0
    return d


def excludes_zero(d):
    return d["lower"] > 0 or d["upper"] < 0


def drift_detected(d):
    return excludes_zero(d) and abs(d["difference"]) >= DRIFT_FLOOR


def _mw(a, b):
    from scipy.stats import mannwhitneyu
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or len(b) < 3:
        return np.nan
    return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)


def spearman_safe(x, y):
    """common.spearman, but return nan with a reason (instead of scipy's ConstantInputWarning)
    when either input is constant over the finite pairs."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return dict(rho=np.nan, p=np.nan, n=int(ok.sum()), reason="fewer than three finite pairs")
    if np.unique(x[ok]).size < 2 or np.unique(y[ok]).size < 2:
        which = "x" if np.unique(x[ok]).size < 2 else "y"
        return dict(rho=np.nan, p=np.nan, n=int(ok.sum()), reason=f"{which} is constant; correlation undefined")
    out = spearman(x[ok], y[ok])
    out["reason"] = None
    return out


def holm_finite(p):
    """Holm over the finite p-values only; nan entries stay nan. Returns (adjusted, n_tested)."""
    p = np.asarray(p, float)
    adj = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        adj[ok] = holm(p[ok])
    return adj, int(ok.sum())


def walk_calls(results_dir, tasks):
    """Per (task, episode, arm): planning-call structure recovered from calls[].

    Assignment rules, in order of preference:
      'iterations_chunk'      - population calls are consumed in order, planning_calls[k].iterations
                                at a time (the contract's documented quantity); used when the total
                                number of population calls equals sum(iterations);
      'mean_rollout_boundary' - a planning call is closed by its mean rollout (candidates == 1);
                                the contract does not fix the mean rollout's position, so this is
                                only a fallback;
      'even_split'            - equal chunks; flagged.
    When both of the first two rules are available and disagree, the record is flagged
    ('boundary_disagreement'). Also records the action_trace length. Returns (call_rows, episode_rows).
    """
    call_rows, ep_rows = [], []
    for task in tasks:
        for episode, arm, rec in iter_records(results_dir, task):
            calls = rec.get("calls", [])
            plan = rec["result"].get("planning_calls", [])
            n_plan = len(plan)
            iters = [int(p.get("iterations", 0) or 0) for p in plan]
            pop_idx = [i for i, c in enumerate(calls) if c.get("candidates", 0) > 1]
            mean_idx = [i for i, c in enumerate(calls) if c.get("candidates", 0) == 1]
            assign_iter, assign_mean = None, None
            if n_plan and pop_idx and sum(iters) == len(pop_idx) and all(v > 0 for v in iters):
                assign_iter, k = {}, 0
                for pc, n_it in enumerate(iters):
                    for it in range(n_it):
                        assign_iter[pop_idx[k]] = (pc, it)
                        k += 1
            if n_plan and pop_idx and len(mean_idx) == n_plan:
                assign_mean, pc, it = {}, 0, 0
                for i in pop_idx:
                    while pc < n_plan - 1 and i > mean_idx[pc]:
                        pc += 1
                        it = 0
                    assign_mean[i] = (pc, it)
                    it += 1
            disagree = bool(assign_iter is not None and assign_mean is not None
                            and any(assign_iter[i][0] != assign_mean[i][0] for i in pop_idx))
            if assign_iter is not None:
                method, assign = "iterations_chunk", assign_iter
            elif assign_mean is not None:
                method, assign = "mean_rollout_boundary", assign_mean
            elif n_plan and pop_idx:
                method, assign = "even_split", {}
                per = int(np.ceil(len(pop_idx) / n_plan))
                for k, i in enumerate(pop_idx):
                    assign[i] = (min(k // per, n_plan - 1), k % per)
            else:
                method, assign = "unassigned", {}
            pop_secs = [float(calls[i].get("seconds", np.nan)) for i in pop_idx]
            mean_secs = [float(calls[i].get("seconds", np.nan)) for i in mean_idx]
            h6_pop = [i for i in pop_idx if calls[i].get("horizon") == 6]
            for i in pop_idx:
                pc, it = assign.get(i, (-1, -1))
                call_rows.append(dict(task=task, episode=episode, arm=arm, call_index=i, planning_call=pc, iteration=it,
                                      horizon=int(calls[i].get("horizon", -1)), seconds=float(calls[i].get("seconds", np.nan))))
            ep_rows.append(dict(task=task, episode=episode, arm=arm, n_population_calls=len(pop_idx),
                                n_h6_population_calls=len(h6_pop), n_mean_rollouts=len(mean_idx), n_planning_calls=n_plan,
                                population_calls_per_planning_call=(len(pop_idx) / n_plan) if n_plan else np.nan,
                                population_seconds_mean=float(np.mean(pop_secs)) if pop_secs else np.nan,
                                population_seconds_sum=float(np.sum(pop_secs)) if pop_secs else np.nan,
                                mean_rollout_seconds_mean=float(np.mean(mean_secs)) if mean_secs else np.nan,
                                assignment=method, boundary_disagreement=disagree,
                                n_action_trace=len(rec.get("action_trace", []))))
    return pd.DataFrame(call_rows), pd.DataFrame(ep_rows)


def paired_cell(df, task, arm, ref, value, family, rename=None):
    """Paired scenario contrast of a per-scenario column (arm minus ref) with a bootstrap CI."""
    sub = df[df.task == task].pivot(index="episode", columns="arm", values=value)
    if arm not in sub.columns or ref not in sub.columns:
        return None
    sub = sub[[arm, ref]].dropna()
    if len(sub) < 2:
        return None
    d = pb(sub[arm].values, sub[ref].values, family)
    d.update(arm=arm, reference=ref, task=task, value=value, exploratory=True,
             mean_arm=float(sub[arm].mean()), mean_reference=float(sub[ref].mean()))
    return d


# ------------------------------------------------------------------------ regime
def regime_from_report(analysis, edf, tasks):
    """Pre-registered arm-vs-native reading plus exploratory learned-vs-control contrasts.

    Regime label (for the narrative): 'positive' / 'negative' / 'mixed' / 'inconclusive',
    derived ONLY from intervals: each learned cell is classed by common.classify_regime
    (registered arm-vs-native simultaneous interval + exploratory arm-vs-control interval)
    and the cells are combined by OVERALL_RULE (the regime_report.py convention). When every
    interval includes zero the label is 'inconclusive' regardless of the point estimates.

    'descriptive_lean' (|point estimate vs native| >= USEFUL_GAIN_PP, symmetric in sign) is
    recorded per cell as metadata only; it never changes the label.
    """
    cells = []
    fam_ctrl = len(LEARNED_ARMS) * len(tasks)
    for task in tasks:
        res = analysis["results"].get(task)
        if res is None:
            LOG.warning("%s: no registered contrasts in analysis report; regime cell skipped", task)
            continue
        sm = success_matrix(edf, task)
        for arm in LEARNED_ARMS:
            c = res["contrasts"].get(f"{arm}-native")
            if c is None:
                LOG.warning("%s/%s: no %s-native contrast in analysis report; regime cell skipped", task, arm, arm)
                continue
            lo, hi = c["simultaneous_95_interval_pp"]
            vs_native = dict(difference=float(c["difference_pp"]), lower=float(lo), upper=float(hi), registered=True)
            ctrl = CONTROL_OF[arm]
            vs_ctrl = None
            if arm in sm.columns and ctrl in sm.columns:
                vs_ctrl = pb(sm[arm].values, sm[ctrl].values, fam_ctrl)
                vs_ctrl.update(exploratory=True, family_description=f"learned-vs-control over {fam_ctrl} cells")
            reg = ("excludes_zero_positive" if lo > 0 else "excludes_zero_negative" if hi < 0 else "includes_zero")
            if vs_native["difference"] >= USEFUL_GAIN_PP:
                lean = "lean_positive"
            elif vs_native["difference"] <= -USEFUL_GAIN_PP:
                lean = "lean_negative"
            else:
                lean = "flat"
            # without a control column the control interval is treated as uninformative (lower = -inf)
            cls = common.classify_regime(vs_native, vs_ctrl if vs_ctrl is not None
                                         else dict(difference=np.nan, lower=-np.inf, upper=np.inf))
            cells.append(dict(task=task, arm=arm, control=ctrl, vs_native=vs_native, vs_control=vs_ctrl,
                              registered_reading=reg, descriptive_lean=lean, classify_regime=cls,
                              control_vs_native_pp=float(res["contrasts"].get(f"{ctrl}-native", {}).get("difference_pp", np.nan))))
    counts = {k: sum(1 for c in cells if c["classify_regime"] == k)
              for k in ("confirmed", "positive_vs_native_only", "positive_vs_control_only", "harmful", "inconclusive")}
    n_conf, n_harm = counts["confirmed"], counts["harmful"]
    n_part = counts["positive_vs_native_only"] + counts["positive_vs_control_only"]
    if n_conf and not n_harm:
        label, basis = "positive", f"{n_conf} confirmed cell(s), no harmful cell"
    elif n_harm and not n_conf and not n_part:
        label, basis = "negative", f"{n_harm} harmful cell(s), no confirmed or partial-positive cell"
    elif n_conf or n_harm or n_part:
        label, basis = "mixed", f"{n_conf} confirmed, {n_harm} harmful, {n_part} partial-positive cell(s)"
    else:
        label, basis = "inconclusive", "every learned cell is inconclusive (no registered interval excludes zero in a way classify_regime scores)"
    pos_lean = [c for c in cells if c["descriptive_lean"] == "lean_positive"]
    neg_lean = [c for c in cells if c["descriptive_lean"] == "lean_negative"]
    return dict(label=label, basis=basis, rule=OVERALL_RULE, cells=cells, cell_class_counts=counts,
                useful_gain_pp=USEFUL_GAIN_PP,
                n_lean_positive=len(pos_lean), n_lean_negative=len(neg_lean),
                lean_note="descriptive_lean is metadata (|point estimate vs native| >= USEFUL_GAIN_PP); it does not enter the label",
                n_registered_positive=sum(1 for c in cells if c["registered_reading"] == "excludes_zero_positive"),
                n_registered_negative=sum(1 for c in cells if c["registered_reading"] == "excludes_zero_negative"))


# ------------------------------------------------------------- 1. planning machinery
def machinery(edf, ep_walk, tasks):
    out, flags, rows = {}, [], []
    fam = len(EDIT_ARMS) * len(tasks)
    df = edf.merge(ep_walk, on=["task", "episode", "arm"], suffixes=("", "_walk"))
    df["iterations_per_call"] = df.cem_iterations / df.n_planning_calls.replace(0, np.nan)
    df["seconds_per_planning_call"] = df.planning_seconds / df.n_planning_calls.replace(0, np.nan)
    for task in tasks:
        t = df[df.task == task]
        if not len(t):
            continue
        iters = t.groupby("arm").iterations_per_call.agg(["min", "max"])
        constant = bool(np.isclose(iters["min"], iters["max"]).all() and np.isclose(iters["min"], iters["min"].iloc[0]).all())
        observed = float(iters["min"].iloc[0]) if constant else np.nan
        exp = EXPECTED_ITERATIONS.get(task)
        if not constant:
            flags.append(f"{task}: CEM iterations per planning call are NOT constant across arms/scenarios: "
                         + ", ".join(f"{a}={r['min']:.1f}..{r['max']:.1f}" for a, r in iters.iterrows()))
        elif exp is not None and not np.isclose(observed, exp):
            flags.append(f"{task}: CEM iterations per planning call = {observed:.0f}, expected {exp}")
        ncalls = t.groupby("arm").n_planning_calls.agg(["min", "max"])
        calls_constant = bool((ncalls["min"] == ncalls["max"]).all() and (ncalls["min"] == ncalls["min"].iloc[0]).all())
        if not calls_constant:
            flags.append(f"{task}: number of planning calls varies: " + ", ".join(f"{a}={r['min']}..{r['max']}" for a, r in ncalls.iterrows()))
        n_dis = int(t.boundary_disagreement.sum()) if "boundary_disagreement" in t else 0
        if n_dis:
            flags.append(f"{task}: {n_dis} record(s) where the iterations-chunk and mean-rollout-boundary assignments of population "
                         "unrolls to planning calls disagree (iterations-chunk used)")
        pop_per = t.population_calls_per_planning_call
        pop_note = None
        if pop_per.notna().any() and constant and not np.isclose(pop_per.mean(), observed):
            pop_note = (f"{task}: {pop_per.mean():.1f} recorded population unroll calls per planning call versus "
                        f"{observed:.0f} CEM iterations per call; the per-population-call timing below covers the recorded calls only")
            flags.append(pop_note)
        per_arm = {}
        for arm in ARMS:
            u = t[t.arm == arm]
            if not len(u):
                continue
            entry = dict(n=int(len(u)), n_planning_calls_mean=float(u.n_planning_calls.mean()),
                         iterations_per_call=float(u.iterations_per_call.mean()),
                         population_calls_per_planning_call=float(u.population_calls_per_planning_call.mean()),
                         seconds_per_planning_call_mean=float(u.seconds_per_planning_call.mean()),
                         seconds_per_population_call_mean=float(u.population_seconds_mean.mean()),
                         mean_rollout_seconds_mean=float(u.mean_rollout_seconds_mean.mean()),
                         episode_planning_seconds_mean=float(u.planning_seconds.mean()))
            if arm != "native":
                o1 = paired_cell(df, task, arm, "native", "seconds_per_planning_call", fam)
                o2 = paired_cell(df, task, arm, "native", "population_seconds_mean", fam)
                nat_pc = float(t[t.arm == "native"].seconds_per_planning_call.mean())
                nat_pop = float(t[t.arm == "native"].population_seconds_mean.mean())
                entry["overhead_per_planning_call_s"] = o1
                entry["overhead_per_population_call_s"] = o2
                entry["overhead_per_planning_call_percent"] = (100 * o1["difference"] / nat_pc) if o1 and nat_pc > 0 else np.nan
                entry["overhead_per_population_call_percent"] = (100 * o2["difference"] / nat_pop) if o2 and nat_pop > 0 else np.nan
                rows.append(dict(task=task, arm=arm, calls=entry["n_planning_calls_mean"], it_per_call=entry["iterations_per_call"],
                                 s_per_call=entry["seconds_per_planning_call_mean"],
                                 overhead_s=o1["difference"] if o1 else np.nan, overhead_lo=o1["lower"] if o1 else np.nan,
                                 overhead_hi=o1["upper"] if o1 else np.nan, overhead_pct=entry["overhead_per_planning_call_percent"],
                                 s_per_pop=entry["seconds_per_population_call_mean"],
                                 pop_overhead_s=o2["difference"] if o2 else np.nan, pop_overhead_lo=o2["lower"] if o2 else np.nan,
                                 pop_overhead_hi=o2["upper"] if o2 else np.nan))
            else:
                rows.append(dict(task=task, arm=arm, calls=entry["n_planning_calls_mean"], it_per_call=entry["iterations_per_call"],
                                 s_per_call=entry["seconds_per_planning_call_mean"], overhead_s=0.0, overhead_lo=np.nan, overhead_hi=np.nan,
                                 overhead_pct=0.0, s_per_pop=entry["seconds_per_population_call_mean"], pop_overhead_s=0.0,
                                 pop_overhead_lo=np.nan, pop_overhead_hi=np.nan))
            per_arm[arm] = entry
        out[task] = dict(iterations_constant_across_arms=constant, iterations_per_call=observed, expected_iterations=exp,
                         planning_calls_constant=calls_constant, arms=per_arm,
                         population_calls_note=pop_note, assignment_methods=t.assignment.value_counts().to_dict(),
                         boundary_disagreements=n_dis)
    return dict(exploratory=True, family=fam, family_description=f"{len(EDIT_ARMS)} edit arms x {len(tasks)} tasks paired overhead contrasts (Bonferroni)",
                per_task=out, flags=flags), pd.DataFrame(rows)


# --------------------------------------------------- 2-4. divergence, timing, commitment
def divergence_block(results_dir, edf, ep_walk, tasks):
    fam_ctrl = len(LEARNED_ARMS) * len(tasks)
    per_task, div_frames, timing_tests, commit_tests = {}, [], [], []
    succ = edf.pivot_table(index=["task", "episode"], columns="arm", values="success", aggfunc="first")
    tlen = ep_walk.pivot_table(index=["task", "episode"], columns="arm", values="n_action_trace", aggfunc="first")
    for task in tasks:
        n_calls_task = None
        arms_out = {}
        frames = {}
        for arm in EDIT_ARMS:
            d = action_divergence(results_dir, task, arm)
            if not len(d):
                LOG.warning("%s/%s: no action traces; divergence skipped", task, arm)
                continue
            d = d.copy()
            d["commitment"] = d.n_divergent_calls / d.n_calls.replace(0, np.nan)
            d["never"] = (d.first_divergence < 0).astype(float)
            s = succ.loc[task] if task in succ.index.get_level_values(0) else None
            if s is not None and arm in s.columns and "native" in s.columns:
                d["success_arm"] = d.episode.map(s[arm]).astype(float)
                d["success_native"] = d.episode.map(s["native"]).astype(float)
                d["delta"] = d.success_arm - d.success_native
                d["outcome_class"] = np.select([d.delta > 0, d.delta < 0], ["rescued", "regressed"], "unchanged")
            else:
                d["delta"] = np.nan
                d["outcome_class"] = "unknown"
            # action_divergence compares only the min-length prefix; a record whose arm and native
            # traces differ in length (early termination) can flip outcome without a hash difference
            # inside the compared prefix, so it is reported separately from a true anomaly.
            if task in tlen.index.get_level_values(0) and arm in tlen.columns and "native" in tlen.columns:
                tl = tlen.loc[task]
                d["length_mismatch"] = (d.episode.map(tl[arm]) != d.episode.map(tl["native"])).astype(bool)
            else:
                d["length_mismatch"] = False
            frames[arm] = d
            div_frames.append(d)
            n_calls_task = int(d.n_calls.max())
            hist_bins = list(range(n_calls_task)) + [-1]
            hist = {("never" if b < 0 else str(b)): int((d.first_divergence == b).sum()) for b in hist_bins}
            divergers = d[d.first_divergence >= 0]
            flipped_never = (d.delta != 0) & (d.first_divergence < 0) & d.delta.notna()
            anomalies = int((flipped_never & ~d.length_mismatch).sum())
            flipped_len = int((flipped_never & d.length_mismatch).sum())
            cls = {}
            for c in ("rescued", "regressed", "unchanged"):
                u = d[d.outcome_class == c]
                ud = u[u.first_divergence >= 0]
                cls[c] = dict(n=int(len(u)), n_diverged=int(len(ud)),
                              first_divergence_median=float(ud.first_divergence.median()) if len(ud) else np.nan,
                              first_divergence_mean=float(ud.first_divergence.mean()) if len(ud) else np.nan,
                              commitment_mean=float(u.commitment.mean()) if len(u) else np.nan)
            entry = dict(n=int(len(d)), n_calls=n_calls_task, first_divergence_histogram=hist,
                         fraction_never_diverging=float(d.never.mean()),
                         first_divergence_mean_among_divergers=float(divergers.first_divergence.mean()) if len(divergers) else np.nan,
                         first_divergence_fraction_mean=float(divergers.first_divergence_fraction.mean()) if len(divergers) else np.nan,
                         commitment_mean=float(d.commitment.mean()),
                         commitment_mean_among_divergers=float(divergers.commitment.mean()) if len(divergers) else np.nan,
                         outcome_flipped_without_divergence=anomalies,
                         outcome_flipped_with_trace_length_mismatch=flipped_len,
                         n_trace_length_mismatch=int(d.length_mismatch.sum()), by_outcome_class=cls)
            # timing vs outcome (only meaningful with >= 2 planning calls)
            if n_calls_task >= 2:
                r = d[(d.outcome_class == "rescued") & (d.first_divergence >= 0)].first_divergence
                g = d[(d.outcome_class == "regressed") & (d.first_divergence >= 0)].first_divergence
                u = d[(d.outcome_class == "unchanged") & (d.first_divergence >= 0)].first_divergence
                f = pd.concat([r, g])
                timing_tests.append(dict(task=task, arm=arm, test="rescued_vs_regressed", n_a=int(len(r)), n_b=int(len(g)),
                                         median_a=float(r.median()) if len(r) else np.nan, median_b=float(g.median()) if len(g) else np.nan,
                                         p=_mw(r, g)))
                timing_tests.append(dict(task=task, arm=arm, test="flipped_vs_unchanged_divergers", n_a=int(len(f)), n_b=int(len(u)),
                                         median_a=float(f.median()) if len(f) else np.nan, median_b=float(u.median()) if len(u) else np.nan,
                                         p=_mw(f, u)))
            else:
                entry["timing_skipped_reason"] = f"{task} plans once per episode ({n_calls_task} planning call): first divergence is 0 or never; timing is degenerate"
            # commitment vs outcome delta, AMONG DIVERGERS ONLY: a flip (delta != 0) requires at
            # least one divergent call, so over all scenarios commitment > 0 is a precondition of
            # any flip and the unconditional correlation is partly structural. On single-call
            # tasks commitment is 1 for every diverger, so the test is undefined there.
            if d.delta.notna().any():
                dv = divergers[divergers.delta.notna()]
                sp = spearman_safe(dv.commitment, dv.delta)
                sp_abs = spearman_safe(dv.commitment, dv.delta.abs())
                commit_tests.append(dict(task=task, arm=arm, n_divergers=int(len(dv)), rho=sp["rho"], p=sp["p"], n=sp["n"],
                                         reason=sp["reason"], rho_abs=sp_abs["rho"], p_abs=sp_abs["p"],
                                         commitment_mean=entry["commitment_mean"],
                                         commitment_mean_among_divergers=entry["commitment_mean_among_divergers"]))
            arms_out[arm] = entry
        # learned vs its control
        contrasts = {}
        for arm, ctrl in LEARNED.items():
            if arm in frames and ctrl in frames:
                m = frames[arm].merge(frames[ctrl], on="episode", suffixes=("_l", "_c"))
                if len(m) >= 2:
                    contrasts[arm] = dict(control=ctrl, n=int(len(m)),
                                          never_pp=pb(m.never_l.values, m.never_c.values, fam_ctrl),
                                          commitment=pb_fraction(m.commitment_l.values, m.commitment_c.values, family=fam_ctrl),
                                          exploratory=True)
        per_task[task] = dict(n_calls=n_calls_task, arms=arms_out, learned_vs_control=contrasts)
    tt = pd.DataFrame(timing_tests)
    n_tt = 0
    if len(tt):
        tt["p_holm"], n_tt = holm_finite(tt.p.values)
    ct = pd.DataFrame(commit_tests)
    n_ct = 0
    if len(ct):
        ct["p_holm"], n_ct = holm_finite(ct.p.values)
        ct["p_abs_holm"], _ = holm_finite(ct.p_abs.values)
    div_all = pd.concat(div_frames, ignore_index=True) if div_frames else pd.DataFrame()
    return per_task, tt, ct, div_all, dict(timing=n_tt, commitment=n_ct)


def commitment_vs_net(ct, analysis):
    """Across (task, arm) cells: does mean commitment track the registered net effect?"""
    rows = []
    for _, r in ct.iterrows():
        c = analysis["results"].get(r.task, {}).get("contrasts", {}).get(f"{r.arm}-native")
        if c is None:
            continue
        rows.append(dict(task=r.task, arm=r.arm, commitment_mean=r.commitment_mean, difference_pp=float(c["difference_pp"]),
                         abs_difference_pp=abs(float(c["difference_pp"]))))
    df = pd.DataFrame(rows)
    if not len(df):
        return dict(exploratory=True, n_cells=0, reason="no cells")
    # commitment is defined on a different scale per task (0/1 on single-call tasks), so compare within task
    df["commitment_within_task"] = df.commitment_mean - df.groupby("task").commitment_mean.transform("mean")
    return dict(exploratory=True, descriptive=True, uncorrected=True, n_cells=int(len(df)),
                family_description="two cross-cell Spearman correlations on task-demeaned commitment: descriptive, uncorrected, "
                                   "and the cells are not independent observations (all cells of a task share the same scenarios; "
                                   "net effect = wins minus losses, which needs divergence, so the relation is partly structural)",
                spearman_commitment_vs_net_pp=spearman_safe(df.commitment_within_task, df.difference_pp),
                spearman_commitment_vs_abs_net_pp=spearman_safe(df.commitment_within_task, df.abs_difference_pp),
                cells=df.to_dict("records"))


# ------------------------------------------------------------------ 5. dose over time
def dose_block(results_dir, call_walk, tasks):
    frames = []
    for arms in (("fixed_rank4", "matched_random_fixed_rank4"), ("coupling_only", "matched_random_coupling")):
        try:
            e = energy_frame(results_dir, tasks=tasks, arms=arms)
        except Exception as ex:  # pragma: no cover - defensive
            LOG.warning("energy_frame failed for %s: %s", arms, ex)
            continue
        if len(e):
            frames.append(e)
    if not frames:
        return dict(exploratory=True, available=False, reason="no H6 population calls with energy summaries found"), pd.DataFrame()
    e = pd.concat(frames, ignore_index=True)
    e = e.merge(call_walk[["task", "episode", "arm", "call_index", "planning_call", "iteration"]],
                on=["task", "episode", "arm", "call_index"], how="left")
    e["dose"] = np.nan
    for arm, key in DOSE_KEY.items():
        if key in e.columns:
            e.loc[e.arm == arm, "dose"] = e.loc[e.arm == arm, key]
    e = e[e.dose.notna() & (e.planning_call >= 0)]
    per_task = {}
    fam = len(DOSE_ARMS) * len(tasks)
    for task in tasks:
        t = e[e.task == task]
        if not len(t):
            per_task[task] = dict(available=False, reason="no delivered-edit energy summaries for this task")
            continue
        arms_out = {}
        for arm in DOSE_ARMS:
            u = t[t.arm == arm]
            if not len(u):
                arms_out[arm] = dict(available=False, reason="no energy rows")
                continue
            # per scenario x planning call mean dose (over iterations)
            pc = u.groupby(["episode", "planning_call"]).dose.mean().reset_index()
            traj = pc.groupby("planning_call").dose.agg(["mean", "std", "count"]).reset_index()
            traj["se"] = traj["std"] / np.sqrt(traj["count"].clip(lower=1))
            it = u.groupby(["episode", "iteration"]).dose.mean().reset_index()
            itraj = it.groupby("iteration").dose.agg(["mean", "std", "count"]).reset_index()
            itraj["se"] = itraj["std"] / np.sqrt(itraj["count"].clip(lower=1))
            entry = dict(available=True, dose_key=DOSE_KEY[arm], unit=DOSE_UNIT[DOSE_KEY[arm]],
                         n_scenarios=int(u.episode.nunique()), n_planning_calls=int(u.planning_call.max() + 1),
                         n_iterations_recorded=int(u.iteration.max() + 1), dose_mean=float(u.dose.mean()),
                         trajectory_by_planning_call=traj.to_dict("records"), trajectory_by_iteration=itraj.to_dict("records"))
            for name, frame, col in (("planning_call", pc, "planning_call"), ("iteration", it, "iteration")):
                slopes, ratios = [], []
                for ep, g in frame.groupby("episode"):
                    if g[col].nunique() < 2 or g.dose.mean() <= 0:
                        continue
                    x, y = g[col].values.astype(float), g.dose.values / g.dose.mean()
                    slopes.append(float(np.polyfit(x, y, 1)[0]))
                    ratios.append(float(g.dose.values[-1] / g.dose.values[0]) if g.dose.values[0] > 0 else np.nan)
                if len(slopes) >= 2:
                    bs = pb_fraction(np.asarray(slopes), np.zeros(len(slopes)), family=fam)
                    bs.update(exploratory=True, unit=f"fraction of scenario-mean dose per {name} step",
                              last_over_first_ratio_mean=float(np.nanmean(ratios)),
                              drift_detected=bool(excludes_zero(bs) and abs(bs["difference"]) >= DRIFT_FLOOR),
                              drift_floor=DRIFT_FLOOR)
                    entry[f"drift_per_{name}"] = bs
                else:
                    entry[f"drift_per_{name}"] = dict(available=False,
                                                       reason=f"fewer than two distinct {name} values per scenario")
            arms_out[arm] = entry
        # learned vs random drift contrast (planning-call axis when available, else iteration)
        contrasts = {}
        for arm, ctrl in LEARNED.items():
            a, c = arms_out.get(arm, {}), arms_out.get(ctrl, {})
            for axis in ("planning_call", "iteration"):
                da, dc = a.get(f"drift_per_{axis}"), c.get(f"drift_per_{axis}")
                if da and dc and da.get("difference") is not None and dc.get("difference") is not None:
                    contrasts[f"{arm}_minus_{ctrl}_{axis}"] = dict(
                        learned=da["difference"], control=dc["difference"], difference=da["difference"] - dc["difference"],
                        note="difference of mean relative slopes; scenarios are paired but slopes are unpaired summaries", exploratory=True)
        per_task[task] = dict(available=True, arms=arms_out, learned_vs_control=contrasts)
    return dict(exploratory=True, available=True, family=fam, drift_floor=DRIFT_FLOOR,
                family_description=f"{len(DOSE_ARMS)} dose arms x {len(tasks)} tasks drift-vs-zero contrasts per axis (Bonferroni within axis); "
                                   f"drift is called only when the interval excludes zero and |slope| >= {DRIFT_FLOOR} per step",
                per_task=per_task), e


# ----------------------------------------------------------------- 6. CEM traces
def cem_traces_block(trace_dir, tasks):
    """Optional per-iteration CEM losses (JSONL; one object per line with CEM_TRACE_FIELDS)."""
    if trace_dir is None:
        return dict(available=False, exploratory=True,
                    reason="--cem-traces not given; upstream keeps per-iteration CEM losses in memory only (DATA_CONTRACT.md s1)"), None
    trace_dir = Path(trace_dir)
    files = sorted(trace_dir.glob("**/*.jsonl")) if trace_dir.exists() else []
    if not files:
        return dict(available=False, exploratory=True, reason=f"no *.jsonl under {trace_dir}"), None
    rows, bad = [], 0
    for f in files:
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if not all(k in o for k in CEM_TRACE_FIELDS):
                bad += 1
                continue
            rows.append({k: o[k] for k in CEM_TRACE_FIELDS})
    if not rows:
        return dict(available=False, exploratory=True, reason=f"{len(files)} file(s) but no rows with fields {CEM_TRACE_FIELDS}"), None
    tr = pd.DataFrame(rows)
    tr = tr[tr.task.isin(tasks)]
    out = dict(available=True, exploratory=True, schema=list(CEM_TRACE_FIELDS), files=[str(f) for f in files],
               n_rows=int(len(tr)), n_malformed_rows=bad, per_task={})
    # Build the contrast list first so the Bonferroni family is the number of contrasts actually computed.
    contrast_list = []
    for task in tr.task.unique():
        arms_here = set(tr[tr.task == task].arm.unique())
        for arm in sorted(arms_here):
            if arm == "native":
                continue
            for ref in ("native", CONTROL_OF.get(arm)):
                if ref is not None and ref in arms_here and ref != arm:
                    contrast_list.append((task, arm, ref))
    fam = max(1, len(contrast_list))
    out["family"] = fam
    out["family_description"] = (f"final-best-loss paired-over-episode contrasts (arm-native, learned-control) actually computed: "
                                 f"{len(contrast_list)}; Bonferroni over that number")
    for task in tr.task.unique():
        t = tr[tr.task == task]
        arms = {}
        for arm in t.arm.unique():
            u = t[t.arm == arm]
            curve = u.groupby("iteration").agg(best_loss=("best_loss", "mean"), best_loss_se=("best_loss", lambda s: s.std() / np.sqrt(max(1, len(s)))),
                                               elite_mean=("elite_mean", "mean"), elite_std=("elite_std", "mean")).reset_index()
            final = u.sort_values("iteration").groupby(["episode", "planning_call"]).best_loss.last()
            first = u.sort_values("iteration").groupby(["episode", "planning_call"]).best_loss.first()
            # iterations to reach 90% of the total decrease
            def _t90(g):
                g = g.sort_values("iteration")
                tot = g.best_loss.iloc[0] - g.best_loss.iloc[-1]
                if tot <= 0:
                    return np.nan
                target = g.best_loss.iloc[0] - 0.9 * tot
                return float(g.iteration[g.best_loss <= target].iloc[0])
            t90 = pd.Series([_t90(g) for _, g in u.groupby(["episode", "planning_call"])], dtype=float)
            arms[arm] = dict(n_episodes=int(u.episode.nunique()), curve=curve.to_dict("records"),
                             final_best_loss_mean=float(final.mean()), initial_best_loss_mean=float(first.mean()),
                             relative_decrease_mean=float(((first - final) / first.replace(0, np.nan)).mean()),
                             iterations_to_90pct_mean=float(np.nanmean(t90.values)) if len(t90) else np.nan,
                             final_elite_std_mean=float(u[u.iteration == u.iteration.max()].elite_std.mean()))
        contrasts = {}
        for ctask, arm, ref in contrast_list:
            if ctask != task:
                continue
            a = t[t.arm == arm].sort_values("iteration").groupby("episode").best_loss.last()
            b = t[t.arm == ref].sort_values("iteration").groupby("episode").best_loss.last()
            m = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
            if len(m) >= 2:
                c = pb(m.a.values, m.b.values, fam)
                c.update(arm=arm, reference=ref, exploratory=True)
                contrasts[f"{arm}-{ref}"] = c
            else:
                LOG.warning("cem traces %s %s-%s: fewer than two paired episodes; contrast skipped", task, arm, ref)
        out["per_task"][task] = dict(arms=arms, final_best_loss_contrasts=contrasts)
    return out, tr


# ---------------------------------------------------------------------- figures
def fig_overhead(mrows, tasks, path):
    _style()
    tasks = [t for t in tasks if (mrows.task == t).any()]
    fig, axes = plt.subplots(2, max(1, len(tasks)), figsize=(3.1 * max(1, len(tasks)) + 0.6, 4.6), squeeze=False)
    for j, task in enumerate(tasks):
        t = mrows[(mrows.task == task) & (mrows.arm != "native")]
        for i, (col, lo, hi, ylabel) in enumerate((("overhead_s", "overhead_lo", "overhead_hi", "overhead per planning call (s)"),
                                                   ("pop_overhead_s", "pop_overhead_lo", "pop_overhead_hi", "overhead per population unroll (ms)"))):
            ax = axes[i, j]
            scale = 1000.0 if i == 1 else 1.0
            y = t[col].values * scale
            err = np.vstack([y - t[lo].values * scale, t[hi].values * scale - y])
            ax.bar(np.arange(len(t)), y, color=[COLOR[a] for a in t.arm], width=0.7, linewidth=0)
            ax.errorbar(np.arange(len(t)), y, yerr=np.nan_to_num(err), fmt="none", ecolor="#0b0b0b", elinewidth=0.6, capsize=1.5)
            ax.axhline(0, color="#9a9891", linewidth=0.6)
            ax.set_xticks(np.arange(len(t)))
            ax.set_xticklabels([SHORT[a] for a in t.arm], fontsize=6, rotation=35, ha="right", rotation_mode="anchor")
            if i == 0:
                ax.set_title(TASK_LABEL.get(task, task))
            if j == 0:
                ax.set_ylabel(ylabel)
            ax.grid(axis="x", visible=False)
    fig.suptitle("Edit overhead relative to the unsteered planner (paired scenarios; Bonferroni 95% intervals)", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_divergence(div_all, tasks, path):
    _style()
    tasks = [t for t in tasks if (div_all.task == t).any()]
    pairs = list(LEARNED.items())
    fig, axes = plt.subplots(len(pairs), max(1, len(tasks)), figsize=(2.3 * max(1, len(tasks)) + 1.6, 2.1 * len(pairs) + 0.6), squeeze=False)
    for j, task in enumerate(tasks):
        t = div_all[div_all.task == task]
        n = int(t.n_calls.max()) if len(t) else 1
        bins = list(range(n)) + [n]  # last bin = never
        for i, (arm, ctrl) in enumerate(pairs):
            ax = axes[i, j]
            w = 0.38
            for k, a in enumerate((arm, ctrl)):
                u = t[t.arm == a]
                if not len(u):
                    continue
                fd = u.first_divergence.replace(-1, n).values
                counts = np.array([(fd == b).sum() for b in bins]) / max(1, len(u)) * 100
                ax.bar(np.arange(len(bins)) + (k - 0.5) * w, counts, width=w, color=COLOR[a], linewidth=0, label=ARM_LABEL[a])
            ax.set_xticks(np.arange(len(bins)))
            ax.set_xticklabels([str(b) for b in bins[:-1]] + ["never"], fontsize=6)
            ax.set_ylim(0, 100)
            ax.grid(axis="x", visible=False)
            if i == 0:
                ax.set_title(TASK_LABEL.get(task, task))
            if j == 0:
                ax.set_ylabel("scenarios (%)")
            if i == len(pairs) - 1:
                ax.set_xlabel("first divergent planning call")
        # one legend per row, outside the axes (avoids covering tall 'never' bars)
    for i, (arm, ctrl) in enumerate(pairs):
        h, lab = axes[i, -1].get_legend_handles_labels()
        if h:
            axes[i, -1].legend(h, lab, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=6, borderaxespad=0)
    fig.suptitle("When the edited planner first chooses a different action than the unsteered planner", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_timing(div_all, tasks, path):
    _style()
    tasks = [t for t in tasks if (div_all.task == t).any() and div_all[div_all.task == t].n_calls.max() >= 2]
    if not tasks:
        return False
    arms = [a for a in EDIT_ARMS if (div_all.arm == a).any()]
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.4 * len(tasks) + 0.4, 2.9), squeeze=False)
    rng = np.random.default_rng(SEED)
    for j, task in enumerate(tasks):
        ax = axes[0, j]
        t = div_all[(div_all.task == task) & (div_all.first_divergence >= 0)]
        for k, arm in enumerate(arms):
            for c_i, cls in enumerate(("rescued", "regressed", "unchanged")):
                u = t[(t.arm == arm) & (t.outcome_class == cls)]
                if not len(u):
                    continue
                x = k + (c_i - 1) * 0.27 + rng.uniform(-0.07, 0.07, len(u))
                ax.scatter(x, u.first_divergence, s=6, color=CLASS_COLOR[cls], alpha=0.55, linewidths=0,
                           label=cls if (j == 0 and k == 0) else None)
                ax.plot([k + (c_i - 1) * 0.27 - 0.1, k + (c_i - 1) * 0.27 + 0.1], [u.first_divergence.median()] * 2,
                        color=CLASS_COLOR[cls], linewidth=1.4)
        ax.set_xticks(np.arange(len(arms)))
        ax.set_xticklabels([SHORT[a] for a in arms], fontsize=6, rotation=35, ha="right", rotation_mode="anchor")
        ax.set_title(TASK_LABEL.get(task, task))
        ax.grid(axis="x", visible=False)
        if j == 0:
            ax.set_ylabel("first divergent planning call")
            ax.legend(loc="upper right", fontsize=6, markerscale=1.5)
    fig.suptitle("First-divergence timing by outcome class (divergers only; bar = median)", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def fig_dose(dose, tasks, path):
    _style()
    tasks = [t for t in tasks if dose["per_task"].get(t, {}).get("available")]
    if not tasks:
        return False
    pairs = list(LEARNED.items())
    fig, axes = plt.subplots(len(pairs), len(tasks), figsize=(2.3 * len(tasks) + 1.6, 2.1 * len(pairs) + 0.6), squeeze=False)
    for j, task in enumerate(tasks):
        arms = dose["per_task"][task]["arms"]
        for i, (arm, ctrl) in enumerate(pairs):
            ax = axes[i, j]
            multi = any(arms.get(a, {}).get("n_planning_calls", 1) > 1 for a in (arm, ctrl))
            key = "trajectory_by_planning_call" if multi else "trajectory_by_iteration"
            xlab = "planning call" if multi else "CEM iteration (single planning call)"
            for a in (arm, ctrl):
                e = arms.get(a, {})
                if not e.get("available"):
                    continue
                tr = pd.DataFrame(e[key])
                x = tr.iloc[:, 0].values
                ax.plot(x, tr["mean"], color=COLOR[a], linewidth=1.6, label=ARM_LABEL[a])
                ax.fill_between(x, tr["mean"] - 1.96 * tr["se"], tr["mean"] + 1.96 * tr["se"], color=COLOR[a], alpha=0.18, linewidth=0)
            ax.set_xlabel(xlab)
            if i == 0:
                ax.set_title(TASK_LABEL.get(task, task))
            if j == 0:
                ax.set_ylabel(("coefficient norm" if arm == "fixed_rank4" else "realized sq. L2 sum"))
    # one legend per row, outside the axes (never drawn over the band or the control line)
    for i in range(len(pairs)):
        for j in range(len(tasks) - 1, -1, -1):
            h, lab = axes[i, j].get_legend_handles_labels()
            if h:
                axes[i, -1].legend(h, lab, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=6, borderaxespad=0)
                break
    fig.suptitle("Delivered edit magnitude over the episode (mean over scenarios, 95% band)", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def fig_cem_traces(cem, path):
    _style()
    tasks = list(cem["per_task"].keys())
    fig, axes = plt.subplots(1, len(tasks), figsize=(max(5.2, 3.0 * len(tasks) + 0.4), 2.8), squeeze=False)
    for j, task in enumerate(tasks):
        ax = axes[0, j]
        for arm, e in cem["per_task"][task]["arms"].items():
            c = pd.DataFrame(e["curve"])
            ax.plot(c.iteration, c.best_loss, color=COLOR.get(arm, "#0b0b0b"), linewidth=1.6, label=ARM_LABEL.get(arm, arm))
            ax.fill_between(c.iteration, c.best_loss - 1.96 * c.best_loss_se, c.best_loss + 1.96 * c.best_loss_se,
                            color=COLOR.get(arm, "#0b0b0b"), alpha=0.15, linewidth=0)
        ax.set_title(TASK_LABEL.get(task, task))
        ax.set_xlabel("CEM iteration")
        if j == 0:
            ax.set_ylabel("best latent goal distance")
        ax.legend(fontsize=6)
    fig.suptitle("CEM convergence from exported traces\n(mean over episodes and planning calls, 95% band)", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------- narrative
def narrative(regime, mach, mrows, div, tt, ct, cross, dose, cem, tasks, prov, args, figs, nfam):
    L = []
    label = regime["label"]
    syn = " (SYNTHETIC FIXTURE - numbers are planted, not measured)" if prov["synthetic"] else ""
    L += [f"# Planner dynamics: what the edit does inside CEM{syn}", "",
          f"Method `{prov['method']}`; results `{prov['results']}`; freeze sha `{prov['freeze_sha256'][:12]}`; "
          f"analysis sha `{prov['analysis_sha256'][:12]}`." + (f" Fixture regime label: `{args.fixture_regime}`." if args.fixture_regime else ""), "",
          "**Scope.** Only the arm-vs-native success contrasts quoted from the frozen analysis are pre-registered. "
          "Every other number in this document (overhead, divergence, timing, commitment, dose, CEM traces) is "
          "EXPLORATORY / post hoc; each block states the family it was corrected over. Nothing here is a new "
          "efficacy claim.", ""]
    # regime
    L += ["## 1. Outcome regime this analysis is read against", ""]
    cc = regime["cell_class_counts"]
    L.append(f"Observed regime: **{label}** ({regime['basis']}). Cell classes (common.classify_regime): "
             f"{cc['confirmed']} confirmed, {cc['positive_vs_native_only']} positive-vs-native-only, "
             f"{cc['positive_vs_control_only']} positive-vs-control-only, {cc['harmful']} harmful, {cc['inconclusive']} inconclusive. "
             f"Rule: {OVERALL_RULE}")
    L.append("")
    L.append(f"Point-estimate leans are recorded for reference only: {regime['n_lean_positive']} learned cell(s) lean positive "
             f"(>= +{USEFUL_GAIN_PP:.0f} pp vs native), {regime['n_lean_negative']} lean negative (<= -{USEFUL_GAIN_PP:.0f} pp). "
             "A lean is not evidence of a sign; only an interval that excludes zero enters the label.")
    L.append("")
    rows = [dict(task=TASK_LABEL[c["task"]], arm=ARM_LABEL[c["arm"]], vs_native_pp=c["vs_native"]["difference"],
                 registered_interval=f"[{c['vs_native']['lower']:+.2f}, {c['vs_native']['upper']:+.2f}]",
                 control_vs_native_pp=c["control_vs_native_pp"],
                 vs_control_pp=(c["vs_control"]["difference"] if c["vs_control"] else np.nan),
                 vs_control_interval=(f"[{c['vs_control']['lower']:+.2f}, {c['vs_control']['upper']:+.2f}]" if c["vs_control"] else "n/a"),
                 reading=c["registered_reading"], cell_class=c["classify_regime"], lean=c["descriptive_lean"]) for c in regime["cells"]]
    L += [md_table(pd.DataFrame(rows), "{:+.1f}"), "",
          f"vs-control intervals are exploratory paired scenario bootstraps, Bonferroni over {len(regime['cells'])} learned cells. "
          "`reading` uses strict inequalities: a bound of exactly 0.00 includes zero.", ""]

    # machinery
    L += ["## 2. Planning machinery per task and arm (exploratory; overhead family = "
          f"{mach['family']} paired contrasts, Bonferroni)", ""]
    for task in tasks:
        m = mach["per_task"].get(task)
        if not m:
            continue
        it = m["iterations_per_call"]
        s = f"- {TASK_LABEL[task]}: {'constant' if m['iterations_constant_across_arms'] else 'NOT constant'} CEM iterations per planning call"
        s += f" ({it:.0f}; expected {m['expected_iterations']})" if np.isfinite(it) else " (varies)"
        s += f"; planning calls per episode {'constant' if m['planning_calls_constant'] else 'VARY'}."
        L.append(s)
    if mach["flags"]:
        L += ["", "Flags:"] + [f"- {f}" for f in mach["flags"]]
    L.append("")
    tab = mrows.copy()
    tab["task"] = tab.task.map(TASK_LABEL)
    tab["arm"] = tab.arm.map(ARM_LABEL)
    tab["overhead per call"] = [("-" if not np.isfinite(lo) else f"{d:+.2f} s [{lo:+.2f}, {hi:+.2f}] ({p:+.1f}%)")
                                for d, lo, hi, p in zip(tab.overhead_s, tab.overhead_lo, tab.overhead_hi, tab.overhead_pct)]
    tab["overhead per pop. unroll"] = [("-" if not np.isfinite(lo) else f"{d * 1000:+.1f} ms [{lo * 1000:+.1f}, {hi * 1000:+.1f}]")
                                       for d, lo, hi in zip(tab.pop_overhead_s, tab.pop_overhead_lo, tab.pop_overhead_hi)]
    L += [md_table(tab[["task", "arm", "calls", "it_per_call", "s_per_call", "overhead per call", "s_per_pop", "overhead per pop. unroll"]]
                   .rename(columns={"calls": "planning calls", "it_per_call": "CEM it./call", "s_per_call": "s / planning call",
                                    "s_per_pop": "s / pop. unroll"})), ""]
    # overhead sentence
    ov = mrows[mrows.arm.isin(LEARNED_ARMS)]
    sig = ov[(ov.overhead_lo > 0)]
    neg = ov[(ov.overhead_hi < 0)]
    if len(ov):
        maxpct = ov.loc[ov.overhead_pct.abs().idxmax()]
        if len(sig):
            L.append(f"The learned edits add measurable planning time in {len(sig)} of {len(ov)} task cells (largest "
                     f"{ARM_LABEL[maxpct.arm]} on {TASK_LABEL[maxpct.task]}: {maxpct.overhead_s:+.2f} s per planning call, "
                     f"{maxpct.overhead_pct:+.1f}% of the unsteered call).")
        elif len(neg):
            L.append(f"Learned-edit planning calls were faster than unsteered in {len(neg)} of {len(ov)} cells with an interval "
                     "excluding zero; that is a scheduling artefact (same GPU per scenario, but not the same wall-clock), not an edit property.")
        else:
            L.append(f"No learned-edit overhead interval excludes zero across {len(ov)} task cells (largest point estimate "
                     f"{ARM_LABEL[maxpct.arm]} on {TASK_LABEL[maxpct.task]}: {maxpct.overhead_s:+.2f} s per planning call, "
                     f"{maxpct.overhead_pct:+.1f}%). The edit is cheap relative to the unroll it modifies.")
    L += ["", f"![overhead]({figs['overhead']})", ""]

    # divergence
    L += ["## 3. First divergence from the unsteered planner (exploratory; learned-vs-control family = "
          f"{len(LEARNED_ARMS) * len(tasks)} cells, Bonferroni)", ""]
    drows = []
    for task in tasks:
        d = div.get(task)
        if not d:
            continue
        for arm, e in d["arms"].items():
            drows.append(dict(task=TASK_LABEL[task], arm=ARM_LABEL[arm], calls=e["n_calls"], never_pct=100 * e["fraction_never_diverging"],
                              first_div_mean=e["first_divergence_mean_among_divergers"], commitment=e["commitment_mean"],
                              flipped_without_divergence=e["outcome_flipped_without_divergence"]))
    L += [md_table(pd.DataFrame(drows)), ""]
    L.append("`never_pct` = scenarios whose action hashes match the unsteered planner at every planning call; `first_div_mean` = mean "
             "index of the first differing call among scenarios that diverge; `commitment` = divergent calls / calls, averaged over all scenarios.")
    L.append("")
    anom = sum(e["outcome_flipped_without_divergence"] for d in div.values() for e in d["arms"].values())
    lenm = sum(e["outcome_flipped_with_trace_length_mismatch"] for d in div.values() for e in d["arms"].values())
    if anom:
        L.append(f"**Anomaly:** {anom} scenario-arm records changed outcome without any action-hash divergence and with equal-length "
                 "action traces; identical actions should give identical outcomes under the same-GPU determinism gate, so these "
                 "records should be checked before interpretation"
                 + (" (in a synthetic fixture this is a property of the generator, not of the evaluator)." if prov["synthetic"] else "."))
    else:
        L.append("Consistency check passed: every outcome flip with equal-length action traces is accompanied by at least one "
                 "divergent planning call.")
    if lenm:
        L.append(f"A further {lenm} record(s) flipped outcome with no divergence inside the compared prefix but with arm and unsteered "
                 "action traces of different length (one episode terminated earlier); the divergence there lies beyond the compared "
                 "prefix and is not an anomaly.")
    L.append("")
    for task in tasks:
        d = div.get(task)
        if not d:
            continue
        for arm, c in d["learned_vs_control"].items():
            nv, cm = c["never_pp"], c["commitment"]
            ctrl = c["control"]
            more_never = "less often" if nv["difference"] < 0 else "more often"
            s = (f"- {TASK_LABEL[task]}, {ARM_LABEL[arm]} vs {ARM_LABEL[ctrl]}: the learned edit leaves the plan untouched {more_never} "
                 f"({fmt_ci(nv, ' pp', 1)} never-diverging), commitment {fmt_ci(cm, '', 3)}")
            if excludes_zero(nv) or excludes_zero(cm):
                s += " - the learned direction perturbs CEM's decisions differently from a dose-matched random direction."
            else:
                s += " - indistinguishable from its dose-matched random direction at this family size."
            L.append(s)
    L += ["", f"![divergence]({figs['divergence']})", ""]

    # timing
    L += ["## 4. Divergence timing versus outcome (exploratory; Mann-Whitney, Holm over "
          f"{nfam['timing']} tests that could be run of {len(tt) if len(tt) else 0} listed)", ""]
    skipped = [(t, d["arms"][next(iter(d["arms"]))]["timing_skipped_reason"]) for t, d in div.items()
               if d["arms"] and "timing_skipped_reason" in d["arms"][next(iter(d["arms"]))]]
    for t, r in skipped:
        L.append(f"- Skipped: {r}.")
    if len(tt):
        show = tt.copy()
        show["task"] = show.task.map(TASK_LABEL)
        show["arm"] = show.arm.map(ARM_LABEL)
        L += ["", md_table(show[["task", "arm", "test", "n_a", "n_b", "median_a", "median_b", "p", "p_holm"]], "{:.3g}"), "",
              "`p = nan`: fewer than three scenarios in a group, test not run. Groups: rescued = arm succeeded where unsteered failed; "
              "regressed = the reverse; medians are first-divergence call indices among divergers.", ""]
        rr = tt[tt.test == "rescued_vs_regressed"]
        sig = rr[rr.p_holm < 0.05]
        if len(sig):
            for _, r in sig.iterrows():
                direction = "earlier" if r.median_a < r.median_b else "later"
                L.append(f"- On {TASK_LABEL[r.task]} ({ARM_LABEL[r.arm]}) rescued scenarios diverge {direction} than regressed ones "
                         f"(median call {r.median_a:.0f} vs {r.median_b:.0f}; Holm p = {r.p_holm:.3g}).")
        elif rr.p.notna().any():
            early = int((rr.median_a < rr.median_b).sum())
            late = int((rr.median_a > rr.median_b).sum())
            L.append(f"- No cell separates rescued from regressed scenarios by divergence timing after Holm correction "
                     f"(rescued median earlier in {early}, later in {late} of {len(rr)} cells; smallest raw p = {rr.p.min():.3g} over "
                     f"{int(rr.p.notna().sum())} runnable tests). "
                     "Whether an early or a late deviation from the unsteered plan helps is not resolved by these data.")
        else:
            L.append(f"- Rescued-vs-regressed timing could not be tested in any of {len(rr)} cells (fewer than three diverging "
                     "scenarios in a group).")
        fu = tt[tt.test == "flipped_vs_unchanged_divergers"]
        sigf = fu[fu.p_holm < 0.05]
        if len(sigf):
            for _, r in sigf.iterrows():
                direction = "earlier" if r.median_a < r.median_b else "later"
                L.append(f"- On {TASK_LABEL[r.task]} ({ARM_LABEL[r.arm]}) scenarios whose outcome flipped diverge {direction} than "
                         f"diverging scenarios whose outcome did not change (median {r.median_a:.0f} vs {r.median_b:.0f}; Holm p = {r.p_holm:.3g}).")
        elif fu.p.notna().any():
            L.append(f"- Flipped and unchanged-but-diverging scenarios do not differ in first-divergence timing after Holm correction "
                     f"(smallest raw p = {fu.p.min():.3g} over {int(fu.p.notna().sum())} runnable tests).")
        else:
            L.append(f"- Flipped-vs-unchanged timing could not be tested in any of {len(fu)} cells (fewer than three scenarios in a group).")
    if figs.get("timing"):
        L += ["", f"![timing]({figs['timing']})"]
    L.append("")

    # commitment
    L += ["## 5. Commitment and net effect (exploratory; per-cell Spearman among divergers, Holm over "
          f"{nfam['commitment']} runnable tests of {len(ct) if len(ct) else 0} cells)", ""]
    L.append("Commitment = divergent planning calls / planning calls. An outcome flip (rescued or regressed) requires at least one "
             "divergent call, so over all scenarios commitment > 0 is a precondition of any flip and the unconditional correlation "
             "between commitment and outcome delta is partly structural; on single-call tasks commitment is literally the "
             "divergence indicator. The tests below are therefore restricted to scenarios that diverged at least once "
             "(`n_divergers`), where commitment varies without that structural link. `rho` is against the signed outcome delta, "
             "`rho_abs` against |delta| (any flip).")
    L.append("")
    if len(ct):
        show = ct.copy()
        show["task"] = show.task.map(TASK_LABEL)
        show["arm"] = show.arm.map(ARM_LABEL)
        show["reason"] = show.reason.fillna("")
        L += [md_table(show[["task", "arm", "commitment_mean", "commitment_mean_among_divergers", "n_divergers", "rho", "p", "p_holm",
                             "rho_abs", "p_abs", "p_abs_holm", "reason"]], "{:.3g}"), ""]
        sig = ct[ct.p_holm < 0.05]
        if len(sig):
            for _, r in sig.iterrows():
                sign = "higher" if r.rho > 0 else "lower"
                L.append(f"- On {TASK_LABEL[r.task]} ({ARM_LABEL[r.arm]}), among diverging scenarios, a larger fraction of divergent "
                         f"calls goes with a {sign} outcome delta (rho = {r.rho:+.2f}, Holm p = {r.p_holm:.3g}). This is a descriptive, "
                         "post hoc association among divergers; it does not say that perturbing more calls causes the outcome.")
        elif ct.p.notna().any():
            L.append(f"- Among diverging scenarios, the fraction of divergent calls does not predict the signed outcome delta in any cell "
                     f"after Holm correction (|rho| <= {ct.rho.abs().max():.2f} over {int(ct.p.notna().sum())} runnable tests). "
                     "Commitment measures how much the edit perturbs the trajectory, not in which direction.")
        else:
            L.append("- No commitment test could be run (every cell has a constant commitment or delta among divergers, e.g. single-call tasks).")
        siga = ct[ct.p_abs_holm < 0.05]
        if len(siga):
            L.append("- Commitment vs |delta| among divergers is Holm-significant in: "
                     + "; ".join(f"{TASK_LABEL[r.task]} {ARM_LABEL[r.arm]} (rho = {r.rho_abs:+.2f}, Holm p = {r.p_abs_holm:.3g})" for _, r in siga.iterrows())
                     + " - more perturbed trajectories are more likely to flip in either direction there (descriptive).")
    cr = cross.get("spearman_commitment_vs_net_pp")
    ca = cross.get("spearman_commitment_vs_abs_net_pp")
    if cr and np.isfinite(cr["rho"]):
        L.append(f"- Across {cross['n_cells']} (task, arm) cells, task-demeaned mean commitment correlates with the registered net "
                 f"effect at rho = {cr['rho']:+.2f} (p = {cr['p']:.2g}) and with its absolute size at rho = {ca['rho']:+.2f} "
                 f"(p = {ca['p']:.2g}). This cross-cell relation is descriptive and uncorrected, the cells share the same scenarios "
                 "within a task so they are not independent observations, and net effect (wins minus losses) needs divergence, so "
                 "part of any positive rho is structural. "
                 + ("In this panel cells with more decision perturbation have larger |net effect|, with those caveats."
                    if (ca['p'] < 0.05 and ca['rho'] > 0) else
                    f"In this panel cells with more decision perturbation tend to have {'lower' if cr['rho'] < 0 else 'higher'} net success, with those caveats."
                    if cr['p'] < 0.05 else
                    "In this panel the amount of decision perturbation does not track the size or sign of the net effect."))
    elif cr:
        L.append(f"- Cross-cell commitment-vs-net-effect correlation not computed ({cr.get('reason')}).")
    L.append("")

    # dose
    L += [f"## 6. Dose over time (exploratory; drift family = {dose.get('family', 0)} arm-task cells per axis, Bonferroni)", ""]
    if not dose.get("available"):
        L.append(f"- Skipped: {dose.get('reason')}.")
    else:
        rows = []
        for task in tasks:
            pt = dose["per_task"].get(task, {})
            if not pt.get("available"):
                L.append(f"- {TASK_LABEL[task]}: skipped ({pt.get('reason')}).")
                continue
            for arm, e in pt["arms"].items():
                if not e.get("available"):
                    L.append(f"- {TASK_LABEL[task]} {ARM_LABEL[arm]}: skipped ({e.get('reason')}).")
                    continue
                dp, di = e.get("drift_per_planning_call", {}), e.get("drift_per_iteration", {})
                rows.append(dict(task=TASK_LABEL[task], arm=ARM_LABEL[arm], dose_mean=e["dose_mean"], calls=e["n_planning_calls"],
                                 drift_per_call=(fmt_ci(dp, "", 4) if "difference" in dp else "n/a (single call)"),
                                 last_over_first=(dp.get("last_over_first_ratio_mean", np.nan) if "difference" in dp else np.nan),
                                 drift_per_iteration=(fmt_ci(di, "", 4) if "difference" in di else "n/a")))
        if rows:
            L += [md_table(pd.DataFrame(rows), "{:.3g}"), "",
                  "Dose = mean coefficient L2 norm over the 300 candidates (refined arms) or realized squared-L2 sum (coupling arms) per H6 "
                  "population unroll; drift = mean over scenarios of the OLS slope of dose (normalised by the scenario mean) against "
                  "planning-call or iteration index, i.e. fractional change per step.", ""]
            drifting, stable = [], []
            for task in tasks:
                pt = dose["per_task"].get(task, {})
                for arm, e in pt.get("arms", {}).items():
                    for axis in ("planning_call", "iteration"):
                        d = e.get(f"drift_per_{axis}", {})
                        if "difference" in d:
                            (drifting if drift_detected(d) else stable).append((task, arm, axis, d))
            if drifting:
                for task, arm, axis, d in drifting:
                    L.append(f"- {TASK_LABEL[task]} {ARM_LABEL[arm]}: the delivered dose drifts {'up' if d['difference'] > 0 else 'down'} "
                             f"by {100 * d['difference']:+.2f}% of its mean per {axis.replace('_', ' ')} ({fmt_ci(d, '', 4)}); the edit "
                             "magnitude is state-dependent, so the energy landscape is not shifted by a constant amount over the episode.")
            if stable and not drifting:
                L.append(f"- The delivered dose is stable over the episode in all {len(stable)} tested arm-task-axis cells (no drift "
                         f"both excluding zero and exceeding {100 * DRIFT_FLOOR:.1f}% per step): the edit shifts every population's "
                         "imagined final latent by a roughly constant magnitude regardless of where the state has moved.")
            elif stable:
                L.append(f"- The remaining {len(stable)} arm-task-axis cells show no drift (interval includes zero or |slope| < "
                         f"{100 * DRIFT_FLOOR:.1f}% per step).")
            lvc = []
            for task in tasks:
                pt = dose["per_task"].get(task, {})
                for k, c in pt.get("learned_vs_control", {}).items():
                    arm, rest = k.split("_minus_")
                    ctrl, axis = rest.rsplit("_", 1) if rest.endswith("_iteration") else (rest.replace("_planning_call", ""), "planning_call")
                    lvc.append(dict(task=TASK_LABEL[task], learned=ARM_LABEL[arm], control=ARM_LABEL[ctrl], axis=axis.replace("_", " "),
                                    learned_slope=c["learned"], control_slope=c["control"], difference=c["difference"]))
            if lvc:
                L += ["", "Learned-vs-control drift (descriptive difference of mean relative slopes, fraction per step; no interval):", "",
                      md_table(pd.DataFrame(lvc), "{:+.4f}")]
        if figs.get("dose"):
            L += ["", f"![dose]({figs['dose']})"]
    L.append("")

    # CEM traces
    L += ["## 7. CEM convergence traces (optional input)", ""]
    if not cem.get("available"):
        L.append(f"- Skipped: {cem.get('reason')}.")
        L.append("- To enable: export one JSON object per line with fields "
                 "`{task, episode, arm, planning_call, iteration, best_loss, elite_mean, elite_std}` (best_loss = lowest latent goal "
                 "distance in the population at that iteration; elite_mean/elite_std = mean and std of the 10 elite losses) into any "
                 "`*.jsonl` under a directory and pass `--cem-traces DIR`.")
    else:
        L.append(f"- Read {cem['n_rows']} rows from {len(cem['files'])} file(s) ({cem['n_malformed_rows']} malformed rows skipped).")
        for task, pt in cem["per_task"].items():
            for arm, e in pt["arms"].items():
                L.append(f"- {TASK_LABEL.get(task, task)} {ARM_LABEL.get(arm, arm)}: best loss {e['initial_best_loss_mean']:.4g} -> "
                         f"{e['final_best_loss_mean']:.4g} ({100 * e['relative_decrease_mean']:.1f}% decrease), 90% of the decrease by "
                         f"iteration {e['iterations_to_90pct_mean']:.1f}, final elite std {e['final_elite_std_mean']:.3g}.")
            for k, c in pt["final_best_loss_contrasts"].items():
                verdict = ("reaches a lower final energy" if c["upper"] < 0 else "ends at a higher final energy" if c["lower"] > 0
                           else "ends at an indistinguishable final energy")
                who = ("the learned-edit planner" if c["arm"] in LEARNED_ARMS else
                       "the control planner" if c["arm"] in CONTROL_OF.values() else f"the {ARM_LABEL.get(c['arm'], c['arm'])} planner")
                L.append(f"- {TASK_LABEL.get(task, task)} {k}: final best loss {fmt_ci(c, '', 4)} - {who} {verdict} relative to "
                         f"{ARM_LABEL.get(c['reference'], c['reference'])} (Bonferroni over {c['family']} contrasts actually computed). "
                         "Note: an edited planner's energy is measured on its own edited landscape, so a lower value need not mean "
                         "a better real outcome.")
        if figs.get("cem"):
            L += ["", f"![cem]({figs['cem']})"]
    L.append("")

    # interpretation
    L += ["## 8. Interpretation in the language of energy-based planning", ""]
    L += interpretation(regime, mach, mrows, div, tt, ct, cross, dose, cem, tasks)
    L.append("")
    L += ["## What this does and does not establish", ""]
    L += establishes(regime, div, tt, ct, dose, cem, tasks, prov)
    return "\n".join(L) + "\n"


def interpretation(regime, mach, mrows, div, tt, ct, cross, dose, cem, tasks):
    P = []
    label = regime["label"]
    P.append("CEM does not follow gradients; it minimises a latent goal distance E(a) = ||f(z_0, a) - z_goal||^2 over candidate action "
             "sequences a by sampling, keeping the 10 lowest-energy candidates and refitting the sampler. The edit at (B3, H3) changes "
             "f, so it moves every candidate's imagined final latent and therefore reshapes the landscape E that the planner "
             "descends; it cannot change the descent rule, the sample budget or the goal. Three things about that reshaping are "
             "visible from the recorded artifacts.")
    P.append("")
    # (a) how often decisions diverge
    never = {a: [] for a in EDIT_ARMS}
    commit = {a: [] for a in EDIT_ARMS}
    for task in tasks:
        d = div.get(task)
        if not d:
            continue
        for arm, e in d["arms"].items():
            never[arm].append(e["fraction_never_diverging"])
            commit[arm].append(e["commitment_mean"])
    ln = [np.mean(never[a]) for a in LEARNED_ARMS if never[a]]
    cn = [np.mean(never[CONTROL_OF[a]]) for a in LEARNED_ARMS if never[CONTROL_OF[a]]]
    lc = [np.mean(commit[a]) for a in LEARNED_ARMS if commit[a]]
    if ln:
        P.append(f"**(a) The landscape change is large enough to move the argmin.** Averaged over tasks, the learned edits leave "
                 f"{100 * np.mean(ln):.0f}% of scenarios with a plan identical to the unsteered planner at every call (random controls: "
                 f"{100 * np.mean(cn):.0f}%), and diverge on {100 * np.mean(lc):.0f}% of planning calls once all scenarios are counted "
                 "(task means; single-call navigation tasks contribute 0/1 per scenario). "
                 + ("Most scenarios are re-planned differently at some point: the shift of the imagined final latent is above the "
                    "resolution of a 300-sample population in the closed loop, where perturbed states compound across calls."
                    if np.mean(ln) < 0.5 else
                    "Most scenarios never see a different plan: the shift of the imagined final latent stays below what separates "
                    "the elites of a 300-sample population, consistent with the development decision diagnostic."))
        P.append("")
    # (b) direction of the shift relative to outcome. The positive / negative / mixed prose fires ONLY
    # when the label was earned by intervals (classify_regime cells + OVERALL_RULE); point-estimate
    # leans are mentioned as leans and never as a sign.
    cells = regime["cells"]
    conf = [c for c in cells if c["classify_regime"] == "confirmed"]
    harm = [c for c in cells if c["classify_regime"] == "harmful"]
    part = [c for c in cells if c["classify_regime"] in ("positive_vs_native_only", "positive_vs_control_only")]
    lean_p = [c for c in cells if c["descriptive_lean"] == "lean_positive"]
    lean_n = [c for c in cells if c["descriptive_lean"] == "lean_negative"]

    def _cellnames(cs):
        return ", ".join(f"{TASK_LABEL[c['task']]} {ARM_LABEL[c['arm']]} ({c['vs_native']['difference']:+.1f} pp)" for c in cs)

    lean_txt = ""
    if lean_p or lean_n:
        lean_dir = "positive" if lean_p and not lean_n else "negative" if lean_n and not lean_p else "both ways"
        lean_txt = (f" Point estimates lean {lean_dir} in " + _cellnames(lean_p + lean_n)
                    + ("; none of those intervals excludes zero, and a lean establishes no sign." if not (conf or harm or part) else
                       "; a lean establishes no sign beyond the interval-classed cells named above."))
    ctrl_below = [c for c in cells if c["vs_control"] and c["vs_control"]["upper"] < 0]
    part_ctrl_only = [c for c in part if c["classify_regime"] == "positive_vs_control_only"]
    part_nat_only = [c for c in part if c["classify_regime"] == "positive_vs_native_only"]
    if label == "positive":
        P.append("**(b) Outcome regime: positive.** At least one learned cell is 'confirmed' - its registered arm-vs-native interval and "
                 "its exploratory arm-vs-control interval both exclude zero above with a gain of at least "
                 f"{USEFUL_GAIN_PP:.0f} pp - and no cell is harmful ({_cellnames(conf)}). In energy terms, on those cells the edited "
                 "landscape's minimum lies nearer the true goal than the native one often enough to raise the success rate; the "
                 "remaining cells are inconclusive and carry no sign." + lean_txt + " " + _timing_clause(tt) + _commit_clause(ct, cross))
    elif label == "negative":
        P.append("**(b) Outcome regime: negative.** At least one learned cell is 'harmful' - its registered arm-vs-native interval "
                 f"excludes zero below ({_cellnames(harm)}) - and no cell is confirmed or partially positive. On those cells the edited "
                 "landscape's minima are displaced away from actions that reach the goal: the edit corrupts rather than corrects the "
                 "imagined final latent, or corrects a component the objective does not score while perturbing one it does. The "
                 "remaining cells are inconclusive and carry no sign." + lean_txt + " " + _timing_clause(tt) + _commit_clause(ct, cross))
    elif label == "mixed":
        P.append("**(b) Outcome regime: mixed.** The interval-based cell classes disagree: "
                 f"{len(conf)} confirmed ({_cellnames(conf) or 'none'}), {len(harm)} harmful ({_cellnames(harm) or 'none'}), "
                 f"{len(part)} partially positive ({_cellnames(part) or 'none'}). "
                 + ("Where the same edit helps on one task and hurts on another, the planner rule being identical across tasks, the sign "
                    "must come from where the edited direction points relative to each task's goal geometry. "
                    if (conf and harm) else "")
                 + (f"In {len(part_ctrl_only)} cell(s) only the EXPLORATORY arm-vs-control interval excludes zero above while the "
                    "REGISTERED arm-vs-native interval includes zero, so the learned direction beats its dose-matched random direction "
                    "there without a registered gain over the unsteered planner. " if part_ctrl_only else "")
                 + (f"In {len(part_nat_only)} cell(s) the registered arm-vs-native interval excludes zero above but the arm-vs-control "
                    "interval does not, so a gain over the unsteered planner is registered there but is not distinguished from a "
                    "dose-matched random edit. " if part_nat_only else "")
                 + ("No cell establishes a sign against both references; "
                    + ("every registered interval includes zero, so this 'mixed' label rests on exploratory control contrasts alone. "
                       if not (conf or harm or part_nat_only) else "the label is not an efficacy claim. ")
                    if not (conf and harm) else "")
                 + lean_txt + " " + _timing_clause(tt) + _commit_clause(ct, cross))
    else:
        P.append("**(b) Outcome regime: inconclusive.** Every learned cell is inconclusive: no registered arm-vs-native interval "
                 "excludes zero and no exploratory arm-vs-control interval excludes zero above"
                 + (f" ({len(ctrl_below)} arm-vs-control interval(s) exclude zero below - " + _cellnames(ctrl_below)
                    + " - which classify_regime does not score; descriptive only)" if ctrl_below else "")
                 + ". The edits are not inert - decisions "
                 "diverge - but nothing here shows that the reshaped landscape sends the planner to minima nearer or farther from the "
                 "goal than the native ones on average; the data are consistent with the perturbation adding variance to which "
                 "actions are chosen without a detectable bias in either direction." + lean_txt + " " + _timing_clause(tt) + _commit_clause(ct, cross))
    P.append("")
    # (c) dose and overhead
    drift_cells = []
    if dose.get("available"):
        for task in tasks:
            for arm, e in dose["per_task"].get(task, {}).get("arms", {}).items():
                for axis in ("planning_call", "iteration"):
                    d = e.get(f"drift_per_{axis}", {})
                    if "difference" in d and drift_detected(d):
                        drift_cells.append((task, arm, axis))
    ov = mrows[mrows.arm.isin(LEARNED_ARMS)]
    ov_sig = int((ov.overhead_lo > 0).sum()) if len(ov) else 0
    P.append("**(c) The edit is a cheap, " + ("state-dependent" if drift_cells else "roughly constant-magnitude") + " reshaping.** "
             + (f"The delivered dose drifts across the episode in {len(drift_cells)} arm-task-axis cells, so the edit responds to the "
                "state the planner has reached rather than adding a fixed offset; whether that adaptation is helpful is a question "
                "for the outcome contrasts, not for this diagnostic. "
                if drift_cells else
                "The delivered dose does not drift measurably across planning calls or iterations, so the landscape is shifted by a "
                "roughly constant magnitude wherever the state goes. ")
             + (f"It costs measurable time in {ov_sig} learned cells" if ov_sig else "It costs no measurable planning time")
             + (f" (largest {ov.overhead_pct.abs().max():.1f}% of an unsteered planning call)." if len(ov) else "."))
    P.append("")
    P.append("**What would require per-iteration traces.** Whether the edit changes the *shape* of the descent - faster or slower "
             "elite convergence, a narrower or wider elite spread, a different final energy on the same scenario - is not recoverable "
             "from planning_calls, calls[] and action hashes; the upstream evaluator keeps the per-iteration losses in memory only. "
             + ("Exported traces were supplied and Section 7 reports the convergence comparison."
                if cem.get("available") else
                "No traces were supplied, so Section 7 is empty; the `--cem-traces` reader documents the schema needed."))
    return [re.sub(r" {2,}", " ", s).rstrip() for s in P]


def _timing_clause(tt):
    if not len(tt):
        return "Timing of divergence was not testable (single planning call per episode). "
    rr = tt[(tt.test == "rescued_vs_regressed")]
    sig = rr[rr.p_holm < 0.05]
    if len(sig):
        earlier = int((sig.median_a < sig.median_b).sum())
        return (f"Divergence timing separates rescues from regressions in {len(sig)} cells (rescues earlier in {earlier}): the edit's "
                "help or harm is concentrated at a particular stage of the episode. ")
    return ("Rescued and regressed scenarios diverge at indistinguishable stages of the episode, so the sign of the effect is not "
            "carried by when the planner first deviates. ")


def _commit_clause(ct, cross):
    if not len(ct):
        return ""
    sig = ct[ct.p_holm < 0.05]
    if len(sig):
        return (f"Among diverging scenarios, commitment is associated with the outcome delta in {len(sig)} cell(s) (descriptive, post hoc); "
                "that is an association among perturbed trajectories, not a benefit of perturbing more. ")
    if ct.p.notna().any():
        return "Among diverging scenarios, commitment (how many calls diverge) does not predict the outcome sign: perturbation size is not benefit. "
    return "Commitment could not be related to outcome (constant among divergers, e.g. single-call tasks). "


def establishes(regime, div, tt, ct, dose, cem, tasks, prov):
    P = []
    if prov["synthetic"]:
        P.append("- **These numbers come from a synthetic fixture.** Every statement above exercises the code path; none is a "
                 "measurement of JEPA-WM.")
    P.append("- **Establishes (descriptively):** how many planning calls and CEM iterations each arm ran, the edit's wall-clock "
             "overhead, the fraction of scenarios in which the edited planner ever chose a different action than the unsteered "
             "planner, when it first did so, how many calls differed, and whether the delivered edit magnitude drifted over the episode.")
    P.append(f"- **Does not establish efficacy.** The outcome regime label ('{regime['label']}') is derived only from intervals: "
             "the frozen arm-vs-native contrasts plus the exploratory arm-vs-control contrasts, via common.classify_regime and the "
             "regime_report overall rule. Point-estimate leans never enter it. Every other quantity here is exploratory, post hoc, "
             "and corrected only within its own stated family. None of the exploratory intervals or p-values is a registered test.")
    P.append("- **Commitment-outcome relations are partly structural.** A flip requires divergence, so the tests are restricted to "
             "divergers and the cross-cell correlation is descriptive and uncorrected over non-independent cells.")
    P.append("- **Does not identify a mechanism of rescue.** Divergence timing and commitment describe *that* and *when* decisions "
             "changed, not *why* one changed decision reached the goal and another did not; that would need per-scenario "
             "counterfactual rollouts, which were not run.")
    P.append("- **Does not see inside a planning call.** Without per-iteration CEM losses (best loss, elite mean/std per iteration) "
             "we cannot say whether the edit changes convergence speed, elite spread, or the final energy on the edited landscape"
             + ("; exported traces partially fill this gap but are measured on the edited objective, not the true one."
                if cem.get("available") else "."))
    single = [TASK_LABEL[t] for t in tasks if div.get(t) and div[t]["n_calls"] is not None and div[t]["n_calls"] < 2]
    if single:
        P.append(f"- **Navigation tasks ({', '.join(single)}) plan once per episode**, so divergence timing there is binary (first "
                 "call or never) and dose can only be followed across CEM iterations, not across planning calls.")
    P.append("- **Action hashes are exact, not graded.** A single differing elementary action counts as divergence; the size of the "
             "action change, and whether it mattered for the trajectory, is not recorded.")
    P.append("- **Dose summaries are per-population aggregates.** Coefficient norms and realized squared-L2 sums are the magnitudes "
             "applied, not their alignment with the goal direction; a constant dose can still have a state-dependent effect on E.")
    return P


# --------------------------------------------------------------------------- main
def main(argv=None):
    p = standard_parser("Planner dynamics: the edit's effect on CEM behaviour over an episode (exploratory).")
    p.add_argument("--cem-traces", type=Path, default=None,
                   help="optional directory of *.jsonl with fields " + ", ".join(CEM_TRACE_FIELDS))
    p.add_argument("--fixture-regime", default=None, help="informational label for synthetic fixtures")
    p.add_argument("--seed", type=int, default=BOOTSTRAP_SEED,
                   help="seed for every paired bootstrap and the strip-plot jitter in this script (default common.BOOTSTRAP_SEED)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    global SEED
    SEED = int(args.seed)
    np.random.seed(SEED)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = [t for t in args.tasks if (Path(args.results) / t).exists()]
    for t in args.tasks:
        if t not in tasks:
            LOG.warning("task %s: no results directory; skipped", t)

    prov = provenance(args.results, args.freeze, args.analysis)
    analysis = load_analysis(args.analysis)
    LOG.info("loading episodes frame")
    edf = episodes_frame(args.results, tasks)
    if not len(edf):
        raise SystemExit("no completed scenarios found")
    LOG.info("walking calls[] for planning-call structure")
    call_walk, ep_walk = walk_calls(args.results, tasks)
    regime = regime_from_report(analysis, edf, tasks)
    LOG.info("regime: %s (%s)", regime["label"], regime["basis"])
    mach, mrows = machinery(edf, ep_walk, tasks)
    for f in mach["flags"]:
        LOG.warning("machinery flag: %s", f)
    LOG.info("action divergence")
    div, tt, ct, div_all, nfam = divergence_block(args.results, edf, ep_walk, tasks)
    cross = commitment_vs_net(ct, analysis) if len(ct) else dict(exploratory=True, n_cells=0)
    LOG.info("dose over time")
    dose, dose_frame = dose_block(args.results, call_walk, tasks)
    cem, cem_frame = cem_traces_block(args.cem_traces, tasks)
    if not cem["available"]:
        LOG.info("CEM traces skipped: %s", cem["reason"])

    figs = {}
    fig_overhead(mrows, tasks, out / f"{SCRIPT}_overhead.png")
    figs["overhead"] = f"{SCRIPT}_overhead.png"
    if len(div_all):
        fig_divergence(div_all, tasks, out / f"{SCRIPT}_divergence.png")
        figs["divergence"] = f"{SCRIPT}_divergence.png"
        if fig_timing(div_all, tasks, out / f"{SCRIPT}_timing.png"):
            figs["timing"] = f"{SCRIPT}_timing.png"
    if dose.get("available") and fig_dose(dose, tasks, out / f"{SCRIPT}_dose.png"):
        figs["dose"] = f"{SCRIPT}_dose.png"
    if cem.get("available"):
        fig_cem_traces(cem, out / f"{SCRIPT}_cem_traces.png")
        figs["cem"] = f"{SCRIPT}_cem_traces.png"

    result = dict(script=SCRIPT, exploratory=True, provenance=prov, fixture_regime=args.fixture_regime, seed=SEED,
                  seed_note="SEED is passed to every paired bootstrap (common.paired_bootstrap seed=) and the strip-plot jitter rng",
                  tasks=tasks, regime=regime, planning_machinery=mach,
                  divergence=dict(exploratory=True, per_task=div,
                                  timing_tests=dict(exploratory=True, family=nfam["timing"],
                                                    family_description=f"Holm over the {nfam['timing']} Mann-Whitney tests with >= 3 scenarios per group "
                                                                       f"(of {len(tt)} listed; the rest have p = nan and are not counted)",
                                                    tests=tt.to_dict("records") if len(tt) else []),
                                  commitment_tests=dict(exploratory=True, family=nfam["commitment"],
                                                        family_description=f"Holm over the {nfam['commitment']} runnable per-cell Spearman tests among divergers "
                                                                           f"(of {len(ct)} cells; constant inputs give p = nan and are not counted); "
                                                                           "restricted to divergers because a flip requires divergence",
                                                        tests=ct.to_dict("records") if len(ct) else []),
                                  commitment_vs_net=cross),
                  dose_over_time=dose, cem_traces=cem, figures=figs,
                  notes=["Only arm-vs-native success contrasts (regime.cells[].vs_native) are pre-registered; every other block is exploratory.",
                         f"Bootstrap: scenario-cluster percentile, {common.BOOTSTRAP_DRAWS} draws, seed {SEED} (common.paired_bootstrap).",
                         "regime.label is derived from intervals only (classify_regime cells + OVERALL_RULE); descriptive_lean is metadata."])
    write_json(out / f"{SCRIPT}.json", result)
    (out / f"{SCRIPT}.md").write_text(narrative(regime, mach, mrows, div, tt, ct, cross, dose, cem, tasks, prov, args, figs, nfam))
    LOG.info("wrote %s", out / f"{SCRIPT}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
