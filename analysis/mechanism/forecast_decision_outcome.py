"""Forecast -> decision -> outcome chain for the fresh four-task confirmation.

Question. Does correcting the predictor's forecast change the CEM planner's decisions, and do
decision changes change closed-loop outcomes? We assemble, per (task, arm) cell, three levels:

  1. forecast   : offline H6 forecast-error reduction vs native (BF16; proprio primary, visual companion),
                  from the frozen offline contrast export (development pools, NOT the fresh scenarios);
  2. decision   : (a) the development first-population diagnostic (Reach/Reach-Wall only), and
                  (b) the FRESH decision proxy: whether the arm's chosen actions at the FIRST planning
                  call already differ from native (identical action hashes = identical actions);
  3. outcome    : fresh paired success gain vs native (rescues, regressions, net), scenario-cluster
                  paired bootstrap.

Only arm-vs-native success contrasts are pre-registered (the frozen report.json). Everything else in
this script is EXPLORATORY / post hoc and is labelled so in every output.

Evidence tiers. A learned-cell effect counts as detected only at a corrected level: tier 1 = the
pre-registered Bonferroni simultaneous interval; tier 2 = this script's exploratory Bonferroni interval
over every success contrast it bootstraps. Unadjusted 95 % intervals are reported as an explicitly
UNCORRECTED SCREENING line that never sets the regime or a verdict (under a global null at least one
of k unadjusted intervals excludes zero with probability 1 - 0.95^k, ~34 % for k = 8).

Per-iteration CEM losses were not recorded upstream, so the decision proxy is the chosen-action hash
(did the planner pick a different sequence?), not the score margin between candidates.

Read-only over finished artifacts. Never calls the model, simulator or GPU. Does not modify
src/offline_study or analysis/mechanism/common.py.
"""
from __future__ import annotations

import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

NAME = "forecast_decision_outcome"

# Exploratory reading thresholds (fixed before running on real data; stated in every output).
DECISION_RARE_FRACTION = 0.05      # first decision changes in < 5 % of scenarios -> "rare"
FLIPS_SUBSTANTIAL_FRACTION = 0.10  # >= 10 % of paired scenarios change outcome -> "substantial"
CONCORDANCE_SUPPORT_FRACTION = 0.75
ARM_CLASS = {a: "learned" for a in common.LEARNED}
ARM_CLASS.update({c: "control" for c in common.LEARNED.values()})
ARM_CLASS.update({a: "component" for a in common.COMPONENT})
SHORT_ARM = {"fixed_rank4": "R4", "matched_random_fixed_rank4": "R4-rand", "coupling_only": "Cpl",
             "matched_random_coupling": "Cpl-rand", "joint": "Joint", "visual_only": "Vis",
             "action_condition_only": "Act"}
SHORT_TASK = {"reach": "Reach", "reach-wall": "RWall", "pointmaze": "PMaze", "wall": "Wall"}
CONTROL_OF = {c: l for l, c in common.LEARNED.items()}
# Determinism: every bootstrap goes through common.paired_bootstrap, which seeds its own generator with
# common.BOOTSTRAP_SEED. No other randomness is used in this script.


# ----------------------------------------------------------------------------- helpers
class Log:
    def __init__(self):
        self.lines = []

    def __call__(self, msg):
        self.lines.append(str(msg))
        print(f"[{NAME}] {msg}", file=sys.stderr)


def fmt(v, nd=2, unit=""):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:.{nd}f}{unit}"


def ci(lo, hi, nd=1):
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return "n/a"
    return f"[{lo:+.{nd}f}, {hi:+.{nd}f}]"


def excludes_zero(d):
    return d is not None and np.isfinite(d.get("lower", np.nan)) and (d["lower"] > 0 or d["upper"] < 0)


def sign_of(d):
    if d is None or not np.isfinite(d.get("lower", np.nan)):
        return "0"
    if d["lower"] > 0:
        return "+"
    if d["upper"] < 0:
        return "-"
    return "0"


def offline_pair(task, arm, log):
    """Offline forecast effect (proprio + visual) for a behavioral arm, or None with a logged reason."""
    names = [common.OFFLINE_ARM_NAME.get(arm, arm)]
    if arm in common.OFFLINE_ARM_NAME:
        names.append(arm)  # literal fallback (reach has a same-named row in the combined sweep)
    for name in names:
        prop = common.offline_effect(task, name, endpoint="proprio_mse_h6")
        vis = common.offline_effect(task, name, endpoint="visual_mse_h6")
        if prop is not None:
            return dict(source_candidate=name, proprio=prop, visual=vis)
    log(f"offline forecast effect absent for task={task} arm={arm} (looked up {names}); forecast leg skipped for this cell")
    return None


def registered_contrast(report, task, arm, log):
    try:
        c = report["results"][task]["contrasts"][f"{arm}-native"]
        lo, hi = c["simultaneous_95_interval_pp"]
        return dict(difference=float(c["difference_pp"]), lower=float(lo), upper=float(hi),
                    family=int(report.get("analysis", {}).get("family_size", 0)))
    except (KeyError, TypeError, ValueError) as e:
        log(f"task {task} arm {arm}: pre-registered contrast missing or malformed in report.json ({type(e).__name__}: {e}); "
            "registered interval absent for this cell")
        return None


def decision_proxy(results, task, arm, index, log, reference="native"):
    """Fresh decision proxy on the scenarios in `index`.

    Returns (summary dict, DataFrame indexed by the VALID episodes) or (None, None). Scenarios whose arm or
    native record has no usable action trace (n_calls == 0, or absent from the divergence frame) are
    excluded from every rate and listed in the summary and the log.
    """
    d = common.action_divergence(results, task, arm, reference=reference)
    if not len(d):
        log(f"task {task} arm {arm}: no action traces at all; decision proxy skipped")
        return None, None
    d = d.set_index("episode").reindex(index)
    missing = d.n_calls.isna() | (d.n_calls.fillna(0) <= 0) | d.first_divergence.isna()
    n_missing = int(missing.sum())
    missing_eps = [int(e) for e in d.index[missing.values]]
    if n_missing:
        log(f"task {task} arm {arm}: {n_missing} scenario(s) without a usable action trace in the arm or native record "
            f"(episodes {missing_eps}); excluded from the decision proxy, not counted as unchanged")
    v = d[~missing.values]
    if not len(v):
        log(f"task {task} arm {arm}: every scenario lacks a usable action trace; decision proxy skipped")
        return None, None
    n = int(len(v))
    diverged = v.first_divergence >= 0
    first0 = (v.first_divergence == 0)
    out = dict(
        n=n, n_missing_trace=n_missing, missing_trace_episodes=missing_eps,
        n_calls_median=float(v.n_calls.median()),
        single_call_task=bool(v.n_calls.max() == 1),
        first_decision_changed=int(first0.sum()),
        first_decision_change_rate=float(first0.mean()),
        any_divergence=int(diverged.sum()),
        any_divergence_rate=float(diverged.mean()),
        identical_throughout=int((~diverged).sum()),
        mean_first_divergence_fraction_among_divergent=float(v.loc[diverged, "first_divergence_fraction"].mean()) if diverged.any() else float("nan"),
        mean_first_divergence_call_among_divergent=float(v.loc[diverged, "first_divergence"].mean()) if diverged.any() else float("nan"),
        mean_divergent_calls=float(v.n_divergent_calls.mean()),
        mean_divergent_call_fraction=float((v.n_divergent_calls / v.n_calls.clip(lower=1)).mean()),
    )
    return out, v


def regime_from(signs):
    s = set(signs.values())
    if "+" in s and "-" in s:
        return "mixed"
    if "+" in s:
        return "positive"
    if "-" in s:
        return "negative"
    return "inconclusive"


def opposite_signs_same_task(signs, label_to_task):
    """For a 'mixed' sign pattern: do a '+' and a '-' occur on the same task?"""
    pos = {label_to_task[k] for k, v in signs.items() if v == "+"}
    neg = {label_to_task[k] for k, v in signs.items() if v == "-"}
    return sorted(pos & neg)


# ----------------------------------------------------------------------------- main
def main():
    ap = common.standard_parser(__doc__.split("\n\n")[0])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = Log()

    prov = common.provenance(args.results, args.freeze, args.analysis)
    prov["generated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report = common.load_analysis(args.analysis)
    planted = report.get("regime") if prov["synthetic"] else None
    tasks = [t for t in args.tasks if t in common.TASKS]

    log(f"loading episodes from {args.results} (tasks={tasks})")
    df = common.episodes_frame(args.results, tasks=tasks)
    if not len(df):
        raise SystemExit("no completed scenarios found")
    tasks_present = [t for t in tasks if (df.task == t).any()]
    for t in tasks:
        if t not in tasks_present:
            log(f"task {t} has no completed scenarios in the results tree; skipped")

    # development decision diagnostic (Reach / Reach-Wall, shared first population)
    try:
        dev = common.load_decision_diagnostic()
        log(f"development decision diagnostic loaded: {len(dev)} rows (reach/reach-wall only)")
    except Exception as e:  # pragma: no cover - degrade
        dev = pd.DataFrame(columns=["task", "arm", "changed_selections", "spearman", "top10_overlap"])
        log(f"development decision diagnostic unavailable ({e}); development decision leg skipped")

    # ---------------------------------------------------------------- per-cell assembly
    cells = []
    arms_non_native = [a for a in common.ARMS if a != "native"]
    # exploratory bootstrap family for SUCCESS contrasts: every one this script bootstraps
    n_vs_native = sum(1 for t in tasks_present for a in arms_non_native if a in set(df[df.task == t].arm))
    n_vs_control = sum(1 for t in tasks_present for a in common.LEARNED
                       if {a, common.LEARNED[a]} <= set(df[df.task == t].arm))
    family_success = n_vs_native + n_vs_control
    family_note = (f"exploratory Bonferroni family for the SUCCESS bootstraps in this script = {family_success} "
                   f"({n_vs_native} arm-vs-native + {n_vs_control} learned-vs-control paired contrasts over the tasks present)")
    log(family_note)

    for task in tasks_present:
        arms_here = [a for a in common.ARMS if a in set(df[df.task == task].arm)]
        for arm in common.ARMS:
            if arm not in arms_here:
                log(f"task {task}: arm {arm} absent from results tree; cell skipped")
        if "native" not in arms_here:
            log(f"task {task}: native arm missing; task skipped")
            continue
        S = common.success_matrix(df, task, arms=arms_here)
        n = int(len(S))
        n_any = int(df[df.task == task].episode.nunique())
        if n < n_any:
            log(f"task {task}: {n_any - n} scenario(s) incomplete in at least one arm; n = {n} complete-case scenarios used for every cell")
        native = S["native"].values
        for arm in arms_here:
            if arm == "native":
                continue
            cell = dict(task=task, arm=arm, arm_class=ARM_CLASS.get(arm, "other"),
                        label=f"{SHORT_TASK.get(task, task)}/{SHORT_ARM.get(arm, arm)}", n=n,
                        exploratory=True)
            a = S[arm].values
            # outcome level
            cell["success_native_pct"] = float(native.mean() * 100)
            cell["success_arm_pct"] = float(a.mean() * 100)
            cell["registered_vs_native"] = registered_contrast(report, task, arm, log)
            cell["gain_vs_native_bonferroni"] = common.paired_bootstrap(a, native, family=family_success)
            cell["gain_vs_native_unadjusted"] = common.paired_bootstrap(a, native, family=1)
            cell["gain_vs_native_unadjusted"]["role"] = "uncorrected screening only; never sets a verdict or the regime"
            disc = common.discordance(a, native)
            cell["rescues"] = disc["wins"]
            cell["regressions"] = disc["losses"]
            cell["flips"] = disc["discordant"]
            cell["flip_rate"] = disc["discordant"] / n
            cell["flip_sign_test_p_unadjusted"] = disc["exact_p"]
            if arm in common.LEARNED and common.LEARNED[arm] in arms_here:
                ctl = S[common.LEARNED[arm]].values
                cell["control_arm"] = common.LEARNED[arm]
                cell["gain_vs_control_bonferroni"] = common.paired_bootstrap(a, ctl, family=family_success)
                cell["gain_vs_control_unadjusted"] = common.paired_bootstrap(a, ctl, family=1)
                cell["flips_vs_control"] = common.discordance(a, ctl)
            # fresh decision proxy (rates over scenarios with a usable trace; CIs added in the second pass)
            proxy, v = decision_proxy(args.results, task, arm, S.index, log)
            if proxy is None:
                cell["decision_fresh"] = None
            else:
                valid = S.index.isin(v.index)
                first0 = (v.first_divergence == 0).values
                identical = (v.first_divergence == -1).values
                flipped = (a != native)[valid]
                proxy["flip_rate_given_first_decision_changed"] = float(flipped[first0].mean()) if first0.any() else float("nan")
                proxy["flip_rate_given_first_decision_unchanged"] = float(flipped[~first0].mean()) if (~first0).any() else float("nan")
                proxy["flips_without_first_decision_change"] = int((flipped & ~first0).sum())
                proxy["share_of_flips_without_first_decision_change"] = (
                    float((flipped & ~first0).sum() / flipped.sum()) if flipped.any() else float("nan"))
                proxy["flips_with_identical_actions_throughout"] = int((flipped & identical).sum())
                proxy["n_identical_actions_throughout"] = int(identical.sum())
                cell["decision_fresh"] = proxy
                cell["_first0"] = pd.Series(first0.astype(float), index=v.index)  # private; dropped before JSON
            # development decision diagnostic
            row = dev[(dev.task == task) & (dev.arm == arm)]
            cell["decision_development"] = (
                dict(changed_selections_of_96=int(row.iloc[0].changed_selections), spearman=float(row.iloc[0].spearman),
                     top10_overlap=float(row.iloc[0].top10_overlap)) if len(row) else None)
            if not len(row) and task in ("reach", "reach-wall"):
                log(f"task {task} arm {arm}: not in the development decision diagnostic (five-arm panel); dev leg absent")
            # offline forecast
            cell["forecast"] = offline_pair(task, arm, log)
            cells.append(cell)

    if not cells:
        raise SystemExit("no cells assembled")

    # ---------------------------------------------------------------- decision-proxy family (second pass)
    by_key = {(c["task"], c["arm"]): c for c in cells}
    n_rate_cis = sum(1 for c in cells if "_first0" in c)
    dose_pairs = [(c, by_key.get((c["task"], c.get("control_arm")))) for c in cells if c["arm_class"] == "learned"]
    dose_pairs = [(c, k) for c, k in dose_pairs if k is not None and "_first0" in c and "_first0" in k]
    family_decision = n_rate_cis + len(dose_pairs)
    decision_family_note = (f"exploratory Bonferroni family for the DECISION-PROXY bootstraps in this script = {family_decision} "
                            f"({n_rate_cis} first-decision change-rate intervals + {len(dose_pairs)} learned-vs-control "
                            f"first-decision contrasts)")
    log(decision_family_note)
    for c in cells:
        if "_first0" in c:
            f0 = c["_first0"].values
            c["decision_fresh"]["first_decision_change_rate_ci"] = common.paired_bootstrap(
                f0, np.zeros(len(f0)), family=family_decision)
    dose_generic = []
    for c, k in dose_pairs:
        shared = c["_first0"].index.intersection(k["_first0"].index)
        fl, fc = c["_first0"].loc[shared].values, k["_first0"].loc[shared].values
        bs = common.paired_bootstrap(fl, fc, family=family_decision)
        disc = common.discordance(fl.astype(bool), fc.astype(bool))
        if bs["lower"] > 0:
            verdict, code = ("learned perturbs the first decision more than its control (exploratory Bonferroni-"
                             f"{family_decision} interval excludes zero)"), "learned_more"
        elif bs["upper"] < 0:
            verdict, code = ("learned perturbs the first decision less than its control (exploratory Bonferroni-"
                             f"{family_decision} interval excludes zero)"), "learned_less"
        else:
            lean = ("higher" if bs["difference"] > 0 else "lower" if bs["difference"] < 0 else "equal")
            verdict, code = (f"point estimate {lean}, not resolved (interval includes zero); consistent with a dose-generic "
                             "perturbation"), "unresolved"
        dose_generic.append(dict(
            task=c["task"], learned=c["arm"], control=k["arm"], n_shared=int(len(shared)),
            learned_changed=int(fl.sum()), control_changed=int(fc.sum()),
            learned_rate=float(fl.mean()), control_rate=float(fc.mean()),
            learned_flips=c["flips"], control_flips=k["flips"],
            contrast_pp=common.paired_bootstrap(fl, fc, family=family_decision),
            discordant_scenarios=dict(learned_only=disc["wins"], control_only=disc["losses"],
                                      exact_sign_test_p_unadjusted=disc["exact_p"]),
            verdict=verdict, code=code, exploratory=True))
    for c in cells:
        c.pop("_first0", None)

    # ---------------------------------------------------------------- chain table
    def g(c, *keys, default=np.nan):
        v = c
        for k in keys:
            if v is None:
                return default
            v = v.get(k) if isinstance(v, dict) else default
        return default if v is None else v

    chain = pd.DataFrame([dict(
        task=c["task"], arm=c["arm"], cls=c["arm_class"], label=c["label"], n=c["n"],
        forecast_proprio=g(c, "forecast", "proprio", "effect"),
        forecast_proprio_lo=g(c, "forecast", "proprio", "lower"), forecast_proprio_hi=g(c, "forecast", "proprio", "upper"),
        forecast_visual=g(c, "forecast", "visual", "effect"),
        forecast_visual_lo=g(c, "forecast", "visual", "lower"), forecast_visual_hi=g(c, "forecast", "visual", "upper"),
        dev_changed_selections=g(c, "decision_development", "changed_selections_of_96"),
        dev_spearman=g(c, "decision_development", "spearman"),
        first_decision_change_rate=g(c, "decision_fresh", "first_decision_change_rate"),
        first_decision_changed=g(c, "decision_fresh", "first_decision_changed"),
        n_traced=g(c, "decision_fresh", "n"),
        n_missing_trace=g(c, "decision_fresh", "n_missing_trace"),
        any_divergence_rate=g(c, "decision_fresh", "any_divergence_rate"),
        mean_first_div_fraction=g(c, "decision_fresh", "mean_first_divergence_fraction_among_divergent"),
        mean_divergent_calls=g(c, "decision_fresh", "mean_divergent_calls"),
        single_call_task=bool(g(c, "decision_fresh", "single_call_task", default=False)),
        rescues=c["rescues"], regressions=c["regressions"], flips=c["flips"], flip_rate=c["flip_rate"],
        net_gain_pp=c["gain_vs_native_bonferroni"]["difference"],
        net_lo=c["gain_vs_native_bonferroni"]["lower"], net_hi=c["gain_vs_native_bonferroni"]["upper"],
        unadj_lo=c["gain_vs_native_unadjusted"]["lower"], unadj_hi=c["gain_vs_native_unadjusted"]["upper"],
        reg_lo=g(c, "registered_vs_native", "lower"), reg_hi=g(c, "registered_vs_native", "upper"),
        gain_vs_control_pp=g(c, "gain_vs_control_bonferroni", "difference"),
        ctl_lo=g(c, "gain_vs_control_bonferroni", "lower"), ctl_hi=g(c, "gain_vs_control_bonferroni", "upper"),
    ) for c in cells])
    chain["abs_net_gain_pp"] = chain.net_gain_pp.abs()

    # ---------------------------------------------------------------- cross-cell correlations (exploratory)
    corr_specs = [
        ("proprio_forecast_vs_success_gain", "forecast_proprio", "net_gain_pp"),
        ("visual_forecast_vs_success_gain", "forecast_visual", "net_gain_pp"),
        ("first_decision_change_vs_abs_success_gain", "first_decision_change_rate", "abs_net_gain_pp"),
        ("first_decision_change_vs_flips", "first_decision_change_rate", "flips"),
        ("proprio_forecast_vs_first_decision_change", "forecast_proprio", "first_decision_change_rate"),
        ("visual_vs_proprio_forecast", "forecast_visual", "forecast_proprio"),
    ]
    corrs = []
    for name, x, y in corr_specs:
        r = common.spearman(chain[x], chain[y])
        corrs.append(dict(name=name, x=x, y=y, **r))
    pvals = [c["p"] if np.isfinite(c["p"]) else 1.0 for c in corrs]
    adj = common.holm(pvals)
    for c, p_adj in zip(corrs, adj):
        c["p_holm"] = float(p_adj) if np.isfinite(c["p"]) else float("nan")
        c["family"] = len(corrs)
    corr_by = {c["name"]: c for c in corrs}
    # concordance of signs (forecast reduction > 0 vs success gain > 0)
    both = chain.dropna(subset=["forecast_proprio"])
    conc = dict(n=int(len(both)),
                agree=int(((both.forecast_proprio > 0) == (both.net_gain_pp > 0)).sum()) if len(both) else 0)
    conc["fraction"] = conc["agree"] / conc["n"] if conc["n"] else float("nan")
    if conc["n"]:
        from scipy.stats import binomtest
        conc["binomial_p_unadjusted"] = float(binomtest(conc["agree"], conc["n"], 0.5).pvalue)

    # ---------------------------------------------------------------- regime (learned arms vs native)
    learned_cells = [c for c in cells if c["arm_class"] == "learned"]
    label_task = {c["label"]: c["task"] for c in cells}
    tier1 = {c["label"]: sign_of(c["registered_vs_native"]) for c in learned_cells}
    tier2 = {c["label"]: sign_of(c["gain_vs_native_bonferroni"]) for c in learned_cells}
    screening = {c["label"]: sign_of(c["gain_vs_native_unadjusted"]) for c in learned_cells}
    k_learned = len(learned_cells)
    false_sign_rate = float(1 - 0.95 ** k_learned) if k_learned else float("nan")
    reg_family = next((c["registered_vs_native"]["family"] for c in learned_cells if c["registered_vs_native"]), None)

    regime_t1, regime_t2, regime_scr = regime_from(tier1), regime_from(tier2), regime_from(screening)
    if regime_t1 != "inconclusive":
        regime, tier_id = regime_t1, 1
        tier = f"tier 1: pre-registered Bonferroni simultaneous 95 % intervals, family {reg_family}"
    elif regime_t2 != "inconclusive":
        regime, tier_id = regime_t2, 2
        tier = (f"tier 2: exploratory Bonferroni simultaneous 95 % intervals, family {family_success} (this script; "
                "not pre-registered)")
    else:
        regime, tier_id = "inconclusive", 0
        tier = f"no learned cell's interval excludes zero at tier 1 (family {reg_family}) or tier 2 (family {family_success})"
    screening_regime = "none" if regime_scr == "inconclusive" else f"suggestive_{regime_scr}"
    per_cell_class = {c["label"]: (common.classify_regime(c["registered_vs_native"], c["gain_vs_control_bonferroni"])
                                   if c.get("registered_vs_native") and c.get("gain_vs_control_bonferroni") else "n/a")
                      for c in learned_cells}
    lchain = chain[chain.cls == "learned"]
    hetero = dict(pos=int((lchain.net_gain_pp >= common.USEFUL_GAIN_PP).sum()),
                  neg=int((lchain.net_gain_pp <= -common.USEFUL_GAIN_PP).sum()), n=int(len(lchain)))
    regime_signs = tier1 if tier_id == 1 else tier2 if tier_id == 2 else {}
    mixed_same_task = opposite_signs_same_task(regime_signs, label_task) if regime == "mixed" else []
    screening_same_task = opposite_signs_same_task(screening, label_task)

    # ---------------------------------------------------------------- mediation-style verdict (rules)
    def cell_verdict(c):
        p = c["decision_fresh"]
        if p is None:
            return dict(code="no_decision_proxy", text="no usable action traces, so the decision level cannot be read")
        rate = p["first_decision_change_rate"]
        rare = rate < DECISION_RARE_FRACTION
        substantial = c["flip_rate"] >= FLIPS_SUBSTANTIAL_FRACTION
        balanced = c["flip_sign_test_p_unadjusted"] >= 0.05
        detected_t1 = excludes_zero(c["registered_vs_native"])
        detected_t2 = excludes_zero(c["gain_vs_native_bonferroni"])
        screened = excludes_zero(c["gain_vs_native_unadjusted"])
        detected = detected_t1 or detected_t2
        b = c["gain_vs_native_bonferroni"]
        span = f"exploratory interval {ci(b['lower'], b['upper'])} pp"
        # direction is read off the interval that triggered detection, never off a bare point estimate
        detecting = c["registered_vs_native"] if detected_t1 else b
        direction = "gain" if detecting["lower"] > 0 else "loss" if detecting["upper"] < 0 else (
            "gain" if b["difference"] > 0 else "loss")
        single = p["single_call_task"]
        det = ("detected at the pre-registered level" if detected_t1 else
               f"detected at the exploratory Bonferroni-{b['family']} level (not pre-registered)" if detected_t2 else
               f"not detected at any corrected level ({span}; the uncorrected screening interval excludes zero, suggestive only)"
               if screened else f"not detected ({span})")
        nn = f"{p['first_decision_changed']}/{p['n']}"
        if single:
            # one planning call: the first decision is the whole plan
            if rare and not substantial:
                return dict(code="inert", text=f"single-plan task: the plan changes in {nn} scenarios and outcomes flip in "
                            f"{c['flips']}; the edit is behaviorally inert at this dose")
            if detected:
                return dict(code="decision_mediated_" + direction, text=f"single-plan task: the plan changes in {nn} scenarios "
                            f"and the net {direction} is {det}; here every outcome change necessarily passes through the (only) decision")
            return dict(code="rerank_equal", text=f"single-plan task: the plan changes in {nn} scenarios, {c['rescues']} rescues vs "
                        f"{c['regressions']} regressions; no net difference detected at this sample size ({span}), so the "
                        "alternatives are not distinguishable from equally good")
        share_wo = p["share_of_flips_without_first_decision_change"]
        if substantial and (rare or (np.isfinite(share_wo) and share_wo >= 0.5)):
            where = (f"{p['flips_without_first_decision_change']} of the {c['flips']} flips occur in scenarios whose first "
                     f"decision was unchanged") if np.isfinite(share_wo) else "no first decision changed"
            if detected:
                return dict(code="replanning_" + direction, text=f"outcomes change ({c['flips']} flips) while the first decision "
                            f"changes in {nn} and {where}; the net {direction} ({det}) accrues through replanning dynamics, not "
                            "through a better initial choice")
            return dict(code="replanning_noise", text=f"outcomes change ({c['flips']} flips: {c['rescues']} rescues, "
                        f"{c['regressions']} regressions) largely without first-decision change ({nn} changed; {where}); any effect "
                        f"would accrue through replanning dynamics, and the net is {det}")
        if not rare and not detected and balanced:
            return dict(code="rerank_equal", text=f"first decisions change in {nn} scenarios with {c['rescues']} rescues vs "
                        f"{c['regressions']} regressions; no net difference detected at this sample size ({span}), so the "
                        "alternatives are not distinguishable from equally good")
        if not rare and detected:
            return dict(code="decision_mediated_" + direction, text=f"first decisions change in {nn} scenarios and the net "
                        f"{direction} is {det}; consistent with a decision-mediated effect")
        if rare and not substantial:
            return dict(code="inert", text=f"neither first decisions ({nn}) nor outcomes ({c['flips']} flips) change materially; "
                        "the edit is behaviorally inert at this dose")
        return dict(code="unresolved", text=f"first decisions change in {nn}, {c['flips']} flips, net effect {det}; no single "
                    "mediation reading fits")

    for c in cells:
        c["verdict"] = cell_verdict(c)

    codes = [c["verdict"]["code"] for c in learned_cells]
    rho_fs = corr_by["proprio_forecast_vs_success_gain"]
    program_support = (np.isfinite(rho_fs["rho"]) and rho_fs["rho"] > 0 and rho_fs["p_holm"] < 0.05
                       and conc["fraction"] >= CONCORDANCE_SUPPORT_FRACTION)
    any_detected = any(excludes_zero(c["registered_vs_native"]) or excludes_zero(c["gain_vs_native_bonferroni"])
                       for c in learned_cells)
    pooled = dict(
        regime=regime, regime_evidence_tier=tier, regime_tier_id=tier_id,
        regime_tier1_signs=tier1, regime_tier2_signs=tier2,
        screening_signs_uncorrected=screening, screening_regime_uncorrected=screening_regime,
        screening_false_sign_rate_under_global_null=false_sign_rate, n_learned_cells=k_learned,
        mixed_opposite_signs_same_task=mixed_same_task, screening_opposite_signs_same_task=screening_same_task,
        any_learned_cell_detected_at_corrected_level=bool(any_detected),
        per_cell_classification=per_cell_class, point_estimate_heterogeneity=hetero,
        learned_cell_verdict_codes=dict(zip([c["label"] for c in learned_cells], codes)),
        dominant_verdict=max(set(codes), key=codes.count) if codes else "n/a",
        forecast_outcome_program_support=bool(program_support),
        concordance=conc, dose_generic_check=dose_generic,
        thresholds=dict(decision_rare_fraction=DECISION_RARE_FRACTION, flips_substantial_fraction=FLIPS_SUBSTANTIAL_FRACTION,
                        concordance_support_fraction=CONCORDANCE_SUPPORT_FRACTION, useful_gain_pp=common.USEFUL_GAIN_PP),
    )
    if planted is not None:
        pooled["fixture_planted_regime_label"] = planted
        expected = "inconclusive" if planted == "null" else planted
        pooled["fixture_label_matches_derived_regime"] = bool(regime == expected)
        pooled["fixture_label_matches_uncorrected_screening"] = bool(
            (screening_regime == "none" and expected == "inconclusive") or screening_regime == f"suggestive_{expected}")

    # ---------------------------------------------------------------- figures
    make_figures(chain, out, prov["synthetic"])

    # ---------------------------------------------------------------- write JSON
    result = dict(
        script=NAME, exploratory=True,
        exploratory_note=("Only arm-vs-native success contrasts are pre-registered (frozen report.json). The chain assembly, "
                          "fresh decision proxy, learned-vs-control contrasts, cross-cell correlations and the mediation "
                          "verdict are exploratory / post hoc."),
        families=dict(success_bootstraps=family_success, success_family_note=family_note,
                      decision_bootstraps=family_decision, decision_family_note=decision_family_note,
                      correlations=len(corrs), correlation_note="Holm over the six cross-cell Spearman correlations",
                      registered=reg_family,
                      screening_note=("unadjusted 95 % intervals are reported as uncorrected screening only; they never set the "
                                      "regime or a verdict")),
        decision_proxy_note=("Per-iteration CEM losses were not recorded upstream; the fresh decision proxy is the SHA-256 of "
                             "the chosen action sequence at each planning call compared with native on the same scenario and "
                             "RNG stream. It detects that the planner chose differently, not by how much its scores moved."),
        provenance=prov, tasks=tasks_present, cells=cells, chain_table=chain.to_dict(orient="records"),
        correlations=corrs, pooled=pooled, log=log.lines,
    )
    common.write_json(out / f"{NAME}.json", _clean(result))
    (out / f"{NAME}.md").write_text(render_md(result, chain, corrs, pooled, cells, log))
    log(f"wrote {out / (NAME + '.json')}, {out / (NAME + '.md')}, figures")


def _clean(o):
    """Replace non-finite floats with None so the JSON is strictly valid."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


# ----------------------------------------------------------------------------- figures
LABEL_OFFSETS = [(4, 3), (4, -9), (-4, 3), (-4, -9), (4, 10), (-4, 10), (4, -16), (-4, -16),
                 (12, 3), (-12, 3), (12, -9), (-12, -9), (0, 12), (0, -18), (18, 3), (-18, 3), (18, -9), (-18, -9)]


def place_labels(ax, renderer, pts, fontsize, color="#333"):
    """Greedy label repulsion in display pixels: try offsets in order, keep the first that overlaps no
    placed label, no data point and stays inside the axes; otherwise keep the least-overlapping one."""
    from matplotlib.transforms import Bbox
    axbb = ax.get_window_extent(renderer)
    all_disp = ax.transData.transform([(x, y) for x, y, _ in pts]) if pts else np.zeros((0, 2))
    placed = []
    for i, (x, y, text) in enumerate(pts):
        best, best_cost = None, None
        for dx, dy in LABEL_OFFSETS:
            ha = "left" if dx > 0 else ("right" if dx < 0 else "center")
            ann = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=fontsize,
                              color=color, ha=ha, va="bottom", zorder=5)
            bb = ann.get_window_extent(renderer).padded(1.0)
            cost = 0.0
            for pb in placed:
                inter = Bbox.intersection(bb, pb)
                if inter is not None:
                    cost += inter.width * inter.height
            others = np.delete(all_disp, i, axis=0)
            if len(others):
                inside = (others[:, 0] >= bb.x0 - 2) & (others[:, 0] <= bb.x1 + 2) & (others[:, 1] >= bb.y0 - 2) & (others[:, 1] <= bb.y1 + 2)
                cost += 40.0 * inside.sum()
            if not (bb.x0 >= axbb.x0 and bb.x1 <= axbb.x1 and bb.y0 >= axbb.y0 and bb.y1 <= axbb.y1):
                cost += 30.0
            if best is None or cost < best_cost:
                if best is not None:
                    best[0].remove()
                best, best_cost = (ann, bb), cost
            else:
                ann.remove()
            if best_cost == 0:
                break
        placed.append(best[1])


def make_figures(chain, out, synthetic):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 200, "savefig.dpi": 200})
    colors = {"learned": "#1A8F7A", "control": "#8C8C8C", "component": "#1F5FA8"}
    markers = {"learned": "o", "control": "s", "component": "^"}
    # learned cells are labelled by name; controls and components carry a number keyed under the figure
    numbered = chain[chain.cls != "learned"].reset_index(drop=True)
    key = {lab: str(i + 1) for i, lab in enumerate(numbered.label)}
    chain = chain.assign(plot_label=[lab if cls == "learned" else key[lab] for lab, cls in zip(chain.label, chain.cls)])

    fig, axs = plt.subplots(2, 2, figsize=(8.6, 7.6))
    specs = [
        (axs[0, 0], "forecast_proprio", "net_gain_pp", ("forecast_proprio_lo", "forecast_proprio_hi"), ("net_lo", "net_hi"),
         "a  Forecast vs success", "H6 proprio error reduction vs native (%, BF16)", "Success gain vs native (pp)"),
        (axs[0, 1], "first_decision_change_rate", "net_gain_pp", None, ("net_lo", "net_hi"),
         "b  First-decision change vs success", "First planning call differs from native (fraction)", "Success gain vs native (pp)"),
        (axs[1, 0], "forecast_proprio", "first_decision_change_rate", ("forecast_proprio_lo", "forecast_proprio_hi"), None,
         "c  Forecast vs first-decision change", "H6 proprio error reduction vs native (%)", "First-decision change rate"),
        (axs[1, 1], "forecast_visual", "forecast_proprio", ("forecast_visual_lo", "forecast_visual_hi"),
         ("forecast_proprio_lo", "forecast_proprio_hi"), "d  Visual vs proprio forecast effect",
         "H6 visual error reduction vs native (%)", "H6 proprio error reduction vs native (%)"),
    ]
    pending = []
    for ax, x, y, xerr, yerr, title, xl, yl in specs:
        pts = []
        for cls, sub in chain.groupby("cls"):
            xs, ys = sub[x].values, sub[y].values
            ok = np.isfinite(xs) & np.isfinite(ys)
            if not ok.any():
                continue
            xe = None
            if xerr:
                xe = np.vstack([xs - sub[xerr[0]].values, sub[xerr[1]].values - xs])[:, ok]
                xe = np.where(np.isfinite(xe), xe, 0)
            ye = None
            if yerr:
                ye = np.vstack([ys - sub[yerr[0]].values, sub[yerr[1]].values - ys])[:, ok]
                ye = np.where(np.isfinite(ye), ye, 0)
            ax.errorbar(xs[ok], ys[ok], xerr=xe, yerr=ye, fmt=markers[cls], ms=4, color=colors[cls], ecolor=colors[cls],
                        elinewidth=0.6, capsize=0, alpha=0.9, label=cls, lw=0)
            pts += [(xi, yi, lab, cls) for xi, yi, lab in zip(xs[ok], ys[ok], sub.plot_label.values[ok])]
        ax.axhline(0, color="#bbb", lw=0.6, zorder=0)
        ax.axvline(0, color="#bbb", lw=0.6, zorder=0)
        ax.set_title(title, loc="left")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.margins(0.12)
        pending.append((ax, pts))
    h, l = axs[0, 0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.075))
    key_txt = "Numbered points (controls / components): " + "; ".join(f"{v} {k}" for k, v in key.items())
    fig.text(0.02, 0.005, "\n".join(textwrap.wrap(key_txt, 150)), fontsize=6.5, color="#333", va="bottom", ha="left")
    fig.suptitle(("SYNTHETIC FIXTURE - " if synthetic else "") + "Forecast, decision and outcome levels per (task, arm) cell; "
                 "y-bars: exploratory Bonferroni paired bootstrap, x-bars: offline simultaneous 95 %", fontsize=7.5, y=0.995)
    fig.tight_layout(rect=(0, 0.09, 1, 0.97))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, pts in pending:
        # learned cells first so they get the best positions, then the numbered controls / components
        ordered = [p for p in pts if p[3] == "learned"] + [p for p in pts if p[3] != "learned"]
        place_labels(ax, renderer, [(x, y, lab) for x, y, lab, _ in ordered], fontsize=6.5)
    fig.savefig(out / f"{NAME}_scatter.png", bbox_inches="tight")
    plt.close(fig)

    # chain bars: per task, per arm: first-decision change rate, flip rate, net gain
    tasks = list(dict.fromkeys(chain.task))
    fig, axs = plt.subplots(1, len(tasks), figsize=(1.9 * len(tasks) + 1, 3.0), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, task in zip(axs, tasks):
        sub = chain[chain.task == task]
        idx = np.arange(len(sub))
        w = 0.27
        ax.bar(idx - w, sub.first_decision_change_rate * 100, w, color="#1F5FA8", label="first decision changed (%)")
        ax.bar(idx, sub.flip_rate * 100, w, color="#E0A100", label="outcome flips (% of scenarios)")
        ax.bar(idx + w, sub.net_gain_pp, w, color="#1A8F7A", label="net success gain (pp)")
        ax.errorbar(idx + w, sub.net_gain_pp, yerr=np.vstack([sub.net_gain_pp - sub.net_lo, sub.net_hi - sub.net_gain_pp]),
                    fmt="none", ecolor="#0f5c4e", elinewidth=0.6)
        ax.axhline(0, color="#bbb", lw=0.6)
        ax.set_xticks(idx)
        ax.set_xticklabels([SHORT_ARM.get(a, a) for a in sub.arm], rotation=60, ha="right")
        ax.set_title(common.TASK_LABEL.get(task, task), loc="left")
    axs[0].set_ylabel("percent / pp")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(("SYNTHETIC FIXTURE - " if synthetic else "") + "Chain per cell: decision change -> flips -> net gain (vs native)",
                 fontsize=7.5)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(out / f"{NAME}_chain.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- markdown
def render_md(result, chain, corrs, pooled, cells, log):
    prov = result["provenance"]
    fam = result["families"]
    regime = pooled["regime"]
    tier_id = pooled["regime_tier_id"]
    L = []
    L.append(f"# Forecast -> decision -> outcome chain ({NAME})\n")
    if prov["synthetic"]:
        L.append("> **SYNTHETIC FIXTURE.** These numbers come from a schema-identical synthetic results tree, not from the "
                 "confirmation run. Nothing here is a scientific result.\n")
    L.append("> **EXPLORATORY.** Only the arm-vs-native success contrasts (frozen `report.json`, Bonferroni family "
             f"{fam['registered']}) are pre-registered. Everything else on this page - the chain assembly, the "
             "fresh decision proxy, learned-vs-control contrasts, the cross-cell correlations and the mediation verdict - is "
             f"post hoc. Families corrected over: {fam['success_family_note']}; {fam['decision_family_note']}; "
             f"{fam['correlation_note']}. {fam['screening_note'].capitalize()}.\n")
    L.append(f"Method `{prov['method']}`; freeze sha256 `{prov['freeze_sha256'][:16]}...`; tasks: "
             f"{', '.join(common.TASK_LABEL.get(t, t) for t in result['tasks'])}. In every cell, n is the number of scenarios "
             "complete in all arms present for that task (complete-case panel).\n")

    # ---- headline
    L.append("## Regime and headline reading\n")
    het = pooled["point_estimate_heterogeneity"]
    k = pooled["n_learned_cells"]
    if tier_id:
        L.append(f"Derived regime for the learned edits vs native: **{regime}** ({pooled['regime_evidence_tier']}).")
    else:
        L.append(f"Derived regime for the learned edits vs native: **inconclusive** ({pooled['regime_evidence_tier']}).")
    signs1 = ", ".join(f"{a}: {v}" for a, v in pooled["regime_tier1_signs"].items())
    signs2 = ", ".join(f"{a}: {v}" for a, v in pooled["regime_tier2_signs"].items())
    L.append(f"Tier 1 (pre-registered) interval signs per learned cell: {signs1}. Tier 2 (exploratory Bonferroni-"
             f"{fam['success_bootstraps']}) signs: {signs2}. Per-cell reading (`classify_regime`, vs native pre-registered and "
             "vs its own control exploratory): " + ", ".join(f"{a}: {v}" for a, v in pooled["per_cell_classification"].items()) + ".")
    lvl = ("pre-registered" if tier_id == 1 else f"exploratory Bonferroni-{fam['success_bootstraps']}")
    if regime == "positive":
        L.append(f"At least one learned cell's {lvl} interval lies above zero and none lies below it"
                 + (". " if tier_id == 1 else " (no pre-registered interval does, so this is an exploratory, not a confirmed, "
                    "positive regime). ")
                 + "This is the regime in which a forecast->outcome chain could exist; the sections below ask whether it "
                 "runs through the first decision.")
    elif regime == "negative":
        if tier_id == 1:
            L.append(f"At least one learned cell's pre-registered interval lies below zero and none above it: the edit is harmful "
                     "where it is detectable. The chain below asks whether the harm enters at the first decision or through replanning.")
        else:
            L.append(f"At least one learned cell's {lvl} interval lies below zero and none above it, while no pre-registered "
                     "interval excludes zero: the edit is harmful at the exploratory level only (not a confirmed harm). The chain "
                     "below asks whether that harm enters at the first decision or through replanning.")
    elif regime == "mixed":
        same = pooled["mixed_opposite_signs_same_task"]
        where = (f"within the same task ({', '.join(common.TASK_LABEL.get(t, t) for t in same)}: its two learned edits disagree)"
                 if same else "across tasks (no task carries both signs)")
        L.append(f"Learned cells disagree in sign at the {lvl} level, {where}. A single mechanism story is therefore not "
                 "available; the chain is read task by task below.")
    else:
        lean = ("with heterogeneous point estimates that the intervals cannot separate from noise" if het["pos"] and het["neg"]
                else "and a majority of learned point estimates lean positive but are not resolved" if het["pos"] > het["n"] / 2
                else "and a majority of learned point estimates lean negative but are not resolved" if het["neg"] > het["n"] / 2
                else "and the point estimates are small or not consistently signed")
        L.append(f"No learned cell's interval excludes zero at either corrected tier. Point estimates reach >= "
                 f"{common.USEFUL_GAIN_PP:g} pp in {het['pos']} and <= -{common.USEFUL_GAIN_PP:g} pp in {het['neg']} of {het['n']} "
                 f"learned cells, so the panel is inconclusive, {lean}.")
    scr = pooled["screening_signs_uncorrected"]
    n_scr = sum(v != "0" for v in scr.values())
    scr_txt = ", ".join(f"{a}: {v}" for a, v in scr.items())
    if n_scr:
        same = pooled["screening_opposite_signs_same_task"]
        pattern = pooled["screening_regime_uncorrected"].replace("suggestive_", "")
        L.append(f"Uncorrected screening (never sets the regime): {n_scr} of {k} learned cells' unadjusted 95 % intervals exclude "
                 f"zero ({scr_txt}), a `{pooled['screening_regime_uncorrected']}` pattern"
                 + (f" with opposite signs within {', '.join(common.TASK_LABEL.get(t, t) for t in same)}" if pattern == "mixed" and same else "")
                 + f". Under a global null the chance that at least one of {k} unadjusted intervals excludes zero is "
                 f"{100 * pooled['screening_false_sign_rate_under_global_null']:.0f} %, so this line is suggestive at most.")
    else:
        L.append(f"Uncorrected screening (never sets the regime): none of the {k} learned cells' unadjusted 95 % intervals excludes "
                 f"zero ({scr_txt}).")
    if "fixture_planted_regime_label" in pooled:
        L.append(f"Fixture sanity check: planted label `{pooled['fixture_planted_regime_label']}`; corrected-tier derived regime "
                 f"`{regime}` ({'match' if pooled['fixture_label_matches_derived_regime'] else 'no match'}); uncorrected screening "
                 f"`{pooled['screening_regime_uncorrected']}` ({'match' if pooled['fixture_label_matches_uncorrected_screening'] else 'no match'}). "
                 + ("A planted effect smaller than the ~%.0f pp the corrected intervals resolve at this n is expected to read "
                    "inconclusive at the corrected tiers and to show only on the screening line. " % common.power_pp(96, family=fam["registered"] or 1)
                    if not pooled["fixture_label_matches_derived_regime"] and regime == "inconclusive" else "")
                 + "The label is read only for this comparison and never enters the analysis.")
    L.append("")

    # ---- chain table
    L.append("## Chain table per (task, arm): forecast -> first-decision change -> flips -> net gain\n")
    L.append("Forecast: H6 proprio error reduction vs native, % of native (BF16, offline development pools, simultaneous 95 %). "
             "Dev sel: development first-population diagnostic (changed selections of 96 / mean Spearman; Reach/Reach-Wall only). "
             "1st chg: fresh scenarios whose FIRST planning call chose a different action sequence than native, over scenarios with "
             "a usable trace. Flips: rescues + regressions vs native. Net: success gain vs native, pp, with exploratory Bonferroni "
             "interval; Reg.: the pre-registered simultaneous interval.\n")
    t = pd.DataFrame(dict(
        task=chain.task, arm=chain.arm, cls=chain.cls,
        forecast_proprio=[f"{fmt(v)} {ci(lo, hi, 2)}" if np.isfinite(v) else "absent" for v, lo, hi in
                          zip(chain.forecast_proprio, chain.forecast_proprio_lo, chain.forecast_proprio_hi)],
        forecast_visual=[fmt(v) for v in chain.forecast_visual],
        dev_sel=[f"{int(s)}/96, rho={r:.6f}" if np.isfinite(s) else "-" for s, r in zip(chain.dev_changed_selections, chain.dev_spearman)],
        first_chg=[f"{int(k_)}/{int(n_)} ({100*r:.1f}%)" if np.isfinite(r) else "-" for k_, n_, r in
                   zip(chain.first_decision_changed.fillna(-1), chain.n_traced.fillna(-1), chain.first_decision_change_rate)],
        div_calls=[fmt(v) for v in chain.mean_divergent_calls],
        flips=[f"{int(f)} ({int(r)}+ / {int(g_)}-)" for f, r, g_ in zip(chain.flips, chain.rescues, chain.regressions)],
        net_pp=[f"{v:+.2f} {ci(lo, hi)}" for v, lo, hi in zip(chain.net_gain_pp, chain.net_lo, chain.net_hi)],
        registered=[ci(lo, hi) for lo, hi in zip(chain.reg_lo, chain.reg_hi)],
    ))
    L.append(common.md_table(t))
    L.append("")
    single = chain[chain.single_call_task]
    if len(single):
        L.append(f"Single-plan tasks ({', '.join(sorted(set(common.TASK_LABEL.get(x, x) for x in single.task)))}) execute one "
                 "planning call, so 'first decision changed' there means the entire executed plan differed; a replanning-dynamics "
                 "explanation is unavailable on those tasks by construction.\n")
    miss = chain[chain.n_missing_trace.fillna(0) > 0]
    if len(miss):
        L.append(f"{len(miss)} cell(s) exclude scenarios without a usable action trace from the decision proxy (listed in the log); "
                 "those scenarios still count in the success contrasts.\n")

    # ---- fresh decision proxy detail
    L.append("## Fresh decision-level proxy (exploratory)\n")
    L.append("Per-iteration CEM losses were not recorded, so the decision proxy is the chosen-action hash, not the score margin: "
             f"it says whether the planner chose differently, not how close the runner-up was. ci_pp: exploratory Bonferroni-"
             f"{fam['decision_bootstraps']} interval of the first-decision change rate (pp).\n")
    rows = []
    for c in cells:
        p = c["decision_fresh"]
        if not p:
            continue
        rows.append(dict(task=c["task"], arm=c["arm"], n=p["n"], missing=p["n_missing_trace"], calls=int(p["n_calls_median"]),
                         first_changed_pct=100 * p["first_decision_change_rate"],
                         ci_pp=ci(p["first_decision_change_rate_ci"]["lower"], p["first_decision_change_rate_ci"]["upper"]),
                         any_divergence_pct=100 * p["any_divergence_rate"],
                         mean_first_div_frac=p["mean_first_divergence_fraction_among_divergent"],
                         mean_divergent_calls=p["mean_divergent_calls"],
                         flip_given_changed=p["flip_rate_given_first_decision_changed"],
                         flip_given_unchanged=p["flip_rate_given_first_decision_unchanged"],
                         share_flips_wo_first_chg=p["share_of_flips_without_first_decision_change"],
                         flips_identical_actions=p["flips_with_identical_actions_throughout"]))
    L.append(common.md_table(pd.DataFrame(rows)) if rows else "No cell has a usable decision proxy.")
    L.append("")
    devcmp = chain.dropna(subset=["dev_changed_selections", "first_decision_changed"])
    if len(devcmp):
        more = int((devcmp.first_decision_changed > devcmp.dev_changed_selections).sum())
        fewer = int((devcmp.first_decision_changed < devcmp.dev_changed_selections).sum())
        L.append(f"Fresh proxy vs development diagnostic on the {len(devcmp)} cells that have both: the fresh first-decision change "
                 f"count exceeds the development changed-selection count in {more}, is below it in {fewer}, and equal in "
                 f"{len(devcmp) - more - fewer}. " + (
                     "The fresh proxy is expected to be the more sensitive of the two because the edit acts on every CEM iteration of "
                     "the first call rather than on one shared scored population, so a higher fresh count is not by itself evidence "
                     "of a larger effect." if more > fewer else
                     "The fresh proxy does not detect more first-decision changes than the shared-population diagnostic, so the "
                     "first decision is at least as stable under the full CEM loop as under a single scored population."))
        L.append("")
    bad = [r for r in rows if r["flips_identical_actions"] > 0]
    if bad:
        L.append(f"Integrity note: {len(bad)} cells have outcome flips in scenarios whose action hashes are identical to native at "
                 "every planning call. In a deterministic simulator on the same RNG stream this should be zero; on real data a "
                 "non-zero count indicates simulator non-determinism or a hashing gap and the decision proxy should be read with care."
                 + (" (Expected here: the synthetic fixture draws outcomes and traces independently.)" if prov["synthetic"] else ""))
        L.append("")
    dg = pooled["dose_generic_check"]
    if dg:
        L.append("Learned vs its dose-matched random control on the decision proxy (is the decision perturbation direction-specific?). "
                 f"Paired per scenario on the shared traced scenarios; exploratory Bonferroni-{fam['decision_bootstraps']} interval "
                 "of the difference in first-decision change rate (pp); discordant counts are scenarios where only one of the two "
                 "changed the first decision:\n")
        for d in dg:
            b = d["contrast_pp"]
            ds = d["discordant_scenarios"]
            L.append(f"- {common.TASK_LABEL[d['task']]}: {d['learned']} {d['learned_changed']}/{d['n_shared']} vs {d['control']} "
                     f"{d['control_changed']}/{d['n_shared']} first-decision changes (difference {b['difference']:+.1f} pp, interval "
                     f"{ci(b['lower'], b['upper'])}; discordant {ds['learned_only']} learned-only vs {ds['control_only']} control-only, "
                     f"exact sign-test p = {ds['exact_sign_test_p_unadjusted']:.3f} unadjusted); flips {d['learned_flips']} vs "
                     f"{d['control_flips']} -> {d['verdict']}.")
        n_res = sum(d["code"] != "unresolved" for d in dg)
        L.append(f"\nDirection-specificity of the decision perturbation is resolved in {n_res} of {len(dg)} learned/control pairs"
                 + ("; where unresolved, the fresh traces cannot distinguish a direction-specific from a dose-generic perturbation "
                    "of the first decision." if n_res < len(dg) else "."))
        L.append("")

    # ---- correlations
    L.append("## Cross-cell Spearman correlations (exploratory; cells are not independent)\n")
    L.append("Each point is one (task, arm) cell. Cells share the native arm within a task and the same scenarios, so n is the "
             "number of cells, not of independent observations; p-values are Holm-adjusted over the six correlations and are "
             "descriptive only.\n")
    ct = pd.DataFrame([dict(correlation=c["name"], n=c["n"], rho=c["rho"], p=c["p"], p_holm=c["p_holm"]) for c in corrs])
    L.append(common.md_table(ct, floatfmt="{:.3f}"))
    L.append("")
    conc = pooled["concordance"]
    cb = {c["name"]: c for c in corrs}
    fs, vs = cb["proprio_forecast_vs_success_gain"], cb["visual_forecast_vs_success_gain"]
    if conc["n"]:
        L.append(f"Sign concordance between forecast improvement and success gain: {conc['agree']}/{conc['n']} cells "
                 f"({100*conc['fraction']:.0f}%; unadjusted binomial p = {conc['binomial_p_unadjusted']:.2f}).")
    if np.isfinite(fs["rho"]) and np.isfinite(vs["rho"]):
        if abs(abs(vs["rho"]) - abs(fs["rho"])) < 1e-9:
            which = "the two endpoints rank cells identically against their success gains, "
        else:
            which = f"the {'visual' if abs(vs['rho']) > abs(fs['rho']) else 'proprio'} endpoint ranks cells more like their success gains, "
        L.append(f"As predictors of success gain, proprio forecast rho = {fs['rho']:+.2f} and visual forecast rho = {vs['rho']:+.2f} "
                 f"(n = {fs['n']}); " + which
                 + ("but neither survives correction." if min(fs['p_holm'], vs['p_holm']) >= 0.05
                    else "and at least one survives Holm correction at 0.05."))
    elif not np.isfinite(fs["rho"]):
        L.append(f"Fewer than three cells carry an offline forecast effect (n = {fs['n']}), so forecast-vs-success correlations "
                 "cannot be computed on this results tree.")
    L.append("")

    # ---- mediation verdict
    L.append("## Mediation-style verdict (rule-generated)\n")
    th = pooled["thresholds"]
    L.append(f"Rules: first-decision change is 'rare' below {100*th['decision_rare_fraction']:.0f}% of scenarios; flips are "
             f"'substantial' at >= {100*th['flips_substantial_fraction']:.0f}% of scenarios; an outcome effect is 'detected' when "
             f"the pre-registered interval excludes zero, or 'detected at the exploratory level' when this script's Bonferroni-"
             f"{fam['success_bootstraps']} interval does. The unadjusted interval is screening only and never sets a verdict. "
             "'Not distinguishable from equally good' means no net difference was detected at this sample size, not that the "
             "alternatives are known to be equal.\n")
    for c in cells:
        if c["arm_class"] != "learned":
            continue
        L.append(f"- **{common.TASK_LABEL[c['task']]} / {common.ARM_LABEL[c['arm']]}** (`{c['verdict']['code']}`): {c['verdict']['text']}.")
    L.append("")
    others = [c for c in cells if c["arm_class"] != "learned"]
    if others:
        L.append("Controls and pathway components (same rules): " + "; ".join(
            f"{c['label']} `{c['verdict']['code']}`" for c in others) + ".\n")
    codes = pooled["learned_cell_verdict_codes"]
    n_learned = len(codes)
    single = {c["label"] for c in cells if c["arm_class"] == "learned" and c["decision_fresh"] and c["decision_fresh"]["single_call_task"]}
    multi_codes = {a: v for a, v in codes.items() if a not in single}
    single_codes = {a: v for a, v in codes.items() if a in single}
    n_replan = sum(v.startswith("replanning") for v in multi_codes.values())
    n_inert = sum(v == "inert" for v in codes.values())
    n_rerank = sum(v == "rerank_equal" for v in codes.values())
    n_dec_multi = sum(v.startswith("decision_mediated") for v in multi_codes.values())
    n_dec_single = sum(v.startswith("decision_mediated") for v in single_codes.values())
    detected_any = pooled["any_learned_cell_detected_at_corrected_level"]
    pooled_txt = (f"**Pooled reading.** Over {n_learned} learned cells ({len(multi_codes)} on replanning tasks, {len(single_codes)} on "
                  f"single-plan tasks): {n_replan}/{len(multi_codes)} replanning-task cells flip outcomes mostly without a first-decision "
                  f"change, {n_dec_multi}/{len(multi_codes)} are consistent with decision mediation, {n_rerank}/{n_learned} cells "
                  f"overall re-rank without a detected net outcome movement, {n_inert}/{n_learned} are inert, and "
                  f"{n_dec_single}/{len(single_codes)} single-plan cells move outcomes at a corrected level (where every outcome change "
                  "passes through the only decision by construction). ")
    if len(multi_codes) and n_dec_multi == 0 and n_replan > 0:
        if detected_any:
            pooled_txt += ("On the replanning tasks, wherever the edit reaches the outcome it does so after the first choice, through "
                           "the closed loop; a 'corrected forecast re-ranks the planner's first choice' story is not what the fresh "
                           "traces show.")
        else:
            pooled_txt += (f"In {n_replan} of the {len(multi_codes)} replanning-task cells the outcome flips that do occur fall mostly "
                           "in scenarios whose first decision was unchanged, so any effect - none is detected at a corrected level "
                           "here - would accrue after the first choice, through the closed loop; a 'corrected forecast re-ranks the "
                           "planner's first choice' story is not what the fresh traces show.")
    elif n_dec_multi > 0:
        pooled_txt += (f"{n_dec_multi} replanning-task cell(s) change the first decision AND move outcomes at a corrected level, the "
                       "pattern a decision-mediated effect of either sign would produce; whether that is direction-specific is "
                       "answered by the learned-vs-control check above.")
    elif n_inert == n_learned:
        pooled_txt += "The edit is behaviorally inert at this dose on every learned cell: neither the first decision nor the outcome moves."
    elif n_rerank == n_learned:
        pooled_txt += ("Every learned cell changes some decisions without a detected net outcome movement; at this sample size the "
                       "alternatives are not distinguishable from equally good.")
    else:
        pooled_txt += "The learned cells do not share one mediation pattern."
    L.append(pooled_txt)
    if pooled["forecast_outcome_program_support"]:
        L.append(f"Across cells, forecast gain and success gain are concordant (rho = {fs['rho']:+.2f}, Holm p = {fs['p_holm']:.3f}, "
                 f"sign agreement {100*conc['fraction']:.0f}%): weak, exploratory support for the forecast->outcome program.")
    else:
        L.append("Across cells, forecast gain and success gain are not concordant enough to support the forecast->outcome program "
                 + (f"(rho = {fs['rho']:+.2f}, Holm p = {fs['p_holm']:.2f}, sign agreement "
                    f"{100*conc['fraction']:.0f}%)." if np.isfinite(fs["rho"]) else "(too few cells with forecast data)."))
    L.append("")

    # ---- does / does not
    L.append("## What this does and does not establish\n")
    L.append("**Does.**")
    L.append(f"- Places every fresh (task, arm) cell on the three levels the study measures and reports, with intervals, whether "
             f"the fresh panel's regime is positive, mixed, negative or inconclusive at a corrected level (here: {regime}).")
    L.append("- Gives a fresh, per-scenario decision proxy that the development diagnostic lacked: whether the planner's first "
             "chosen sequence under the edit differs from native on the same scenario and RNG stream, and how outcome flips "
             "distribute over scenarios with and without a first-decision change.")
    L.append("- Ties the three levels together with rule-based readings whose thresholds are stated above and applied identically "
             "to learned arms, controls and pathway components.")
    L.append("\n**Does not.**")
    L.append("- Establish a causal mediation path. The forecast leg is measured on development pools (not the fresh scenarios), "
             "the decision proxy is a hash equality (no score margin: per-iteration CEM losses were not recorded), and cross-cell "
             "correlations pool non-independent cells. None of these is a pre-registered test.")
    L.append("- Turn an inconclusive success interval into a null: with 96 paired scenarios the panel resolves roughly "
             f"{common.power_pp(96, family=fam['registered'] or 1):.0f} pp at the registered family size, not the "
             f"{common.USEFUL_GAIN_PP:g} pp pre-registered as useful. 'Not distinguishable from equally good' is a statement about "
             "resolution, not equality.")
    L.append(f"- Promote the uncorrected screening line to a finding: under a global null at least one of {k} unadjusted intervals "
             f"excludes zero with probability {100 * pooled['screening_false_sign_rate_under_global_null']:.0f} %.")
    L.append("- Identify which physical situations are rescued or broken; that would require post hoc subgroup selection, which is not done here.")
    L.append("- Assign physical semantics to the edited directions. In the terms of the 'physics emergence zone' framing, a decodable "
             "variable, a correctable forecast and a decision-relevant mechanism are three different things; this page tests only "
             "whether the second reaches the third.")
    L.append("")

    # ---- degradations and provenance
    L.append("## Degradations and log\n")
    for line in log.lines:
        L.append(f"- {line}")
    L.append("")
    L.append("## Provenance\n")
    L.append(f"- results: `{prov['results']}`\n- freeze sha256: `{prov['freeze_sha256']}`\n- analysis sha256: `{prov['analysis_sha256']}`"
             f"\n- synthetic: {prov['synthetic']}\n- generated: {prov['generated_utc']}\n- bootstrap: {common.BOOTSTRAP_DRAWS} draws, seed {common.BOOTSTRAP_SEED}")
    L.append(f"\nFigures: `{NAME}_scatter.png` (2x2 scatter matrix with intervals; learned cells named, controls/components numbered "
             f"with a key), `{NAME}_chain.png` (per-cell chain bars).\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()
