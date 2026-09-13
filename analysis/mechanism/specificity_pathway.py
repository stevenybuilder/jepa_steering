"""Specificity and pathway analysis of the fresh four-task eight-arm confirmation.

Question. Is any closed-loop success effect of the learned edits due to the fitted DIRECTIONS
(model biology) or to generic perturbation at the (B3, H3) site, and how do the visual and
action-conditioning pathways combine?

Computes, per task, from the finished results tree only (no model, simulator or GPU):
  1. generic-perturbation share  = (control - native) / (learned - native) for each learned arm,
     with a paired scenario-cluster bootstrap interval on the ratio ONLY when the registered
     learned-vs-native lower bound is > 0; otherwise the ratio is reported as undefined and the
     absolute differences (learned-native, control-native, learned-control) are given instead;
  2. pathway factorial on SUCCESS: interaction = joint - visual_only - action_condition_only + native,
     visual-only vs action-only, equal-budget coupling vs unscaled joint, additivity verdict;
  3. cross-task consistency: sign concordance of each effect across tasks and Spearman correlation
     of the 7-arm effect profile between every pair of tasks (with a permutation reference);
  4. dose verification from common.energy_frame (realized vs requested energy; learned vs control);
  5. a per-task success-by-arm bar figure (Wilson 95% whiskers, descriptive) for the paper.

Only arm-vs-native contrasts are pre-registered (read from ANALYSIS/report.json). Everything else
here is EXPLORATORY; every exploratory interval is Bonferroni-corrected over the fixed family of
EXPLORATORY_FAMILY intervals this script can report (7 per task x 4 tasks), and the outputs say so.
All interpretation text is generated from the numbers and branches on the observed regime.

Usage (from the repository root):
  .venv/bin/python analysis/mechanism/specificity_pathway.py --results R --freeze F --analysis A --out OUT
"""
from __future__ import annotations

import itertools
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

SCRIPT = "specificity_pathway"
LOG = logging.getLogger(SCRIPT)

# Fixed exploratory family: per task -> [fixed_rank4 - control, coupling_only - control,
# share(fixed_rank4), share(coupling_only), interaction, visual - action, coupling_only - joint].
EXPLORATORY_PER_TASK = 7
EXPLORATORY_FAMILY = EXPLORATORY_PER_TASK * len(C.TASKS)
CRITERION = ("two-reference criterion (registered lower bound > 0 vs unsteered AND exploratory Bonferroni-%d lower "
             "bound > 0 vs its random control, gain >= 5 pp; the vs-control half is not pre-registered for this "
             "confirmation)" % EXPLORATORY_FAMILY)
REGISTERED_FAMILY_FALLBACK = 48
PERMUTATIONS = 4000
PERMUTATION_SEED = 20260913
ENERGY_TOLERANCE = 0.05  # relative difference in delivered energy tolerated between learned and control
ROUNDING_TOLERANCE = 0.02  # tolerated mean rounding loss (requested -> realized)
DOSE_CV_VARIES = 0.05  # across-scenario coefficient of variation above which an arm's dose is called state-dependent

# The per-cell reading combines the REGISTERED vs-native interval with the EXPLORATORY vs-control interval. That
# joint two-reference rule is the development-panel criterion (paper/workshop/main.tex); it is NOT pre-registered
# for this confirmation, whose registered contrasts are arm-vs-native only. Every mention of it says so.
READING_CRITERION = "two_reference_development_style_not_preregistered"
CRITERION_SHORT = "two-reference criterion"

SHORT = {"native": "Unsteered", "fixed_rank4": "Rank-4 edit", "matched_random_fixed_rank4": "Rand. subspace",
         "coupling_only": "Eq. coupling", "matched_random_coupling": "Rand. direction", "joint": "Joint",
         "visual_only": "Visual-only", "action_condition_only": "Action-only"}
COL_LEARNED, COL_NATIVE, COL_CONTROL, COL_PATHWAY = "#1A8F7A", "#8C8C8C", "#C8C8C8", "#7F9CB8"
ARM_KIND = {"native": "native", "fixed_rank4": "learned", "coupling_only": "learned",
            "matched_random_fixed_rank4": "control", "matched_random_coupling": "control",
            "joint": "pathway", "visual_only": "pathway", "action_condition_only": "pathway"}


# ------------------------------------------------------------------ helpers (script-local)
def wilson(k, n, z=1.959964):
    """Wilson score interval for a binomial proportion, in percent."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return 100 * (centre - half), 100 * (centre + half)


def boot_ratio(num, den, family, draws=C.BOOTSTRAP_DRAWS, seed=C.BOOTSTRAP_SEED, alpha=0.05):
    """Paired scenario-cluster percentile bootstrap of mean(num)/mean(den).

    Uses the same resampling scheme (seed, index draw) as common.paired_bootstrap so the ratio is
    computed on exactly the resampled scenario sets that produced the difference intervals.
    """
    num = np.asarray(num, float)
    den = np.asarray(den, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(den), size=(draws, len(den)))
    nm, dm = num[idx].mean(1), den[idx].mean(1)
    ok = dm > 0
    ratios = nm[ok] / dm[ok]
    q = alpha / family
    lo, hi = np.quantile(ratios, [q / 2, 1 - q / 2]) if ok.sum() else (np.nan, np.nan)
    return dict(ratio=float(num.mean() / den.mean()), lower=float(lo), upper=float(hi),
                draws_with_nonpositive_denominator=int((~ok).sum()), draws=int(draws), seed=int(seed),
                family=int(family))


def contrast(a, b, family):
    """common.paired_bootstrap on binary success, plus discordance counts."""
    r = C.paired_bootstrap(a, b, family=family)
    r.update(C.discordance(a, b))
    r["excludes_zero"] = bool(r["lower"] > 0 or r["upper"] < 0)
    return r


def registered(report, task, arm):
    """Pre-registered arm-vs-native contrast from the frozen analysis report."""
    c = report["results"][task]["contrasts"][f"{arm}-native"]
    lo, hi = c["simultaneous_95_interval_pp"]
    return dict(difference=float(c["difference_pp"]), lower=float(lo), upper=float(hi),
                family=int(report.get("analysis", {}).get("family_size", REGISTERED_FAMILY_FALLBACK)),
                registered=True)


def fmt(x, nd=1):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:+.{nd}f}"


def ci(d, nd=1):
    return f"{fmt(d['difference'], nd)} [{fmt(d['lower'], nd)}, {fmt(d['upper'], nd)}]"


def sign_word(x, tol=0.0):
    return "positive" if x > tol else ("negative" if x < -tol else "zero")


def fmt_p(m):
    """Permutation p as (b+1)/(N+1); printed as a bound when no permutation reached the observed value."""
    return "p < 0.001" if m["exceedances"] == 0 else f"p = {m['permutation_p']:.3f}"


def cells_str(pairs):
    return ", ".join(f"{C.TASK_LABEL[t]}/{SHORT[a]}" for t, a in pairs) or "none"


def lc_resolved(spec, pairs):
    """Cells whose exploratory learned-minus-control interval excludes zero."""
    return [(t, a) for t, a in pairs if spec[t][a]["learned_vs_control_exploratory"]["excludes_zero"]]


def cells_with_reading(spec, tasks, reading):
    return [(t, a) for t in tasks for a in C.LEARNED if spec[t][a]["reading"] == reading]


def share_resolution_sentence(spec, pairs):
    """How many defined share intervals resolve away from one half, generated from the numbers."""
    defined = [(t, a) for t, a in pairs if spec[t][a]["generic_perturbation_share"]["defined"]]
    if not defined:
        return "No generic-perturbation share interval is defined for these cells."
    iv = lambda t, a: spec[t][a]["generic_perturbation_share"]["interval"]  # noqa: E731
    below = [(t, a) for t, a in defined if iv(t, a)["upper"] < 0.5]
    above = [(t, a) for t, a in defined if iv(t, a)["lower"] > 0.5]
    unres = [p for p in defined if p not in below and p not in above]
    bits = []
    if below:
        bits.append(f"resolved below one half (predominantly direction-specific) on {cells_str(below)}")
    if above:
        bits.append(f"resolved above one half (predominantly generic perturbation) on {cells_str(above)}")
    if unres:
        bits.append(f"straddling one half (split unresolved) on {cells_str(unres)}")
    return (f"Generic-perturbation share intervals (ratio bootstrap, Bonferroni-{EXPLORATORY_FAMILY}) are defined for "
            f"{len(defined)} cell(s): " + "; ".join(bits) + ".")


# ------------------------------------------------------------------ 1. specificity
def specificity_for_task(S, report, task):
    out = {}
    nat = S["native"].to_numpy()
    for learned, control in C.LEARNED.items():
        L, K = S[learned].to_numpy(), S[control].to_numpy()
        reg_l, reg_c = registered(report, task, learned), registered(report, task, control)
        lc = contrast(L, K, EXPLORATORY_FAMILY)
        own_l = C.paired_bootstrap(L, nat, family=reg_l["family"])  # reproduction of the registered estimator
        cell = dict(learned=learned, control=control, n=int(len(nat)),
                    learned_vs_native_registered=reg_l, control_vs_native_registered=reg_c,
                    learned_vs_native_reproduced=own_l, learned_vs_control_exploratory=lc,
                    registered_point_reproduced=bool(abs(own_l["difference"] - reg_l["difference"]) < 0.05),
                    registered_lower_abs_diff_pp=float(abs(own_l["lower"] - reg_l["lower"])),
                    reading=C.classify_regime(reg_l, lc), reading_criterion=READING_CRITERION,
                    reading_note=(f"'confirmed' = {CRITERION}; the other labels use the same two intervals"))
        ratio_defined = reg_l["lower"] > 0
        share_point = (reg_c["difference"] / reg_l["difference"]) if reg_l["difference"] != 0 else np.nan
        cell["generic_perturbation_share"] = dict(
            defined=bool(ratio_defined), point_estimate=float(share_point) if np.isfinite(share_point) else None,
            point_estimate_note=("(control-native)/(learned-native) at the point estimate; only interpretable "
                                 "when the learned-vs-native registered lower bound is > 0"),
            interval=(boot_ratio(K.astype(float) - nat.astype(float), L.astype(float) - nat.astype(float),
                                 EXPLORATORY_FAMILY) if ratio_defined else None),
            undefined_reason=(None if ratio_defined else
                              f"registered learned-vs-native lower bound {reg_l['lower']:+.2f} pp is not > 0; "
                              "the denominator is not resolved from zero, so the ratio has no meaningful interval "
                              "and absolute differences are reported instead"))
        # point-estimate pattern (explicitly descriptive)
        cell["point_pattern"] = ("favorable" if reg_l["difference"] >= C.USEFUL_GAIN_PP and lc["difference"] >= C.USEFUL_GAIN_PP
                                 else "unfavorable" if reg_l["difference"] <= -C.USEFUL_GAIN_PP else "flat")
        out[learned] = cell
    return out


def share_reading(cell):
    """Sentence(s) for one (task, learned arm) cell, generated from the numbers."""
    reg_l, reg_c, lc = (cell["learned_vs_native_registered"], cell["control_vs_native_registered"],
                        cell["learned_vs_control_exploratory"])
    L, K = C.ARM_LABEL[cell["learned"]], C.ARM_LABEL[cell["control"]]
    s = cell["generic_perturbation_share"]
    parts = [f"{L}: {fmt(reg_l['difference'])} pp vs unsteered (registered {ci(reg_l)}), "
             f"its {K} {fmt(reg_c['difference'])} pp (registered {ci(reg_c)}); learned minus control "
             f"{ci(lc)} pp (exploratory, family {lc['family']}; {lc['wins']} wins / {lc['losses']} losses)."]
    if s["defined"]:
        iv = s["interval"]
        parts.append(f"The generic-perturbation share is {iv['ratio']:.2f} [{iv['lower']:.2f}, {iv['upper']:.2f}] "
                     f"(ratio bootstrap, family {iv['family']}).")
        if iv["upper"] < 0.5:
            parts.append("Less than half of the native-relative gain is reproduced by the control across the interval: "
                         "on this task the gain is predominantly direction-specific.")
        elif iv["lower"] > 0.5:
            parts.append("More than half of the native-relative gain is reproduced by the control across the interval: "
                         "generic perturbation at the site accounts for most of it.")
        else:
            parts.append("The interval straddles one half, so the split between direction-specific and generic "
                         "perturbation is not resolved.")
    else:
        parts.append(f"The share is undefined ({s['undefined_reason']}).")
        if abs(reg_l["difference"]) >= C.USEFUL_GAIN_PP:
            pt = s["point_estimate"]
            parts.append(f"At the point estimates only, the control reproduces {pt:.2f} of the learned arm's "
                         f"native-relative change; this is a description of two unresolved numbers, not an estimate "
                         f"with an interval.")
    r = cell["reading"]
    if r == "confirmed":
        parts.append(f"Reading: meets the {CRITERION_SHORT} (registered lower bound > 0 vs unsteered, exploratory lower "
                     "bound > 0 vs control, gain >= 5 pp; the vs-control half is not pre-registered).")
    elif r == "positive_vs_native_only":
        parts.append("Reading: better than unsteered but not resolved from its own control, so direction-specificity "
                     "is not established.")
    elif r == "positive_vs_control_only":
        parts.append("Reading: resolved from its control but not from unsteered; consistent with the control being "
                     "harmful rather than the learned edit being helpful.")
    elif r == "harmful":
        parts.append("Reading: harmful relative to unsteered (registered upper bound < 0).")
    else:
        parts.append("Reading: inconclusive; neither interval excludes zero.")
    return " ".join(parts)


# ------------------------------------------------------------------ 2. pathway factorial
def pathway_for_task(S, report, task):
    nat, J = S["native"].to_numpy(), S["joint"].to_numpy()
    V, A, E = S["visual_only"].to_numpy(), S["action_condition_only"].to_numpy(), S["coupling_only"].to_numpy()
    inter = C.paired_bootstrap(J.astype(float) + nat.astype(float), V.astype(float) + A.astype(float),
                               family=EXPLORATORY_FAMILY)
    if inter["scale"] == 1.0:  # inputs were not all in {0,1}; convert to percentage points
        for k in ("difference", "lower", "upper"):
            inter[k] *= 100.0
        inter["scale"] = 100.0
    per_scen = J.astype(int) - V.astype(int) - A.astype(int) + nat.astype(int)
    inter["per_scenario_counts"] = {str(v): int((per_scen == v).sum()) for v in sorted(set(per_scen.tolist()))}
    inter["excludes_zero"] = bool(inter["lower"] > 0 or inter["upper"] < 0)
    va = contrast(V, A, EXPLORATORY_FAMILY)
    ej = contrast(E, J, EXPLORATORY_FAMILY)
    regs = {a: registered(report, task, a) for a in C.COMPONENT + ("coupling_only",)}
    additive_pred = regs["visual_only"]["difference"] + regs["action_condition_only"]["difference"]
    if inter["excludes_zero"]:
        verdict = "super-additive" if inter["difference"] > 0 else "sub-additive"
    else:
        verdict = "additivity not rejected"
    return dict(registered_vs_native=regs, interaction_exploratory=inter, visual_vs_action_exploratory=va,
                equal_budget_vs_unscaled_joint_exploratory=ej, additive_prediction_pp=float(additive_pred),
                observed_joint_pp=float(regs["joint"]["difference"]), additivity_verdict=verdict,
                interaction_resolution_halfwidth_pp=float((inter["upper"] - inter["lower"]) / 2))


def pathway_reading(task, p):
    regs, inter, va, ej = (p["registered_vs_native"], p["interaction_exploratory"],
                           p["visual_vs_action_exploratory"], p["equal_budget_vs_unscaled_joint_exploratory"])
    T = C.TASK_LABEL[task]
    parts = [f"{T}: visual-only {fmt(regs['visual_only']['difference'])} pp, action-only "
             f"{fmt(regs['action_condition_only']['difference'])} pp, unscaled joint "
             f"{fmt(regs['joint']['difference'])} pp vs unsteered (registered intervals "
             f"{ci(regs['visual_only'])}, {ci(regs['action_condition_only'])}, {ci(regs['joint'])})."]
    parts.append(f"Additive prediction for joint = {fmt(p['additive_prediction_pp'])} pp; observed "
                 f"{fmt(p['observed_joint_pp'])} pp; interaction {ci(inter)} pp (exploratory, family "
                 f"{inter['family']}), so {p['additivity_verdict']} at a resolution of about "
                 f"+/-{p['interaction_resolution_halfwidth_pp']:.0f} pp.")
    if inter["excludes_zero"]:
        parts.append("An interaction this size means the two pathways do not act independently on success on this "
                     "task; because every registered main effect is read separately, this is a post-hoc "
                     "observation to be confirmed, not a mechanism.")
    else:
        parts.append("Interactions smaller than that resolution cannot be excluded; the factorial does not "
                     "resolve whether the pathways combine additively.")
    if va["excludes_zero"]:
        parts.append(f"Visual-only differs from action-only by {ci(va)} pp ({va['wins']} / {va['losses']} "
                     f"discordant), favouring the {'visual' if va['difference'] > 0 else 'action-conditioning'} pathway.")
    else:
        parts.append(f"Visual-only minus action-only is {ci(va)} pp: the pathways are not separated on success.")
    if ej["excludes_zero"]:
        parts.append(f"Equal-budget coupling differs from the unscaled joint edit by {ci(ej)} pp, so on this task the "
                     f"dose scaling itself changes the outcome.")
    else:
        parts.append(f"Equal-budget coupling minus unscaled joint is {ci(ej)} pp: halving the squared standardized "
                     f"dose is not distinguishable from the full-dose edit on success.")
    return " ".join(parts)


# ------------------------------------------------------------------ 3. cross-task consistency
def profiles_from(report, tasks):
    """7-arm effect profiles (arm - native, pp) per task, from the registered point estimates."""
    arms = [a for a in C.ARMS if a != "native"]
    return {t: np.array([report["results"][t]["contrasts"][f"{a}-native"]["difference_pp"] for a in arms])
            for t in tasks}, arms


def cross_task(report, tasks, spec):
    prof, arms = profiles_from(report, tasks)
    out = dict(arms=arms, profiles_pp={t: prof[t].tolist() for t in tasks}, tasks=list(tasks))
    # sign concordance per arm (vs native) and per learned arm (vs control)
    conc = {}
    for i, a in enumerate(arms):
        vals = [prof[t][i] for t in tasks]
        conc[f"{a}-native"] = _concordance(vals, tasks)
    for learned in C.LEARNED:
        vals = [spec[t][learned]["learned_vs_control_exploratory"]["difference"] for t in tasks]
        conc[f"{learned}-{C.LEARNED[learned]}"] = _concordance(vals, tasks)
    out["sign_concordance"] = conc
    # Spearman of the 7-arm profile between every pair of tasks
    pairs = {}
    for t1, t2 in itertools.combinations(tasks, 2):
        pairs[f"{t1}|{t2}"] = C.spearman(prof[t1], prof[t2])
    out["pairwise_spearman"] = pairs
    if len(tasks) >= 2:
        obs = float(np.mean([v["rho"] for v in pairs.values()]))
        rng = np.random.default_rng(PERMUTATION_SEED)
        null = np.empty(PERMUTATIONS)
        for k in range(PERMUTATIONS):
            perm = {t: prof[t][rng.permutation(len(arms))] for t in tasks}
            null[k] = np.mean([C.spearman(perm[a], perm[b])["rho"] for a, b in itertools.combinations(tasks, 2)])
        b = int((null >= obs).sum())
        out["mean_pairwise_rho"] = dict(observed=obs, permutation_p=float((b + 1) / (PERMUTATIONS + 1)), exceedances=b,
                                        permutations=PERMUTATIONS, seed=PERMUTATION_SEED,
                                        null_mean=float(null.mean()), null_sd=float(null.std()),
                                        p_definition="(exceedances + 1) / (permutations + 1), one-sided; family 1, uncorrected",
                                        note="arm labels permuted independently within each task; descriptive only")
    else:
        out["mean_pairwise_rho"] = None
    return out


def _concordance(vals, tasks):
    signs = [sign_word(v) for v in vals]
    npos, nneg, nzero = signs.count("positive"), signs.count("negative"), signs.count("zero")
    nonzero = npos + nneg
    return dict(values_pp={t: float(v) for t, v in zip(tasks, vals)}, positive=npos, negative=nneg, zero=nzero,
                concordant=bool(nonzero >= 2 and (npos == nonzero or nneg == nonzero)),
                majority_sign=("positive" if npos > nneg else "negative" if nneg > npos else "tie"))


def cross_task_reading(x):
    tasks = x["tasks"]
    parts = []
    if len(tasks) < 2:
        return "Fewer than two tasks are available, so cross-task consistency cannot be assessed."
    for learned in C.LEARNED:
        c = x["sign_concordance"][f"{learned}-native"]
        cc = x["sign_concordance"][f"{learned}-{C.LEARNED[learned]}"]
        L = C.ARM_LABEL[learned]
        parts.append(f"{L} vs unsteered is {c['positive']} positive / {c['negative']} negative / {c['zero']} zero "
                     f"across {len(tasks)} tasks ({'sign-concordant' if c['concordant'] else 'not sign-concordant'}); "
                     f"vs its control {cc['positive']} / {cc['negative']} / {cc['zero']} "
                     f"({'concordant' if cc['concordant'] else 'not concordant'}).")
    m = x["mean_pairwise_rho"]
    rhos = [v["rho"] for v in x["pairwise_spearman"].values()]
    parts.append(f"Spearman correlation of the 7-arm effect profile between task pairs ranges "
                 f"{min(rhos):+.2f} to {max(rhos):+.2f} (mean {m['observed']:+.2f}; permutation reference mean "
                 f"{m['null_mean']:+.2f}, one-sided {fmt_p(m)} over {m['permutations']} arm-label permutations, "
                 f"{m['exceedances']} exceedances; a single uncorrected descriptive test).")
    if m["permutation_p"] < 0.05 and m["observed"] > 0:
        parts.append("The same ordering of arms recurs across tasks more than arm-label permutation would produce; "
                     "a shared ordering is consistent with a site-generic effect of the edit family or with a "
                     "shared direction-specific effect, and the specificity table above is what separates those.")
    elif m["observed"] < 0 and m["permutation_p"] > 0.95:
        parts.append("Arm orderings are anti-correlated across tasks more than permutation would produce; the "
                     "effect profile is task-specific in a way that argues against a single transferable mechanism.")
    else:
        parts.append("The ordering of arms does not recur across tasks beyond what arm-label permutation produces; "
                     "there is no evidence of a shared effect profile.")
    return " ".join(parts)


# ------------------------------------------------------------------ 4. dose verification
def dose_verification(results, tasks, tolerance=ENERGY_TOLERANCE, rounding_tol=ROUNDING_TOLERANCE):
    out = dict(tolerance_relative=tolerance, rounding_tolerance=rounding_tol, cv_varies_threshold=DOSE_CV_VARIES,
               refined={}, coupling={}, flags=[])
    try:
        ef_r = C.energy_frame(results, tasks=tasks, arms=("fixed_rank4", "matched_random_fixed_rank4"))
    except Exception as e:  # noqa: BLE001
        LOG.warning("refined energy frame unavailable: %s", e)
        ef_r = pd.DataFrame()
    try:
        ef_c = C.energy_frame(results, tasks=tasks, arms=("coupling_only", "matched_random_coupling", "joint",
                                                          "visual_only", "action_condition_only"))
    except Exception as e:  # noqa: BLE001
        LOG.warning("coupling energy frame unavailable: %s", e)
        ef_c = pd.DataFrame()
    need_r = {"requested_l2_mean", "realized_l2_mean", "rounding_loss", "active_fraction"}
    need_c = {"edited_candidates", "requested_sq_l2_sum", "realized_sq_l2_sum"}

    def cv(x):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        return float(x.std() / x.mean()) if x.size and x.mean() > 0 else float("nan")

    for task in tasks:
        # refined pair: realized vs requested L2 per candidate
        r = {}
        sub = ef_r[ef_r.task == task] if len(ef_r) else pd.DataFrame()
        if len(sub) and {"fixed_rank4", "matched_random_fixed_rank4"} <= set(sub.arm.unique()) and need_r <= set(sub.columns):
            missing = {arm: int((g.realized_l2_mean.isna() | g.requested_l2_mean.isna()).sum())
                       for arm, g in sub.groupby("arm")}
            per_ep = sub.groupby(["arm", "episode"]).agg(requested=("requested_l2_mean", "mean"),
                                                         realized=("realized_l2_mean", "mean"),
                                                         rounding_loss=("rounding_loss", "mean"),
                                                         active=("active_fraction", "mean"),
                                                         calls=("call_index", "count")).reset_index()
            arms = {}
            for arm, g in per_ep.groupby("arm"):
                arms[arm] = dict(episodes=int(len(g)), population_calls=int(g.calls.sum()),
                                 n_missing_energy=int(missing.get(arm, 0)),
                                 requested_l2_mean=float(g.requested.mean()), realized_l2_mean=float(g.realized.mean()),
                                 realized_cv_across_scenarios=cv(g.realized),
                                 rounding_loss_mean=float(g.rounding_loss.mean()),
                                 rounding_loss_max=float(g.rounding_loss.max()), active_fraction=float(g.active.mean()))
            pv = per_ep.pivot(index="episode", columns="arm", values="realized").dropna()
            ratio = arms["fixed_rank4"]["realized_l2_mean"] / arms["matched_random_fixed_rank4"]["realized_l2_mean"]
            per_ratio = (pv["fixed_rank4"] / pv["matched_random_fixed_rank4"]).to_numpy()
            equal = abs(np.log(ratio)) <= np.log1p(tolerance)
            complete = all(a["n_missing_energy"] == 0 for a in arms.values())
            r = dict(status="ok", arms=arms, learned_over_control_realized=float(ratio),
                     per_scenario_ratio=dict(median=float(np.median(per_ratio)), min=float(per_ratio.min()),
                                             max=float(per_ratio.max()),
                                             fraction_outside_tolerance=float((np.abs(np.log(per_ratio)) > np.log1p(tolerance)).mean())),
                     equal_energy_within_tolerance=bool(equal), energy_records_complete=bool(complete),
                     rounding_within_tolerance=bool(max(a["rounding_loss_mean"] for a in arms.values()) <= rounding_tol),
                     active_fraction_equal=bool(abs(arms["fixed_rank4"]["active_fraction"]
                                                    - arms["matched_random_fixed_rank4"]["active_fraction"]) < 1e-6))
            if not equal:
                out["flags"].append(f"{task}: refined learned/control realized L2 ratio {ratio:.3f} outside +/-{tolerance:.0%}")
            if not r["rounding_within_tolerance"]:
                out["flags"].append(f"{task}: refined rounding loss exceeds {rounding_tol:.0%}")
            if not complete:
                out["flags"].append(f"{task}: refined population calls with missing energy records: "
                                    + ", ".join(f"{a}={m}" for a, m in missing.items() if m))
                LOG.warning("%s: refined energy records incomplete (%s); cell marked dose-unverified", task, missing)
        else:
            r = dict(status="unavailable", reason="no H6 population energy records (with coefficient statistics) for both refined arms")
            LOG.warning("%s: refined dose verification skipped (%s)", task, r["reason"])
        out["refined"][task] = r
        # coupling family: edited candidates and squared-L2 sums
        c = {}
        sub = ef_c[ef_c.task == task] if len(ef_c) else pd.DataFrame()
        if len(sub) and {"coupling_only", "matched_random_coupling"} <= set(sub.arm.unique()) and need_c <= set(sub.columns):
            g = sub.assign(edited_fraction=sub.edited_candidates / sub.candidates,
                           realized_over_requested=sub.realized_sq_l2_sum / sub.requested_sq_l2_sum)
            arms = {}
            for arm, gg in g.groupby("arm"):
                per_ep_real = gg.groupby("episode").realized_sq_l2_sum.mean()
                arms[arm] = dict(population_calls=int(len(gg)), episodes=int(gg.episode.nunique()),
                                 n_missing_energy=int((gg.realized_sq_l2_sum.isna() | gg.requested_sq_l2_sum.isna()).sum()),
                                 edited_fraction=float(gg.edited_fraction.mean()),
                                 requested_sq_l2_sum_mean=float(gg.requested_sq_l2_sum.mean()),
                                 realized_sq_l2_sum_mean=float(gg.realized_sq_l2_sum.mean()),
                                 realized_cv_across_scenarios=cv(per_ep_real),
                                 realized_over_requested=float(gg.realized_over_requested.mean()),
                                 realized_per_edited_candidate=float((gg.realized_sq_l2_sum / gg.edited_candidates.clip(lower=1)).mean()))
            ratio = arms["coupling_only"]["realized_sq_l2_sum_mean"] / arms["matched_random_coupling"]["realized_sq_l2_sum_mean"]
            equal = abs(np.log(ratio)) <= np.log1p(tolerance)
            rounding_ok = all(1 - a["realized_over_requested"] <= rounding_tol for a in arms.values())
            pair_missing = {a: arms[a]["n_missing_energy"] for a in ("coupling_only", "matched_random_coupling")}
            complete = all(m == 0 for m in pair_missing.values())
            c = dict(status="ok", arms=arms, learned_over_control_realized=float(ratio),
                     equal_energy_within_tolerance=bool(equal), energy_records_complete=bool(complete),
                     rounding_within_tolerance=bool(rounding_ok),
                     edited_fraction_equal=bool(abs(arms["coupling_only"]["edited_fraction"]
                                                    - arms["matched_random_coupling"]["edited_fraction"]) < 1e-6),
                     pathway_over_equal_budget={a: float(arms[a]["realized_sq_l2_sum_mean"] / arms["coupling_only"]["realized_sq_l2_sum_mean"])
                                                for a in ("joint", "visual_only", "action_condition_only") if a in arms})
            if not equal:
                out["flags"].append(f"{task}: coupling learned/control realized squared-L2 ratio {ratio:.3f} outside +/-{tolerance:.0%}")
            if not rounding_ok:
                out["flags"].append(f"{task}: coupling rounding loss exceeds {rounding_tol:.0%}")
            if not c["edited_fraction_equal"]:
                out["flags"].append(f"{task}: coupling arms edited different candidate fractions")
            if not complete:
                out["flags"].append(f"{task}: coupling population calls with missing energy records: "
                                    + ", ".join(f"{a}={m}" for a, m in pair_missing.items() if m))
                LOG.warning("%s: coupling energy records incomplete (%s); cell marked dose-unverified", task, pair_missing)
            other_missing = {a: arms[a]["n_missing_energy"] for a in arms if a not in pair_missing and arms[a]["n_missing_energy"]}
            if other_missing:
                LOG.warning("%s: pathway arms with missing energy records: %s", task, other_missing)
        else:
            c = dict(status="unavailable", reason="no H6 population energy records (with squared-L2 sums) for both coupling arms")
            LOG.warning("%s: coupling dose verification skipped (%s)", task, c["reason"])
        out["coupling"][task] = c
    out["specificity_claim_dose_valid"] = {
        t: dict(fixed_rank4=bool(out["refined"][t].get("status") == "ok" and out["refined"][t]["equal_energy_within_tolerance"]
                                 and out["refined"][t]["energy_records_complete"]),
                coupling_only=bool(out["coupling"][t].get("status") == "ok" and out["coupling"][t]["equal_energy_within_tolerance"]
                                   and out["coupling"][t]["energy_records_complete"]))
        for t in tasks}
    out["dose_valid_definition"] = ("status ok AND |log(learned/control realized energy)| <= log(1+tolerance) AND no "
                                    "population call of either arm lacks an energy record")
    return out


def dose_reading(task, d):
    r, c = d["refined"][task], d["coupling"][task]
    thr = d["cv_varies_threshold"]
    parts = []

    def varies_clause(a, b):
        cva, cvb = a["realized_cv_across_scenarios"], b["realized_cv_across_scenarios"]
        who = [n for n, v in (("learned", cva), ("control", cvb)) if np.isfinite(v) and v > thr]
        desc = f"across-scenario CV of realized energy: learned {cva:.2f}, control {cvb:.2f}"
        if len(who) == 2:
            return f"{desc}; both doses vary by scenario (CV > {thr:.2f})"
        if len(who) == 1:
            return f"{desc}; only the {who[0]} dose varies by scenario (CV > {thr:.2f})"
        return f"{desc}; neither varies appreciably (CV <= {thr:.2f})"

    if r.get("status") == "ok":
        a, b = r["arms"]["fixed_rank4"], r["arms"]["matched_random_fixed_rank4"]
        parts.append(f"Refined pair: realized per-candidate L2 {a['realized_l2_mean']:.4f} (learned) vs "
                     f"{b['realized_l2_mean']:.4f} (control), ratio {r['learned_over_control_realized']:.3f}; rounding loss "
                     f"{a['rounding_loss_mean']:.2%} / {b['rounding_loss_mean']:.2%}; per-scenario ratio median "
                     f"{r['per_scenario_ratio']['median']:.2f} (range {r['per_scenario_ratio']['min']:.2f}-"
                     f"{r['per_scenario_ratio']['max']:.2f}, {r['per_scenario_ratio']['fraction_outside_tolerance']:.0%} of "
                     f"scenarios outside +/-{d['tolerance_relative']:.0%}).")
        if not r["energy_records_complete"]:
            parts.append(f"FLAG: {a['n_missing_energy']} learned / {b['n_missing_energy']} control population calls lack an "
                         "energy record, so the means above are computed on a subset and the refined specificity contrast "
                         "is dose-unverified.")
        elif r["equal_energy_within_tolerance"]:
            parts.append("The learned and control edits delivered equal mean energy, so the refined specificity contrast is "
                         "dose-valid.")
        else:
            parts.append("FLAG: the learned and control edits did NOT deliver equal mean energy; the refined specificity "
                         "contrast confounds direction with dose and should not be read as direction-specific.")
        if r["per_scenario_ratio"]["fraction_outside_tolerance"] > 0.5:
            parts.append(f"Per-scenario doses differ even though the means match ({varies_clause(a, b)}), so equal energy "
                         "holds on average, not per scenario.")
    else:
        parts.append(f"Refined pair: dose verification unavailable ({r.get('reason')}); the refined specificity contrast "
                     "is dose-unverified.")
    if c.get("status") == "ok":
        a, b = c["arms"]["coupling_only"], c["arms"]["matched_random_coupling"]
        parts.append(f"Coupling pair: edited fraction {a['edited_fraction']:.2f} / {b['edited_fraction']:.2f}, realized "
                     f"squared-L2 sum per population call {a['realized_sq_l2_sum_mean']:.3f} / {b['realized_sq_l2_sum_mean']:.3f} "
                     f"(ratio {c['learned_over_control_realized']:.3f}), realized/requested "
                     f"{a['realized_over_requested']:.4f} / {b['realized_over_requested']:.4f}; {varies_clause(a, b)}.")
        if not c["energy_records_complete"]:
            parts.append(f"FLAG: {a['n_missing_energy']} learned / {b['n_missing_energy']} control population calls lack an "
                         "energy record, so the coupling specificity contrast is dose-unverified.")
        elif c["equal_energy_within_tolerance"]:
            parts.append("Equal-budget coupling and its random-direction control delivered equal energy, so that specificity "
                         "contrast is dose-valid.")
        else:
            parts.append("FLAG: equal-budget coupling and its control did NOT deliver equal energy; that specificity contrast "
                         "is confounded with dose.")
        pj = c["pathway_over_equal_budget"]
        if pj:
            parts.append("Pathway arms relative to the equal-budget arm (realized squared L2): " +
                         ", ".join(f"{SHORT[a]} {v:.2f}x" for a, v in pj.items()) +
                         "; the unscaled joint edit is expected to carry more energy than the equal-budget edit, so "
                         "coupling_only-vs-joint compares dose as well as direction.")
    else:
        parts.append(f"Coupling pair: dose verification unavailable ({c.get('reason')}); that specificity contrast is "
                     "dose-unverified.")
    return " ".join(parts)


# ------------------------------------------------------------------ 5. offline context (read-only join)
def offline_context(tasks):
    out = {}
    for task in tasks:
        row = {}
        for learned, control in C.LEARNED.items():
            cand = C.OFFLINE_ARM_NAME.get(learned, learned)
            ctrl = C.OFFLINE_ARM_NAME.get(control, control)
            try:
                eL, eC = C.offline_effect(task, cand), C.offline_effect(task, ctrl)
            except Exception as e:  # noqa: BLE001
                LOG.warning("offline contrasts unavailable: %s", e)
                eL = eC = None
            if eL is None or eC is None:
                row[learned] = dict(status="unavailable", reason=f"no BF16 proprio_mse_h6 contrast for {cand}/{ctrl} on {task}")
                continue
            share = eC["effect"] / eL["effect"] if eL["effect"] else np.nan
            row[learned] = dict(status="ok", learned_vs_native_percent=eL, control_vs_native_percent=eC,
                                forecast_generic_share_point=float(share) if np.isfinite(share) else None,
                                learned_resolved=bool(eL["lower"] > 0), learned_resolved_harmful=bool(eL["upper"] < 0))
        out[task] = row
    return out


def offline_reading(task, spec_task, off_task):
    parts = []
    for learned in C.LEARNED:
        o = off_task[learned]
        L = C.ARM_LABEL[learned]
        if o["status"] != "ok":
            parts.append(f"{L}: no development forecast contrast available for this task ({o['reason']}), so the "
                         "forecast-level share cannot be compared.")
            continue
        eL, eC = o["learned_vs_native_percent"], o["control_vs_native_percent"]
        beh = spec_task[learned]["learned_vs_native_registered"]["difference"]
        parts.append(f"{L}: development forecast effect {eL['effect']:+.2f}% [{eL['lower']:+.2f}, {eL['upper']:+.2f}] "
                     f"of native error, control {eC['effect']:+.2f}%; forecast-level generic share "
                     f"{o['forecast_generic_share_point']:.2f} at the point estimate; fresh success change {beh:+.1f} pp.")
        if o["learned_resolved"] and beh >= C.USEFUL_GAIN_PP:
            parts.append("Forecast and success point in the same direction here; whether the success gain is "
                         "direction-specific is answered by the success share above, not by the forecast share.")
        elif o["learned_resolved"] and beh <= -C.USEFUL_GAIN_PP:
            parts.append("The forecast improves while success falls: a controlled forecast correction that is not "
                         "the planner's objective can move outcomes the wrong way.")
        elif o["learned_resolved"]:
            parts.append("The forecast effect is resolved while the success change is within +/-5 pp: consistent with "
                         "a forecast gain that does not reach the decision.")
        elif o["learned_resolved_harmful"]:
            parts.append("The development forecast effect was resolved HARMFUL on this task (the edit worsened the "
                         "forecast), so a success gain here would be evidence against the forecast-to-outcome chain and a "
                         "success loss would be consistent with it; the fresh change is "
                         + ("within +/-5 pp" if abs(beh) < C.USEFUL_GAIN_PP else ("a loss" if beh < 0 else "a gain")) + ".")
        else:
            parts.append("The development forecast effect was itself unresolved on this task, so no forecast-to-"
                         "outcome chain is expected.")
    return " ".join(parts)


# ------------------------------------------------------------------ regime
def observed_regime(spec, tasks):
    cells = [spec[t][a] for t in tasks for a in C.LEARNED]
    readings = [c["reading"] for c in cells]
    pats = [c["point_pattern"] for c in cells]
    n_conf, n_harm = readings.count("confirmed"), readings.count("harmful")
    n_pos_native = readings.count("positive_vs_native_only")
    n_pos_ctrl = readings.count("positive_vs_control_only")
    n_fav, n_unf = pats.count("favorable"), pats.count("unfavorable")
    # Registered tiers are decided by registered (vs-native) intervals; 'confirmed' additionally needs the
    # exploratory vs-control interval. 'positive_vs_control_only' is exploratory-only and never reaches a registered tier.
    if n_conf >= 1 and n_harm == 0:
        reg, tier = "positive", "registered"
    elif n_conf == 0 and n_pos_native >= 1 and n_harm == 0:
        reg, tier = "positive", "registered_vs_native_only"
    elif n_harm >= 1 and n_conf + n_pos_native == 0:
        reg, tier = "negative", "registered"
    elif (n_conf + n_pos_native) >= 1 and n_harm >= 1:
        reg, tier = "mixed", "registered"
    elif n_fav >= 2 and n_unf == 0:
        reg, tier = "positive", "point_estimate_pattern"
    elif n_unf >= 2 and n_fav == 0:
        reg, tier = "negative", "point_estimate_pattern"
    elif n_fav >= 1 and n_unf >= 1:
        reg, tier = "mixed", "point_estimate_pattern"
    else:
        reg, tier = "inconclusive", "point_estimate_pattern"
    return dict(regime=reg, tier=tier, cells=len(cells), confirmed=n_conf, harmful=n_harm,
                positive_vs_native_only=n_pos_native, positive_vs_control_only=n_pos_ctrl,
                positive_vs_one_reference=n_pos_native + n_pos_ctrl, inconclusive=readings.count("inconclusive"),
                favorable_point_cells=n_fav, unfavorable_point_cells=n_unf,
                reading_criterion=READING_CRITERION,
                rule=(f"registered tier: positive if any cell meets the {CRITERION_SHORT} and none is harmful; "
                      "registered_vs_native_only tier: positive if no cell is confirmed or harmful but at least one has a "
                      "registered vs-native lower bound > 0; negative if any harmful and none resolved above unsteered; "
                      "mixed if both. Otherwise the point-estimate tier: favorable = learned >= +5 pp vs unsteered AND "
                      ">= +5 pp vs its control; unfavorable = <= -5 pp vs unsteered; positive needs >= 2 favorable and 0 "
                      "unfavorable cells, negative the reverse, mixed both, else inconclusive. The two-reference rule is "
                      "the development-panel criterion and is not pre-registered for this confirmation."))


def headline(reg, spec, tasks, dose):
    r, tier = reg["regime"], reg["tier"]
    parts = [f"Across {reg['cells']} learned-arm cells ({len(tasks)} tasks x 2 learned arms), {reg['confirmed']} meet "
             f"the {CRITERION}, {reg['harmful']} are harmful (registered upper bound < 0 vs unsteered), "
             f"{reg['positive_vs_native_only']} are resolved above unsteered only (registered interval), "
             f"{reg['positive_vs_control_only']} above their control only (exploratory interval) and "
             f"{reg['inconclusive']} are inconclusive."]
    nf, nu = reg["favorable_point_cells"], reg["unfavorable_point_cells"]
    parts.append(f"On point estimates, {nf} cell{'s are' if nf != 1 else ' is'} favorable (>= +5 pp vs both references) "
                 f"and {nu} {'are' if nu != 1 else 'is'} unfavorable (<= -5 pp vs unsteered).")
    fav = [(t, a) for t in tasks for a in C.LEARNED if spec[t][a]["point_pattern"] == "favorable"]
    unf = [(t, a) for t in tasks for a in C.LEARNED if spec[t][a]["point_pattern"] == "unfavorable"]
    conf = cells_with_reading(spec, tasks, "confirmed")
    posn = cells_with_reading(spec, tasks, "positive_vs_native_only")
    harm = cells_with_reading(spec, tasks, "harmful")

    def spread(pairs, key):
        v = [spec[t][a][key]["difference"] for t, a in pairs]
        return f"{min(v):+.1f} to {max(v):+.1f} pp (median {np.median(v):+.1f})"

    if tier == "registered":
        parts.append(f"Observed regime: {r.upper()} at the registered tier.")
        if r == "positive":
            parts.append(f"Cells meeting the {CRITERION_SHORT}: {cells_str(conf)}"
                         + (f"; resolved above unsteered only: {cells_str(posn)}" if posn else "") + ".")
            parts.append(share_resolution_sentence(spec, conf + posn))
        elif r == "negative":
            res = lc_resolved(spec, harm)
            rest = [p for p in harm if p not in res]
            parts.append(f"Registered-harmful cells: {cells_str(harm)}.")
            if res:
                parts.append(f"On {cells_str(res)} the learned arm is also resolved below its matched random control "
                             f"(exploratory, Bonferroni-{EXPLORATORY_FAMILY}), so the harm there is specific to the learned direction.")
            if rest:
                parts.append(f"On {cells_str(rest)} the learned-minus-control interval includes zero, so direction-specific "
                             "harm is not separated from generic perturbation at the site.")
        else:  # mixed at the registered tier
            parts.append(f"Resolved above unsteered: {cells_str(conf + posn)}; registered-harmful: {cells_str(harm)}. "
                         "The sign of the registered learned effect depends on the task, which argues against a single "
                         "site-generic mechanism and for task-specific fitted directions whose behavioral consequence is "
                         "not uniformly beneficial.")
            parts.append(share_resolution_sentence(spec, conf + posn))
    elif tier == "registered_vs_native_only":
        parts.append(f"Observed regime: POSITIVE at the registered-vs-native tier: {cells_str(posn)} "
                     f"{'is' if len(posn) == 1 else 'are'} resolved above unsteered by the registered interval but not above "
                     "the matched random control (the exploratory learned-minus-control interval includes zero), so an "
                     "improvement over unsteered is established while its direction-specificity is not.")
        parts.append(share_resolution_sentence(spec, posn))
    else:
        parts.append(f"Observed regime: {r.upper()} as a point-estimate pattern; no cell meets the {CRITERION_SHORT}, none is "
                     "resolved above unsteered by a registered interval and none is registered-harmful, so this label "
                     "describes the direction of the estimates, not a resolved effect.")
        if r == "positive":
            res = lc_resolved(spec, fav)
            parts.append(f"The favorable cells are {cells_str(fav)}. Their learned-arm point changes vs unsteered are "
                         f"{spread(fav, 'learned_vs_native_registered')}; the matched random controls' are "
                         f"{spread(fav, 'control_vs_native_registered')}.")
            if res:
                parts.append(f"{len(res)} of {len(fav)} learned-minus-control exploratory intervals (Bonferroni-"
                             f"{EXPLORATORY_FAMILY}) exclude zero ({cells_str(res)}): on those cells the advantage over the "
                             "control is resolved; on the others the smaller control change is a description of point "
                             "estimates, not a resolved direction-specific effect.")
            else:
                parts.append(f"None of the {len(fav)} learned-minus-control exploratory intervals excludes zero, so the "
                             "smaller control changes describe point estimates only; direction-specificity is not "
                             "established for any cell.")
        elif r == "negative":
            res = lc_resolved(spec, unf)
            n_ctrl = sum(spec[t][a]["control_vs_native_registered"]["difference"] <= -C.USEFUL_GAIN_PP for t, a in unf)
            parts.append(f"The unfavorable cells are {cells_str(unf)}. Their learned-arm point changes vs unsteered are "
                         f"{spread(unf, 'learned_vs_native_registered')}; the matched random controls' are "
                         f"{spread(unf, 'control_vs_native_registered')} ({n_ctrl} of {len(unf)} controls also <= -5 pp).")
            if res:
                parts.append(f"{len(res)} of {len(unf)} learned-minus-control exploratory intervals (Bonferroni-"
                             f"{EXPLORATORY_FAMILY}) exclude zero ({cells_str(res)}): on those cells the loss relative to the "
                             "control is resolved; on the others whether the loss is specific to the learned direction or "
                             "shared with generic perturbation at the site is not resolved.")
            else:
                parts.append(f"None of the {len(unf)} learned-minus-control exploratory intervals excludes zero, so whether "
                             "the point losses are specific to the learned directions or shared with generic perturbation "
                             "at the site is not resolved.")
        elif r == "mixed":
            res = lc_resolved(spec, fav + unf)
            parts.append(f"Favorable: {cells_str(fav)}; unfavorable: {cells_str(unf)}. These are opposite-signed point "
                         "estimates whose registered intervals all include zero; the pattern is compatible with "
                         "task-specific directions and with noise at this n, and does not by itself argue for or against "
                         "a site-generic mechanism.")
            if res:
                parts.append(f"({len(res)} of these cells, {cells_str(res)}, has a learned-minus-control exploratory "
                             "interval excluding zero; that separates the learned arm from its control there, not from "
                             "unsteered.)")
        else:
            parts.append("No cell shows a point change of 5 pp against both references, so neither direction-specificity "
                         "nor generic perturbation can be attributed a behavioral effect; the analysis below quantifies "
                         "what the panel can and cannot resolve.")
    bad = [t for t in tasks if not all(dose["specificity_claim_dose_valid"][t].values())]
    if bad:
        parts.append("Dose caveat: on " + ", ".join(C.TASK_LABEL[t] for t in bad) + " at least one learned/control pair did "
                     "not deliver verified equal energy (or its energy records are incomplete), so the corresponding "
                     "specificity readings are not dose-valid.")
    else:
        parts.append("Dose check: every learned/control pair delivered equal mean energy within tolerance with complete "
                     "energy records, so the specificity contrasts compare direction, not dose.")
    return " ".join(parts)


# ------------------------------------------------------------------ figures
def figure_success(df, tasks, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 7.5, "axes.labelsize": 6.5, "xtick.labelsize": 6,
                         "ytick.labelsize": 6, "axes.spines.top": False, "axes.spines.right": False,
                         "font.family": "sans-serif", "hatch.linewidth": 0.5})
    n = max(len(tasks), 1)
    fig, axes = plt.subplots(1, n, figsize=(7.0, 2.2), sharey=True, squeeze=False)
    for ax, task in zip(axes[0], tasks):
        S = C.success_matrix(df, task)
        xs = np.arange(len(C.ARMS))
        for i, arm in enumerate(C.ARMS):
            k, m = int(S[arm].sum()), int(len(S))
            p = 100 * k / m
            lo, hi = wilson(k, m)
            kind = ARM_KIND[arm]
            color = {"learned": COL_LEARNED, "native": COL_NATIVE, "control": COL_CONTROL, "pathway": COL_PATHWAY}[kind]
            ax.bar(i, p, width=0.72, color=color, edgecolor="#555555" if kind == "control" else "none",
                   hatch="////" if kind == "control" else None, linewidth=0.4, zorder=2)
            ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], fmt="none", ecolor="#333333", elinewidth=0.6, capsize=1.5,
                        capthick=0.6, zorder=3)
            inside = p >= 12
            ax.text(i, (p / 2 if inside else p + 6), f"{p:.0f}", ha="center", va="center", fontsize=5.5,
                    color=("white" if inside and kind != "control" else "#222222"), zorder=4,
                    fontweight="bold" if kind == "learned" else "normal")
        ax.set_xticks(xs)
        ax.set_xticklabels([SHORT[a] for a in C.ARMS], rotation=35, ha="right", rotation_mode="anchor")
        ax.set_title(f"{C.TASK_LABEL[task]} (n={len(S)})", pad=3)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.5, zorder=0)
        ax.tick_params(length=2, pad=1.5)
    axes[0][0].set_ylabel("Success (%)")
    fig.subplots_adjust(left=0.065, right=0.995, top=0.88, bottom=0.30, wspace=0.12)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def figure_contrasts(spec, path_, tasks, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 7.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
                         "axes.spines.top": False, "axes.spines.right": False, "font.family": "sans-serif"})
    rows = [("Rank-4 − rand. subspace", lambda t: spec[t]["fixed_rank4"]["learned_vs_control_exploratory"], COL_LEARNED),
            ("Eq. coupling − rand. dir.", lambda t: spec[t]["coupling_only"]["learned_vs_control_exploratory"], COL_LEARNED),
            ("Interaction (J−V−A+N)", lambda t: path_[t]["interaction_exploratory"], COL_PATHWAY),
            ("Visual − action", lambda t: path_[t]["visual_vs_action_exploratory"], COL_PATHWAY),
            ("Eq. coupling − joint", lambda t: path_[t]["equal_budget_vs_unscaled_joint_exploratory"], COL_PATHWAY)]
    n = max(len(tasks), 1)
    fig, axes = plt.subplots(1, n, figsize=(7.0, 2.3), sharex=True, sharey=True, squeeze=False)
    ys = np.arange(len(rows))[::-1]
    for ax, task in zip(axes[0], tasks):
        ax.axvline(0, color="#999999", linewidth=0.6, zorder=1)
        for y, (label, get, color) in zip(ys, rows):
            d = get(task)
            ax.plot([d["lower"], d["upper"]], [y, y], color=color, linewidth=1.2, zorder=2, solid_capstyle="butt")
            ax.plot(d["difference"], y, "o", color=color, markersize=3.2, zorder=3)
        ax.set_yticks(ys)
        ax.set_yticklabels([r[0] for r in rows])
        ax.set_title(C.TASK_LABEL[task], pad=3)
        ax.grid(axis="x", color="#E6E6E6", linewidth=0.5, zorder=0)
        ax.tick_params(length=2, pad=1.5)
    fig.text(0.6, 0.03, "Success difference (pp); exploratory paired contrasts, Bonferroni-%d simultaneous 95%%"
             % EXPLORATORY_FAMILY, ha="center", va="bottom", fontsize=6.5)
    fig.subplots_adjust(left=0.20, right=0.985, top=0.88, bottom=0.20, wspace=0.10)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def figure_profiles(x, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 7.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
                         "axes.spines.top": False, "axes.spines.right": False, "font.family": "sans-serif"})
    tasks, arms = x["tasks"], x["arms"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.3), gridspec_kw=dict(width_ratios=[2.2, 1]))
    palette = ["#1A8F7A", "#1F5FA8", "#C77A2B", "#7A4E9E"]
    for t, col in zip(tasks, palette):
        a1.plot(np.arange(len(arms)), x["profiles_pp"][t], "-o", color=col, markersize=3, linewidth=1.0, label=C.TASK_LABEL[t])
    a1.axhline(0, color="#999999", linewidth=0.6)
    a1.set_xticks(np.arange(len(arms)))
    a1.set_xticklabels([SHORT[a] for a in arms], rotation=35, ha="right", rotation_mode="anchor")
    a1.set_ylabel("Success − unsteered (pp)")
    a1.set_title("7-arm effect profile", loc="left")
    a1.legend(frameon=False, fontsize=6, ncol=len(tasks), loc="lower right", bbox_to_anchor=(1.0, 1.0),
              handlelength=1.4, columnspacing=0.9, borderaxespad=0.0)
    a1.grid(axis="y", color="#E6E6E6", linewidth=0.5)
    M = np.full((len(tasks), len(tasks)), np.nan)
    for i, t1 in enumerate(tasks):
        for j, t2 in enumerate(tasks):
            if i == j:
                M[i, j] = 1.0
            else:
                key = f"{t1}|{t2}" if f"{t1}|{t2}" in x["pairwise_spearman"] else f"{t2}|{t1}"
                M[i, j] = x["pairwise_spearman"][key]["rho"]
    im = a2.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
    for i in range(len(tasks)):
        for j in range(len(tasks)):
            a2.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(M[i, j]) > 0.6 else "#222222")
    a2.set_xticks(range(len(tasks)))
    a2.set_yticks(range(len(tasks)))
    a2.set_xticklabels([C.TASK_LABEL[t] for t in tasks], rotation=35, ha="right", rotation_mode="anchor")
    a2.set_yticklabels([C.TASK_LABEL[t] for t in tasks])
    a2.set_title("Spearman of profiles")
    for s in ("top", "right"):
        a2.spines[s].set_visible(True)
    cb = fig.colorbar(im, ax=a2, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=5.5, length=2)
    fig.subplots_adjust(left=0.08, right=0.93, top=0.88, bottom=0.28, wspace=0.35)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ markdown
def render_md(res, tasks):
    prov, reg = res["provenance"], res["observed_regime"]
    spec, path_, x, dose, off = (res["specificity"], res["pathway"], res["cross_task"], res["dose_verification"],
                                 res["offline_context"])
    L = []
    L.append(f"# Specificity and pathway analysis ({SCRIPT}) - EXPLORATORY\n")
    L.append(f"Source: `{prov['results']}` (method `{prov['method']}`; freeze sha256 `{prov['freeze_sha256'][:12]}`, "
             f"analysis sha256 `{prov['analysis_sha256'][:12]}`). Synthetic fixture: **{prov['synthetic']}**. "
             f"Tasks analysed: {', '.join(C.TASK_LABEL[t] for t in tasks)}"
             + (f"; skipped: {', '.join(res['skipped_tasks'])}" if res["skipped_tasks"] else "") + ".\n")
    L.append("> **Scope.** Only arm-vs-native contrasts are pre-registered (read from the frozen analysis report, "
             f"Bonferroni family {res['registered_family']}). Every other interval in this document is **exploratory / "
             f"post hoc**, computed with the same paired scenario-cluster percentile bootstrap "
             f"({C.BOOTSTRAP_DRAWS} draws, seed {C.BOOTSTRAP_SEED}) and Bonferroni-corrected over a fixed family of "
             f"**{EXPLORATORY_FAMILY}** intervals (per task: learned-minus-control x2, generic-share ratio x2, "
             "interaction, visual-minus-action, equal-budget-minus-joint; x4 tasks). Ratios that are undefined still "
             "count toward the family. Nothing here was selected after seeing outcomes; the family is the full set the "
             "script can report. The permutation p for the mean pairwise Spearman is a single descriptive test (family 1, "
             "uncorrected); wins/losses are counts, not tests. The per-cell 'reading' label combines the registered "
             f"vs-native interval with the exploratory vs-control interval ({CRITERION_SHORT}); that joint rule is the "
             "development-panel criterion and is **not pre-registered** for this confirmation.\n")
    L.append("## Headline\n")
    L.append(res["headline"] + "\n")
    # 1
    L.append("## 1. Direction-specificity: generic-perturbation share\n")
    L.append("share = (control - native) / (learned - native). A ratio interval is reported only when the registered "
             "learned-vs-native lower bound is > 0; otherwise the ratio is undefined and absolute differences are given. "
             f"The point-only share is suppressed when |L-N| < {C.USEFUL_GAIN_PP:.0f} pp (near-zero denominator). "
             f"'reading' is the {CRITERION_SHORT} label (confirmed / positive_vs_native_only / positive_vs_control_only / "
             "harmful / inconclusive); its vs-control half is exploratory, not pre-registered.\n")
    rows = []
    for t in tasks:
        for a in C.LEARNED:
            c = spec[t][a]
            s = c["generic_perturbation_share"]
            iv = s["interval"]
            rows.append({"task": C.TASK_LABEL[t], "learned arm": SHORT[a],
                         "L-N pp [reg. CI]": ci(c["learned_vs_native_registered"]),
                         "C-N pp [reg. CI]": ci(c["control_vs_native_registered"]),
                         "L-C pp [expl. CI]": ci(c["learned_vs_control_exploratory"]),
                         "wins/losses": f"{c['learned_vs_control_exploratory']['wins']}/{c['learned_vs_control_exploratory']['losses']}",
                         "share": ("undefined" if not s["defined"] else f"{iv['ratio']:.2f} [{iv['lower']:.2f}, {iv['upper']:.2f}]"),
                         "share (point only)": ("n/a" if s["point_estimate"] is None else
                                                f"n/a (|L-N| < {C.USEFUL_GAIN_PP:.0f} pp)"
                                                if abs(c["learned_vs_native_registered"]["difference"]) < C.USEFUL_GAIN_PP
                                                else f"{s['point_estimate']:.2f}"),
                         "reading": c["reading"]})
    L.append(C.md_table(pd.DataFrame(rows)) + "\n")
    for t in tasks:
        for a in C.LEARNED:
            L.append(f"- **{C.TASK_LABEL[t]}** - " + share_reading(spec[t][a]))
    L.append("")
    # 2
    L.append("## 2. Pathway factorial on success\n")
    rows = []
    for t in tasks:
        p = path_[t]
        r = p["registered_vs_native"]
        rows.append({"task": C.TASK_LABEL[t], "visual-N [reg.]": ci(r["visual_only"]),
                     "action-N [reg.]": ci(r["action_condition_only"]), "joint-N [reg.]": ci(r["joint"]),
                     "eq. coupling-N [reg.]": ci(r["coupling_only"]),
                     "interaction [expl.]": ci(p["interaction_exploratory"]),
                     "visual-action [expl.]": ci(p["visual_vs_action_exploratory"]),
                     "eq. coupling-joint [expl.]": ci(p["equal_budget_vs_unscaled_joint_exploratory"]),
                     "verdict": p["additivity_verdict"]})
    L.append(C.md_table(pd.DataFrame(rows)) + "\n")
    for t in tasks:
        L.append("- " + pathway_reading(t, path_[t]))
    L.append("")
    # 3
    L.append("## 3. Cross-task consistency\n")
    if len(tasks) >= 2:
        rows = []
        for k, v in x["sign_concordance"].items():
            rows.append({"effect": k, **{C.TASK_LABEL[t]: f"{v['values_pp'][t]:+.1f}" for t in tasks},
                         "+/-/0": f"{v['positive']}/{v['negative']}/{v['zero']}",
                         "concordant": v["concordant"]})
        L.append(C.md_table(pd.DataFrame(rows)) + "\n")
        rows = [{"pair": k.replace("|", " vs "), "rho": v["rho"], "n arms": v["n"]} for k, v in x["pairwise_spearman"].items()]
        L.append(C.md_table(pd.DataFrame(rows)) + "\n")
    L.append(cross_task_reading(x) + "\n")
    # 4
    L.append("## 4. Dose verification (learned vs control delivered energy)\n")
    L.append(f"Tolerance: +/-{dose['tolerance_relative']:.0%} relative on mean delivered energy; rounding tolerance "
             f"{dose['rounding_tolerance']:.0%}. The specificity claim is dose-valid only where the pair delivered equal energy.\n")
    for t in tasks:
        L.append(f"- **{C.TASK_LABEL[t]}** - " + dose_reading(t, dose))
    if dose["flags"]:
        L.append("\nFlags: " + "; ".join(dose["flags"]))
    else:
        L.append("\nNo dose flags raised.")
    L.append("")
    # 5
    L.append("## 5. Forecast-level context (development offline contrasts, read-only join)\n")
    L.append("Development BF16 H6 proprioceptive forecast contrasts from `paper/data/all_task_ablation_contrasts.csv` "
             "(exposed pools, not the fresh cohort). Shown to relate the forecast-level specificity, which was resolved "
             "in development, to the success-level specificity above.\n")
    for t in tasks:
        L.append(f"- **{C.TASK_LABEL[t]}** - " + offline_reading(t, spec[t], off[t]))
    L.append("")
    # figures
    L.append("## Figures\n")
    L.append(f"- `{SCRIPT}_success_by_arm.png`: success rate per arm and task with **95% Wilson intervals**. These "
             "whiskers are per-arm descriptive intervals for a single proportion; they ignore the pairing of scenarios "
             "across arms and are NOT the inference. The inference is the paired within-scenario contrasts (registered "
             "arm-vs-native; exploratory learned-vs-control and factorial contrasts), whose intervals are typically "
             "narrower than the overlap of two Wilson whiskers would suggest. Learned arms teal, unsteered gray, random "
             "controls hatched gray, pathway arms muted blue; value printed in each bar.")
    L.append(f"- `{SCRIPT}_contrasts.png`: exploratory paired contrasts per task (Bonferroni-{EXPLORATORY_FAMILY}).")
    L.append(f"- `{SCRIPT}_profiles.png`: 7-arm effect profiles and their pairwise Spearman correlations.\n")
    # what it does / does not establish
    L.append("## What this does and does not establish\n")
    L.extend(what_it_establishes(res, tasks))
    L.append("")
    L.append(f"Provenance: `{SCRIPT}.json` holds every number; `exploratory: true`; deterministic "
             f"(bootstrap seed {C.BOOTSTRAP_SEED}, permutation seed {PERMUTATION_SEED}).")
    return "\n".join(L) + "\n"


def what_it_establishes(res, tasks):
    reg, spec, path_, x, dose = (res["observed_regime"], res["specificity"], res["pathway"], res["cross_task"],
                                 res["dose_verification"])
    r, tier = reg["regime"], reg["tier"]
    L = []
    defined = [(t, a) for t in tasks for a in C.LEARNED if spec[t][a]["generic_perturbation_share"]["defined"]]
    lc_cells = lc_resolved(spec, [(t, a) for t in tasks for a in C.LEARNED])
    n_lc = len(lc_cells)
    L.append("**Does establish (given the data):**")
    L.append(f"- Every learned-arm cell is classified against both references (registered vs-native interval, family "
             f"{res['registered_family']}; exploratory vs-control interval, family {EXPLORATORY_FAMILY}); "
             f"{reg['confirmed']} of {reg['cells']} meet the {CRITERION_SHORT} (a development-style rule, not "
             f"pre-registered for this run), {reg['positive_vs_native_only']} are resolved above unsteered only and "
             f"{reg['harmful']} are registered-harmful.")
    if defined:
        L.append("- " + share_resolution_sentence(spec, defined))
    else:
        L.append("- No generic-perturbation share has a defined interval: no registered learned-vs-native lower bound "
                 "exceeds zero, so the question 'what fraction of the gain is generic?' has no resolved denominator.")
    L.append(f"- {n_lc} of {2 * len(tasks)} learned-minus-control exploratory intervals exclude zero"
             + (f" ({cells_str(lc_cells)}); only there is the learned arm separated from its matched random control" if n_lc
                else "; no learned arm is separated from its matched random control") + ".")
    n_int = sum(path_[t]["interaction_exploratory"]["excludes_zero"] for t in tasks)
    L.append(f"- {n_int} of {len(tasks)} pathway interactions exclude zero at a resolution of roughly "
             f"+/-{np.mean([path_[t]['interaction_resolution_halfwidth_pp'] for t in tasks]):.0f} pp; additivity is "
             + ("rejected on " + ", ".join(C.TASK_LABEL[t] for t in tasks if path_[t]["interaction_exploratory"]["excludes_zero"])
                if n_int else "not rejected on any task") + ".")
    ok_dose = [t for t in tasks if all(dose["specificity_claim_dose_valid"][t].values())]
    L.append(f"- Delivered energy is verified equal (within +/-{dose['tolerance_relative']:.0%}, complete records) for both "
             f"learned/control pairs on {len(ok_dose)} of {len(tasks)} tasks"
             + (": " + ", ".join(C.TASK_LABEL[t] for t in ok_dose) if ok_dose else "") + ".")
    if x.get("mean_pairwise_rho"):
        m = x["mean_pairwise_rho"]
        L.append(f"- The arm ordering {'does' if (m['permutation_p'] < 0.05 and m['observed'] > 0) else 'does not'} recur "
                 f"across tasks beyond permutation (mean pairwise Spearman {m['observed']:+.2f}, {fmt_p(m)}; single "
                 "uncorrected descriptive test).")
    L.append("")
    L.append("**Does not establish:**")
    if r == "positive" and tier == "registered":
        below = [(t, a) for t, a in defined if spec[t][a]["generic_perturbation_share"]["interval"]["upper"] < 0.5]
        above = [(t, a) for t, a in defined if spec[t][a]["generic_perturbation_share"]["interval"]["lower"] > 0.5]
        if not defined:
            L.append(f"- The size of the generic component: no ratio interval is defined, so the {CRITERION_SHORT} "
                     "asserts separation from both references but not how much of the gain is direction-specific.")
        elif not below and not above:
            L.append(f"- The split between direction-specific and generic perturbation: all {len(defined)} defined share "
                     "intervals straddle one half, so 'mostly direction-specific' is a point-estimate statement only.")
        else:
            L.append(f"- A uniform mechanism: the share is resolved below one half on {cells_str(below)} and above one half "
                     f"on {cells_str(above)}; the remaining defined cells are unresolved.")
        L.append("- Pre-registered confirmation of direction-specificity: the vs-control half of the criterion is "
                 "exploratory, so 'confirmed' here is a development-style reading pending a registered vs-control contrast.")
    elif r == "positive" and tier == "registered_vs_native_only":
        posn = cells_with_reading(spec, tasks, "positive_vs_native_only")
        L.append(f"- Direction-specificity: on {cells_str(posn)} the registered vs-native interval excludes zero but the "
                 "exploratory learned-minus-control interval does not, so the improvement is not separated from generic "
                 "perturbation at the site.")
    elif r == "positive":
        L.append("- A behavioral improvement: no registered interval excludes zero. The 'positive' label is the direction "
                 f"of point estimates and would not survive the {CRITERION_SHORT}.")
        L.append(f"- Direction-specificity: {n_lc} of {2 * len(tasks)} learned-minus-control intervals exclude zero, so the "
                 "smaller control changes are point-estimate descriptions" + (" except on " + cells_str(lc_cells) if n_lc else "") + ".")
    elif r == "negative":
        if tier != "registered":
            L.append("- A behavioral harm: no registered upper bound is below zero, so the 'negative' label is the direction "
                     "of point estimates, not a resolved loss.")
            L.append(f"- Direction-specificity of the point losses: {n_lc} of {2 * len(tasks)} learned-minus-control intervals "
                     "exclude zero" + (f" ({cells_str(lc_cells)})" if n_lc else "") + "; elsewhere learned-direction harm and "
                     "generic perturbation are not separated.")
        else:
            harm = cells_with_reading(spec, tasks, "harmful")
            rest = [p for p in harm if p not in lc_cells]
            if rest:
                L.append(f"- Direction-specificity of the harm on {cells_str(rest)}: the learned-minus-control interval "
                         "includes zero there.")
            L.append("- Which component of the edit is harmful: the factorial resolves only the main effects listed above.")
    elif r == "mixed":
        L.append("- A task-general mechanism: opposite-signed " + ("registered" if tier == "registered" else "point")
                 + " estimates across tasks are compatible with task-specific fitted directions"
                 + (" and with noise around zero at this n; the intervals do not separate these." if tier != "registered"
                    else "; which property of a task flips the sign is not identified here."))
    else:
        hw = np.mean([(spec[t][a]["learned_vs_native_registered"]["upper"] - spec[t][a]["learned_vs_native_registered"]["lower"]) / 2
                      for t in tasks for a in C.LEARNED])
        L.append(f"- An effect of either sign: the registered learned-vs-native intervals have a mean half-width of "
                 f"{hw:.0f} pp (normal-approximation 80%-power detectable difference about "
                 f"{C.power_pp(family=res['registered_family']):.0f} pp), so effects below that scale are not excluded.")
    L.append("- Mechanism: a success-level share or interaction says nothing about which activations changed or "
             "why; the decision-level and geometry analyses are separate scripts.")
    L.append("- Transfer: each task uses its own fitted directions; cross-task concordance of signs is a property of "
             "the fitting recipe, not of one shared direction.")
    L.append("- Any claim selected after seeing these numbers; the exploratory family is fixed and reported, but the "
             "readings are post hoc and would need a fresh confirmation to become evidence.")
    L.append("- Anything about the decodability of physical variables at the site (in the sense of a 'physics zone'): "
             "a controllable behavioral share is not a decoded variable.")
    return L


# ------------------------------------------------------------------ main
def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = C.standard_parser("Specificity (learned vs generic perturbation) and pathway factorial analysis; exploratory.")
    ap.add_argument("--fixture-regime", default=None, help="declared fixture regime (recorded only; never used for text)")
    ap.add_argument("--energy-tolerance", type=float, default=ENERGY_TOLERANCE)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    prov = C.provenance(args.results, args.freeze, args.analysis)
    report = C.load_analysis(args.analysis)
    registered_family = int(report.get("analysis", {}).get("family_size", REGISTERED_FAMILY_FALLBACK))
    LOG.info("loading episodes frame")
    df = C.episodes_frame(args.results, tasks=args.tasks)

    tasks, skipped = [], []
    for t in args.tasks:
        if t not in report.get("results", {}):
            skipped.append(f"{t} (not in analysis report)")
            continue
        if not len(df) or t not in set(df.task):
            skipped.append(f"{t} (no completed scenarios in results tree)")
            continue
        S = C.success_matrix(df, t)
        missing = [a for a in C.ARMS if a not in S.columns]
        if missing or len(S) == 0:
            skipped.append(f"{t} (missing arms {missing} or no complete scenarios)")
            continue
        tasks.append(t)
    for s in skipped:
        LOG.warning("skipping task %s", s)
    if not tasks:
        raise SystemExit("no task with a complete eight-arm panel; nothing to analyse")

    spec, path_, S_by_task = {}, {}, {}
    for t in tasks:
        S = C.success_matrix(df, t)
        S_by_task[t] = S
        if len(S) != report["results"][t]["n"]:
            LOG.warning("%s: %d complete scenarios in tree vs n=%d in report", t, len(S), report["results"][t]["n"])
        spec[t] = specificity_for_task(S, report, t)
        path_[t] = pathway_for_task(S, report, t)
        for a, c in spec[t].items():
            if not c["registered_point_reproduced"]:
                LOG.warning("%s/%s: reproduced learned-vs-native POINT differs from report (tree %.2f vs report %.2f)", t, a,
                            c["learned_vs_native_reproduced"]["difference"], c["learned_vs_native_registered"]["difference"])
            elif c["registered_lower_abs_diff_pp"] > 0.05:
                LOG.info("%s/%s: registered lower bound differs from a fresh-seed reproduction by %.2f pp (resampling "
                         "stream differs; point estimate identical)", t, a, c["registered_lower_abs_diff_pp"])
    x = cross_task(report, tasks, spec)
    dose = dose_verification(args.results, tasks, tolerance=args.energy_tolerance)
    off = offline_context(tasks)
    reg = observed_regime(spec, tasks)

    res = dict(script=SCRIPT, exploratory=True,
               exploratory_family=EXPLORATORY_FAMILY,
               exploratory_family_definition=("per task: fixed_rank4-control, coupling_only-control, share(fixed_rank4), "
                                              "share(coupling_only), interaction, visual-action, coupling_only-joint; x4 tasks"),
               registered_family=registered_family,
               registered_contrasts="arm-vs-native only, from ANALYSIS/report.json",
               reading_criterion=READING_CRITERION, reading_criterion_definition=CRITERION,
               provenance=prov, declared_fixture_regime=args.fixture_regime, tasks=tasks, skipped_tasks=skipped,
               observed_regime=reg, specificity=spec, pathway=path_, cross_task=x, dose_verification=dose,
               offline_context=off,
               success_counts={t: {a: int(S_by_task[t][a].sum()) for a in C.ARMS} for t in tasks},
               n_scenarios={t: int(len(S_by_task[t])) for t in tasks},
               wilson_95_percent={t: {a: list(wilson(int(S_by_task[t][a].sum()), len(S_by_task[t]))) for a in C.ARMS} for t in tasks},
               seeds=dict(bootstrap=C.BOOTSTRAP_SEED, permutation=PERMUTATION_SEED))
    res["headline"] = headline(reg, spec, tasks, dose)

    figure_success(df, tasks, out / f"{SCRIPT}_success_by_arm.png")
    figure_contrasts(spec, path_, tasks, out / f"{SCRIPT}_contrasts.png")
    if len(tasks) >= 2:
        figure_profiles(x, out / f"{SCRIPT}_profiles.png")
    else:
        LOG.warning("profiles figure skipped: fewer than two tasks")
    C.write_json(out / f"{SCRIPT}.json", res)
    (out / f"{SCRIPT}.md").write_text(render_md(res, tasks))
    LOG.info("regime=%s (%s); wrote %s", reg["regime"], reg["tier"], out)
    print(f"{SCRIPT}: regime={reg['regime']} tier={reg['tier']} tasks={tasks} out={out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
