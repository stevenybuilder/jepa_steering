"""regime_report: first-look reading of the fresh four-task eight-arm confirmation.

Answers four questions from finished artifacts only (no model, simulator or GPU):
  1. What did the pre-registered analysis say?  Reproduces EVERY registered contrast in the
     frozen protocol (12 per task: 7 arm-vs-native, refined-random, coupling-random, joint-visual,
     joint-action, factorial-interaction; 12 x 4 = family 48) from the raw records with
     common.paired_bootstrap at the protocol family and checks agreement with the frozen report.
     A second, stricter check replays the analyzer's single RNG stream (defined here, not in
     common) and is expected to match to floating-point precision.
  2. Which regime is each (task, learned arm) cell in, and what is the overall regime?
     common.classify_regime with the registered vs-native interval and the registered
     learned-vs-its-control interval, both Bonferroni-48 from the frozen report.
  3. Does the fresh (untouched) panel replicate the development (exposed) panels?
  4. Which conclusions are licensed by the observed numbers, and which are not.

Pre-registered: the 12 registered contrasts per task at family 48 (as read from the protocol).
EXPLORATORY (stamped `exploratory: true` in the JSON, labelled in the Markdown, with the family
corrected over): the discordance p-values (Holm over the two-arm registered contrasts), the
family-8 control-leg sensitivity, the unadjusted family-1 orientation reading, the replication
comparison and the power table. Every interpretive sentence is generated from the numbers and
branches on the observed regime; nothing is hard-coded.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT = "regime_report"
CONTROL_SENSITIVITY_FAMILY = 8   # exploratory: 2 learned arms x 4 tasks, control leg only
REPRO_DIFF_TOL_PP = 1e-6         # point estimates are deterministic given the records
REPRO_BOUND_TOL_PP = 2.5         # Monte-Carlo tolerance on Bonferroni-48 percentile endpoints (fresh RNG per call)
REPLAY_TOL_PP = 1e-6             # analyzer-stream replay should match to floating point
NAV_REPORT = C.ROOT / "artifacts/offline_study/table-completion-20260911-v1/navigation-final-analysis-v1/analysis/report.json"
FIG_SCRIPT = C.ROOT / "paper/scripts/build_workshop_figures.py"
C_PRE = "#1f5fa8"
C_EXP = "#8c8c8c"
C_HARM = "#c2492b"
C_OK = "#1a8f7a"
REGIME_COLOR = {"confirmed": C_OK, "positive_vs_native_only": "#7fb3d5", "positive_vs_control_only": "#b8b8b8",
                "harmful": C_HARM, "inconclusive": "#ececec"}
REGIME_TEXT = {
    "confirmed": "confirmed",
    "positive_vs_native_only": "positive vs unsteered only",
    "positive_vs_control_only": "positive vs its random control only",
    "harmful": "harmful",
    "inconclusive": "inconclusive",
}
LOG = []


def log(msg):
    LOG.append(msg)
    print(f"[{SCRIPT}] {msg}")


def pp(x):
    return f"{x:+.1f}" if np.isfinite(x) else "n/a"


def ci(lo, hi):
    return f"[{lo:+.1f}, {hi:+.1f}]"


def names(cells):
    return ", ".join(f"{C.TASK_LABEL[c['task']]} {C.ARM_LABEL[c['arm']]}" for c in cells)


# ----------------------------------------------------------- registered contrasts
def contrast_kind(weights):
    """Classify a registered contrast from its weight dict; returns (kind, arm, control)."""
    pos = [a for a, w in weights.items() if w > 0]
    neg = [a for a, w in weights.items() if w < 0]
    if len(pos) == 1 and len(neg) == 1:
        arm, ctrl = pos[0], neg[0]
        if ctrl == "native":
            return "arm_vs_native", arm, ctrl
        if C.LEARNED.get(arm) == ctrl:
            return "learned_vs_control", arm, ctrl
        return "pathway", arm, ctrl
    return "interaction", None, None


def contrast_label(name, kind, arm, ctrl, weights):
    if kind == "arm_vs_native":
        return f"{C.ARM_LABEL[arm]} - unsteered"
    if kind == "learned_vs_control":
        return f"{C.ARM_LABEL[arm]} - its random control"
    if kind == "pathway":
        return f"{C.ARM_LABEL[arm]} - {C.ARM_LABEL[ctrl]}"
    ordered = sorted(weights.items(), key=lambda kv: (kv[1] < 0, kv[0] == "native",
                                                     C.ARMS.index(kv[0]) if kv[0] in C.ARMS else 99))
    terms = " ".join(("+ " if w > 0 else "- ") + C.ARM_LABEL.get(a, a) for a, w in ordered)
    return f"Interaction ({terms[2:]})"


def registered_contrasts(protocol, report):
    """Ordered dict name -> weights from the frozen protocol (fallback: the 7 arm-vs-native)."""
    cons = protocol.get("contrasts")
    if not cons:
        log("protocol has no 'contrasts' block; falling back to the 7 arm-vs-native contrasts (family as reported)")
        cons = {f"{a}-native": {a: 1, "native": -1} for a in C.ARMS[1:]}
    order = [f"{a}-native" for a in C.ARMS[1:]] + ["refined-random", "coupling-random", "joint-visual",
                                                     "joint-action", "factorial-interaction"]
    ordered = {k: cons[k] for k in order if k in cons}
    ordered.update({k: v for k, v in cons.items() if k not in ordered})
    return ordered


def split_weights(values, weights):
    """a = positive part, b = negative part, both in pp so common.paired_bootstrap returns pp."""
    wp = np.asarray([max(weights.get(a, 0), 0) for a in C.ARMS], float)
    wn = np.asarray([max(-weights.get(a, 0), 0) for a in C.ARMS], float)
    return 100.0 * (values @ wp), 100.0 * (values @ wn)


def analyzer_stream_indices(seed, draws, n_by_task):
    """Replay of fresh_confirmation.analyze: ONE rng stream, one index matrix per task in TASKS
    order, shared by all contrasts of that task. Data never enters the stream."""
    rng = np.random.default_rng(seed)
    out = {}
    for task in C.TASKS:
        n = int(n_by_task.get(task, 96))
        out[task] = rng.integers(0, n, size=(draws, n))
    return out


def preregistered(report, S, task, family, contrasts, replay_idx):
    """Frozen registered contrasts for one task + reproduction from the raw success matrix."""
    frozen = report["results"][task]
    values = S[list(C.ARMS)].values.astype(float)
    alpha = 0.05 / family
    out = []
    for name, weights in contrasts.items():
        if name not in frozen["contrasts"]:
            log(f"{task}: registered contrast '{name}' absent from the frozen report; skipped")
            continue
        fz = frozen["contrasts"][name]
        kind, arm, ctrl = contrast_kind(weights)
        a, b = split_weights(values, weights)
        rep = C.paired_bootstrap(a, b, family=family)
        rep1 = C.paired_bootstrap(a, b, family=1)
        d = values @ np.asarray([weights.get(x, 0) for x in C.ARMS], float)
        if replay_idx is not None:
            lo_r, hi_r = np.quantile(d[replay_idx].mean(1), [alpha / 2, 1 - alpha / 2]) * 100
        else:
            lo_r = hi_r = float("nan")
        row = dict(
            task=task, contrast=name, kind=kind, arm=arm, control=ctrl, weights=dict(weights),
            label=contrast_label(name, kind, arm, ctrl, weights),
            preregistered=True, exploratory=False, family=family, n=int(len(S)),
            difference=float(fz["difference_pp"]),
            lower=float(fz["simultaneous_95_interval_pp"][0]), upper=float(fz["simultaneous_95_interval_pp"][1]),
            reproduced=dict(difference=rep["difference"], lower=rep["lower"], upper=rep["upper"],
                            draws=rep["draws"], seed=rep["seed"], estimator="common.paired_bootstrap (fresh RNG per call)"),
            reproduction_abs_deviation_pp=dict(difference=abs(rep["difference"] - fz["difference_pp"]),
                                               lower=abs(rep["lower"] - fz["simultaneous_95_interval_pp"][0]),
                                               upper=abs(rep["upper"] - fz["simultaneous_95_interval_pp"][1])),
            analyzer_replay=dict(lower=float(lo_r), upper=float(hi_r),
                                 abs_deviation_pp=dict(lower=abs(lo_r - fz["simultaneous_95_interval_pp"][0]),
                                                       upper=abs(hi_r - fz["simultaneous_95_interval_pp"][1]))),
            unadjusted_family1=dict(exploratory=True, family=1, lower=rep1["lower"], upper=rep1["upper"]),
        )
        if arm is not None:
            row["success_percent_arm"] = float(frozen["success_percent"][arm])
            row["success_percent_control"] = float(frozen["success_percent"][ctrl])
            disc = C.discordance(S[arm].values, S[ctrl].values)
            row.update(wins=disc["wins"], losses=disc["losses"], discordant=disc["discordant"], exact_p=disc["exact_p"])
        out.append(row)
    return out


def control_sensitivity(S, task, family):
    """EXPLORATORY: learned-vs-control interval at Bonferroni over the 8 control contrasts only."""
    out = []
    for arm, ctrl in C.LEARNED.items():
        b = C.paired_bootstrap(S[arm].values, S[ctrl].values, family=family)
        out.append(dict(task=task, arm=arm, control=ctrl, contrast=f"{arm}-{ctrl}", exploratory=True,
                        preregistered=False, family=family, difference=b["difference"], lower=b["lower"], upper=b["upper"],
                        note="sensitivity only: the registered family-48 interval is the classification input"))
    return out


# ---------------------------------------------------------------------- regime
def overall_regime(cells):
    labels = [c["regime"] for c in cells]
    n_conf = labels.count("confirmed")
    n_harm = labels.count("harmful")
    n_part = labels.count("positive_vs_native_only") + labels.count("positive_vs_control_only")
    if n_conf and not n_harm:
        return "positive"
    if n_harm and not n_conf and not n_part:
        return "negative"
    if n_conf or n_harm or n_part:
        return "mixed"
    return "inconclusive"


OVERALL_RULE = ("positive: >=1 confirmed cell and no harmful cell; negative: >=1 harmful cell, no confirmed and "
                "no partial-positive cell; mixed: any other combination with at least one non-inconclusive cell "
                "(confirmed together with harmful, harmful together with partial-positive, or partial-positive only); "
                "inconclusive: every cell inconclusive.")


# ----------------------------------------------------------------- development
def load_refined_transcription():
    """REFINED dict transcribed in paper/scripts/build_workshop_figures.py (no import: side effects)."""
    if not FIG_SCRIPT.exists():
        log(f"development REFINED transcription absent ({FIG_SCRIPT}); skipping PointMaze/Wall refined replication")
        return None
    text = FIG_SCRIPT.read_text()
    m = re.search(r"^REFINED\s*=\s*(\{.*?^\})", text, re.S | re.M)
    if not m:
        log("could not locate REFINED dict in build_workshop_figures.py; skipping")
        return None
    body = re.sub(r"#[^\n]*", "", m.group(1))
    return ast.literal_eval(body)


def development_contrasts():
    """Development point estimates + intervals per (task, arm, control) with their source."""
    rows = {}
    try:
        core = C.load_core_panel()
        for c in core["contrasts"]:
            lo, hi = c["simultaneous_95_interval_percentage_points"]
            rows[(c["task"], c["arm"], c["control"])] = dict(
                difference=float(c["gain_percentage_points"]), lower=float(lo), upper=float(hi),
                family=int(core["analysis"]["family"]), source="reports/wm-approaches/core-analysis.json",
                exposed=True)
    except FileNotFoundError as e:
        log(f"core development panel absent ({e}); skipping Reach/Reach-Wall replication")
    refined = load_refined_transcription()
    if refined:
        for task in ("pointmaze", "wall"):
            if task not in refined:
                continue
            for key, (e, lo, hi) in refined[task].items():
                arm, ctrl = key.split("-", 1)
                rows[(task, arm, ctrl)] = dict(difference=float(e), lower=float(lo), upper=float(hi), family=12,
                                               source="paper/scripts/build_workshop_figures.py REFINED (transcribed)",
                                               exposed=True)
    if NAV_REPORT.exists():
        nav = C.read_json(NAV_REPORT)
        nm = {"joint_equal_standardized_energy_vs_native": ("coupling_only", "native"),
              "equal_energy_joint_vs_matched_random": ("coupling_only", "matched_random_coupling"),
              "matched_random_equal_standardized_energy_vs_native": ("matched_random_coupling", "native")}
        for c in nav["contrasts"]:
            if c["contrast"] in nm and c["task"] in ("pointmaze", "wall"):
                arm, ctrl = nm[c["contrast"]]
                lo, hi = c["simultaneous_95_interval_percentage_points"]
                rows[(c["task"], arm, ctrl)] = dict(difference=float(c["difference_percentage_points"]),
                                                   lower=float(lo), upper=float(hi), family=32,
                                                   source=str(NAV_REPORT.relative_to(C.ROOT)), exposed=True)
    else:
        log(f"navigation development report absent ({NAV_REPORT}); skipping PointMaze/Wall coupling replication")
    return rows


def replication_rows(pre, dev, tasks):
    fresh = {(r["task"], r["arm"], r["control"]): r for r in pre if r["arm"] is not None}
    rows = []
    for task in tasks:
        for arm, ctrl in C.LEARNED.items():
            for control in ("native", ctrl):
                f = fresh.get((task, arm, control))
                d = dev.get((task, arm, control))
                if f is None:
                    continue
                if d is None:
                    log(f"no development estimate for {task} {arm} vs {control}; replication row skipped")
                    rows.append(dict(task=task, arm=arm, control=control, available=False))
                    continue
                fd, dd = f["difference"], d["difference"]
                if dd == 0 or fd == 0:
                    sign = "flat" if dd == 0 else "fresh_flat"
                else:
                    sign = "agree" if np.sign(fd) == np.sign(dd) else "disagree"
                rows.append(dict(
                    task=task, arm=arm, control=control, available=True, exploratory=True,
                    fresh_difference=fd, fresh_lower=f["lower"], fresh_upper=f["upper"], fresh_family=f["family"],
                    dev_difference=dd, dev_lower=d["lower"], dev_upper=d["upper"], dev_family=d["family"],
                    dev_source=d["source"], dev_scenarios_exposed=True, fresh_scenarios_exposed=False,
                    sign=sign, dev_point_inside_fresh_interval=bool(f["lower"] <= dd <= f["upper"]),
                    fresh_point_inside_dev_interval=bool(d["lower"] <= fd <= d["upper"]),
                    shrinkage_ratio=(fd / dd) if abs(dd) >= 1.0 else float("nan"),
                    shrinkage_note=None if abs(dd) >= 1.0 else "development estimate within 1 pp of zero; ratio undefined",
                ))
    return rows


# ------------------------------------------------------------------------ power
def power_table(report, tasks, n_per_task, family):
    rows = []
    for task in tasks:
        if task not in report["results"]:
            continue
        p_nat = report["results"][task]["success_percent"]["native"] / 100.0
        n = int(n_per_task[task])
        for fam in (1, family):
            rows.append(dict(task=task, family=fam, n=n, native_rate=p_nat, exploratory=True,
                             detectable_pp_80pct_at_native_rate=C.power_pp(n, p_nat, family=fam),
                             detectable_pp_80pct_at_p50=C.power_pp(n, 0.5, family=fam),
                             preregistered_useful_pp=C.USEFUL_GAIN_PP))
    return rows


# ------------------------------------------------------------- licensed claims
def control_vs_native_text(ctrl):
    if ctrl["upper"] < 0:
        return (f"its random control is itself below unsteered (interval entirely below zero: {pp(ctrl['difference'])} pp "
                f"{ci(ctrl['lower'], ctrl['upper'])}), so the contrast is consistent with the control being harmful "
                "rather than the edit helping")
    if ctrl["difference"] < 0:
        return (f"its random control's point estimate is below unsteered ({pp(ctrl['difference'])} pp "
                f"{ci(ctrl['lower'], ctrl['upper'])}; interval includes zero), so a harmful control is not excluded as "
                "the explanation")
    return (f"its random control is not below unsteered ({pp(ctrl['difference'])} pp {ci(ctrl['lower'], ctrl['upper'])}), "
            "so the advantage is not explained by a harmful control")


def licensed_claims(cell, pre_task, half_width):
    task, arm = cell["task"], cell["arm"]
    tl, al = C.TASK_LABEL[task], C.ARM_LABEL[arm]
    reg = cell["regime"]
    n, c = cell["vs_native"], cell["vs_control"]
    ctrl = pre_task[C.LEARNED[arm]]
    lic, not_lic = [], []
    if reg == "confirmed":
        lic.append(f"A direction-specific closed-loop success improvement of the {al} on {tl} on this released "
                   f"checkpoint: {pp(n['difference'])} pp over unsteered {ci(n['lower'], n['upper'])} and "
                   f"{pp(c['difference'])} pp over its dose-matched random control {ci(c['lower'], c['upper'])} (both "
                   f"pre-registered, Bonferroni {n['family']}), with the point gain at or above the pre-registered "
                   f"{C.USEFUL_GAIN_PP:.0f} pp useful effect.")
        not_lic += ["transfer of the effect to other tasks, other checkpoints or other training seeds",
                    "a mechanism for the improvement (the decision-level diagnostic is a separate analysis)",
                    "that the offline forecast gain caused the behavioral gain",
                    "any claim about the size beyond the reported interval (the half-width is "
                    f"{half_width:.1f} pp)"]
    elif reg == "positive_vs_native_only":
        lic.append(f"The {al} improves success over the unsteered model on {tl} on this checkpoint "
                   f"({pp(n['difference'])} pp {ci(n['lower'], n['upper'])}, pre-registered, family {n['family']}).")
        not_lic += [f"that the improvement is due to the learned directions: the registered contrast against the "
                    f"dose-matched random control is {pp(c['difference'])} pp {ci(c['lower'], c['upper'])} and does not "
                    f"exclude zero, so the gain is not distinguishable from a generic perturbation of matched dose",
                    "transfer, mechanism or multi-seed generality"]
    elif reg == "positive_vs_control_only":
        lic.append(f"On {tl}, the {al} outperforms its dose-matched random control ({pp(c['difference'])} pp "
                   f"{ci(c['lower'], c['upper'])}; pre-registered, family {c['family']}), but not the unsteered model "
                   f"({pp(n['difference'])} pp {ci(n['lower'], n['upper'])}).")
        not_lic += ["any improvement over the unsteered planner (the registered contrast includes zero)",
                    f"a learned-direction advantage as a behavioral improvement: {control_vs_native_text(ctrl)}"]
    elif reg == "harmful":
        lic.append(f"The {al} reduces success on {tl} on this checkpoint ({pp(n['difference'])} pp "
                   f"{ci(n['lower'], n['upper'])}, pre-registered interval entirely below zero).")
        not_lic += [("that the harm is direction-specific: " +
                     (f"the random control is also below zero ({pp(ctrl['difference'])} pp {ci(ctrl['lower'], ctrl['upper'])}), "
                      "so matched-dose perturbation as such is harmful here" if ctrl["upper"] < 0 else
                      f"the random control vs unsteered is {pp(ctrl['difference'])} pp {ci(ctrl['lower'], ctrl['upper'])} "
                      "(interval includes zero); the learned-vs-control contrast is "
                      f"{pp(c['difference'])} pp {ci(c['lower'], c['upper'])}" +
                      (", entirely below zero, so the fitted directions are worse than a random direction of the same dose"
                       if c["upper"] < 0 else ", not distinguishable from zero"))),
                    "generalisation of the harm to other tasks or checkpoints"]
    else:
        lic.append(f"On {tl}, no success change of the {al} at or beyond the panel's resolution was detected "
                   f"({pp(n['difference'])} pp {ci(n['lower'], n['upper'])} vs unsteered; {pp(c['difference'])} pp "
                   f"{ci(c['lower'], c['upper'])} vs its control; both pre-registered, family {n['family']}). The "
                   f"interval half-width is about {half_width:.1f} pp against a pre-registered useful effect of "
                   f"{C.USEFUL_GAIN_PP:.0f} pp.")
        if c["upper"] < 0:
            lic.append(f"The {al} does worse than its dose-matched random control on {tl} ({pp(c['difference'])} pp "
                       f"{ci(c['lower'], c['upper'])}, registered, family {c['family']}); a random direction of the same "
                       "dose is less harmful than the fitted directions. `classify_regime` has no category for this.")
        not_lic += ["that the effect is zero (a real effect smaller than the resolution is not excluded)",
                    "that the edit is inert (see the discordant-outcome counts: "
                    f"{n['wins']} wins / {n['losses']} losses vs unsteered)"]
    return lic, not_lic


# ---------------------------------------------------------------------- figures
ROW_ORDER = ["fixed_rank4-native", "matched_random_fixed_rank4-native", "refined-random",
             "coupling_only-native", "matched_random_coupling-native", "coupling-random",
             "joint-native", "visual_only-native", "action_condition_only-native",
             "joint-visual", "joint-action", "factorial-interaction"]


def forest_figure(pre, dev, out_png, synthetic, tasks, n_per_task, family):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                         "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "axes.spines.top": False, "axes.spines.right": False})
    lookup = {(r["task"], r["contrast"]): r for r in pre}
    present = [k for k in ROW_ORDER if any((t, k) in lookup for t in tasks)]
    present += sorted({r["contrast"] for r in pre} - set(present))
    labels = {r["contrast"]: r["label"] for r in pre}
    fig, axes = plt.subplots(1, len(tasks), figsize=(2.5 * len(tasks), 0.32 * len(present) + 1.6),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes[0]
    ys = np.arange(len(present))[::-1]
    for ax, task in zip(axes, tasks):
        ax.axvspan(-C.USEFUL_GAIN_PP, C.USEFUL_GAIN_PP, color="#000000", alpha=0.04, lw=0)
        ax.axvline(0, color="k", lw=0.6, alpha=0.7)
        for y, key in zip(ys, present):
            r = lookup.get((task, key))
            if r is None:
                continue
            u = r["unadjusted_family1"]
            ax.plot([r["lower"], r["upper"]], [y, y], color=C_PRE, lw=1.3, solid_capstyle="round", zorder=2)
            ax.plot([u["lower"], u["upper"]], [y, y], color=C_EXP, lw=3.2, alpha=0.45, solid_capstyle="butt", zorder=1)
            ax.plot(r["difference"], y, marker="o", ms=4.2, color=C_PRE, mec="white", mew=0.5, zorder=3)
            d = dev.get((task, r["arm"], r["control"])) if r["arm"] is not None else None
            if d is not None:
                ax.plot([d["lower"], d["upper"]], [y + 0.34, y + 0.34], color=C_PRE, lw=0.6, alpha=0.5, ls=":")
                ax.plot(d["difference"], y + 0.34, marker="o", ms=4.2, mfc="white", mec=C_PRE, mew=1.0, zorder=3)
        ax.set_title(f"{C.TASK_LABEL[task]} (n = {n_per_task[task]})")
        ax.set_yticks(ys)
        ax.set_yticklabels([labels[k] for k in present])
        ax.set_ylim(-0.7, len(present) - 0.3)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#dddddd", lw=0.5)
        ax.set_axisbelow(True)
    fig.supxlabel("Success difference (percentage points, paired scenarios)", fontsize=8, y=0.02)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=C_PRE, marker="o", ms=4, label=f"fresh, pre-registered (Bonferroni {family})"),
               Line2D([], [], color=C_EXP, lw=3.2, alpha=0.45, label="fresh, exploratory unadjusted (family 1)"),
               Line2D([], [], color=C_PRE, marker="o", ms=4, mfc="white", ls=":", lw=0.6,
                      label="development (exposed scenarios; own family)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.055))
    title = "Fresh confirmation contrasts vs development" + (" [SYNTHETIC FIXTURE]" if synthetic else "")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


REGIME_SHORT = {"confirmed": "confirmed", "positive_vs_native_only": "positive vs\nunsteered only",
                "positive_vs_control_only": "positive vs\ncontrol only", "harmful": "harmful",
                "inconclusive": "inconclusive"}


def regime_figure(cells, overall, out_png, synthetic, tasks):
    fig, ax = plt.subplots(figsize=(1.4 * len(tasks) + 0.6, 2.2))
    arms = list(C.LEARNED)
    for i, task in enumerate(tasks):
        for j, arm in enumerate(arms):
            c = next(x for x in cells if x["task"] == task and x["arm"] == arm)
            dark = c["regime"] in ("confirmed", "harmful")
            ax.add_patch(plt.Rectangle((i, j), 1, 1, color=REGIME_COLOR[c["regime"]], ec="white", lw=1.5))
            ax.text(i + 0.5, j + 0.64, REGIME_SHORT[c["regime"]], ha="center", va="center", fontsize=6.5,
                    color="white" if dark else "black")
            ax.text(i + 0.5, j + 0.22, f"{pp(c['vs_native']['difference'])} / {pp(c['vs_control']['difference'])} pp",
                    ha="center", va="center", fontsize=6.5, color="white" if dark else "black")
    ax.set_xlim(0, len(tasks))
    ax.set_ylim(0, len(arms))
    ax.set_xticks(np.arange(len(tasks)) + 0.5)
    ax.set_xticklabels([C.TASK_LABEL[t] for t in tasks])
    ax.set_yticks(np.arange(len(arms)) + 0.5)
    ax.set_yticklabels([C.ARM_LABEL[a] for a in arms])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Cell regimes (pre-registered reading); overall: {overall}" + (" [SYNTHETIC]" if synthetic else ""),
                 fontsize=8)
    ax.text(0, -0.55, "cell text: gain vs unsteered / gain vs its random control (both registered)", fontsize=6.5,
            color="#555555")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------- markdown
def write_markdown(path, J):
    prov, syn = J["provenance"], J["provenance"]["synthetic"]
    ov = J["overall_regime"]
    tasks = J["tasks"]
    pre = J["preregistered_contrasts"]
    fam = J["preregistered_family"]
    n_reg = len(pre)
    per_task = len(J["registered_contrast_names"])
    n_cells = len(J["cells"])
    L = []
    L.append("# Regime report: fresh four-task eight-arm confirmation\n")
    if syn:
        L.append("> **SYNTHETIC FIXTURE.** These records were generated by `make_fixture.py`; every number below is "
                 "planted, not measured. The text is produced by the same rules that will run on the real results.\n")
    n_txt = ", ".join(f"{C.TASK_LABEL[t]} {J['n_scenarios_per_task'][t]}" for t in tasks)
    L.append(f"Method `{prov['method']}`; freeze sha256 `{prov['freeze_sha256'][:16]}…`; analysis report sha256 "
             f"`{prov['analysis_sha256'][:16]}…`. Tasks read: {len(tasks)} ({', '.join(C.TASK_LABEL[t] for t in tasks)}); "
             f"paired scenarios per task: {n_txt}. Bootstrap: {C.BOOTSTRAP_DRAWS} paired scenario-cluster draws, seed "
             f"{C.BOOTSTRAP_SEED}.\n")
    L.append(f"**Scope of pre-registration.** The frozen protocol registers {per_task} contrasts per task "
             f"({', '.join(J['registered_contrast_names'])}), Bonferroni family {fam} "
             f"({per_task} x {J['protocol_task_count']} tasks); {n_reg} of them are read here. " +
             ("Both legs of the cell classification (learned vs unsteered, learned vs its dose-matched random control) "
              "are registered. " if J["control_leg_frozen"] else
              "**The learned-vs-control leg is not in this frozen report**; it was computed from the raw records at "
              f"family {fam} (see Log) and is NOT pre-registered, so every cell classification below rests on one "
              "registered leg and one unregistered leg. ") +
             "**Exploratory** (stamped `exploratory: true` in the JSON): the discordance p-values (Holm over the "
             f"{J['discordance_holm_family']} two-arm registered contrasts read), the family-"
             f"{CONTROL_SENSITIVITY_FAMILY} control-leg sensitivity, the unadjusted family-1 orientation reading, and the "
             "replication and power tables.\n")

    # ---- 1 pre-registered
    L.append("## 1. Pre-registered analysis (all registered contrasts)\n")
    rc = J["reproduction_check"]
    replay_txt = (f"The analyzer-stream replay (one RNG stream, one index matrix per task) matches the frozen endpoints to "
                  f"{rc['replay_max_bound_deviation_pp']:.1e} pp." if rc["replay_matches"] else
                  f"The analyzer-stream replay deviates by up to {rc['replay_max_bound_deviation_pp']:.2f} pp, so the frozen "
                  "report was not produced by the documented single-stream estimator (or n differs); logged, not fatal.")
    if rc["passed"]:
        L.append(f"Reproduction from the raw records agrees with the frozen report: all {n_reg} point estimates match to "
                 f"{REPRO_DIFF_TOL_PP:g} pp and the largest `common.paired_bootstrap` endpoint deviation is "
                 f"{rc['max_bound_deviation_pp']:.2f} pp (tolerance {REPRO_BOUND_TOL_PP} pp for percentile endpoints at "
                 f"alpha/{fam} with a fresh RNG per call). {replay_txt} " +
                 ("Every endpoint agrees with the frozen report on its sign relative to zero.\n" if rc["all_signs_agree"] else
                  f"**{rc['sign_disagreements']} endpoint(s) differ in sign relative to zero** "
                  f"({', '.join(rc['disagreeing_contrasts'])}); the frozen report is authoritative for the classification "
                  "below.\n"))
    else:
        L.append(f"**Reproduction check FAILED**: {rc['failure']}. {replay_txt} The frozen report is still used below, "
                 "but the discrepancy must be resolved before any number here is quoted.\n")
    for task in tasks:
        rows = [r for r in pre if r["task"] == task]
        if not rows:
            continue
        df = pd.DataFrame([dict(contrast=r["label"], gain_pp=r["difference"], lower=r["lower"], upper=r["upper"],
                                wins=r.get("wins", "-"), losses=r.get("losses", "-")) for r in rows])
        nat = J["native_success_percent"][task]
        L.append(f"**{C.TASK_LABEL[task]}** (unsteered success {nat:.1f}%; family {fam}):\n")
        L.append(C.md_table(df, "{:+.1f}") + "\n")
    excl = [r for r in pre if r["lower"] > 0 or r["upper"] < 0]
    if excl:
        L.append(f"Registered intervals excluding zero ({len(excl)} of {n_reg}): " + "; ".join(
            f"{C.TASK_LABEL[r['task']]} {r['label']} {pp(r['difference'])} pp {ci(r['lower'], r['upper'])}"
            for r in excl) + ".\n")
    else:
        L.append(f"None of the {n_reg} registered intervals excludes zero.\n")
    rnd = [r for r in pre if r["kind"] == "arm_vs_native" and r["arm"] in C.LEARNED.values()
           and (r["lower"] > 0 or r["upper"] < 0)]
    if rnd:
        L.append("A random control differs from unsteered on its own: " + "; ".join(
            f"{C.TASK_LABEL[r['task']]} {C.ARM_LABEL[r['arm']]} {pp(r['difference'])} pp" for r in rnd)
            + ". Generic perturbation of matched dose moves outcomes on those tasks, so learned-vs-unsteered gains there "
              "cannot be attributed to the fitted directions without the control contrast.\n")

    # ---- 2 learned vs control
    L.append("## 2. Learned arm vs its dose-matched random control (registered) with exploratory sensitivities\n")
    L.append(f"`gain`, `lower`, `upper`: the registered contrast (Bonferroni {fam}) that enters the classification. "
             f"`lower{CONTROL_SENSITIVITY_FAMILY}`/`upper{CONTROL_SENSITIVITY_FAMILY}`: exploratory sensitivity, Bonferroni "
             f"over the {CONTROL_SENSITIVITY_FAMILY} control contrasts only. `lower1`/`upper1`: exploratory, unadjusted. "
             f"`exact_p`: exact paired sign test on discordant scenarios, exploratory; `holm_p`: Holm over the "
             f"{J['discordance_holm_family']} two-arm registered contrasts read.\n")
    ctrl_rows = [r for r in pre if r["kind"] == "learned_vs_control"]
    sens = {(r["task"], r["arm"]): r for r in J["control_sensitivity_family8"]}
    df = pd.DataFrame([dict(task=C.TASK_LABEL[r["task"]], arm=C.ARM_LABEL[r["arm"]], gain_pp=r["difference"],
                            lower=r["lower"], upper=r["upper"],
                            **{f"lower{CONTROL_SENSITIVITY_FAMILY}": sens[(r["task"], r["arm"])]["lower"],
                               f"upper{CONTROL_SENSITIVITY_FAMILY}": sens[(r["task"], r["arm"])]["upper"]},
                            lower1=r["unadjusted_family1"]["lower"], upper1=r["unadjusted_family1"]["upper"],
                            wins=r["wins"], losses=r["losses"], exact_p=f"{r['exact_p']:.3f}",
                            holm_p=f"{r['exploratory_discordance']['holm_p']:.3f}") for r in ctrl_rows])
    L.append(C.md_table(df, "{:+.2f}") + "\n")
    above = [r for r in ctrl_rows if r["lower"] > 0]
    below = [r for r in ctrl_rows if r["upper"] < 0]
    if above:
        L.append("Registered control intervals entirely above zero (learned edit beats its dose-matched control): " +
                 "; ".join(f"{C.TASK_LABEL[r['task']]} {C.ARM_LABEL[r['arm']]} {pp(r['difference'])} pp "
                           f"{ci(r['lower'], r['upper'])}" for r in above) + ".\n")
    if below:
        L.append("Registered control intervals entirely below zero (learned edit does *worse* than its dose-matched "
                 "control): " + "; ".join(f"{C.TASK_LABEL[r['task']]} {C.ARM_LABEL[r['arm']]} {pp(r['difference'])} pp "
                                          f"{ci(r['lower'], r['upper'])}" for r in below) +
                 ". `classify_regime` has no category for this; it is recorded here and in the cell notes.\n")
    if not above and not below:
        L.append(f"No registered learned-vs-control interval excludes zero at Bonferroni {fam}.\n")
    flips = [c for c in J["cells"] if c["regime_control_family8_sensitivity"] != c["regime"]]
    if flips:
        L.append(f"*Sensitivity (exploratory):* with the control leg at Bonferroni {CONTROL_SENSITIVITY_FAMILY} instead of "
                 f"{fam}, the classification would change for " + "; ".join(
                     f"{C.TASK_LABEL[c['task']]} {C.ARM_LABEL[c['arm']]} ({REGIME_TEXT[c['regime']]} -> "
                     f"{REGIME_TEXT[c['regime_control_family8_sensitivity']]})" for c in flips) +
                 ". The registered reading stands; the narrower family is reported only to show how close the cell is.\n")
    else:
        L.append(f"*Sensitivity (exploratory):* no cell classification changes when the control leg is corrected over "
                 f"{CONTROL_SENSITIVITY_FAMILY} instead of {fam}.\n")
    sig = [r for r in ctrl_rows if r["exploratory_discordance"]["holm_p"] < 0.05]
    if sig:
        L.append("Holm-adjusted discordance below 0.05 among the control contrasts: " + "; ".join(
            f"{C.TASK_LABEL[r['task']]} {C.ARM_LABEL[r['arm']]} (p = {r['exploratory_discordance']['holm_p']:.3f})"
            for r in sig) + ".\n")
    elif ctrl_rows:
        L.append(f"Smallest Holm-adjusted discordance p over the {len(ctrl_rows)} control contrasts: "
                 f"{min(r['exploratory_discordance']['holm_p'] for r in ctrl_rows):.3f}.\n")

    # ---- 3 regime
    L.append("## 3. Regime per (task, learned arm) and overall\n")
    L.append(f"Cell rule (`common.classify_regime`): *confirmed* = both lower bounds > 0 and gain vs unsteered >= "
             f"{C.USEFUL_GAIN_PP:.0f} pp; *positive vs unsteered only*; *positive vs control only*; *harmful* = upper bound "
             f"vs unsteered < 0; otherwise *inconclusive*. Both intervals are the registered Bonferroni-{fam} intervals "
             "from the frozen report.\n")
    L.append(f"Overall rule: {OVERALL_RULE}\n")
    df = pd.DataFrame([dict(task=C.TASK_LABEL[c["task"]], arm=C.ARM_LABEL[c["arm"]],
                            vs_unsteered=f"{pp(c['vs_native']['difference'])} {ci(c['vs_native']['lower'], c['vs_native']['upper'])}",
                            vs_control=f"{pp(c['vs_control']['difference'])} {ci(c['vs_control']['lower'], c['vs_control']['upper'])}",
                            regime=REGIME_TEXT[c["regime"]]) for c in J["cells"]])
    L.append(C.md_table(df) + "\n")
    counts = J["regime_counts"]
    L.append(f"**Overall regime (pre-registered reading): `{ov}`** — "
             f"{counts['confirmed']} confirmed, {counts['positive_vs_native_only']} positive vs unsteered only, "
             f"{counts['positive_vs_control_only']} positive vs control only, {counts['harmful']} harmful, "
             f"{counts['inconclusive']} inconclusive of {n_cells} cells.\n")
    L.append(regime_paragraph(J))
    ovu = J["overall_regime_unadjusted"]
    dfu = pd.DataFrame([dict(task=C.TASK_LABEL[c["task"]], arm=C.ARM_LABEL[c["arm"]],
                             **{"unadjusted (exploratory, family 1)": REGIME_TEXT[c["regime_unadjusted"]]})
                        for c in J["cells"] if c["regime_unadjusted"] != c["regime"]])
    if ovu != ov:
        L.append(f"*Exploratory orientation only:* with no multiplicity correction (family 1 on both legs) the overall "
                 f"regime would read `{ovu}` ({J['regime_counts_unadjusted']['confirmed']} confirmed, "
                 f"{J['regime_counts_unadjusted']['harmful']} harmful). This is the regime the data *suggest*; it is not "
                 "the regime the pre-registered analysis *establishes*, and it must not be quoted as a result. Cells "
                 "that change:\n")
        L.append(C.md_table(dfu) + "\n")
    elif len(dfu):
        L.append("The unadjusted (family 1, exploratory) reading gives the same overall regime; cells whose label "
                 "changes without correction:\n")
        L.append(C.md_table(dfu) + "\n")
    else:
        L.append("The unadjusted (family 1, exploratory) reading changes no cell label and no overall regime, so the "
                 "conclusion does not hinge on the multiplicity correction.\n")

    # ---- 4 replication
    L.append("## 4. Replication against the development panels (exploratory)\n")
    L.append("Development scenarios were **exposed** (selection and iteration happened on them); fresh scenarios were "
             "**not**. Development intervals come from their own frozen families (core panel 8; refined navigation "
             f"transcription 12; navigation coupling report 32); fresh intervals from family {fam}. Families are not "
             "re-harmonised. Shrinkage = fresh / development point estimate, undefined when the development estimate is "
             "within 1 pp of zero.\n")
    rep = [r for r in J["replication"] if r.get("available")]
    miss = [r for r in J["replication"] if not r.get("available")]
    if rep:
        df = pd.DataFrame([dict(task=C.TASK_LABEL[r["task"]], arm=C.ARM_LABEL[r["arm"]],
                                vs="unsteered" if r["control"] == "native" else "control",
                                dev_pp=r["dev_difference"], fresh_pp=r["fresh_difference"], sign=r["sign"],
                                dev_in_fresh_CI=r["dev_point_inside_fresh_interval"],
                                fresh_in_dev_CI=r["fresh_point_inside_dev_interval"],
                                shrinkage=r["shrinkage_ratio"]) for r in rep])
        L.append(C.md_table(df, "{:+.2f}") + "\n")
        L.append(replication_paragraph(rep))
    else:
        L.append("No comparable development contrast was available for the tasks read.\n")
    if miss:
        L.append("No development estimate available for: " + "; ".join(
            f"{C.TASK_LABEL[r['task']]} {C.ARM_LABEL[r['arm']]} vs {r['control']}" for r in miss) + ".\n")
    if J["development_confirmed_cells"]:
        L.append("Development cells that met the frozen rule on exposed scenarios: " +
                 ", ".join(J["development_confirmed_cells"]) + ".\n")
    else:
        L.append("No development cell met the frozen rule (both lower bounds > 0 and gain >= "
                 f"{C.USEFUL_GAIN_PP:.0f} pp) on exposed scenarios.\n")

    # ---- 5 power
    L.append("## 5. Power and resolution (exploratory)\n")
    L.append("Approximate paired difference detectable at 80% power with the observed number of paired binary outcomes "
             "(normal approximation assuming maximal discordance, orientation only), against the pre-registered useful "
             f"effect of {C.USEFUL_GAIN_PP:.0f} pp. `observed_half_width_pp` is the mean half-width of the fresh "
             "arm-vs-unsteered intervals.\n")
    df = pd.DataFrame([dict(task=C.TASK_LABEL[r["task"]], n=r["n"], family=r["family"],
                            native_rate=f"{100 * r['native_rate']:.1f}%",
                            detectable_pp_at_native_rate=r["detectable_pp_80pct_at_native_rate"],
                            detectable_pp_at_50pct=r["detectable_pp_80pct_at_p50"],
                            observed_half_width_pp=J["resolution_pp"][r["task"]]) for r in J["power"]])
    L.append(C.md_table(df, "{:.1f}") + "\n")
    L.append(power_paragraph(J))

    # ---- 6 licensed claims
    L.append("## 6. Licensed claims (generated by rule from the numbers)\n")
    for c in J["cells"]:
        L.append(f"**{C.TASK_LABEL[c['task']]} / {C.ARM_LABEL[c['arm']]} — {REGIME_TEXT[c['regime']]}.**")
        for s in c["licensed"]:
            L.append(f"- Licensed: {s}")
        for s in c["not_licensed"]:
            L.append(f"- Not licensed: {s}")
        L.append("")
    L.append("**Panel-level.**")
    for s in J["panel_claims"]["licensed"]:
        L.append(f"- Licensed: {s}")
    for s in J["panel_claims"]["not_licensed"]:
        L.append(f"- Not licensed: {s}")
    L.append("")

    # ---- 7 physics-zone framing
    L.append("## 7. Relation to the 'physics emergence zone' framing\n")
    L.append(physics_paragraph(J))

    # ---- 8
    L.append("## 8. What this does and does not establish\n")
    L.append(establish_paragraph(J))
    L.append("\n## Figures\n")
    L.append(f"- `{SCRIPT}_forest.png`: every registered fresh contrast per task (blue, Bonferroni {fam}; gray band = "
             "exploratory unadjusted interval) with the development estimates as hollow markers on dotted lines.")
    L.append(f"- `{SCRIPT}_regimes.png`: the {len(tasks)} x 2 cell-regime grid.\n")
    L.append("## Log\n")
    L += [f"- {m}" for m in J["log"]]
    Path(path).write_text("\n".join(L) + "\n")


def regime_paragraph(J):
    ov, cells, pre = J["overall_regime"], J["cells"], J["preregistered_contrasts"]
    fam = J["preregistered_family"]
    conf = [c for c in cells if c["regime"] == "confirmed"]
    harm = [c for c in cells if c["regime"] == "harmful"]
    pn = [c for c in cells if c["regime"] == "positive_vs_native_only"]
    pc = [c for c in cells if c["regime"] == "positive_vs_control_only"]
    below_ctrl = [c for c in cells if c["vs_control"]["upper"] < 0]
    near_zero = [c for c in cells if c["regime"] != "confirmed" and abs(c["vs_native"]["lower"]) <= 1.0]
    excl_all = [r for r in pre if r["lower"] > 0 or r["upper"] < 0]
    excl_other = [r for r in excl_all if not
                  (r["kind"] in ("arm_vs_native", "learned_vs_control") and r["arm"] in C.LEARNED)]
    s = []
    if ov == "positive":
        s.append("Reading: the pre-registered criterion is met in " + names(conf) +
                 ". On those cells a learned edit beats both the unsteered planner and its dose-matched random control "
                 "on scenarios that were never used during development" +
                 ("; no development cell met the same rule, so this is the first behavioral result in the project that "
                  "survives the frozen selection." if not J["development_confirmed_cells"] else
                  "; the development panel had already met the rule on " + ", ".join(J["development_confirmed_cells"]) + "."))
        if pn or pc:
            s.append("Partial signal elsewhere (" + names(pn + pc) + ") does not meet the criterion.")
    elif ov == "negative":
        s.append("Reading: no cell is confirmed and the pre-registered interval is entirely below zero for " +
                 names(harm) + ". The learned edit is behaviorally active on those tasks, in the wrong direction; a "
                 "forecast improvement, where one exists, did not translate into better plans.")
    elif ov == "mixed":
        parts = []
        if conf:
            parts.append("confirmed on " + names(conf))
        if harm:
            parts.append("harmful on " + names(harm))
        if pn:
            parts.append("positive over unsteered but not over its random control on " + names(pn))
        if pc:
            parts.append("positive over its control but not over unsteered on " + names(pc))
        s.append("Reading: the cells do not all meet one rule — " + "; ".join(parts) + ".")
        signs = {np.sign(c["vs_native"]["difference"]) for c in conf + harm + pn + pc}
        if len(signs) > 1:
            s.append("The non-inconclusive cells differ in sign, so the effect is task-dependent: no statement about "
                     "'the edit' as such is licensed without naming the task.")
        elif pn and pc and not conf:
            s.append("Every non-inconclusive cell points the same way, but each clears only one registered leg: " +
                     names(pn) + " beats unsteered without beating its random control, and " + names(pc) +
                     " beats its random control without beating unsteered. Neither pattern is the frozen rule, and the "
                     "two do not combine into it across cells.")
        elif pc and not conf and not pn:
            s.append("Every non-inconclusive cell points the same way and the learned edit beats its dose-matched control "
                     "there, but the registered contrast against unsteered does not clear zero at Bonferroni "
                     f"{fam}; this is a consistent direction that falls short of the frozen rule, not a confirmed effect.")
        elif pn and not conf:
            s.append("The gain over unsteered is real where it appears but the registered control contrast is not, so the "
                     "improvement is not distinguishable from a generic perturbation of matched dose.")
    else:
        s.append("Reading: every learned-arm cell is inconclusive: each learned-arm interval vs unsteered includes zero "
                 "and neither registered learned-vs-control interval is positive, so the fresh panel neither confirms "
                 "nor refutes a success effect of either learned edit at the pre-registered resolution.")
        if excl_other:
            s.append("Registered intervals that do exclude zero are confined to arms outside the learned cells: " +
                     "; ".join(f"{C.TASK_LABEL[r['task']]} {r['label']} {pp(r['difference'])} pp "
                               f"{ci(r['lower'], r['upper'])}" for r in excl_other) +
                     ". Those are pathway-level or control-arm results and carry no learned-direction claim.")
        elif excl_all:
            s.append(f"The only registered intervals excluding zero at Bonferroni {fam} belong to the learned cells "
                     "themselves (" + "; ".join(f"{C.TASK_LABEL[r['task']]} {r['label']} {pp(r['difference'])} pp "
                                                f"{ci(r['lower'], r['upper'])}" for r in excl_all) +
                     "), none of them in the direction the rule rewards.")
        else:
            s.append(f"No registered interval on the panel excludes zero at Bonferroni {fam}.")
    if below_ctrl:
        s.append("No regime category: the learned edit is *below* its own dose-matched random control on " +
                 ", ".join(f"{C.TASK_LABEL[c['task']]} {C.ARM_LABEL[c['arm']]} ({pp(c['vs_control']['difference'])} pp "
                           f"{ci(c['vs_control']['lower'], c['vs_control']['upper'])})" for c in below_ctrl) +
                 "; whatever the fitted directions do there, a random direction of the same dose does it less harm.")
    if near_zero:
        s.append("Lower bounds within 1 pp of zero against unsteered (a classification that would flip with a handful "
                 "of scenarios): " + ", ".join(f"{C.TASK_LABEL[c['task']]} {C.ARM_LABEL[c['arm']]} "
                                              f"({ci(c['vs_native']['lower'], c['vs_native']['upper'])})"
                                              for c in near_zero) + ".")
    return " ".join(s) + "\n"


def replication_paragraph(rep):
    agree = [r for r in rep if r["sign"] == "agree"]
    disagree = [r for r in rep if r["sign"] == "disagree"]
    flat = [r for r in rep if r["sign"] in ("flat", "fresh_flat")]
    inside = [r for r in rep if r["dev_point_inside_fresh_interval"]]
    ratios = [r["shrinkage_ratio"] for r in rep if np.isfinite(r["shrinkage_ratio"])]
    name = lambda r: f"{C.TASK_LABEL[r['task']]} {C.ARM_LABEL[r['arm']]} vs {'unsteered' if r['control'] == 'native' else 'control'}"
    s = [f"Of {len(rep)} comparable contrasts, {len(agree)} agree in sign with development, {len(disagree)} disagree, "
         f"{len(flat)} have a flat estimate on one side; {len(inside)} of {len(rep)} development point estimates lie "
         "inside the fresh interval."]
    if disagree:
        s.append("Sign reversals: " + "; ".join(f"{name(r)} (dev {pp(r['dev_difference'])} -> fresh {pp(r['fresh_difference'])} pp)"
                                                for r in disagree) + ".")
    if ratios:
        med = float(np.median(ratios))
        wdev = float(np.mean([r["dev_upper"] - r["dev_lower"] for r in rep]))
        wfr = float(np.mean([r["fresh_upper"] - r["fresh_lower"] for r in rep]))
        s.append(f"Median fresh/development ratio over the {len(ratios)} contrasts with a non-trivial development "
                 f"estimate: {med:.2f}. Mean interval width: development {wdev:.1f} pp, fresh {wfr:.1f} pp.")
        if med < 0:
            s.append("A negative median ratio means the development effects, on average, reversed on untouched scenarios; "
                     "the exposed development pools should be read as selection, not as evidence.")
        elif med < 0.5:
            s.append("Ratios well below one are the expected signature of selection on exposed scenarios (winner's curse); "
                     "development point estimates should not be quoted as effect sizes.")
        elif med <= 1.5:
            s.append("Ratios near one mean the fresh panel reproduces the development magnitudes; " +
                     ("with development intervals no narrower than the fresh ones, " if wdev >= 0.8 * wfr else
                      "with development intervals narrower than the fresh ones, ") +
                     "this is consistency, not independent confirmation, unless the fresh cell itself is confirmed above.")
        else:
            s.append("Ratios above one mean the fresh effects are larger than the exposed development estimates; with "
                     "intervals this wide, that is as likely sampling variation as a real difference between cohorts.")
    else:
        s.append("Every development estimate is within 1 pp of zero, so no shrinkage ratio is defined.")
    outside = [r for r in rep if not r["dev_point_inside_fresh_interval"]]
    if outside:
        s.append("Development estimates outside the fresh interval: " + "; ".join(name(r) for r in outside) + ".")
    return " ".join(s) + "\n"


def power_paragraph(J):
    res = J["resolution_pp"]
    fam = J["preregistered_family"]
    worst, best = max(res.values()), min(res.values())
    pf = [r for r in J["power"] if r["family"] == fam]
    p1 = [r for r in J["power"] if r["family"] == 1]
    df_ = float(np.mean([r["detectable_pp_80pct_at_native_rate"] for r in pf]))
    d1 = float(np.mean([r["detectable_pp_80pct_at_native_rate"] for r in p1]))
    s = [f"At the pre-registered family the panel resolves differences of roughly {df_:.0f} pp (80% power, averaged over "
         f"the {len(pf)} task(s)) and the observed half-widths run {best:.1f} to {worst:.1f} pp; even uncorrected it "
         f"resolves about {d1:.0f} pp. The pre-registered useful effect of {C.USEFUL_GAIN_PP:.0f} pp is therefore "
         + ("below" if d1 > C.USEFUL_GAIN_PP else "within") + " the panel's resolution "
         + ("under any correction." if d1 > C.USEFUL_GAIN_PP else "without correction" +
            (" but not at the registered family." if df_ > C.USEFUL_GAIN_PP else " and at the registered family."))]
    if worst < df_:
        s.append("The observed half-widths are narrower than the normal approximation because most paired outcomes are "
                 "concordant (the approximation assumes maximal discordance).")
    if J["overall_regime"] in ("inconclusive", "mixed"):
        s.append("'Not detected' here means 'not resolved', not 'zero': a real effect at the useful-effect scale would "
                 + ("usually be missed by this design." if df_ > C.USEFUL_GAIN_PP else "usually be detected by this design, "
                    "so an inconclusive cell is informative about the size of any effect."))
    if J["overall_regime"] == "positive":
        s.append("A confirmed cell under this correction therefore reflects an effect " +
                 ("well above the useful-effect threshold, not a marginal one." if df_ > C.USEFUL_GAIN_PP else
                  "resolvable by design; its size is bounded by the reported interval."))
    return " ".join(s) + "\n"


def physics_paragraph(J):
    ov = J["overall_regime"]
    cells, pre = J["cells"], J["preregistered_contrasts"]
    conf = [c for c in cells if c["regime"] == "confirmed"]
    off = J["offline_forecast_context"]
    with_forecast = [k for k, v in off.items() if v and v.get("result") == "positive"]
    other_excl = [r for r in pre if (r["lower"] > 0 or r["upper"] < 0) and r["kind"] in ("pathway", "interaction")
                  or (r["kind"] == "arm_vs_native" and r["arm"] in C.COMPONENT and (r["lower"] > 0 or r["upper"] < 0))]
    s = ["Joseph et al. (arXiv 2602.07050) find that a physics variable can be linearly readable at a layer and still "
         "resist steering unless many directions are moved together; representational accessibility and behavioral "
         "leverage are separate questions. This panel is the behavioral-leverage test for a world model whose forecast "
         "we can already move."]
    if with_forecast:
        s.append("Offline forecast gains are on record for " + ", ".join(
            f"{C.TASK_LABEL[t.split('/')[0]]} {C.ARM_LABEL[t.split('/')[1]]} ({off[t]['effect']:.2f}% of native error)"
            for t in with_forecast) + ".")
    else:
        s.append("No positive offline forecast contrast is on record for the learned cells read here, so the "
                 "forecast -> behavior comparison has no forecast side on this panel.")
    if ov == "positive":
        s.append("With " + names(conf) + " confirmed, a four-direction edit at one site does have behavioral leverage on "
                 "that task; the accessible structure is, there, also a lever. That is the opposite of their "
                 "single-direction steering null and should be stated as task-specific, not as a general property of "
                 "the predictor.")
    elif ov == "negative":
        s.append("With harmful cells and no confirmed cell, the edit has leverage but the wrong sign: the accessible "
                 "forecast structure is coupled to behavior, and moving it along the fitted directions degrades the "
                 "planner. This is a stronger statement than their steering null, and it argues against reading a "
                 "forecast probe as a control knob.")
    elif ov == "mixed":
        active = [c for c in cells if c["regime"] != "inconclusive"]
        signs = {np.sign(c["vs_native"]["difference"]) for c in active}
        if len(signs) > 1:
            s.append("With active cells of opposite sign, leverage exists but is task-dependent; the tie to their framing "
                     "is that the same accessible structure can be a lever on one task and inert or harmful on another, "
                     "so a 'physics variable' in the forecast is not a task-general control variable.")
        else:
            s.append("The active cells share a sign but none clears the frozen rule against both references; the "
                     "accessible forecast structure shows partial leverage on " + names(active) +
                     ", short of the coordinated-steering standard their paper sets for a variable that is actually "
                     "under control.")
    else:
        s.append("With every learned cell inconclusive, the panel matches their pattern for low-dimensional steering: the "
                 "forecast structure is accessible and correctable, but the dose-matched fitted directions at one site "
                 "do not measurably change task success. Whether a coordinated many-direction edit would is untested "
                 "here.")
        if other_excl:
            s.append("The site is not inert, however: " + "; ".join(
                f"{C.TASK_LABEL[r['task']]} {r['label']} {pp(r['difference'])} pp {ci(r['lower'], r['upper'])}"
                for r in other_excl) + " exclude zero. The null is specific to the dose-matched learned edits, not to "
                "editing this site as such, which is closer to their 'moving many directions' regime than to inertness.")
    return " ".join(s) + "\n"


def establish_paragraph(J):
    ov = J["overall_regime"]
    counts = J["regime_counts"]
    rc = J["reproduction_check"]
    syn = J["provenance"]["synthetic"]
    tasks = J["tasks"]
    fam = J["preregistered_family"]
    task_txt = (f"any of the {len(tasks)} tasks read ({', '.join(C.TASK_LABEL[t] for t in tasks)})" if len(tasks) > 1
                else f"the one task read ({C.TASK_LABEL[tasks[0]]})")
    est, not_est = [], []
    if syn:
        est.append("Nothing about the model: the inputs are a synthetic fixture. What it establishes is that the "
                   "pipeline reproduces frozen contrasts, classifies cells and generates regime-dependent text end to end.")
    else:
        est.append(f"The {len(J['preregistered_contrasts'])} registered contrasts on untouched scenarios, " +
                   ("reproduced independently from the raw records." if rc["passed"] else
                    "with an unresolved reproduction discrepancy (see Section 1)."))
        if ov == "positive":
            est.append(f"{counts['confirmed']} cell(s) meet the frozen criterion: a direction-specific closed-loop "
                       "improvement on the named task(s) and released checkpoint.")
        elif ov == "negative":
            est.append(f"{counts['harmful']} cell(s) with a pre-registered interval below zero: the edit is behaviorally "
                       "active and harmful on the named task(s).")
        elif ov == "mixed":
            active = [c for c in J["cells"] if c["regime"] != "inconclusive"]
            if len({np.sign(c["vs_native"]["difference"]) for c in active}) > 1:
                est.append("A task-dependent pattern with no single sign; each cell's claim is listed in Section 6 and "
                           "none generalises across tasks.")
            else:
                est.append(f"{len(active)} cell(s) with partial signal in one direction that do not clear the frozen "
                           "rule; each cell's claim is listed in Section 6 and none is confirmatory.")
        else:
            dev_conf = J["development_confirmed_cells"]
            est.append(f"That no effect at the panel's resolution was detected for either learned edit on {task_txt}; " +
                       ("this reproduces the development panels' reading (no development cell met the rule either) on "
                        "scenarios that were never exposed." if not dev_conf else
                        "this does not reproduce the development panels, where " + ", ".join(dev_conf) +
                        " met the rule on exposed scenarios; the exposed result did not survive untouched scenarios."))
    not_est += [
        "Transfer to other tasks, checkpoints or training seeds (one released checkpoint per task was evaluated).",
        "Any mechanism: this script reads outcomes only; the decision-level and geometry analyses are separate and "
        "exploratory.",
        "Effects smaller than the resolution (Section 5): an interval that includes zero is not evidence of zero.",
        f"Anything from the family-{CONTROL_SENSITIVITY_FAMILY} control sensitivity, the unadjusted family-1 reading, the "
        "discordance p-values or the replication ratios beyond orientation: they are exploratory and were not part of "
        f"the frozen family of {fam}.",
    ]
    if ov == "positive":
        not_est.append("That the forecast improvement caused the behavioral improvement; the chain forecast -> decision "
                       "-> outcome is not tested here.")
    if counts["positive_vs_native_only"]:
        not_est.append("That a gain over unsteered reflects the fitted directions where the control contrast is not "
                       "positive; matched-dose random perturbation is a competing explanation.")
    if len(tasks) < len(C.TASKS):
        not_est.append(f"Anything about the {len(C.TASKS) - len(tasks)} task(s) not read in this run.")
    out = ["**Establishes:**"] + [f"- {s}" for s in est] + ["", "**Does not establish:**"] + [f"- {s}" for s in not_est]
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------------- main
def main():
    ap = C.standard_parser("First-look regime report for the fresh four-task eight-arm confirmation.")
    ap.add_argument("--fixture-regime", default=None, help="label only (synthetic fixtures); never used for conclusions")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(0)

    prov = C.provenance(a.results, a.freeze, a.analysis)
    protocol = C.load_freeze(a.freeze)
    report = C.load_analysis(a.analysis)
    an = report.get("analysis", {}) or protocol.get("analysis", {})
    family = int(an.get("family_size", 48))
    seed = int(an.get("bootstrap_seed", C.BOOTSTRAP_SEED))
    draws = int(an.get("bootstrap_draws", C.BOOTSTRAP_DRAWS))
    if family != 48:
        log(f"protocol family_size is {family}, not 48; using the protocol's value")
    if (seed, draws) != (C.BOOTSTRAP_SEED, C.BOOTSTRAP_DRAWS):
        log(f"protocol bootstrap seed/draws {seed}/{draws} differ from common defaults; replay uses the protocol's")
    contrasts = registered_contrasts(protocol, report)
    protocol_tasks = list(protocol.get("tasks", {}).keys()) or list(report["results"].keys())
    if len(contrasts) * len(protocol_tasks) != family:
        log(f"{len(contrasts)} registered contrasts x {len(protocol_tasks)} tasks != family_size {family}; "
            "the frozen family is used as reported")
    tasks = [t for t in a.tasks if t in report["results"]]
    for t in a.tasks:
        if t not in tasks:
            log(f"task {t} absent from frozen report; skipped")
    if not tasks:
        log("no requested task is present in the frozen report; nothing to do")
        sys.exit(2)

    log(f"loading episodes from {a.results}")
    df = C.episodes_frame(a.results, tasks)
    n_report = {t: int(report["results"][t].get("n", 96)) for t in report["results"]}
    replay = analyzer_stream_indices(seed, draws, n_report)
    pre, sens, n_per_task, S_by_task = [], [], {}, {}
    for task in tasks:
        S = C.success_matrix(df, task)
        S_by_task[task] = S
        n_per_task[task] = int(len(S))
        idx = replay[task] if len(S) == n_report.get(task, 96) else None
        if idx is None:
            log(f"{task}: {len(S)} complete scenarios in the tree vs n={n_report.get(task)} in the report; "
                "analyzer-stream replay skipped for this task")
        pre += preregistered(report, S, task, family, contrasts, idx)
        sens += control_sensitivity(S, task, CONTROL_SENSITIVITY_FAMILY)

    # reproduction check (frozen report is authoritative; this is the audit)
    max_diff = max(r["reproduction_abs_deviation_pp"]["difference"] for r in pre)
    max_bound = max(max(r["reproduction_abs_deviation_pp"]["lower"], r["reproduction_abs_deviation_pp"]["upper"]) for r in pre)
    replay_devs = [max(r["analyzer_replay"]["abs_deviation_pp"]["lower"], r["analyzer_replay"]["abs_deviation_pp"]["upper"])
                   for r in pre if np.isfinite(r["analyzer_replay"]["lower"])]
    replay_max = float(max(replay_devs)) if replay_devs else float("nan")
    signs = [(np.sign(r["lower"]) == np.sign(r["reproduced"]["lower"]) or r["lower"] == 0 or r["reproduced"]["lower"] == 0)
             and (np.sign(r["upper"]) == np.sign(r["reproduced"]["upper"]) or r["upper"] == 0 or r["reproduced"]["upper"] == 0)
             for r in pre]
    failure = None
    if max_diff > REPRO_DIFF_TOL_PP:
        failure = f"point estimates deviate by up to {max_diff:.3g} pp (tolerance {REPRO_DIFF_TOL_PP:g})"
    elif max_bound > REPRO_BOUND_TOL_PP:
        failure = f"interval endpoints deviate by up to {max_bound:.2f} pp (tolerance {REPRO_BOUND_TOL_PP})"
    repro = dict(passed=failure is None, failure=failure, n_contrasts=len(pre),
                 max_difference_deviation_pp=max_diff, max_bound_deviation_pp=max_bound,
                 bound_tolerance_pp=REPRO_BOUND_TOL_PP, difference_tolerance_pp=REPRO_DIFF_TOL_PP,
                 replay_max_bound_deviation_pp=replay_max, replay_tolerance_pp=REPLAY_TOL_PP,
                 replay_matches=bool(replay_devs) and replay_max <= REPLAY_TOL_PP,
                 replay_contrasts=len(replay_devs),
                 all_signs_agree=all(signs), sign_disagreements=int(len(signs) - sum(signs)), family=family,
                 disagreeing_contrasts=[r["contrast"] + "@" + r["task"] for r, ok in zip(pre, signs) if not ok])
    log(f"reproduction ({len(pre)} contrasts): max point deviation {max_diff:.2e} pp, max endpoint deviation "
        f"{max_bound:.2f} pp (common.paired_bootstrap), analyzer-stream replay max deviation {replay_max:.2e} pp, "
        f"passed={repro['passed']}, signs agree={repro['all_signs_agree']}")
    if not repro["passed"]:
        log(f"REPRODUCTION FAILED: {failure}; outputs are still written, exit status will be 1")
    if replay_devs and not repro["replay_matches"]:
        log("analyzer-stream replay does not match the frozen endpoints; the frozen report was not produced by the "
            "documented single-stream estimator (informational)")

    # exploratory discordance p-values, Holm over all two-arm registered contrasts read
    two_arm = [r for r in pre if r["arm"] is not None]
    hp = C.holm([r["exact_p"] for r in two_arm])
    for r, p in zip(two_arm, hp):
        r["exploratory_discordance"] = dict(exploratory=True, exact_p=r["exact_p"], holm_p=float(p), holm_family=len(two_arm))

    # cells
    nat_by = {(r["task"], r["arm"]): r for r in pre if r["kind"] == "arm_vs_native"}
    ctl_by = {(r["task"], r["arm"]): r for r in pre if r["kind"] == "learned_vs_control"}
    sens_by = {(r["task"], r["arm"]): r for r in sens}
    cells = []
    resolution = {task: float(np.mean([(r["upper"] - r["lower"]) / 2 for r in pre if r["task"] == task
                                       and r["kind"] == "arm_vs_native"])) for task in tasks}
    for task in tasks:
        half = resolution[task]
        for arm, ctrl in C.LEARNED.items():
            n = nat_by[(task, arm)]
            c = ctl_by.get((task, arm))
            frozen_ctrl = c is not None
            if c is None:
                b = C.paired_bootstrap(S_by_task[task][arm].values, S_by_task[task][ctrl].values, family=family)
                d = C.discordance(S_by_task[task][arm].values, S_by_task[task][ctrl].values)
                b1 = C.paired_bootstrap(S_by_task[task][arm].values, S_by_task[task][ctrl].values, family=1)
                c = dict(difference=b["difference"], lower=b["lower"], upper=b["upper"], family=family, wins=d["wins"],
                         losses=d["losses"], exploratory_discordance=dict(holm_p=float("nan")),
                         unadjusted_family1=dict(lower=b1["lower"], upper=b1["upper"]))
                log(f"{task} {arm}: learned-vs-control contrast not in the frozen report; computed from raw records at "
                    f"family {family} (NOT frozen) for the classification")
            vn = dict(difference=n["difference"], lower=n["lower"], upper=n["upper"], family=n["family"],
                      wins=n["wins"], losses=n["losses"], preregistered=True, contrast=n["contrast"])
            vc = dict(difference=c["difference"], lower=c["lower"], upper=c["upper"], family=c["family"],
                      wins=c["wins"], losses=c["losses"], preregistered=frozen_ctrl, frozen=frozen_ctrl,
                      contrast=c.get("contrast"), holm_p_exploratory=c["exploratory_discordance"]["holm_p"])
            reg = C.classify_regime(vn, vc)
            s8 = sens_by[(task, arm)]
            reg_s8 = C.classify_regime(vn, dict(difference=s8["difference"], lower=s8["lower"], upper=s8["upper"]))
            u_n, u_c = n["unadjusted_family1"], c["unadjusted_family1"]
            reg_u = C.classify_regime(dict(difference=n["difference"], lower=u_n["lower"], upper=u_n["upper"]),
                                      dict(difference=c["difference"], lower=u_c["lower"], upper=u_c["upper"]))
            cell = dict(task=task, arm=arm, control=ctrl, regime=reg, vs_native=vn, vs_control=vc,
                        regime_control_family8_sensitivity=reg_s8,
                        control_sensitivity_family8=dict(exploratory=True, family=CONTROL_SENSITIVITY_FAMILY,
                                                         lower=s8["lower"], upper=s8["upper"]),
                        regime_unadjusted=reg_u,
                        unadjusted=dict(exploratory=True, family=1,
                                        vs_native=dict(difference=n["difference"], lower=u_n["lower"], upper=u_n["upper"]),
                                        vs_control=dict(difference=c["difference"], lower=u_c["lower"], upper=u_c["upper"])),
                        useful_pp=C.USEFUL_GAIN_PP, resolution_half_width_pp=float(half))
            task_pre = {r["arm"]: r for r in pre if r["task"] == task and r["kind"] == "arm_vs_native"}
            cell["licensed"], cell["not_licensed"] = licensed_claims(cell, task_pre, float(half))
            cells.append(cell)
    ov = overall_regime(cells)
    ov_u = overall_regime([dict(regime=c["regime_unadjusted"]) for c in cells])
    counts = {k: sum(c["regime"] == k for c in cells) for k in REGIME_TEXT}
    counts_u = {k: sum(c["regime_unadjusted"] == k for c in cells) for k in REGIME_TEXT}
    log(f"overall regime (pre-registered reading): {ov}; unadjusted orientation (exploratory): {ov_u}")

    # replication, power, offline context
    dev = development_contrasts()
    rep = replication_rows(pre, dev, tasks)
    dev_confirmed = []
    for task in tasks:
        for arm, ctrl in C.LEARNED.items():
            dn, dc = dev.get((task, arm, "native")), dev.get((task, arm, ctrl))
            if dn and dc and C.classify_regime(dn, dc) == "confirmed":
                dev_confirmed.append(f"{C.TASK_LABEL[task]} {C.ARM_LABEL[arm]}")
    power = power_table(report, tasks, n_per_task, family)
    offline = {}
    for task in tasks:
        for arm in C.LEARNED:
            try:
                o = C.offline_effect(task, C.OFFLINE_ARM_NAME.get(arm, arm))
            except FileNotFoundError:
                o = None
            if o is None:
                log(f"no offline forecast contrast for {task} {arm} (BF16 proprio H6); context omitted")
            offline[f"{task}/{arm}"] = o

    # panel-level claims
    comp_excl = [r for r in pre if r["kind"] == "arm_vs_native" and r["arm"] in C.COMPONENT and (r["lower"] > 0 or r["upper"] < 0)]
    path_excl = [r for r in pre if r["kind"] in ("pathway", "interaction") and (r["lower"] > 0 or r["upper"] < 0)]
    rnd_excl = [r for r in pre if r["kind"] == "arm_vs_native" and r["arm"] in C.LEARNED.values() and (r["lower"] > 0 or r["upper"] < 0)]
    task_list = ", ".join(C.TASK_LABEL[t] for t in tasks)
    panel_lic, panel_not = [], []
    if ov == "inconclusive":
        panel_lic.append("The development-panel statement 'no learned edit improves task success over both its unsteered "
                         f"and its matched-random reference' is reproduced on untouched scenarios for {task_list}.")
    elif counts["confirmed"]:
        panel_lic.append("The development-panel statement 'no learned edit improves task success over both references' "
                         f"is overturned on {counts['confirmed']} cell(s); it stands on the other {len(cells) - counts['confirmed']}.")
    else:
        panel_lic.append("No cell meets the frozen selection rule, so the development-panel statement 'no learned edit "
                         f"improves task success over both references' stands on untouched scenarios for {task_list}.")
    gap = [k for k, v in offline.items() if v and v.get("result") == "positive"
           and next(c for c in cells if f"{c['task']}/{c['arm']}" == k)["regime"] != "confirmed"]
    if gap:
        panel_lic.append("The forecast -> behavior gap persists for " + ", ".join(
            f"{C.TASK_LABEL[k.split('/')[0]]} {C.ARM_LABEL[k.split('/')[1]]}" for k in gap) +
            ": a recorded offline forecast gain with no confirmed success gain.")
    if comp_excl:
        panel_lic.append("Registered component contrasts excluding zero: " + "; ".join(
            f"{C.TASK_LABEL[r['task']]} {C.ARM_LABEL[r['arm']]} {pp(r['difference'])} pp {ci(r['lower'], r['upper'])}"
            for r in comp_excl) + " (vs unsteered; pathway-level, no dose-matched control arm).")
    if path_excl:
        panel_lic.append("Registered pathway/interaction contrasts excluding zero: " + "; ".join(
            f"{C.TASK_LABEL[r['task']]} {r['label']} {pp(r['difference'])} pp {ci(r['lower'], r['upper'])}"
            for r in path_excl) + ".")
    if rnd_excl:
        panel_not.append("Attribution of any same-task learned-vs-unsteered gain to the fitted directions, because a "
                         "random control also moved: " + "; ".join(
                             f"{C.TASK_LABEL[r['task']]} {C.ARM_LABEL[r['arm']]} {pp(r['difference'])} pp" for r in rnd_excl) + ".")
    panel_not += ["Multi-seed generality (one released checkpoint per task).",
                  "A transferable recipe: each cell is a task-specific fit and a task-specific reading."]

    J = dict(
        script=SCRIPT, provenance=dict(**prov, fixture_regime_label=a.fixture_regime,
                                       fixture_declared_regime=report.get("regime") if prov["synthetic"] else None,
                                       note="fixture labels are recorded only; no conclusion uses them"),
        tasks=tasks, protocol_task_count=len(protocol_tasks),
        preregistered_family=family, registered_contrast_names=list(contrasts),
        registered_contrast_weights=contrasts,
        n_scenarios_per_task=n_per_task,
        native_success_percent={t: float(report["results"][t]["success_percent"]["native"]) for t in tasks},
        reproduction_check=repro,
        preregistered_contrasts=pre,
        discordance_holm_family=len(two_arm),
        control_sensitivity_family8=dict(exploratory=True, family=CONTROL_SENSITIVITY_FAMILY, rows=sens,
                                         note="sensitivity only; the registered family-48 control interval classifies"),
        cells=cells, control_leg_frozen=all(c["vs_control"]["frozen"] for c in cells),
        overall_regime=ov, overall_rule=OVERALL_RULE, regime_counts=counts,
        overall_regime_unadjusted=ov_u, regime_counts_unadjusted=counts_u,
        unadjusted_reading=dict(exploratory=True, family=1, note="orientation only; never a result"),
        replication=dict(exploratory=True, rows=rep, development_confirmed_cells=dev_confirmed,
                         note="development scenarios exposed during selection; fresh scenarios untouched"),
        development_confirmed_cells=dev_confirmed,
        power=dict(exploratory=True, rows=power, note="normal approximation, orientation only"),
        resolution_pp=resolution, useful_pp=C.USEFUL_GAIN_PP,
        offline_forecast_context=offline,
        panel_claims=dict(licensed=panel_lic, not_licensed=panel_not),
        log=LOG,
    )
    # markdown helpers expect flat lists
    Jmd = dict(J, replication=rep, power=power, control_sensitivity_family8=sens)
    C.write_json(out / f"{SCRIPT}.json", J)
    forest_figure(pre, dev, out / f"{SCRIPT}_forest.png", prov["synthetic"], tasks, n_per_task, family)
    regime_figure(cells, ov, out / f"{SCRIPT}_regimes.png", prov["synthetic"], tasks)
    write_markdown(out / f"{SCRIPT}.md", Jmd)
    print(f"[{SCRIPT}] wrote {out / (SCRIPT + '.json')}, .md, _forest.png, _regimes.png; overall regime: {ov}")
    if not repro["passed"]:
        print(f"[{SCRIPT}] reproduction of registered contrasts failed: {failure}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
