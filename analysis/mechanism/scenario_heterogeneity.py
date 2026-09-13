"""Scenario heterogeneity: WHICH scenarios does an edit rescue or break, and is that
structured (a mechanism) or fragility (noise)?

EXPLORATORY. Only the arm-vs-native contrasts in ANALYSIS/report.json are pre-registered.
Every quantity computed here is post hoc; each JSON block carries `exploratory: true` and
names the multiplicity family it was corrected over.

Reads finished artifacts only (see DATA_CONTRACT.md); never calls the model, the simulator
or a GPU. Deterministic (fixed seeds). Outputs:
  OUT/scenario_heterogeneity.json
  OUT/scenario_heterogeneity.md
  OUT/scenario_heterogeneity_flips.png          flip heatmap, one panel per task
  OUT/scenario_heterogeneity_concordance.png    Jaccard vs stratified permutation null
  OUT/scenario_heterogeneity_tertiles.png       net effect by native-distance tertile

Design notes
  * A rescue can only happen where native failed and a regression only where native
    succeeded. Features that are determined by the native outcome (native final distance,
    native reward) are therefore tautologically associated with a pooled signed outcome.
    The PRIMARY association design is within-stratum: among native failures, does the
    feature predict rescue; among native successes, does it predict regression. The pooled
    signed-outcome design requested by the spec is also reported, restricted to features
    that are not determined by the native outcome.
  * The concordance null permutes the control's flip labels within the same eligible
    stratum for the same reason.
  * "Near-miss" = native failed but ended close to the goal; the near-miss test is the
    rescue rate across native-distance tertiles computed AMONG NATIVE FAILURES.

Usage (from the repository root):
  .venv/bin/python analysis/mechanism/scenario_heterogeneity.py --results R --freeze F --analysis A --out OUT
"""
from __future__ import annotations

import hashlib
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu, norm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

SCRIPT = "scenario_heterogeneity"
LOG = logging.getLogger(SCRIPT)

PERM_DRAWS = 5000
PERM_SEED = 20260913
TERTILE_SEED = 20260914
MIN_FLIPS_FOR_CONCORDANCE = 5      # below this a Jaccard index is not informative
MIN_GROUP_FOR_TEST = 3             # smallest group a rank test is run on
MIN_TESTED_FOR_STRONG = 4          # concordance: a 'fragility' reading needs at least this many tested cells
ALPHA = 0.05

# arms whose flip structure is analysed: the two learned edits and their dose-matched controls
LEARNED_AND_CONTROLS = ("fixed_rank4", "matched_random_fixed_rank4", "coupling_only", "matched_random_coupling")
FLIP_ARMS = tuple(a for a in C.ARMS if a != "native")
REFINED = ("fixed_rank4", "matched_random_fixed_rank4")

# scenario-level features (shared by every arm) and arm-level features (delivered edit / planner trace)
SCENARIO_FEATURES = {
    "native_distance": "native final distance to goal (task units)",
    "native_reward": "native cumulative simulator reward",
    "expert_success": "expert rollout that defined the goal succeeded (MetaWorld; constant on navigation)",
    "native_planning_seconds": "native wall-clock planning seconds (sum over planning calls)",
    "logical_rank": "logical rank of the scenario in the frozen protocol (0..7)",
    "initial_goal_distance": "||goal_state - initial_state|| over the full state vector (PROXY: MetaWorld state "
                             "vectors include non-positional dimensions, navigation states include velocities)",
}
ARM_FEATURES = {
    "coef_norm_mean": "mean delivered coefficient norm over H6 population calls (refined arms only)",
    "active_fraction": "fraction of candidates with an active edit (refined arms only)",
    "first_divergence": "first planning call at which the arm's chosen actions differ from native "
                        "(diverged scenarios only; a never-diverging scenario cannot flip, so it is excluded)",
    "n_divergent_calls": "number of planning calls whose chosen actions differ from native (DESCRIPTIVE ONLY: once an "
                         "arm diverges it stays diverged, so this equals n_calls - first_divergence and is not tested)",
}
DIVERGENCE_FEATURES = ("first_divergence", "n_divergent_calls")
DESCRIPTIVE_ONLY_FEATURES = ("n_divergent_calls",)   # excluded from every test family (duplicate of first_divergence)
NATIVE_OUTCOME_FEATURES = ("native_distance", "native_reward")   # determined by the native outcome
STRATA = (("native_failures", "rescue"), ("native_successes", "regression"))

# palette (validated with the dataviz validator): rescue blue, regression red, unchanged neutral
COL_RESCUE, COL_REGRESS, COL_UNCH = "#2a78d6", "#e34948", "#e6e5e0"
ARM_COLOURS = {"fixed_rank4": "#2a78d6", "matched_random_fixed_rank4": "#eb6834",
               "coupling_only": "#1baf7a", "matched_random_coupling": "#4a3aa7"}
TEXT_MUTED = "#52514e"
TL = lambda t: C.TASK_LABEL.get(t, t)  # noqa: E731
AL = lambda a: C.ARM_LABEL.get(a, a)   # noqa: E731


# ----------------------------------------------------------------------------- helpers
def _finite(x):
    try:
        return float(x) if x is not None and np.isfinite(x) else None
    except TypeError:
        return None


def jaccard(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    union = int((a | b).sum())
    if union == 0:
        return np.nan
    return float((a & b).sum() / union)


def flip_labels(arm_success, native_success):
    """+1 rescue (native failed, arm succeeded), -1 regression, 0 unchanged."""
    a, n = np.asarray(arm_success, bool), np.asarray(native_success, bool)
    return (a & ~n).astype(int) - (~a & n).astype(int)


def cell_seed(task, arm, kind):
    """Per-cell seed so a cell's permutation null does not depend on which other tasks/cells are present."""
    t_idx = C.TASKS.index(task) if task in C.TASKS else 100 + int(hashlib.sha256(task.encode()).hexdigest()[:6], 16) % 1000
    a_idx = LEARNED_AND_CONTROLS.index(arm) if arm in LEARNED_AND_CONTROLS else 50
    k_idx = ("rescued", "regressed", "any_flip").index(kind)
    return [int(PERM_SEED), t_idx, a_idx, k_idx]


def permutation_jaccard(learned, control, eligible, draws, seed):
    """Jaccard of two flip sets against a null that permutes the CONTROL's flip labels.

    `stratified` permutes only within the eligible stratum (native failures for rescues,
    native successes for regressions); `naive` permutes across all scenarios of the task.
    `seed` is a per-cell seed sequence (see cell_seed), so each cell's null is invariant to task order.
    """
    rng = np.random.default_rng(seed)
    learned, control, eligible = np.asarray(learned, bool), np.asarray(control, bool), np.asarray(eligible, bool)
    obs = jaccard(learned, control)
    out = dict(observed=_finite(obs), n_learned=int(learned.sum()), n_control=int(control.sum()),
               n_overlap=int((learned & control).sum()), n_eligible=int(eligible.sum()), draws=int(draws),
               seed=[int(v) for v in seed])
    if not np.isfinite(obs) or min(learned.sum(), control.sum()) < MIN_FLIPS_FOR_CONCORDANCE:
        out.update(status="too_few_flips", stratified=None, naive=None)
        return out
    for name, mask in (("stratified", eligible), ("naive", np.ones_like(eligible))):
        idx = np.where(mask)[0]
        base = control[idx]
        null = np.empty(draws)
        for d in range(draws):
            perm = np.zeros_like(control)
            perm[idx] = base[rng.permutation(len(idx))]
            null[d] = jaccard(learned, perm)
        null = np.nan_to_num(null, nan=0.0)
        p_high = float((np.sum(null >= obs) + 1) / (draws + 1))
        p_low = float((np.sum(null <= obs) + 1) / (draws + 1))
        sd = float(null.std())
        out[name] = dict(null_mean=float(null.mean()), null_q025=float(np.quantile(null, .025)),
                         null_q975=float(np.quantile(null, .975)), p_above_null=p_high, p_below_null=p_low,
                         z=float((obs - null.mean()) / sd) if sd > 0 else None)
    out["status"] = "tested"
    return out


def min_detectable_rho(n, alpha, power=0.8):
    """Fisher-z approximation to the smallest |Spearman rho| detectable at `power`."""
    if n is None or n <= 3:
        return None
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return float(np.tanh(z / np.sqrt(n - 3)))


def stouffer(rhos, ns):
    """Combine per-task Spearman rhos: z_i = atanh(rho_i)*sqrt(n_i-3), weights sqrt(n_i-3)."""
    rhos, ns = np.asarray(rhos, float), np.asarray(ns, float)
    ok = np.isfinite(rhos) & (ns > 3)
    if ok.sum() == 0:
        return None, None
    w = np.sqrt(ns[ok] - 3)
    zi = np.arctanh(np.clip(rhos[ok], -0.999999, 0.999999)) * w
    z = float(np.sum(w * zi) / np.sqrt(np.sum(w ** 2)))
    return z, float(2 * norm.sf(abs(z)))


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None, None)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(p), float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def bootstrap_diff_of_means(a, b, draws, seed, family=1):
    """Independent-groups percentile bootstrap of mean(a) - mean(b), in pp."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return dict(difference=None, lower=None, upper=None, n_a=int(len(a)), n_b=int(len(b)), family=family)
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(a), size=(draws, len(a)))
    ib = rng.integers(0, len(b), size=(draws, len(b)))
    d = a[ia].mean(1) - b[ib].mean(1)
    q = ALPHA / family
    lo, hi = np.quantile(d, [q / 2, 1 - q / 2])
    return dict(difference=float((a.mean() - b.mean()) * 100), lower=float(lo * 100), upper=float(hi * 100),
                n_a=int(len(a)), n_b=int(len(b)), family=int(family), draws=int(draws), seed=int(seed))


def tertiles_of(values):
    """0/1/2 by rank (stable), 0 = smallest; -1 for non-finite values (never silently binned)."""
    values = np.asarray(values, float)
    t = np.full(len(values), -1, int)
    ok = np.where(np.isfinite(values))[0]
    if len(ok) == 0:
        return t
    order = ok[np.argsort(values[ok], kind="stable")]
    t[order] = np.floor(np.arange(len(ok)) * 3 / len(ok)).astype(int)
    return t


def fmt(x, nd=2):
    x = _finite(x)
    return "n/a" if x is None else f"{x:.{nd}f}"


def fmt_p(p):
    p = _finite(p)
    if p is None:
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def holm_col(tab, col):
    """Holm-adjust one p-value column in place (over its finite entries); returns family size."""
    out = col + "_holm"
    tab[out] = np.nan
    if not len(tab):
        return 0
    ok = tab[col].notna()
    if ok.sum():
        tab.loc[ok, out] = C.holm(tab.loc[ok, col].values.astype(float))
    return int(ok.sum())


# ----------------------------------------------------------------------------- loading
def scenario_feature_frame(protocol, df, task):
    """One row per scenario with the shared (arm-independent) features."""
    rows = C.scenario_rows(protocol, task)
    nat = df[(df.task == task) & (df.arm == "native")].set_index("episode")
    out = []
    for r in rows:
        ep = int(r["episode"])
        if ep not in nat.index:
            continue
        init, goal = np.asarray(r.get("initial_state", []), float), np.asarray(r.get("goal_state", []), float)
        igd = float(np.linalg.norm(goal - init)) if init.size and init.shape == goal.shape else np.nan
        n = nat.loc[ep]
        out.append(dict(episode=ep, native_success=bool(n.success), native_distance=float(n.distance),
                        native_reward=float(n.reward), expert_success=float(n.expert_success),
                        native_planning_seconds=float(n.planning_seconds), logical_rank=int(r.get("logical_rank", -1)),
                        initial_goal_distance=igd, state_dim=int(init.size)))
    return pd.DataFrame(out).set_index("episode").sort_index()


def arm_feature_frame(results, task, arm, ef, notes):
    """Arm-specific per-scenario features: delivered edit (refined arms) and action divergence."""
    frames = []
    if arm in REFINED:
        if len(ef) and "coef_norm_mean" in ef.columns:
            sub = ef[(ef.task == task) & (ef.arm == arm)]
            if len(sub):
                g = sub.groupby("episode").agg(coef_norm_mean=("coef_norm_mean", "mean"),
                                               active_fraction=("active_fraction", "mean"),
                                               n_h6_calls=("call_index", "size"))
                frames.append(g)
            else:
                notes.append(f"{task}/{arm}: no H6 population calls with coefficients in energy_frame; "
                             "coef_norm_mean/active_fraction skipped")
        else:
            notes.append(f"{task}/{arm}: energy_frame has no refined-arm coefficient columns; edit features skipped")
    try:
        div = C.action_divergence(results, task, arm)
    except Exception as exc:  # pragma: no cover - defensive
        notes.append(f"{task}/{arm}: action_divergence failed ({exc}); divergence features skipped")
        div = pd.DataFrame()
    if len(div):
        d = div.set_index("episode")[["n_calls", "first_divergence", "n_divergent_calls"]].copy()
        d["diverged"] = d.n_divergent_calls > 0
        frames.append(d)
    else:
        notes.append(f"{task}/{arm}: no action traces; divergence features skipped")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()


# ----------------------------------------------------------------------------- regime
def regime_table(report, df, tasks):
    """Pre-registered arm-vs-native reading per task + exploratory learned-vs-control bootstrap."""
    cells, notes = [], []
    pairs = [(t, a) for t in tasks for a in C.LEARNED]
    for task, arm in pairs:
        res = report.get("results", {}).get(task)
        if res is None:
            notes.append(f"{task}: no pre-registered results in analysis report")
            continue
        ctr = res.get("contrasts", {}).get(f"{arm}-native")
        if ctr is None:
            notes.append(f"{task}/{arm}: pre-registered contrast '{arm}-native' absent from analysis report; regime cell skipped")
            continue
        vs_native = dict(difference=float(ctr["difference_pp"]), lower=float(ctr["simultaneous_95_interval_pp"][0]),
                         upper=float(ctr["simultaneous_95_interval_pp"][1]), source="pre-registered (analysis/report.json)")
        m = C.success_matrix(df, task)
        control = C.LEARNED[arm]
        if arm in m.columns and control in m.columns and len(m):
            b = C.paired_bootstrap(m[arm].values, m[control].values, family=len(pairs))
            vs_control = dict(difference=b["difference"], lower=b["lower"], upper=b["upper"],
                              source=f"EXPLORATORY paired bootstrap, Bonferroni family={len(pairs)}")
        else:
            vs_control = dict(difference=None, lower=None, upper=None, source="missing")
            notes.append(f"{task}/{arm}: control arm missing; learned-vs-control skipped")
        label = C.classify_regime(vs_native, vs_control) if vs_control["lower"] is not None else "inconclusive"
        cells.append(dict(task=task, arm=arm, control=control, vs_native=vs_native, vs_control=vs_control, regime=label))
    labels = [c["regime"] for c in cells]
    pos = sum(l in ("confirmed", "positive_vs_native_only") for l in labels)
    neg = sum(l == "harmful" for l in labels)
    overall = ("positive" if pos and not neg else "negative" if neg and not pos else
               "mixed" if pos and neg else "inconclusive")
    return dict(overall=overall, cells=cells, n_positive_cells=pos, n_harmful_cells=neg, notes=notes)


# ----------------------------------------------------------------------------- analysis
def analyse_task(task, protocol, df, ef, results, notes):
    m = C.success_matrix(df, task)
    if not len(m) or "native" not in m.columns:
        notes.append(f"{task}: no complete scenarios; task skipped")
        return None
    n_dirs = len(list((Path(results) / task).glob("scenario-*")))
    n_proto = len(C.scenario_rows(protocol, task)) if task in protocol.get("tasks", {}) else 0
    n_any = int(df[df.task == task].episode.nunique())
    if len(m) < max(n_dirs, n_proto, n_any):
        notes.append(f"{task}: {max(n_dirs, n_proto, n_any) - len(m)} scenario(s) incomplete in >=1 arm "
                     f"(scenario dirs={n_dirs}, protocol rows={n_proto}, with any arm file={n_any}, complete={len(m)}); "
                     "dropped from ALL arms by the complete-case success matrix")
    native = m["native"].values
    scen = scenario_feature_frame(protocol, df, task).reindex(m.index)
    flips = {}
    for arm in FLIP_ARMS:
        if arm not in m.columns:
            notes.append(f"{task}/{arm}: arm missing from success matrix")
            continue
        flips[arm] = flip_labels(m[arm].values, native)
    census = {}
    for arm, f in flips.items():
        census[arm] = dict(rescued=int((f == 1).sum()), regressed=int((f == -1).sum()), unchanged=int((f == 0).sum()),
                           net=int((f == 1).sum() - (f == -1).sum()), n=int(len(f)),
                           rescued_episodes=[int(e) for e in m.index[f == 1]],
                           regressed_episodes=[int(e) for e in m.index[f == -1]])
    concordance = {}
    for arm, control in C.LEARNED.items():
        if arm not in flips or control not in flips:
            continue
        fl, fc = flips[arm], flips[control]
        concordance[arm] = dict(control=control,
                                rescued=permutation_jaccard(fl == 1, fc == 1, ~native, PERM_DRAWS, cell_seed(task, arm, "rescued")),
                                regressed=permutation_jaccard(fl == -1, fc == -1, native, PERM_DRAWS, cell_seed(task, arm, "regressed")),
                                any_flip=permutation_jaccard(fl != 0, fc != 0, np.ones_like(native), PERM_DRAWS, cell_seed(task, arm, "any_flip")))
    arm_feats = {arm: arm_feature_frame(results, task, arm, ef, notes) for arm in LEARNED_AND_CONTROLS if arm in flips}
    return dict(matrix=m, native=native, scen=scen, flips=flips, census=census, concordance=concordance, arm_feats=arm_feats)


def _feature_matrix(T, arm):
    feats = T["scen"].copy()
    af = T["arm_feats"].get(arm, pd.DataFrame())
    if len(af):
        feats = feats.join(af, how="left")
    return feats


def association_tests(per_task, notes):
    """Primary: within-stratum tests. Secondary (spec): pooled signed-outcome tests."""
    strat_rows, pooled_rows = [], []
    for task, T in per_task.items():
        native = T["native"]
        for arm in LEARNED_AND_CONTROLS:
            if arm not in T["flips"]:
                continue
            f = T["flips"][arm]
            feats = _feature_matrix(T, arm)
            for feat in list(SCENARIO_FEATURES) + [f for f in ARM_FEATURES if f not in DESCRIPTIVE_ONLY_FEATURES]:
                if feat not in feats.columns:
                    continue
                x = feats[feat].values.astype(float)
                mask = np.isfinite(x)
                if feat in DIVERGENCE_FEATURES and "diverged" in feats.columns:
                    mask &= feats["diverged"].fillna(False).values.astype(bool)
                # --- primary: within stratum
                for stratum, flip_kind in STRATA:
                    elig = (~native) if stratum == "native_failures" else native
                    sel = mask & elig
                    y = (f[sel] == (1 if flip_kind == "rescue" else -1)).astype(int)
                    k, n = int(y.sum()), int(sel.sum())
                    reason = (f"too few {flip_kind}s ({k} of {n})" if k < MIN_GROUP_FOR_TEST else
                              f"too few non-{flip_kind} scenarios ({n - k} of {n})" if n - k < MIN_GROUP_FOR_TEST or n < 2 * MIN_GROUP_FOR_TEST else
                              "constant feature" if n and np.std(x[sel]) == 0 else None)
                    if reason:
                        notes.append(f"skip: {task}/{arm}/{feat}/{stratum}: {reason}")
                        continue
                    xs = x[sel]
                    sp = C.spearman(xs, y)
                    mw = mannwhitneyu(xs[y == 1], xs[y == 0], alternative="two-sided").pvalue
                    strat_rows.append(dict(task=task, arm=arm, feature=feat, stratum=stratum, outcome=flip_kind, n=n,
                                           n_flipped=k, median_flipped=_finite(np.median(xs[y == 1])),
                                           median_not_flipped=_finite(np.median(xs[y == 0])),
                                           spearman_rho=_finite(sp["rho"]), spearman_p=_finite(sp["p"]), mwu_p=_finite(mw)))
                # --- secondary: pooled signed outcome (spec), excluding native-outcome-determined features
                if feat in NATIVE_OUTCOME_FEATURES:
                    continue
                if mask.sum() < 3 * MIN_GROUP_FOR_TEST or np.std(x[mask]) == 0:
                    continue
                xs, fs = x[mask], f[mask]
                if (fs != 0).sum() < MIN_GROUP_FOR_TEST:
                    continue
                sp = C.spearman(xs, fs)
                groups = {"rescued": xs[fs == 1], "regressed": xs[fs == -1], "unchanged": xs[fs == 0]}
                row = dict(task=task, arm=arm, feature=feat, n=int(mask.sum()),
                           n_rescued=len(groups["rescued"]), n_regressed=len(groups["regressed"]), n_unchanged=len(groups["unchanged"]),
                           spearman_rho=_finite(sp["rho"]), spearman_p=_finite(sp["p"]),
                           kruskal_p=_finite(kruskal(*groups.values()).pvalue) if all(len(g) >= MIN_GROUP_FOR_TEST for g in groups.values()) else None)
                for a, b in (("rescued", "regressed"), ("rescued", "unchanged"), ("regressed", "unchanged")):
                    ok = len(groups[a]) >= MIN_GROUP_FOR_TEST and len(groups[b]) >= MIN_GROUP_FOR_TEST
                    row[f"mwu_{a}_vs_{b}_p"] = _finite(mannwhitneyu(groups[a], groups[b], alternative="two-sided").pvalue) if ok else None
                pooled_rows.append(row)
    strat, pooled = pd.DataFrame(strat_rows), pd.DataFrame(pooled_rows)
    fam = dict(stratified_spearman=holm_col(strat, "spearman_p"), stratified_mwu=holm_col(strat, "mwu_p"),
               pooled_spearman=holm_col(pooled, "spearman_p"))
    # pooled MWU: one Holm family over all three pairwise comparisons
    mwu_cols = [c for c in pooled.columns if c.startswith("mwu_") and c.endswith("_p")]
    entries = [(i, c) for c in mwu_cols for i in pooled.index if _finite(pooled.at[i, c]) is not None]
    for c in mwu_cols:
        pooled[c + "_holm"] = np.nan
    if entries:
        adj = C.holm([pooled.at[i, c] for i, c in entries])
        for (i, c), v in zip(entries, adj):
            pooled.at[i, c + "_holm"] = float(v)
    fam["pooled_mwu"] = len(entries)
    return strat, pooled, fam


def tertile_analysis(per_task, notes):
    net_rows, miss_rows, contrast_rows = [], [], []
    cells = [(t, a) for t, T in per_task.items() for a in LEARNED_AND_CONTROLS if a in T["flips"]]
    family_net = 3 * len(cells)
    for task, T in per_task.items():
        d = T["scen"]["native_distance"].values
        native = T["native"]
        n_nan = int((~np.isfinite(d)).sum())
        if n_nan:
            notes.append(f"{task}: {n_nan} scenario(s) with non-finite native_distance excluded from every tertile")
        tert_all = tertiles_of(d)
        for arm in LEARNED_AND_CONTROLS:
            if arm not in T["flips"]:
                continue
            arm_s, f = T["matrix"][arm].values, T["flips"][arm]
            for k in range(3):
                sel = tert_all == k
                b = C.paired_bootstrap(arm_s[sel], native[sel], family=family_net, seed=C.BOOTSTRAP_SEED + k)
                net_rows.append(dict(task=task, arm=arm, tertile=k + 1, n=int(sel.sum()), distance_max=float(d[sel].max()),
                                     native_success_pct=float(native[sel].mean() * 100), net_pp=b["difference"],
                                     lower_pp=b["lower"], upper_pp=b["upper"], rescued=int((f[sel] == 1).sum()),
                                     regressed=int((f[sel] == -1).sum())))
            # The all-scenario net effect by tertile is DESCRIPTIVE ONLY: the near tertile is mostly native successes
            # (can only regress) and the far tertile mostly native failures (can only be rescued), so any all-scenario
            # near-vs-far contrast is determined by composition. The test is done WITHIN the stratum below.
            row = dict(task=task, arm=arm)
            # near-miss: tertiles of native distance AMONG NATIVE FAILURES (rescue rate); symmetric for successes
            for stratum, flip_kind in STRATA:
                elig = (~native) if stratum == "native_failures" else native
                idx = np.where(elig & np.isfinite(d))[0]
                if len(idx) < 3 * MIN_GROUP_FOR_TEST:
                    notes.append(f"skip: {task}/{arm}/{stratum}: only {len(idx)} scenarios; distance tertiles not formed")
                    row[f"{flip_kind}_rate_chi2_p"] = None
                    row.update({f"{flip_kind}_near_minus_far_pp": None, f"{flip_kind}_near_minus_far_lower": None,
                                f"{flip_kind}_near_minus_far_upper": None})
                    continue
                t = tertiles_of(d[idx])
                flipped = (f[idx] == (1 if flip_kind == "rescue" else -1))
                c13 = bootstrap_diff_of_means(flipped[t == 0].astype(float), flipped[t == 2].astype(float),
                                              C.BOOTSTRAP_DRAWS, TERTILE_SEED, family=len(cells))
                row.update({f"{flip_kind}_near_minus_far_pp": c13["difference"], f"{flip_kind}_near_minus_far_lower": c13["lower"],
                            f"{flip_kind}_near_minus_far_upper": c13["upper"], f"{flip_kind}_n_near": c13["n_a"],
                            f"{flip_kind}_n_far": c13["n_b"]})
                table = []
                for k in range(3):
                    sel = t == k
                    kk, nn = int(flipped[sel].sum()), int(sel.sum())
                    r, lo, hi = wilson(kk, nn)
                    miss_rows.append(dict(task=task, arm=arm, stratum=stratum, outcome=flip_kind, tertile=k + 1, n=nn,
                                          distance_min=float(d[idx][sel].min()), distance_max=float(d[idx][sel].max()),
                                          flipped=kk, rate=r, rate_lo=lo, rate_hi=hi))
                    table.append([kk, nn - kk])
                table = np.array(table)
                ok = (table.sum(1) >= MIN_GROUP_FOR_TEST).all() and (table.sum(0) > 0).all()
                row[f"{flip_kind}_rate_chi2_p"] = _finite(chi2_contingency(table, correction=False)[1]) if ok else None
            contrast_rows.append(row)
    net, miss, ctab = pd.DataFrame(net_rows), pd.DataFrame(miss_rows), pd.DataFrame(contrast_rows)
    fam = dict(net=family_net, near_far=len(cells))
    for kind in ("rescue", "regression"):
        col = f"{kind}_rate_chi2_p"
        if col not in ctab.columns:
            ctab[col] = np.nan
        fam[f"{kind}_chi2"] = holm_col(ctab, col)
    return net, miss, ctab, fam


def cross_task_consistency(strat):
    rows = []
    if not len(strat):
        return pd.DataFrame(), 0
    for (arm, feat, stratum), g in strat.groupby(["arm", "feature", "stratum"]):
        rhos, ns = g.spearman_rho.values.astype(float), g.n.values.astype(float)
        z, p = stouffer(rhos, ns)
        ok = np.isfinite(rhos)
        signs = np.sign(rhos[ok])
        rows.append(dict(arm=arm, feature=feat, stratum=stratum, outcome=g.outcome.iloc[0], n_tasks=int(ok.sum()),
                         n_total=int(ns.sum()), rhos={t: _finite(r) for t, r in zip(g.task, rhos)},
                         n_positive=int((signs > 0).sum()), n_negative=int((signs < 0).sum()),
                         all_same_sign=bool(ok.sum() >= 2 and abs(signs.sum()) == ok.sum()),
                         stouffer_z=z, stouffer_p=p))
    tab = pd.DataFrame(rows)
    fam = holm_col(tab, "stouffer_p")
    return tab, fam


# ----------------------------------------------------------------------------- figures
def _style():
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#b0afa9", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b", "figure.dpi": 200,
                         "savefig.dpi": 200, "figure.facecolor": "white"})


def fig_flips(per_task, tasks, path):
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    _style()
    tasks = [t for t in tasks if t in per_task]
    if not tasks:
        return False
    fig, axes = plt.subplots(len(tasks), 1, figsize=(7.2, 1.15 * len(tasks) + 0.8), squeeze=False)
    cmap = ListedColormap([COL_REGRESS, COL_UNCH, COL_RESCUE])
    for ax, task in zip(axes[:, 0], tasks):
        T = per_task[task]
        order = np.argsort(T["scen"]["native_distance"].values, kind="stable")
        arms = [a for a in FLIP_ARMS if a in T["flips"]]
        mat = np.stack([T["flips"][a][order] for a in arms])
        ax.imshow(mat, cmap=cmap, vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels([AL(a) for a in arms])
        ax.set_xticks([])
        nat = T["native"][order]
        ax.scatter(np.where(nat)[0], np.full(nat.sum(), -0.75), marker="|", s=14, color="#0b0b0b", linewidths=0.6)
        ax.set_ylim(len(arms) - 0.5, -1.1)
        ax.set_title(f"{TL(task)}  (native success {nat.mean()*100:.0f}%; ticks above = native successes)", loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
    axes[-1, 0].set_xlabel("scenarios, sorted by native final distance (closest → farthest)")
    fig.legend(handles=[Patch(color=COL_RESCUE, label="rescue (native fail → arm success)"),
                        Patch(color=COL_REGRESS, label="regression (native success → arm fail)"),
                        Patch(color=COL_UNCH, label="unchanged")],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


def fig_concordance(per_task, tasks, path):
    _style()
    items = []
    for task in tasks:
        T = per_task.get(task)
        if not T:
            continue
        for arm, cc in T["concordance"].items():
            for kind in ("rescued", "regressed"):
                items.append((task, arm, kind, cc[kind]))
    if not items:
        return False
    fig, ax = plt.subplots(figsize=(max(7.2, 0.58 * len(items) + 1.5), 2.8))
    xs = np.arange(len(items))
    for x, (task, arm, kind, r) in zip(xs, items):
        col = COL_RESCUE if kind == "rescued" else COL_REGRESS
        obs = r["observed"] if r["observed"] is not None else 0.0
        ax.bar(x, obs, width=0.6, color=col, alpha=0.9 if r["status"] == "tested" else 0.35, linewidth=0)
        s = r.get("stratified")
        if s:
            ax.errorbar(x, s["null_mean"], yerr=[[max(s["null_mean"] - s["null_q025"], 0)], [max(s["null_q975"] - s["null_mean"], 0)]],
                        fmt="_", color="#0b0b0b", markersize=9, capsize=2, elinewidth=0.8)
        ax.text(x, max(obs, s["null_q975"] if s else 0) + 0.02,
                f"{r['n_learned']}/{r['n_control']}" if r["status"] == "tested" else "few", ha="center", va="bottom",
                fontsize=6, color=TEXT_MUTED)
    ax.set_xticks(xs)
    short = {"reach": "Reach", "reach-wall": "R-Wall", "pointmaze": "PMaze", "wall": "Wall"}
    ax.set_xticklabels([f"{short.get(t, TL(t))}\n{'refined' if a == 'fixed_rank4' else 'coupling'}\n{k}" for t, a, k, _ in items], fontsize=6)
    ax.set_ylabel("Jaccard(learned set, control set)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Flip-set concordance of each learned edit with its random control\n"
                 "bar = observed Jaccard; marker = stratified permutation null mean with 95% range; text = |learned|/|control|; "
                 "pale = too few flips", loc="left", fontsize=7)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


def fig_tertiles(net, miss, tasks, path):
    _style()
    tasks = [t for t in tasks if len(net) and t in set(net.task)]
    if not tasks:
        return False
    fig, axes = plt.subplots(2, len(tasks), figsize=(1.9 * len(tasks) + 1.3, 4.4), squeeze=False)
    arms = [a for a in LEARNED_AND_CONTROLS if a in set(net.arm)]
    for j_t, task in enumerate(tasks):
        ax = axes[0, j_t]
        for j, arm in enumerate(arms):
            g = net[(net.task == task) & (net.arm == arm)].sort_values("tertile")
            x = g.tertile.values + (j - (len(arms) - 1) / 2) * 0.16
            ax.errorbar(x, g.net_pp, yerr=[np.maximum(g.net_pp - g.lower_pp, 0), np.maximum(g.upper_pp - g.net_pp, 0)], fmt="o", ms=3.5,
                        color=ARM_COLOURS[arm], capsize=1.5, elinewidth=0.9, label=AL(arm))
        ax.axhline(0, color="#b0afa9", linewidth=0.8)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["near", "mid", "far"])
        ax.set_title(TL(task), loc="left")
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.6)
        ax.set_axisbelow(True)
        ax = axes[1, j_t]
        for j, arm in enumerate(arms):
            g = miss[(miss.task == task) & (miss.arm == arm) & (miss.stratum == "native_failures")].sort_values("tertile") if len(miss) else pd.DataFrame()
            g = g[g.rate.notna()] if len(g) else g
            if not len(g):
                continue
            x = g.tertile.values + (j - (len(arms) - 1) / 2) * 0.16
            r, lo, hi = (g.rate.values.astype(float) * 100, g.rate_lo.values.astype(float) * 100,
                         g.rate_hi.values.astype(float) * 100)
            ax.errorbar(x, r, yerr=[np.maximum(r - lo, 0), np.maximum(hi - r, 0)], fmt="s", ms=3.5, color=ARM_COLOURS[arm],
                        capsize=1.5, elinewidth=0.9)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["near-miss", "mid", "far"])
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0, 0].set_ylabel("net effect, all scenarios\narm − native success (pp)")
    axes[1, 0].set_ylabel("rescue rate among native failures\nrescued / native failures (%)")
    fig.supxlabel("native final-distance tertile (near = closest to goal)", fontsize=8, y=0.01)
    axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


# ----------------------------------------------------------------------------- narrative
def concordance_reading(r):
    """Uncorrected two-sided reading; Holm-corrected versions of BOTH tails are in p_above_null_holm / p_below_null_holm."""
    if r["status"] != "tested":
        return "too_few_flips"
    s = r["stratified"]
    if s["p_above_null"] < ALPHA:
        return "overlap_above_null"        # candidate scenario fragility (same scenarios flip under either perturbation)
    if s["p_below_null"] < ALPHA:
        return "overlap_below_null"        # anti-concordant: fewer shared flips than chance
    return "no_excess_overlap"             # direction-specific effects OR independent planner noise


def _sig_rows(records, col):
    return [r for r in records if _finite(r.get(col)) is not None and r[col] < ALPHA]


def build_markdown(J, tasks):
    md = []
    prov, reg = J["provenance"], J["regime"]
    md.append("# Scenario heterogeneity: which scenarios flip, and is it structured?\n")
    md.append("**EXPLORATORY.** Only the arm-vs-native contrasts in the frozen analysis report are pre-registered. "
              "Every per-scenario quantity below is post hoc; each table states the multiplicity family it was "
              "corrected over. Nothing here changes the primary endpoint's reading.\n")
    if prov["synthetic"]:
        md.append("> **Synthetic fixture.** These results were computed on a schema-identical synthetic tree "
                  f"(regime label: `{prov.get('fixture_regime') or J['analysis_report'].get('regime', 'unknown')}`), "
                  "not on the confirmation data. Numbers are for exercising the pipeline only.\n")
    md.append(f"Provenance: results `{prov['results']}`; protocol sha256 `{prov['freeze_sha256'][:12]}…`; "
              f"analysis sha256 `{prov['analysis_sha256'][:12]}…`; method `{prov['method']}`.\n")

    # ---- 0 regime
    md.append("## 0. Observed regime (pre-registered contrasts, for orientation)\n")
    rows = []
    for c in reg["cells"]:
        vc = c["vs_control"]
        rows.append({"task": TL(c["task"]), "learned arm": AL(c["arm"]),
                     "vs native pp [Bonferroni-48]": f"{c['vs_native']['difference']:+.1f} [{c['vs_native']['lower']:+.1f}, {c['vs_native']['upper']:+.1f}]",
                     "vs control pp [expl., family 8]": f"{vc['difference']:+.1f} [{vc['lower']:+.1f}, {vc['upper']:+.1f}]" if vc["lower"] is not None else "n/a",
                     "reading": c["regime"]})
    md.append(C.md_table(pd.DataFrame(rows)) if rows else "_no pre-registered cells found_")
    md.append("")
    ov = reg["overall"]
    if ov == "positive":
        md.append(f"Overall regime: **positive** ({reg['n_positive_cells']} learned-arm cell(s) beat native on the "
                  "pre-registered contrast and none is harmful). The heterogeneity question is therefore whether the "
                  "rescues are concentrated on identifiable scenarios (a mechanism) or spread like the random control's flips.\n")
    elif ov == "negative":
        md.append(f"Overall regime: **negative** ({reg['n_harmful_cells']} learned-arm cell(s) have an upper bound "
                  "below zero vs native). The heterogeneity question becomes whether the regressions are concentrated on "
                  "identifiable scenarios, and whether the random control breaks the same ones.\n")
    elif ov == "mixed":
        md.append(f"Overall regime: **mixed** ({reg['n_positive_cells']} positive and {reg['n_harmful_cells']} harmful "
                  "learned-arm cell(s)). Task-by-task readings below matter more than any pooled statement.\n")
    else:
        md.append("Overall regime: **inconclusive** (no learned-arm cell beats native or is harmful on the "
                  "pre-registered contrast). Flip structure is then the only place a behavioral signature could hide, "
                  "and everything below is a search, not a confirmation.\n")
    n_vc = sum(c["regime"] == "positive_vs_control_only" for c in reg["cells"])
    if n_vc:
        md.append(f"{n_vc} learned-arm cell(s) beat their dose-matched random control on the *exploratory* learned-vs-control "
                  "bootstrap without beating native on the pre-registered contrast; that comparison is not pre-registered and "
                  "does not change the regime.\n")

    # ---- 1 census
    md.append("## 1. Flip census per task and arm\n")
    md.append("A *rescue* is native failure → arm success on the same frozen scenario; a *regression* is the reverse.\n")
    rows = []
    for task in tasks:
        T = J["tasks"].get(task)
        if not T:
            continue
        for arm, c in T["census"].items():
            rows.append({"task": TL(task), "arm": AL(arm), "rescued": c["rescued"], "regressed": c["regressed"],
                         "unchanged": c["unchanged"], "net": c["net"], "n": c["n"]})
    md.append(C.md_table(pd.DataFrame(rows), floatfmt="{:.0f}") if rows else "_no completed tasks_")
    md.append("")
    for task in tasks:
        T = J["tasks"].get(task)
        if not T:
            continue
        for arm in C.LEARNED:
            c, cc = T["census"].get(arm), T["census"].get(C.LEARNED[arm])
            if not c or not cc:
                continue
            flips, cflips = c["rescued"] + c["regressed"], cc["rescued"] + cc["regressed"]
            if flips == 0:
                md.append(f"- {TL(task)} / {AL(arm)}: behaviorally inert on this panel (0 flips of {c['n']}); its control flips {cflips}.")
                continue
            ratio = flips / max(cflips, 1)
            lead = ("flips more scenarios than its control" if ratio > 1.25 else
                    "flips fewer scenarios than its control" if ratio < 0.8 else "flips about as many scenarios as its control")
            tail = ("the net change is small relative to the gross churn: the edit is not inert, it is bidirectional."
                    if abs(c["net"]) < 0.5 * flips else "most of the churn is one-directional.")
            md.append(f"- {TL(task)} / {AL(arm)}: {flips} flips ({c['rescued']} rescues, {c['regressed']} regressions, "
                      f"net {c['net']:+d}) — {lead} ({cflips} flips, net {cc['net']:+d}); {tail}")
    md.append("")

    # ---- 2 concordance
    md.append("## 2. Does the random control flip the *same* scenarios? (flip-set concordance)\n")
    md.append(f"Jaccard index of the learned arm's flip set with its dose-matched random control's flip set, against a "
              f"permutation null ({J['concordance']['permutation_draws']} draws, seed {J['concordance']['seed']}) that shuffles "
              "the control's labels **within the eligible stratum** (native failures for rescues, native successes for "
              "regressions; a naive within-task shuffle is in the JSON). High concordance = the same scenarios flip under "
              "*any* perturbation of this dose (scenario fragility); low concordance with many flips = direction-specific "
              f"effects or independent noise. Cells with fewer than {MIN_FLIPS_FOR_CONCORDANCE} flips in either arm are not "
              f"tested. Holm over the {J['concordance']['family_size']} tested (task × learned arm × flip type) cells.\n")
    rows, readings = [], []
    for task in tasks:
        T = J["tasks"].get(task)
        if not T:
            continue
        for arm, cc in T["concordance"].items():
            for kind in ("rescued", "regressed"):
                r = cc[kind]
                s = r.get("stratified")
                rows.append({"task": TL(task), "learned arm": AL(arm), "flip type": kind,
                             "|learned|": r["n_learned"], "|control|": r["n_control"], "overlap": r["n_overlap"],
                             "Jaccard": fmt(r["observed"]), "null mean": fmt(s["null_mean"]) if s else "n/a",
                             "null 95% range": f"[{s['null_q025']:.2f}, {s['null_q975']:.2f}]" if s else "n/a",
                             "p(above null)": fmt_p(s["p_above_null"]) if s else "n/a",
                             "Holm p": fmt_p(r.get("p_above_null_holm")), "reading": r["reading"]})
                readings.append((task, arm, kind, r))
    md.append(C.md_table(pd.DataFrame(rows), floatfmt="{:.0f}") if rows else "_no concordance cells_")
    md.append("")
    tested = [x for x in readings if x[3]["status"] == "tested"]
    n_untested = len(readings) - len(tested)
    frag = [x for x in tested if x[3]["reading"] == "overlap_above_null"]
    frag_holm = [x for x in frag if (_finite(x[3].get("p_above_null_holm")) or 1.0) < ALPHA]
    anti_holm = [x for x in tested if (_finite(x[3].get("p_below_null_holm")) or 1.0) < ALPHA]
    spec = [x for x in tested if x[3]["reading"] == "no_excess_overlap"]
    # the flip type that carries the regime's claim: rescues unless the regime is negative
    relevant_kind = "regressed" if reg["overall"] == "negative" else "rescued"
    frag_relevant = [x for x in frag_holm if x[2] == relevant_kind]
    # strong reading only when a MAJORITY of tested cells overlap above the null, the tested cells are not a sliver of the
    # panel (>= MIN_TESTED_FOR_STRONG), and the relevant flip type is among them
    majority = len(frag_holm) >= 2 and len(frag_holm) > len(tested) / 2 and bool(frag_relevant)
    fragile_strong = majority and len(tested) >= MIN_TESTED_FOR_STRONG
    fragile_weak = bool(frag_holm) and not fragile_strong
    if not tested:
        md.append("Too few flips in every cell for a concordance test: the edits and their controls barely move any "
                  "scenario, so there is no flip structure to characterise.")
    else:
        if frag_holm:
            md.append(f"**Above-null flip-set overlap** in {len(frag_holm)} of {len(tested)} tested cells after Holm "
                      f"({n_untested} of {len(readings)} cells untested for too few flips): " +
                      "; ".join(f"{TL(t)}/{AL(a)}/{k}: J={r['observed']:.2f} vs null mean {r['stratified']['null_mean']:.2f}, "
                                f"{r['n_overlap']} shared of {r['n_learned']}|{r['n_control']}"
                                for t, a, k, r in frag_holm) +
                      ". In those cells the control's flip set overlaps the learned edit's more than the stratified null "
                      "predicts (a Jaccard of J means the two sets share a fraction J of their union — not that they coincide). ")
            if fragile_strong:
                md.append(f"This holds in a majority of tested cells and includes the {relevant_kind} flips that carry the "
                          "regime's claim, so on this panel the flipped scenarios are predominantly ones that flip under *any* "
                          "perturbation of this dose (scenario fragility) rather than a selection specific to the fitted directions.")
            elif majority:
                md.append(f"This is a majority of the tested cells ({len(frag_holm)} of {len(tested)}), but only {len(tested)} of "
                          f"{len(readings)} cells could be tested at all (fewer than {MIN_TESTED_FOR_STRONG}), so the evidence is "
                          "confined to those cells: weak evidence of dose-level fragility there, not a characterisation of the "
                          "fitted directions or of the panel.")
            else:
                md.append(f"This is a minority of tested cells ({len(frag_holm)} of {len(tested)}" +
                          (f"; none of them a {relevant_kind} cell" if not frag_relevant else "") +
                          "): weak evidence of dose-level fragility in those cells, not a characterisation of the fitted directions.")
        elif frag:
            md.append(f"{len(frag)} of {len(tested)} tested cells exceed the null uncorrected but none survives Holm "
                      f"({n_untested} cells untested): weak or no evidence that the learned edit and its control flip the same scenarios.")
        else:
            md.append(f"No tested cell ({len(tested)}; {n_untested} untested) shows overlap above the stratified null: the learned edits "
                      "and their random controls flip largely *different* scenarios.")
        if anti_holm:
            md.append(f" {len(anti_holm)} cell(s) overlap *less* than the null after Holm (" +
                      "; ".join(f"{TL(t)}/{AL(a)}/{k}" for t, a, k, _ in anti_holm) +
                      "): the learned edit and its control flip disjoint scenario sets more than chance — a direction-specific "
                      "signature worth a pre-registered follow-up, if it survives the feature tests below.")
        many = [x for x in spec if x[3]["n_learned"] >= 10]
        if many:
            md.append(f" In {len(many)} cell(s) the learned arm flips ≥10 scenarios without above-null overlap with its control (" +
                      "; ".join(f"{TL(t)}/{AL(a)}/{k}" for t, a, k, _ in many) +
                      "): consistent with direction-specific effects, but equally consistent with independent planner noise "
                      "— concordance alone cannot separate the two; the feature associations in §3 are the test.")
    md.append("")

    # ---- 3 associations
    A = J["associations"]
    fs = A["family_sizes"]
    md.append("## 3. Which scenario features predict rescue vs regression? (EXPLORATORY)\n")
    md.append("**Primary design (within stratum).** Among native failures: does the feature predict *rescue*? Among native "
              "successes: does it predict *regression*? Spearman rank correlation with the binary outcome (equivalent to a "
              "rank-biserial test) and a Mann-Whitney U (flipped vs not), per (task, arm, feature, stratum), run only when "
              f"both groups have ≥{MIN_GROUP_FOR_TEST} scenarios. Holm over **all** {fs['stratified_spearman']} Spearman tests "
              f"and separately over all {fs['stratified_mwu']} MWU tests. This design is immune to the tautology that a rescue "
              "requires a native failure.\n")
    md.append("**Secondary design (pooled signed outcome, as specified).** Spearman between the feature and (+1 rescue, −1 "
              "regression, 0 unchanged) over all scenarios plus pairwise MWU between the three groups. `native_distance` and "
              "`native_reward` are excluded here because they are determined by the native outcome and would score as "
              f"associations for every arm. Holm over {fs['pooled_spearman']} Spearman and {fs['pooled_mwu']} MWU tests.\n")
    md.append("Features: " + "; ".join(f"`{k}` = {v}" for k, v in {**SCENARIO_FEATURES, **ARM_FEATURES}.items()) + ". "
              f"`{'`, `'.join(DESCRIPTIVE_ONLY_FEATURES)}` is recorded in the JSON but enters no test family (it is a deterministic "
              "function of `first_divergence`, so testing both would only inflate the family).\n")
    md.append(f"Resolution: the median within-stratum cell has n = {A['n_typical_stratified']} scenarios, so the smallest "
              f"|rho| detectable at 80% power is ≈{fmt(A['min_detectable_rho_stratified_uncorrected'])} uncorrected and "
              f"≈{fmt(A['min_detectable_rho_stratified_bonferroni'])} at the Bonferroni-equivalent threshold α/{fs['stratified_spearman']} "
              "(a lower bound on what Holm can detect); the cross-task combination in "
              f"§5 pools ≈{A['n_typical_cross_task']} scenarios per (arm, feature, stratum) and is the more powerful test of a "
              "task-general association.\n")
    top = pd.DataFrame(A["top_stratified"])
    if len(top):
        show = pd.DataFrame({"task": top.task.map(TL), "arm": top.arm.map(AL), "feature": top.feature, "outcome": top.outcome,
                             "n": top.n, "flipped": top.n_flipped, "median flipped": top.median_flipped.map(fmt),
                             "median not": top.median_not_flipped.map(fmt), "rho": top.spearman_rho.map(lambda v: fmt(v, 3)),
                             "p": top.spearman_p.map(fmt_p), "Holm p": top.spearman_p_holm.map(fmt_p),
                             "MWU p": top.mwu_p.map(fmt_p), "MWU Holm p": top.mwu_p_holm.map(fmt_p)})
        md.append("Strongest within-stratum associations (by uncorrected Spearman p):\n")
        md.append(C.md_table(show, floatfmt="{:.0f}"))
    else:
        md.append("_no within-stratum association test could be run (too few flips or no finite features)_")
    md.append("")
    sig = A["holm_significant_stratified"]
    sig_mwu = A["holm_significant_stratified_mwu"]
    sig_pooled = A["holm_significant_pooled"]
    if sig:
        md.append(f"**{len(sig)} within-stratum association(s) survive Holm** over the Spearman family: " +
                  "; ".join(f"{TL(s['task'])}/{AL(s['arm'])}: `{s['feature']}` → {s['outcome']} rho={s['spearman_rho']:+.2f} "
                            f"(Holm p={fmt_p(s['spearman_p_holm'])})" for s in sig) + ". ")
        learned_sig = [s for s in sig if s["arm"] in C.LEARNED]
        control_sig = [s for s in sig if s["arm"] not in C.LEARNED]
        if learned_sig and not control_sig:
            md.append("They hold for learned arms only, not for their random controls: which scenarios the learned edit flips "
                      "is predictable from scenario/edit features in a way the control's flips are not — the signature "
                      "expected of a direction-specific mechanism rather than dose-level fragility.")
        elif control_sig and not learned_sig:
            md.append("They hold for the random controls only: whatever structure exists is a property of perturbing the "
                      "predictor at this dose, not of the fitted directions.")
        else:
            md.append("They appear for both learned arms and controls, so the structure is at least partly a property of "
                      "the perturbation dose rather than the fitted directions.")
        if any(s["feature"] in ("coef_norm_mean", "active_fraction") for s in sig):
            md.append(" A delivered-edit feature (coefficient norm / active fraction) is among them: scenarios receiving a "
                      "larger edit flip differently, the dose-response signature one would want from a mechanism — but the "
                      "coefficient is set by the planner's own candidates, so this is correlational.")
    else:
        best = top.iloc[0] if len(top) else None
        md.append("**No within-stratum association survives Holm** over the Spearman family (" +
                  (f"best uncorrected p = {fmt_p(best.spearman_p)} for {TL(best.task)}/{AL(best.arm)}/`{best.feature}` → {best.outcome}"
                   if best is not None else "no tests run") + "). ")
        if sig_mwu:
            md.append(f"{len(sig_mwu)} MWU comparison(s) survive their own Holm family: " +
                      "; ".join(f"{TL(s['task'])}/{AL(s['arm'])}/`{s['feature']}` → {s['outcome']} (Holm p={fmt_p(s['mwu_p_holm'])})" for s in sig_mwu) +
                      ". Treat as leads for a pre-registered follow-up, not findings.")
        else:
            md.append("No MWU comparison survives its Holm family either. Within the resolution stated above, which scenarios "
                      "flip is not predictable from native outcome, scenario geometry, planner effort, delivered edit size or "
                      "when the actions diverged — per task, the flips look like unstructured planner sensitivity rather than "
                      "a scenario-selective mechanism.")
    if sig_pooled:
        md.append(f" Pooled signed-outcome design: {len(sig_pooled)} association(s) survive Holm (" +
                  "; ".join(f"{TL(s['task'])}/{AL(s['arm'])}/`{s['feature']}` rho={s['spearman_rho']:+.2f}" for s in sig_pooled) +
                  "); these mix rescue and regression and are reported for completeness.")
    else:
        md.append(" Pooled signed-outcome design: nothing survives Holm.")
    md.append("")

    # ---- 4 tertiles
    Tt = J["tertiles"]
    md.append("## 4. Are gains concentrated in near-miss scenarios? (native-distance tertiles, EXPLORATORY)\n")
    md.append(f"(a) **Descriptive only.** Net effect by tertile of native final distance over **all** scenarios: paired arm − native "
              f"success with a scenario-cluster bootstrap ({C.BOOTSTRAP_DRAWS} draws, Bonferroni family {Tt['family_sizes']['net']} = "
              "task × arm × tertile). The near tertile is dominated by native *successes*, which can only regress, and the far "
              "tertile by native *failures*, which can only be rescued, so the tertile pattern — and any all-scenario near-vs-far "
              "contrast — is determined by tertile composition. Table (a) shows where the net effect sits; it is not evidence "
              "about near-misses and no inference is drawn from it. "
              "(b) **Near-miss test:** tertiles of native distance **among native failures only**; rescue rate per tertile "
              f"(Wilson 95%), a chi-square test of homogeneity (Holm over {Tt['family_sizes']['rescue_chi2']} task × arm cells), and a "
              f"near-miss − far rescue-rate bootstrap contrast (Bonferroni family {Tt['family_sizes']['near_far']} = task × arm). "
              "The symmetric regression-rate analysis among native successes is in the JSON.\n")
    net = pd.DataFrame(Tt["net"])
    if len(net):
        show = pd.DataFrame({"task": net.task.map(TL), "arm": net.arm.map(AL), "tertile": net.tertile, "n": net.n,
                             "native succ %": net.native_success_pct.map(lambda v: fmt(v, 0)),
                             "net pp [CI]": [f"{a:+.1f} [{b:+.1f}, {c:+.1f}]" for a, b, c in zip(net.net_pp, net.lower_pp, net.upper_pp)],
                             "rescued": net.rescued, "regressed": net.regressed})
        md.append("(a) Net effect by tertile, all scenarios (descriptive; confounded by composition — see column `native succ %`):\n")
        md.append(C.md_table(show, floatfmt="{:.0f}"))
        md.append("")
    miss = pd.DataFrame(Tt["near_miss"])
    if len(miss):
        mm = miss[miss.stratum == "native_failures"]
        show = pd.DataFrame({"task": mm.task.map(TL), "arm": mm.arm.map(AL), "tertile": mm.tertile.map({1: "near-miss", 2: "mid", 3: "far"}),
                             "native failures": mm.n, "distance range": [f"{a:.2f}–{b:.2f}" for a, b in zip(mm.distance_min, mm.distance_max)],
                             "rescued": mm.flipped,
                             "rescue rate [Wilson 95%]": [f"{r*100:.0f}% [{lo*100:.0f}, {hi*100:.0f}]" if r is not None else "n/a"
                                                          for r, lo, hi in zip(mm.rate, mm.rate_lo, mm.rate_hi)]})
        md.append("(b) Rescue rate by native-distance tertile among native failures:\n")
        md.append(C.md_table(show, floatfmt="{:.0f}"))
        md.append("")
    ct = pd.DataFrame(Tt["contrasts"])
    for _, r in (ct.iterrows() if len(ct) else []):
        if r.arm not in C.LEARNED:
            continue
        lab = f"{TL(r.task)} / {AL(r.arm)}"
        g = net[(net.task == r.task) & (net.arm == r.arm)].sort_values("tertile")
        near_succ = _finite(g.native_success_pct.iloc[0]) if len(g) else None
        far_succ = _finite(g.native_success_pct.iloc[-1]) if len(g) else None
        md.append(f"- {lab}: (a) descriptive only — the near tertile is {fmt(near_succ, 0)}% native successes and the far tertile "
                  f"{fmt(far_succ, 0)}%, so its tertile pattern is set by composition; the test is (b).")
        mm = miss[(miss.task == r.task) & (miss.arm == r.arm) & (miss.stratum == "native_failures")].sort_values("tertile") if len(miss) else pd.DataFrame()
        chi_h = _finite(r.get("rescue_rate_chi2_p_holm"))
        nf_d, nf_lo, nf_hi = (_finite(r.get("rescue_near_minus_far_pp")), _finite(r.get("rescue_near_minus_far_lower")),
                              _finite(r.get("rescue_near_minus_far_upper")))
        nf = (f"near-miss − far rescue-rate contrast {nf_d:+.1f} pp [{nf_lo:+.1f}, {nf_hi:+.1f}] (Bonferroni)"
              if nf_d is not None else "near-miss − far contrast n/a")
        if len(mm) == 3 and mm.rate.notna().all():
            rates = mm.rate.values.astype(float)
            k = int(np.argmax(rates))
            name = ["near-miss", "mid", "far"][k]
            nf_sig = nf_d is not None and (nf_lo > 0 or nf_hi < 0)
            if chi_h is not None and chi_h < ALPHA:
                md.append(f"  (b) Rescue rate among native failures differs across tertiles (chi-square Holm p={fmt_p(chi_h)}); "
                          f"highest in the {name} tertile ({rates[k]*100:.0f}%); {nf}: " +
                          ("rescues concentrate in near-miss scenarios." if k == 0 and nf_sig and nf_d > 0 else
                           "rescues concentrate in far (not near-miss) scenarios." if k == 2 and nf_sig and nf_d < 0 else
                           "the tertiles differ but the near-miss − far contrast itself does not exclude zero, so the pattern is "
                           "not a monotone near-miss concentration."))
            elif nf_sig:
                md.append(f"  (b) Rescue rate among native failures is highest in the {name} tertile ({rates[k]*100:.0f}%); the "
                          f"chi-square homogeneity test does not reject after Holm (p={fmt_p(chi_h)}) but the {nf} excludes zero: "
                          "treat as a weak lead, not a finding (the two tests disagree).")
            else:
                md.append(f"  (b) Rescue rate among native failures is highest in the {name} tertile ({rates[k]*100:.0f}%) but not "
                          f"distinguishable across tertiles (chi-square Holm p={fmt_p(chi_h)}; {nf}): no evidence that rescues are "
                          "concentrated in near-miss scenarios.")
        else:
            md.append("  (b) too few native failures for a near-miss tertile test.")
    md.append("")

    # ---- 5 cross-task
    X = J["cross_task"]
    md.append("## 5. Is any within-stratum feature association consistent across tasks? (EXPLORATORY)\n")
    md.append(f"Per (arm, feature, stratum): per-task Spearman rhos combined by Stouffer's method (z_i = atanh(rho_i)·√(n_i−3), "
              f"weights √(n_i−3)); Holm over the {X['family_size']} combinations. `same sign` requires every task's rho to share a sign.\n")
    xt = pd.DataFrame(X["table"])
    control_echo, control_quiet = [], []
    if len(xt):
        xs = xt.sort_values("stouffer_p").head(15)
        show = pd.DataFrame({"arm": xs.arm.map(AL), "feature": xs.feature, "outcome": xs.outcome, "tasks": xs.n_tasks, "n": xs.n_total,
                             "rhos by task": [", ".join(f"{TL(t)} {fmt(v)}" for t, v in d.items()) for d in xs.rhos],
                             "same sign": xs.all_same_sign, "Stouffer z": xs.stouffer_z.map(fmt), "p": xs.stouffer_p.map(fmt_p),
                             "Holm p": xs.stouffer_p_holm.map(fmt_p)})
        md.append(C.md_table(show, floatfmt="{:.0f}"))
        md.append("")
        cons = _sig_rows(X["table"], "stouffer_p_holm")
        if cons:
            same = [r for r in cons if r["all_same_sign"]]
            md.append(f"**{len(cons)} combined association(s) survive Holm**: " +
                      "; ".join(f"{AL(r['arm'])}/`{r['feature']}` → {r['outcome']} z={r['stouffer_z']:+.2f}, same sign on all tasks: {r['all_same_sign']}" for r in cons) + ". ")
            if same:
                md.append("Sign-consistent across every task with data: this is the cross-task regularity a scenario-selective "
                          "mechanism would produce, and it is the first thing a pre-registered follow-up should test.")
            else:
                md.append("But the per-task signs disagree, so the combined result is driven by a subset of tasks and should not "
                          "be read as a task-general mechanism.")
            learned_c = [r for r in cons if r["arm"] in C.LEARNED]
            control_c = [r for r in cons if r["arm"] not in C.LEARNED]
            # guard against a 'learned-only' overclaim: does the matched control show the same-sign tendency (uncorrected)?
            for r in learned_c:
                ctrl = C.LEARNED[r["arm"]]
                m = [x for x in X["table"] if x["arm"] == ctrl and x["feature"] == r["feature"] and x["stratum"] == r["stratum"]]
                if m and _finite(m[0].get("stouffer_p")) is not None:
                    same_sign = np.sign(m[0]["stouffer_z"] or 0) == np.sign(r["stouffer_z"] or 0)
                    if m[0]["stouffer_p"] < ALPHA and same_sign:
                        control_echo.append((r, m[0]))
                    else:
                        control_quiet.append((r, m[0]))
            if learned_c and not control_c and not control_echo:
                md.append(" It survives Holm for the learned arm(s) only, and the matched random control shows no comparable "
                          "tendency (" + "; ".join(f"{AL(m['arm'])}/`{m['feature']}` z={fmt(m['stouffer_z'])}, p={fmt_p(m['stouffer_p'])}"
                                                   for _, m in control_quiet) + ").")
            elif learned_c and not control_c and control_echo:
                md.append(" It survives Holm for the learned arm(s) only, **but** the matched random control shows the same-sign "
                          "tendency uncorrected (" + "; ".join(f"{AL(m['arm'])}/`{m['feature']}` z={fmt(m['stouffer_z'])}, p={fmt_p(m['stouffer_p'])}"
                                                               for _, m in control_echo) +
                          "), so 'learned-only' is not established: the association is at least partly a dose-level property.")
            elif control_c:
                md.append(" It also (or only) holds for a random control, so it is at least partly a dose-level property.")
        else:
            best = xt.sort_values("stouffer_p").iloc[0]
            md.append(f"**No combined association survives Holm** (best: {AL(best.arm)}/`{best.feature}` → {best.outcome} "
                      f"z={fmt(best.stouffer_z)}, uncorrected p={fmt_p(best.stouffer_p)}). Pooling the tasks does not reveal a "
                      "feature that consistently separates rescued from unrescued (or regressed from unregressed) scenarios.")
    else:
        md.append("_no cross-task table (no within-stratum tests ran)_")
        cons = []
    md.append("")

    # ---- 6 offline context
    off = J.get("offline_context", {})
    if any(v for d in off.values() for v in d.values()):
        md.append("## 6. Offline forecast context (development sweeps, read-only join)\n")
        for task in tasks:
            for arm, v in (off.get(task) or {}).items():
                if v:
                    md.append(f"- {TL(task)} / {AL(arm)}: BF16 H6 proprio error reduction {v['effect']:+.2f}% "
                              f"[{v['lower']:+.2f}, {v['upper']:+.2f}] ({v['result']}).")
        md.append("")
        md.append("A forecast gain that is uniform across scenarios next to a behavioral flip pattern that is unstructured is "
                  "the combination the development results already suggested (better forecast ≠ better plan); the sections "
                  "above test the behavioral half of that on the fresh panel.\n")

    # ---- 7 physics tie-back
    md.append("## 7. Reading against the 'Interpreting Physics in Video World Models' framing\n")
    structured = bool(sig) or bool(cons)
    # dose-level if any surviving association belongs to a control arm, or the matched control echoes a learned-arm hit
    structured_dose_level = (any(s_["arm"] not in C.LEARNED for s_ in sig) or any(r["arm"] not in C.LEARNED for r in cons)
                             or bool(control_echo))
    k_frag, m_tested = len(frag_holm), len(tested)
    frag_summary = (f"{k_frag} of {m_tested} tested concordance cells overlap above the stratified null after Holm "
                    f"({n_untested} cells untested)")
    if structured and not fragile_strong and structured_dose_level:
        md.append("Joseph et al. (2026) find that physical variables in video world models are distributed, and that steering "
                  "them needs coordinated many-dimensional edits; a single direction does little. Here a feature association "
                  "survives correction, but the same tendency is present for the dose-matched random control (§3/§5), so the "
                  "structure is a property of perturbing the predictor at this dose — which scenarios are near a decision "
                  "boundary — and says nothing specific about the fitted directions. That is consistent with their "
                  "distributed-code picture: any perturbation of comparable size moves the same marginal scenarios.")
    elif structured and not fragile_strong:
        md.append("Joseph et al. (2026) find that physical variables in video world models are distributed, and that steering "
                  "them needs coordinated many-dimensional edits; a single direction does little. Here the flips of the "
                  "learned edit are predictable from scenario/edit features in a way that its random control's are not (or "
                  "consistently across tasks), which is the one pattern that would argue a four-direction edit at B3/H3 has found "
                  "a *usable* handle rather than merely perturbing a distributed code. It is a lead to pre-register, not a mechanism.")
        if fragile_weak:
            md.append(f" ({frag_summary}; a minority, so it does not overturn the structured reading but should be reported with it.)")
    elif fragile_strong and not structured:
        md.append(f"Joseph et al. (2026) find that physical variables in video world models are distributed and that single "
                  f"directions do not steer them. The concordance result here ({frag_summary}, including the {relevant_kind} cells) "
                  "is the closest behavioral analogue available from outcomes alone: the scenarios the random control flips "
                  "overlap those the learned edit flips more than chance, and no feature association separates the two, so at "
                  "this dose the outcome-level effect of the four fitted directions is not distinguishable from that of a "
                  "matched random perturbation of a distributed code. This is a statement about closed-loop outcomes on this "
                  "panel, not a characterisation of what the directions encode; it is consistent with their 'coarse causal "
                  "influence without a compact state variable' reading.")
    elif fragile_strong and structured:
        md.append(f"Both signatures appear: {frag_summary} (fragility) and some feature associations survive correction "
                  "(structure). Given the distributed-code picture of Joseph et al. (2026), the parsimonious reading is dose-level "
                  "fragility modulated by a scenario feature that applies to any perturbation; a direction-specific mechanism "
                  "would require the association to hold for learned arms only.")
    elif fragile_weak:
        md.append(f"Joseph et al. (2026) find physical variables in video world models to be distributed and steerable only by "
                  f"coordinated many-dimensional edits. Here the evidence is thin on both sides: {frag_summary}, and no feature "
                  "association survives correction. Above-null overlap in a minority of cells is weak evidence of dose-level "
                  "fragility in those cells — not a characterisation of the fitted directions, and not evidence that they act "
                  "as a generic perturbation. The honest summary is that at this resolution the learned edit's flips are neither "
                  "shown to be structured nor shown to coincide with its control's beyond a few cells.")
    else:
        md.append("Joseph et al. (2026) find physical variables in video world models to be distributed and steerable only by "
                  "coordinated many-dimensional edits. Here neither signature is present: the learned edit's flips are not "
                  "concordant with its control's and not predictable from any scenario or edit feature at this resolution. "
                  "At the level of closed-loop outcomes a rank-4 edit at one block and one imagined step behaves like "
                  "unstructured planner sensitivity — what one expects if the quantity it corrects is a small component of a "
                  "distributed code that the planner's objective does not isolate.")
    md.append("")

    # ---- does / does not
    md.append("## What this does and does not establish\n")
    md.append("**Does:**")
    md.append(f"- Gives the complete rescue/regression census for every arm on every completed task "
              f"({sum(len(J['tasks'][t]['census']) for t in J['tasks'])} arm-task cells), read off the frozen paired outcomes.")
    md.append(f"- Tests, in {len(tested)} cell(s), whether the learned edit and its dose-matched random control flip the same "
              "scenarios, against a null that respects that only native failures can be rescued and only native successes can regress.")
    md.append(f"- Tests {fs['stratified_spearman']} within-stratum feature associations (+{fs['pooled_spearman']} pooled) and "
              f"{fs['stratified_mwu']} + {fs['pooled_mwu']} rank comparisons with a single Holm correction per family, and states "
              "the resolution of those tests.")
    md.append("**Does not:**")
    md.append("- Establish or refute any behavioral effect: only the arm-vs-native contrasts are pre-registered, and this "
              "analysis conditions on them post hoc.")
    md.append("- Identify a mechanism. A surviving association is a *lead* for a pre-registered follow-up; an absent one only "
              f"bounds per-task within-stratum associations to |rho| ≲ {fmt(A['min_detectable_rho_stratified_uncorrected'])} "
              f"(≈{fmt(A['min_detectable_rho_stratified_bonferroni'])} at the corrected threshold).")
    md.append("- Separate direction-specific effects from planner noise when concordance is low: both give low Jaccard. Nor does "
              "above-null overlap in a minority of cells characterise the fitted directions: the concordance test measures set "
              "overlap of outcomes, and shared scenario difficulty alone produces it.")
    md.append("- Test near-miss concentration with the all-scenario tertile table (a): it is confounded by tertile composition and "
              "is reported as description only; the within-failure rate contrast (b) is the test.")
    md.append("- Use closed-loop state beyond the recorded scalars: `initial_goal_distance` is a proxy over the full state vector "
              "(MetaWorld states include object/gripper dimensions; navigation states include velocities), and the divergence "
              "features are tautologically linked to flipping (a non-diverging scenario cannot flip), so they are tested only "
              "among diverged scenarios.")
    md.append("- Jointly correct across the *sections* (concordance, associations, tertiles, cross-task): each family is corrected "
              "on its own and the number of sections is itself a multiplicity.")
    if prov["synthetic"]:
        md.append("- Say anything about the real confirmation: the input is a synthetic fixture.")
    notes = J.get("notes", [])
    skips = [n for n in notes if n.startswith("skip: ")]
    others = [n for n in notes if not n.startswith("skip: ")]
    if notes:
        md.append("\n### Degradations and skipped items\n")
        if others:
            md.append(f"{len(others)} missing/degraded artifact(s):")
            md.extend(f"- {n}" for n in others)
        if skips:
            from collections import Counter
            kinds = Counter(n.rsplit(": ", 1)[-1].split(" (")[0] for n in skips)
            md.append(f"\n{len(skips)} individual tests not run (all listed in the JSON `notes`): " +
                      "; ".join(f"{v} × {k}" for k, v in kinds.most_common()) + ".")
    md.append(f"\n_Reproduce:_ `{J['command']}`")
    return "\n".join(md) + "\n"


# ----------------------------------------------------------------------------- main
def main(argv=None):
    global PERM_DRAWS, PERM_SEED
    p = C.standard_parser("Scenario heterogeneity of rescues/regressions vs native (EXPLORATORY)")
    p.add_argument("--fixture-regime", default=None, help="label only; recorded in provenance")
    p.add_argument("--permutations", type=int, default=PERM_DRAWS)
    p.add_argument("--seed", type=int, default=PERM_SEED)
    a = p.parse_args(argv)
    PERM_DRAWS, PERM_SEED = a.permutations, a.seed
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    notes = []

    protocol = C.load_freeze(a.freeze)
    report = C.load_analysis(a.analysis)
    prov = C.provenance(a.results, a.freeze, a.analysis)
    prov["fixture_regime"] = a.fixture_regime
    prov["generated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tasks = [t for t in a.tasks if t in protocol.get("tasks", {})]
    for t in a.tasks:
        if t not in tasks:
            notes.append(f"{t}: not in protocol; skipped")

    LOG.info("loading episodes")
    df = C.episodes_frame(a.results, tasks)
    if not len(df):
        raise SystemExit("no completed scenarios found under --results")
    LOG.info("loading refined-arm energy")
    try:
        ef = C.energy_frame(a.results, tasks)
    except Exception as exc:
        notes.append(f"energy_frame failed ({exc}); delivered-edit features unavailable")
        ef = pd.DataFrame()

    regime = regime_table(report, df, tasks)
    notes.extend(regime.pop("notes"))

    per_task = {}
    for task in tasks:
        LOG.info("analysing %s", task)
        T = analyse_task(task, protocol, df, ef, a.results, notes)
        if T:
            per_task[task] = T

    conc_cells = [(t, arm, k) for t, T in per_task.items() for arm in T["concordance"]
                  for k in ("rescued", "regressed") if T["concordance"][arm][k]["status"] == "tested"]
    if conc_cells:
        for tail in ("p_above_null", "p_below_null"):   # each tail Holm-corrected over the same family of tested cells
            adj = C.holm([per_task[t]["concordance"][arm][k]["stratified"][tail] for t, arm, k in conc_cells])
            for (t, arm, k), v in zip(conc_cells, adj):
                per_task[t]["concordance"][arm][k][tail + "_holm"] = float(v)
    for T in per_task.values():
        for cc in T["concordance"].values():
            for k in ("rescued", "regressed", "any_flip"):
                cc[k]["reading"] = concordance_reading(cc[k])

    strat, pooled, fam = association_tests(per_task, notes)
    net, miss, tert_c, tert_fam = tertile_analysis(per_task, notes)
    xt, xfam = cross_task_consistency(strat)

    offline = {}
    for task in tasks:
        offline[task] = {}
        for arm in C.LEARNED:
            try:
                offline[task][arm] = C.offline_effect(task, C.OFFLINE_ARM_NAME.get(arm, arm))
            except Exception as exc:
                offline[task][arm] = None
                notes.append(f"{task}/{arm}: offline contrasts unavailable ({exc})")
            if offline[task][arm] is None:
                notes.append(f"{task}/{arm}: no offline forecast contrast row; context omitted")

    figs = {"flips": fig_flips(per_task, tasks, out / f"{SCRIPT}_flips.png"),
            "concordance": fig_concordance(per_task, tasks, out / f"{SCRIPT}_concordance.png"),
            "tertiles": fig_tertiles(net, miss, tasks, out / f"{SCRIPT}_tertiles.png")}

    n_strat = int(np.median(strat.n)) if len(strat) else None
    n_cross = int(np.median(xt.n_total)) if len(xt) else None
    strat_recs = _nan_to_none(strat.to_dict(orient="records")) if len(strat) else []
    pooled_recs = _nan_to_none(pooled.to_dict(orient="records")) if len(pooled) else []
    top_strat = _nan_to_none(strat.sort_values("spearman_p").head(12).to_dict(orient="records")) if len(strat) else []
    J = dict(
        script=SCRIPT, exploratory=True,
        preregistered="only arm-vs-native contrasts in analysis/report.json; everything in this file is post hoc",
        provenance=prov, command=" ".join([Path(sys.argv[0]).name] + [str(x) for x in (argv or sys.argv[1:])]),
        analysis_report=dict(method=report.get("method"), regime=report.get("regime"), synthetic=report.get("synthetic", False)),
        regime=regime,
        tasks={t: dict(n=int(len(T["native"])), native_success_pct=float(T["native"].mean() * 100), census=T["census"],
                       concordance=T["concordance"],
                       flip_labels={arm: dict(episodes=[int(e) for e in T["matrix"].index], labels=[int(x) for x in f])
                                    for arm, f in T["flips"].items()},
                       scenario_features=T["scen"].reset_index().to_dict(orient="records"),
                       arm_features={arm: af.reset_index().to_dict(orient="records") for arm, af in T["arm_feats"].items() if len(af)})
               for t, T in per_task.items()},
        concordance=dict(exploratory=True, permutation_draws=PERM_DRAWS, seed=PERM_SEED, min_flips=MIN_FLIPS_FOR_CONCORDANCE,
                         null="control flip labels permuted within the eligible stratum (native failures for rescues, "
                              "native successes for regressions); naive = within task",
                         family="Holm over tested task x learned-arm x flip-type cells, separately for each tail "
                                "(p_above_null_holm, p_below_null_holm)", family_size=len(conc_cells),
                         per_cell_seed="np.random.default_rng([seed, task index in common.TASKS, arm index, flip-type index])"),
        associations=dict(exploratory=True, family_sizes=fam,
                          family="stratified: Holm over ALL task x arm x feature x stratum Spearman tests, separately over all MWU; "
                                 "pooled: Holm over ALL task x arm x feature Spearman tests (native-outcome features excluded), "
                                 "separately over all pairwise MWU",
                          native_outcome_features_excluded_from_pooled=list(NATIVE_OUTCOME_FEATURES),
                          n_typical_stratified=n_strat, n_typical_cross_task=n_cross,
                          min_detectable_rho_stratified_uncorrected=min_detectable_rho(n_strat, ALPHA),
                          min_detectable_rho_stratified_bonferroni=min_detectable_rho(n_strat, ALPHA / max(fam["stratified_spearman"], 1)),
                          descriptive_only_features=list(DESCRIPTIVE_ONLY_FEATURES),
                          features={**SCENARIO_FEATURES, **ARM_FEATURES},
                          stratified=strat_recs, pooled=pooled_recs, top_stratified=top_strat,
                          holm_significant_stratified=_sig_rows(strat_recs, "spearman_p_holm"),
                          holm_significant_stratified_mwu=_sig_rows(strat_recs, "mwu_p_holm"),
                          holm_significant_pooled=_sig_rows(pooled_recs, "spearman_p_holm")),
        tertiles=dict(exploratory=True, family_sizes=tert_fam,
                      family="net (DESCRIPTIVE ONLY; confounded by tertile composition): Bonferroni over task x arm x tertile "
                             "paired bootstraps; near-far rescue/regression-rate contrast WITHIN the eligible stratum: Bonferroni "
                             "over task x arm; rescue/regression-rate chi-square: Holm over task x arm",
                      net=net.to_dict(orient="records") if len(net) else [],
                      near_miss=miss.to_dict(orient="records") if len(miss) else [],
                      contrasts=tert_c.to_dict(orient="records") if len(tert_c) else []),
        cross_task=dict(exploratory=True, family="Holm over arm x feature x stratum Stouffer combinations", family_size=xfam,
                        table=xt.to_dict(orient="records") if len(xt) else []),
        offline_context=offline, figures={k: str(out / f"{SCRIPT}_{k}.png") if v else None for k, v in figs.items()},
        notes=notes)
    J = _nan_to_none(J)
    C.write_json(out / f"{SCRIPT}.json", J)
    (out / f"{SCRIPT}.md").write_text(build_markdown(J, tasks))
    LOG.info("wrote %s", out / f"{SCRIPT}.json")
    LOG.info("regime=%s; concordance cells tested=%d; stratified spearman family=%d; holm-significant=%d; cross-task holm-significant=%d",
             regime["overall"], len(conc_cells), fam["stratified_spearman"], len(J["associations"]["holm_significant_stratified"]),
             len(_sig_rows(J["cross_task"]["table"], "stouffer_p_holm")))
    return 0


def _nan_to_none(o):
    if isinstance(o, dict):
        return {str(k): _nan_to_none(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_nan_to_none(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _nan_to_none(o.tolist())
    if isinstance(o, pd.DataFrame):
        return _nan_to_none(o.to_dict(orient="records"))
    return o


if __name__ == "__main__":
    sys.exit(main())
