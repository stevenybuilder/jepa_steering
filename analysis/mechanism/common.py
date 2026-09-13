"""Shared loaders and statistics for the post-confirmation mechanism analyses.

Read-only over finished artifacts. See DATA_CONTRACT.md. No model or simulator calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TASKS = ("reach", "reach-wall", "pointmaze", "wall")
ARMS = ("native", "fixed_rank4", "matched_random_fixed_rank4", "coupling_only",
        "matched_random_coupling", "joint", "visual_only", "action_condition_only")
LEARNED = {"fixed_rank4": "matched_random_fixed_rank4", "coupling_only": "matched_random_coupling"}
COMPONENT = ("joint", "visual_only", "action_condition_only")
ARM_LABEL = {
    "native": "Unsteered", "fixed_rank4": "Refined rank-4 edit",
    "matched_random_fixed_rank4": "Random-subspace control", "coupling_only": "Equal-budget coupling",
    "matched_random_coupling": "Random-direction control", "joint": "Unscaled joint",
    "visual_only": "Visual-only", "action_condition_only": "Action-only"}
TASK_LABEL = {"reach": "Reach", "reach-wall": "Reach-Wall", "pointmaze": "PointMaze", "wall": "Wall"}
BOOTSTRAP_DRAWS = 20000
BOOTSTRAP_SEED = 2026091221
USEFUL_GAIN_PP = 5.0


# ----------------------------------------------------------------------------- io
def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_default) + "\n")


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def standard_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--results", type=Path, required=True, help="fresh confirmation results tree")
    p.add_argument("--freeze", type=Path, required=True, help="freeze dir containing protocol.json")
    p.add_argument("--analysis", type=Path, required=True, help="frozen analysis dir containing report.json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tasks", nargs="*", default=list(TASKS))
    return p


# ------------------------------------------------------------------------ loading
def load_freeze(freeze_dir):
    return read_json(Path(freeze_dir) / "protocol.json")


def scenario_rows(protocol, task):
    rows = [r for r in protocol["tasks"][task]["records"] if r["role"] == "scientific_candidates"]
    return sorted(rows, key=lambda r: r["episode"])


def load_analysis(analysis_dir):
    return read_json(Path(analysis_dir) / "report.json")


def iter_records(results_dir, task, arms=ARMS, require_done=True):
    """Yield (episode, arm, record) for every completed scenario of a task."""
    task_dir = Path(results_dir) / task
    if not task_dir.exists():
        return
    for scen in sorted(task_dir.glob("scenario-*")):
        if require_done and not (scen / "DONE.json").exists():
            continue
        episode = int(scen.name.split("-")[1])
        for arm in arms:
            path = scen / f"{arm}.json"
            if path.exists():
                yield episode, arm, read_json(path)


def episodes_frame(results_dir, tasks=TASKS):
    """One row per (task, episode, arm) with the scalar outcome fields."""
    rows = []
    for task in tasks:
        for episode, arm, rec in iter_records(results_dir, task):
            res = rec["result"]
            pc = res.get("planning_calls", [])
            h6 = [c for c in rec.get("calls", []) if c.get("horizon") == 6 and c.get("candidates", 0) > 1]
            rows.append(dict(
                task=task, episode=episode, arm=arm,
                success=bool(res["native_success"]),
                distance=float(res["native_state_distance"]),
                reward=float(res["native_reward"]),
                expert_success=float(res.get("expert_success", np.nan)),
                elementary_steps=int(res.get("elementary_steps", 0)),
                n_planning_calls=len(pc),
                cem_iterations=int(np.sum([c.get("iterations", 0) for c in pc])) if pc else 0,
                planning_seconds=float(np.sum([c.get("seconds", 0.0) for c in pc])) if pc else np.nan,
                n_unroll_calls=len(rec.get("calls", [])),
                n_h6_population_calls=len(h6),
                episode_seconds=float(rec.get("seconds", np.nan)),
                device_uuid=rec.get("device_uuid"),
                logical_rank=int(rec["scenario"].get("logical_rank", -1)),
                environment_seed=int(rec["scenario"].get("environment_seed", -1)),
            ))
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["task", "episode", "arm"]).reset_index(drop=True)
    return df


def success_matrix(df, task, arms=ARMS):
    """episodes × arms boolean matrix (only scenarios complete in all requested arms)."""
    sub = df[df.task == task].pivot(index="episode", columns="arm", values="success")
    sub = sub.dropna(subset=[a for a in arms if a in sub.columns])
    return sub[[a for a in arms if a in sub.columns]].astype(bool)


def energy_frame(results_dir, tasks=TASKS, arms=("fixed_rank4", "matched_random_fixed_rank4")):
    """One row per (task, episode, arm, H6 population call) with delivered-edit summaries.

    For refined arms: per-call coefficient statistics over the 300 candidates.
    For coupling arms: edited_candidates and requested/realized squared-L2 sums.
    """
    rows = []
    for task in tasks:
        for episode, arm, rec in iter_records(results_dir, task, arms=arms):
            for i, c in enumerate(rec.get("calls", [])):
                if c.get("horizon") != 6 or c.get("candidates", 0) <= 1:
                    continue
                e = c.get("energy", {})
                row = dict(task=task, episode=episode, arm=arm, call_index=i,
                           candidates=int(c["candidates"]), seconds=float(c.get("seconds", np.nan)))
                if "coefficients" in e:
                    coef = np.asarray(e["coefficients"], dtype=float)  # [cand, 4]
                    req = np.asarray(e.get("requested_l2", []), dtype=float)
                    rea = np.asarray(e.get("realized_l2", []), dtype=float)
                    act = np.asarray(e.get("active", []), dtype=bool)
                    row.update(
                        coef_mean=coef.mean(0).tolist(), coef_abs_mean=np.abs(coef).mean(0).tolist(),
                        coef_norm_mean=float(np.linalg.norm(coef, axis=1).mean()),
                        coef_norm_std=float(np.linalg.norm(coef, axis=1).std()),
                        coef_cosine_to_call_mean=float(_mean_cosine(coef)),
                        requested_l2_mean=float(req.mean()) if req.size else np.nan,
                        realized_l2_mean=float(rea.mean()) if rea.size else np.nan,
                        rounding_loss=float(1 - rea.mean() / req.mean()) if req.size and req.mean() > 0 else np.nan,
                        active_fraction=float(act.mean()) if act.size else np.nan)
                else:
                    # Fresh static adapters store population means, not sums.
                    # Missing active counts remain unknown; positive mean energy
                    # alone cannot establish that every candidate was edited.
                    count = int(c["candidates"])
                    if "requested_squared_l2_mean" in e and int(e.get("candidates", -1)) != count:
                        raise ValueError("Energy population size differs from forecast population")
                    row.update(edited_candidates=float(e.get("edited_candidates", np.nan)),
                               requested_sq_l2_sum=float(e.get("requested_squared_l2_sum",
                                   e.get("requested_squared_l2_mean", np.nan) * count)),
                               realized_sq_l2_sum=float(e.get("realized_squared_l2_sum",
                                   e.get("realized_squared_l2_mean", np.nan) * count)))
                rows.append(row)
    return pd.DataFrame(rows)


def _mean_cosine(coef):
    n = np.linalg.norm(coef, axis=1, keepdims=True)
    ok = n[:, 0] > 0
    if ok.sum() < 2:
        return np.nan
    u = coef[ok] / n[ok]
    m = u.mean(0)
    mn = np.linalg.norm(m)
    return float((u @ m / mn).mean()) if mn > 0 else np.nan


def action_divergence(results_dir, task, arm, reference="native"):
    """First planning call at which `arm`'s chosen actions differ from `reference`.

    Returns DataFrame(episode, n_calls, first_divergence (call index or -1 if identical),
    n_divergent_calls). Uses the recorded action hashes; identical hashes mean identical actions.
    """
    ref = {e: r for e, a, r in iter_records(results_dir, task, arms=(reference,))}
    rows = []
    for episode, _, rec in iter_records(results_dir, task, arms=(arm,)):
        if episode not in ref:
            continue
        a = [t["actions_sha256"] for t in rec.get("action_trace", [])]
        b = [t["actions_sha256"] for t in ref[episode].get("action_trace", [])]
        n = min(len(a), len(b))
        diff = [i for i in range(n) if a[i] != b[i]]
        rows.append(dict(task=task, episode=episode, arm=arm, n_calls=n,
                         first_divergence=diff[0] if diff else -1, n_divergent_calls=len(diff),
                         first_divergence_fraction=(diff[0] / n) if diff and n else np.nan))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- statistics
def paired_bootstrap(a, b, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED, family=1, alpha=0.05):
    """Paired scenario-cluster percentile bootstrap of mean(a - b), Bonferroni over `family`.

    a, b: arrays of per-scenario outcomes (bool or float), same length and order.
    Returns dict(difference, lower, upper, family, draws, seed). Success inputs are
    reported in percentage points.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape and a.ndim == 1
    d = a - b
    scale = 100.0 if set(np.unique(np.concatenate([a, b]))) <= {0.0, 1.0} else 1.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(draws, len(d)))
    means = d[idx].mean(1)
    q = alpha / family
    lo, hi = np.quantile(means, [q / 2, 1 - q / 2])
    return dict(difference=float(d.mean() * scale), lower=float(lo * scale), upper=float(hi * scale),
                n=int(len(d)), family=int(family), draws=int(draws), seed=int(seed), scale=scale)


def discordance(a, b):
    """Paired binary outcomes: wins (a & ~b), losses (~a & b), exact two-sided sign-test p."""
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    wins, losses = int((a & ~b).sum()), int((~a & b).sum())
    n = wins + losses
    p = 1.0
    if n:
        from scipy.stats import binomtest
        p = float(binomtest(wins, n, 0.5).pvalue)
    return dict(wins=wins, losses=losses, discordant=n, exact_p=p)


def holm(pvalues):
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def spearman(x, y):
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return dict(rho=np.nan, p=np.nan, n=int(ok.sum()))
    r = spearmanr(x[ok], y[ok])
    return dict(rho=float(r.statistic), p=float(r.pvalue), n=int(ok.sum()))


def classify_regime(contrast_vs_native, contrast_vs_control, useful=USEFUL_GAIN_PP):
    """Pre-registered-style reading of one learned arm on one task.

    contrast dicts need keys difference/lower/upper (pp). Returns one of
    'confirmed' (both lower bounds > 0 and gain >= useful), 'positive_vs_native_only',
    'positive_vs_control_only', 'harmful' (upper < 0 vs native), 'inconclusive'.
    """
    n, c = contrast_vs_native, contrast_vs_control
    if n["lower"] > 0 and c["lower"] > 0 and n["difference"] >= useful:
        return "confirmed"
    if n["lower"] > 0 and c["lower"] <= 0:
        return "positive_vs_native_only"
    if c["lower"] > 0 and n["lower"] <= 0:
        return "positive_vs_control_only"
    if n["upper"] < 0:
        return "harmful"
    return "inconclusive"


def power_pp(n=96, p=0.5, alpha=0.05, family=1):
    """Approximate detectable paired difference (pp) at 80% power for n paired binary
    outcomes with discordance-heavy variance; normal approximation, for orientation only."""
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / family / 2) + norm.ppf(0.8)
    return float(100 * z * np.sqrt(2 * p * (1 - p) / n))


# --------------------------------------------------------------- prior artifacts
def load_offline_contrasts():
    return pd.read_csv(ROOT / "paper/data/all_task_ablation_contrasts.csv")


def offline_effect(task, candidate, control="native", precision="bfloat16", endpoint="proprio_mse_h6"):
    d = load_offline_contrasts()
    q = d[(d.task == {"reach-wall": "reach-wall"}.get(task, task)) & (d.candidate == candidate) & (d.control == control)
          & (d.precision == precision) & (d.endpoint == endpoint)]
    if not len(q):
        return None
    r = q.iloc[0]
    return dict(effect=float(r.error_reduction_percent_of_native), lower=float(r.simultaneous_95_lower),
                upper=float(r.simultaneous_95_upper), result=str(r.statistical_result), n=int(r.independent_lineages))


OFFLINE_ARM_NAME = {"coupling_only": "joint_equal_standardized_energy",
                    "matched_random_coupling": "matched_random_equal_standardized_energy"}


def load_core_panel():
    return read_json(ROOT / "reports/wm-approaches/core-analysis.json")


def load_decision_diagnostic():
    """Parse the two tables of reports/PLANNER_DECISION_DIAGNOSTIC_RESULTS.md."""
    text = (ROOT / "reports/PLANNER_DECISION_DIAGNOSTIC_RESULTS.md").read_text()
    rows = []
    for line in text.splitlines():
        m = re.match(r"\|\s*(reach(?:-wall)?)\s*\|\s*(\w+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", line)
        if m:
            rows.append(dict(task=m.group(1), arm=m.group(2), changed_selections=int(m.group(3)),
                             spearman=float(m.group(4)), top10_overlap=float(m.group(5))))
    return pd.DataFrame(rows)


def load_bank(task):
    """Fitted refined operator bank for a task, or None. Requires torch."""
    candidates = [ROOT / f"artifacts/offline_study/fixed-response-20260908-v1/fits/{task}/operator_bank.pt",
                  ROOT / f"artifacts/offline_study/fresh-campaign-stage-20260912-v1/local/fits/{task}/refined/operator_bank.pt"]
    for path in candidates:
        if path.exists():
            import torch
            return dict(path=str(path), sha256=sha256(path),
                        bank=torch.load(path, map_location="cpu", weights_only=True))
    return None


# ------------------------------------------------------------------- reporting
def md_table(df, floatfmt="{:.2f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(floatfmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def provenance(results_dir, freeze_dir, analysis_dir):
    p = load_freeze(freeze_dir)
    a = load_analysis(analysis_dir)
    return dict(results=str(results_dir), freeze_sha256=sha256(Path(freeze_dir) / "protocol.json"),
                analysis_sha256=sha256(Path(analysis_dir) / "report.json"),
                method=a.get("method"), synthetic=bool(p.get("synthetic", False) or a.get("synthetic", False)))
