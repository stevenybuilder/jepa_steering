"""Post-confirmation descriptive analyses. No model calls or intervention selection.

The fixed scope is configs/mechanism_followup_20260913.json. Raw replay is optional;
figures can be rebuilt from the committed, verified derived data on CPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "paper/data"
FIG = ROOT / "docs/figures"
PROTOCOL = ROOT / "configs/mechanism_followup_20260913.json"
SUPPORTS = [f"single_block{i}" for i in range(6)] + ["intermediate_blocks2_3", "all_six_blocks"]
LABELS = [f"B{i}" for i in range(6)] + ["B2+B3", "All six"]
TASKS = ("reach", "reach-wall", "pointmaze", "wall")
ARMS = ("fixed_rank4", "matched_random_fixed_rank4")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decompose(calls):
    """Exact nested ANOVA identity in coefficient coordinates, not causal mediation."""
    if not calls:
        raise ValueError("No candidate calls")
    arrays = [np.asarray(c, dtype=np.float64) for c in calls]
    if any(c.shape != (300, 4) or not np.isfinite(c).all() for c in arrays):
        raise ValueError("Expected finite 300 by 4 coefficients in every call")
    means = np.stack([c.mean(axis=0) for c in arrays])
    total = sum(float(np.square(c).sum()) for c in arrays)
    if total <= 0:
        raise ValueError("Zero total coefficient energy")
    within = sum(float(np.square(c - m).sum()) for c, m in zip(arrays, means))
    between = float(300 * np.square(means - means.mean(axis=0)).sum())
    mean_energy = float(300 * len(means) * np.square(means.mean(axis=0)).sum())
    if not np.isclose(total, within + between + mean_energy, rtol=1e-12, atol=1e-10):
        raise ValueError("Nested energy identity failed")
    return {"candidate_centered_fraction": within / total,
            "between_call_fraction": between / total,
            "scenario_mean_fraction": mean_energy / total,
            "common_to_candidates_fraction": (between + mean_energy) / total,
            "total_coefficient_energy": total, "calls": len(arrays)}


def call_fraction(c):
    return decompose([c])["candidate_centered_fraction"]


def bootstrap_mean(values, rng):
    values = np.asarray(values, dtype=float)
    draws = values[rng.integers(0, len(values), size=(20000, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "marginal_95_interval": np.quantile(draws, [.025, .975]).tolist()}


def raw_candidate_analysis(results):
    rows, sources = [], {}
    if sha(ROOT / "reports/fresh-confirmation/report.json") != "8123d71497835fc164f09f5094c430308647a3ee2462c66746baea32633a0b15":
        raise ValueError("Frozen final report changed")
    frozen = json.loads((ROOT / "reports/fresh-confirmation/report.json").read_text())
    for task in TASKS:
        scenarios = sorted((results / task).glob("scenario-*"))
        if len(scenarios) != 96:
            raise ValueError(f"Incomplete scenario set for {task}")
        for scenario in scenarios:
            rp = scenario / "report.json"
            record = json.loads(rp.read_text())
            done = json.loads((scenario / "DONE.json").read_text())
            if sha(rp) != done["report_sha256"]:
                raise ValueError(f"Completion hash mismatch: {rp}")
            # The immutable final report binds each completed scenario report.
            matches = [v for k, v in frozen["source_reports_sha256"].items()
                       if k.endswith(f"/{task}/{scenario.name}")]
            if len(matches) != 1:
                raise ValueError("Missing or ambiguous final scenario binding")
            expected = matches[0]
            if sha(rp) != expected:
                raise ValueError(f"Final report binding mismatch: {rp}")
            for arm in ARMS:
                p = scenario / f"{arm}.json"
                if sha(p) != record["records_sha256"][arm]:
                    raise ValueError(f"Raw arm hash mismatch: {p}")
                raw = json.loads(p.read_text())
                if raw["scientific_efficacy_measurement"] is not True:
                    raise ValueError("Engineering record in scientific set")
                calls = [x for x in raw["calls"] if x["horizon"] == 6 and x["candidates"] == 300]
                coeff = [x["energy"]["coefficients"] for x in calls]
                norm_errors = []
                for c, call in zip(coeff, calls):
                    requested = np.asarray(call["energy"]["requested_l2"], dtype=float)
                    cnorm = np.linalg.norm(np.asarray(c, dtype=float), axis=1)
                    if requested.shape != (300,) or not np.allclose(cnorm, requested, rtol=2e-5, atol=1e-6):
                        raise ValueError("Coefficient norm disagrees with requested activation edit norm")
                    norm_errors.append(float(np.max(np.abs(cnorm - requested) / np.maximum(requested, 1e-12))))
                first_iterations = int(raw["result"]["planning_calls"][0]["iterations"])
                expected_calls = 75 if task in ("reach", "reach-wall") else 30
                first_batches = [x for x in raw["calls"] if x["candidates"] == 300][:first_iterations]
                if len(coeff) != expected_calls or len(first_batches) != first_iterations or any(x["horizon"] != 6 for x in first_batches):
                    raise ValueError("Incomplete first CEM planning call")
                r = decompose(coeff)
                r.update(task=task, arm=arm, scenario=scenario.name,
                         max_relative_coefficient_vs_requested_norm_error=max(norm_errors),
                         first_iteration_fraction=call_fraction(coeff[0]),
                         last_first_plan_iteration_fraction=call_fraction(coeff[first_iterations - 1]))
                rows.append(r)
                sources[f"{task}/{scenario.name}/{arm}.json"] = sha(p)
        print(f"Verified and decomposed {task}: 192 arm records", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(DATA / "candidate_specificity_scenarios.csv", index=False)
    rng, summary = np.random.default_rng(20260913), {}
    quantities = ("candidate_centered_fraction", "common_to_candidates_fraction", "between_call_fraction",
                  "scenario_mean_fraction", "first_iteration_fraction", "last_first_plan_iteration_fraction")
    for task in TASKS:
        summary[task] = {}
        for arm in ARMS:
            f = frame[(frame.task == task) & (frame.arm == arm)]
            summary[task][arm] = {k: bootstrap_mean(f[k], rng) for k in quantities}
            summary[task][arm].update(scenarios=len(f), calls=int(f.calls.sum()))
    out = {"scope": "Exploratory coefficient geometry; not causal mediation or fresh efficacy inference",
           "protocol_sha256": sha(PROTOCOL), "analysis_source_sha256": sha(__file__),
           "raw_sha256": sources, "summary": summary,
           "uncertainty": "Marginal descriptive scenario bootstrap; not simultaneous discovery intervals"}
    (DATA / "candidate_specificity.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    return summary


def layer_reanalysis():
    """Recount each saved lineage aggregate, then verify the existing CSV export."""
    manifest = json.loads((DATA / "all_task_ablation_sources.json").read_text())
    exported = pd.read_csv(DATA / "all_task_ablation_metrics.csv")
    rows = []
    for s in manifest["forecast_sources"]:
        if s["category"] != "distribution_layer":
            continue
        path = ROOT / s["source"]
        if sha(path) != s["sha256"]:
            raise ValueError(f"Layer source hash mismatch: {path}")
        report = json.loads(path.read_text())
        groups = report["aggregation"]["per_arm"]
        ids = {r["lineage_group"] for r in groups["native"]["per_lineage_group"]}
        n = s["independent_lineage_groups"]
        if len(ids) != n:
            raise ValueError("Wrong native lineage count")
        for arm in ["native", "zero_dose"] + SUPPORTS + ["matched_random_" + a for a in SUPPORTS]:
            records = groups[arm]["per_lineage_group"]
            if len(records) != n or {r["lineage_group"] for r in records} != ids:
                raise ValueError("Unpaired layer lineage set")
        for kind in ("proprio_mse", "visual_mse"):
            for horizon in range(1, 7):
                endpoint = f"{kind}_h{horizon}"
                def mean(arm):
                    value = float(np.mean([r["metrics"][endpoint] for r in groups[arm]["per_lineage_group"]]))
                    f = exported[(exported.source_id == s["id"]) & (exported.arm == arm) & (exported.endpoint == endpoint)]
                    if len(f) != 1 or not np.isclose(value, f.iloc[0].lineage_weighted_mean, rtol=1e-10, atol=1e-14):
                        raise ValueError("Recount disagrees with published aggregate")
                    return value
                native = mean("native")
                if not np.isclose(native, mean("zero_dose"), rtol=0, atol=1e-14):
                    raise ValueError("Zero-dose identity failed")
                for arm in SUPPORTS:
                    learned, random = mean(arm), mean("matched_random_" + arm)
                    rows.append(dict(task=s["task"], precision=s["precision"], arm=arm,
                        endpoint=kind, horizon=horizon, n=n, native_mean=native,
                        learned_mean=learned, random_mean=random,
                        reduction_vs_native_percent=100 * (native - learned) / native,
                        advantage_vs_random_percent=100 * (random - learned) / native,
                        source_sha256=s["sha256"]))
    frame = pd.DataFrame(rows)
    if len(frame) != 576:
        raise ValueError(f"Expected 576 layer cells, received {len(frame)}")
    frame.to_csv(DATA / "layer_mechanism_grid.csv", index=False)
    print("Verified 6 source reports and recounted 576 layer/modality/horizon cells", flush=True)
    return frame


def save_figure(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight", facecolor="white")
    svg = FIG / f"{name}.svg"
    svg.write_text("\n".join(x.rstrip() for x in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def plots():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    frame = pd.read_csv(DATA / "layer_mechanism_grid.csv")
    # Do not share color scales across modalities with very different numerical effects.
    for precision in ("bfloat16", "float32"):
        for metric, suffix, heading in (
            ("reduction_vs_native_percent", "native", "Forecast correction by intervention site"),
            ("advantage_vs_random_percent", "random", "Does fitted geometry beat a matched random subspace?")):
            fig, axes = plt.subplots(3, 2, figsize=(12, 12), layout="constrained")
            fig.suptitle(f"{heading}\n{precision.upper()} · recorded-action development evaluation", fontsize=17)
            for col, kind in enumerate(("proprio_mse", "visual_mse")):
                subset = frame[(frame.precision == precision) & (frame.endpoint == kind)]
                vmax = max(float(subset[metric].abs().max()), .01)
                for row, task in enumerate(("reach", "reach-wall", "pusht")):
                    f = subset[subset.task == task]
                    matrix = f.pivot(index="arm", columns="horizon", values=metric).loc[SUPPORTS].values
                    ax = axes[row, col]
                    im = ax.imshow(matrix, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
                    ax.set(xticks=range(6), xticklabels=[f"H{i}" for i in range(1, 7)],
                           yticks=range(8), yticklabels=LABELS,
                           title=f"{task} · {'proprioceptive' if col == 0 else 'visual'} embedding MSE · n={int(f.n.iloc[0])}")
                    ax.axvline(1.5, color="#333333", lw=1, ls="--")
                    for (y, x), value in np.ndenumerate(matrix):
                        ax.text(x, y, f"{value:+.2f}", ha="center", va="center", fontsize=8,
                                color="white" if abs(value) > .55*vmax else "#111111")
                fig.colorbar(im, ax=axes[:, col], shrink=.62, label="% of native MSE · positive = better")
            fig.supxlabel("Rank-one, equal requested energy, edit at H3 · H1–H2 are pre-edit checks\nLayer effect, not attention or physical-task success; delivered BF16 dose can differ", fontsize=10)
            save_figure(fig, f"layer_mechanism_{precision}_{suffix}")
    summary = json.loads((DATA / "candidate_specificity.json").read_text())["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    fig.suptitle("How much of the edit distinguishes candidate plans?", fontsize=17)
    colors = ("#187c80", "#b2b7c0")
    for i, arm in enumerate(ARMS):
        x = np.arange(4) + (i-.5)*.34
        vals = [100*summary[t][arm]["candidate_centered_fraction"]["mean"] for t in TASKS]
        intervals = np.array([summary[t][arm]["candidate_centered_fraction"]["marginal_95_interval"] for t in TASKS]).T*100
        axes[0].bar(x, vals, .32, color=colors[i], label=("Refined" if i == 0 else "Calibrated random subspace"))
        axes[0].errorbar(x, vals, yerr=np.array([vals-intervals[0],intervals[1]-vals]), fmt="none", color="#333333", capsize=3)
        for pos, val in zip(x, vals):
            axes[0].text(pos, val+.8, f"{val:.1f}", ha="center", fontsize=9)
    axes[0].set(xticks=range(4), xticklabels=["Reach", "Reach-Wall", "PointMaze", "Wall"], ylabel="Candidate-centered coefficient energy (%)", ylim=(0,23))
    axes[0].legend(fontsize=8)
    for i, (key, label, color) in enumerate((("scenario_mean_fraction", "Scenario mean", "#187c80"),
                 ("between_call_fraction", "Changing mean across calls", "#79bdba"),
                 ("candidate_centered_fraction", "Differences within candidate batch", "#ebae69"))):
        bottom = np.array([sum(summary[t][ARMS[0]][k]["mean"] for k in ("scenario_mean_fraction", "between_call_fraction", "candidate_centered_fraction")[:i]) for t in TASKS])*100
        axes[1].bar(range(4), [100*summary[t][ARMS[0]][key]["mean"] for t in TASKS], bottom=bottom, color=color, label=label)
    axes[1].set(xticks=range(4), xticklabels=["Reach", "Reach-Wall", "PointMaze", "Wall"], ylabel="Refined coefficient energy (%)", ylim=(0,100))
    axes[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5,-.12))
    fig.supxlabel("96 scenarios/task · scenario means and marginal 95% bootstrap intervals · descriptive, not a score-ranking test", fontsize=9)
    save_figure(fig, "candidate_specificity")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, help="Optional verified raw results-v2 tree")
    ap.add_argument("--recount-layers", action="store_true")
    args = ap.parse_args()
    if args.recount_layers:
        layer_reanalysis()
    if args.results:
        raw_candidate_analysis(args.results)
    plots()
