"""Readable plots for the complete, fixed CEM extension; no partial-case plots.

Primary curves use the 56 new development contexts only. The original eight
remain in the source tables, not silently pooled into the displayed cohort.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from build_publication_figures import ROOT, INK, MUTED, TEAL, AMBER, clean, save, sha, style

TASKS = ("reach", "reach-wall")
ARMS = ("fixed_rank4", "matched_random_fixed_rank4")
LABELS = ("Reach", "Reach-Wall")
NAMES = ("Intervention (rank-four)", "Calibrated random")
PROTOCOL_SHA = "4766488e4907dff62f588894845f5b5aecf084e2c47a5ab4a1605cc2f07463db"


def validate_frames(iterations, prefixes, summaries):
    for frame, extra in ((iterations, ["iteration"]), (prefixes, [])):
        keys = ["task", "episode", "arm", *extra]
        expected = {(t, e, a, *([i] if extra else [])) for t in TASKS
                    for e in range(32) for a in ARMS for i in (range(15) if extra else [0])}
        if frame.duplicated(keys).any() or set(frame[keys].itertuples(index=False, name=None)) != expected:
            raise ValueError("Require complete64 source cases, both arms and every iteration")
        expected_cohort = np.where(frame.episode < 4, "initial8", "extension56")
        if not np.array_equal(frame.cohort.to_numpy(), expected_cohort):
            raise ValueError("Initial and extension cohorts were mixed")
    curves = summaries[summaries.population == "extension56"]
    for metric in ("proposal_mean_delta_rms", "proposal_entropy_delta_nats"):
        rows = curves[curves.metric == metric]
        keys = ["task", "arm", "iteration"]
        expected = {(t, a, i) for t in TASKS for a in ARMS for i in range(15)}
        if rows.duplicated(keys).any() or set(rows[keys].itertuples(index=False, name=None)) != expected:
            raise ValueError("Missing extension curve cell")
        for row in rows.itertuples(index=False):
            raw = iterations[(iterations.cohort == "extension56") & (iterations.task == row.task)
                             & (iterations.arm == row.arm) & (iterations.iteration == row.iteration)][metric].to_numpy()
            if row.n != 28 or len(raw) != 28 or not np.isfinite(raw).all():
                raise ValueError("Require all28 finite scenario measurements per curve cell")
            if not np.allclose([row.mean, row.scenario_se], [raw.mean(), raw.std(ddof=1)/np.sqrt(28)], rtol=1e-10, atol=1e-12):
                raise ValueError("Displayed mean/SE differs from the scenario-level data")
    values = prefixes.selected_prefix_delta_rms.to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid selected-prefix RMS")


def load(root=ROOT):
    root = Path(root)
    receipt_path = root/"paper/data/cem_expansion_summary.json"
    receipt = json.loads(receipt_path.read_text())
    required = {"status": "complete56_extension_with_separate_initial8", "primary_cases": 56,
                "primary_n_per_task": 28, "initial_cases": 8, "complete_full_raw_preservation": True,
                "physical_outcomes_measured": False, "fresh_confirmation": False,
                "protocol_sha256": PROTOCOL_SHA}
    if any(receipt.get(key) != value for key, value in required.items()):
        raise ValueError("CEM extension is not complete and fully preserved under its fixed protocol")
    required_tables = {f"paper/data/cem_expansion_{name}.csv" for name in
                       ("iteration_cases", "selected_prefix_cases", "iteration_summary")}
    if not required_tables.issubset(receipt.get("outputs", {})):
        raise ValueError("CEM plot tables are missing source bindings")
    for name, info in receipt["outputs"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts or sha(root/name) != info["sha256"]:
            raise ValueError("Public CEM table identity changed")
    frames = [pd.read_csv(root/f"paper/data/cem_expansion_{name}.csv") for name in
              ("iteration_cases", "selected_prefix_cases", "iteration_summary")]
    validate_frames(*frames)
    return receipt, frames, sha(receipt_path)


def provenance(receipt, receipt_sha):
    return dict(cem_plot_source_sha256=sha(__file__), receipt_sha256=receipt_sha,
                source_tables=receipt["outputs"], plotted_cohort="extension56",
                n_per_task=28, uncertainty="one scenario SE; not a confidence interval or training-seed variation",
                scope="Same-device native comparator; all cases and both fixed edit arms; not physical efficacy")


def curve_figure(summary, metric, title, ylabel, name, meta):
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.8), sharey=True)
    fig.subplots_adjust(left=.115, right=.965, top=.69, bottom=.23, wspace=.19)
    fig.text(.055, .94, title, fontsize=18, weight="bold", color=INK)
    fig.text(.055, .875, "Same initial candidate actions. Separate adaptive searches.", fontsize=11, color=MUTED)
    selected = summary[(summary.population == "extension56") & (summary.metric == metric)]
    for ax, task, label in zip(axes, TASKS, LABELS):
        for arm, color in zip(ARMS, (TEAL, AMBER)):
            rows = selected[(selected.task == task) & (selected.arm == arm)].sort_values("iteration")
            x, mean, se = rows.iteration.to_numpy()+1, rows["mean"].to_numpy(), rows.scenario_se.to_numpy()
            ax.plot(x, mean, color=color, lw=2.2)
            ax.fill_between(x, mean-se, mean+se, color=color, alpha=.15, lw=0)
        ax.axhline(0, color=MUTED, lw=.8, ls="--")
        ax.set_title(label, loc="left", fontsize=13, weight="bold", color=INK, pad=10)
        ax.set(xticks=[1, 5, 10, 15], xlim=(1, 15), xlabel="CEM iteration")
        clean(ax)
    axes[0].set_ylabel(ylabel)
    fig.legend([Line2D([0], [0], color=c, lw=2.2) for c in (TEAL, AMBER)], NAMES,
               loc="upper left", bbox_to_anchor=(.055, .82), ncol=2, frameon=False, fontsize=11)
    fig.text(.055, .096, "56 new development contexts · n=28/task · means ±1 scenario SE", fontsize=10.5, color=INK)
    fig.text(.055, .044, "Concurrent unsteered reference. Original eight cases excluded. Different does not mean better.", fontsize=9.2, color=MUTED)
    save(fig, name, meta | {"metric": metric})


def prefixes_figure(prefixes, meta):
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.8), sharey=True)
    fig.subplots_adjust(left=.115, right=.965, top=.77, bottom=.24, wspace=.19)
    fig.text(.055, .94, "How much does the returned action plan change?", fontsize=17, weight="bold", color=INK)
    fig.text(.055, .875, "Every paired context, not just the mean.", fontsize=11, color=MUTED)
    selected = prefixes[prefixes.cohort == "extension56"]
    for ax, task, label in zip(axes, TASKS, LABELS):
        grid = selected[selected.task == task].pivot(index="episode", columns="arm", values="selected_prefix_delta_rms").sort_index()
        for left, right in grid[list(ARMS)].itertuples(index=False, name=None):
            ax.plot([0, 1], [left, right], color="#cbd4d8", lw=.65, alpha=.65, zorder=1)
        for i, (arm, color) in enumerate(zip(ARMS, (TEAL, AMBER))):
            values = grid[arm].to_numpy()
            ax.scatter(np.full(28, i), values, color=color, alpha=.65, s=16, zorder=2)
            ax.errorbar(i+.12, values.mean(), yerr=values.std(ddof=1)/np.sqrt(28), fmt="D",
                        color=INK, ms=5, capsize=4, lw=1.2, zorder=3)
        ax.set_title(label, loc="left", fontsize=13, weight="bold", color=INK, pad=10)
        ax.set(xticks=[0, 1], xticklabels=["Intervention", "Calibrated random"], xlim=(-.3, 1.35))
        clean(ax)
    axes[0].set_ylabel("Prefix RMS difference from unsteered\n60 model-action coordinates")
    fig.text(.055, .104, "All 28 paired contexts/task. Black diamonds: means ±1 scenario SE.", fontsize=10.5, color=INK)
    fig.text(.055, .048, "Returned three-step plans; this figure measures model-action differences, not robot displacement.", fontsize=9.2, color=MUTED)
    save(fig, "cem_expansion_prefixes", meta | {"metric": "selected_prefix_delta_rms", "all_tails_retained": True})


if __name__ == "__main__":
    receipt, (iterations, prefixes, summary), receipt_sha = load()
    style()
    meta = provenance(receipt, receipt_sha)
    curve_figure(summary, "proposal_mean_delta_rms", "Small forecast edits. Different search paths.",
                 "Proposal-mean RMS from unsteered\n120 model-action coordinates", "cem_expansion_search", meta)
    curve_figure(summary, "proposal_entropy_delta_nats", "Does steering expand or contract the search?",
                 "Proposal entropy change from unsteered\nPreclip diagonal Gaussian (nats)", "cem_expansion_entropy", meta)
    prefixes_figure(prefixes, meta)
    print("PASS: complete56 curves, source-bound values and scenario SE; all paired endpoints retained")
