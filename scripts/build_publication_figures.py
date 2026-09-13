"""Readable README figures from complete, provenance-bound measurements.

These are presentation changes, not a new fit or replacement of paired inference.
Benchmark maxima are explicitly post hoc across all seven edited arms. Published
SD and local episode SE are distinct; neither adjusts for maximum selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

from build_comparison_figures import load_data, display
from check_public_results import ARMS, best_observed_edit

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, TEAL, AMBER, BLUE = "#24343e", "#63737c", "#247f85", "#bd7836", "#4b72a3"
TASKS = ("reach", "reach-wall", "pointmaze", "wall")
LABELS = {"reach": "Reach", "reach-wall": "Reach-Wall", "pointmaze": "PointMaze", "wall": "Wall"}
BENCHMARK_LOCAL_LABELS = ("Unsteered", "Best edit")
EDIT_LABELS = {
    "fixed_rank4": "Four-direction", "matched_random_fixed_rank4": "Random subspace",
    "coupling_only": "Equal-budget visual + action", "matched_random_coupling": "Random visual + action",
    "joint": "Unscaled joint", "visual_only": "Visual-only", "action_condition_only": "Action-only",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.labelsize": 11, "xtick.labelsize": 10,
                         "ytick.labelsize": 10, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "svg.hashsalt": "jepa-publication-v2"})


def clean(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#cad4d9")
    ax.tick_params(length=0, pad=6, colors=MUTED)
    ax.grid(axis="y", color="#e5ebee", linewidth=.65)
    ax.set_axisbelow(True)


def save(fig, name, provenance):
    description = json.dumps({"generator_sha256": sha(__file__), **provenance}, sort_keys=True)
    for ext in ("png", "svg", "pdf"):
        metadata = ({"Description": description, "Date": None} if ext == "svg" else
                    {"Subject": description, "CreationDate": None, "ModDate": None} if ext == "pdf" else
                    {"Description": description})
        fig.savefig(ROOT/f"docs/figures/{name}.{ext}", dpi=240, facecolor="white", metadata=metadata)
    plt.close(fig)


def episode_se(percent, n):
    if n < 2 or not 0 <= percent <= 100:
        raise ValueError("Invalid binary evaluation inputs")
    p = percent/100
    return 100*np.sqrt(p*(1-p)/n)


def benchmark():
    source, report = load_data()
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 7.3), sharey=True)
    fig.subplots_adjust(left=.085, right=.975, top=.78, bottom=.18, hspace=.65, wspace=.23)
    fig.text(.055, .94, "Best observed activation edit by task", fontsize=19, weight="bold", color=INK)
    fig.text(.055, .89, "Highest success among seven edit arms · same 96 unseen scenarios per task", fontsize=11, color=MUTED)
    names = ["DINO-WM", "JEPA-WM", *BENCHMARK_LOCAL_LABELS]
    colors = ["#d7ddd9", "#b0bcb6", BLUE, TEAL]
    measures = []
    for ax, task in zip(axes.flat, TASKS):
        j = source["task_order"].index(task)
        values = [r["values"][j] for r in source["author_rows"][:2]]
        errors = [r["published_dispersion"][j] for r in source["author_rows"][:2]]
        n = report["results"][task]["n"]
        if n != source["fresh_source"]["n_per_task_arm"]:
            raise ValueError("Benchmark cohort size disagrees with provenance")
        selected = best_observed_edit(report, task)
        local = [report["results"][task]["success_percent"]["native"], selected["success_percent"]]
        local_errors = [episode_se(v, n) for v in local]
        values += local
        errors += local_errors
        ax.axvspan(-.55, 1.5, color="#f4f6f3", zorder=0)
        ax.bar(np.arange(4), values, width=.65, color=colors, edgecolor="white", linewidth=.5,
               yerr=errors, capsize=3, error_kw={"elinewidth": .9, "capthick": .9, "ecolor": INK})
        for i, (v, err) in enumerate(zip(values, errors)):
            ax.text(i, v+err+2.1, display(v), ha="center", fontsize=9.7, color=INK)
        ax.axvline(1.5, color="#bcc9c2", lw=.7, ls=(0, (3, 3)))
        ax.set_title(LABELS[task], loc="left", fontsize=12.5, weight="bold", color=INK, pad=30)
        selected_label = " / ".join(EDIT_LABELS[arm] for arm in selected["arms"])
        if len(selected["arms"]) > 1:
            selected_label += " (tie)"
        ax.text(0, 1.035, selected_label, transform=ax.transAxes, fontsize=9, color=TEAL, va="bottom")
        ax.text(.5, 109, "Published", ha="center", fontsize=9, color=MUTED)
        ax.text(2.5, 109, "Our evaluation", ha="center", fontsize=9, color=INK)
        ax.set(ylim=(0, 116), xlim=(-.6, 3.6), xticks=np.arange(4), xticklabels=names, yticks=[0, 25, 50, 75, 100])
        clean(ax)
        ax.tick_params(axis="x", labelsize=9)
        for tick, color in zip(ax.get_xticklabels()[2:], (BLUE, TEAL)):
            tick.set_color(color)
        measures.append(dict(task=task, values=values, error_bars=errors,
                             labels=names, unsteered_arm="native", selected_arms=selected["arms"], n=n,
                             published="reported late-epoch SD", local="one Bernoulli episode SE, n96"))
    axes[0, 0].set_ylabel("Success (%)")
    axes[1, 0].set_ylabel("Success (%)")
    fig.text(.055, .105, "Best edit = post-hoc taskwise maximum; not one fixed method. Winning edits are named above.", fontsize=9.5, color=INK)
    fig.text(.055, .066, "Error bars: published SD; local ±1 episode SE, not adjusted for selecting the maximum.", fontsize=9.5, color=MUTED)
    fig.text(.055, .030, "Published models are external context. Full results retain all arms, ties and paired comparisons.", fontsize=9.5, color=MUTED)
    save(fig, "benchmark_readable", dict(source_sha256=sha(ROOT/"paper/data/benchmark_comparison_sources.json"),
         report_sha256=source["fresh_source"]["sha256"], values=measures,
         selection_pool=[arm for arm in ARMS if arm != "native"],
         scope="Post-hoc observed maximum across all seven activation edits, including randomized comparators; every tie retained. Not a confirmed fixed method. Episode SE is not selection-adjusted. Frozen paired inference unchanged."))


def load_margins(root=ROOT):
    root = Path(root)
    path = root/"paper/data/decision_geometry_scenarios.csv"
    if sha(path) != "0e4d381f4e9b9fe8b0b2d81d482fdf58b6334a74e5d0cd7550995a3fa0abb604":
        raise ValueError("Archived scenario table changed")
    receipt = json.loads((root/"paper/data/decision_geometry.json").read_text())
    frame = pd.read_csv(path)
    expected = {(t, e, a) for t in TASKS[:2] for e in range(96)
                for a in ("refined", "random_refined", "coupling", "random_coupling")}
    if len(frame) != 768 or set(frame[["task", "episode", "arm"]].itertuples(index=False, name=None)) != expected:
        raise ValueError("Require all 192 contexts and four edited arms")
    if not np.isfinite(frame.range_to_margin).all() or (frame.range_to_margin <= 0).any():
        raise ValueError("Ratio cannot be placed on the registered log axis")
    for row in receipt["summaries"]:
        block = frame[(frame.task == row["task"]) & (frame.arm == row["arm"])]
        if int(block.changed.sum()) != row["changed"] or int(block.no_flip_certified.sum()) != row["no_flip_certified"]:
            raise ValueError("Counts disagree with source summary")
    if not np.array_equal((frame.range_to_margin < 1).astype(int), frame.no_flip_certified):
        raise ValueError("Margin certificate identity failed")
    return frame


def margins():
    frame = load_margins()
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 5.3), sharex=True, sharey=True)
    fig.subplots_adjust(left=.095, right=.97, top=.63, bottom=.25, wspace=.16)
    fig.text(.055, .94, "Score changes rarely displace the best plan", fontsize=18, weight="bold", color=INK)
    fig.text(.055, .877, "Below 1: the original winning candidate is guaranteed to stay first.", fontsize=11, color=MUTED)
    arms = ("refined", "random_refined", "coupling", "random_coupling")
    names = ("Learned edit", "Random subspace", "Visual + action", "Random visual + action")
    colors = (TEAL, TEAL, AMBER, AMBER)
    styles = ("-", "--", "-", "--")
    lo = 10**np.floor(np.log10(frame.range_to_margin.min()))
    hi = 10**np.ceil(np.log10(frame.range_to_margin.max()))
    for ax, task in zip(axes, TASKS[:2]):
        ax.axvspan(lo, 1, color="#edf5f3")
        ax.axvline(1, color=INK, lw=1, ls=(0, (3, 3)))
        for arm, color, ls in zip(arms, colors, styles):
            values = np.sort(frame[(frame.task == task) & (frame.arm == arm)].range_to_margin)
            ax.step(np.r_[lo, values, hi], np.r_[0, np.arange(1, 97)/96*100, 100], where="post", color=color, ls=ls, lw=1.8)
        ax.set_title(LABELS[task], loc="left", fontsize=12.5, weight="bold", color=INK, pad=12)
        ax.set_xscale("log")
        ax.set(xlim=(lo, hi), ylim=(0, 102), yticks=[0, 25, 50, 75, 100])
        ax.set_xlabel("Score-change range / original winner's lead", fontsize=10.5, labelpad=8)
        clean(ax)
    axes[0].set_ylabel("Cumulative scenarios (%)")
    fig.legend([Line2D([0], [0], color=c, lw=2, ls=s) for c, s in zip(colors, styles)], names,
               loc="upper left", bbox_to_anchor=(.045, .82), ncol=2, frameon=False,
               fontsize=10, columnspacing=1.3, handlelength=1.8)
    refined = frame[frame.arm == "refined"]
    fig.text(.055, .092, f"Learned edit: {int(refined.no_flip_certified.sum())}/192 certified unchanged; {int(refined.changed.sum())}/192 observed winner changes.", fontsize=10.5, weight="bold", color=TEAL)
    fig.text(.055, .04, "96 contexts/task · same 300 actions per comparison · initial population, not an entire CEM search", fontsize=9.5, color=MUTED)
    save(fig, "decision_margin_story", dict(source_sha256=sha(ROOT/"paper/data/decision_geometry_scenarios.csv"),
         scope="Complete empirical cumulative distributions; all arms, all observations; algebraic certificate, no statistical test"))


def architecture():
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    fig.subplots_adjust(left=.03, right=.98, top=.97, bottom=.02)
    ax.set(xlim=(0, 10), ylim=(0, 6))
    ax.axis("off")
    def box(x, y, w, h, label, color="#f3f6f7", edge="#d2dde1", size=10):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.035,rounding_size=.07", facecolor=color, edgecolor=edge, lw=1, zorder=3 if label else 0))
        ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=size, color=INK, linespacing=1.3, zorder=4)
    def arrow(points, color=MUTED):
        if len(points)>2:
            ax.plot(*zip(*points[:-1]), color=color, lw=1.15, zorder=2)
        ax.annotate("", xy=points[-1], xytext=points[-2], arrowprops=dict(arrowstyle="->", color=color, lw=1.15), zorder=2)
    ax.text(.1, 5.63, "How JEPA-WM plans with predicted futures", fontsize=18, weight="bold", color=INK)
    ax.text(.1, 5.22, "Frozen JEPA-WM  ·  alternative intervention sites at imagined step H3", fontsize=10.5, color=MUTED)
    box(.12, 3.97, 2.02, .65, "Image + robot state")
    box(.12, 2.94, 2.02, .65, "Frozen encoders")
    arrow([(1.13, 3.93), (1.13, 3.63)])
    box(2.72, 2.77, 4.44, 1.7, "", color="#f6f8f7")
    ax.text(4.94, 4.27, "Action-conditioned predictor", ha="center", fontsize=11, weight="bold", color=INK)
    xs = np.linspace(2.94, 6.35, 6)
    for i, x in enumerate(xs):
        box(x, 3.22, .49, .59, f"B{i}", color="#e5eef6" if i == 3 else "white", edge=BLUE if i == 3 else "#d2dde1", size=9.5)
        if i<5: arrow([(x+.52, 3.515), (xs[i+1]-.055, 3.515)])
    arrow([(2.18, 3.265), (2.47, 3.265), (2.47, 3.515), (2.90, 3.515)])
    box(2.27, 3.69, .34, .35, "V", "#e4f3ee", TEAL, 10)
    ax.text(2.25, 4.20, "visual", fontsize=8.5, color=TEAL)
    box(xs[3]+.51, 3.91, .27, .26, "R", "#e8eef7", BLUE, 9)
    ax.plot([xs[3]+.59, xs[3]+.59], [3.55, 3.88], color=BLUE, lw=1, zorder=2)
    # Preserve topological order: R follows B3 and precedes B4; compact marker
    # is offset above the connecting edge to avoid becoming a seventh block.
    box(xs[3]+.055, 2.87, .35, .28, "A", "#faf0df", AMBER, 9)
    box(7.75, 3.13, 2.05, .78, "Predicted future\nembeddings")
    arrow([(7.20, 3.515), (7.71, 3.515)])
    box(7.75, 1.70, 2.05, .80, "Distance to\nencoded goal")
    arrow([(8.78, 3.08), (8.78, 2.54)])
    box(.12, 1.71, 2.02, .78, "300 candidate\naction sequences")
    box(2.83, 1.71, 2.17, .78, "Action encoder")
    arrow([(2.18, 2.10), (2.79, 2.10)])
    ax.plot([3.92, 3.92], [2.53, 2.71], color=AMBER, lw=1.1, zorder=2)
    ax.plot([xs[0]+.245, xs[-1]+.245], [2.71, 2.71], color=AMBER, lw=1.1, zorder=2)
    for x in xs:
        arrow([(x+.245, 2.71), (x+.245, 3.17)], AMBER)
    ax.text(5.20, 2.53, "every block", fontsize=8.5, color=MUTED)
    box(4.60, .40, 2.30, .72, "CEM: sample around\nthe best ten plans", color="#eaf4f2", edge=TEAL)
    arrow([(8.78, 1.66), (8.78, 1.35), (5.75, 1.35), (5.75, 1.16)])
    arrow([(4.56, .76), (1.13, .76), (1.13, 1.67)])
    ax.text(2.85, .87, "resample + rescore", ha="center", fontsize=9, color=TEAL)
    box(7.75, .40, 2.05, .72, "Start executing\nObserve + replan", size=9.5)
    arrow([(6.94, .76), (7.71, .76)])
    save(fig, "architecture_readable", dict(scope="Inference schematic, not result. V=visual input only; A=B3 action condition; R=B3 output. Alternative arms; action conditioning reaches all blocks. No model weights updated."))


def geometry():
    receipt = json.loads((ROOT/"paper/data/controlled_geometry_summary.json").read_text())
    if receipt["cohort"] != 64 or not receipt["all_cloud_verified"] or not receipt["all_gram_audits_passed"]:
        raise ValueError("Controlled geometry cohort not complete")
    for name, info in receipt["outputs"].items():
        if sha(ROOT/"paper/data"/name) != info["sha256"]:
            raise ValueError("Controlled geometry table changed")
    table = pd.read_csv(ROOT/"paper/data/controlled_geometry_summary.csv")
    table = table[table.metric == "log_ratio"]
    if len(table) != 36 or not (table.n == 32).all():
        raise ValueError("Need six blocks, three conditions, two tasks")
    conditions = list(table.condition.unique())
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.6), sharey=True)
    fig.subplots_adjust(left=.11, right=.975, top=.67, bottom=.22, wspace=.17)
    fig.text(.055, .94, "Precision changes what looks locally curved", fontsize=18, weight="bold", color=INK)
    fig.text(.055, .876, "Same weights, same action perturbations, every predictor block.", fontsize=11, color=MUTED)
    # Exact source condition names are bound below; never guess a missing arm.
    if set(conditions) != {"float32", "float32_field_bfloat16_roundtrip", "bfloat16"}:
        raise ValueError(f"Unexpected numerical conditions: {conditions}")
    conditions = ["float32", "float32_field_bfloat16_roundtrip", "bfloat16"]
    labels = ["FP32", "Output rounding", "BF16 computation"]
    for ax, task in zip(axes, TASKS[:2]):
        for condition, color in zip(conditions, (TEAL, "#927cad", AMBER)):
            rows = table[(table.task == task) & (table.condition == condition)].sort_values("layer")
            if list(rows.layer) != list(range(6)):
                raise ValueError("Missing layer")
            ax.plot(rows.layer, np.exp(rows["mean"]), color=color, lw=2, marker="o", ms=4)
            ax.fill_between(rows.layer, np.exp(rows.marginal_95_low), np.exp(rows.marginal_95_high), color=color, alpha=.15, lw=0)
        ax.set_title(LABELS[task], loc="left", fontsize=12.5, weight="bold", color=INK, pad=12)
        ax.axhline(1, color=MUTED, lw=.8, ls="--")
        ax.set_yscale("log")
        ax.set(ylim=(.0002, 2), xticks=range(6), xticklabels=[f"B{i}" for i in range(6)], yticks=[.001, .01, .1, 1], yticklabels=["0.001", "0.01", "0.1", "1"])
        ax.set_xlabel("Predictor block")
        clean(ax)
    axes[0].set_ylabel("Cubic / linear reconstruction error")
    fig.legend([Line2D([0], [0], color=c, lw=2, marker="o", ms=4) for c in (TEAL, "#927cad", AMBER)], labels,
               loc="upper left", bbox_to_anchor=(.055, .82), ncol=3, frameon=False, fontsize=10, columnspacing=1.4)
    fig.text(.055, .092, "Below 1: cubic is better. BF16 reverses the FP32 comparison at every block.", fontsize=10.5, color=INK)
    fig.text(.055, .040, "32 contexts/task · geometric means, full marginal 95% bands · activation reconstruction, not robot success", fontsize=9.2, color=MUTED)
    save(fig, "geometry_control_story", dict(source_receipt_sha256=sha(ROOT/"paper/data/controlled_geometry_summary.json"),
         source_tables=receipt["outputs"], scope="All six layers/three numerical conditions. Geometric mean ratios from saved mean logs. Output rounding does not quantitatively reproduce actual BF16 computation."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=["benchmark", "margins", "architecture", "geometry"])
    args = parser.parse_args()
    style()
    for name in args.only or ["benchmark", "margins", "architecture", "geometry"]:
        globals()[name]()
        print(f"Rendered {name}")
