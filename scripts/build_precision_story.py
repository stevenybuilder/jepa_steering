"""Reframe the existing reconstruction metric; do not refit or change intervals.

The complete downstream forecast panel remains in pathway_geometry_precision.
This figure displays every task/precision condition for the raw interpolation test.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TABLE = "paper/data/pathway_geometry_geometry_summary.csv"
TASKS = ("reach", "reach-wall", "pusht", "wall", "pointmaze")
COUNTS = (33, 27, 21, 192, 200)
PRECISIONS = ("float32", "bfloat16")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_source(root=ROOT):
    root = Path(root)
    path = root / "paper/data/pathway_geometry.json"
    receipt = json.loads(path.read_text())
    if receipt["status"] != "completed_exploratory_archival_reanalysis":
        raise ValueError("Require completed archival reanalysis")
    tables = {}
    for name, info in receipt["outputs"].items():
        if sha(root / name) != info["sha256"]:
            raise ValueError("Source table hash changed")
        with (root / name).open(newline="") as stream:
            tables[name] = list(csv.DictReader(stream))
        if len(tables[name]) != info["rows"]:
            raise ValueError("Source table row count changed")
    rows = [r for r in tables[TABLE] if r["metric"] == "omitted_activation_mse"
            and r["control"] == "equal_anchor_linear" and r["intervention"] == "cubic"]
    expected = {(t, p) for t in TASKS for p in PRECISIONS}
    if len(rows) != 10 or {(r["task"], r["precision"]) for r in rows} != expected:
        raise ValueError("Require all ten task/precision conditions")
    values = []
    for row in rows:
        task, precision = row["task"], row["precision"]
        if int(row["n"]) != COUNTS[TASKS.index(task)]:
            raise ValueError("Changed lineage count")
        control, intervention = float(row["control_mean"]), float(row["intervention_mean"])
        if not (np.isfinite([control, intervention]).all() and control > 0 and intervention > 0):
            raise ValueError("Invalid reconstruction means")
        ratio = intervention / control
        # Monotone decreasing transform reverses interval endpoints.
        low = 1 - float(row["percent_95_high"]) / 100
        high = 1 - float(row["percent_95_low"]) / 100
        if not np.isclose(ratio, 1 - float(row["percent_of_reference"]) / 100, rtol=1e-9):
            raise ValueError("Inconsistent reported ratio")
        if not (np.isfinite([low, high]).all() and 0 < low <= ratio <= high):
            raise ValueError("Invalid full interval")
        values.append(dict(task=task, precision=precision, ratio=ratio, low=low, high=high))
    provenance = dict(source_receipt_sha256=sha(path), source_tables=receipt["outputs"],
                      figure_source_sha256=sha(__file__), metric="cubic / linear omitted activation MSE",
                      interval="Full marginal paired-bootstrap 95% interval; monotone endpoint transform",
                      scope="All five archival tasks, both precision-specific fits; not a controlled rounding test",
                      downstream_panel="docs/figures/pathway_geometry_precision.png")
    return values, provenance


def make_figure(values):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ink, muted, teal, amber = "#22313a", "#67767e", "#147d83", "#be762c"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5,
                         "pdf.fonttype": 42, "svg.fonttype": "none",
                         "svg.hashsalt": "precision-reconstruction-story-v1"})
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    fig.subplots_adjust(left=.20, right=.94, top=.72, bottom=.24)
    fig.text(.055, .93, "The reconstruction comparison reverses.",
             fontsize=17, weight="bold", color=ink)
    fig.text(.055, .864, "Same four-anchor test. Two precision conditions.", color=muted)
    fig.legend(handles=[Line2D([0], [0], marker="o", color=teal, lw=0, label="FP32 sensitivity"),
                        Line2D([0], [0], marker="s", color=amber, lw=0, label="BF16 primary")],
               loc="upper left", bbox_to_anchor=(.047, .823), ncol=2, frameon=False,
               handletextpad=.25, columnspacing=1.6)
    lookup = {(r["task"], r["precision"]): r for r in values}
    for y, task in enumerate(TASKS):
        pair = [lookup[task, p] for p in PRECISIONS]
        ax.plot([r["ratio"] for r in pair], [y, y], color="#d4dedf", lw=1.7, zorder=1)
        for row, color, marker in zip(pair, (teal, amber), ("o", "s")):
            artist = ax.errorbar(row["ratio"], y,
                                xerr=[[row["ratio"]-row["low"]], [row["high"]-row["ratio"]]],
                                fmt=marker, color=color, capsize=3, ms=5, lw=1.2, zorder=3)
            artist.lines[0].set_gid(f"condition-{task}-{row['precision']}")
    ax.axvline(1, ls=(0, (3, 3)), lw=1, color=muted)
    ax.text(1, 1.025, "Equal error", transform=ax.get_xaxis_transform(), ha="center", color=muted)
    labels = ("Reach", "Reach-Wall", "Push-T", "Wall", "PointMaze")
    ax.set_yticks(range(5), [f"{label}\n(n={n})" for label, n in zip(labels, COUNTS)])
    ax.set_ylim(4.55, -.55)
    ax.set_xscale("log")
    low = min(r["low"] for r in values)
    high = max(r["high"] for r in values)
    ax.set_xlim(min(6e-5, low / 1.5), max(3, high * 1.5))
    ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1], ["0.0001", "0.001", "0.01", "0.1", "1"])
    ax.set_xlabel("Cubic / linear activation MSE · log scale · lower is better", labelpad=11, color=muted)
    ax.grid(axis="x", color="#e4e9eb", lw=.7)
    ax.tick_params(axis="both", which="both", length=0, pad=7, colors=muted)
    ax.minorticks_off()
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(.055, .087, "All 5 tasks · full paired-bootstrap 95% intervals", color=muted)
    fig.text(.055, .037, "Raw reconstruction, not forecast accuracy or robot success.", color=muted)
    return fig


def build(root=ROOT):
    import matplotlib.pyplot as plt
    root = Path(root)
    values, provenance = load_source(root)
    fig = make_figure(values)
    output = root / "docs/figures"
    output.mkdir(parents=True, exist_ok=True)
    description = json.dumps(provenance, sort_keys=True)
    for extension in ("png", "svg", "pdf"):
        metadata = ({"Description": description, "Date": None} if extension == "svg" else
                    {"Subject": description, "CreationDate": None, "ModDate": None} if extension == "pdf" else
                    {"Description": description})
        fig.savefig(output / f"precision_reconstruction_story.{extension}", dpi=240,
                    facecolor="white", metadata=metadata)
    plt.close(fig)
    return provenance


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
