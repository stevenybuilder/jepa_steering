"""README hero from the complete, source-bound eight-case CEM diagnostic.

No simulation, new statistics, selected scenarios, or confidence-band filtering.
Every individual trajectory is rendered on a shared axis; bold lines are means.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("reach", "reach-wall")
ARMS = ("fixed_rank4", "matched_random_fixed_rank4")
METRIC = "proposal_mean_delta_rms"
TABLES = {f"paper/data/cem_steering_{name}.csv" for name in (
    "iteration_cases", "iteration_summary", "selected_prefix_cases",
    "selected_prefix_summary", "shared_iteration0_summary")}
INK, MUTED, TEAL, AMBER = "#22313a", "#67767e", "#147d83", "#be762c"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_source(root=ROOT):
    root = Path(root)
    receipt_path = root/"paper/data/cem_steering_summary.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "complete_fixed8_instrumentation_only" or receipt.get("scenarios") != 8 or receipt.get("n_per_task") != 4:
        raise ValueError("Hero requires the complete fixed-eight source receipt")
    identities = [(s["task"],s["episode"]) for s in receipt["sources"]]
    if len(identities) != 8 or set(identities) != {(t,e) for t in TASKS for e in range(4)}:
        raise ValueError("Receipt contains missing or duplicate scenarios")
    if set(receipt["outputs"]) != TABLES:
        raise ValueError("Require all five bound public CEM tables")
    tables = {}
    for name, info in receipt["outputs"].items():
        if sha(root/name) != info["sha256"]:
            raise ValueError("Public CEM table hash changed")
        tables[name] = pd.read_csv(root/name)
        if len(tables[name]) != info["rows"]:
            raise ValueError("Public CEM table row count changed")
    cases = tables["paper/data/cem_steering_iteration_cases.csv"]
    expected = {(t,e,a,i) for t in TASKS for e in range(4) for a in ARMS for i in range(15)}
    actual = list(cases[["task","episode","arm","iteration"]].itertuples(index=False,name=None))
    if len(actual) != 240 or set(actual) != expected:
        raise ValueError("Require all four scenarios per task/arm and all fifteen iterations")
    values = cases[METRIC].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Invalid RMS values")
    initial = cases[cases.iteration == 0]
    if len(initial) != 16 or not (initial.shared_population_elite_overlap_count == 10).all() or not (initial.same_candidate_actions_verified.astype(str).str.lower() == "true").all():
        raise ValueError("All ten shared initial elites are not established")
    if not (initial[METRIC] == 0).all():
        raise ValueError("Initial proposals are not identical")
    later = cases[cases.iteration != 0]
    if later.shared_population_elite_overlap_count.notna().any():
        raise ValueError("Later candidate-ID overlap is not a common-action measurement")
    summary = tables["paper/data/cem_steering_iteration_summary.csv"]
    selected = summary[summary.metric == METRIC]
    if len(selected) != 60 or not (selected.n == 4).all() or selected.duplicated(["task","arm","iteration"]).any():
        raise ValueError("Incomplete scenario means")
    recomputed = cases.groupby(["task","arm","iteration"])[METRIC].mean().sort_index()
    reported = selected.set_index(["task","arm","iteration"])["mean"].sort_index()
    if not recomputed.index.equals(reported.index) or not np.allclose(recomputed,reported,rtol=1e-12,atol=1e-14):
        raise ValueError("Plotted means disagree with source scenarios")
    elite_summary = tables["paper/data/cem_steering_shared_iteration0_summary.csv"]
    elite_summary = elite_summary[elite_summary.metric == "shared_population_elite_overlap_count"]
    if len(elite_summary) != 4 or not (elite_summary.n == 4).all() or not (elite_summary["mean"] == 10).all():
        raise ValueError("Initial-elite aggregate disagrees with cases")
    provenance = dict(source_receipt_sha256=sha(receipt_path),plot_source_sha256=sha(__file__),
                      source_tables=receipt["outputs"],scenario_count=8,n_per_task=4,
                      individual_traces=16,metric=METRIC,coordinates=120,maximum_individual_value=float(values.max()),
                      initial_elite_overlap="10/10 for all sixteen case/arm pairs; exact initial action identity verified",
                      scope="Exploratory historical development inputs; no executed actions; individual traces instead of confidence bands")
    return cases, provenance


def make_figure(cases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch

    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10.5,
                         "axes.labelsize":10.5,"xtick.labelsize":10.5,"ytick.labelsize":10.5,
                         "pdf.fonttype":42,"svg.fonttype":"none","svg.hashsalt":"steering-search-story-v1"})
    fig, axes = plt.subplots(1,2,figsize=(6.8,5.2),sharey=True)
    fig.subplots_adjust(left=.105,right=.97,bottom=.265,top=.615,wspace=.18)
    fig.text(.075,.945,"Same initial choices.",fontsize=20,weight="bold",color=INK)
    fig.text(.075,.868,"Different search trajectories.",fontsize=20,weight="bold",color=TEAL)
    # This statement is guarded against every actual case and the source summary.
    fig.add_artist(FancyBboxPatch((.075,.758),.48,.063,transform=fig.transFigure,
                                  boxstyle="round,pad=.009,rounding_size=.012",
                                  edgecolor="none",facecolor="#edf5f4"))
    fig.text(.091,.78,"All 10 initial elites shared",fontsize=11,color=TEAL,weight="bold")
    handles = [Line2D([0],[0],color=TEAL,lw=2.8,label="Learned"),
               Line2D([0],[0],color=AMBER,lw=2.8,label="Calibrated random")]
    fig.legend(handles=handles,loc="upper right",bbox_to_anchor=(.982,.823),frameon=False,
               fontsize=10.5,handlelength=1.4,labelspacing=.4,borderaxespad=0)
    fig.text(.105,.69,"Proposal-mean RMS change from native · 120 coordinates",fontsize=10.5,color=MUTED)
    peak = float(cases[METRIC].max())
    upper = max(.1,math.ceil(peak*1.08/.1)*.1)
    for ax, task, title in zip(axes,TASKS,("Reach","Reach-Wall")):
        ax.set_title(title,loc="left",fontsize=12,weight="bold",color=INK,pad=10)
        ax.axhline(0,color="#7e8c92",lw=.9,ls=(0,(2,2)),zorder=1)
        for arm,color in zip(ARMS,(TEAL,AMBER)):
            block = cases[(cases.task==task)&(cases.arm==arm)]
            for episode in range(4):
                values = block[block.episode==episode].sort_values("iteration")
                line, = ax.plot(values.iteration+1,values[METRIC],color=color,alpha=.36,lw=1.15,zorder=2)
                line.set_gid(f"individual-{task}-{arm}-{episode}")
                ax.scatter([15],[values[METRIC].iloc[-1]],s=13,color=color,alpha=.55,edgecolors="none",zorder=3)
            means = block.groupby("iteration")[METRIC].mean().sort_index()
            line, = ax.plot(means.index+1,means.to_numpy(),color=color,lw=2.6,zorder=4)
            line.set_gid(f"mean-{task}-{arm}")
        ax.set_xlim(.8,15.4)
        ax.set_ylim(-upper*.035,upper)
        ax.set_xticks([1,5,10,15])
        ax.set_yticks(np.arange(0,upper+.01,.25 if upper>=1 else upper/4))
        ax.set_xlabel("CEM iteration",color=MUTED,labelpad=9)
        ax.tick_params(axis="both",length=0,pad=6,colors=MUTED)
        ax.grid(axis="y",color="#e4e9eb",lw=.65,zorder=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(.075,.105,"Thin lines: every scenario. Bold lines: scenario means.",fontsize=10.5,color=MUTED)
    fig.text(.075,.048,"All 8 cases · n=4/task · exploratory · no actions executed",fontsize=10.5,color=MUTED)
    return fig


def build(root=ROOT):
    import matplotlib.pyplot as plt
    root = Path(root)
    cases, provenance = load_source(root)
    fig = make_figure(cases)
    output = root/"docs/figures"
    output.mkdir(parents=True,exist_ok=True)
    description = json.dumps(provenance,sort_keys=True)
    for extension in ("png","svg","pdf"):
        metadata = ({"Description":description,"Date":None} if extension=="svg" else
                    {"Subject":description,"CreationDate":None,"ModDate":None} if extension=="pdf" else
                    {"Description":description})
        fig.savefig(output/f"steering_search_story.{extension}",dpi=240,facecolor="white",metadata=metadata)
    plt.close(fig)
    return provenance


if __name__ == "__main__":
    print(json.dumps(build(),indent=2))
