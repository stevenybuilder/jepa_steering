"""Display the complete all-block action-history counterfactual, not a best layer."""
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from build_publication_figures import ROOT, INK, MUTED, TEAL, AMBER, clean, save, sha, style


def load():
    path = ROOT/"paper/data/action_counterfactual_summary.json"
    receipt = json.loads(path.read_text())
    if receipt.get("status") != "complete16_action_counterfactual_development" or not receipt["all_cloud_verified"]:
        raise ValueError("Complete C analysis required")
    if receipt["cohort"] != 16 or receipt["n_per_task"] != 8 or receipt["physical_outcomes_measured"]:
        raise ValueError("Unexpected scientific scope")
    source = ROOT/"paper/data/action_counterfactual_summary.csv"
    if sha(source) != receipt["outputs"][source.name]["sha256"]:
        raise ValueError("Counterfactual summary changed")
    table = pd.read_csv(source)
    selected = table[(table.modality == "official") & (table.metric == "donor_reconstruction") &
                     table.arm.isin(["all_blocks_h3_donor", "all_blocks_persistent_donor"])]
    keys = ["task", "bank", "arm", "horizon"]
    expected = {(t, b, a, h) for t in ("reach", "reach-wall") for b in ("original", "fresh")
                for a in ("all_blocks_h3_donor", "all_blocks_persistent_donor") for h in (3, 4, 6)}
    if len(selected) != 24 or selected.duplicated(keys).any() or set(selected[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError("Require all24 registered task/bank/history/endpoint cells")
    if not (selected.n == 8).all() or not (selected.n_defined == 8).all():
        raise ValueError("No reduced-cohort reconstruction means")
    if not np.isfinite(selected[["mean", "marginal_95_low", "marginal_95_high"]]).all().all():
        raise ValueError("Undefined reconstruction must not be hidden")
    return selected, dict(source_receipt_sha256=sha(path), source_table_sha256=sha(source),
                         plot_source_sha256=sha(__file__), cohort=16, n_per_task=8,
                         endpoint="official weighted forecast distance from coherent raw-action counterfactual",
                         scope="All-block donor interventions at recorded endpoints; no physical actions executed")


def render():
    table, provenance = load()
    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.9), sharey=True)
    fig.subplots_adjust(left=.12, right=.96, top=.67, bottom=.29, wspace=.20)
    fig.text(.055, .94, "An action returns in the next context window.", fontsize=17, weight="bold", color=INK)
    fig.text(.055, .875, "A one-time patch is not the same as changing that action.", fontsize=11.5, color=MUTED)
    for ax, task, label in zip(axes, ("reach", "reach-wall"), ("Reach", "Reach-Wall")):
        ax.axvline(4, color="#bac7cc", lw=.9, ls=":")
        for arm, color in (("all_blocks_h3_donor", AMBER), ("all_blocks_persistent_donor", TEAL)):
            for bank, ls in (("original", "-"), ("fresh", "--")):
                rows = table[(table.task == task) & (table.arm == arm) & (table.bank == bank)].sort_values("horizon")
                x = rows.horizon.to_numpy()
                ax.plot(x, rows["mean"], color=color, ls=ls, lw=2, marker="o", ms=4)
                ax.fill_between(x, rows.marginal_95_low, rows.marginal_95_high, color=color, alpha=.13, lw=0)
        ax.set_title(label, loc="left", fontsize=13, weight="bold", color=INK, pad=10)
        ax.set(xticks=[3, 4, 6], xticklabels=["H3", "H4", "H6"], xlim=(2.9, 6.1),
               yticks=[0, .5, 1], ylim=(-.03, 1.07), xlabel="Measured forecast endpoint")
        clean(ax)
    axes[0].set_ylabel("Counterfactual reconstruction R\n1 = exact; 0 = native forecast")
    fig.legend([Line2D([0], [0], color=c, lw=2) for c in (AMBER, TEAL)],
               ["Patch at H3 only", "Patch both H3 and H4 appearances"],
               loc="upper left", bbox_to_anchor=(.055, .81), ncol=2, frameon=False, fontsize=10.3)
    fig.text(.055, .142, "All six blocks patched. The donor action reappears as history at H4.", fontsize=10.5, color=INK)
    fig.text(.055, .091, "Solid / dashed: original / fresh action banks · all 16 contexts, n=8/task · full marginal 95% bands", fontsize=8.8, color=MUTED)
    fig.text(.055, .043, "Predictor-internal action counterfactuals, not physical outcomes or protected-confirmation mechanisms.", fontsize=8.8, color=MUTED)
    save(fig, "action_history_consistency", provenance)
    print("PASS: all24 source-bound history cells, complete16 contexts, no selected layer")


if __name__ == "__main__":
    render()
