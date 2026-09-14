"""Print-sized scientific figures; no model execution or result selection.

Architecture: fixed_response.py/operator_fit.py; plots read existing aggregates.
README figures and source-bound analysis scripts remain untouched.
Render at 6.8 inches (10.5 pt minimum); placement at 5.5 inches is >=8.49 pt.
"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/figures"
INK, MUTED = "#253044", "#647184"
GREEN, BLUE, AMBER = "#237969", "#3f68a6", "#a46c22"


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5,
                         "svg.fonttype": "none", "svg.hashsalt": "paper-architecture-v1",
                         "pdf.fonttype": 42})


def save(fig, name):
    for ext in ("pdf", "svg", "png"):
        meta = {"CreationDate": None, "ModDate": None} if ext == "pdf" else {"Date": None} if ext == "svg" else None
        fig.savefig(OUT / f"{name}.{ext}", dpi=220, facecolor="white", metadata=meta)
    plt.close(fig)


def architecture():
    fig = plt.figure(figsize=(6.8, 4.45), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 6.8), ylim=(0, 4.45))
    ax.axis("off")

    def text(x, y, label, *, size=10.5, color=INK, weight="normal", ha="center"):
        return ax.text(x, y, label, fontsize=size, color=color, weight=weight,
                       ha=ha, va="center", linespacing=1.25, zorder=5)

    def box(x, y, w, h, label, *, fill="white", edge="#bfc8d3", size=10.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.025,rounding_size=.045",
                                   fc=fill, ec=edge, lw=.9, zorder=3))
        text(x+w/2, y+h/2, label, size=size)

    def arrow(points, *, color=MUTED, dash=False):
        if len(points) > 2:
            ax.plot(*zip(*points[:-1]), color=color, lw=1, zorder=1,
                    linestyle="--" if dash else "-")
        ax.annotate("", xy=points[-1], xytext=points[-2], zorder=2,
                    arrowprops={"arrowstyle": "->", "color": color, "lw": 1,
                                "linestyle": "--" if dash else "-", "shrinkA": 0, "shrinkB": 0})

    text(.16, 4.22, "Frozen model; alternative activation edits", size=12, weight="bold", ha="left")
    text(.16, 3.95, "H6 imagined rollout · hooks active only at H3", color=MUTED, ha="left")
    box(.18, 3.12, 1.30, .56, "Image\nDINO encoder", fill="#f5f7fa")
    box(.18, 2.32, 1.30, .56, "Proprio\nState encoder", fill="#f5f7fa")
    box(1.64, 3.25, .32, .30, "V", fill="#e4f1eb", edge=GREEN)
    arrow([(1.51, 3.40), (1.61, 3.40)])
    arrow([(1.99, 3.40), (2.13, 3.40), (2.13, 3.18), (2.22, 3.18)])
    arrow([(1.51, 2.60), (2.04, 2.60), (2.04, 3.18), (2.22, 3.18)])

    # Six transformer blocks. R is specifically after B3, not after B5.
    xs = [2.25, 2.87, 3.49, 4.11, 5.12, 5.74]
    text(4.23, 3.64, "Six predictor blocks", weight="bold")
    for i, x in enumerate(xs):
        box(x, 2.92, .43, .52, f"B{i}", fill="#edf2fa" if i == 3 else "white")
        if i < 5:
            arrow([(x+.46, 3.18), (xs[i+1]-.03, 3.18)])
    box(4.70, 3.03, .29, .30, "R", fill="#e6eefb", edge=BLUE)
    arrow([(6.20, 3.18), (6.48, 3.18), (6.48, 1.99), (6.26, 1.99)])

    # Every block is action-conditioned. A changes only the B3 condition.
    box(.18, 1.61, 1.54, .56, "Action encoder", fill="#f5f7fa")
    arrow([(1.75, 1.89), (1.90, 1.89), (1.90, 2.39), (5.96, 2.39)])
    for i, x in enumerate(xs):
        arrow([(x+.215, 2.39), (x+.215, 2.89)], color=AMBER if i == 3 else MUTED)
    box(4.17, 2.55, .30, .28, "A", fill="#fbefde", edge=AMBER)
    text(3.86, 2.12, "Actions condition every block", color=MUTED)

    box(5.12, 1.68, 1.11, .62, "Latent\nforecast")
    box(4.76, .82, 1.76, .58, "Encoded-goal\nL2 cost")
    arrow([(5.67, 1.65), (5.67, 1.43)])
    text(4.40, 1.58, "Goal", color=MUTED)
    arrow([(4.55, 1.58), (4.65, 1.58), (4.65, 1.30), (4.73, 1.30)])
    box(2.49, .82, 1.70, .58, "CEM: score + refit\n300 candidates", fill="#f5f7fa")
    arrow([(4.73, 1.11), (4.22, 1.11)])
    arrow([(3.34, 1.43), (3.34, 1.52), (.95, 1.52), (.95, 1.58)])
    box(.18, .82, 1.73, .58, "Execute prefix\nObserve + replan", fill="#f5f7fa")
    arrow([(2.46, 1.11), (1.94, 1.11)])
    arrow([(.15, 1.11), (.065, 1.11), (.065, 3.40), (.15, 3.40)], dash=True)

    text(.17, .47, "V  visual input", color=GREEN, ha="left")
    text(2.25, .47, "A  B3 condition", color=AMBER, ha="left")
    text(4.36, .47, "R  rank-four B3 output", color=BLUE, ha="left")
    text(.17, .18, "Eight alternative arms; no arm combines R with V or A.", ha="left", color=MUTED)
    fig.canvas.draw()
    assert min(t.get_fontsize() for t in ax.texts) >= 10.5
    save(fig, "paper_architecture")


def layer_response():
    with (ROOT / "paper/data/layer_mechanism_grid.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 576
    tasks = ("reach", "reach-wall", "pusht")
    arms = [f"single_block{i}" for i in range(6)] + ["intermediate_blocks2_3", "all_six_blocks"]
    values = []
    for precision in ("bfloat16", "float32"):
        grid = []
        for arm in arms:
            line = []
            for task in tasks:
                matches = [r for r in rows if r["precision"] == precision and r["arm"] == arm
                           and r["task"] == task and r["endpoint"] == "proprio_mse" and r["horizon"] == "6"]
                assert len(matches) == 1
                line.append(float(matches[0]["reduction_vs_native_percent"]))
            grid.append(line)
        values.append(np.array(grid))
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.85), sharey=True)
    fig.subplots_adjust(left=.15, right=.88, bottom=.18, top=.87, wspace=.14)
    bound = max(abs(np.array(values).min()), abs(np.array(values).max()))
    for ax, data, title in zip(axes, values, ("BF16", "FP32")):
        im = ax.imshow(data, cmap="RdBu", vmin=-bound, vmax=bound, aspect="auto")
        for i in range(8):
            for j in range(3):
                ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center", fontsize=10.5,
                        color="white" if abs(data[i,j]) > bound*.55 else INK)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xticks(range(3), ["Reach", "R.-Wall", "Push-T"])
        ax.set_yticks(range(8), [f"B{i}" for i in range(6)] + ["B2+B3", "All six"])
        ax.tick_params(length=0)
    cb = fig.colorbar(im, cax=fig.add_axes([.91, .18, .025, .69]))
    cb.ax.tick_params(labelsize=10.5)
    fig.text(.5, .035, "H6 proprioceptive MSE reduction vs native (%)", ha="center", fontsize=10.5)
    save(fig, "paper_layer_response")


def decision_geometry():
    data = json.loads((ROOT / "paper/data/decision_geometry.json").read_text())
    assert data["scenarios"] == 192 and data["candidate_scores_per_arm"] == 300
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.65))
    fig.subplots_adjust(left=.105, right=.98, bottom=.23, top=.85, wspace=.39)
    for ti, task in enumerate(("reach", "reach-wall")):
        group = [s for s in data["summaries"] if s["task"] == task]
        assert [s["arm"] for s in group] == ["refined", "random_refined", "coupling", "random_coupling"]
        pos = np.arange(4) + (ti-.5)*.36
        color = ("#286e78", "#a3b3b9")[ti]
        counts = [s["no_flip_certified"] for s in group]
        axes[0].bar(pos, counts, .34, color=color, label=("Reach", "Reach-Wall")[ti])
        for x, y in zip(pos, counts):
            axes[0].text(x, y+1, str(y), ha="center", va="bottom", fontsize=10.5)
        axes[1].bar(pos, [100*s["metrics"]["centered_rms_to_native_std"]["mean"] for s in group], .34, color=color)
    axes[0].set_title("Sufficient no-flip certificate", fontsize=11, weight="bold")
    axes[0].set_ylabel("Scenarios (of 96)")
    axes[0].set_ylim(0, 106)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=10.5,
               ncol=2, loc="upper center", bbox_to_anchor=(.5, 1.01))
    axes[1].set_title("Candidate-cost change", fontsize=11, weight="bold")
    axes[1].set_ylabel("Centered RMS / native SD (%)")
    for ax in axes:
        ax.set_xticks(range(4), ["R", "Random\nR", "V/A", "Random\nV/A"])
        ax.tick_params(axis="x", length=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.text(.5, .025, "Same 300 candidates per arm; initial population, not full CEM.", ha="center", fontsize=10.5)
    save(fig, "paper_decision_geometry")


def pathway_geometry():
    with (ROOT / "paper/data/pathway_geometry_geometry_summary.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r["control"] == "equal_anchor_linear" and r["intervention"] == "cubic"]
    tasks = ("reach", "reach-wall", "pusht", "wall", "pointmaze")
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 4.1))
    fig.subplots_adjust(left=.105, right=.98, bottom=.26, top=.85, wspace=.41)
    for precision, color, offset, label in (("bfloat16", "#c57d36", -.12, "BF16"), ("float32", "#187c80", .12, "FP32")):
        def metric(name):
            selected = []
            for task in tasks:
                matching = [r for r in rows if r["precision"] == precision and r["task"] == task and r["metric"] == name]
                assert len(matching) == 1
                selected.append(matching[0])
            return selected
        local, future = metric("omitted_activation_mse"), metric("recorded_proprio_mse_h6")
        pos = np.arange(5)+offset
        ratio = np.array([float(r["intervention_mean"])/float(r["control_mean"]) for r in local])
        lo = np.array([1-float(r["percent_95_high"])/100 for r in local])
        hi = np.array([1-float(r["percent_95_low"])/100 for r in local])
        axes[0].errorbar(pos, ratio, yerr=[ratio-lo, hi-ratio], fmt="o", color=color, label=label, capsize=3, markersize=5)
        vals = np.array([float(r["percent_of_reference"]) for r in future])
        lo = np.array([float(r["percent_95_low"]) for r in future])
        hi = np.array([float(r["percent_95_high"]) for r in future])
        axes[1].errorbar(pos, vals, yerr=[vals-lo, hi-vals], fmt="o", color=color, capsize=3, markersize=5)
    axes[0].set(yscale="log", ylim=(.00003, 3), ylabel="Cubic / linear activation MSE")
    axes[0].set_title("Local reconstruction", fontsize=11, weight="bold")
    axes[0].axhline(1, color=MUTED, ls="--", lw=1)
    axes[0].legend(frameon=False, fontsize=10.5, loc="lower left")
    axes[1].set_title("H6 forecast advantage", fontsize=11, weight="bold")
    axes[1].set_ylabel("Cubic advantage / native MSE (%)")
    axes[1].axhline(0, color=MUTED, ls="--", lw=1)
    for ax in axes:
        ax.set_xticks(range(5), ["Reach", "R.-Wall", "Push-T", "Wall", "P.Maze"], rotation=45, ha="right")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
    fig.text(.5, .065, "Right: matched requested dose; realized energy can differ.", ha="center", fontsize=10.5)
    fig.text(.5, .015, "Marginal paired 95% intervals; neither axis is robot success.", ha="center", fontsize=10.5)
    save(fig, "paper_pathway_geometry")


def benchmark():
    from build_comparison_figures import load_data, COLORS
    from check_public_results import ARMS, TASKS
    source, report = load_data()
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.65))
    fig.subplots_adjust(left=.095, right=.985, bottom=.20, top=.89, wspace=.20, hspace=.65)
    for ax, task, title in zip(axes.flat, TASKS, ("Reach", "Reach-Wall", "PointMaze", "Wall")):
        ti = source["task_order"].index(task)
        values = [row["values"][ti] for row in source["author_rows"]] + [report["results"][task]["success_percent"][arm] for arm in ARMS]
        assert len(values) == 11
        ax.axvspan(-.6, 2.45, color="#edf0e7", zorder=0)
        ax.bar(range(11), values, color=["#c4c7c3", "#92998f", "#a7b09e", *COLORS], width=.77, zorder=3)
        ax.axvline(2.5, color=MUTED, ls="--", lw=.7)
        ax.set_xticks(range(11), ["D", "I", "F", "N", "R", "rR", "C", "rC", "VA", "V", "A"], fontsize=10.5)
        ax.set(xlim=(-.65, 10.65), ylim=(0, 105), yticks=[0, 50, 100])
        ax.set_title(title, fontsize=12, weight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
    axes[0,0].set_ylabel("Success (%)")
    axes[1,0].set_ylabel("Success (%)")
    fig.text(.5, .975, "Published references | All eight fresh protected arms", ha="center", fontsize=11, weight="bold")
    fig.text(.5, .115, "Shaded: D = DINO-WM, I = JEPA improved, F = JEPA final.", ha="center", fontsize=10.5)
    fig.text(.5, .070, "N = native; R/rR = learned/random rank-four; C/rC = coupling/random.", ha="center", fontsize=10.5)
    fig.text(.5, .025, "VA/V/A = component arms. Cross-study bars are not paired effects.", ha="center", fontsize=10.5)
    save(fig, "paper_benchmark")


def action_interaction():
    """Render the saved same-input factorial terms, without new estimates."""
    key = "paper/data/pathway_geometry_coupling_summary.csv"
    receipt = json.loads((ROOT / "paper/data/pathway_geometry.json").read_text())
    raw = (ROOT / key).read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt["outputs"][key]["sha256"]:
        raise ValueError("Coupling summary SHA differs from the completed pathway receipt")
    rows = list(csv.DictReader(raw.decode().splitlines()))
    assert len(rows) == receipt["outputs"][key]["rows"]
    selected = [r for r in rows if r["modality"] == "proprio" and r["horizon"] == "6"]
    cells = {
        ("bfloat16", "reach"): "S01", ("float32", "reach"): "S02",
        ("bfloat16", "reach-wall"): "S15", ("float32", "reach-wall"): "S16",
        ("bfloat16", "pusht"): "S27", ("float32", "pusht"): "S28",
        ("bfloat16", "wall"): "S37",
    }
    assert {(r["precision"], r["task"]) for r in selected} == set(cells)
    metrics = ("factorial_from_arm_losses", "additive_quadratic_cross",
               "nonadditive_mse_remainder")
    grouped = {}
    for pair, source in cells.items():
        group = [r for r in selected if (r["precision"], r["task"]) == pair]
        assert {r["source_id"] for r in group} == {source}
        lookup = {r["metric"]: r for r in group}
        assert len(lookup) == len(group) and all(m in lookup for m in metrics)
        # Reuse the source recount's numerical identity tolerance. The remainder
        # already contains q²; output_interaction_mse must not be added again.
        d, cross, remainder = [float(lookup[m]["mean"]) for m in metrics]
        native = float(lookup[metrics[0]]["native_mse"])
        assert abs(d-cross-remainder) <= 1e-12 + 1e-6*native
        grouped[pair] = lookup

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 4.5))
    fig.subplots_adjust(left=.105, right=.98, bottom=.31, top=.77, wspace=.33)
    styles = (
        (metrics[0], "Loss factorial D", INK, "D", -.19),
        (metrics[1], "Quadratic cross", "#b77931", "o", 0),
        (metrics[2], "Nonadditive remainder", "#277b83", "s", .19),
    )
    for ax, precision, title in zip(axes, ("bfloat16", "float32"), ("BF16", "FP32")):
        tasks = ("reach", "reach-wall", "pusht", "wall") if precision == "bfloat16" else ("reach", "reach-wall", "pusht")
        for metric, label, color, marker, offset in styles:
            for i, task in enumerate(tasks):
                row = grouped[precision, task][metric]
                value = float(row["percent_of_reference"])
                lo, hi = row["percent_95_low"], row["percent_95_high"]
                if lo and hi:
                    ax.errorbar(i+offset, value,
                                yerr=[[value-float(lo)], [float(hi)-value]],
                                fmt=marker, color=color, capsize=3, markersize=5,
                                label=label if i == 0 else None)
                else:
                    assert (precision, task, metric) == ("bfloat16", "wall", metrics[0])
                    assert row["missing_reason"] and row["n"] == "0"
                    ax.plot(i+offset, value, marker=marker, color=color,
                            markerfacecolor="white", markersize=5, linestyle="none")
        ax.axhline(0, color=MUTED, linewidth=.8, linestyle="--")
        ax.set_xticks(range(len(tasks)), ["Reach", "R.-Wall", "Push-T", "Wall"][:len(tasks)])
        ax.set_xlim(-.55, len(tasks)-.45)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_ylabel("Term / native H6 MSE (%)", fontsize=10.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
    fig.text(.5, .97, "Same-input forecast interaction: D = cross + remainder",
             ha="center", fontsize=11, weight="bold")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, ncol=3,
               loc="upper center", bbox_to_anchor=(.5, .925), fontsize=10.5,
               handletextpad=.35, columnspacing=.8)
    fig.text(.5, .22, "Different y-scales. Remainder already includes output-interaction q².",
             ha="center", fontsize=10.5)
    fig.text(.5, .16, "Wall D: aggregate only, no CI. Other intervals: marginal paired 95%.",
             ha="center", fontsize=10.5)
    fig.text(.5, .10, "No PointMaze or FP32 Wall source. This is not a success factorial.",
             ha="center", fontsize=10.5)
    fig.text(.5, .04, "Precision-specific fits and doses preclude a precision-causal claim.",
             ha="center", fontsize=10.5)
    save(fig, "paper_action_interaction")
    print("Action interaction: 7 source cells, 21 saved terms, 20 saved CIs; SHA + identity verified.")


def pilot_tables(root=ROOT):
    """Fail closed on partial/mismatched exports; absence is a normal skip."""
    path = root / "paper/data/pilot_summary.json"
    if not path.exists():
        return None
    receipt = json.loads(path.read_text())
    if (receipt["status"] != "complete64_exploratory_development"
            or receipt["cohort"] != 64 or receipt["n_per_task"] != 32
            or receipt["physical_outcomes_measured"] is not False
            or receipt["fresh_confirmation"] is not False):
        raise ValueError("Paper pilot figures require the complete64 development receipt")
    protocol_path = root / "paper/data/pilot_summary_protocol.json"
    if hashlib.sha256(protocol_path.read_bytes()).hexdigest() != receipt["protocol_sha256"]:
        raise ValueError("Pilot protocol SHA mismatch")
    protocol = json.loads(protocol_path.read_text())
    if protocol["attention"]["candidate"] != "original candidate0 zero-action plan":
        raise ValueError("Unexpected pilot attention input scope")
    frames = {}
    for name, evidence in receipt["outputs"].items():
        source = root / name
        if not source.resolve().is_relative_to(root.resolve()):
            raise ValueError("Pilot export path escapes the project")
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != evidence["sha256"]:
            raise ValueError(f"Pilot export SHA mismatch: {name}")
        rows = list(csv.DictReader(raw.decode().splitlines()))
        if len(rows) != evidence["rows"]:
            raise ValueError(f"Pilot export row-count mismatch: {name}")
        frames[name] = rows
    attention = frames["paper/data/pilot_summary_attention_summary.csv"]
    spatial = [r for r in attention if r["metric"] == "spatial_distance_patches"]
    expected = {(t, l, h, head) for t in ("reach", "reach-wall")
                for l in range(6) for h in range(1, 7) for head in range(16)}
    keys = [(r["task"], int(r["layer"]), int(r["horizon"]), int(r["head"])) for r in spatial]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Pilot attention lacks the complete all-head/all-block H1–H6 grid")
    for row in spatial:
        point = pilot_point(row)
        if point is None or point[0] < 0:
            raise ValueError("Undefined or negative spatial attention distance")
    component = frames["paper/data/pilot_summary_component_summary.csv"]
    wanted = ("common_field_energy_fraction", "centered_field_energy_fraction",
              "common_centered_cost_reconstruction", "centered_centered_cost_reconstruction")
    relevant = [r for r in component if r["metric"] in wanted]
    expected = {(t, a, m) for t in ("reach", "reach-wall")
                for a in ("fixed_rank4", "matched_random_fixed_rank4") for m in wanted}
    keys = [(r["task"], r["arm"], r["metric"]) for r in relevant]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Pilot component summary lacks complete learned/random component cells")
    for row in relevant:
        point = pilot_point(row)
        if row["metric"].endswith("energy_fraction") and (point is None or not 0 <= point[0] <= 1):
            raise ValueError("Invalid pilot component energy fraction")
    return spatial, relevant


def pilot_point(row):
    """Return saved mean/interval, preserving undefined full-cohort ratios."""
    if int(row["n"]) != 32:
        raise ValueError("Pilot mean is not over all32 scenarios/task")
    if not row["mean"]:
        if (not row.get("undefined_reason") or row["marginal_95_low"]
                or row["marginal_95_high"] or int(row["n_defined"]) >= 32):
            raise ValueError("Missing pilot mean without explicit undefined-ratio provenance")
        return None
    if int(row["n_defined"]) != 32:
        raise ValueError("A pilot mean selectively excludes undefined scenarios")
    values = tuple(float(row[k]) for k in ("mean", "marginal_95_low", "marginal_95_high"))
    if not np.isfinite(values).all() or not values[1] <= values[0] <= values[2]:
        raise ValueError("Nonfinite or malformed saved pilot interval")
    return values


def pilot_attention(spatial):
    selected = [r for r in spatial if int(r["horizon"]) == 6]
    assert len(selected) == 192
    lookup = {(r["task"], int(r["head"]), int(r["layer"])): pilot_point(r)[0] for r in selected}
    low, high = min(lookup.values()), max(lookup.values())
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 5.1), sharey=True)
    fig.subplots_adjust(left=.10, right=.80, bottom=.22, top=.82, wspace=.18)
    for ax, task, title in zip(axes, ("reach", "reach-wall"), ("Reach", "Reach-Wall")):
        matrix = np.array([[lookup[task, head, layer] for layer in range(6)] for head in range(16)])
        im = ax.imshow(matrix, aspect="auto", cmap="viridis", origin="upper", vmin=low, vmax=high)
        ax.set_xticks(range(6), [f"B{i}" for i in range(6)])
        ax.set_yticks(range(16))
        ax.tick_params(length=0)
        ax.set_title(title, fontsize=12, weight="bold")
    axes[0].set_ylabel("Head (zero-indexed)")
    cb = fig.colorbar(im, cax=fig.add_axes([.83, .22, .023, .60]))
    cb.ax.tick_params(labelsize=10.5)
    cb.set_label("Mean attention distance\n(patch spacings)", fontsize=10.5, labelpad=12)
    fig.text(.5, .955, "Attention distance across layers and heads", ha="center", fontsize=13, weight="bold")
    fig.text(.5, .90, "H6 forecast · visual queries to visual keys · shared color scale", ha="center", fontsize=10.5)
    fig.text(.5, .13, "Unsteered · 32 starting states per task · zero-action candidate", ha="center", fontsize=10.5)
    fig.text(.5, .065, "Shorter distances indicate more local attention on the image patch grid.", ha="center", fontsize=10.5)
    save(fig, "paper_pilot_attention")


def pilot_replay(component):
    lookup = {(r["task"], r["arm"], r["metric"]): r for r in component}
    groups = [(t, a) for t in ("reach", "reach-wall")
              for a in ("fixed_rank4", "matched_random_fixed_rank4")]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 4.25))
    fig.subplots_adjust(left=.10, right=.98, bottom=.30, top=.80, wspace=.34)
    for ax, suffix, title in zip(axes, ("field_energy_fraction", "centered_cost_reconstruction"),
                                 ("Activation-edit energy", "Centered cost reconstruction")):
        for component_name, offset, color, marker in (
                ("common", -.13, "#287c83", "o"), ("centered", .13, "#bd7c32", "s")):
            for i, (task, arm) in enumerate(groups):
                point = pilot_point(lookup[task, arm, f"{component_name}_{suffix}"])
                if point is None:
                    ax.text(i+offset, .07, "undefined", color=color, rotation=90,
                            ha="center", va="bottom", transform=ax.get_xaxis_transform(), fontsize=10.5)
                else:
                    mean, low, high = point
                    ax.errorbar(i+offset, mean, yerr=[[mean-low], [high-mean]], fmt=marker,
                                color=color, capsize=3, markersize=5,
                                label=component_name.capitalize() if i == 0 else None)
        ax.set_xticks(range(4), ["Reach\nR", "Reach\nrR", "R.-Wall\nR", "R.-Wall\nrR"])
        ax.set_xlim(-.5, 3.5)
        ax.axhline(0, color=MUTED, lw=.7)
        ax.axhline(1, color=MUTED, lw=.7, ls="--")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
        ax.set_title(title, fontsize=11, weight="bold")
    axes[0].set_ylabel("Fraction of full edit energy")
    axes[1].set_ylabel("1 − SSE / full centered cost energy")
    # No hard reconstruction limits: negative values and all saved CIs remain.
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, ncol=2,
               loc="upper center", bbox_to_anchor=(.5, .985), fontsize=10.5)
    fig.text(.5, .18, "R = refined; rR = calibrated random. Same 300 actions/scenario.", ha="center", fontsize=10.5)
    fig.text(.5, .115, "All 64 development scenarios; marginal paired 95% intervals.", ha="center", fontsize=10.5)
    fig.text(.5, .05, "Natural component doses; no clipping of reconstruction scores.", ha="center", fontsize=10.5)
    save(fig, "paper_pilot_replay")


def pilot_figures():
    tables = pilot_tables()
    if tables is None:
        print("SKIP pilot print figures: no completed pilot_summary.json yet.")
        return False
    pilot_attention(tables[0])
    pilot_replay(tables[1])
    print("Pilot print figures: complete64 + export hashes validated; all192 H6 attention cells.")
    return True


def main():
    style()
    for build in (architecture, layer_response, decision_geometry, pathway_geometry, benchmark, action_interaction):
        build()
    print("Wrote six print-specific PDF/PNG/SVG figures from existing verified aggregates.")
    pilot_figures()


if __name__ == "__main__":
    main()
