"""Render completed, checksum-bound evidence. Never evaluates a model or partial panel."""
from pathlib import Path
import csv
import hashlib
import json
import platform
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
DURABLE = ROOT / "artifacts/offline_study/primary-durable-20260907"
FIXED = ROOT / "artifacts/offline_study/fixed-response-20260908-v1"
FIG = PAPER / "figures"
DATA = PAPER / "data"
for folder in (FIG, DATA):
    folder.mkdir(parents=True, exist_ok=True)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

sources = {}
reports = {}
rows = []
closure = DURABLE / "three-task-offline-closure-20260907/metrics/report.json"
closure_data = json.loads(closure.read_text())
assert len(closure_data["verified_analysis_scopes"]) == 30

def register(path, expected=None):
    digest = sha(path)
    if expected is not None:
        assert digest == expected, f"Source checksum mismatch: {path}"
    rel = str(path.relative_to(ROOT))
    sources[rel] = digest
    return json.loads(path.read_text())

register(closure)
def ingest(path, report, category, task, precision):
    task = {"mw-reach":"reach", "mw-reach-wall":"reach-wall"}.get(task, task)
    reports[(category, task, precision)] = report
    assert report.get("fresh_confirmation") is False
    for c in report["contrasts"]:
        if "candidate" not in c:
            continue
        ci = c["simultaneous_95_reduction_percent_of_observed_native"]
        rows.append(dict(category=category, task=task, precision=precision,
                         candidate=c["candidate"], control=c["control"],
                         endpoint=c["endpoint"],
                         error_reduction_percent_of_native=c["error_reduction_percent_of_native"],
                         simultaneous_95_lower=ci[0], simultaneous_95_upper=ci[1],
                         independent_lineages=report["independent_lineage_groups"],
                         source=str(path.relative_to(ROOT)), source_sha256=sha(path)))

for binding in closure_data["verified_analysis_scopes"]:
    path = DURABLE / binding["path"].split("/workspace/jepa-runtime/", 1)[1]
    report = register(path, binding["sha256"])
    ingest(path, report, report["category"], report["task"], report["precision"])

for precision in ("bfloat16", "float32"):
    path = DURABLE / f"combined-reach-author-20260907/analysis-v1/{precision}/report.json"
    done = json.loads(path.with_name("DONE.json").read_text())
    expected = done.get("report_sha256", done.get("report.json"))
    assert expected, "Combined report must have a hash-bound receipt"
    report = register(path, expected)
    ingest(path, report, "combined", "reach", precision)
    for task in ("reach", "reach-wall"):
        path = FIXED / f"offline-analysis-v2/{task}/{precision}/report.json"
        done = register(path.with_name("DONE.json"))
        report = register(path, done["report_sha256"])
        assert report["status"] == "verified_fixed_response_offline_complete"
        assert report["independent_lineage_groups"] == {"reach":33,"reach-wall":27}[task]
        ingest(path, report, "fixed_response", task, precision)

def get(category, task, candidate, control="native", precision="bfloat16"):
    selected = [r for r in rows if r["category"] == category and r["task"] == task
                and r["candidate"] == candidate and r["control"] == control
                and r["precision"] == precision and r["endpoint"] == "proprio_mse_h6"]
    assert len(selected) == 1, (category, task, candidate, control, precision)
    return selected[0]

with (DATA / "reported_contrasts.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

BLUE, TEAL, ORANGE, RED, GRAY = "#3268a8", "#138578", "#cf8638", "#b6474a", "#9ba3ab"
sns.set_theme(style="whitegrid", context="paper", font="DejaVu Sans")
plt.rcParams.update({"font.size":10, "axes.titlesize":12, "axes.labelsize":10,
                    "svg.fonttype":"none", "pdf.fonttype":42,
                    "axes.spines.top":False, "axes.spines.right":False,
                    "grid.alpha":.22, "savefig.dpi":220})

def save(fig, stem, caption):
    fig.savefig(FIG / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.png", bbox_inches="tight", dpi=220)
    (FIG / f"{stem}.caption.txt").write_text(caption.strip()+"\n")
    plt.close(fig)

def point(ax, row, y, color, marker="o", annotate=False):
    v = row["error_reduction_percent_of_native"]
    lo, hi = row["simultaneous_95_lower"], row["simultaneous_95_upper"]
    assert lo <= v <= hi
    ax.errorbar(v, y, xerr=np.array([[v-lo], [hi-v]]), fmt=marker,
                color=color, markersize=5.5, capsize=3, linewidth=1.5)
    if annotate:
        ax.text(hi+.07, y, f"{v:+.2f}", va="center", fontsize=8, color=color)

def forest_style(ax, labels, title, lim):
    ax.axvline(0, color="#434950", lw=1)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels)-.45, -.65)
    ax.set_xlim(*lim)
    ax.set_title(title, loc="left", fontweight="bold", pad=12)
    ax.set_xlabel("H6 embedding error reduction (%)\nPositive values favor the edit")
    ax.grid(axis="y", visible=False)

# Figure 1: a conceptual schematic; no fabricated empirical trajectory.
fig, ax = plt.subplots(figsize=(11.5, 4.8))
ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
def box(x,y,w,h,text,color="#edf1f5",edge=BLUE):
    ax.add_patch(FancyBboxPatch((x,y), w,h,boxstyle="round,pad=0.08,rounding_size=.09",
                               fc=color,ec=edge,lw=1.2))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=10)
def arrow(x1,y1,x2,y2,dashed=False):
    ax.annotate("",(x2,y2),(x1,y1),arrowprops=dict(arrowstyle="->",lw=1.3,
                color="#4a5661",linestyle="--" if dashed else "-"))
box(.1,2.25,2,1.1,"Observation history\n+ candidate actions")
box(2.65,2.25,3.3,1.1,"Frozen dynamics predictor\nOne B3 edit at imagined H3")
box(6.5,2.25,2.1,1.1,"H6 forecast\nPredicted consequences")
box(9.2,2.25,2.6,1.1,"Unchanged CEM planner\nScore against the goal")
arrow(2.2,2.8,2.55,2.8); arrow(6.05,2.8,6.4,2.8); arrow(8.7,2.8,9.1,2.8)
box(2.65,.5,3.3,.9,"Offline fit: four directions B\nand coefficient map A",color="#e9f5f1",edge=TEAL)
arrow(4.3,1.5,4.3,2.15)
ax.text(6.4,.95,"One distributed correction: h' = h + Bc\nZero online response-probe forecasts",
        ha="left",va="center",color=TEAL,fontsize=11)
ax.text(.1,4.4,"FROM A REPRESENTATION TO A USEFUL FORECAST",fontsize=15,fontweight="bold",color="#26364a")
ax.text(.1,3.88,"Conceptual pipeline for the fixed-response successor; arrows do not claim improved behavior.",fontsize=10,color="#53616d")
ax.text(7.55,1.87,"Offline: score recorded future",ha="center",fontsize=8,color=TEAL)
ax.text(10.5,1.87,"Behavioral comparison pending",ha="center",fontsize=9,color=RED)
save(fig,"fig01_intervention_pipeline", "Figure 1. Fixed-response intervention pipeline. The base model remains frozen. A map calibrated offline computes four coefficients from native H3 activations and applies one distributed edit at predictor block B3. Corrected offline forecasts are measured; the full behavioral comparison remains incomplete. This diagram is conceptual, not a measured rollout or proof of action-ranking improvement.")

# Figure 2: selected methods AND their matched controls, not success-rate bars.
fig, axes = plt.subplots(1,3,figsize=(15,5.8),sharex=True)
specs=[("operator_rank","rank4","Original rank 4",BLUE),
       ("operator_rank","matched_random_rank4","  Matched random",GRAY),
       ("vision_action_coupling","joint_equal_standardized_energy","Equal-energy coupling",ORANGE),
       ("vision_action_coupling","matched_random_equal_standardized_energy","  Matched random",GRAY),
       ("fixed_response","fixed_rank4","Fixed-response rank 4",TEAL),
       ("fixed_response","matched_random_fixed_rank4","  Matched random",GRAY),
       ("combined","combined","Original combined",RED),
       ("combined","matched_random_combined","  Matched random",GRAY)]
for ax,task,title,n in zip(axes,["reach","reach-wall","pusht"],["Reach","Reach-Wall","Push-T"],[33,27,21]):
    assert ("operator_rank", task, "bfloat16") in reports
    assert ("vision_action_coupling", task, "bfloat16") in reports
    for y,(cat,arm,label,col) in enumerate(specs):
        if (cat,task,"bfloat16") in reports:
            point(ax,get(cat,task,arm),y,col,annotate=True)
        else:
            ax.text(.12,y,"Not measured",color=GRAY,va="center",fontsize=8)
    forest_style(ax,[x[2] for x in specs],f"{title} | {n} trajectories",(-3.5,7.0))
    ax.axhline(5.5, color=GRAY,lw=.7,ls="--")
fig.suptitle("Forecast corrections in frozen world models",x=.03,ha="left",fontweight="bold",fontsize=17)
fig.text(.03,.012,"BF16 primary. Original and successor studies have separate simultaneous 95% interval families. No behavioral outcomes are plotted.",fontsize=10,color="#535c66")
fig.tight_layout(rect=(0,.06,1,.94),w_pad=2)
save(fig,"fig02_forecast_effects", "Figure 2. Selected completed BF16 H6 proprioceptive embedding-MSE contrasts versus each study's paired native arm, alongside matched-random arms. Intervals are the source reports' paired-lineage simultaneous 95% intervals, scaled by the observed native mean. They are simultaneous within each task/category/precision, not globally across this figure. The original combined study (below dashed separator) and fixed-response successor have separate inference families. Missing methods are unmeasured, not zero. Estimates do not establish cross-study equivalence or a difference between old and new operators. These are forecast-error changes, not success-rate gains; 33/27/21 development trajectories are reused across arms.")

# Figure 3: pathways plus registered direct contrasts; no new statistical tests.
fig,axes=plt.subplots(1,3,figsize=(15,4.7))
arms=[("visual_only","Visual only"),("action_condition_only","Action-conditioning only"),
      ("joint","Joint (unscaled)"),("joint_equal_standardized_energy","Joint (equal energy)"),
      ("permuted_joint","Spatially permuted joint")]
for ax,task,title in zip(axes[:2],["reach","pusht"],["A  Reach | 33 trajectories","B  Push-T | 21 trajectories"]):
    for y,(arm,_) in enumerate(arms):point(ax,get("vision_action_coupling",task,arm),y,BLUE if task=="reach" else ORANGE,annotate=True)
    forest_style(ax,[x[1] for x in arms],title,(-2.1,4.7))
contrasts=[("reach","joint","permuted_joint","Reach: joint vs permuted"),
           ("reach","joint_equal_standardized_energy","visual_only","Reach: equal-energy joint\nvs visual only"),
           ("pusht","joint","action_condition_only","Push-T: joint vs action only")]
for y,(task,c,ctrl,_) in enumerate(contrasts):point(axes[2],get("vision_action_coupling",task,c,ctrl),y,TEAL,annotate=True)
forest_style(axes[2],[x[3] for x in contrasts],"C  Registered paired contrasts",(-2.1,4.7))
axes[2].set_xlabel("Error reduction vs named control\n(% of native error; positive favors first arm)")
fig.suptitle("Spatial arrangement and pathway choice change the effect",x=.03,ha="left",fontweight="bold",fontsize=16)
fig.tight_layout(rect=(0,.03,1,.91),w_pad=2)
save(fig,"fig03_pathway_and_spatial_controls", "Figure 3. BF16 pathway decomposition and existing registered paired contrasts. Visual and action-conditioning edits have opposing effects on Push-T. Reach's joint edit outperforms its spatial permutation, while equal-energy joint steering does not establish superiority over visual-only steering. Unscaled joint has a different total energy from equal-energy joint. Panels A/B compare with native; panel C compares with its named control, always scaling differences by native error. Source simultaneous 95% intervals are reused without new tests. These interventions do not identify semantic variables or establish equal-energy synergy.")

# Figure 4: spatial support and depth are distinct rank-1 sweeps.
fig,axes=plt.subplots(1,2,figsize=(11.5,5.3))
depth=[(f"single_block{i}",f"Block {i}") for i in range(6)]+[("intermediate_blocks2_3","Blocks 2 + 3"),("all_six_blocks","All six blocks")]
spatial=[("one_patch","One patch"),("contiguous_group","16 contiguous patches"),("equal_size_scattered_group","16 scattered patches"),("all_patches","All 256 patches")]
for ax,cat,items,title in zip(axes,["distribution_layer","distribution_spatial"],[depth,spatial],["A  Depth support","B  Spatial support"]):
    for y,(arm,label) in enumerate(items):
        point(ax,get(cat,"reach",arm),y-.09,BLUE,annotate=False)
        point(ax,get(cat,"reach","matched_random_"+arm),y+.13,GRAY,marker="s")
    forest_style(ax,[x[1] for x in items],title,(-3.1,4.3))
fig.suptitle("Broad spatial support does not imply editing every layer",x=.03,ha="left",fontweight="bold",fontsize=16)
fig.text(.03,.012,"Reach, n = 33. Blue: structured edit. Gray: matched random. Rank and total delivered energy are held fixed within each sweep.",fontsize=9)
fig.tight_layout(rect=(0,.06,1,.92),w_pad=2)
save(fig,"fig04_distribution", "Figure 4. Reach BF16 depth and spatial-support sweeps at fixed total rank 1 and matched delivered energy within each sweep, with source simultaneous 95% intervals. Depth varies B0-B5, B2+B3, or all six blocks while retaining all patches. Spatial support varies at the registered B3 site. Several singleton blocks help; only all-patch spatial support passed the frozen spatial gates. These are separate sweeps, not a crossed layer-by-position experiment. The result does not automatically identify the best site for the rank-4 successor.")

# Figure 5: separate reconstruction diagnostic from forecast endpoint.
fig,axes=plt.subplots(1,2,figsize=(11,4.3))
ratios=[]
for precision in ["float32","bfloat16"]:
    r=reports[("action_response_geometry","pusht",precision)]
    metrics=r["mechanism_diagnostics"]["per_task_group_weighted"]["pusht"]["metrics"]
    ratio=metrics["cubic/omitted_activation_mse"]/metrics["equal_anchor_linear/omitted_activation_mse"]
    ratios.append(ratio)
axes[0].barh([0,1],ratios,color=[TEAL,ORANGE],height=.45)
axes[0].set_xscale("log");axes[0].set_xlim(.0003,6)
axes[0].axvline(1,color="#434950",ls="--",lw=1)
axes[0].set_yticks([0,1],["FP32 sensitivity","BF16 primary"]);axes[0].invert_yaxis()
axes[0].set_title("A  Omitted-activation reconstruction",loc="left",fontweight="bold")
axes[0].set_xlabel("Cubic / equal-anchor linear MSE (log scale)\nBelow 1 favors cubic; descriptive ratios")
for i,v in enumerate(ratios):axes[0].text(v*1.16,i,f"{v:.5f}" if v<.01 else f"{v:.3f}",va="center",fontsize=10)
for y,precision in enumerate(["float32","bfloat16"]):point(axes[1],get("action_response_geometry","pusht","cubic",precision=precision),y,TEAL if y==0 else ORANGE,annotate=False)
forest_style(axes[1],["FP32 sensitivity","BF16 primary"],"B  Forecast correction versus native",(-.25,1.1))
axes[1].axvline(1,color=GRAY,ls=":",lw=1)
axes[1].text(.97,.6,"1% scale reference",rotation=90,va="center",ha="right",fontsize=8,color=GRAY)
fig.suptitle("Better local reconstruction need not improve forecasts",x=.03,ha="left",fontweight="bold",fontsize=16)
fig.tight_layout(rect=(0,0,1,.9),w_pad=3)
save(fig,"fig05_geometry_boundary", "Figure 5. Push-T (21 trajectories): two different endpoints. Left: ratios of group-weighted mean omitted-activation MSE; no uncertainty interval is inferred for these descriptive ratios. Cubic reconstruction is approximately 1,192 times better in FP32 but about 35% worse in BF16. Right: the dose-controlled cubic intervention's H6 proprioceptive embedding error reduction versus native, with source simultaneous 95% intervals. Both forecast intervals include zero. The 1% line is a scale reference, not a recomputed source eligibility boundary. The diagnostic reconstructs a local action-response curve; it does not test all manifold steering methods or establish a cubic physical law.")

# Figure 6: the actual fitted directions, with no invented semantic labels.
basis_summary = json.loads((DATA / "basis_spatial_summary.json").read_text())
fig, axes = plt.subplots(2, 5, figsize=(12, 5.8), layout="constrained")
all_maps = []
for task in ("reach", "reach-wall"):
    data = basis_summary["tasks"][task]
    path = ROOT / data["source"]
    assert sha(path) == data["source_sha256"]
    sources[data["source"]] = data["source_sha256"]
    maps = np.concatenate([np.array(data["patch_squared_loadings"]),
                           np.array(data["subspace_mean_patch_squared_loadings"])[None]], axis=0)
    assert np.allclose(maps.sum(1), 1, atol=2e-5)
    all_maps.append(maps.reshape(5,16,16) * 100)
vmax = max(x.max() for x in all_maps)
for row, (task, maps) in enumerate(zip(("Reach", "Reach-Wall"), all_maps)):
    for col, patch_map in enumerate(maps):
        ax = axes[row,col]
        im = ax.imshow(patch_map, cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest")
        ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
        if row == 0: ax.set_title(f"Direction {col+1}" if col < 4 else "Subspace average", fontsize=11)
        if col == 0: ax.set_ylabel(task, fontsize=12, fontweight="bold")
        if row == 1: ax.set_xlabel("Patch column (16 positions)", fontsize=8)
fig.colorbar(im, ax=axes, shrink=.72, label="Squared loading mass per patch (%)")
fig.suptitle("What the fitted correction basis actually looks like", fontsize=16, fontweight="bold")
save(fig,"fig06_fitted_basis", "Figure 6. Spatial squared loadings of the actual fixed-response rank-4 correction bases, extracted from checksum-verified fitting banks. Each direction contains 256 patches by 400 features. Color sums squared feature loadings within a patch; each panel sums to 100%. The fifth column averages the four unit-direction maps and is invariant to an orthogonal rotation of this subspace. Individual axes depend on the fitted basis; they are not named physical variables or aligned across tasks. These are fixed basis weights, not a per-example edit, attention map, saliency map, image segmentation, or a discovered physical manifold. Common color limits permit comparison without separately rescaling each map. Patch layout follows the model's 16-by-16 raster ordering. No additional model evaluations or fitting were performed.")

manifest={"research_date":"2026-09-08", "scope":"Completed offline reports and existing fitted-basis summaries",
          "new_model_evaluations":0,"partial_behavioral_outcomes_read":False,
          "report_scopes_verified":len(reports),"sources":sources,
          "software":{"python":platform.python_version(),"numpy":np.__version__,"matplotlib":matplotlib.__version__,"seaborn":sns.__version__},
          "statistical_note":"Reuse reported simultaneous intervals; no new bootstrap, tests, selection, or pooled inference.",
          "outputs":{str(p.relative_to(PAPER)):sha(p) for p in sorted(FIG.glob('*'))}}
(DATA/"figure_provenance.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(f"Verified {len(reports)} report scopes; exported {len(rows)} registered contrasts and six PNG/SVG figures.")
