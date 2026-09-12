"""Render existing completed reports; no fitting, inference, or statistical reruns."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "paper/data"
TABLE = ROOT / "paper/tables/all_task_ablation_audit.md"
TASKS = ["reach", "reach-wall", "pusht", "wall", "pointmaze"]
NAMES = {"reach": "Reach", "reach-wall": "Reach-Wall", "pusht": "Push-T", "wall": "Wall", "pointmaze": "PointMaze"}
CATS = ["vision_action_coupling", "action_response_geometry", "operator_rank", "distribution_layer", "distribution_spatial", "combined", "fixed_response"]
CAT_NAMES = dict(zip(CATS, ["Vision/action coupling", "Action-response geometry", "Operator rank", "Layer distribution (total rank 1)", "Spatial support (rank 1)", "Original combined/drop-one", "Revised fixed response"]))
EPS = ["proprio_mse_h6", "visual_mse_h6"]


def digest(b):
    return hashlib.sha256(b).hexdigest()


def archive_reports(path, prefix):
    blobs = {}
    with tarfile.open(path) as t:
        for m in t:
            if m.isfile() and m.name.startswith(prefix) and m.name.endswith(("/report.json", "/DONE.json")):
                blobs[m.name] = t.extractfile(m).read()
    result = []
    for name, b in blobs.items():
        if not name.endswith("/report.json"):
            continue
        done = json.loads(blobs[str(Path(name).parent / "DONE.json")])
        assert done["report_sha256"] == digest(b), name
        result.append((json.loads(b), f"{path.relative_to(ROOT)}::{name}", digest(b)))
    return result


def sign(c):
    lo, hi = c["simultaneous_95_reduction_percent_of_observed_native"]
    if lo > 0:
        return "positive"
    if hi < 0:
        return "negative"
    if lo == hi == 0:
        return "identity"
    return "inconclusive"


def cell(c):
    x = c["error_reduction_percent_of_native"]
    lo, hi = c["simultaneous_95_reduction_percent_of_observed_native"]
    return f"{x:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def explanation(task, cat, arm):
    if arm == "zero_dose":
        return "Identity check: zero edit reproduces the native output."
    if "matched_random" in arm:
        return "Matched direction control: tests whether the learned basis is more useful than perturbation alone; a positive control is not evidence for the proposed direction."
    if "random_position" in arm:
        return "Position control: tests the chosen patch placement at fixed support size; does not identify a physical object or circuit."
    if cat == "vision_action_coupling":
        if "permuted" in arm:
            return "Alignment control: rearranges patch assignment; direct joint-versus-permuted contrasts are in the full contrast export."
        if arm == "action_condition_only":
            return "Small conditioning-path effect may reflect correctable action encoding; raw actions are unchanged. Why it helps or fails is untested."
        if task == "pusht":
            return "Visual-only and unscaled joint harm the primary endpoint. Possible mismatch with contact-dependent dynamics; no contact-mediated causal test establishes this explanation."
        if task == "wall":
            return "Learned visual/joint directions do not reliably reduce native error; positive random-control effects weaken a learned-direction explanation. Physical cause unknown."
        if task == "reach-wall":
            return "Proprioceptive gains coexist with inconclusive visual changes. A modality-specific correction is plausible; success and wall-clearance effects are unmeasured here."
        return "Consistent with correction through visual and conditioning pathways. Joint doses and drop-one energy differences preclude an automatic synergy claim."
    if cat == "action_response_geometry":
        return "Local reconstruction and forecast correction are different targets. A small correction, readout mismatch, and quantization are possible explanations; forecast nulls do not identify which dominates."
    if cat == "operator_rank":
        if task == "pusht":
            return "All tested ranks are inconclusive; the selected subspace/site may miss useful error structure, or effects may be small/variable. Neither explanation is established."
        return "Consistent with compact correctable error. Four/eight have similar point means; rank denotes correction capacity, not the dimension of physics."
    if cat == "distribution_layer":
        if task == "pusht":
            return "No tested placement establishes an effect; moving this rank-1 edit does not establish a useful Push-T correction. Cause unresolved."
        return "Earlier blocks have larger point effects, possibly because more downstream computation propagates the edit. This is not a localized circuit or an independent rank-4 placement test."
    if cat == "distribution_spatial":
        if task == "pusht":
            return "No tested support establishes an effect; spatial redistribution alone has not rescued this rank-1 correction. Cause unresolved."
        return "Distributed support and patch arrangement affect point gains. Small supports can help statistically while failing usefulness/control gates; semantic spatial localization is untested."
    if cat == "combined":
        return "Combined effects support complementary correction in this Reach configuration; removal also removes energy. These are the original expensive operators, not the revised combination."
    return "An offline mean response retains forecast benefits. This supports amortizing response estimation, not old/new equivalence or demonstrated task-success gains."


def main():
    rows = list(csv.DictReader((DATA / "reported_contrasts.csv").open()))
    reports = []
    for source in dict.fromkeys(r["source"] for r in rows):
        row = next(r for r in rows if r["source"] == source)
        p = ROOT / source
        b = p.read_bytes()
        assert digest(b) == row["source_sha256"], source
        done = json.loads((p.parent / "DONE.json").read_text())
        assert done["report_sha256"] == digest(b), source
        r = json.loads(b)
        r["category"] = row["category"]
        r["task"] = row["task"]
        reports.append((r, source, digest(b), "report hash matches export and adjacent DONE"))
    compact = ROOT / "artifacts/offline_study/navigation-analysis-compact-20260907-v1/bfloat16/wall/vision_action_coupling/report.json"
    b = compact.read_bytes()
    reports.append((json.loads(b), str(compact.relative_to(ROOT)), digest(b), "preserved compact report; original full hash recorded, full raw report not reread"))
    nav_archive = ROOT / "artifacts/offline_study/navigation-geometry-closure-preservation-20260908-v1/navigation-geometry-results.tar.gz"
    assert digest(nav_archive.read_bytes()) == "63ff98086d7be6534494a4d45075b02d05eea52f4d9395e30a93c5e3cfffcb1a"
    for r, src, sha in archive_reports(nav_archive, "navigation-analysis-"):
        reports.append((r, src, sha, "archive checksum and member report/DONE match"))
    reports.sort(key=lambda x: (TASKS.index(x[0]["task"]), CATS.index(x[0]["category"]), x[0]["precision"]))
    sources = []
    all_contrasts = []
    metrics = []
    for i, (r, src, sha, check) in enumerate(reports, 1):
        r["audit_source_id"] = f"S{i:02d}"
        source = dict(id=r["audit_source_id"], task=r["task"], category=r["category"], precision=r["precision"], source=src, sha256=sha, verification=check, independent_lineage_groups=r["independent_lineage_groups"], protocol_sha256=r.get("protocol_sha256"))
        sources.append(source)
        for c in r["contrasts"]:
            lo, hi = c["simultaneous_95_reduction_percent_of_observed_native"]
            all_contrasts.append(dict(source_id=source["id"], task=r["task"], category=r["category"], precision=r["precision"], candidate=c["candidate"], control=c["control"], endpoint=c["endpoint"], error_reduction_percent_of_native=c["error_reduction_percent_of_native"], simultaneous_95_lower=lo, simultaneous_95_upper=hi, statistical_result=sign(c), beats_control_by_frozen_minimum=c["beats_control_by_frozen_minimum"], independent_lineages=r["independent_lineage_groups"], source=src, source_sha256=sha))
        for arm, a in r["arms"].items():
            for endpoint, value in a["lineage_weighted_metrics"].items():
                metrics.append(dict(source_id=source["id"], task=r["task"], category=r["category"], precision=r["precision"], arm=arm, endpoint=endpoint, lineage_weighted_mean=value, clip_weighted_mean=a.get("clip_weighted_metrics", {}).get(endpoint), error_reduction_percent_vs_native=a["error_reduction_percent_vs_native"][endpoint], realized_energy_match_fraction=a.get("realized_energy_match_fraction")))
    for name, values in [("all_task_ablation_contrasts.csv", all_contrasts), ("all_task_ablation_metrics.csv", metrics)]:
        with (DATA / name).open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(values[0])); w.writeheader(); w.writerows(values)
    droid_path = ROOT / "artifacts/offline_study/live-20260908T151000Z/instance-50259194-droid-parallel/instance-50259194-results.tar.gz"
    droid, droid_src, droid_sha = archive_reports(droid_path, "droid-coupling-behavior-20260908-v3/analysis/")[0]
    assert droid_sha == "b998218db8f1b87d18150bed04b9eec29731ef50af5e8c203f7ba028b9f83de8"
    (DATA / "all_task_ablation_sources.json").write_text(json.dumps(dict(checked_at=datetime.now(timezone.utc).isoformat(), verification_scope="Completed aggregate hashes and completion receipts, not a new raw-trajectory or statistical reproduction", forecast_sources=sources, droid_source=dict(source=droid_src, sha256=droid_sha, verification="report and archived DONE match"), unresolved=["PointMaze coupling BF16/FP32 and Wall coupling FP32 are reported complete in execution records but their numerical aggregate reports were not recovered for this audit", "Navigation rank/layer/spatial fits are not completed comparison panels"]), indent=2)+"\n")
    lines = ["# All-task ablation results and interpretation", "", "Checked September 8, 2026. This renders completed, preserved development reports. No experiment, fitting, new statistical test, or incomplete behavioral-outcome inspection was performed.", "",
        "## How to read the tables", "",
        "Each forecast cell is **percentage error reduction versus native [simultaneous 95% interval]**. Positive means lower error; negative means higher error. P = H6 proprioceptive embedding MSE; V = H6 visual embedding MSE. Neither is task success, and P is not decoded physical-state error.", "",
        "Intervals are paired by source trajectory/lineage, simultaneous across the registered contrasts and both endpoints **within task/category/precision**, not across this entire document. They are difference intervals scaled by the observed native mean, not confidence intervals for a population error ratio. BF16 is primary; FP32 is sensitivity. A positive interval is statistical evidence for that comparison, not necessarily a practically useful effect. The frozen minimum is 1% of fit-only native error; qualifying also requires the registered controls and implementation/energy checks. It is not a requirement that each observed percentage simply exceed 1%.", "",
        "Reach has 33 development lineages, Reach-Wall 27, Push-T 21, Wall 192 and PointMaze 200. The same lineages recur across arms and precisions. Reach and Reach-Wall share the released MetaWorld predictor but have separate fitted corrections. Many comparisons reuse the same intervention: rank 1 at B3, layer B3, and all-patch rank 1 have matching point estimates but different simultaneous contrast families. These are not three independent discoveries.", "",
        "The interpretations below distinguish observed statistical patterns from **possible, untested mechanisms**. An inconclusive interval does not prove zero effect. A negative interval establishes harm to this endpoint, not necessarily to task success.", "",
        "## Why these comparisons were selected", "",
        "JEPA-WM compares inherited architecture/training/planning alternatives one factor at a time, then evaluates behavior. Its choices test concrete concerns such as rollout error, action conditioning and missing state information; the paper does not establish when each rationale was formulated relative to its experiments. See [Table 1 and Section 4](https://arxiv.org/html/2512.24497v4#S4).", "",
        "Our [frozen plan](../../docs/EXPERIMENT_PLAN.md) asks whether frozen-model interventions have selective forecast effects that survive later candidate ranking and control. Rank tests correction capacity; pathway controls test vision/action alignment and dose; layer and spatial sweeps separate depth from support; linear/cubic and curvature controls test local response geometry; the revised operator tests moving response estimation offline. Registration and controls make these questions interpretable, but do not prove that the sites, dose, metric, or compute allocation were optimal.", "",
        "The B3/H3 reference is an experimental choice, not an independently discovered physics zone. The original online response diagnostics were expensive inside planning. The present evidence supports a task-specific adaptation study, not transfer of one fitted operator or universal steering. The individual visual/action coupling arms lack their own scope-matched random-direction arms in the original registry; their native effects do not alone establish learned-direction specificity.", "",
        "## Coverage", "",
        "| Family | Reach | Reach-Wall | Push-T | Wall | PointMaze | DROID |", "|---|---|---|---|---|---|---|",
        "| Coupling | Complete BF16/FP32 | Complete BF16/FP32 | Complete BF16/FP32 | BF16 numbers below; FP32 reported complete, aggregate unavailable here | Reported complete BF16/FP32; aggregates unavailable here | Complete recorded-plan score panel |",
        "| Geometry | Complete BF16/FP32 | Complete BF16/FP32 | Complete BF16/FP32 | Complete BF16/FP32 | Complete BF16/FP32 | No completed panel verified |",
        "| Rank / layer / spatial | Complete BF16/FP32 | Complete BF16/FP32 | Complete BF16/FP32 | Fits complete; comparison panels not verified complete | Fits complete; comparison panels not verified complete | No completed panels verified |",
        "| Original combined/drop-one | Complete BF16/FP32 | Not run | Not run | Not run | Not run | Not run |",
        "| Revised rank 4 | Complete BF16/FP32 | Complete BF16/FP32 | Not run | Not run | Not run | Not run |",
        "| HMM / revised combined behavior | No completed efficacy panel | No completed HMM efficacy panel | Not run | Not run | Not run | Not run |", "",
        "The closed-loop comparison is a separate endpoint. HMM, additional combined behavior, navigation expansion and training histories are paused under the existing core-panel priority. Missing or paused comparisons are not null findings. No completed task-success result is inferred from this audit.", "",
        "## BF16 primary results: every measured arm versus native", ""]
    for precision in ["bfloat16", "float32"]:
        if precision == "float32":
            lines += ["## FP32 sensitivity: every measured arm versus native", "", "These rows do not replace the BF16 primary analysis. Original sweeps use their precision-specific fits; the revised fixed-response sensitivity uses the same frozen BF16-fitted map.", ""]
        for task in TASKS:
            lines += [f"### {NAMES[task]} — {precision}", ""]
            for r, _, _, _ in reports:
                if r["task"] != task or r["precision"] != precision:
                    continue
                cat = r["category"]
                native = r["arms"]["native"]["lineage_weighted_metrics"]
                lines += [f"#### {CAT_NAMES[cat]}", "", f"Source {r['audit_source_id']}; n={r['independent_lineage_groups']}. Native absolute H6 MSE: P={native[EPS[0]]:.10g}, V={native[EPS[1]]:.10g}.", "", "| Arm | P reduction % [95% interval] | V reduction % [95% interval] | Statistical result P / V | Interpretation / possible explanation |", "|---|---:|---:|---|---|"]
                contrasts = {(c["candidate"], c["endpoint"]): c for c in r["contrasts"] if c["control"] == "native"}
                for arm in dict.fromkeys(c[0] for c in contrasts):
                    p, v = [contrasts[(arm, e)] for e in EPS]
                    lines.append(f"| {arm} | {cell(p)} | {cell(v)} | {sign(p)} / {sign(v)} | {explanation(task, cat, arm)} |")
                lines += [""]
            if task == "pointmaze" or (task == "wall" and precision == "float32"):
                lines += ["**Coupling numerical-evidence gap:** completion is recorded, but the relevant aggregate was not available in the verified export. No zero or negative effect is assigned. The registered non-native arms are visual_only, action_condition_only, joint, joint_equal_standardized_energy, permuted_visual, permuted_joint, matched_random, matched_random_equal_standardized_energy, and zero_dose.", ""]
    lines += ["## DROID: all completed coupling arms", "", "Metric: recorded-plan action score, higher is better. These are score-point differences, not error-reduction percentages or robot success. Each arm has 64 endpoints from the same 15 recordings; zero physical robot executions. All 16 registered simultaneous 95% contrast intervals include zero.", "", "| Arm | Action score | XYZ action error | Score difference vs native [95% interval] | Interpretation / possible explanation |", "|---|---:|---:|---:|---|"]
    for arm, a in droid["arm_summaries"].items():
        if arm == "native":
            diff = "Reference"
        else:
            c = next(c for c in droid["contrasts"] if c["contrast"] == arm + "_vs_native")
            lo, hi = c["simultaneous_95_interval_score_points"]
            diff = f"{c['score_difference']:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        why = "Baseline." if arm == "native" else "Inconclusive. Small/variable score changes and only 15 recording clusters limit inference; action imitation is not task success. No established causal explanation."
        lines.append(f"| {arm} | {a['checkpoint_score']:.3f} | {a['mean_xyz_action_error']:.6f} | {diff} | {why} |")
    lines += ["", "### All DROID registered paired contrasts", "", "| Contrast | Score difference [simultaneous 95% interval] |", "|---|---:|"]
    for c in droid["contrasts"]:
        lo, hi = c["simultaneous_95_interval_score_points"]
        lines.append(f"| {c['contrast']} | {c['score_difference']:+.4f} [{lo:+.4f}, {hi:+.4f}] |")
    lines += ["", "## Control comparisons and all-horizon metrics", "", "The [full H6 contrast CSV](../data/all_task_ablation_contrasts.csv) retains every registered contrast, including candidate-versus-random, cross-rank/support and drop-one contrasts, both endpoints and both precisions. Its frozen-minimum column is **per contrast**, not a claim that the complete arm gate passed. The [all-horizon metric CSV](../data/all_task_ablation_metrics.csv) retains all reported per-arm L1/MSE horizon means and error reductions, including native. Statistical intervals were registered for the H6 contrast family; no new intervals or tests were manufactured for other horizons.", "", "## Verified sources", "", "[Machine-readable source ledger](../data/all_task_ablation_sources.json). This audit checks existing aggregate hashes/completion bindings; it does not repeat the underlying trajectory bootstrap or every raw-file check. Original archives and scientific records remain unchanged.", ""]
    for s in sources:
        path = s["source"].split("::")[0]
        lines.append(f"- **{s['id']}** {NAMES[s['task']]} / {s['category']} / {s['precision']}: [{path}](../../{path}). SHA256 `{s['sha256']}`. {s['verification']}.")
    lines += [f"- **DROID** [{droid_src.split('::')[0]}](../../{droid_src.split('::')[0]}). Member `{droid_src.split('::')[1]}`; report SHA256 `{droid_sha}` matches archived DONE.", ""]
    TABLE.write_text("\n".join(lines))
    print(json.dumps(dict(forecast_reports=len(reports), forecast_contrasts=len(all_contrasts), all_horizon_metric_rows=len(metrics), droid_contrasts=len(droid["contrasts"]), markdown=str(TABLE), source_sha256=digest((DATA / "all_task_ablation_sources.json").read_bytes())), indent=2))


if __name__ == "__main__":
    main()
