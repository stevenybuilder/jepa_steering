"""Fixed CPU recount of archived pathway factorials and action interpolation.

All source IDs and estimands are fixed in pathway_geometry_protocol.json. No model
calls, refitting, outcome-selected subsets, or scientific-method selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "paper/data"
PROTOCOL = DATA / "pathway_geometry_protocol.json"
GEOMETRY_ARMS = ("equal_anchor_linear", "cubic", "projected_cubic", "reflected_curvature")
MODALITIES = ("proprio", "visual")


def sha(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def factorial_terms(native, visual, action, joint, target):
    """Reference algebra; raw vectors are not reconstructed from saved scalars."""
    n, v, a, j, y = map(lambda x: np.asarray(x, dtype=float),
                         (native, visual, action, joint, target))
    r, dv, da = j - v - a + n, v - n, a - n
    cross = 2 * np.mean(dv * da)
    interaction = np.mean((j-y)**2 - (v-y)**2 - (a-y)**2 + (n-y)**2)
    return dict(output_interaction_mse=float(np.mean(r*r)),
                additive_quadratic_cross=float(cross),
                factorial_mse_interaction=float(interaction),
                nonadditive_mse_remainder=float(interaction-cross))


def group_records(records, expected=None):
    result = {r["lineage_group"]: r for r in records}
    if len(result) != len(records):
        raise ValueError("Duplicate lineage IDs")
    if expected is not None and set(result) != set(expected):
        raise ValueError("Unpaired lineage sets")
    for row in result.values():
        if not np.isfinite(list(row["metrics"].values())).all():
            raise ValueError("Nonfinite source metric")
    return result


def bootstrap_weights(n, seed, replicates):
    return np.random.default_rng(seed).multinomial(n, np.full(n, 1/n), size=replicates) / n


def estimate(values, weights, denominator=None):
    values = np.asarray(values, dtype=float)
    draws = weights @ values
    out = dict(mean=float(values.mean()), marginal_95_low=float(np.quantile(draws, .025)),
               marginal_95_high=float(np.quantile(draws, .975)), n=len(values))
    if denominator is not None:
        denominator = np.asarray(denominator, dtype=float)
        d = float(denominator.mean())
        if d <= 0:
            raise ValueError("Nonpositive normalization denominator")
        normalized = 100 * draws / (weights @ denominator)
        out.update(percent_of_reference=100*out["mean"]/d,
                   percent_95_low=float(np.quantile(normalized, .025)),
                   percent_95_high=float(np.quantile(normalized, .975)))
    return out


def correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"pearson": None, "spearman": None, "n": len(x)}
    return dict(pearson=float(np.corrcoef(x, y)[0, 1]),
                spearman=float(np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1]), n=len(x))


def load_source(source, protocol, verified_archives):
    location = source["source"]
    if "::" in location:
        archive_name, member = location.split("::", 1)
        archive = ROOT / archive_name
        if archive_name not in verified_archives:
            digest = sha(archive)
            if digest != protocol["navigation_archive_sha256"]:
                raise ValueError("Navigation archive checksum mismatch")
            verified_archives[archive_name] = digest
        with tarfile.open(archive, "r:gz") as tf:
            payload = tf.extractfile(member).read()
            done = json.load(tf.extractfile(str(Path(member).with_name("DONE.json"))))
    else:
        path = ROOT / location
        payload = path.read_bytes()
        done_path = path.with_name("DONE.json")
        done = json.loads(done_path.read_text()) if done_path.exists() else None
    actual = hashlib.sha256(payload).hexdigest()
    if actual != source["sha256"] or (done is not None and done["report_sha256"] != actual):
        raise ValueError(f"Source checksum mismatch: {source['id']}")
    report = json.loads(payload)
    if report["protocol_sha256"] != source["protocol_sha256"]:
        raise ValueError("Protocol binding mismatch")
    return report, done is not None


def validate_report(report, source, exported):
    """Verify group identity and every published arm mean before new arithmetic."""
    n = source["independent_lineage_groups"]
    diagnostics = group_records(report["mechanism_diagnostics"]["per_lineage_group"])
    if len(diagnostics) != n or report["independent_lineage_groups"] != n:
        raise ValueError("Wrong independent lineage count")
    full = report.get("aggregation", {}).get("per_arm")
    groups = {a: group_records(v["per_lineage_group"], diagnostics) for a, v in full.items()} if full else None
    means = {}
    for arm, values in report["arms"].items():
        means[arm] = values["lineage_weighted_metrics"]
        for endpoint, mean in means[arm].items():
            e = exported[(exported.source_id == source["id"]) & (exported.arm == arm) & (exported.endpoint == endpoint)]
            if len(e) != 1 or not np.isclose(mean, e.iloc[0].lineage_weighted_mean, rtol=1e-10, atol=1e-14):
                raise ValueError("Published arm means disagree with raw source")
            if groups:
                recounted = np.mean([r["metrics"][endpoint] for r in groups[arm].values()])
                if not np.isclose(recounted, mean, rtol=1e-10, atol=1e-14):
                    raise ValueError("Lineage recount disagrees with source arm mean")
    for endpoint, native in means["native"].items():
        if not np.isclose(means["zero_dose"][endpoint], native, rtol=1e-10, atol=1e-14):
            raise ValueError("Zero-dose identity failed")
        if endpoint.endswith(("_h1", "_h2")):
            if any(not np.isclose(v[endpoint], native, rtol=1e-10, atol=1e-14) for v in means.values()):
                raise ValueError("Pre-edit arm identity failed")
        if groups:
            for key in diagnostics:
                if groups["native"][key]["metrics"][endpoint] != groups["zero_dose"][key]["metrics"][endpoint]:
                    raise ValueError("Per-lineage zero dose is not identical")
    return diagnostics, groups, means


def coupling(source, report, diagnostics, groups, means, weights):
    rows, summary = [], []
    ids = sorted(diagnostics)
    for modality in MODALITIES:
        for horizon in range(1, 7):
            endpoint = f"{modality}_mse_h{horizon}"
            expected = means["joint"][endpoint] - means["visual_only"][endpoint] - means["action_condition_only"][endpoint] + means["native"][endpoint]
            for key in ids:
                diagnostic = diagnostics[key]["metrics"]
                prefix = dict(source_id=source["id"], task=source["task"], precision=source["precision"],
                              lineage_group=key, modality=modality, horizon=horizon)
                values = {}
                if groups:
                    for arm in ("native", "visual_only", "action_condition_only", "joint", "joint_equal_standardized_energy", "matched_random_equal_standardized_energy"):
                        values[f"{arm}_mse"] = groups[arm][key]["metrics"][endpoint]
                    values["factorial_from_arm_losses"] = values["joint_mse"] - values["visual_only_mse"] - values["action_condition_only_mse"] + values["native_mse"]
                for name in ("output_interaction_mse", "additive_quadratic_cross", "factorial_mse_interaction", "nonadditive_mse_remainder"):
                    dkey = f"{modality}_{name}_h{horizon}"
                    if dkey in diagnostic:
                        values[name] = diagnostic[dkey]
                if "factorial_mse_interaction" in values:
                    value = values["factorial_mse_interaction"]
                    cross = values["additive_quadratic_cross"] + values["nonadditive_mse_remainder"]
                    tolerance = 1e-12 + 1e-6 * means["native"][endpoint]
                    if abs(value-cross) > tolerance or (groups and abs(value-values["factorial_from_arm_losses"]) > tolerance):
                        raise ValueError("Saved factorial decomposition identity failed")
                    if values["output_interaction_mse"] < 0:
                        raise ValueError("Negative output interaction squared norm")
                if values:
                    rows.append(dict(prefix, **values))
            block = pd.DataFrame([r for r in rows if r["modality"] == modality and r["horizon"] == horizon])
            base = dict(source_id=source["id"], task=source["task"], precision=source["precision"],
                        modality=modality, horizon=horizon, source_n=len(ids), native_mse=means["native"][endpoint])
            for metric in ("factorial_from_arm_losses", "output_interaction_mse", "additive_quadratic_cross", "nonadditive_mse_remainder"):
                if len(block) and metric in block and block[metric].notna().all():
                    denominator = block.native_mse if groups else np.full(len(ids), means["native"][endpoint])
                    summary.append(dict(base, metric=metric, **estimate(block[metric], weights, denominator)))
            if not groups:
                summary.append(dict(base, metric="factorial_from_arm_losses", mean=expected,
                                    percent_of_reference=100*expected/means["native"][endpoint], n=0,
                                    missing_reason="Compact source omitted per-arm per-lineage losses; only aggregate factorial available"))
            else:
                if abs(block.factorial_from_arm_losses.mean()-expected) > 1e-12:
                    raise ValueError("Mean factorial identity failed")
    return rows, summary


def geometry(source, report, diagnostics, groups, means, weights, contrasts):
    if not groups:
        raise ValueError("Geometry source lacks lineage losses")
    rows, summaries, associations = [], [], []
    ids = sorted(diagnostics)
    for key in ids:
        diag = diagnostics[key]["metrics"]
        for arm in GEOMETRY_ARMS:
            row = dict(source_id=source["id"], task=source["task"], precision=source["precision"],
                       lineage_group=key, arm=arm,
                       omitted_activation_mse=diag[f"{arm}/omitted_activation_mse"],
                       endpoint_line_valid=diag["endpoint_line_valid"],
                       common_energy_edit_eligible=diag["common_energy_edit_eligible"])
            for modality in MODALITIES:
                for h in range(1, 7):
                    row[f"recorded_{modality}_mse_h{h}"] = groups[arm][key]["metrics"][f"{modality}_mse_h{h}"]
                for h in (3, 6):
                    row[f"raw_native_{modality}_fidelity_mse_h{h}"] = diag[f"{arm}/raw_native_{modality}_fidelity_mse_h{h}"]
                    row[f"delivered_native_{modality}_fidelity_mse_h{h}"] = diag[f"{arm}_{modality}_native_fidelity_mse_h{h}"]
                    for zero in ("native", "zero_dose"):
                        if diag[f"{zero}_{modality}_native_fidelity_mse_h{h}"] != 0:
                            raise ValueError("Geometry native-fidelity zero check failed")
            rows.append(row)
    frame = pd.DataFrame(rows)
    metrics = [k for k in frame if "mse" in k]
    for control, intervention in contrasts:
        a = frame[frame.arm == control].set_index("lineage_group").loc[ids]
        b = frame[frame.arm == intervention].set_index("lineage_group").loc[ids]
        for metric in metrics:
            denominator = a[metric]
            if metric.startswith("recorded_"):
                native_endpoint = metric.removeprefix("recorded_")
                denominator = [groups["native"][key]["metrics"][native_endpoint] for key in ids]
            summaries.append(dict(source_id=source["id"], task=source["task"], precision=source["precision"],
                                  control=control, intervention=intervention, metric=metric,
                                  control_mean=float(a[metric].mean()), intervention_mean=float(b[metric].mean()),
                                  **estimate(a[metric]-b[metric], weights, denominator)))
    linear = frame[frame.arm == "equal_anchor_linear"].set_index("lineage_group").loc[ids]
    cubic = frame[frame.arm == "cubic"].set_index("lineage_group").loc[ids]
    x = cubic.omitted_activation_mse-linear.omitted_activation_mse
    for modality in MODALITIES:
        for h in (3, 6):
            y = cubic[f"recorded_{modality}_mse_h{h}"]-linear[f"recorded_{modality}_mse_h{h}"]
            associations.append(dict(source_id=source["id"], task=source["task"], precision=source["precision"],
                                     modality=modality, horizon=h, **correlation(x, y)))
    return rows, summaries, associations


def cross_precision(frame, kind):
    rows = []
    index = ["lineage_group"] + (["arm"] if kind == "geometry" else ["modality", "horizon"])
    metrics = ["omitted_activation_mse", "recorded_proprio_mse_h6", "recorded_visual_mse_h6"] if kind == "geometry" else ["factorial_from_arm_losses", "output_interaction_mse"]
    for task, task_frame in frame.groupby("task", sort=True):
        bf, fp = (task_frame[task_frame.precision == p].set_index(index) for p in ("bfloat16", "float32"))
        if not len(bf) or not len(fp):
            continue
        if set(bf.index) != set(fp.index):
            raise ValueError("Cross-precision lineage coverage differs")
        fp = fp.reindex(bf.index)
        grouping = ["arm"] if kind == "geometry" else ["modality", "horizon"]
        for group, b in bf.groupby(level=grouping[0] if len(grouping) == 1 else grouping):
            f = fp.loc[b.index]
            labels = dict(zip(grouping, group if isinstance(group, tuple) else (group,)))
            for metric in metrics:
                if metric not in b or b[metric].isna().all():
                    continue
                rows.append(dict(kind=kind, task=task, metric=metric, **labels,
                                 bfloat16_mean=float(b[metric].mean()), float32_mean=float(f[metric].mean()),
                                 sign_agreement=float((np.sign(b[metric]) == np.sign(f[metric])).mean()),
                                 **correlation(b[metric], f[metric])))
    return rows


def plots():
    """Rebuild the figure from public derived tables without private raw records."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = pd.read_csv(DATA/"pathway_geometry_geometry_summary.csv")
    summary = summary[summary.control == "equal_anchor_linear"]
    tasks = ("reach", "reach-wall", "pusht", "wall", "pointmaze")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "svg.hashsalt": "pathway_geometry_v1"})
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9))
    fig.subplots_adjust(bottom=.23, top=.81, wspace=.29)
    fig.suptitle("Better local reconstruction does not guarantee a better forecast", fontsize=15, y=.98)
    fig.text(.5, .905, "Four symmetric action donors · omitted native center · development trajectories", ha="center", color="#4e5661", fontsize=10)
    for precision, color, offset, label in (("bfloat16", "#c57d36", -.10, "BF16 primary"),
                                             ("float32", "#187c80", .10, "FP32 sensitivity")):
        f = summary[summary.precision == precision]
        local = f[f.metric == "omitted_activation_mse"].set_index("task").loc[list(tasks)]
        future = f[f.metric == "recorded_proprio_mse_h6"].set_index("task").loc[list(tasks)]
        x = np.arange(len(tasks))+offset
        ratio = local.intervention_mean/local.control_mean
        low, high = 1-local.percent_95_high/100, 1-local.percent_95_low/100
        axes[0].errorbar(x, ratio, yerr=[ratio-low, high-ratio], fmt="o", color=color, label=label, capsize=3, markersize=6)
        values = future.percent_of_reference
        axes[1].errorbar(x, values, yerr=[values-future.percent_95_low, future.percent_95_high-values],
                         fmt="o", color=color, label=label, capsize=3, markersize=6)
    axes[0].axhline(1, color="#7c8790", ls="--", lw=1)
    axes[0].set(yscale="log", ylim=(.00003, 3), ylabel="Cubic / linear activation MSE · lower is better",
                title="Raw reconstruction: precision reverses the comparison")
    axes[0].legend(loc="lower left", frameon=False, fontsize=9)
    axes[1].axhline(0, color="#7c8790", ls="--", lw=1)
    axes[1].set(ylabel="Cubic advantage (% of native H6 proprio MSE)", title="Requested-dose-matched forecast effects are mixed")
    for ax in axes:
        ax.set_xticks(range(5), ["Reach", "Reach-Wall", "Push-T", "Wall", "PointMaze"], rotation=18)
        ax.grid(axis="y", alpha=.15)
    fig.text(.5, .02, "n = 33 / 27 / 21 / 192 / 200 lineages; marginal paired-bootstrap 95% intervals.\n"
             "Left: raw interpolation. Right: matched requested doses, same recorded action; realized energy can differ. Neither is robot success.",
             ha="center", fontsize=9, color="#4e5661")
    directory = ROOT/"docs/figures"
    directory.mkdir(exist_ok=True)
    for extension in ("png", "svg", "pdf"):
        metadata = {"Date": None} if extension == "svg" else {"CreationDate": None, "ModDate": None} if extension == "pdf" else None
        fig.savefig(directory/f"pathway_geometry_precision.{extension}", dpi=180, facecolor="white",
                    metadata=metadata)
    plt.close(fig)


def run():
    protocol = json.loads(PROTOCOL.read_text())
    for key in ("sources_registry", "published_metrics"):
        if sha(ROOT/protocol[key]) != protocol[f"{key}_sha256"]:
            raise ValueError("Frozen source inventory changed")
    registry = json.loads((ROOT/protocol["sources_registry"]).read_text())
    exported = pd.read_csv(ROOT/protocol["published_metrics"])
    sources = [s for s in registry["forecast_sources"] if s["id"] in protocol["source_ids"]]
    if [s["id"] for s in sources] != protocol["source_ids"]:
        raise ValueError("Frozen source coverage differs")
    coupling_rows, coupling_summary, geometry_rows, geometry_summary, associations, audit = [], [], [], [], [], []
    verified_archives = {}
    for source in sources:
        report, done = load_source(source, protocol, verified_archives)
        diagnostics, groups, means = validate_report(report, source, exported)
        weights = bootstrap_weights(len(diagnostics), protocol["bootstrap_seed"]+int(source["id"][1:]), protocol["bootstrap_replicates"])
        if source["category"] == "vision_action_coupling":
            rows, summary = coupling(source, report, diagnostics, groups, means, weights)
            coupling_rows.extend(rows)
            coupling_summary.extend(summary)
        else:
            rows, summary, association = geometry(source, report, diagnostics, groups, means, weights, protocol["geometry_contrasts"])
            geometry_rows.extend(rows)
            geometry_summary.extend(summary)
            associations.extend(association)
        audit.append(dict(source, adjacent_done_verified=done, per_arm_lineage_losses_available=bool(groups),
                          per_lineage_diagnostics=len(diagnostics), original_protocol_sha256=report.get("original_protocol_sha256"),
                          full_analysis_report_sha256=report.get("full_analysis_report_sha256"),
                          arm_energy={a: {k: values[k] for k in ("requested_l2_mean", "realized_l2_mean", "realized_energy_match_fraction") if k in values}
                                      for a, values in report["arms"].items()}))
        print(f"Verified {source['id']}: {source['task']} {source['precision']} {source['category']} n={len(diagnostics)}", flush=True)
    frames = {"coupling_lineages": pd.DataFrame(coupling_rows), "coupling_summary": pd.DataFrame(coupling_summary),
              "geometry_lineages": pd.DataFrame(geometry_rows), "geometry_summary": pd.DataFrame(geometry_summary),
              "associations": pd.DataFrame(associations)}
    frames["cross_precision"] = pd.DataFrame(cross_precision(frames["coupling_lineages"], "coupling") + cross_precision(frames["geometry_lineages"], "geometry"))
    outputs = {}
    for name, frame in frames.items():
        path = DATA/f"pathway_geometry_{name}.csv"
        frame.to_csv(path, index=False)
        outputs[str(path.relative_to(ROOT))] = {"sha256": sha(path), "rows": len(frame)}
    result = dict(schema_version=1, status="completed_exploratory_archival_reanalysis", protocol_sha256=sha(PROTOCOL),
                  analysis_source_sha256=sha(__file__), source_audit=audit, verified_archives=verified_archives,
                  outputs=outputs, interval=protocol["interval"],
                  interpretation="Output nonadditivity is a recorded scalar squared norm, not reconstructed output vectors; interpolation fidelity is not physical future accuracy or a dense manifold.",
                  limitations=["H2/H4/H5 lack saved output decomposition", "Wall coupling compact source lacks per-arm lineage losses; aggregate loss interactions retained",
                               "No PointMaze coupling or FP32 Wall coupling in frozen registered source set", "One original checkpoint per task; precision-specific fitted banks; no training-seed replication",
                               "Saved scalar diagnostics are recounted and identity-checked; original output tensors are not reexecuted"])
    (DATA/"pathway_geometry.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plots-only", action="store_true", help="Rebuild the figure from public derived tables")
    args = parser.parse_args()
    if not args.plots_only:
        result = run()
        print(f"Completed {len(result['source_audit'])} verified source reports; {len(result['outputs'])} derived tables")
    plots()
