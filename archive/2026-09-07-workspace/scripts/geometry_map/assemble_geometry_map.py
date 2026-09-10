#!/usr/bin/env python3
"""Join measured geometry evidence into JSON/CSV and traceable SVG summaries.

This assembles evidence, not a completion claim. A missing measurement is never
filled from a different target, a whole-block patch, or a mislabeled proxy.
Existing source reports stay immutable; corrected labels are explicit aliases.
Only the standard library is required, so assembly needs no GPU or model load.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path


ALIASES = {
    "progress": "cumulative_progress",
    "straight_path_intersects_wall": "hand_to_goal_segment_intersects_wall",
    "action_translation_magnitude": "action_chunk_frobenius_magnitude",
    "motion_magnitude": "realized_hand_delta_magnitude",
    "wall_clearance": "wall_signed_distance",
}
CORE_FIELDS = (
    "linear_score", "nonlinear_gain", "best_coordinate_frame", "geometry",
    "cross_trajectory_stability", "regime_dependence", "causal_patch_effect",
    "specificity", "manifold_distance", "recommended_operator",
)


def canonical(target):
    return ALIASES.get(target, target)


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    return value


def dump(path, value):
    path.write_text(json.dumps(finite(value), indent=2, allow_nan=False) + "\n")


def load_source(path, sources):
    if not path.is_file():
        return None
    value = json.loads(path.read_text())
    sources[str(path)] = {"path": str(path.resolve()), "sha256": sha256(path)}
    return value


def key(family, layer, target):
    return family, int(layer), canonical(target)


def new_row(family, layer, target):
    target = canonical(target)
    return {
        "row_id": f"reach-wall/{family}/block-{layer}/spatial-mean/{target}",
        "task": "reach-wall", "model": "released_jepa_wm_metaworld",
        "module": family, "block": int(layer), "hook": "block_residual_output",
        "spatial_region": "mean_of_256_spatial_tokens", "task_variable": target,
        "imagined_timestep": 1 if family == "predictor" else None,
        "imagined_timestep_status": "isolated_one_step" if family == "predictor" else "not_applicable_encoder",
        "evidence_scope": "discovery_cross_validation_not_untouched_confirmation",
        **{field: None for field in CORE_FIELDS},
        "field_status": {field: "pending_measurement" for field in CORE_FIELDS},
        "recommended_operator": "none_validated",
        "operator_reason": "No complete held-out causal validation package yet; not evidence all operators fail.",
        "sources": [],
    }


def measured(row, field, value, status="measured_discovery_only"):
    row[field] = value
    row["field_status"][field] = status


def source_ref(row, path, item):
    row["sources"].append({"path": str(path), "item": item})


def join_evidence(root, geometry_path=None, causal_path=None):
    sources, index = {}, {}

    def row_for(family, layer, target):
        k = key(family, layer, target)
        return index.setdefault(k, new_row(*k))

    discovery_path = root / "discovery-screen/discovery_screen.json"
    discovery = load_source(discovery_path, sources)
    if discovery is None:
        raise FileNotFoundError(discovery_path)
    for number, old in enumerate(discovery["rows"]):
        row = row_for(old["family"], old["layer"], old["target"])
        measured(row, "linear_score", {k: old[k] for k in ("metric", "score", "time_baseline", "delta_vs_time")})
        measured(row, "geometry", {k: old[k] for k in ("participation_ratio", "top10_variance_fraction", "adjacent_timestep_cosine_mean")}, "descriptive_activation_geometry")
        row["independent_episode_count"] = discovery["episode_count"]
        row["sample_count"] = discovery["sample_count"]
        source_ref(row, discovery_path, f"rows[{number}]")

    emergence_path = root / "emergence-geometry/emergence_geometry.json"
    emergence = load_source(emergence_path, sources)
    if emergence:
        for family, variables in emergence["curves"].items():
            for target, curve in variables.items():
                for point in curve["rows"]:
                    row = row_for(family, point["layer"], target)
                    if row["linear_score"] is None:
                        measured(row, "linear_score", {"metric": "r2", "score": point["r2"], "time_baseline": None, "delta_vs_time": None})
                    row["direction_magnitude_curve"] = point
                    source_ref(row, emergence_path, f"curves.{family}.{target}.rows.layer={point['layer']}")
        # Historical cross-metric principal angles are intentionally NOT imported.

    nonlinear_path = root / "cached-geometry-v2/nonlinear_screen.json"
    nonlinear = load_source(nonlinear_path, sources)
    if nonlinear:
        for number, old in enumerate(nonlinear["rows"]):
            row = row_for(old["family"], old["block"], old["target"])
            measured(row, "nonlinear_gain", {k: old[k] for k in ("knn_score", "linear_score", "nonlinear_gain")})
            source_ref(row, nonlinear_path, f"rows[{number}]")

    spatial_path = root / "spatial-screen/spatial_screen.json"
    spatial = load_source(spatial_path, sources)
    if spatial:
        for name, scores in spatial["maps"].items():
            family, block, target = name.split(".", 2)
            row = row_for(family, int(block.removeprefix("block")), target)
            row["spatial_decodability"] = {"grid_shape": [16, 16], "scores": scores, "scope": "fixed_grid_population_readability_not_causal_token_localization"}
            source_ref(row, spatial_path, f"maps.{name}")

    geometry_path = geometry_path or root / "cached-geometry-v2/cached_geometry.json"
    cached = load_source(geometry_path, sources)
    if cached:
        for number, item in enumerate(cached.get("rows", [])):
            row = row_for(item["family"], item["layer"], item["target"])
            if row["linear_score"] is not None:
                row["earlier_readout_measurement"] = row["linear_score"]
            measured(row, "linear_score", {k: item.get(k) for k in ("metric", "score", "time_baseline", "delta_vs_time")})
            row["readout_protocol"] = {"source": str(geometry_path), "samples": item.get("samples"), "folds": item.get("folds"), "note": "Use this report's sample masks and multivariate score; do not splice into an older differently masked emergence curve."}
            for field, source in (("regime_dependence", "regime_dependence"), ("cross_trajectory_stability", "bootstrap_stability")):
                if item.get(source) is not None:
                    measured(row, field, item[source])
            source_ref(row, geometry_path, f"rows[{number}]")
        for site_number, site in enumerate(cached.get("sites", [])):
            for (family, layer, _target), row in index.items():
                if (family, layer) != (site["family"], site["layer"]):
                    continue
                site_id = f"{family}.block{layer}"
                row["site_geometry_id"] = site_id
                row["geometry"] = {**(row["geometry"] or {}), "eigenspectrum_reference": f"sites.{site_id}.eigenspectrum", "common_metric_subspace_comparisons_reference": f"sites.{site_id}.subspace_comparisons"}
                row["field_status"]["geometry"] = "descriptive_common_metric_discovery_geometry"
                # A natural-state distance alone cannot measure edited-state support.
                row["natural_activation_support_reference"] = f"sites.{site_id}.natural_support"
                if canonical(row["task_variable"]) == "realized_motion_world" and site.get("coordinate_frames"):
                    measured(row, "best_coordinate_frame", site["coordinate_frames"], "measured_candidate_frame_comparison_not_unique_native_coordinates")
                source_ref(row, geometry_path, f"sites[{site_number}]")

    causal = load_source(causal_path, sources) if causal_path else None
    if causal:
        row = row_for("predictor", causal.get("block", 3), "realized_xz_direction")
        measured(row, "causal_patch_effect", {"reference": "causal_measurements", "independent_starts": len({(r['episode'], r['replan']) for r in causal.get('rows', [])}), "arm_count": len(causal.get('rows', []))}, "measured_development_diagnostic_not_confirmed_mechanism")
        source_ref(row, causal_path, "rows_and_timelines")
        for field in ("specificity", "manifold_distance"):
            if causal.get(field) is not None:
                measured(row, field, causal[field], "development_diagnostic")

    rows = sorted(index.values(), key=lambda r: (r["module"], r["block"], r["task_variable"]))
    for row in rows:
        row["field_status"]["recommended_operator"] = "not_yet_justified"
    return rows, sources, cached, causal


def svg_open(width, height, title, subtitle=""):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<style>text{font-family:Arial,sans-serif;fill:#17212b;font-size:12px}</style>',
            f'<text x="22" y="28" style="font-size:19px">{html.escape(title)}</text>',
            f'<text x="22" y="49">{html.escape(subtitle)}</text>']


def color(value, limit=0.5):
    t = min(abs(value) / limit, 1.0)
    target = (39, 104, 168) if value >= 0 else (180, 60, 48)
    return "#" + "".join(f"{round(248 + t * (v - 248)):02x}" for v in target)


def heatmap(rows, path):
    targets = list(dict.fromkeys(r["task_variable"] for r in rows))
    sites = [(family, n) for family, count in (("encoder", 12), ("predictor", 6)) for n in range(count)]
    lookup = {(r["module"], r["block"], r["task_variable"]): r for r in rows}
    svg = svg_open(1390, 108 + len(targets) * 27, "Layer × variable: held-collection-seed gain over time", "Numbers are score differences (R² or AUROC by target); color clips at ±0.5; gray = not measured.")
    for col, (family, block) in enumerate(sites):
        svg.append(f'<text x="{340 + col * 56}" y="77">{family[0].upper()}{block}</text>')
    for line, target in enumerate(targets):
        y = 86 + line * 27
        svg.append(f'<text x="20" y="{y + 17}">{html.escape(target)}</text>')
        for col, (family, block) in enumerate(sites):
            row = lookup.get((family, block, target))
            score = row["linear_score"] if row else None
            value = score.get("delta_vs_time") if score else None
            valid = isinstance(value, (int, float)) and math.isfinite(value)
            fill, label = (color(value), f"{value:.2f}") if valid else ("#dddddd", "—")
            x = 324 + col * 56
            svg.append(f'<rect x="{x}" y="{y}" width="54" height="24" fill="{fill}"/><text x="{x + 5}" y="{y + 17}" style="font-size:11px">{label}</text>')
    path.write_text("\n".join(svg + ["</svg>"]))


def spatial_plot(rows, path, frame=None):
    selected = [r for r in rows if r.get("spatial_decodability") and r["task_variable"] in ("cumulative_progress", "wall_signed_distance", "realized_dz", "hand_to_goal_segment_intersects_wall")]
    svg = svg_open(1150, 95 + math.ceil(len(selected) / 4) * 300, "Spatial token readability", "Held-seed population scores; fixed-grid association, not a causal localization or single-image attribution.")
    for i, row in enumerate(selected):
        x, y = 20 + i % 4 * 282, 95 + i // 4 * 300
        svg.append(f'<text x="{x}" y="{y - 14}">{row["module"]} {row["block"]} · {html.escape(row["task_variable"])}</text>')
        if frame:
            uri = "data:image/png;base64," + base64.b64encode(frame.read_bytes()).decode()
            svg.append(f'<image x="{x}" y="{y}" width="256" height="256" href="{uri}"/>')
        for token, score in enumerate(row["spatial_decodability"]["scores"]):
            if score is None or not math.isfinite(score):
                fill = "#cccccc"
            else:
                fill = color(score, 1.0)
            svg.append(f'<rect x="{x + token % 16 * 16}" y="{y + token // 16 * 16}" width="16" height="16" fill="{fill}" opacity="{0.52 if frame else 1}"><title>{html.escape(row["row_id"])} token={token} score={score}</title></rect>')
    path.write_text("\n".join(svg + ["</svg>"]))


def line_plot(path, title, series, xlabel, ylabel, subtitle=""):
    if not series:
        return False
    values = [(float(x), float(y)) for _, points in series for x, y in points if math.isfinite(float(x)) and math.isfinite(float(y))]
    if not values:
        return False
    xs, ys = zip(*values)
    x0, x1, y0, y1 = min(xs), max(xs), min(0, min(ys)), max(ys)
    sx = lambda x: 90 + 680 * (x - x0) / max(x1 - x0, 1e-9)
    sy = lambda y: 410 - 320 * (y - y0) / max(y1 - y0, 1e-9)
    svg = svg_open(1050, 495, title, subtitle)
    svg += ['<path d="M90 85 V410 H785" fill="none" stroke="#777"/>', f'<text x="365" y="470">{html.escape(xlabel)}</text>', f'<text x="20" y="75">{html.escape(ylabel)}</text>']
    palette = ["#286aa6", "#c8543b", "#478750", "#8a5eab", "#ba8c27", "#3d9198"]
    for tick in range(6):
        xv, yv = x0 + (x1 - x0) * tick / 5, y0 + (y1 - y0) * tick / 5
        svg += [f'<text x="{sx(xv) - 8}" y="433">{xv:.2g}</text>', f'<text x="25" y="{sy(yv) + 4}">{yv:.3g}</text>']
    for i, (label, points) in enumerate(series):
        c = palette[i % len(palette)]
        pts = " ".join(f"{sx(float(x)):.2f},{sy(float(y)):.2f}" for x, y in points if math.isfinite(float(x)) and math.isfinite(float(y)))
        svg.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2"/>')
        valid_points = [(x, y) for x, y in points if math.isfinite(float(x)) and math.isfinite(float(y))]
        if len(valid_points) == 1:
            x, y = valid_points[0]
            svg.append(f'<circle cx="{sx(float(x)):.2f}" cy="{sy(float(y)):.2f}" r="5" fill="{c}"><title>{html.escape(label)}: {float(y):.6g}</title></circle>')
        svg.append(f'<text x="800" y="{90 + i * 20}" style="fill:{c}">{html.escape(label)}</text>')
    path.write_text("\n".join(svg + ["</svg>"]))
    return True


def geometry_visuals(output, cached, causal):
    visuals = []
    if cached:
        sites = cached.get("sites", [])
        series = []
        for site in sites:
            spectrum = site["eigenspectrum"]["standardized_activation_covariance_eigenvalues"]
            series.append((f"{site['family']} {site['layer']}", [(i + 1, math.log10(max(v, 1e-12))) for i, v in enumerate(spectrum)]))
        if line_plot(output / "shortlist_eigenspectra.svg", "Shortlisted activation eigenspectra", series, "Eigenvalue rank", "log10 variance", "Same per-site standardized coordinates; dimensionality is descriptive, not causal intervention rank."):
            visuals.append({"path": "shortlist_eigenspectra.svg", "evidence": "geometry_map.json sites.*.eigenspectrum"})
        for site in sites:
            series = [(f"seed {t['seed']} episode {t['episode']}", list(zip(t["cumulative_progress"], [v[0] for v in t["coordinates"]]))) for t in site["pca_trajectories"]["trajectories"]]
            name = f"latent_progress_{site['family']}_{site['layer']}.svg"
            if line_plot(output / name, f"{site['family']} {site['layer']}: latent trajectory through progress", series, "Cumulative goal-distance reduction (m)", "PC1 coordinate", "Predetermined discovery episodes 0,1,2; points connected in time order, progress may reverse."):
                visuals.append({"path": name, "evidence": f"geometry_map.json sites.{site['family']}.block{site['layer']}.pca_trajectories"})
        # Directly plot target-pair angles only from the corrected common metric.
        series = []
        for site in sites:
            values = [(i + 1, sum(c["principal_angles_deg"]) / len(c["principal_angles_deg"])) for i, c in enumerate(site["subspace_comparisons"]) if c.get("principal_angles_deg")]
            series.append((f"{site['family']} {site['layer']}", values))
        if line_plot(output / "common_metric_subspace_angles.svg", "Corrected subspace comparisons", series, "1: progress/wall · 2: progress/direction · 3: wall/direction", "Mean angle (degrees)", "Shared standardization within each site; angles do not prove a native coordinate system."):
            visuals.append({"path": "common_metric_subspace_angles.svg", "evidence": "geometry_map.json sites.*.subspace_comparisons"})
    if causal:
        arms = list(dict.fromkeys(t["arm"] for t in causal.get("timelines", [])))
        for field, name, title, ylabel in (
            ("latent_l2_change_mean_per_candidate", "causal_latent_timeline.svg", "Intervention effect over imagined rollout time", "Mean candidate latent L2 change"),
            ("decoded_displacement_change_mean_norm", "causal_decoded_motion_timeline.svg", "Probe-decoded physical effect over imagined time", "Mean decoded displacement change (m)"),
        ):
            series = [(arm, [(t["imagined_step"] + 1, t[field]) for t in causal["timelines"] if t["arm"] == arm and t.get(field) is not None]) for arm in arms]
            if line_plot(output / name, title, series, "Imagined model step (five raw actions each)", ylabel, "Same 300 initial candidates; one development start, not 300 independent episodes; no confidence interval."):
                visuals.append({"path": name, "evidence": f"geometry_map.json causal_measurements.timelines[].{field}"})
        series = []
        for row in causal.get("rows", []):
            points = row["realized"]["displacement_xyz"]
            series.append((row["arm"], [(i, xyz[2]) for i, xyz in enumerate(points)]))
        if line_plot(output / "causal_realized_z.svg", "Actually realized vertical motion", series, "Executed raw simulator step", "Z displacement from identical start (m)", "Actual short simulator forks, not probe output; these are not full-episode task-success results."):
            visuals.append({"path": "causal_realized_z.svg", "evidence": "geometry_map.json causal_measurements.rows[].realized.displacement_xyz"})
    return visuals


def attach_validation(rows, sources, paths):
    """Attach measured follow-ups without converting proxies into acceptance."""
    extras = {name: load_source(path, sources) for name, path in paths.items() if path}
    target = next(r for r in rows if r["module"] == "predictor" and r["block"] == 3
                  and r["task_variable"] == "realized_xz_direction")
    if extras.get("replication"):
        report = extras["replication"]
        measured(target, "causal_patch_effect", {
            "reference": "followups.replication", "independent_starts": report["N_independent_development_starts"],
            "held_confirmation": report["held_confirmation_run"],
            "effects": report["combined_directions_semantic_minus_sham"],
        }, "measured_five_start_development_not_confirmed_mechanism")
        source_ref(target, paths["replication"], "combined_directions_semantic_minus_sham")
    if extras.get("support"):
        measured(target, "manifold_distance", {
            "reference": "followups.support.rows", "measurement": "nearest_reference_and_shrinkage_Mahalanobis",
            "true_manifold_membership": "not_established", "independent_starts": 1,
            "interpretation": "paired edited-state distributional support proxies only",
        }, "measured_edited_state_support_proxy_not_true_manifold")
        source_ref(target, paths["support"], "rows[].distances")
    if extras.get("specificity"):
        measured(target, "specificity", {
            "reference": "followups.specificity.rows", "independent_starts": 1,
            "location_and_time_controls": "executed", "behavioral_specificity": "not_established",
            "scope": "fixed_candidates_only_not_final_CEM_or_full_episodes",
        }, "measured_controls_do_not_establish_specificity")
        source_ref(target, paths["specificity"], "rows")
    if extras.get("interface_response"):
        target["actual_forward_response"] = {
            "reference": "followups.interface_response.rows", "independent_starts": 1,
            "directions_tested": 1, "doses": extras["interface_response"]["doses"],
            "interpretation": "finite-dose directional response, not a global manifold test"}
        source_ref(target, paths["interface_response"], "rows")
    if extras.get("nonlinear_interfaces"):
        for site in extras["nonlinear_interfaces"]["rows"]:
            if site["site"] not in ("predictor_block3", "predictor_block5"):
                continue  # Encoder/output interfaces remain separately named, not fake blocks.
            block = int(site["site"][-1])
            scores = site.get("held_validation", {})
            for variable in ("step_progress", "wall_clearance", "realized_xz_direction"):
                row = next(r for r in rows if r["module"] == "predictor" and r["block"] == block
                           and r["task_variable"] == canonical(variable))
                row["held_validation_readouts"] = {name: values[variable] for name, values in scores.items()}
                row["held_validation_episodes"] = 75
                row["evidence_scope"] = "discovery_plus_held_validation_not_confirmation"
                row["earlier_nonlinear_measurement"] = row["nonlinear_gain"]
                measured(row, "nonlinear_gain", {
                    "reference": f"followups.nonlinear_interfaces.rows.{site['site']}",
                    "held_validation_rbf_minus_full_ridge": scores["rbf_ridge"][variable] - scores["full_ridge"][variable],
                    "held_validation_rbf_minus_same_PCA_ridge": scores["rbf_ridge"][variable] - scores["pca_ridge"][variable],
                    "interpretation": "fixed readout comparison; no manifold/causal conclusion",
                }, "measured_held_validation_readout_gain")
                source_ref(row, paths["nonlinear_interfaces"], f"rows.{site['site']}.held_validation")
    for source_name, task in (("coordinate_time", "reach-wall"), ("pusht_coordinate_time", "push-t")):
        report = extras.get(source_name)
        if not report:
            continue
        for variable, values in report["validation"].items():
            row = new_row("predictor", 3, "coordinate_probe/"+variable)
            row.update(task=task, model="released_jepa_wm_metaworld" if task == "reach-wall" else "released_jepa_wm_pusht",
                       row_id=f"{task}/predictor/block-3/spatial-mean/coordinate_probe/{variable}",
                       evidence_scope="whole_episode_validation_after_discovery_freeze",
                       held_validation_episodes=report["validation_episodes"],
                       coordinate_probe_measurements=values,
                       time_warning="Episode time decodability is not an internal-clock claim; imagined horizon is separate.")
            measured(row, "linear_score", {"metric": "r2", "score": values["linear"]["r2"],
                                          "time_baseline": None, "delta_vs_time": None}, "measured_whole_episode_validation")
            measured(row, "nonlinear_gain", {"rbf_minus_linear_r2": values["rbf"]["r2"]-values["linear"]["r2"],
                                             "paired_mse_difference": values["nonlinear_minus_linear_mse"],
                                             "equal_full_inputs_and_tuning_count": True,
                                             "manifold_evidence": "not_established"}, "measured_readout_gain_not_causal_geometry")
            measured(row, "best_coordinate_frame", {"discovery_selected": report["candidate"]["target"],
                                                     "selection": report["candidate"]["selection"],
                                                     "native_coordinate_claim": False}, "measured_discovery_frame_comparison")
            source_ref(row, paths[source_name], f"validation.{variable}")
            rows.append(row)
    return extras


def collect_pilot(root, sources):
    """Only locally present, hash-verified completed episode files; partial study."""
    if not root:
        return None
    selected = {}
    for directory in sorted(root.glob("worker-*/results-v1")):
        candidates = [directory/"DONE.json"] if (directory/"DONE.json").is_file() else sorted(directory.glob("progress*.json"))
        for path in candidates:
            receipt = load_source(path, sources)
            for item in receipt.get("outputs", []):
                payload = directory/item["path"]
                if not payload.is_file() or sha256(payload) != item["sha256"]:
                    continue
                k = item["episode"], item["arm"]
                if k in selected and selected[k]["sha256"] != item["sha256"]:
                    raise ValueError("Conflicting pilot episode")
                selected[k] = {**item, "source_receipt": str(path), "verified_payload": str(payload)}
    episodes = sorted({k[0] for k in selected})
    arms = ("unsteered", "static_candidate", "matched_sham")
    paired = [e for e in episodes if all((e, a) in selected for a in arms)]
    return {"complete": len(paired) == 5, "planned_starts": [0, 1, 4, 7, 10],
            "fully_paired_starts": paired, "completed_arm_runs": len(selected),
            "scope": "exploratory full99-step development episodes; NOT residual intervention or held efficacy",
            "intervention_site": "returned predicted visual latents, not block3 residual",
            "rows": [selected[k] for k in sorted(selected)],
            "operator_validated": False,
            "warning": "Final goal distance, minimum goal distance, final success and ever-success are distinct; do not select the favorable endpoint."}


def attach_action_time(rows, sources, extras, paths):
    report = extras.get("action_interactions")
    if not report:
        return
    for block in (0, 3, 5):
        selected = [r for r in report["group_rows"] if r["stage"] == f"P{block}"
                    and r["representation"] == "full_tensor"]
        row = new_row("predictor", block, "finite_dose_action_response")
        row.update(spatial_region="all_256_newest_frame_tokens",
                   imagined_timestep=list(range(1, 7)),
                   imagined_timestep_status="measured_separately_at_six_horizons",
                   evidence_scope="three_development_scenes_fixed_action_perturbations")
        measured(row, "geometry", {"response_by_horizon": selected,
                                   "measurement": "native_input_response_curvature_and_XZ_interaction",
                                   "independent_scenes": len(report["episodes"]),
                                   "scope": report["scope"],
                                   "intrinsic_manifold_dimension": "not_established"},
                 "measured_finite_dose_response_not_readout_gain")
        source_ref(row, paths["action_interactions"], "group_rows")
        if extras.get("token_time"):
            measured(row, "cross_trajectory_stability", {
                "spatial_rank_stability": [r for r in extras["token_time"]["spatial_rank_stability"]
                                           if r["stage"] == f"P{block}"],
                "individual_token_causality": "not_tested",
                "interpretation": "token response ranks across imagined horizons, not an accepted intervention site"},
                "descriptive_three_scene_spatial_response_stability")
            source_ref(row, paths["token_time"], "spatial_rank_stability")
        rows.append(row)


def attach_residual_physics(rows, sources, paths):
    reports = [load_source(path, sources) for path in paths]
    if not reports:
        return None
    measurements = [r for report in reports for r in report["rows"]]
    indexed = {(r["episode"], r["arm"]): r for r in measurements}
    if len(indexed) != len(measurements):
        raise ValueError("Duplicate residual physical episode/arm")
    episodes = sorted({r["episode"] for r in measurements})
    contrasts = {}
    for arm in ("positive", "negative"):
        values = []
        for episode in episodes:
            semantic, sham = indexed[episode, arm], indexed[episode, arm+"_sham"]
            values.append({"episode": episode,
                           "progress_vs_baseline_m": semantic["hand_goal_progress_change_vs_baseline_m"],
                           "progress_vs_sham_m": semantic["hand_goal_progress_m"]-sham["hand_goal_progress_m"],
                           "initial_candidates_predicted_z_shift_h3_m": semantic["common_first300_predicted_z_shift_h3_m"],
                           "actual_z_shift_m": semantic["physical_end_xyz_shift"][2]})
        contrasts[arm] = {"episode_values": values,
                          "mean_progress_vs_baseline_m": sum(r["progress_vs_baseline_m"] for r in values)/len(values),
                          "mean_progress_vs_sham_m": sum(r["progress_vs_sham_m"] for r in values)/len(values)}
    result = {"complete": True, "episodes": episodes, "independent_starts": len(episodes),
              "rows": measurements, "paired_contrasts": contrasts,
              "scope": "15 raw simulator steps after native CEM; not full episodes or held confirmation",
              "success_warning": "Hand-goal distance is not native finger-TCP success; raw report provenance retained",
              "operator_validated": False}
    row = new_row("predictor", 3, "world_z_donor_pulse_rank8")
    row.update(evidence_scope="three_development_starts_short_native_planner_physics",
               operator_reason="Neither signed rank8 pulse1 edit improved mean progress beyond sham in these development forks.")
    measured(row, "causal_patch_effect", result, "measured_short_physical_forks_not_full_episode_efficacy")
    for path in paths:
        source_ref(row, path, "rows")
    rows.append(row)
    return result


def coordinate_heatmap(path, report, title):
    names = list(report["validation"])
    svg = svg_open(1030, 115+len(names)*38, title,
                   f"{report['validation_episodes']} whole validation trajectories; same400 inputs,2 settings/model; R2 is readout accuracy, not steering.")
    for j, model in enumerate(("linear", "rbf")):
        svg.append(f'<text x="{420+j*200}" y="90">{model}</text>')
    for i, name in enumerate(names):
        y = 110+i*38
        svg.append(f'<text x="20" y="{y+21}">{html.escape(name)}</text>')
        for j, model in enumerate(("linear", "rbf")):
            value = report["validation"][name][model]["r2"]
            fill = color(value, 1.) if value is not None else "#ccc"
            label = "constant target" if value is None else f"{value:.4f}"
            svg.append(f'<rect x="{410+j*200}" y="{y}" width="170" height="30" fill="{fill}"/><text x="{420+j*200}" y="{y+21}">{label}</text>')
    path.write_text("\n".join(svg+["</svg>"]))


def build(root, output, geometry_path=None, causal_path=None, followup_paths=None):
    output.mkdir(parents=True, exist_ok=False)
    rows, sources, cached, causal = join_evidence(root, geometry_path, causal_path)
    followup_paths = dict(followup_paths or {})
    pilot_root = followup_paths.pop("pilot_root", None)
    physical_paths = followup_paths.pop("residual_physical", []) or []
    extras = attach_validation(rows, sources, followup_paths)
    attach_action_time(rows, sources, extras, followup_paths)
    physical = attach_residual_physics(rows, sources, physical_paths)
    if physical:
        extras["residual_physical"] = physical
    pilot = collect_pilot(pilot_root, sources)
    if pilot:
        extras["coordinate_pilot"] = pilot
    visuals = []
    heatmap([r for r in rows if r["task"] == "reach-wall"], output / "layer_variable_heatmap.svg")
    visuals.append({"path": "layer_variable_heatmap.svg", "evidence": "geometry_map.json rows[].linear_score"})
    frame = root / "shortlist-components-v1/representative_frame.png"
    spatial_plot(rows, output / "spatial_token_readability.svg", frame if frame.is_file() else None)
    visuals.append({"path": "spatial_token_readability.svg", "evidence": "geometry_map.json rows[].spatial_decodability", "image_overlay": frame.is_file()})
    if frame.is_file():
        sources[str(frame)] = {"path": str(frame.resolve()), "sha256": sha256(frame)}
    series = []
    for family in ("encoder", "predictor"):
        # Keep one original measurement protocol across every block in this curve.
        points = [(r["block"], r["direction_magnitude_curve"]["r2"]) for r in rows if r["module"] == family and r["task_variable"] == "realized_xz_direction" and r.get("direction_magnitude_curve")]
        series.append((family, sorted(points)))
    if line_plot(output / "motion_direction_emergence.svg", "Realized motion-direction readability", series, "Block index (separate encoder/predictor modules)", "Held-seed R²", "Discovery CV only; decodability does not establish use by the planner."):
        visuals.append({"path": "motion_direction_emergence.svg", "evidence": "geometry_map.json realized_xz_direction rows[].direction_magnitude_curve; original consistently masked report"})
    visuals.extend(geometry_visuals(output, cached, causal))
    if extras.get("replication"):
        effects = extras["replication"]["combined_directions_semantic_minus_sham"]
        series = [(name, list(enumerate(effects[name]["episode_values"], 1))) for name in
                  ("signed_target_z_effect_m", "goal_improvement_vs_unsteered_m")]
        line_plot(output / "five_start_causal_effects.svg", "Five-start semantic-minus-sham effects",
                  series, "Development start index", "Metres; positive is prescribed improvement",
                  "Short forks, not full episodes; descriptive bootstrap intervals are in JSON.")
        visuals.append({"path": "five_start_causal_effects.svg", "evidence": "followups.replication.combined_directions_semantic_minus_sham"})
    if extras.get("specificity"):
        series = [(r["arm"], list(enumerate(r["mean_absolute_rank_change_by_future_step"], 1)))
                  for r in extras["specificity"]["rows"] if r["arm"] not in ("unsteered", "identity")]
        line_plot(output / "location_time_rank_controls.svg", "Off-site and delayed edit controls",
                  series, "Imagined model step", "Mean absolute candidate-rank change",
                  "Same initial candidate bank; not final CEM selection or behavioral specificity.")
        visuals.append({"path": "location_time_rank_controls.svg", "evidence": "followups.specificity.rows"})
    if extras.get("nonlinear_interfaces"):
        selected = extras["nonlinear_interfaces"]["rows"]
        series = [(model, [(i, r["held_validation"][model]["realized_xz_direction"])
                          for i, r in enumerate(selected)]) for model in ("full_ridge", "pca_ridge", "rbf_ridge")]
        line_plot(output / "held_validation_interface_readouts.svg", "Direction readouts on 75 held validation trajectories",
                  series, "0 encoder output; 1 predictor block3; 2 block5; 3 predicted visual output", "Held-validation R²",
                  "Frozen readouts; pooled interfaces and unequal model capacities; not a causal depth comparison.")
        visuals.append({"path": "held_validation_interface_readouts.svg", "evidence": "followups.nonlinear_interfaces.rows[].held_validation"})
    if extras.get("interface_response"):
        stage_series = {}
        for row in extras["interface_response"]["rows"]:
            values = row["even_to_odd_norm_ratio"]
            stage_series.setdefault(row["stage"], []).append((row["beta"], sum(values) / len(values)))
        line_plot(output / "actual_forward_interface_response.svg", "Actual forward response along one fixed edit direction",
                  list(stage_series.items()), "Absolute edit dose", "Symmetric second difference / plus-minus response",
                  "First 8 fixed candidates from one development start; small ratio does not exclude odd or other-direction nonlinearity.")
        visuals.append({"path": "actual_forward_interface_response.svg", "evidence": "followups.interface_response.rows"})
    for name, title in (("coordinate_time", "Reach-Wall coordinate and time probes"),
                        ("pusht_coordinate_time", "Push-T coordinate and time probes")):
        if extras.get(name):
            filename = name+"_heatmap.svg"
            coordinate_heatmap(output/filename, extras[name], title)
            visuals.append({"path": filename, "evidence": f"followups.{name}.validation"})
    if extras.get("transform_null"):
        metrics = extras["transform_null"]["rmse_m"]
        line_plot(output/"coordinate_transform_control.svg", "Does known coordinate conversion explain the nonlinear advantage?",
                  [(name, [(i, value*1000)]) for i, (name, value) in enumerate(metrics.items())],
                  "Frozen readout / control", "Physical error (mm)",
                  "Post-hoc75-episode control; external current-hand/goal anchors; does not identify native coordinates.")
        visuals.append({"path": "coordinate_transform_control.svg", "evidence": "followups.transform_null.rmse_m"})
    if pilot and pilot["fully_paired_starts"]:
        paired = set(pilot["fully_paired_starts"])
        series = [(arm, [(r["episode"], r["final_goal_distance_m"]*1000) for r in pilot["rows"]
                          if r["arm"] == arm and r["episode"] in paired]) for arm in ("unsteered", "static_candidate", "matched_sham")]
        line_plot(output/"full_episode_pilot_endpoints.svg", "Output-calibration pilot: final goal error",
                  series, "Development episode ID", "Final goal error (mm)",
                  f"{len(paired)}/5 paired starts available; native success outcomes are separately shown below; not a residual intervention.")
        visuals.append({"path": "full_episode_pilot_endpoints.svg", "evidence": "followups.coordinate_pilot.rows"})
    if extras.get("action_interactions"):
        selected = [r for r in extras["action_interactions"]["group_rows"]
                    if r["representation"] == "full_tensor" and r["scale"] == 1.
                    and r["kind"] == "axis_curvature"]
        series = [(stage, [(r["imagined_step"], 100*r["mean"]) for r in selected if r["stage"] == stage])
                  for stage in sorted({r["stage"] for r in selected})]
        line_plot(output/"native_action_curvature_over_time.svg", "Native action-response curvature across imagined time",
                  series, "Imagined model step", "Curvature / odd response (%)",
                  "Three scenes, raw action amplitude0.05; newest256 tokens; not manifold dimension or steering efficacy.")
        visuals.append({"path": "native_action_curvature_over_time.svg", "evidence": "followups.action_interactions.group_rows"})
    if extras.get("token_time"):
        source_dir = followup_paths["token_time"].parent
        for name in ("P3_H1_H3_H6.svg", "returned_visual_H1_H3_H6.svg"):
            source = source_dir/name
            sources[str(source)] = {"path": str(source.resolve()), "sha256": sha256(source)}
            shutil.copyfile(source, output/name)
            visuals.append({"path": name, "evidence": "followups.token_time.rows; token response, not individual-token causality"})
    if physical:
        series = [(arm, [(r["episode"], r["progress_vs_sham_m"]*1000) for r in values["episode_values"]])
                  for arm, values in physical["paired_contrasts"].items()]
        line_plot(output/"residual_physical_progress.svg", "Residual pulse: actual task progress beyond matched sham",
                  series, "Development episode ID", "Semantic minus sham progress (mm)",
                  "Rank8, pulse1,15 raw steps; positive means better; neither sign has positive mean here.")
        visuals.append({"path": "residual_physical_progress.svg", "evidence": "followups.residual_physical.paired_contrasts"})
    result = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_type": "causal_geometry_map", "complete": False,
        "scope": "task-conditioned Reach-Wall geometry with separate Push-T coordinate follow-up; causal confirmation still required",
        "label_corrections": ALIASES,
        "excluded_evidence": ["Historical principal angles fitted with inconsistent scaling", "Whole-block latent transfer as evidence of a selective planner control interface"],
        "rows": rows, "sources": list(sources.values()), "visuals": visuals,
        "sites": {f"{s['family']}.block{s['layer']}": s for s in cached.get("sites", [])} if cached else {},
        "causal_measurements": causal,
        "followups": extras,
        "cached_geometry_present": cached is not None, "causal_measurements_present": causal is not None,
        "field_coverage": {f: {"measured_rows": sum(r["field_status"][f].startswith(("measured", "descriptive", "development")) for r in rows), "total_rows": len(rows)} for f in CORE_FIELDS},
        "completion_blockers": ["Untouched confirmation and full required causal controls are not yet complete", "No validated operator card exists"],
    }
    dump(output / "geometry_map.json", result)
    csv_fields = ["row_id", "task", "module", "block", "hook", "spatial_region", "imagined_timestep", "task_variable", *CORE_FIELDS, "field_status", "sources"]
    with (output / "geometry_map.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(finite(row[k]), allow_nan=False) if isinstance(row[k], (list, dict)) else row[k] for k in csv_fields})
    page = '<!doctype html><meta charset="utf-8"><title>Reach-Wall geometry map</title><style>body{font:16px system-ui;margin:2rem;max-width:1400px}img{max-width:100%}</style><h1>Reach-Wall geometry map: discovery evidence</h1><p>Not a completed causal map or validated steering operator. Raw data: <a href="geometry_map.json">JSON</a> · <a href="geometry_map.csv">CSV</a>.</p>'
    if extras:
        page += '<p>Follow-up evidence: five-start short-fork replication, edited-state support proxies, location/time controls, and frozen nonlinear interface readouts on 75 validation trajectories. No validated steering operator; confirmation trajectories remain unopened.</p>'
    page += f'<p>Assembled: {html.escape(result["created_utc"])}; new coordinate findings are block3-only, not a new all-layer sweep.</p>'
    if extras.get("coordinate_time"):
        page += '<p>Latest Reach finding: readout nonlinearity depends on the coordinate definition; elapsed episode time is readable but is not proof of a native clock. Rank1/2/8 residual candidates are hypotheses, not accepted operators.</p>'
    if extras.get("transform_null"):
        values = extras["transform_null"]["rmse_m"]
        page += '<p>Coordinate-transform control: '+html.escape("; ".join(f"{k}: {1000*v:.3f}mm" for k, v in values.items()))+'. A linear world readout plus known geometry is a competing explanation; hidden nonlinear structure is not ruled out.</p>'
    if extras.get("pusht_coordinate_time"):
        page += '<p>Push-T warning: a nonlinear advantage over a weak linear probe is not sufficient; compare the physical-persistence baseline recorded in the JSON. The large fit-versus-validation gap remains unresolved, and no Push-T operator is validated.</p>'
    if extras.get("action_interactions"):
        page += '<p>Actual action perturbations reveal increasing nonlinear response over imagined time, independently of the coordinate probes. The new spatial plots localize response, not causal importance of individual tokens; only three development scenes were tested.</p>'
    if physical:
        page += '<p>Residual rank8/pulse1 result: neither signed candidate improved average15-step physical progress beyond its matched sham. Predicted and executed shifts can oppose one another; this is compatible with planner compensation, not proof of that mediation mechanism.</p>'
    if pilot:
        page += f'<h2>Full-episode pilot snapshot: {len(pilot["fully_paired_starts"])}/5 paired development starts</h2><p>{html.escape(pilot["scope"])}; {html.escape(pilot["warning"])}</p><table><tr><th>Episode</th><th>Arm</th><th>Final distance mm</th><th>Ever success</th><th>Final success</th><th>Cap saturation</th></tr>'
        for row in pilot["rows"]:
            page += f'<tr><td>{row["episode"]}</td><td>{html.escape(row["arm"])}</td><td>{1000*row["final_goal_distance_m"]:.3f}</td><td>{row["ever_success"]}</td><td>{row["final_success"]}</td><td>{row["operator_statistics"]["cap_saturation_fraction"]:.1%}</td></tr>'
        page += '</table>'
    for visual in visuals:
        page += f'<h2>{html.escape(visual["path"])}</h2><img src="{visual["path"]}"><p>Evidence: {html.escape(visual["evidence"])}</p>'
    (output / "index.html").write_text(page)
    dump(output / "ASSEMBLY_DONE.json", {"assembly_complete": True, "experiment_complete": False, "rows": len(rows), "outputs": [{"path": p.name, "sha256": sha256(p)} for p in sorted(output.iterdir()) if p.is_file()]})
    print(json.dumps({"output": str(output), "rows": len(rows), "visuals": len(visuals), "experiment_complete": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--geometry", type=Path)
    parser.add_argument("--causal", type=Path)
    parser.add_argument("--support", type=Path)
    parser.add_argument("--replication", type=Path)
    parser.add_argument("--specificity", type=Path)
    parser.add_argument("--nonlinear-interfaces", type=Path)
    parser.add_argument("--interface-response", type=Path)
    parser.add_argument("--coordinate-time", type=Path)
    parser.add_argument("--pusht-coordinate-time", type=Path)
    parser.add_argument("--transform-null", type=Path)
    parser.add_argument("--pilot-root", type=Path)
    parser.add_argument("--action-interactions", type=Path)
    parser.add_argument("--token-time", type=Path)
    parser.add_argument("--residual-physical", type=Path, nargs="+")
    args = parser.parse_args()
    build(args.root, args.output_dir, args.geometry, args.causal, {
        "support": args.support, "replication": args.replication, "specificity": args.specificity,
        "nonlinear_interfaces": args.nonlinear_interfaces, "interface_response": args.interface_response,
        "coordinate_time": args.coordinate_time, "pusht_coordinate_time": args.pusht_coordinate_time,
        "transform_null": args.transform_null, "pilot_root": args.pilot_root,
        "action_interactions": args.action_interactions, "token_time": args.token_time,
        "residual_physical": args.residual_physical})


if __name__ == "__main__":
    main()
