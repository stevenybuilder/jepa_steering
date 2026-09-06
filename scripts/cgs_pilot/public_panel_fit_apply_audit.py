#!/usr/bin/env python3
"""Audit chosen-unroll fitting states against the actual CEM hook population.

The fitter uses one selected-action extra unroll per physical replan, while the
online hook edits every CEM candidate forward. This script requires an
``all_plans`` capture and compares every retained pre-outcome replan against its
separately tagged candidate population. It compares density,
regime, energy-field, conceptor-field, edit magnitude, and predicted action
sensitivity.  Failure requires candidate-population refitting or a narrowed
hook schedule; it may not be waived by regime agreement alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from public_panel_factor_gate import gmm_log_prob
from public_panel_sonar_fit import static_responsibilities
from public_panel_sonar_math import capped_metric_step, outcome_energy_contrast_gradient


def transform_hidden(rows: np.ndarray, operator: dict[str, np.ndarray]) -> np.ndarray:
    return (np.asarray(rows, dtype=np.float64) - operator["hidden_mean"]) @ operator["encoder"]


def metric_norm(rows: np.ndarray, metrics: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(np.einsum("ni,nij,nj->n", rows, metrics, rows), 0.0))


def cap_final(rows: np.ndarray, metrics: np.ndarray, radius: float) -> np.ndarray:
    """Apply the same final radial trust-region cap as the online runtime."""
    norm = metric_norm(rows, metrics)
    scale = np.minimum(radius / np.maximum(norm, 1e-30), 1.0)
    return rows * scale[:, None]


def cosine_to(reference: np.ndarray, rows: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=np.float64).reshape(1, -1)
    rows = np.atleast_2d(np.asarray(rows, dtype=np.float64))
    denominator = np.linalg.norm(reference, axis=1)[0] * np.linalg.norm(rows, axis=1)
    return (rows @ reference[0]) / np.maximum(denominator, 1e-30)


def fields(rows: np.ndarray, operator: dict[str, np.ndarray], step_size: float, radius: float) -> dict:
    emission_dim = operator["regime_means"].shape[1]
    regime_model = {
        "weights": operator["regime_weights"],
        "means": operator["regime_means"],
        "var": operator["regime_var"],
    }
    responsibility = static_responsibilities(rows[:, :emission_dim], regime_model)
    metrics = np.einsum("nk,kij->nij", responsibility, operator["trust_metrics"])
    _energy, gradient, _parts = outcome_energy_contrast_gradient(
        rows,
        {
            "weights": operator["success_weights"], "means": operator["success_means"],
            "covariances": operator["success_covariances"],
        },
        {
            "weights": operator["failure_weights"], "means": operator["failure_means"],
            "covariances": operator["failure_covariances"],
        },
    )
    energy_delta = capped_metric_step(gradient, metrics, step_size=step_size, radius=radius)
    local = np.einsum("nk,kij->nij", responsibility, operator["local_contrastive_conceptors"])
    conceptor_delta = step_size * (np.einsum("ni,nji->nj", rows, local) - rows)
    conceptor_delta = cap_final(conceptor_delta, metrics, radius)
    action_gram = operator.get("action_gram", np.zeros((rows.shape[1], rows.shape[1])))
    return {
        "responsibility": responsibility,
        "metrics": metrics,
        "energy_delta": energy_delta,
        "conceptor_delta": conceptor_delta,
        "energy_norm": metric_norm(energy_delta, metrics),
        "conceptor_norm": metric_norm(conceptor_delta, metrics),
        "energy_action_effect": np.sqrt(np.maximum(np.einsum(
            "ni,ij,nj->n", energy_delta, action_gram, energy_delta
        ), 0.0)),
    }


def ratio_log(candidate: np.ndarray, chosen: float) -> float:
    return float(np.median(np.abs(np.log((np.asarray(candidate) + 1e-12) / (float(chosen) + 1e-12)))))


def audit_episode(chosen: np.ndarray, candidates: np.ndarray, operator: dict, step_size: float, radius: float) -> dict:
    all_rows = np.r_[chosen.reshape(1, -1), candidates]
    value = fields(all_rows, operator, step_size, radius)
    chosen_field = {key: array[:1] for key, array in value.items()}
    candidate_field = {key: array[1:] for key, array in value.items()}
    chosen_regime = int(chosen_field["responsibility"][0].argmax())
    regime_agreement = float(np.mean(candidate_field["responsibility"].argmax(1) == chosen_regime))
    regime_model = {
        "weights": operator["regime_weights"], "means": operator["regime_means"], "var": operator["regime_var"]
    }
    ll = gmm_log_prob(all_rows[:, : operator["regime_means"].shape[1]], regime_model)
    dimension = operator["regime_means"].shape[1]
    return {
        "n_candidates": int(len(candidates)),
        "regime_agreement": regime_agreement,
        "density_gap_nats_per_dimension": float(abs(ll[1:].mean() - ll[0]) / dimension),
        "energy_edit_cosine_median": float(np.median(cosine_to(
            chosen_field["energy_delta"][0], candidate_field["energy_delta"]
        ))),
        "conceptor_edit_cosine_median": float(np.median(cosine_to(
            chosen_field["conceptor_delta"][0], candidate_field["conceptor_delta"]
        ))),
        "energy_edit_abs_log_norm_ratio_median": ratio_log(
            candidate_field["energy_norm"], chosen_field["energy_norm"][0]
        ),
        "conceptor_edit_abs_log_norm_ratio_median": ratio_log(
            candidate_field["conceptor_norm"], chosen_field["conceptor_norm"][0]
        ),
        "energy_action_effect_abs_log_ratio_median": ratio_log(
            candidate_field["energy_action_effect"], chosen_field["energy_action_effect"][0]
        ),
    }


AUDIT_METRICS = (
    "regime_agreement",
    "density_gap_nats_per_dimension",
    "energy_edit_cosine_median",
    "conceptor_edit_cosine_median",
    "energy_edit_abs_log_norm_ratio_median",
    "conceptor_edit_abs_log_norm_ratio_median",
    "energy_action_effect_abs_log_ratio_median",
)


def aggregate(per_episode: list[dict], thresholds: dict[str, float]) -> dict:
    """Aggregate replan checks with episodes as the independent unit."""
    if not per_episode:
        raise ValueError("no fit/apply comparisons")
    episode_ids = sorted({int(row.get("ep", index)) for index, row in enumerate(per_episode)})
    episode_metrics = []
    for episode in episode_ids:
        rows = [row for index, row in enumerate(per_episode) if int(row.get("ep", index)) == episode]
        episode_metrics.append({
            "ep": episode,
            **{key: float(np.median([row[key] for row in rows])) for key in AUDIT_METRICS},
        })
    metrics = {
        key: float(np.median([row[key] for row in episode_metrics]))
        for key in AUDIT_METRICS
    }
    checks = {
        "regime_agreement": metrics["regime_agreement"] >= thresholds["min_regime_agreement"],
        "density": metrics["density_gap_nats_per_dimension"] <= thresholds["max_density_gap"],
        "energy_direction": metrics["energy_edit_cosine_median"] >= thresholds["min_edit_cosine"],
        "conceptor_direction": metrics["conceptor_edit_cosine_median"] >= thresholds["min_edit_cosine"],
        "energy_norm": metrics["energy_edit_abs_log_norm_ratio_median"] <= thresholds["max_abs_log_ratio"],
        "conceptor_norm": metrics["conceptor_edit_abs_log_norm_ratio_median"] <= thresholds["max_abs_log_ratio"],
        "predicted_action_effect": (
            metrics["energy_action_effect_abs_log_ratio_median"] <= thresholds["max_abs_log_ratio"]
        ),
    }
    return {
        "median_metrics": metrics,
        "episode_metrics": episode_metrics,
        "checks": checks,
        "fit_apply_equivalent": bool(all(checks.values())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--operator", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument("--radius", type=float, default=0.25)
    parser.add_argument("--min-regime-agreement", type=float, default=0.75)
    parser.add_argument("--min-edit-cosine", type=float, default=0.80)
    parser.add_argument("--max-density-gap", type=float, default=0.25)
    parser.add_argument("--max-abs-log-ratio", type=float, default=float(np.log(2.0)))
    args = parser.parse_args()

    with np.load(args.operator) as raw:
        operator = {key: np.asarray(raw[key], dtype=np.float64) for key in raw.files if raw[key].dtype.kind not in "USO"}
        site = str(raw["site"].item())
    index = json.loads((args.capture / "activations" / "index.json").read_text())
    if index.get("planner_capture_scope") != "all_plan_calls_tagged_by_physical_replan":
        raise SystemExit(
            "fit/apply audit requires a fresh --planner-capture-scope all_plans capture; "
            "first-plan rows cannot license an all-replan hook"
        )
    window = index.get("pre_outcome_window")
    if not window or len(window) != 2 or int(window[1]) <= int(window[0]):
        raise SystemExit(f"invalid or empty frozen pre-outcome window: {window}")
    start, stop = map(int, window)
    per_episode = []
    for row in index["episodes"]:
        with np.load(args.capture / "activations" / row["file"]) as capture:
            chosen_key = f"{site}__mean"
            if chosen_key not in capture.files or len(capture[chosen_key]) < stop:
                raise SystemExit(f"{row['file']} lacks {chosen_key}[{start}:{stop}]")
            for replan in range(start, stop):
                candidate_key = f"{site}__planner_call{replan:03d}"
                if candidate_key not in capture.files:
                    raise SystemExit(
                        f"{row['file']} lacks {candidate_key}; every retained physical replan must be tagged"
                    )
                chosen = transform_hidden(capture[chosen_key][replan], operator)
                candidates = transform_hidden(capture[candidate_key], operator)
                result = audit_episode(chosen, candidates, operator, args.step_size, args.radius)
                result.update({
                    "ep": int(row["ep"]),
                    "success": int(row["success"]),
                    "replan": int(replan),
                })
                per_episode.append(result)
    thresholds = {
        "min_regime_agreement": args.min_regime_agreement,
        "min_edit_cosine": args.min_edit_cosine,
        "max_density_gap": args.max_density_gap,
        "max_abs_log_ratio": args.max_abs_log_ratio,
    }
    summary = aggregate(per_episode, thresholds)
    report = {
        "status": "pre_intervention_gate",
        "capture": str(args.capture.resolve()),
        "capture_index_sha256": hashlib.sha256(
            (args.capture / "activations" / "index.json").read_bytes()
        ).hexdigest(),
        "operator": str(args.operator.resolve()),
        "operator_sha256": hashlib.sha256(args.operator.read_bytes()).hexdigest(),
        "site": site,
        "pre_outcome_window": [start, stop],
        "fitted_population": "selected-action extra unroll at every retained physical replan",
        "runtime_population": "tagged CEM candidate forwards at those exact physical replans",
        "independent_unit": "episode; replan metrics collapse within episode before the across-episode median",
        "thresholds": thresholds,
        **summary,
        "per_episode": per_episode,
        "failure_action": "refit on tagged CEM candidates or narrow the hook schedule; do not launch steering",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "fit_apply_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"fit_apply_equivalent": report["fit_apply_equivalent"], **report["median_metrics"]}, indent=2))
    return 0 if report["fit_apply_equivalent"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
