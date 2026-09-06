#!/usr/bin/env python3
"""Fit/evaluate the outcome-free modules of the Panel-P Frankenstein recipe.

This gate runs only after the factor-coordinate artifact is frozen.  It writes
one JSON admission report and one NPZ artifact for later outcome-operator fitting.
Missing action, attention, coordinate, or paired-transport evidence fails only
that module; it never silently substitutes a different mathematical object.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from public_panel_factor_gate import load_site_sequences, log_gaussian_diag
from public_panel_coordinate_gate import load_aligned_targets, selected_replan_indices
from public_panel_frankenstein_math import (
    attention_retrieval_gate,
    fit_orthogonal_transport,
    pattern_separation_gate,
    probability_geometry_gate,
    sparse_superposition_gate,
)
from public_panel_sonar_fit import transform


def static_responsibility(sequence: np.ndarray, coordinate: dict[str, np.ndarray]) -> np.ndarray:
    joint = log_gaussian_diag(
        sequence[:, : coordinate["regime_means"].shape[1]],
        coordinate["regime_means"],
        coordinate["regime_var"],
    ) + np.log(coordinate["regime_weights"] + 1e-30)
    joint -= joint.max(1, keepdims=True)
    probability = np.exp(joint)
    return probability / probability.sum(1, keepdims=True)


def load_attention_sequences(capture: Path, index: dict, layer: str, window: tuple[int, int]) -> dict[str, list[np.ndarray]] | None:
    names = ("query_mean", "query_var", "key_mean", "key_var", "attention_entropy", "attention_key_frame_mass")
    values = {name: [] for name in names}
    for episode in index["episodes"]:
        with np.load(capture / "activations" / episode["file"]) as arrays:
            for name in names:
                key = f"{layer}.{name}__attention"
                if key not in arrays.files:
                    return None
                values[name].append(np.asarray(arrays[key][window[0] : window[1]], dtype=np.float64))
    return values


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def strip_artifacts(report: dict) -> dict:
    """Remove large arrays/models from JSON while preserving every verdict."""
    if isinstance(report, dict):
        return {
            key: strip_artifacts(value)
            for key, value in report.items()
            if key not in {"models", "model", "basis", "pairs"}
        }
    if isinstance(report, list):
        return [strip_artifacts(value) for value in report]
    return json_safe(report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--coordinates", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--pool", default="mean", choices=["mean", "mean_all"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--sparse-components", type=int, default=16)
    parser.add_argument("--sparse-alpha", type=float, default=0.10)
    parser.add_argument("--sparse-iterations", type=int, default=250)
    parser.add_argument("--transport-capture", type=Path, default=None)
    parser.add_argument("--transport-coordinates", type=Path, default=None)
    parser.add_argument("--coordinate-report", type=Path, default=None)
    parser.add_argument("--model-native-artifact", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()

    with np.load(args.coordinates) as raw:
        coordinate = {key: np.asarray(raw[key]) for key in raw.files}
    site = str(coordinate["site"].item())
    raw_hidden, raw_actions, index = load_site_sequences(args.capture, site, args.pool)
    capture_index_sha256 = hashlib.sha256(
        (args.capture / "activations" / "index.json").read_bytes()
    ).hexdigest()
    window_raw = index.get("pre_outcome_window") or [0, min(len(value) for value in raw_hidden)]
    window = (int(window_raw[0]), int(window_raw[1]))
    hidden = [value[window[0] : window[1]] for value in raw_hidden]
    actions = (
        [value[window[0] : window[1]] for value in raw_actions]
        if raw_actions is not None else None
    )
    factors = [transform(value, coordinate) for value in hidden]
    responsibilities = [static_responsibility(value, coordinate) for value in factors]

    # Load and bind the independently audited physical-coordinate artifact before
    # constructing lures.  Simulator targets are used only for the offline match;
    # they are never exposed to the runtime intervention.
    coordinate_report = None
    native_artifact = args.model_native_artifact
    if args.coordinate_report is not None:
        coordinate_report = json.loads(args.coordinate_report.read_text())
        sibling = args.coordinate_report.with_suffix(".npz")
        if native_artifact is None and sibling.is_file():
            native_artifact = sibling
    native_arrays = None
    if native_artifact is not None:
        with np.load(native_artifact) as raw:
            native_arrays = {key: np.asarray(raw[key]) for key in raw.files}
        if str(native_arrays["site"].item()) != site:
            raise SystemExit("model-native artifact and factor coordinates select different sites")
    if coordinate_report is not None:
        if coordinate_report.get("capture_index_sha256") != capture_index_sha256:
            raise SystemExit("model-native report was fitted from a different capture index")
        if str(coordinate_report.get("site")) != site:
            raise SystemExit("model-native report and factor coordinates select different sites")
        if str(Path(coordinate_report.get("coordinates", "")).resolve()) != str(args.coordinates.resolve()):
            raise SystemExit("model-native report was fitted from different factor coordinates")
        if native_arrays is not None and "report_sha256" in native_arrays:
            expected_report_sha256 = str(native_arrays["report_sha256"].item())
            actual_report_sha256 = hashlib.sha256(args.coordinate_report.read_bytes()).hexdigest()
            if expected_report_sha256 != actual_report_sha256:
                raise SystemExit("model-native NPZ is not bound to the supplied JSON report")
    model_native = {
        "decidable": coordinate_report is not None and native_arrays is not None,
        "eligible": bool(
            coordinate_report
            and native_arrays is not None
            and coordinate_report.get("model_native_relational_candidate", False)
            and bool(native_arrays["eligible"].item())
        ),
        "source": str(args.coordinate_report.resolve()) if args.coordinate_report else None,
        "artifact": str(native_artifact.resolve()) if native_artifact else None,
        "reason": (
            None if coordinate_report is not None and native_arrays is not None
            else "model-native JSON report and NPZ row-space artifact are both required"
        ),
    }

    probability = probability_geometry_gate(
        factors,
        responsibilities,
        n_folds=min(args.folds, len(factors)),
        seed=args.seed,
    )
    sparse = sparse_superposition_gate(
        factors,
        actions,
        responsibilities=responsibilities,
        n_components=max(args.sparse_components, factors[0].shape[1] + 1),
        alpha=args.sparse_alpha,
        n_folds=min(args.folds, len(factors)),
        seed=args.seed + 101,
        max_iter=args.sparse_iterations,
    )
    pattern_factors, pattern_actions = factors, actions
    pattern_nuisance = None
    nuisance_source = "episode_progress_only_fallback"
    if coordinate_report is not None and native_arrays is not None:
        absolute_replans = selected_replan_indices(window)
        task = str(coordinate_report["adapter"]["task"])
        native_targets = load_aligned_targets(
            args.capture, index, absolute_replans, task=task
        )
        pattern_factors = [
            transform(value[absolute_replans], coordinate) for value in raw_hidden
        ]
        pattern_actions = (
            [value[absolute_replans] for value in raw_actions]
            if raw_actions is not None else None
        )
        pattern_nuisance = [
            np.concatenate(
                [
                    target["absolute_hand_world"],
                    target["absolute_goal_world"],
                    target["distance_hand_to_goal"],
                ],
                axis=1,
            )
            for target in native_targets
        ]
        nuisance_source = "audited_simulator_hand_goal_geometry_offline_only"
    pattern = pattern_separation_gate(
        pattern_factors,
        pattern_actions,
        nuisance=pattern_nuisance,
        rank=min(2, factors[0].shape[1]),
        seed=args.seed + 201,
    )
    pattern["nuisance_source"] = nuisance_source
    pattern["privileged_simulator_state_used_online"] = False

    layer = site.split(".")[0]
    attention_rows = load_attention_sequences(args.capture, index, layer, window)
    if attention_rows is None:
        attention = {
            "decidable": False,
            "eligible": False,
            "reason": "capture predates bounded DINO Q/K and attention-summary hooks",
        }
    else:
        query = np.concatenate(attention_rows["query_mean"])[:, :, None, :]
        key = np.concatenate(attention_rows["key_mean"])[:, :, None, :]
        attention_episode_ids = np.concatenate([
            np.full(len(value), episode, dtype=np.int32) for episode, value in enumerate(factors)
        ])
        attention = attention_retrieval_gate(
            np.concatenate(factors), query, key, np.concatenate(responsibilities),
            episode_ids=attention_episode_ids, seed=args.seed + 301,
        )
        entropy = np.concatenate(attention_rows["attention_entropy"])
        key_mass = np.concatenate(attention_rows["attention_key_frame_mass"])
        attention["captured_attention_entropy_mean"] = float(entropy.mean())
        attention["captured_key_frame_mass_mean"] = key_mass.mean(axis=0).tolist()

    transport = {"decidable": False, "eligible": False, "reason": "no paired capture supplied"}
    transport_model = None
    if args.transport_capture is not None:
        if args.transport_coordinates is None:
            raise SystemExit("--transport-capture requires --transport-coordinates")
        with np.load(args.transport_coordinates) as raw:
            other_coordinate = {key: np.asarray(raw[key]) for key in raw.files}
        other_site = str(other_coordinate["site"].item())
        other_hidden, _other_actions, other_index = load_site_sequences(
            args.transport_capture, other_site, args.pool
        )
        other_window_raw = other_index.get("pre_outcome_window") or [0, min(len(value) for value in other_hidden)]
        stop = min(window[1], int(other_window_raw[1]))
        n = min(len(hidden), len(other_hidden))
        source = np.stack([factors[i][:stop].mean(0) for i in range(n)])
        other_factors = [
            transform(value[window[0] : stop], other_coordinate) for value in other_hidden[:n]
        ]
        target = np.stack([value.mean(0) for value in other_factors])
        if source.shape[1] != target.shape[1]:
            raise SystemExit(
                f"factor transport requires equal ranks, got {source.shape[1]} and {target.shape[1]}"
            )
        transport_model = fit_orthogonal_transport(source, target)
        loo = []
        for held in range(n):
            keep = np.arange(n) != held
            fitted = fit_orthogonal_transport(source[keep], target[keep])
            prediction = (
                (source[held] - fitted["mean_source"]) @ fitted["rotation"] + fitted["mean_target"]
            )
            denominator = max(float(np.linalg.norm(target[held] - target[keep].mean(0))), 1e-12)
            loo.append(float(np.linalg.norm(prediction - target[held]) / denominator))
        transport = {
            "decidable": True,
            "n_matched_episode_anchors": int(n),
            "fit_relative_residual": float(transport_model["relative_residual"]),
            "leave_one_episode_out_relative_error_median": float(np.median(loo)),
            "eligible": bool(n >= 8 and np.median(loo) < 1.0),
            "row_vector_convention": "target=(source-mean_source)@rotation+mean_target",
            "source_coordinates": str(args.coordinates.resolve()),
            "target_coordinates": str(args.transport_coordinates.resolve()),
            "source_site": site,
            "target_site": other_site,
        }

    modules = {
        "probability_geometry": {
            "decidable": True,
            "eligible": bool(
                probability["all_blocks_calibrated"]
                and probability["at_least_one_regime_distribution_difference"]
            ),
        },
        "sparse_superposition": {"decidable": True, "eligible": bool(sparse["eligible"])},
        "pattern_separation": {
            "decidable": bool(pattern.get("decidable")), "eligible": bool(pattern.get("eligible"))
        },
        "attention_hopfield": {
            "decidable": bool(attention.get("decidable")), "eligible": bool(attention.get("eligible"))
        },
        "model_native_coordinates": model_native,
        "relational_transport": {
            "decidable": bool(transport.get("decidable")), "eligible": bool(transport.get("eligible"))
        },
    }
    report = {
        "status": "development_only_outcome_free",
        "capture": str(args.capture.resolve()),
        "capture_index_sha256": capture_index_sha256,
        "coordinates": str(args.coordinates.resolve()),
        "coordinates_sha256": hashlib.sha256(args.coordinates.read_bytes()).hexdigest(),
        "site": site,
        "pool": args.pool,
        "pre_outcome_window": list(window),
        "modules": modules,
        "probability_geometry": probability,
        "sparse_superposition": sparse,
        "pattern_separation": pattern,
        "attention_hopfield": attention,
        "relational_transport": transport,
        "model_native_coordinates": model_native,
        "outcome_labels_used": False,
        "all_required_modules_decidable": all(value["decidable"] for value in modules.values()),
        "all_decidable_modules_eligible": all(
            value["eligible"] for value in modules.values() if value["decidable"]
        ),
    }

    arrays = {
        "site": np.asarray(site),
        "sparse_mean": np.asarray(sparse["model"]["mean"], dtype=np.float32),
        "sparse_dictionary": np.asarray(sparse["model"]["dictionary"], dtype=np.float32),
        "sparse_alpha": np.asarray(sparse["model"]["alpha"], dtype=np.float32),
        "sparse_active_epsilon": np.asarray(sparse["active_epsilon"], dtype=np.float32),
        "sparse_ista_iterations": np.asarray(sparse["ista_iterations"], dtype=np.int32),
        "sparse_eligible": np.asarray(sparse["eligible"]),
        "sparse_activation_probability": np.asarray(
            sparse["feature_activation_probability"], dtype=np.float32
        ),
        "sparse_activation_probability_by_regime": np.asarray(
            sparse["feature_activation_probability_by_regime"], dtype=np.float32
        ),
        "sparse_support_nll_q99_by_regime": np.asarray(
            sparse["support_negative_log_likelihood_q99_by_regime"], dtype=np.float32
        ),
        "sparse_active_count_bounds": np.asarray(
            sparse["active_count_empirical_q01_q99"],
            dtype=np.float32,
        ),
        "pattern_basis": np.asarray(pattern.get("basis", np.empty((factors[0].shape[1], 0))), dtype=np.float32),
        "pattern_eligible": np.asarray(pattern.get("eligible", False)),
        "attention_hopfield_eligible": np.asarray(attention.get("eligible", False)),
        "attention_restore_site": np.asarray(f"{layer}.attn_out"),
        "model_native_eligible": np.asarray(model_native["eligible"]),
        "probability_geometry_eligible": np.asarray(modules["probability_geometry"]["eligible"]),
        "density_n_blocks": np.asarray(len(probability["blocks"]), dtype=np.int32),
        "density_block_bounds": np.asarray(
            [row["block"] for row in probability["blocks"]], dtype=np.int32
        ),
    }
    if native_arrays is not None:
        arrays.update({
            "model_native_factor_mean": np.asarray(native_arrays["factor_mean"], dtype=np.float32),
            "model_native_target_mean": np.asarray(native_arrays["target_mean"], dtype=np.float32),
            "model_native_readout_weights": np.asarray(native_arrays["readout_weights"], dtype=np.float32),
            "model_native_rowspace_basis": np.asarray(native_arrays["rowspace_basis"], dtype=np.float32),
        })
    for block, models in probability["models"].items():
        arrays[f"density_block{block}_means"] = np.stack(
            [np.asarray(model["mean"]) for model in models]
        ).astype(np.float32)
        arrays[f"density_block{block}_covariances"] = np.stack(
            [np.asarray(model["covariance"]) for model in models]
        ).astype(np.float32)
        arrays[f"density_block{block}_support_q99"] = np.asarray(
            [model["support_q99"] for model in models], dtype=np.float32
        )
        for regime, model in enumerate(models):
            prefix = f"density_block{block}_regime{regime}"
            arrays[f"{prefix}_mean"] = np.asarray(model["mean"], dtype=np.float32)
            arrays[f"{prefix}_covariance"] = np.asarray(model["covariance"], dtype=np.float32)
            arrays[f"{prefix}_family"] = np.asarray(model["family"])
            arrays[f"{prefix}_support_q99"] = np.asarray(model["support_q99"], dtype=np.float32)
    if transport_model is not None:
        arrays.update({
            "transport_mean_source": np.asarray(transport_model["mean_source"], dtype=np.float32),
            "transport_mean_target": np.asarray(transport_model["mean_target"], dtype=np.float32),
            "transport_rotation": np.asarray(transport_model["rotation"], dtype=np.float32),
            "transport_eligible": np.asarray(transport["eligible"]),
        })

    args.out.mkdir(parents=True, exist_ok=True)
    artifact = args.out / "frankenstein_modules.npz"
    np.savez_compressed(artifact, **arrays)
    report["artifact"] = str(artifact.resolve())
    report["artifact_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    report_path = args.out / "frankenstein_gate.json"
    report_path.write_text(json.dumps(strip_artifacts(report), indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "site": site,
        "modules": modules,
        "all_required_modules_decidable": report["all_required_modules_decidable"],
        "report": str(report_path.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
