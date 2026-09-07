"""Freeze the action-geometry protocol using only registered fitting trajectories."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from . import VENDOR_COMMIT
from .action_geometry import GEOMETRY_ARMS, central_estimates, collect_donors
from .backends import JepaBackend
from .benchmark import batches, filter_reviewed_development
from .interventions import validate_frozen_protocol
from .operator_fit import FIT_SEED, _model_versions, make_operator_bank, select_fit_rows
from .protocol import sha256, validate_manifest, write_json


def make_geometry_protocol(receipt, receipt_hash, frozen_at):
    names = ["native", "zero_dose", *GEOMETRY_ARMS, "matched_random"]
    edit = {"site": "block_output", "block": 3, "horizon": 3,
            "token_start": -256, "scale": 1., "tensor": "dynamic_geometry_residual"}
    pairs = [("cubic", name) for name in GEOMETRY_ARMS if name != "cubic"]
    pairs += [(name, control) for name in GEOMETRY_ARMS for control in ("native", "matched_random")]
    threshold = .01 * receipt["fit_native_proprio_mse_h6"]
    protocol = {
        "schema_version": 1, "status": "frozen", "category": "action_response_geometry",
        "vendor_commit": VENDOR_COMMIT, "manifest_sha256": receipt["manifest_sha256"],
        "checkpoint_sha256": receipt["checkpoint_sha256"], "fit_receipt_sha256": receipt_hash,
        "fit_split": "fit", "evaluation_split": "development", "tasks": [receipt["task"]],
        "hypothesis": "At fixed delivered energy, curvature-sensitive four-donor reconstruction improves recorded H6 futures beyond linear, projected, reflected, and random controls.",
        "frozen_at": frozen_at,
        "geometry": {"anchors": [-1., -.5, .5, 1.], "omitted_coordinate": 0.,
                     "action_direction": "fixed_seed_unit_Rademacher_in_normalized_H3_action_space",
                     "normalized_action_radius": .1, "action_seed": 2026090710,
                     "random_control_seed": 2026090711,
                     "delivered_l2": receipt["delivered_l2"],
                     "support": "newest_256_visual_patch_tokens_at_P3_H3",
                     "dimension": "full_patch_field_no_PCA_one_residual_vector_per_arm_and_window",
                     "degenerate_policy": "zero_all_common_energy_arms_keep_window_flag_and_pairing",
                     "donor_ground_truth": "none_model_responses_only",
                     "physical_target": "recorded_future_of_unmodified_recorded_action_recipient",
                     "raw_and_energy_matched_interpretation": "raw_projection_reflection_geometry_is_not_preserved_by_armwise_residual_normalization",
                     "midpoint_identification": "cubic_equals_quadratic_LS_at_t0"},
        "dose_budget": {"rule": "one_common_L2_for_all_active_arms_equal_to_fit_median_linear_reconstruction_residual",
                        "energy_rule": "equal_total_delivered_squared_l2_per_window_across_active_arms",
                        "raw_fidelity_pass": "unscaled_estimates_separate_from_efficacy",
                        "rank": "one_dense_residual_vector_per_window_and_arm"},
        "arms": [{"name": name, "edits": [] if name == "native" else
                  [{**edit, "scale": 0. if name == "zero_dose" else 1.}]} for name in names],
        "primary_contrasts": [{"name": a + "_vs_" + b, "candidate": a, "control": b} for a, b in pairs],
        "development_analysis_plan": {
            "resampling_unit": "lineage_group", "task_pooling": False,
            "primary_forecast_endpoint": "proprio_mse_h6", "bootstrap_replicates": 10000,
            "bootstrap_seed": 2026090704, "interval": "paired_max_standardized_bootstrap_simultaneous_95",
            "smallest_useful_effect": threshold, "equivalence_margin": threshold,
            "threshold_rule": "one_percent_of_fit_only_native_H6_proprio_MSE",
            "arm_selection": "development_only_after_identity_and_fidelity_checks",
            "confirmation": False,
        },
    }
    validate_frozen_protocol(protocol)
    return protocol


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("vendor", "checkpoint", "manifest", "exposure-registry", "data-root", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1 or not args.device.startswith("cuda:"):
        parser.error("Positive batch size and CUDA are required")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        manifest_hash = sha256(args.manifest)
        rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
        validate_manifest(rows)
        registry = json.loads(args.exposure_registry.read_text())
        fit = select_fit_rows(rows, args.task, 128, FIT_SEED)
        development = [row for row in filter_reviewed_development(rows, registry, manifest_hash)
                       if row["task"] == args.task and row["split"] == "development"]
        if not development:
            raise ValueError("No reviewed development population")
        write_json(args.output / "fit_selection.json", fit)
        backend = JepaBackend(args.vendor, args.checkpoint, args.checkpoint_sha256,
                              fit[0]["dataset"], args.device)
        versions = _model_versions(backend.model)
        direction, random = None, None
        residual_norms, errors, window_metadata = [], [], []
        for meta, frames, proprio, raw_actions in batches(fit, args.batch_size, backend, args.data_root, True):
            encoded = backend.encode(frames, proprio)
            actions = backend.normalize_actions(raw_actions)
            if direction is None:
                generator = torch.Generator().manual_seed(2026090710)
                direction = (2 * torch.randint(0, 2, (actions.shape[-1],), generator=generator) - 1).float()
                direction = (direction / direction.norm()).to(backend.device)
            donors, native, predictions = collect_donors(backend, backend.context(encoded), actions, direction, .1)
            estimates, valid = central_estimates(donors)
            if random is None:
                random = torch.randn(native.shape[1:], generator=torch.Generator().manual_seed(2026090711))
                random /= random.norm()
            residual_norms.extend((estimates["equal_anchor_linear"] - native).flatten(1).norm(dim=1).cpu().tolist())
            errors.extend((predictions["proprio"][6] - encoded["proprio"][:, 6]).square().flatten(1).mean(1).cpu().tolist())
            window_metadata.extend(meta)
        dose = float(torch.tensor(residual_norms).median())
        if not torch.isfinite(torch.tensor(dose)) or dose <= 1e-10:
            raise ValueError("Fit interpolation residual is degenerate; no nonzero dose can be frozen")
        if versions != _model_versions(backend.model):
            raise RuntimeError("Frozen model parameters or buffers changed")
        frozen_at = datetime.now(timezone.utc).isoformat()
        receipt = {"schema_version": 1, "status": "fit_only_complete", "category": "action_response_geometry",
                   "task": args.task, "vendor_commit": VENDOR_COMMIT,
                   "manifest_sha256": manifest_hash, "checkpoint_sha256": args.checkpoint_sha256,
                   "fit_split": "fit", "fit_seed": FIT_SEED,
                   "fit_lineage_groups": [row["lineage_group"] for row in fit],
                   "fit_trajectory_ids": [row["trajectory_id"] for row in fit],
                   "fit_window_count": len(window_metadata), "fit_lineage_group_count": len(fit),
                   "development_outcomes_accessed": False, "holdout_outcomes_accessed": False,
                   "fit_native_proprio_mse_h6": sum(errors) / len(errors), "delivered_l2": dose,
                   "direction_choice": "fixed_seed_no_outcome_selection", "created_at": frozen_at,
                   "model_parameters_and_buffers_unchanged": True}
        write_json(args.output / "fit_receipt.json", receipt)
        protocol = make_geometry_protocol(receipt, sha256(args.output / "fit_receipt.json"), frozen_at)
        write_json(args.output / "protocol.json", protocol)
        bank = make_operator_bank(sha256(args.output / "protocol.json"), development,
                                  {"action_direction": direction, "random_direction": random})
        torch.save(bank, args.output / "operator_bank.pt")
        done = {"status": "action_geometry_protocol_frozen", "task": args.task,
                "fit_lineage_groups": len(fit), "fit_windows": len(window_metadata)}
        for name in ("fit_selection.json", "fit_receipt.json", "protocol.json", "operator_bank.pt"):
            done[name.rsplit(".", 1)[0] + "_sha256"] = sha256(args.output / name)
        write_json(args.output / "DONE.json", done)
        print(json.dumps(done), flush=True)
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"status": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
