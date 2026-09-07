"""Fit one target and all fixed support bases without development outcomes."""
from __future__ import annotations

import argparse
import itertools
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import torch

from . import VENDOR_COMMIT
from .backends import JepaBackend
from .benchmark import batches, filter_reviewed_development
from .interventions import validate_frozen_protocol
from .operator_fit import FIT_SEED, _model_versions, make_operator_bank, select_fit_rows
from .protocol import sha256, validate_manifest, write_json
from .support_operator import CATEGORIES, NativeFieldCapture, fit_bases, fit_target


def make_support_protocol(receipt, receipt_hash, category, fitted):
    mapping = fitted["maps"][category]
    names = ["native", "zero_dose", *mapping]
    arms, blocks_all = [], set()
    for name in names:
        if name in ("native", "zero_dose"):
            arms.append({"name": name, "edits": []})
            continue
        key, rank = mapping[name]
        blocks = fitted["bases"][key]["blocks"]
        blocks_all.update(blocks)
        arms.append({"name": name, "operator_rank": rank, "edits": [
            {"site": "block_output", "block": block, "horizon": 3,
             "token_start": -256, "tensor": "dynamic_support_correction", "scale": 1.}
            for block in blocks]})
    arms[1]["edits"] = [{"site": "block_output", "block": block, "horizon": 3,
                         "token_start": -256, "tensor": "dynamic_support_correction", "scale": 0.}
                        for block in sorted(blocks_all)]
    mechanisms = [name for name in mapping if not name.startswith(("matched_random_", "random_position_"))]
    pairs, controls = [], {}
    for name in mechanisms:
        controls[name] = ["native", "matched_random_" + name]
        if "random_position_" + name in mapping:
            controls[name].append("random_position_" + name)
        pairs.extend((name, control) for control in controls[name])
    pairs.extend(itertools.combinations(mechanisms, 2))
    threshold = .01 * receipt["fit_native_proprio_mse_h6"]
    protocol = {
        "schema_version": 1, "status": "frozen", "category": category,
        "vendor_commit": VENDOR_COMMIT, "manifest_sha256": receipt["manifest_sha256"],
        "checkpoint_sha256": receipt["checkpoint_sha256"], "fit_receipt_sha256": receipt_hash,
        "fit_split": "fit", "evaluation_split": "development", "tasks": [receipt["task"]],
        "hypothesis": "At fixed energy, the registered rank/support choices improve H6 recorded-future error beyond native and matched controls.",
        "frozen_at": receipt["created_at"], "arms": arms,
        "support_operator": {"target": "fit_only_PCA3_OLS_H6_visual_residual_from_native_P3_H3",
                             "basis": "raw_activation_PCA_nested_or_direct_sum",
                             "delivered_l2": receipt["delivered_l2"],
                             "response_radius": receipt["delivered_l2"], "probe_chunk_size": 8,
                             "response_horizon": 6, "edit_horizon": 3,
                             "random_spectrum": "orthogonal_projector_not_damped_inverse_map",
                             "degenerate_policy": "zero_all_arms_keep_paired_window",
                             "position_selection": fitted["positions"]},
        "dose_budget": {"energy_rule": "equal_total_delivered_squared_l2_across_ranks" if category == "operator_rank" else "equal_total_delivered_squared_l2_per_arm",
                        "capacity_rule": "fixed_total_direct_sum_rank_per_arm",
                        "random_control_rule": "same_support_rank_spectrum_and_energy",
                        "spectrum_definition": "orthogonal_basis_projector"},
        "primary_contrasts": [{"name": a + "_vs_" + b, "candidate": a, "control": b} for a, b in pairs],
        "development_analysis_plan": {
            "resampling_unit": "lineage_group", "task_pooling": False,
            "primary_forecast_endpoint": "proprio_mse_h6", "bootstrap_replicates": 10000,
            "bootstrap_seed": 2026090704, "interval": "paired_max_standardized_bootstrap_simultaneous_95",
            "smallest_useful_effect": threshold, "equivalence_margin": threshold,
            "threshold_rule": "one_percent_of_fit_only_native_H6_proprio_MSE",
            "mechanism_controls": controls, "confirmation": False},
    }
    validate_frozen_protocol(protocol)
    return protocol


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    return value


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("vendor", "checkpoint", "manifest", "exposure-registry", "data-root", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--native-fit", type=Path)
    args = parser.parse_args()
    if not args.device.startswith("cuda:") or args.batch_size < 1:
        parser.error("Positive batch size and CUDA required")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        manifest_hash = sha256(args.manifest)
        rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
        validate_manifest(rows)
        registry = json.loads(args.exposure_registry.read_text())
        fit = select_fit_rows(rows, args.task, 128, FIT_SEED)
        development = [row for row in filter_reviewed_development(rows, registry, manifest_hash)
                       if row["task"] == args.task and row["split"] == "development"]
        if len(fit) != 128 or not development:
            raise ValueError("Expected 128 fit lineages and reviewed development population")
        write_json(args.output / "fit_selection.json", fit)
        backend, versions, reused_from = None, None, None
        if args.native_fit:
            previous = json.loads((args.native_fit.parent / "fit_receipt.json").read_text())
            for key, expected in (("task", args.task), ("manifest_sha256", manifest_hash),
                                  ("checkpoint_sha256", args.checkpoint_sha256),
                                  ("fit_trajectory_ids", [row["trajectory_id"] for row in fit])):
                if previous.get(key) != expected:
                    raise ValueError(f"Cached fitting provenance mismatch: {key}")
            if previous["native_fit_sha256"] != sha256(args.native_fit):
                raise ValueError("Cached native fitting tensors failed checksum")
            cached = torch.load(args.native_fit, map_location="cpu", weights_only=True)
            fields, errors = cached["fields"], cached["errors"]
            native_error, metadata = cached["native_proprio_error"], cached["metadata"]
            reused_from = {"native_fit_sha256": previous["native_fit_sha256"],
                           "fit_receipt_sha256": sha256(args.native_fit.parent / "fit_receipt.json")}
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
        else:
            backend = JepaBackend(args.vendor, args.checkpoint, args.checkpoint_sha256,
                                  fit[0]["dataset"], args.device)
            versions = _model_versions(backend.model)
            fields, errors, native_error, metadata = [], [], [], []
            for meta, frames, proprio, raw_actions in batches(fit, args.batch_size, backend, args.data_root, True):
                encoded = backend.encode(frames, proprio)
                with NativeFieldCapture(backend.predictor) as capture:
                    predictions = backend.predict(backend.context(encoded), backend.normalize_actions(raw_actions))
                fields.append(torch.stack([capture.values[i] for i in range(6)], 1).cpu())
                errors.append((encoded["visual"][:, 6] - predictions["visual"][6]).flatten(1).cpu())
                native_error.extend((encoded["proprio"][:, 6] - predictions["proprio"][6]).square().flatten(1).mean(1).cpu().tolist())
                metadata.extend(meta)
            fields, errors = torch.cat(fields), torch.cat(errors)
        torch.save({"fields": fields, "errors": errors, "metadata": metadata,
                    "native_proprio_error": native_error}, args.output / "native_fit.pt")
        fields, errors = fields.to(args.device), errors.to(args.device)
        features = fields[:, 3].flatten(1)
        target = fit_target(features, errors)
        bases, maps, positions = fit_bases(fields, errors)
        dose = .005 * float((features - features.mean(0)).square().sum(1).mean().sqrt())
        if not 1e-10 < dose < float("inf"):
            raise ValueError("Invalid frozen dose")
        if backend is not None and versions != _model_versions(backend.model):
            raise RuntimeError("Frozen model parameters or buffers changed")
        fitted = cpu_tree({"target": target, "bases": bases, "maps": maps, "positions": positions})
        receipt = {"schema_version": 1, "status": "fit_only_complete", "task": args.task,
                   "vendor_commit": VENDOR_COMMIT, "manifest_sha256": manifest_hash,
                   "checkpoint_sha256": args.checkpoint_sha256, "fit_split": "fit", "fit_seed": FIT_SEED,
                   "fit_lineage_groups": [row["lineage_group"] for row in fit],
                   "fit_trajectory_ids": [row["trajectory_id"] for row in fit],
                   "fit_window_count": len(metadata), "fit_lineage_group_count": len(fit),
                   "development_outcomes_accessed": False, "holdout_outcomes_accessed": False,
                   "fit_native_proprio_mse_h6": sum(native_error) / len(native_error), "delivered_l2": dose,
                   "native_fit_sha256": sha256(args.output / "native_fit.pt"),
                   "reused_native_fit": reused_from,
                   "model_parameters_and_buffers_unchanged": True,
                   "created_at": datetime.now(timezone.utc).isoformat()}
        source_hash = hashlib.sha256()
        for path in sorted(Path(__file__).parent.glob("*.py")):
            source_hash.update(path.name.encode() + b"\0" + path.read_bytes())
        receipt["harness_source_sha256"] = source_hash.hexdigest()
        write_json(args.output / "fit_receipt.json", receipt)
        for category in CATEGORIES:
            directory = args.output / category
            directory.mkdir()
            write_json(directory / "fit_receipt.json", receipt)
            write_json(directory / "fit_selection.json", fit)
            protocol = make_support_protocol(receipt, sha256(directory / "fit_receipt.json"), category, fitted)
            write_json(directory / "protocol.json", protocol)
            bank = make_operator_bank(sha256(directory / "protocol.json"), development, {})
            bank["support_bank"] = fitted
            torch.save(bank, directory / "operator_bank.pt")
            done = {"status": "support_protocol_frozen", "task": args.task, "category": category}
            for name in ("fit_selection.json", "fit_receipt.json", "protocol.json", "operator_bank.pt"):
                done[name.rsplit(".", 1)[0] + "_sha256"] = sha256(directory / name)
            write_json(directory / "DONE.json", done)
        write_json(args.output / "DONE.json", {"status": "all_support_protocols_frozen", "task": args.task,
                   "seconds": time.monotonic() - started, "fit_windows": len(metadata), "fit_lineages": len(fit),
                   "fit_receipt_sha256": sha256(args.output / "fit_receipt.json"),
                   "native_fit_sha256": receipt["native_fit_sha256"]})
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc)})
        raise


if __name__ == "__main__":
    main()
