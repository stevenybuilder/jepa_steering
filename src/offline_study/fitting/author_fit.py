"""Refit the five fixed sweeps on unprotected, author-train-only families."""
from __future__ import annotations
from offline_study._paths import package_source_hash

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from offline_study import VENDOR_COMMIT
from offline_study.interventions.action_geometry import central_estimates, collect_donors
from offline_study.evaluation.checkpoint.author_runtime import AuthorBackend, encoded_batches, examples, open_normalized_dataset, prefix_batches, validate_cohort
from offline_study.fitting.geometry_fit import make_geometry_protocol
from offline_study.fitting.operator_fit import NativeCouplingCapture, fit_coupled_directions, make_protocol, orthogonal_random_control, permute_visual_direction, _model_versions
from offline_study.core.protocol import sha256, write_json
from offline_study.fitting.support_fit import cpu_tree, make_support_protocol
from offline_study.interventions.support_operator import CATEGORIES, NativeFieldCapture, fit_bases, fit_target


def source_hash():
    return package_source_hash()


def add_reporting_contract(protocol, receipt, cohort_hash):
    """Register all native comparisons before outcomes; keep all existing arms."""
    names = [arm["name"] for arm in protocol["arms"]]
    existing = {(item["candidate"], item["control"]) for item in protocol["primary_contrasts"]}
    for name in names:
        if name != "native" and (name, "native") not in existing:
            protocol["primary_contrasts"].append({"name": name + "_vs_native", "candidate": name, "control": "native"})
    protocol["author_validation"] = {"cohort_sha256": cohort_hash, "precision": receipt["precision"],
        "context": 3, "prefixes": "all", "coverage": receipt["coverage"],
        "metrics": "official embedding L1/L2 H1-H6; no decoded state head loaded",
        "original_arms_unchanged": True, "protected_rows_excluded": True,
        "clip_weighted_point_estimates": "full available unprotected author-validation clips",
        "statistical_unit": "prefixes then clips then trajectory then lineage",
        "precision_selection": "forbidden; bfloat16 primary, float32 sensitivity"}
    plan = protocol["development_analysis_plan"]
    plan.update(interval="paired_max_standardized_bootstrap_simultaneous_95",
                bootstrap_replicates=10000, bootstrap_seed=2026090704,
                primary_forecast_endpoint="proprio_mse_h6",
                smallest_useful_effect=.01 * receipt["fit_native_proprio_mse_h6"],
                equivalence_margin=.01 * receipt["fit_native_proprio_mse_h6"],
                confirmation=False,
                reporting_family="all registered contrasts per task/category; both H6 visual and proprio endpoints jointly",
                secondary_metrics="L1/L2 at every H1-H6; no endpoint selection",
                arm_selection=False)
    return protocol


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("vendor", "checkpoint", "cohort", "data-root", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--precision", choices=["bfloat16", "float32"], required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        started = time.monotonic()
        cohort = json.loads(args.cohort.read_text())
        validate_cohort(cohort)
        config, fit = cohort["reference_config"], cohort["fit"]
        backend = AuthorBackend(args.vendor, args.checkpoint, args.checkpoint_sha256,
                                cohort["dataset"], args.device, args.precision)
        versions = _model_versions(backend.model)
        dataset = open_normalized_dataset(cohort["dataset"], args.data_root, config, True)
        fields, errors, visuals, conditions, native_error, visual_error, residuals, metadata = [], [], [], [], [], [], [], []
        direction, random, parity = None, None, None
        for clip_meta, encoded in encoded_batches(backend, dataset, fit, config, fitting=True):
            if parity is None:
                parity = backend.verify_reference(encoded)
                write_json(args.output / "PARITY.json", {"precision": args.precision, "source": "actual core VideoWM.rollout and compute_loss",
                                                        "fit_only": True, "checks": parity})
            for meta, context, actions, target in prefix_batches(backend, clip_meta, encoded):
                with NativeCouplingCapture(backend.predictor) as coupling, NativeFieldCapture(backend.predictor) as capture:
                    pred = backend.predict(context, actions)
                visuals.append(coupling.visual)
                conditions.append(coupling.condition)
                fields.append(torch.stack([capture.values[i] for i in range(6)], 1).float().cpu())
                errors.append((target["visual"][:, 6].float() - pred["visual"][6].float()).flatten(1).cpu())
                scores = backend.metrics(pred, target)
                native_error.extend(item["proprio_mse_h6"] for item in scores)
                visual_error.extend(item["visual_mse_h6"] for item in scores)
                if direction is None:
                    generator = torch.Generator().manual_seed(2026090710)
                    direction = (2 * torch.randint(0, 2, (actions.shape[-1],), generator=generator) - 1).float()
                    direction = (direction / direction.norm()).to(backend.device)
                donors, native, _ = collect_donors(backend, context, actions, direction, .1)
                estimates, _ = central_estimates(donors.float())
                if random is None:
                    random = torch.randn(native.shape[1:], generator=torch.Generator().manual_seed(2026090711))
                    random /= random.norm()
                residuals.extend((estimates["equal_anchor_linear"] - native.float()).flatten(1).norm(dim=1).cpu().tolist())
                metadata.extend(meta)
            if len(metadata) % 32 == 0:
                print(json.dumps({"stage": "fit_capture", "task": cohort["task"], "precision": args.precision,
                                  "examples": len(metadata), "seconds": time.monotonic() - started}), flush=True)
        if len(metadata) != 512 or len({(m["trajectory_id"], m["start"]) for m in metadata}) != 512:
            raise ValueError("Fit coverage differs from the frozen 128 x 4 examples")
        vd, ad, diagnostic = fit_coupled_directions(torch.cat(visuals), torch.cat(conditions), 12)
        del visuals, conditions
        fields, errors = torch.cat(fields), torch.cat(errors)
        torch.save({"fields": fields, "errors": errors, "metadata": metadata,
                    "native_proprio_error": native_error, "native_visual_error": visual_error}, args.output / "native_fit.pt")
        fields, errors = fields.to(backend.device), errors.to(backend.device)
        features = fields[:, 3].flatten(1)
        target_fit = fit_target(features, errors)
        bases, maps, positions = fit_bases(fields, errors)
        support_dose = .005 * float((features - features.mean(0)).square().sum(1).mean().sqrt())
        geometry_dose = float(torch.tensor(residuals).median())
        if not all(math.isfinite(dose) and dose > 1e-10 for dose in (support_dose, geometry_dose)):
            raise ValueError("A fixed fit dose is degenerate; do not invent a replacement dose")
        if versions != _model_versions(backend.model):
            raise RuntimeError("Frozen model changed during fit")
        fitted = cpu_tree({"target": target_fit, "bases": bases, "maps": maps, "positions": positions})
        receipt = {"status": "author_train_fit_only_complete", "task": cohort["task"],
                   "vendor_commit": VENDOR_COMMIT, "manifest_sha256": cohort["source_manifest_sha256"],
                   "cohort_sha256": sha256(args.cohort), "checkpoint_sha256": args.checkpoint_sha256,
                   "precision": args.precision, "fit_split": "fit", "fit_lineage_group_count": len(fit),
                   "fit_lineage_groups": [row["lineage_group"] for row in fit],
                   "fit_trajectory_ids": [row["trajectory_id"] for row in fit],
                   "fit_window_count": len(metadata), "coverage": cohort["coverage"],
                   "fit_native_proprio_mse_h6": sum(native_error) / len(native_error),
                   "fit_native_visual_mse_h6": sum(visual_error) / len(visual_error),
                   "development_outcomes_accessed": False, "holdout_outcomes_accessed": False,
                   "created_at": datetime.now(timezone.utc).isoformat(),
                   "native_fit_sha256": sha256(args.output / "native_fit.pt"),
                   "harness_source_sha256": source_hash(), "backend": backend.provenance,
                   "geometry_delivered_l2": geometry_dose, "support_delivered_l2": support_dose,
                   "model_parameters_and_buffers_unchanged": True}
        write_json(args.output / "fit_receipt.json", receipt)
        write_json(args.output / "fit_selection.json", fit)
        write_json(args.output / "cohort.json", cohort)
        eval_meta = examples(cohort["evaluation"], config)
        for category in ("vision_action_coupling", "action_response_geometry", *CATEGORIES):
            directory = args.output / category
            directory.mkdir()
            local_receipt = {**receipt, "category": category,
                             "delivered_l2": geometry_dose if category == "action_response_geometry" else support_dose}
            write_json(directory / "fit_receipt.json", local_receipt)
            receipt_hash = sha256(directory / "fit_receipt.json")
            if category == "vision_action_coupling":
                protocol = make_protocol(cohort["task"], receipt["manifest_sha256"], args.checkpoint_sha256, receipt_hash,
                    .1 * diagnostic["visual_score_robust_sigma"], .1 * diagnostic["action_score_robust_sigma"], receipt["created_at"])
                tensors = {"visual_direction": vd, "action_direction": ad,
                    "permuted_visual_direction": permute_visual_direction(vd, 2026090702),
                    "random_visual_direction": orthogonal_random_control(vd, 2026090703),
                    "random_action_direction": orthogonal_random_control(ad, 2026090704)}
            elif category == "action_response_geometry":
                protocol = make_geometry_protocol(local_receipt, receipt_hash, receipt["created_at"])
                tensors = {"action_direction": direction, "random_direction": random}
            else:
                protocol = make_support_protocol(local_receipt, receipt_hash, category, fitted)
                tensors = {}
            protocol = add_reporting_contract(protocol, receipt, sha256(args.cohort))
            write_json(directory / "protocol.json", protocol)
            bank = {"schema_version": 1, "protocol_sha256": sha256(directory / "protocol.json"),
                    "global_tensors": {k: v.detach().float().cpu() for k, v in tensors.items()}}
            # Use the established key function, not an independently invented encoding.
            from offline_study.interventions.interventions import window_key
            bank["rows"] = {window_key(m): {**m, "tensors": {}} for m in eval_meta}
            if category in CATEGORIES:
                bank["support_bank"] = fitted
            torch.save(bank, directory / "operator_bank.pt")
            write_json(directory / "DONE.json", {"status": "author_protocol_frozen",
                "cohort_sha256": sha256(args.cohort), "protocol_sha256": sha256(directory / "protocol.json"),
                "operator_bank_sha256": sha256(directory / "operator_bank.pt"),
                "fit_receipt_sha256": receipt_hash})
        write_json(args.output / "DONE.json", {"status": "all_five_author_protocols_frozen", "task": cohort["task"],
            "seconds": time.monotonic() - started, "fit_receipt_sha256": sha256(args.output / "fit_receipt.json"),
            "cohort_sha256": sha256(args.cohort), "native_fit_sha256": receipt["native_fit_sha256"]})
        print(json.dumps({"status": "fit_complete", "task": cohort["task"], "precision": args.precision,
                          "seconds": time.monotonic() - started}), flush=True)
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"status": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
