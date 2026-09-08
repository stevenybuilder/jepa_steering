"""Bounded fitting-only response calibration for the user-selected Design 1.

No development/protected data, candidate selection, planner, or simulator calls.
The audit partition holds out response measurements only: the inherited PCA/error
readout already used the original 128 fitting families. It is not confirmation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import torch

from .author_evaluate import verify_fit
from .author_fit import source_hash
from .author_runtime import (AuthorBackend, encoded_batches, examples,
    open_normalized_dataset, prefix_batches, validate_cohort)
from .fixed_response import ARMS, METHOD, FixedResponseIntervention, make_bank, load_fitted_bank
from .interventions import PredictorIntervention
from .operator_fit import _model_versions
from .protocol import sha256, write_json
from .support_operator import compile_fields, expand_basis


def select_families(cohort, config):
    validate_cohort(cohort)
    if cohort["task"] not in config["tasks"] or cohort["dataset"] != "metaworld":
        raise ValueError("Initial successor fit is restricted to Reach and Reach-Wall")
    ordered = sorted(cohort["fit"], key=lambda row: hashlib.sha256(
        (str(config["selection_seed"]) + ":" + row["lineage_group"]).encode()).hexdigest())
    n, m = config["response_calibration_families"], config["response_audit_families"]
    if (n, m, config["examples_per_family"]) != (24, 8, 4) or len(ordered) < n + m:
        raise ValueError("Bounded 24/8 family protocol changed")
    return {"calibration": ordered[:n], "response_audit": ordered[n:n + m]}


def collect_responses(backend, context, actions, fitted, radius, chunk_size=4):
    """Sixteen fitting-only probes, four learned and four matched-random axes."""
    expanded = {key: expand_basis(fitted["bases"][key], 4).to(backend.device)
                for key in ("rank", "rank_random")}
    probes = [(key, j, sign) for key in expanded for j in range(4) for sign in (-1, 1)]
    outputs, batch = {}, actions.shape[1]
    for start in range(0, len(probes), chunk_size):
        chunk = probes[start:start + chunk_size]
        delta = torch.stack([expanded[key][j] * sign * radius for key, j, sign in chunk])
        fields = compile_fields(delta[None].expand(batch, -1, -1, -1, -1), {3})
        with PredictorIntervention(backend.predictor, fields):
            pred = backend.predict(backend.expand_context(context, len(chunk)),
                                   actions.repeat_interleave(len(chunk), dim=1))
        final = pred["visual"][6].float().reshape(batch, len(chunk), -1)
        for i, probe in enumerate(chunk):
            outputs[probe] = final[:, i].clone()
    result = {key: torch.stack([(outputs[key, j, 1] - outputs[key, j, -1]) / (2 * radius)
                               for j in range(4)], 1) for key in expanded}
    if any(not torch.isfinite(value).all() for value in result.values()):
        raise ValueError("Nonfinite fitting responses")
    return result


def check_examples(actual, rows, config):
    expected = examples(rows, config, fitting=True)
    keys = lambda items: [(m["trajectory_id"], m["start"]) for m in items]
    if (len(keys(actual)) != len(set(keys(actual))) or set(keys(actual)) != set(keys(expected)) or
            set(Counter(m["lineage_group"] for m in actual).values()) != {4}):
        raise ValueError("Missing/duplicate fitting examples or unequal family weights")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "cohort", "fit", "vendor", "checkpoint", "data-root", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if (config.get("method") != METHOD or config.get("selected_design") != 1 or
            config.get("fit_precision") != "bfloat16" or config.get("probe_chunk_size") != 4 or
            config.get("runtime_arms") != list(ARMS) or config.get("max_fit_seconds_per_task") != 3600):
        raise ValueError("Unregistered fitting configuration")
    cohort = json.loads(args.cohort.read_text())
    selected = select_families(cohort, config)
    receipt, protocol = verify_fit(args.fit, sha256(args.cohort), args.checkpoint_sha256, "bfloat16")
    if (receipt["fit_lineage_groups"] != [r["lineage_group"] for r in cohort["fit"]] or
            receipt["task"] != cohort["task"] or protocol["category"] != "operator_rank"):
        raise ValueError("Source fit population/task/category changed")
    source = torch.load(args.fit / "operator_bank.pt", map_location="cpu", weights_only=True)
    if source["protocol_sha256"] != sha256(args.fit / "protocol.json"):
        raise ValueError("Source bank/protocol mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        binding = {"method": METHOD, "task": cohort["task"],
            "checkpoint_sha256": args.checkpoint_sha256, "cohort_sha256": sha256(args.cohort),
            "config_sha256": sha256(args.config), "source_bank_sha256": sha256(args.fit / "operator_bank.pt"),
            "source_protocol_sha256": sha256(args.fit / "protocol.json"),
            "source_fit_receipt_sha256": sha256(args.fit / "fit_receipt.json"),
            "source_sha256": source_hash(), "fit_precision": "bfloat16"}
        write_json(args.output / "contract.json", {**config, "binding": binding,
            "selection": selected, "response_audit_is_independent_of_inherited_basis_fit": False})
        write_json(args.output / "cohort.json", cohort)
        backend = AuthorBackend(args.vendor, args.checkpoint, args.checkpoint_sha256,
                                "metaworld", args.device, "bfloat16")
        versions = _model_versions(backend.model)
        dataset = open_normalized_dataset("metaworld", args.data_root, cohort["reference_config"], True)
        response_sum, calibration_meta, audit_meta, audit_rows = {}, [], [], []
        bank, adapters, parity = None, None, None
        radius = protocol["support_operator"]["response_radius"]
        for role, rows in selected.items():
            for clips, encoded in encoded_batches(backend, dataset, rows, cohort["reference_config"], fitting=True):
                if parity is None:
                    parity = backend.verify_reference(encoded)
                    write_json(args.output / "PARITY.json", {"fit_only": True, "checks": parity})
                for meta, context, actions, target in prefix_batches(backend, clips, encoded):
                    if time.monotonic() - started > config["max_fit_seconds_per_task"]:
                        raise TimeoutError("Fixed response fitting time cap; no partial bank promoted")
                    responses = collect_responses(backend, context, actions, source["support_bank"], radius)
                    if role == "calibration":
                        for key, response in responses.items():
                            part = response.double().sum(0)
                            response_sum[key] = response_sum.get(key, 0) + part
                        calibration_meta.extend(meta)
                    else:
                        audit_meta.extend(meta)
                        native = adapters["native"](context, actions)
                        for arm, adapter in adapters.items():
                            prediction = native if arm == "native" else adapter(context, actions)
                            if arm == "zero_dose" and any(not torch.equal(native[k], prediction[k]) for k in native):
                                raise ValueError("One-pass zero-dose identity failed")
                            metrics = backend.metrics(prediction, target)
                            record = adapter.last_record
                            for i, item in enumerate(meta):
                                row = {**item, "arm": arm, "metrics": metrics[i]}
                                if arm in ARMS[2:]:
                                    key = "rank_random" if arm.startswith("matched_random") else "rank"
                                    row["relative_response_deviation"] = float(
                                        (responses[key][i].double() - means[key]).norm() / means[key].norm().clamp_min(1e-18))
                                    row["requested_l2"] = float(record["requested_l2"][i])
                                    row["realized_l2"] = float(record["realized_l2"][i])
                                    row["coefficients"] = record["coefficients"][i].cpu().tolist()
                                audit_rows.append(row)
                print(json.dumps({"stage": role, "calibration_examples": len(calibration_meta),
                    "response_audit_examples": len(audit_meta), "seconds": time.monotonic() - started}), flush=True)
            check_examples(calibration_meta if role == "calibration" else audit_meta, rows, cohort["reference_config"])
            if role == "calibration":
                means = {k: v / len(calibration_meta) for k, v in response_sum.items()}
                bank = make_bank(source, means, protocol["support_operator"]["delivered_l2"], binding)
                torch.save(bank, args.output / "operator_bank.pt")
                torch.save({k: v.cpu() for k, v in means.items()}, args.output / "mean_responses.pt")
                adapters = {arm: FixedResponseIntervention(backend, bank, arm) for arm in ARMS}
        if _model_versions(backend.model) != versions:
            raise ValueError("Frozen model changed")
        if source_hash() != binding["source_sha256"] or sha256(args.config) != binding["config_sha256"]:
            raise ValueError("Source/config changed during fitting; immutable staging required")
        write_json(args.output / "audit_metrics.json", audit_rows)
        write_json(args.output / "report.json", {"status": "fit_only_calibration_and_response_audit_complete",
            "binding": binding, "calibration_families": 24, "response_audit_families": 8,
            "calibration_examples": len(calibration_meta), "response_audit_examples": len(audit_meta),
            "response_probe_forecasts": 16 * (len(calibration_meta) + len(audit_meta)),
            "runtime_probe_forecasts": 0, "runtime_native_shadow_forecasts": 0,
            "source_basis_readout_fit_families": 128, "response_audit_is_untouched_confirmation": False,
            "development_or_protected_outcomes_accessed": False, "zero_dose_identity": True,
            "behavioral_launch_ready": False, "fresh_confirmation": False,
            "measured_planning_speedup": None, "wall_seconds": time.monotonic() - started})
        write_json(args.output / "DONE.json", {"status": "fit_only_complete_not_behavioral_clearance", **{
            name: sha256(args.output / name) for name in ("contract.json", "cohort.json", "operator_bank.pt",
                "mean_responses.pt", "PARITY.json", "audit_metrics.json", "report.json")}})
        load_fitted_bank(args.output, task=cohort["task"], checkpoint_sha256=args.checkpoint_sha256)
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "partial_not_eligible": True})
        raise


if __name__ == "__main__":
    main()
