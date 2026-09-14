"""Frozen, full previously exposed MetaWorld comparisons for Design 1.

This is an append-only successor evaluation, not a rewrite of the old access
registry or permission to open any other protected trajectories.
"""
import argparse
import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from offline_study.evaluation.checkpoint.author_access import CATEGORIES, PRECISIONS, digest, verify_runtime_access
from offline_study.evaluation.checkpoint.author_analyze import analyze, expected_shards
from offline_study.evaluation.checkpoint.author_evaluate import check_coverage
from offline_study.fitting.author_fit import source_hash
from offline_study.evaluation.checkpoint.author_runtime import AuthorBackend, encoded_batches, examples, open_normalized_dataset, prefix_batches, validate_cohort
from offline_study.interventions.fixed_response import ARMS, METHOD, FixedResponseIntervention, load_fitted_bank
from offline_study.fitting.operator_fit import _model_versions
from offline_study.planning.planning_native_smoke import CHECKPOINTS
from offline_study.core.protocol import sha256, write_json

COUNTS = {"reach": 33, "reach-wall": 27}
PAIRS = [(arm, "native") for arm in ARMS[1:]] + [(ARMS[2], ARMS[3])]
ROLE = "previously_exposed_fixed_response_development"


def validate_measurements(rows, expected):
    check_coverage(rows, expected, ARMS)
    identities = {(row["trajectory_id"], row["start"]): row for row in expected}
    for row in rows:
        identity = identities[row["trajectory_id"], row["start"]]
        if any(row.get(key) != value for key, value in identity.items()):
            raise ValueError("A measurement changed its source lineage, task or prefix identity")
        if not all(math.isfinite(value) for value in row["metrics"].values()):
            raise ValueError("Nonfinite recorded-future metric")


def exposed_population(evidence, task):
    """Recheck original authorization AND completed exposure, preserving labels."""
    extension_path = evidence / "extension/cohort.json"
    extension = json.loads(extension_path.read_text())
    base_path = evidence / "extension/source_cohort.json"
    base = json.loads(base_path.read_text())
    validate_cohort(base)
    validate_cohort(extension)
    if (not verify_runtime_access(extension_path, extension) or base["task"] != "mw-" + task or
            base["fit"] != extension["fit"] or base["evaluation"] != extension["reused_evaluation"] or
            base["protected_official_validation"] != extension["evaluation"]):
        raise ValueError("Prior exposure/fitting history changed")
    rows = base["evaluation"] + extension["evaluation"]
    if (len(rows) != COUNTS[task] or len({r["lineage_group"] for r in rows}) != COUNTS[task] or
            {r["lineage_group"] for r in rows} & {r["lineage_group"] for r in base["fit"]}):
        raise ValueError("Wrong full population or fitting overlap")
    for precision in PRECISIONS:
        for category in CATEGORIES:
            directory = evidence / "prior-full-pool" / precision / category
            prior = json.loads((directory / "report.json").read_text())
            done = json.loads((directory / "DONE.json").read_text())
            if ((directory / "FAILED.json").exists() or done["report_sha256"] != sha256(directory / "report.json") or
                    prior.get("status") != "verified_full_primary_author_split_replication_complete" or
                    prior.get("task") != "mw-" + task or prior.get("precision") != precision or
                    prior.get("category") != category or prior.get("rollout_trajectories") != COUNTS[task] or
                    prior.get("fresh_confirmation") is not False or
                    prior.get("extension_cohort_sha256") != sha256(extension_path)):
                raise ValueError("Missing completed prior full-pool exposure evidence")
    return base, rows


def validate_protocol(protocol):
    if (protocol.get("schema_version") != 1 or protocol.get("status") != "frozen" or
            protocol.get("method") != METHOD or protocol.get("task") not in COUNTS or
            protocol.get("evaluation_role") != ROLE or protocol.get("fresh_confirmation") is not False or
            protocol.get("newly_opened_untouched_rows") != 0 or protocol.get("shard_count") != 4 or
            protocol.get("precisions") != list(PRECISIONS) or protocol.get("primary_precision") != "bfloat16" or
            protocol.get("arms") != [{"name": arm} for arm in ARMS] or
            protocol.get("primary_contrasts") != [{"name": a + "_vs_" + b, "candidate": a, "control": b}
                                                   for a, b in PAIRS] or
            protocol.get("offline_significance_required_for_behavioral_admission") is not False):
        raise ValueError("Fixed successor development registry changed")


def load_protocol(path, fit):
    path, fit = Path(path), Path(fit)
    frozen = json.loads((path.parent / "FROZEN.json").read_text())
    if (path.parent / "FAILED.json").exists() or sha256(path) != frozen["protocol_sha256"]:
        raise ValueError("Offline successor freeze changed/failed")
    protocol = json.loads(path.read_text())
    validate_protocol(protocol)
    for relative, expected in protocol["evidence_sha256"].items():
        item = path.parent / relative
        if not item.resolve().is_relative_to(path.parent.resolve()) or sha256(item) != expected:
            raise ValueError("Frozen exposure evidence changed")
    base, rows = exposed_population(path.parent / "evidence", protocol["task"])
    bank = load_fitted_bank(fit, task="mw-" + protocol["task"], checkpoint_sha256=CHECKPOINTS["metaworld"])
    if (sha256(fit / "DONE.json") != protocol["fit_done_sha256"] or
            sha256(fit / "operator_bank.pt") != protocol["fit_bank_sha256"] or
            bank["binding"]["cohort_sha256"] != sha256(path.parent / "evidence/extension/source_cohort.json") or
            digest(rows) != protocol["evaluation_rows_sha256"] or
            len(rows) != protocol["trajectory_count"]):
        raise ValueError("Frozen fit or full population changed")
    return protocol, base, rows, bank


def freeze(args):
    load_fitted_bank(args.fit, task="mw-" + args.task, checkpoint_sha256=CHECKPOINTS["metaworld"])
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        evidence = args.output / "evidence"
        original = args.original_root
        extension = original / "metaworld-author-extension-20260907"
        shutil.copytree(extension / "cohorts" / args.task, evidence / "extension")
        for precision in PRECISIONS:
            for category in CATEGORIES:
                source = extension / "analysis-full-v1" / precision / args.task / category
                destination = evidence / "prior-full-pool" / precision / category
                destination.mkdir(parents=True)
                for name in ("report.json", "DONE.json"):
                    shutil.copyfile(source / name, destination / name)
        base, rows = exposed_population(evidence, args.task)
        source_receipt_path = original / "fits-v1/bfloat16" / args.task / "operator_rank/fit_receipt.json"
        source_receipt = json.loads(source_receipt_path.read_text())
        fit_contract = json.loads((args.fit / "contract.json").read_text())
        if sha256(source_receipt_path) != fit_contract["binding"]["source_fit_receipt_sha256"]:
            raise ValueError("Minimum-effect reference is not the inherited source fit")
        shutil.copyfile(source_receipt_path, evidence / "source_fit_receipt.json")
        protocol = {"schema_version": 1, "status": "frozen", "method": METHOD, "task": args.task,
            "frozen_at": datetime.now(timezone.utc).isoformat(), "source_sha256": source_hash(),
            "authority": "user_20260907_supplied_Design1_and_authorized_affected_comparison_reruns",
            "evaluation_role": ROLE, "fresh_confirmation": False, "newly_opened_untouched_rows": 0,
            "historical_protection_retained": True, "fit_done_sha256": sha256(args.fit / "DONE.json"),
            "fit_bank_sha256": sha256(args.fit / "operator_bank.pt"), "evaluation_rows_sha256": digest(rows),
            "trajectory_count": len(rows), "prefixes_per_arm": len(examples(rows, base["reference_config"])),
            "arms": [{"name": arm} for arm in ARMS], "shard_count": 4,
            "primary_contrasts": [{"name": a + "_vs_" + b, "candidate": a, "control": b} for a, b in PAIRS],
            "hypothesis": "The fixed-response successor reduces recorded-future embedding error beyond native and its matched-random map; old operator gains are not assumed to transfer.",
            "precisions": list(PRECISIONS), "primary_precision": "bfloat16",
            "fit_precision": "bfloat16", "same_frozen_BF16_fitted_map_in_FP32_sensitivity": True,
            "offline_context": 3, "horizon": 6, "recorded_actions_unchanged": True,
            "all_author_clips_and_prefixes": True, "encoding_batch_size": 4,
            "offline_significance_required_for_behavioral_admission": False,
            "analysis": {"primary_endpoint": "proprio_mse_h6", "bootstrap_seed": 2026090704,
                "bootstrap_replicates": 10000, "family": "all four contrasts x both H6 embedding MSE endpoints per task/precision",
                "minimum_useful_effect": "1% of inherited BF16 fit-only native error; same reference in FP32 sensitivity",
                "unit": "prefixes within clip, clips within trajectory, trajectories within lineage",
                "winner_selection": False},
            "evidence_sha256": {str(p.relative_to(args.output)): sha256(p)
                                for p in sorted(evidence.rglob("*")) if p.is_file()}}
        validate_protocol(protocol)
        write_json(args.output / "protocol.json", protocol)
        write_json(args.output / "FROZEN.json", {"protocol_sha256": sha256(args.output / "protocol.json"),
            "new_development_outcomes_accessed": False, "fresh_confirmation": False})
        load_protocol(args.output / "protocol.json", args.fit)
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "no_new_outcomes_accessed": True})
        raise


def run_batch(backend, adapters, metadata, context, actions, target):
    native = adapters["native"](context, actions)
    rows = []
    for arm, adapter in adapters.items():
        prediction = native if arm == "native" else adapter(context, actions)
        if arm == "zero_dose" and any(not torch.equal(native[k], prediction[k]) for k in ("visual", "proprio")):
            raise ValueError("Successor zero-dose identity failed")
        metrics = backend.metrics(prediction, target)
        record = adapter.last_record
        for i, item in enumerate(metadata):
            requested = float(record["requested_l2"][i]) if arm in ARMS[2:] else 0.
            realized = float(record["realized_l2"][i]) if arm in ARMS[2:] else 0.
            rows.append({**item, "arm": arm, "metrics": metrics[i], "requested_l2": requested,
                "realized_l2": realized,
                "realized_energy_within_fp32_tolerance": abs(realized - requested) <= 1e-5 + 1e-3 * abs(requested)})
    return rows


@torch.no_grad()
def evaluate(args):
    protocol, base, population, bank = load_protocol(args.protocol, args.fit)
    if (source_hash() != protocol["source_sha256"] or args.precision not in PRECISIONS or
            not 0 <= args.shard_index < protocol["shard_count"]):
        raise ValueError("Unfrozen source, precision or shard")
    selected = [row for i, row in enumerate(population) if i % protocol["shard_count"] == args.shard_index]
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        write_json(args.output / "protocol.json", protocol)
        write_json(args.output / "selection.json", selected)
        backend = AuthorBackend(args.vendor, args.checkpoint, CHECKPOINTS["metaworld"], "metaworld", "cuda:0", args.precision)
        versions = _model_versions(backend.model)
        dataset = open_normalized_dataset("metaworld", args.data_root, base["reference_config"], False)
        adapters = {arm: FixedResponseIntervention(backend, bank, arm) for arm in ARMS}
        # Complete source-native parity on one permitted fitting trajectory before
        # successor development outcomes, independently in each frozen precision.
        fitting_rows, checks = [], []
        for clips, encoded in encoded_batches(backend, dataset, base["fit"][:1], base["reference_config"], fitting=True):
            checks.append(backend.verify_reference(encoded))
            for meta, context, actions, target in prefix_batches(backend, clips, encoded):
                fitting_rows.extend(run_batch(backend, adapters, meta, context, actions, target))
        validate_measurements(fitting_rows, examples(base["fit"][:1], base["reference_config"], fitting=True))
        write_json(args.output / "PARITY.json", {"fit_only": True, "checks": checks,
            "fit_trajectory": base["fit"][0]["trajectory_id"], "zero_dose_identity": True,
            "precision": args.precision, "scientific_efficacy_measurement": False})
        measurements = []
        for clips, encoded in encoded_batches(backend, dataset, selected, base["reference_config"], role=ROLE):
            for meta, context, actions, target in prefix_batches(backend, clips, encoded):
                if time.monotonic() - started > 7200:
                    raise TimeoutError("Offline shard time cap; partial result is not complete")
                measurements.extend(run_batch(backend, adapters, meta, context, actions, target))
            write_json(args.output / "progress.json", {"prefixes_per_arm": len(measurements) // len(ARMS),
                "seconds": time.monotonic() - started, "task": protocol["task"], "precision": args.precision})
        expected = examples(selected, base["reference_config"], role=ROLE)
        validate_measurements(measurements, expected)
        if _model_versions(backend.model) != versions or source_hash() != protocol["source_sha256"]:
            raise ValueError("Frozen source/model changed")
        write_json(args.output / "window_metrics.json", measurements)
        report = {"status": "fixed_response_offline_shard_complete", "method": METHOD,
            "protocol_sha256": sha256(args.protocol), "task": protocol["task"], "precision": args.precision,
            "shard_index": args.shard_index, "shard_count": protocol["shard_count"],
            "trajectory_count": len(selected), "prefixes_per_arm": len(expected), "arms": list(ARMS),
            "fit_done_sha256": sha256(args.fit / "DONE.json"), "source_sha256": source_hash(),
            "zero_dose_identity": True, "parameters_unchanged": True, "backend": backend.provenance,
            "seconds": time.monotonic() - started, "fresh_confirmation": False,
            "newly_opened_untouched_rows": 0, "evaluation_role": ROLE}
        write_json(args.output / "report.json", report)
        write_json(args.output / "DONE.json", {name: sha256(args.output / name) for name in
            ("protocol.json", "selection.json", "PARITY.json", "window_metrics.json", "report.json")})
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "partial_not_complete": True})
        raise


def summarize(args):
    protocol, base, population, _ = load_protocol(args.protocol, args.fit)
    if args.precision not in PRECISIONS:
        raise ValueError("Unregistered analysis precision")
    measurements, inputs = [], {}
    for directory in expected_shards(args.shards, protocol["shard_count"]):
        done = json.loads((directory / "DONE.json").read_text())
        if (directory / "FAILED.json").exists():
            raise ValueError("Failed shard cannot enter analysis")
        for name in ("protocol.json", "selection.json", "PARITY.json", "window_metrics.json", "report.json"):
            if sha256(directory / name) != done[name]:
                raise ValueError("Changed offline shard")
            inputs[str(directory / name)] = done[name]
        report = json.loads((directory / "report.json").read_text())
        if (report["status"] != "fixed_response_offline_shard_complete" or
                report["protocol_sha256"] != sha256(args.protocol) or report["task"] != protocol["task"] or
                report["precision"] != args.precision or report["shard_count"] != protocol["shard_count"] or
                directory.name != "shard-" + str(report["shard_index"]) or
                report["fit_done_sha256"] != sha256(args.fit / "DONE.json") or
                not report["zero_dose_identity"] or not report["parameters_unchanged"]):
            raise ValueError("Incompatible successor shards")
        selection = [r for i, r in enumerate(population) if i % protocol["shard_count"] == report["shard_index"]]
        if json.loads((directory / "selection.json").read_text()) != selection:
            raise ValueError("Wrong trajectory shard")
        rows = json.loads((directory / "window_metrics.json").read_text())
        validate_measurements(rows, examples(selection, base["reference_config"], role=ROLE))
        measurements.extend(rows)
    validate_measurements(measurements, examples(population, base["reference_config"], role=ROLE))
    receipt = json.loads((args.protocol.parent / "evidence/source_fit_receipt.json").read_text())
    report = analyze(measurements, protocol, receipt, "mw-" + protocol["task"])
    report.update(status="verified_fixed_response_offline_complete", method=METHOD, task=protocol["task"],
        precision=args.precision, input_sha256=inputs, protocol_sha256=sha256(args.protocol),
        trajectory_count=len(population), prefixes_per_arm=protocol["prefixes_per_arm"],
        evaluation_role=ROLE, offline_significance_required_for_behavioral_admission=False)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "report.json", report)
    write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "evaluate", "analyze"))
    for name in ("fit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("original-root", "protocol", "vendor", "checkpoint", "data-root", "shards"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--task", choices=tuple(COUNTS))
    parser.add_argument("--precision", choices=PRECISIONS)
    parser.add_argument("--shard-index", type=int)
    args = parser.parse_args()
    required = {"freeze": ("original_root", "task"),
                "evaluate": ("protocol", "vendor", "checkpoint", "data_root", "precision", "shard_index"),
                "analyze": ("protocol", "precision", "shards")}[args.mode]
    if any(getattr(args, name) is None for name in required):
        parser.error("Missing inputs for " + args.mode + ": " + ", ".join(required))
    {"freeze": freeze, "evaluate": evaluate, "analyze": summarize}[args.mode](args)


if __name__ == "__main__":
    main()
