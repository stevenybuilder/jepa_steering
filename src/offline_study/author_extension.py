"""Explicitly authorized seven-row MetaWorld replication extension; no refitting.

Only these seven source rows are authorized here. Historical manifests and the
original 53-row sweep remain immutable. Full-pool claims require a verified merge.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch

from .author_access import CATEGORIES, PRECISIONS, digest
from .protocol import sha256, write_json


SCOPE = "metaworld_seven_row_replication_extension"
INDICES = {"mw-reach": {10624, 10774, 10782, 10653},
           "mw-reach-wall": {10476, 10201, 10350}}
FULL_COUNTS = {"mw-reach": 33, "mw-reach-wall": 27}


def validate_extension_access(cohort):
    access = cohort["replication_access"]
    task = cohort["task"]
    if task not in INDICES or cohort["dataset"] != "metaworld" or cohort.get("measurement_role") != "author_replication":
        raise ValueError("Extension authorizes only the two primary MetaWorld tasks")
    if access.get("scope") != SCOPE or access.get("authority") != "user_20260907_metaworld_conditional_approval":
        raise ValueError("Missing seven-row user authorization")
    if access.get("purpose") != "one_shot_released_author_validation_replication":
        raise ValueError("Unexpected extension purpose")
    if access.get("retune_from_outcomes") is not False or access.get("fresh_confirmation") is not False:
        raise ValueError("Extension is not tuning permission or confirmation")
    if digest(access) != cohort.get("access_authorization_sha256"):
        raise ValueError("Extension authorization digest mismatch")
    for key in ("manifest", "registry"):
        if access[key + "_sha256"] != cohort["source_" + key + "_sha256"]:
            raise ValueError("Extension source mismatch")
    for role in ("fit", "evaluation", "reused_evaluation"):
        if digest(cohort[role]) != access[role + "_rows_sha256"]:
            raise ValueError("Extension membership changed: " + role)
    rows = cohort["evaluation"]
    if len(rows) != len(INDICES[task]) or {r["index"] for r in rows} != INDICES[task]:
        raise ValueError("Extension is restricted to the exact seven approved rows")
    if any(r["source_pool"] != "all" or r["task"] != task or r["split"] != "holdout" or
           r["trajectory_id"] != f"metaworld:all:{r['index']}" or
           r["lineage_group"] not in cohort["all_protected_groups"] for r in rows):
        raise ValueError("Original protected identities must remain intact")
    if len(cohort["reused_evaluation"]) + len(rows) != FULL_COUNTS[task]:
        raise ValueError("Incomplete primary author pool")
    if {r["trajectory_id"] for r in rows} & {r["trajectory_id"] for r in cohort["reused_evaluation"]}:
        raise ValueError("An extension must not rerun the original rows")
    if set(access["frozen_protocols"]) != {f"{p}/{c}" for p in PRECISIONS for c in CATEGORIES}:
        raise ValueError("All ten scientific protocols must already be frozen")
    return True


def verify_extension_access(cohort_path, cohort):
    validate_extension_access(cohort)
    directory, access = cohort_path.parent, cohort["replication_access"]
    if json.loads((directory / "access_authorization.json").read_text()) != access:
        raise ValueError("Extension authorization sidecar mismatch")
    for filename, key in (("source_manifest.jsonl", "manifest_sha256"), ("source_registry.json", "registry_sha256"),
                          ("source_cohort.json", "original_cohort_sha256")):
        if sha256(directory / filename) != access[key]:
            raise ValueError("Original extension evidence changed: " + filename)
    original = json.loads((directory / "source_cohort.json").read_text())
    if (original["fit"] != cohort["fit"] or original["evaluation"] != cohort["reused_evaluation"] or
            original["protected_official_validation"] != cohort["evaluation"] or
            original["all_protected_groups"] != cohort["all_protected_groups"]):
        raise ValueError("Extension rewrites prior fitting, evaluation or exposure history")
    registry = json.loads((directory / "source_registry.json").read_text())
    if registry["manifest_sha256"] != access["manifest_sha256"]:
        raise ValueError("Registry manifest mismatch")
    rows = {r["trajectory_id"]: r for r in map(json.loads, (directory / "source_manifest.jsonl").read_text().splitlines())}
    for row in cohort["fit"] + cohort["evaluation"] + cohort["reused_evaluation"]:
        if rows.get(row["trajectory_id"]) != row:
            raise ValueError("Extension row differs from original source")
    return True


def bind_task(root, output, task):
    from .author_evaluate import verify_fit
    from .author_runtime import examples, validate_cohort
    from .interventions import validate_frozen_protocol, window_key
    original_path = root / "cohorts" / task / "cohort.json"
    original = json.loads(original_path.read_text())
    validate_cohort(original)
    manifest, registry = root / "inputs/metaworld-manifest.jsonl", root / "inputs/metaworld-registry.json"
    if original["source_manifest_sha256"] != sha256(manifest) or original["source_registry_sha256"] != sha256(registry):
        raise ValueError("Unverified original MetaWorld inputs")
    protocols = {}
    for precision in PRECISIONS:
        for category in CATEGORIES:
            source = root / "fits-v1" / precision / task / category
            receipt = json.loads((source / "fit_receipt.json").read_text())
            verify_fit(source, sha256(original_path), receipt["checkpoint_sha256"], precision)
            protocols[f"{precision}/{category}"] = {"protocol_sha256": sha256(source / "protocol.json"),
                "fit_receipt_sha256": sha256(source / "fit_receipt.json"), "bank_sha256": sha256(source / "operator_bank.pt")}
    evaluation = original["protected_official_validation"]
    access = {"scope": SCOPE, "purpose": "one_shot_released_author_validation_replication",
        "authority": "user_20260907_metaworld_conditional_approval",
        "approval_evidence": "User asked whether including the seven rows follows JEPA-WM and authorized proceeding if so. They belong to the official validation pool; inclusion is author-split replication, not untouched confirmation.",
        "frozen_at": datetime.now(timezone.utc).isoformat(), "retune_from_outcomes": False, "fresh_confirmation": False,
        "manifest_sha256": sha256(manifest), "registry_sha256": sha256(registry),
        "original_cohort_sha256": sha256(original_path), "fit_rows_sha256": digest(original["fit"]),
        "evaluation_rows_sha256": digest(evaluation), "reused_evaluation_rows_sha256": digest(original["evaluation"]),
        "frozen_protocols": protocols}
    cohort = {**original, "evaluation": evaluation, "reused_evaluation": original["evaluation"],
        "protected_official_validation": [], "historically_protected_validation": evaluation,
        "role": SCOPE, "measurement_role": "author_replication", "replication_access": access,
        "access_authorization_sha256": digest(access), "fresh_confirmation": False,
        "evaluation_permission": "exact_seven_rows_user_authorized_no_other_protected_rows",
        "access_review": "Original protection preserved; one fixed replication sweep, no refit. Fresh simulator confirmation remains distinct.",
        "coverage": {"evaluated_rows": len(evaluation), "reused_rows": len(original["evaluation"]),
                     "official_rows": FULL_COUNTS[original["task"]], "complete_author_split": False,
                     "full_author_split_requires_verified_merge": True}}
    validate_cohort(cohort)
    directory = output / "cohorts" / task
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "cohort.json", cohort)
    write_json(directory / "access_authorization.json", access)
    for source, name in ((manifest, "source_manifest.jsonl"), (registry, "source_registry.json"), (original_path, "source_cohort.json")):
        shutil.copy2(source, directory / name)
    verify_extension_access(directory / "cohort.json", cohort)
    metadata = examples(evaluation, cohort["reference_config"], role="author_replication")
    for precision in PRECISIONS:
        parent = output / "fits-v1" / precision / task
        parent.mkdir(parents=True)
        shutil.copy2(root / "fits-v1" / precision / task / "PARITY.json", parent / "PARITY.json")
        for category in CATEGORIES:
            source = root / "fits-v1" / precision / task / category
            dest = parent / category
            dest.mkdir()
            old_receipt = json.loads((source / "fit_receipt.json").read_text())
            receipt = {**old_receipt, "cohort_sha256": sha256(directory / "cohort.json"), "coverage": cohort["coverage"],
                "access_authorization_sha256": digest(access), "fitting_repeated": False,
                "source_frozen_artifacts": protocols[f"{precision}/{category}"]}
            write_json(dest / "fit_receipt.json", receipt)
            protocol = json.loads((source / "protocol.json").read_text())
            protocol.update(fit_receipt_sha256=sha256(dest / "fit_receipt.json"),
                            evaluation_split="author_replication", frozen_at=access["frozen_at"])
            protocol["author_validation"].update(cohort_sha256=sha256(directory / "cohort.json"), coverage=cohort["coverage"],
                protected_rows_excluded=False, access_authorization_sha256=digest(access), access_scope=SCOPE,
                retune_from_outcomes=False, fresh_confirmation=False)
            validate_frozen_protocol(protocol)
            write_json(dest / "protocol.json", protocol)
            bank = torch.load(source / "operator_bank.pt", weights_only=True, map_location="cpu")
            bank.update(protocol_sha256=sha256(dest / "protocol.json"), evaluation_role="author_replication",
                access_authorization_sha256=digest(access), rows={window_key(m): {**m, "tensors": {}} for m in metadata})
            torch.save(bank, dest / "operator_bank.pt")
            shutil.copy2(source / "protocol.json", dest / "source_protocol.json")
            shutil.copy2(source / "fit_receipt.json", dest / "source_fit_receipt.json")
            write_json(dest / "DONE.json", {"status": "authorized_seven_row_extension_frozen",
                "protocol_sha256": sha256(dest / "protocol.json"), "fit_receipt_sha256": sha256(dest / "fit_receipt.json"),
                "operator_bank_sha256": sha256(dest / "operator_bank.pt")})
        write_json(parent / "DONE.json", {"status": "all_five_protocols_rebound_without_refit"})
    return {"cohort_sha256": sha256(directory / "cohort.json"), "access_authorization_sha256": digest(access)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tasks = {task: bind_task(args.root, args.output, task) for task in ("reach", "reach-wall")}
    write_json(args.output / "ACCESS_READY.json", {"tasks": tasks, "protected_outcomes_accessed": False,
                                                 "authorized_rows": 7, "old_rows_rerun": 0})


if __name__ == "__main__":
    main()
