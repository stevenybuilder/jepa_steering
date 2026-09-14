"""Cohort-bound one-shot Push-T replication, preserving the original protection.

This is not a generic holdout override. It requires the complete original manifest,
registry, fit-only cohort, and all ten already-frozen task/precision protocols.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch

from offline_study.core.protocol import sha256, write_json


CATEGORIES = ("vision_action_coupling", "action_response_geometry", "operator_rank",
              "distribution_layer", "distribution_spatial")
PRECISIONS = ("bfloat16", "float32")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_access(cohort):
    access = cohort.get("replication_access")
    if not access:
        return False
    if cohort.get("dataset") == "metaworld":
        from offline_study.evaluation.checkpoint.author_extension import validate_extension_access
        return validate_extension_access(cohort)
    if cohort["dataset"] != "pusht" or cohort["task"] != "pusht" or cohort.get("measurement_role") != "author_replication":
        raise ValueError("Authorization is restricted to the released Push-T replication")
    if digest(access) != cohort.get("access_authorization_sha256"):
        raise ValueError("Replication authorization digest mismatch")
    if access.get("purpose") != "one_shot_released_author_validation_replication" or access.get("authority") != "user_20260907":
        raise ValueError("Missing explicit replication authority")
    if access.get("retune_from_outcomes") is not False or access.get("fresh_confirmation") is not False:
        raise ValueError("Replication cannot authorize tuning or claim fresh confirmation")
    if access["manifest_sha256"] != cohort["source_manifest_sha256"] or access["registry_sha256"] != cohort["source_registry_sha256"]:
        raise ValueError("Replication source binding mismatch")
    if digest(cohort["evaluation"]) != access["evaluation_rows_sha256"] or digest(cohort["fit"]) != access["fit_rows_sha256"]:
        raise ValueError("Frozen replication membership changed")
    evaluation = cohort["evaluation"]
    if len(evaluation) != 21 or {row["index"] for row in evaluation} != set(range(21)) or any(row["source_pool"] != "val" for row in evaluation):
        raise ValueError("Authorization covers exactly the 21 released validation rows")
    if any(row["lineage_group"] not in cohort["all_protected_groups"] for row in evaluation):
        raise ValueError("Historical protected status must remain preserved")
    expected = {f"{precision}/{category}" for precision in PRECISIONS for category in CATEGORIES}
    if set(access["frozen_protocols"]) != expected:
        raise ValueError("All fixed precision/category protocols must be frozen before reveal")
    return True


def verify_runtime_access(cohort_path, cohort):
    if not validate_access(cohort):
        return False
    if cohort.get("dataset") == "metaworld":
        from offline_study.evaluation.checkpoint.author_extension import verify_extension_access
        return verify_extension_access(cohort_path, cohort)
    directory = cohort_path.parent
    access = cohort["replication_access"]
    authorization = json.loads((directory / "access_authorization.json").read_text())
    if authorization != access:
        raise ValueError("Authorization sidecar differs from bound cohort")
    for filename, key in (("source_manifest.jsonl", "manifest_sha256"), ("source_registry.json", "registry_sha256"),
                          ("source_cohort.json", "original_cohort_sha256")):
        if sha256(directory / filename) != access[key]:
            raise ValueError("Original exposure evidence changed: " + filename)
    registry = json.loads((directory / "source_registry.json").read_text())
    if registry["manifest_sha256"] != access["manifest_sha256"]:
        raise ValueError("Original registry does not bind the manifest")
    rows = {row["trajectory_id"]: row for row in map(json.loads, (directory / "source_manifest.jsonl").read_text().splitlines())}
    original = json.loads((directory / "source_cohort.json").read_text())
    if original["fit"] != cohort["fit"] or original["evaluation"] or original["all_protected_groups"] != cohort["all_protected_groups"]:
        raise ValueError("Fit-only cohort or protection history was rewritten")
    for row in cohort["fit"] + cohort["evaluation"]:
        if rows.get(row["trajectory_id"]) != row:
            raise ValueError("Authorized row is not the original manifest record")
    if any(registry["trajectories"][row["trajectory_id"]]["use"] != "protected" for row in cohort["evaluation"]):
        raise ValueError("Protected registry was relabelled")
    return True


def main():
    from offline_study.evaluation.checkpoint.author_evaluate import verify_fit
    from offline_study.evaluation.checkpoint.author_runtime import examples, validate_cohort
    from offline_study.interventions.interventions import validate_frozen_protocol, window_key
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("cohort", "manifest", "registry", "fits", "output"):
        parser.add_argument("--" + flag, required=True, type=Path)
    args = parser.parse_args()
    original = json.loads(args.cohort.read_text())
    validate_cohort(original)
    registry = json.loads(args.registry.read_text())
    if original["task"] != "pusht" or original["evaluation"] or registry["manifest_sha256"] != sha256(args.manifest):
        raise ValueError("Expected the verified Push-T fit-only inputs")
    if original["source_manifest_sha256"] != sha256(args.manifest) or original["source_registry_sha256"] != sha256(args.registry):
        raise ValueError("Original cohort source hashes differ")
    evaluation = original["protected_official_validation"]
    protocols = {}
    for precision in PRECISIONS:
        for category in CATEGORIES:
            directory = args.fits / precision / "pusht" / category
            receipt = json.loads((directory / "fit_receipt.json").read_text())
            verify_fit(directory, sha256(args.cohort), receipt["checkpoint_sha256"], precision)
            protocols[f"{precision}/{category}"] = {"protocol_sha256": sha256(directory / "protocol.json"),
                "fit_receipt_sha256": sha256(directory / "fit_receipt.json"), "bank_sha256": sha256(directory / "operator_bank.pt")}
    access = {"purpose": "one_shot_released_author_validation_replication", "authority": "user_20260907",
        "approval_evidence": "User explicitly authorized corrected reruns and agreed that running the released Push-T validation set is the closest paper-methodology replication.",
        "frozen_at": datetime.now(timezone.utc).isoformat(), "retune_from_outcomes": False, "fresh_confirmation": False,
        "manifest_sha256": sha256(args.manifest), "registry_sha256": sha256(args.registry),
        "original_cohort_sha256": sha256(args.cohort), "evaluation_rows_sha256": digest(evaluation),
        "fit_rows_sha256": digest(original["fit"]), "frozen_protocols": protocols}
    cohort = {**original, "evaluation": evaluation, "protected_official_validation": [],
        "historically_protected_validation": evaluation,
        "coverage": {"evaluated_rows": 21, "official_rows": 21, "complete_author_split": True, "authorized_protected_rows": 21},
        "role": "exposed_protected_author_split_one_shot_replication", "measurement_role": "author_replication",
        "evaluation_permission": "cohort_bound_one_shot_replication", "replication_access": access,
        "access_authorization_sha256": digest(access),
        "access_review": "Explicit user-approved one-shot replication; original registry and protection retained. No retuning and no fresh-confirmation claim."}
    validate_cohort(cohort)
    args.output.mkdir(parents=True, exist_ok=False)
    cohort_dir = args.output / "cohorts/pusht"
    cohort_dir.mkdir(parents=True)
    write_json(cohort_dir / "cohort.json", cohort)
    write_json(cohort_dir / "access_authorization.json", access)
    for source, name in ((args.manifest, "source_manifest.jsonl"), (args.registry, "source_registry.json"), (args.cohort, "source_cohort.json")):
        shutil.copy2(source, cohort_dir / name)
    verify_runtime_access(cohort_dir / "cohort.json", cohort)
    metadata = examples(evaluation, cohort["reference_config"], role="author_replication")
    for precision in PRECISIONS:
        parent = args.output / "fits-v1" / precision / "pusht"
        parent.mkdir(parents=True)
        shutil.copy2(args.fits / precision / "pusht/PARITY.json", parent / "PARITY.json")
        for category in CATEGORIES:
            source = args.fits / precision / "pusht" / category
            dest = parent / category
            dest.mkdir()
            old_receipt = json.loads((source / "fit_receipt.json").read_text())
            receipt = {**old_receipt, "cohort_sha256": sha256(cohort_dir / "cohort.json"), "coverage": cohort["coverage"],
                "access_authorization_sha256": digest(access), "fitting_repeated": False,
                "rebind_reason": "Access/membership binding only; all fitted tensors, arms, doses and analyses unchanged",
                "source_frozen_artifacts": protocols[f"{precision}/{category}"]}
            write_json(dest / "fit_receipt.json", receipt)
            protocol = json.loads((source / "protocol.json").read_text())
            protocol["fit_receipt_sha256"] = sha256(dest / "fit_receipt.json")
            protocol["evaluation_split"] = "author_replication"
            protocol["frozen_at"] = access["frozen_at"]
            protocol["author_validation"].update(cohort_sha256=sha256(cohort_dir / "cohort.json"), coverage=cohort["coverage"],
                protected_rows_excluded=False, access_authorization_sha256=digest(access), retune_from_outcomes=False,
                fresh_confirmation=False, source_protocol_sha256=protocols[f"{precision}/{category}"]["protocol_sha256"])
            validate_frozen_protocol(protocol)
            write_json(dest / "protocol.json", protocol)
            bank = torch.load(source / "operator_bank.pt", map_location="cpu", weights_only=True)
            bank.update(protocol_sha256=sha256(dest / "protocol.json"), evaluation_role="author_replication",
                access_authorization_sha256=digest(access), rows={window_key(m): {**m, "tensors": {}} for m in metadata})
            # Existing fitted tensors are retained verbatim, with no fit/selection call.
            torch.save(bank, dest / "operator_bank.pt")
            shutil.copy2(source / "protocol.json", dest / "source_protocol.json")
            shutil.copy2(source / "fit_receipt.json", dest / "source_fit_receipt.json")
            write_json(dest / "DONE.json", {"status": "user_authorized_replication_protocol_frozen",
                "protocol_sha256": sha256(dest / "protocol.json"), "fit_receipt_sha256": sha256(dest / "fit_receipt.json"),
                "operator_bank_sha256": sha256(dest / "operator_bank.pt")})
        write_json(parent / "DONE.json", {"status": "all_five_replication_protocols_bound_without_refit", "access_authorization_sha256": digest(access)})
    write_json(args.output / "ACCESS_READY.json", {"cohort_sha256": sha256(cohort_dir / "cohort.json"),
        "access_authorization_sha256": digest(access), "protected_outcomes_accessed": False})


if __name__ == "__main__":
    main()
