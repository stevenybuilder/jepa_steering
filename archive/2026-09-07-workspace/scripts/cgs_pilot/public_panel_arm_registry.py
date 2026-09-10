#!/usr/bin/env python3
"""Freeze a separate, hash-bound Panel-P intervention-arm registry.

Stimulus manifests are immutable and arm-independent. This command therefore
never edits ``manifest_meta.json``. It consumes a reviewed JSON specification,
verifies the required pre-intervention evidence, hashes every referenced file,
and creates one new registry with exclusive-create semantics.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path


SCHEMA_VERSION = "panel-p-arm-registry-v1"
BASE_EVIDENCE = {"fit_apply", "identity_action_hash", "action_dose", "development_utility"}
HMM_EVIDENCE = {"hmm_deployment", "hmm_outcome_routing", "treatment_separation"}
FRANKENSTEIN_EVIDENCE = {"frankenstein_modules"}
FRANKENSTEIN_MODULES = {
    "probability_geometry", "sparse_superposition", "pattern_separation",
    "model_native_coordinates",
}
NON_OPERATOR_ARMS = {"unsteered", "identity"}
KNOWN_ARMS = NON_OPERATOR_ARMS | {
    "sonar_coast_global",
    "sonar_coast_static",
    "sonar_coast_hmm",
    "sonar_coast_sham",
    "sonar_coast_label_shuffled",
    "sonar_energy",
    "sonar_energy_boundary",
    "sonar_ot",
    "sonar_reverse",
    "sonar_sham",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lookup(document: dict, dotted_key: str):
    value = document
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"missing evidence field {dotted_key!r}")
        value = value[part]
    return value


def validate_arm(arm: dict, artifact_names: set[str]) -> None:
    name = arm.get("name")
    condition_id = arm.get("id", name)
    if not isinstance(condition_id, str) or not condition_id:
        raise ValueError("every arm must have a nonempty string id")
    if name not in KNOWN_ARMS:
        raise ValueError(f"unknown arm {name!r}")
    if name in NON_OPERATOR_ARMS:
        unexpected = set(arm) - {"name", "id"}
        if unexpected:
            raise ValueError(f"{name} must not carry treatment parameters: {sorted(unexpected)}")
        return
    required = {"operator_artifact", "site", "beta", "hook_schedule", "displacement_mode"}
    missing = required - set(arm)
    if missing:
        raise ValueError(f"{name} missing {sorted(missing)}")
    if arm["operator_artifact"] not in artifact_names:
        raise ValueError(f"{name} references unknown operator artifact {arm['operator_artifact']!r}")
    if arm["hook_schedule"] not in {"all_physical_replans", "first_physical_replan"}:
        raise ValueError(f"{name} has an unrecognized hook schedule")
    if name.startswith("sonar_coast_") and arm["displacement_mode"] != "joint":
        raise ValueError(f"{name} must use joint multiplicative conceptor gating")
    if not name.startswith("sonar_coast_") and "trust_radius" not in arm:
        raise ValueError(f"{name} must freeze trust_radius")
    if arm.get("frankenstein_mode", "off") not in {"off", "admitted"}:
        raise ValueError(f"{name} has an invalid frankenstein_mode")
    ablations = arm.get("frankenstein_ablate", [])
    if not isinstance(ablations, list) or not set(ablations).issubset(FRANKENSTEIN_MODULES):
        raise ValueError(f"{name} has invalid Frankenstein ablations")
    if ablations and arm.get("frankenstein_mode", "off") != "admitted":
        raise ValueError(f"{name} cannot ablate Frankenstein modules while the recipe is off")
    native_mode = arm.get("model_native_mode", "rowspace")
    if native_mode not in {"rowspace", "complement"}:
        raise ValueError(f"{name} has an invalid model_native_mode")
    if native_mode == "complement" and arm.get("frankenstein_mode", "off") != "admitted":
        raise ValueError(f"{name} cannot use the model-native complement while the recipe is off")
    attention_site = arm.get("attention_restore_site")
    if attention_site is not None:
        if arm.get("frankenstein_mode", "off") != "admitted":
            raise ValueError(f"{name} cannot restore attention while the recipe is off")
        if not str(attention_site).endswith(".attn_out"):
            raise ValueError(f"{name} has an invalid attention_restore_site")
        strength = arm.get("attention_restore_strength")
        if not isinstance(strength, (int, float)) or not 0.0 <= float(strength) <= 1.0:
            raise ValueError(f"{name} must freeze attention_restore_strength in [0,1]")


def freeze_registry(spec_path: Path, stimulus_dir: Path, out: Path) -> dict:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite frozen registry: {out}")
    spec = json.loads(spec_path.read_text())
    for key in (
        "primary_endpoint", "primary_comparison", "selected_on_split", "arms",
        "artifacts", "evidence", "method_hyperparameters",
    ):
        if key not in spec:
            raise ValueError(f"spec missing {key!r}")
    if spec["selected_on_split"] != "development":
        raise ValueError("arms must be selected on the development split")
    if spec["primary_comparison"] != "selected_utility_arm_vs_unsteered":
        raise ValueError("the primary utility baseline must be unsteered")
    required_hyperparameters = {
        "coordinate_rank", "shrinkage", "covariance_floor", "support_quantile",
        "conceptor_aperture", "conceptor_rcond", "bootstrap_draws",
    }
    missing_hyperparameters = sorted(required_hyperparameters - set(spec["method_hyperparameters"]))
    if missing_hyperparameters:
        raise ValueError(f"method_hyperparameters missing {missing_hyperparameters}")

    stimulus_files = [
        stimulus_dir / "manifest_meta.json",
        stimulus_dir / "fit.jsonl",
        stimulus_dir / "evaluation.jsonl",
    ]
    missing_stimulus = [str(path) for path in stimulus_files if not path.is_file()]
    if missing_stimulus:
        raise ValueError(f"missing stimulus files: {missing_stimulus}")

    artifacts = {}
    for name, raw_path in spec["artifacts"].items():
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"artifact {name!r} does not exist: {path}")
        artifacts[name] = {"path": str(path), "sha256": sha256_file(path)}

    arms = list(spec["arms"])
    names = [arm.get("name") for arm in arms]
    condition_ids = [arm.get("id", arm.get("name")) for arm in arms]
    if len(condition_ids) != len(set(condition_ids)):
        raise ValueError("arm condition ids must be unique")
    if not {"unsteered", "identity"}.issubset(names):
        raise ValueError("unsteered and identity arms are mandatory")
    for arm in arms:
        validate_arm(arm, set(artifacts))

    evidence = {}
    for item in spec["evidence"]:
        name = str(item["name"])
        if name in evidence:
            raise ValueError(f"duplicate evidence name {name!r}")
        path = Path(item["path"]).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"evidence {name!r} does not exist: {path}")
        document = json.loads(path.read_text())
        field = str(item["required_true_field"])
        if lookup(document, field) is not True:
            raise ValueError(f"evidence {name!r} did not pass {field!r}")
        evidence[name] = {"path": str(path), "sha256": sha256_file(path), "passed_field": field}
    required_evidence = set(BASE_EVIDENCE)
    if "sonar_coast_hmm" in names:
        required_evidence |= HMM_EVIDENCE
    if any(arm.get("frankenstein_mode") == "admitted" for arm in arms):
        required_evidence |= FRANKENSTEIN_EVIDENCE
    missing_evidence = sorted(required_evidence - set(evidence))
    if missing_evidence:
        raise ValueError(f"missing mandatory evidence: {missing_evidence}")

    registry = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_spec": {"path": str(spec_path.resolve()), "sha256": sha256_file(spec_path)},
        "stimulus_manifest": {
            path.name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in stimulus_files
        },
        "primary_endpoint": spec["primary_endpoint"],
        "primary_comparison": spec["primary_comparison"],
        "selected_on_split": spec["selected_on_split"],
        "method_hyperparameters": spec["method_hyperparameters"],
        "arm_condition_ids": condition_ids,
        "arms": arms,
        "artifacts": artifacts,
        "evidence": evidence,
        "protected_outcomes_seen": False,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as handle:
        json.dump(registry, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return registry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--stimulus-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    registry = freeze_registry(args.spec, args.stimulus_dir, args.out)
    print(json.dumps({
        "registry": str(args.out.resolve()),
        "schema_version": registry["schema_version"],
        "n_arms": len(registry["arms"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
