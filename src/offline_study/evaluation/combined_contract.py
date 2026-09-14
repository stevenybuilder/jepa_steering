"""Freeze the already-planned Reach coupling/rank combination before its outcomes.

No additional fit, dose, site search, newly untouched-row access or confirmation
is authorized here. The full previously measured 33-row Reach pool is reused.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from offline_study.evaluation.checkpoint.author_evaluate import verify_fit
from offline_study.evaluation.checkpoint.author_access import digest, verify_runtime_access
from offline_study.evaluation.checkpoint.author_runtime import validate_cohort
from offline_study.evaluation.checkpoint.author_summary import CATEGORIES
from offline_study.interventions.combined_operator import PAIRS, RECIPE_ARMS
from offline_study.planning.planning_native_smoke import CHECKPOINTS
from offline_study.core.protocol import sha256, write_json


def validate_contract(contract):
    if (contract.get("schema_version") != 1 or contract.get("status") != "frozen" or
            contract.get("category") != "combined_development" or contract.get("task") != "mw-reach" or
            contract.get("evaluation_role") != "previously_exposed_full_primary_development" or contract.get("fresh_confirmation") is not False):
        raise ValueError("Invalid narrowly scoped combined-development contract")
    if contract["recipe_arms"] != [list(row) for row in RECIPE_ARMS]:
        raise ValueError("Combined arm registry changed")
    if contract["primary_contrasts"] != [{"name": a + "_vs_" + b, "candidate": a, "control": b} for a, b in PAIRS]:
        raise ValueError("Combined contrast registry changed")
    if contract["selected_choices"] != {"coupling": "joint_equal_standardized_energy", "rank": "rank4",
            "layer_change": None, "spatial_change": None, "reference_block": 3, "reference_patches": 256}:
        raise ValueError("Combined choices no longer follow the frozen source design")
    if contract["shard_count"] != 4 or contract["precisions"] != ["bfloat16", "float32"]:
        raise ValueError("Changed cohort sharding or precision registry")
    if contract["primary_precision"] != "bfloat16" or contract["selection_between_precisions"]:
        raise ValueError("Precision selection is not allowed")
    if contract["trajectory_count"] != 33 or contract["newly_opened_untouched_rows"] != 0:
        raise ValueError("Do not narrow the full primary pool or open another reserve")


def already_exposed_cohort(contract):
    base_path = Path(contract["source_cohort_path"])
    extension_path = Path(contract["extension_cohort_path"])
    for path, expected in ((base_path, contract["source_cohort_sha256"]),
                           (extension_path, contract["extension_cohort_sha256"])):
        if sha256(path) != expected:
            raise ValueError("A source exposure cohort changed")
    base, extension = json.loads(base_path.read_text()), json.loads(extension_path.read_text())
    validate_cohort(base)
    validate_cohort(extension)
    if not verify_runtime_access(extension_path, extension):
        raise ValueError("The four added Reach rows lack their original authorization")
    if (base["task"] != "mw-reach" or base["fit"] != extension["fit"] or
            base["evaluation"] != extension["reused_evaluation"] or
            base["protected_official_validation"] != extension["evaluation"]):
        raise ValueError("Combined stage rewrites fitting or historical exposure")
    evaluation = base["evaluation"] + extension["evaluation"]
    if len(evaluation) != 33 or len({r["lineage_group"] for r in evaluation}) != 33:
        raise ValueError("Wrong full primary Reach population")
    if digest(evaluation) != contract["evaluation_rows_sha256"]:
        raise ValueError("Frozen combined population changed")
    scopes = set()
    for binding in contract["prior_full_pool_analyses"]:
        path = Path(binding["path"])
        if sha256(path) != binding["sha256"]:
            raise ValueError("Prior full-pool exposure evidence changed")
        prior = json.loads(path.read_text())
        done = json.loads((path.parent / "DONE.json").read_text())
        if (done["report_sha256"] != binding["sha256"] or prior["rollout_trajectories"] != 33 or
                prior["task"] != "mw-reach" or prior["fresh_confirmation"] or
                prior["extension_cohort_sha256"] != contract["extension_cohort_sha256"] or
                prior["status"] != "verified_full_primary_author_split_replication_complete"):
            raise ValueError("No verified prior full-pool exposure")
        scopes.add((prior["category"], prior["precision"]))
    if len(contract["prior_full_pool_analyses"]) != 10 or scopes != {(c, p) for c in CATEGORIES for p in ("bfloat16", "float32")}:
        raise ValueError("The whole frozen prior sweep must finish before this stage")
    # Distinct schema: this does NOT masquerade as an unprotected author cohort
    # or weaken validate_cohort/verify_runtime_access for any other evaluator.
    return {"schema_kind": "verified_previously_exposed_combination_cohort", "task": base["task"],
        "dataset": base["dataset"], "fit": base["fit"], "evaluation": evaluation,
        "reference_config": base["reference_config"],
        "coverage": {"evaluated_rows": 33, "official_rows": 33, "complete_author_split": True},
        "historical_protection_retained": True, "fresh_confirmation": False}


def load_components(contract, precision):
    validate_contract(contract)
    if precision not in contract["precisions"]:
        raise ValueError("Unregistered precision")
    cohort = already_exposed_cohort(contract)
    result = {}
    for category in ("vision_action_coupling", "operator_rank"):
        binding = contract["components"][precision][category]
        directory = Path(binding["path"])
        receipt, protocol = verify_fit(directory, contract["source_cohort_sha256"], CHECKPOINTS["metaworld"], precision)
        for filename, key in (("protocol.json", "protocol_sha256"), ("operator_bank.pt", "bank_sha256")):
            if sha256(directory / filename) != binding[key]:
                raise ValueError("A combined source component changed")
        result[category] = (directory, receipt, protocol)
    return cohort, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original-root", "extension-root", "parsimony", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    done = json.loads((args.parsimony / "DONE.json").read_text())
    if sha256(args.parsimony / "report.json") != done["report_sha256"]:
        raise ValueError("Parsimony receipt changed")
    audit = json.loads((args.parsimony / "report.json").read_text())
    for source, expected in audit["input_sha256"].items():
        if sha256(Path(source)) != expected:
            raise ValueError("Parsimony input changed")
    choices = audit["decisions"]["reach"]
    if (choices["vision_action_coupling"]["selected_arm"] != "joint_equal_standardized_energy" or
            choices["operator_rank"]["selected_arm"] != "rank4" or
            choices["distribution_layer"]["selected_arm"] is not None or
            choices["distribution_spatial"]["selected_arm"] != "all_patches" or
            choices["action_response_geometry"]["selected_arm"] != "native"):
        raise ValueError("Selection differs from this fixed compatible recipe")
    cohort = args.original_root / "cohorts/reach/cohort.json"
    extension = args.extension_root / "cohorts/reach/cohort.json"
    original_rows = json.loads(cohort.read_text())["evaluation"]
    added_rows = json.loads(extension.read_text())["evaluation"]
    prior = []
    for precision in ("bfloat16", "float32"):
        for category in CATEGORIES:
            path = args.extension_root / "analysis-full-v1" / precision / "reach" / category / "report.json"
            prior.append({"path": str(path), "sha256": sha256(path)})
    components = {}
    for precision in ("bfloat16", "float32"):
        components[precision] = {}
        for category in ("vision_action_coupling", "operator_rank"):
            path = args.original_root / "fits-v1" / precision / "reach" / category
            _, protocol = verify_fit(path, sha256(cohort), CHECKPOINTS["metaworld"], precision)
            components[precision][category] = {"path": str(path), "protocol_sha256": sha256(path / "protocol.json"),
                                               "bank_sha256": sha256(path / "operator_bank.pt")}
    contract = {"schema_version": 1, "status": "frozen", "category": "combined_development", "task": "mw-reach",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "source_cohort_path": str(cohort), "source_cohort_sha256": sha256(cohort),
        "extension_cohort_path": str(extension), "extension_cohort_sha256": sha256(extension),
        "evaluation_rows_sha256": digest(original_rows + added_rows), "prior_full_pool_analyses": prior,
        "parsimony_report_sha256": done["report_sha256"], "components": components,
        "hypothesis": "The planned additive equal-budget coupling plus rank-4 correction retains useful H6 forecast improvement beyond native and matched controls; drop-one comparisons measure dependence on each component and rank choice.",
        "recipe_arms": [list(row) for row in RECIPE_ARMS], "arms": [{"name": r[0]} for r in RECIPE_ARMS],
        "primary_contrasts": [{"name": a + "_vs_" + b, "candidate": a, "control": b} for a, b in PAIRS],
        "selected_choices": {"coupling": "joint_equal_standardized_energy", "rank": "rank4", "layer_change": None,
                             "spatial_change": None, "reference_block": 3, "reference_patches": 256},
        "unresolved_layer_policy": "preserve registered category reference; do not call B3 best or claim localization",
        "layer_position_2x2": "not eligible: no non-global spatial arm passes the frozen gates; do not invent a selected non-global support",
        "drop_one_definitions": {"remove_coupling": "rank4_only", "remove_rank_operator": "coupling_only",
                                 "remove_added_rank_capacity": "combined_rank1"},
        "drop_one_energy_interpretation": "Removing a component also removes its fixed budget; do not call those contrasts equal-total-energy mechanism tests. Combined rank1/rank4 and corresponding random controls retain matched total component energy.",
        "composition": "add the unchanged source edits at distinct hooks, with support responses and target computed on native unedited shadow; no combined dose renormalization",
        "matched_controls": "combine the corresponding scope/rank/spectrum/energy-matched source random edits",
        "source_full_rank_probe_and_common_degeneracy_registry_retained": True,
        "precisions": ["bfloat16", "float32"], "primary_precision": "bfloat16", "selection_between_precisions": False,
        "evaluation_role": "previously_exposed_full_primary_development", "trajectory_count": 33, "shard_count": 4,
        "authority": "user_20260907_autonomous_predeclared_post_selection_advancement",
        "access_scope": "only the exact previously measured full Reach pool; old registry and one-factor protocols unchanged",
        "newly_opened_untouched_rows": 0,
        "full_primary_offline_replication_already_complete": True, "complete_author_pool_claim_for_this_secondary_stage": True,
        "analysis": {"primary_endpoint": "proprio_mse_h6", "interval_family": "all registered contrasts x visual/proprio H6 MSE",
                     "bootstrap_seed": 2026090704, "bootstrap_replicates": 10000,
                     "minimum_and_equivalence_margin": "same precision-specific 1% fit-only native H6 error as source protocols"},
        "candidate_controls": {"combined": ["native", "matched_random_combined"],
            "combined_rank1": ["native", "matched_random_combined_rank1"],
            "coupling_only": ["native", "matched_random_coupling"],
            "rank4_only": ["native", "matched_random_rank4"]},
        "fresh_confirmation": False, "historically_protected_rows_reused": 4, "new_fit_or_dose": False,
        "launch_requires": "this contract plus a matching fit-only combination parity receipt"}
    load_components(contract, "bfloat16")
    load_components(contract, "float32")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "protocol.json", contract)
    write_json(args.output / "FROZEN.json", {"protocol_sha256": sha256(args.output / "protocol.json"),
                                           "evaluation_launched": False, "fresh_confirmation": False})


if __name__ == "__main__":
    main()
