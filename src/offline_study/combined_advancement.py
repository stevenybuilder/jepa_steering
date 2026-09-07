"""Verify durable combined evidence and apply the existing non-routed rank rule.

This selects no new method or dose and grants no protected-evaluation access.
Conditional temporal routing and the final confirmation contract remain separate.
"""
import argparse
import json
import math
from pathlib import Path

from .author_evaluate import check_coverage, verify_fit
from .author_runtime import examples, validate_cohort
from .combined_contract import validate_contract
from .combined_operator import PAIRS, RECIPE_ARMS
from .planning_native_smoke import CHECKPOINTS
from .protocol import sha256, write_json


def decide_combination(report, margin):
    if report["precision"] != "bfloat16" or not math.isfinite(margin) or margin <= 0:
        raise ValueError("Selection requires primary BF16 and the existing positive fit margin")
    family = {(a, b, e) for a, b in PAIRS for e in ("visual_mse_h6", "proprio_mse_h6")}
    rows = report["contrasts"]
    observed = [(r["candidate"], r["control"], r["endpoint"]) for r in rows]
    if len(observed) != len(family) or set(observed) != family:
        raise ValueError("Missing or duplicate members of the frozen simultaneous family")
    contrasts = {}
    for row in rows:
        if row["endpoint"] != "proprio_mse_h6":
            continue
        low, high = row["simultaneous_95_difference_interval"]
        if not all(math.isfinite(x) for x in (low, high)) or low > high:
            raise ValueError("Invalid simultaneous interval")
        if row["minimum_useful_error_reduction"] != margin:
            raise ValueError("The frozen fit-only margin changed")
        if row["beats_control_by_frozen_minimum"] != (high < -margin):
            raise ValueError("Stored eligibility disagrees with its interval")
        contrasts[row["candidate"], row["control"]] = row
    if set(report["arms"]) != {r[0] for r in RECIPE_ARMS}:
        raise ValueError("Incomplete fixed arm registry")
    if any(a["realized_energy_match_fraction"] != 1.0 for a in report["arms"].values()):
        raise ValueError("Do not select from unverified delivered-energy comparisons")
    required = ("native", "matched_random_combined", "coupling_only", "rank4_only")
    gates = {control: contrasts["combined", control]["beats_control_by_frozen_minimum"] for control in required}
    rank_comparison = contrasts["combined", "combined_rank1"]
    low, high = rank_comparison["simultaneous_95_difference_interval"]
    rank1_equivalent = -margin < low <= high < margin
    rank1_eligible = all(contrasts["combined_rank1", control]["beats_control_by_frozen_minimum"]
                         for control in ("native", "matched_random_combined_rank1"))
    selected = None
    if all(gates.values()):
        selected = "combined_rank1" if rank1_eligible and rank1_equivalent else "combined"
    return {"selected_nonrouted_recipe": selected, "combined_advancement_gates": gates,
        "rank1_eligible": rank1_eligible, "rank1_equivalence_established": rank1_equivalent,
        "rank4_beats_rank1_by_frozen_minimum": rank_comparison["beats_control_by_frozen_minimum"],
        "rank_capacity_comparison": rank_comparison,
        "rank_selection_basis": "retain the originally selected rank4 unless eligible rank1 establishes equivalence under the frozen primary-precision margin",
        "native_comparison": contrasts["combined", "native"],
        "confirmation_or_planning_launch_authorized": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--durable-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    inputs = {}
    def locate(source):
        relative = Path(source).relative_to("/workspace/jepa-runtime")
        if relative.parts[0] == "author-correction-20260907":
            relative = Path(*relative.parts[1:])
        path = args.durable_root / relative
        if not path.resolve().is_relative_to(args.durable_root.resolve()):
            raise ValueError("Source binding escapes the durable root")
        return path
    def checked(path, expected=None):
        actual = sha256(path)
        if expected is not None and actual != expected:
            raise ValueError("Durable checksum mismatch: " + str(path))
        inputs[str(path)] = actual
        return json.loads(path.read_text()) if path.suffix == ".json" else None
    root = args.durable_root / "combined-reach-author-20260907"
    frozen = checked(root / "freeze-v1/FROZEN.json")
    contract = checked(root / "freeze-v1/protocol.json", frozen["protocol_sha256"])
    validate_contract(contract)
    source = checked(locate(contract["source_cohort_path"]), contract["source_cohort_sha256"])
    extension = checked(locate(contract["extension_cohort_path"]), contract["extension_cohort_sha256"])
    validate_cohort(source)
    validate_cohort(extension)
    population = source["evaluation"] + extension["evaluation"]
    if len(population) != 33 or len({r["lineage_group"] for r in population}) != 33:
        raise ValueError("Wrong complete primary population")
    expected = examples(population, source["reference_config"], role="combined_development")
    parsimony_root = args.durable_root / "primary-parsimony-audit-20260907-v2"
    parsimony_done = checked(parsimony_root / "DONE.json")
    parsimony = checked(parsimony_root / "report.json", parsimony_done["report_sha256"])
    if parsimony_done["report_sha256"] != contract["parsimony_report_sha256"]:
        raise ValueError("Source non-routed decisions changed")
    for path, checksum in parsimony["input_sha256"].items():
        checked(locate(path), checksum)
    analyses, margins = {}, {}
    for precision in contract["precisions"]:
        for category, binding in contract["components"][precision].items():
            directory = locate(binding["path"])
            receipt, protocol = verify_fit(directory, contract["source_cohort_sha256"], CHECKPOINTS["metaworld"], precision)
            checked(directory / "protocol.json", binding["protocol_sha256"])
            checked(directory / "operator_bank.pt", binding["bank_sha256"])
            if category == "operator_rank":
                margins[precision] = .01 * receipt["fit_native_proprio_mse_h6"]
        directory = root / "analysis-v1" / precision
        done = checked(directory / "DONE.json")
        report = checked(directory / "report.json", done["report_sha256"])
        if (report["status"] != "verified_full_reach_combination_development_complete" or
                report["precision"] != precision or report["protocol_sha256"] != frozen["protocol_sha256"] or
                report["rollout_trajectories"] != 33 or report["independent_lineage_groups"] != 33 or
                report["fresh_confirmation"] or report["newly_opened_untouched_rows"] != 0 or
                report["bootstrap_seed"] != 2026090704 or report["bootstrap_replicates"] != 10000):
            raise ValueError("Incompatible completed combined analysis")
        for path, checksum in report["input_sha256"].items():
            checked(locate(path), checksum)
        measured, diagnostics, selections = [], [], []
        shards = sorted((root / "evaluation-v1" / precision).glob("shard-*"))
        if [p.name for p in shards] != [f"shard-{i:03d}" for i in range(4)]:
            raise ValueError("Missing or extra combined shard")
        for index, shard in enumerate(shards):
            done = checked(shard / "DONE.json")
            values = {name: checked(shard / (name + ".json"), done[name + "_sha256"])
                      for name in ("report", "window_metrics", "diagnostics", "selection")}
            state = values["report"]
            if (state["shard_index"] != index or state["precision"] != precision or
                    state["protocol_sha256"] != frozen["protocol_sha256"] or
                    not all(state[key] for key in ("frozen_weights_unchanged", "native_fidelity", "zero_dose_identity"))):
                raise ValueError("Shard fidelity or protocol mismatch")
            measured.extend(values["window_metrics"])
            diagnostics.extend((r["trajectory_id"], r["start"]) for r in values["diagnostics"])
            selections.extend(r["trajectory_id"] for r in values["selection"])
        check_coverage(measured, expected, [r[0] for r in RECIPE_ARMS])
        if (len(selections) != 33 or len(set(selections)) != 33 or
                set(selections) != {r["trajectory_id"] for r in population} or
                len(diagnostics) != len(set(diagnostics)) or
                set(diagnostics) != {(r["trajectory_id"], r["start"]) for r in expected}):
            raise ValueError("Incomplete or duplicated population/diagnostics")
        analyses[precision] = report
    decision = decide_combination(analyses["bfloat16"], margins["bfloat16"])
    previous = parsimony["decisions"]
    if (previous["reach-wall"]["operator_rank"]["selected_arm"] != "rank4" or
            any(row["selected_arm"] != "native" for row in previous["pusht"].values())):
        raise ValueError("Other primary task decisions changed")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "report.json", {
        "status": "complete_combined_development_verified_nonrouted_selection_audited",
        "input_sha256": inputs, "verified_artifact_hashes": len(inputs),
        "primary_precision": "bfloat16", "selection_between_precisions": False,
        "decisions": {"reach": decision, "reach-wall": {"selected_nonrouted_recipe": "rank4"},
                      "pusht": {"selected_nonrouted_recipe": "native"}},
        "fp32_sensitivity_combined_native_comparison": next(r for r in analyses["float32"]["contrasts"]
            if (r["candidate"], r["control"], r["endpoint"]) == ("combined", "native", "proprio_mse_h6")),
        "full_reach_population": 33, "prefix_rollouts_per_arm_per_precision": len(expected),
        "new_methods_doses_or_outcomes": False, "fresh_confirmation": False,
        "conditional_temporal_routing_eligibility": "unresolved_not_silently_omitted",
        "confirmation_protocol_frozen": False, "protected_outcomes_opened": False,
        "confirmation_or_planning_launch_authorized": False, "full_study_complete": False})
    write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})


if __name__ == "__main__":
    main()
