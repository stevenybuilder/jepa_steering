"""Audit fixed primary-precision eligibility and registered equivalence evidence.

Operator rank has an explicit total ordering/selection rule. Coupling efficacy
uses the equal-budget arm, per EXPERIMENT_PLAN.md, not the unscaled factorial.
Do not invent a tie breaker for other multi-arm sets or confuse localization with
a point winner.
"""
import argparse
import json
import math
from pathlib import Path

from .author_advancement import qualify
from .author_summary import CATEGORIES, COUNTS, verified_analysis
from .protocol import sha256, write_json


def interval_between(report, candidate, control):
    if candidate == control:
        return [0., 0.]
    values = []
    for row in report["contrasts"]:
        if row["endpoint"] != "proprio_mse_h6":
            continue
        pair = row["candidate"], row["control"]
        if pair == (candidate, control):
            values.append(row["simultaneous_95_difference_interval"])
        elif pair == (control, candidate):
            low, high = row["simultaneous_95_difference_interval"]
            values.append([-high, -low])
    if len(values) > 1:
        raise ValueError("Duplicate registered comparison")
    if not values:
        return None
    low, high = values[0]
    if not all(math.isfinite(x) for x in (low, high)) or low > high:
        raise ValueError("Invalid equivalence interval")
    return [low, high]


def audit(report, protocol):
    if report["precision"] != "bfloat16":
        raise ValueError("Primary-precision selection cannot switch to FP32")
    gates = qualify(report, protocol)
    candidates = [name for name in gates["statistically_eligible_arms"]
                  if gates["arms"][name]["all_delivered_energies_within_frozen_fp32_tolerance"]]
    margin = protocol["development_analysis_plan"]["equivalence_margin"]
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("Equivalence margin must already be frozen and positive")
    result = {"eligible_with_verified_energy": candidates, "fixed_gates": gates,
        "frozen_equivalence_margin": margin, "planning_or_confirmation_authorized": False,
        "selected_arm": None, "unique_combined_recipe_frozen": False}
    if protocol["category"] == "vision_action_coupling":
        # The governing plan explicitly reserves the unscaled factorial for
        # interaction and the equal-budget arm for efficacy. No new tie rule.
        name = "joint_equal_standardized_energy"
        eligible = name in candidates
        return {**result, "decision": "planned_equal_budget_efficacy_arm_selected" if eligible else "retain_native",
            "selected_arm": name if eligible else "native",
            "selection_basis": "EXPERIMENT_PLAN.md: use unscaled factorial for interaction and equal-budget arms for efficacy",
            "unscaled_factorial_is_not_promoted_for_efficacy": True}
    if not candidates:
        return {**result, "decision": "retain_native", "selected_arm": "native"}
    best = max(candidates, key=lambda n: gates["arms"][n]["native_error_reduction_percent"])
    evidence = {}
    for name in candidates:
        interval = interval_between(report, name, best)
        evidence[name] = {"difference_interval_against_observed_best": interval,
            "equivalence_established": interval is not None and -margin < interval[0] <= interval[1] < margin}
    result.update(observed_best_eligible_arm=best, equivalence_to_observed_best=evidence,
                  observed_best_is_not_a_statistical_superiority_claim=True)
    if protocol["category"] == "operator_rank":
        if any(n not in ("rank1", "rank4", "rank8") for n in candidates):
            raise ValueError("Unregistered rank")
        selected = min((n for n in candidates if evidence[n]["equivalence_established"]), key=lambda n: int(n[4:]))
        return {**result, "selected_arm": selected, "decision": "fixed_smallest_equivalent_eligible_rank_selected"}
    if len(candidates) == 1:
        return {**result, "selected_arm": candidates[0], "decision": "single_eligible_choice_pending_combination"}
    return {**result, "decision": "multiple_eligible_choices_no_unique_frozen_tie_rule",
        "next": "Retain all registered equivalence evidence; do not infer equivalence from a missing contrast or silently introduce a tie breaker."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("metaworld-extension-root", "pusht-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    results, inputs, checked = {}, {}, {}
    for task in COUNTS:
        root = args.pusht_root if task == "pusht" else args.metaworld_extension_root
        analyses = "analysis-v1" if task == "pusht" else "analysis-full-v1"
        results[task] = {}
        for category in CATEGORIES:
            report, binding = verified_analysis(root / analyses / "bfloat16" / task / category,
                                                task, "bfloat16", category, checked)
            path = root / "fits-v1/bfloat16" / task / category / "protocol.json"
            if report["protocol_sha256"] != sha256(path):
                raise ValueError("Frozen protocol binding changed")
            inputs[binding["path"]] = binding["sha256"]
            inputs[str(path)] = sha256(path)
            results[task][category] = audit(report, json.loads(path.read_text()))
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "report.json", {"status": "fixed_primary_parsimony_audited",
        "decisions": results, "input_sha256": inputs, "primary_precision": "bfloat16",
        "new_methods_or_doses": False, "confirmation_jobs_launched": False,
        "combined_recipe_and_drop_one_executed": False})
    write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})


if __name__ == "__main__":
    main()
