"""Apply the existing offline advancement gates without optimizing new methods.

BF16 remains the primary offline analysis; FP32 is reported separately. This receipt
does not substitute for compatibility/drop-one work or a frozen confirmation design.
"""
import argparse
import json
from pathlib import Path

from .author_summary import CATEGORIES, COUNTS, verified_analysis
from .protocol import sha256, write_json


def required_controls(protocol):
    category = protocol["category"]
    if category == "vision_action_coupling":
        # No scope-matched random single-pathway arm exists in this fixed registry.
        # Do not promote visual/action-only using the joint random as a substitute.
        return {"joint": ["native", "matched_random", "permuted_joint"],
                "joint_equal_standardized_energy": ["native", "matched_random_equal_standardized_energy"]}
    if category == "action_response_geometry":
        return {name: ["native", "matched_random"] for name in
                ("equal_anchor_linear", "cubic", "projected_cubic", "reflected_curvature")}
    return protocol["development_analysis_plan"]["mechanism_controls"]


def qualify(report, protocol):
    if report["category"] != protocol["category"]:
        raise ValueError("Wrong protocol category")
    endpoint = protocol["development_analysis_plan"]["primary_forecast_endpoint"]
    if endpoint != "proprio_mse_h6":
        raise ValueError("Refusing endpoint selection")
    threshold = protocol["development_analysis_plan"]["smallest_useful_effect"]
    contrasts = {(r["candidate"], r["control"]): r for r in report["contrasts"] if r["endpoint"] == endpoint}
    decisions = {}
    for candidate, controls in required_controls(protocol).items():
        reasons = []
        for control in controls:
            row = contrasts.get((candidate, control))
            if row is None:
                reasons.append("missing_registered_contrast:" + control)
            elif row["simultaneous_95_difference_interval"][1] >= -threshold:
                reasons.append("minimum_effect_not_established:" + control)
        relevant = [candidate, *[c for c in controls if c != "native"]]
        energy = all(report["arms"][arm]["realized_energy_match_fraction"] == 1. for arm in relevant)
        decisions[candidate] = {"effect_and_control_intervals_pass": not reasons,
            "reasons": reasons, "all_delivered_energies_within_frozen_fp32_tolerance": energy,
            "exact_equal_energy_mechanism_claim_supported": energy,
            "native_error_reduction_percent": report["arms"][candidate]["error_reduction_percent_vs_native"][endpoint]}
    eligible = [name for name, row in decisions.items() if row["effect_and_control_intervals_pass"]]
    return {"endpoint": endpoint, "minimum_useful_effect": threshold, "arms": decisions,
            "statistically_eligible_arms": eligible,
            "decision": "retain_native" if not eligible else "eligible_pending_energy_parsimony_and_recipe_verification",
            "confirmation_launch_authorized_by_this_receipt": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("metaworld-extension-root", "pusht-root", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    decisions, inputs, checked = {}, {}, {}
    for task in COUNTS:
        root = args.pusht_root if task == "pusht" else args.metaworld_extension_root
        analyses = "analysis-v1" if task == "pusht" else "analysis-full-v1"
        decisions[task] = {}
        for category in CATEGORIES:
            for precision in ("bfloat16", "float32"):
                directory = root / analyses / precision / task / category
                report, binding = verified_analysis(directory, task, precision, category, checked)
                path = root / "fits-v1" / precision / task / category / "protocol.json"
                if report["protocol_sha256"] != sha256(path):
                    raise ValueError("Analysis no longer binds the frozen protocol")
                inputs[str(path)] = sha256(path)
                inputs[binding["path"]] = binding["sha256"]
                decisions[task].setdefault(precision, {})[category] = qualify(report, json.loads(path.read_text()))
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "report.json", {"status": "fixed_offline_advancement_gates_evaluated",
        "primary_offline_precision": "bfloat16", "sensitivity_precision": "float32",
        "precision_or_endpoint_selection": False, "decisions": decisions, "input_sha256": inputs,
        "fresh_confirmation": False, "confirmation_jobs_launched": False,
        "next": "Verify parsimony, energy and any required combined-recipe/drop-one checks; freeze planning cohort and analysis before reveal."})
    write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})


if __name__ == "__main__":
    main()
