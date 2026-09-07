"""Freeze whole-RNG-stream repairs from stimulus hashes only, never policy outcomes."""
import argparse
import json
from pathlib import Path

from .behavioral_development import assigned_rows, schedule, validate_coverage, verified_report
from .planning_goal_bank import GoalBank, native_records, verify_delivery
from .protocol import sha256, write_json


def changed_streams(native, candidates):
    reference = {row["episode"]: row for row in native}
    if len(reference) != len(native) or len({row["episode"] for row in candidates}) != len(candidates):
        raise ValueError("Duplicate stimulus identity")
    mismatches = []
    for row in candidates:
        baseline = reference[row["episode"]]
        if (any(row[key] != baseline[key] for key in
                ("logical_rank", "local_seed", "environment_seed", "initial_state_vector")) or
                row["result"]["initial_sha256"] != baseline["result"]["initial_sha256"]):
            raise ValueError("This renderer repair cannot fix changed physical tasks, initial images or seeds")
        if row["result"]["goal_sha256"] != baseline["result"]["goal_sha256"]:
            mismatches.append({"episode": row["episode"], "logical_rank": row["logical_rank"],
                "original_native_goal_sha256": baseline["result"]["goal_sha256"],
                "candidate_goal_sha256": row["result"]["goal_sha256"]})
    return sorted({row["logical_rank"] for row in mismatches}), mismatches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("freeze", "reference-root", "candidate-shard", "goal-bank", "goal-delivery-proof", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--task", choices=("reach", "reach-wall"), required=True)
    parser.add_argument("--arm", choices=("coupling_only", "matched_random_coupling"), required=True)
    args = parser.parse_args()
    frozen_hash = sha256(args.freeze / "protocol.json")
    if json.loads((args.freeze / "FROZEN.json").read_text())["protocol_sha256"] != frozen_hash:
        raise ValueError("Scientific freeze changed")
    originals, native_bindings = native_records(args.reference_root, args.task, frozen_hash)
    bank = GoalBank(args.goal_bank, args.task, frozen_hash)
    delivery_hash = verify_delivery(args.goal_delivery_proof, bank.report_hash, frozen_hash)
    report, report_hash = verified_report(args.candidate_shard)
    launch = json.loads((args.candidate_shard / "protocol.json").read_text())
    expected = assigned_rows(schedule(), launch["logical_ranks"])
    if (report["status"] != "fixed_planning_development_candidate_shard_complete" or
            report["task"] != args.task or report["arm"] != args.arm or
            report["protocol_sha256"] != sha256(args.candidate_shard / "protocol.json") or
            launch["freeze_sha256"] != frozen_hash or report["episodes"] != len(expected) or
            launch["expected_episodes"] != expected or report["parameters_unchanged"] is not True):
        raise ValueError("Require the complete bound original physical shard")
    candidates, inputs = [], {}
    for name, digest in report["episode_files_sha256"].items():
        if Path(name).name != name or sha256(args.candidate_shard / name) != digest:
            raise ValueError("Original candidate evidence changed")
        row = json.loads((args.candidate_shard / name).read_text())
        if row["arm"] != args.arm:
            raise ValueError("Original treatment label changed")
        trace = args.candidate_shard / f"calls-{row['episode']:03d}" / "unroll_calls.json"
        if sha256(trace) != row["unroll_calls_sha256"]:
            raise ValueError("Original call trace changed")
        inputs[name] = digest
        candidates.append(row)
    candidates.sort(key=lambda row: row["episode"])
    validate_coverage(candidates, expected)
    ranks, mismatches = changed_streams(originals, candidates)
    args.output.mkdir(parents=True, exist_ok=False)
    plan = {"status": "input_only_whole_stream_repair_frozen", "task": args.task, "arm": args.arm,
        "freeze_sha256": frozen_hash, "original_candidate_report_sha256": report_hash,
        "original_candidate_shard": str(args.candidate_shard), "native_reference_reports_sha256": native_bindings,
        "original_episode_files_sha256": inputs, "goal_bank_report_sha256": bank.report_hash,
        "goal_delivery_report_sha256": delivery_hash, "mismatches": mismatches, "logical_ranks": ranks,
        "replacement_episodes": [row for row in schedule() if row["logical_rank"] in ranks],
        "original_episodes": len(expected), "episodes_to_repeat": 12 * len(ranks),
        "rule": "replace entire original 12-episode logical stream including previously matching episodes; do not double count",
        "outcomes_used_for_repair_selection": False, "sample_size_reduced": False,
        "unchanged_scientific_panel_seeds_rng_budget": True, "confirmation_authorized": False,
        "source_sha256": sha256(Path(__file__))}
    write_json(args.output / "plan.json", plan)
    write_json(args.output / "FROZEN.json", {"plan_sha256": sha256(args.output / "plan.json"),
        "replacement_outcomes_not_yet_generated": True})
    print(json.dumps({key: plan[key] for key in ("task", "arm", "logical_ranks", "episodes_to_repeat")}))


if __name__ == "__main__":
    main()
