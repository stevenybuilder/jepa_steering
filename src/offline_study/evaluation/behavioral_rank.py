"""Run already-engineered rank/combined members of the frozen behavioral panel.

This admits no new method, fit, dose, seed or shortened planner. Other drop-one
members stay disabled until their own full integration receipts are available.
"""
from offline_study._paths import source_path
import argparse
import json
from pathlib import Path

import torch

from offline_study.evaluation.checkpoint.author_evaluate import verify_fit
from offline_study.evaluation.behavioral_candidate import immutable_source_hash, run_shard
from offline_study.evaluation.behavioral_development import ARMS, DEVELOPMENT_SEED, assigned_rows, verified_report
from offline_study.planning.planning_contract import prepare
from offline_study.planning.planning_native_smoke import CHECKPOINTS
from offline_study.planning.planning_support import PlanningSupportIntervention, TRANSFER_POLICY
from offline_study.core.protocol import sha256
from offline_study.models.vendor import use_vendor


ENABLED = {("reach", "combined"), ("reach", "matched_random_combined"),
           ("reach-wall", "rank4_only"), ("reach-wall", "matched_random_rank4")}
CRITICAL = ("backends.py", "model_loader.py", "planning_native_smoke.py", "planning_contract.py",
            "planning_support.py", "support_operator.py", "support_fit.py", "interventions.py",
            "planning_intervention.py", "planning_support_smoke.py")

# Audited complete code-v28 -> current diffs, not a generic compatibility bypass.
# First three only extend dataset dispatch to navigation; MetaWorld constructor,
# normalization and returned config are unchanged (also bound to code-v33 below).
# Last extracts prepare_support into a forwarding method; this runner constructs
# exactly PlanningSupportIntervention, never the optional cached subclass.
REVIEWED_EQUIVALENT = {
    "backends.py": ("ddce8aaf463dfd39dee47ea06f6ac3e810e232579229c835d14bdaeea9bf6fa1",
                    "a64ad6a31fa98b866d17880bb15ac68f31fc9731bbb144478a1cc6c6799fda46"),
    "model_loader.py": ("c504e6b227ff98c3e66ab37ed724e01a347a89401671156e5dbdb373ffcc1db3",
                        "01cc4ca4b90ab92b2e7bd3c4b99bd3f5a305f16a6c14e62816481af5bace68f5"),
    "planning_contract.py": ("fa51f28f9ee31117c2e6e6aa492da3aac3fc476ed360a363cc4c8466e7217307",
                             "741d1716394820f7a471c3e8edf4013b85660e37df8f05ab6bd3847e874bada0"),
    "planning_support.py": ("01bd5b64f6ff85bb2afb55af895e8d456de2b923c43714c402d7790f2f85bc96",
                            "48baa2ac7640865fb5038033430a5bae6fd3ea125d226e46c4a18b373b9cdc63"),
}


def verify_source_pair(name, old, current):
    if old != current and REVIEWED_EQUIVALENT.get(name) != (old, current):
        raise ValueError("Engineering operator path changed without equivalence audit: " + name)


def components(task, arm):
    if (task, arm) not in ENABLED:
        raise ValueError("This panel member has no bound complete engineering yet")
    _, coupling, rank = next(row for row in ARMS if row[0] == arm)
    return coupling, rank


def validate_engineering(report, ep, task, arm, rank_binding, coupling_binding):
    coupling, rank = components(task, arm)
    combined = coupling != "native"
    if (report.get("status") != "selected_primary_recipe_full_cem_simulator_smoke_complete" or
            report.get("task") != task or report.get("arm") != rank or
            report.get("combined") != combined or ep.get("combined") != combined or
            ep.get("task") != task or ep.get("arm") != rank or
            ep.get("rank_binding") != rank_binding or ep.get("coupling_binding") != coupling_binding or
            ep.get("transfer_policy") != TRANSFER_POLICY or
            not report.get("actual_cem_integration_validated") or
            not report.get("full_simulator_episode_validated") or
            not report.get("parameters_unchanged") or
            report.get("scientific_efficacy_measurement") is not False or
            report.get("protected_outcomes_accessed") is not False or
            report.get("result", {}).get("elementary_steps") != 100):
        raise ValueError("Missing exact full-budget task/component integration evidence")


def launch_contract(args):
    coupling_arm, rank_arm = components(args.task, args.arm)
    frozen = json.loads((args.freeze / "FROZEN.json").read_text())
    if sha256(args.freeze / "protocol.json") != frozen["protocol_sha256"]:
        raise ValueError("Scientific freeze changed")
    protocol = json.loads((args.freeze / "protocol.json").read_text())
    if (protocol["role"] != "frozen_planning_development_not_confirmation" or
            protocol["arms"] != [list(row) for row in ARMS] or
            immutable_source_hash(args.baseline_code / "src/offline_study") != protocol["source_sha256"]):
        raise ValueError("Unverified original scientific panel/source")
    for name in ("backends.py", "model_loader.py", "planning_native_smoke.py", "planning_scenarios.py",
                 "planning_contract.py", "behavioral_development.py", "planning_env_smoke.py"):
        if sha256(args.baseline_code / "src/offline_study" / name) != sha256(source_path(name)):
            raise ValueError("Native comparison path changed: " + name)
    expected = assigned_rows(protocol["episodes"], args.logical_ranks)
    task = protocol["tasks"][args.task]
    cfg = prepare(args.vendor, args.task)["config"]
    cfg["meta"]["seed"] = DEVELOPMENT_SEED
    if cfg != task["planning"]["config"]:
        raise ValueError("Native planning config changed")
    fits = {}
    for category in ("operator_rank", "vision_action_coupling"):
        root = args.original_root / "fits-v1/bfloat16" / args.task / category
        for name, digest in task["components"][category].items():
            if sha256(root / name) != digest:
                raise ValueError("Frozen component changed: " + category)
        _, fit = verify_fit(root, task["cohort_sha256"], CHECKPOINTS["metaworld"], "bfloat16")
        fits[category] = (root, fit)
    def binding(category):
        root = fits[category][0]
        return {"protocol_sha256": sha256(root / "protocol.json"),
                "bank_sha256": sha256(root / "operator_bank.pt")}
    report, digest = verified_report(args.engineering)
    ep = json.loads((args.engineering / "protocol.json").read_text())
    if (report["protocol_sha256"] != sha256(args.engineering / "protocol.json") or
            report["unroll_calls_sha256"] != sha256(args.engineering / "unroll_calls.json") or
            immutable_source_hash(args.engineering_code / "src/offline_study") != ep["source_sha256"]):
        raise ValueError("Engineering receipt/source changed")
    for name in CRITICAL:
        verify_source_pair(name, sha256(args.engineering_code / "src/offline_study" / name),
                           sha256(source_path(name)))
    validate_engineering(report, ep, args.task, args.arm, binding("operator_rank"),
                         binding("vision_action_coupling") if coupling_arm != "native" else None)
    rank_root, rank_protocol = fits["operator_rank"]
    rank_bank = torch.load(rank_root / "operator_bank.pt", map_location="cpu", weights_only=True)
    coupling = None
    if coupling_arm != "native":
        root, fit = fits["vision_action_coupling"]
        coupling = (fit, torch.load(root / "operator_bank.pt", map_location="cpu", weights_only=True), coupling_arm)
    return protocol, expected, digest, rank_protocol, rank_bank, rank_arm, coupling


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vendor", "freeze", "baseline-code", "engineering-code", "original-root", "checkpoint",
                 "engineering", "output", "goal-bank", "goal-delivery-proof"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--task", choices=("reach", "reach-wall"), required=True)
    parser.add_argument("--arm", choices=tuple(sorted({arm for _, arm in ENABLED})), required=True)
    parser.add_argument("--logical-ranks", nargs="+", type=int, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    use_vendor(args.vendor)
    protocol, expected, digest, fit, bank, rank_arm, coupling = launch_contract(args)
    from offline_study.planning.planning_goal_bank import GoalBank, verify_delivery
    goals = GoalBank(args.goal_bank, args.task, sha256(args.freeze / "protocol.json"))
    verify_delivery(args.goal_delivery_proof, goals.report_hash, sha256(args.freeze / "protocol.json"))
    if args.verify_only:
        print(json.dumps({"status": "fixed_rank_launch_binding_passed", "task": args.task,
            "arm": args.arm, "episodes": len(expected), "engineering_report_sha256": digest,
            "canonical_native_goal_bank_report_sha256": goals.report_hash,
            "model_or_simulator_calls": 0}))
        return
    run_shard(args, protocol, expected, digest,
        lambda backend: PlanningSupportIntervention(backend, fit, bank, rank_arm, coupling),
        {"source_rank_arm": rank_arm, "source_coupling_arm": coupling[2] if coupling else "native",
         "transfer_policy": TRANSFER_POLICY, "no_unvalidated_runtime_optimization": True,
         "reviewed_equivalent_source_pairs": REVIEWED_EQUIVALENT})


if __name__ == "__main__":
    main()
