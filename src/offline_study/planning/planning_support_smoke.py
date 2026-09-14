"""Full official simulator/CEM integration on the already-used smoke seed.

The exact selected non-routed MW recipes and their matched controls are fixed.
These episodes cannot select a candidate or supply confirmation evidence.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from offline_study.evaluation.checkpoint.author_evaluate import verify_fit
from offline_study.fitting.author_fit import source_hash
from offline_study.evaluation.checkpoint.author_runtime import validate_cohort
from offline_study.models.backends import JepaBackend
from offline_study.runtime.intervention_runner import _model_versions
from offline_study.planning.planning_contract import prepare
from offline_study.planning.planning_native_smoke import CHECKPOINTS, SMOKE_SEED, run_episode
from offline_study.planning.planning_support import PlanningSupportIntervention, TRANSFER_POLICY
from offline_study.core.protocol import sha256, write_json
from offline_study.models.vendor import use_vendor


TRANSFER_RECEIPTS = {
    ("reach", "rank4"): "e40c3421f8b3e4fd823538caad99fa70967d89310482193d5a8d94fb1457853e",
    ("reach", "matched_random_rank4"): "b28336c4c9379eac1ae6d2e5e9501b3efbe7226acea44c583fa3b321c3f0be42",
    ("reach-wall", "rank4"): "c9c20732f474ce54f35aa2ab6fa8bc968da5ea5a1e82caa4b15eaad1cd4357c8",
    ("reach-wall", "matched_random_rank4"): "37de0eeccdae62483f65e82aa8616479d280ae083bb4b7397c755ee4ad5a50da",
}
SIMULATOR_RECEIPT = "60fd2871a278acc2ad76a82e0b766d84b50abde1270820ee73bf59d7199b7842"


def verified_report(root, expected):
    done = json.loads((root / "DONE.json").read_text())
    if done["report_sha256"] != expected or sha256(root / "report.json") != expected:
        raise ValueError("Required completed receipt changed: " + str(root))
    return json.loads((root / "report.json").read_text())


def validate_transfer(report, contract, task, arm, rank_binding, coupling_binding):
    if (report.get("status") != "primary_rank_planning_transfer_fit_only_passed" or
            report.get("profiled_rank_arm") != arm or
            report.get("combined_planning_transfer_validated") != (task == "reach") or
            not report.get("diverse_fit_actions_checked") or
            not report.get("parameters_unchanged") or
            report.get("development_or_protected_outcomes_accessed") is not False or
            report.get("planning_precision") != "float32_strict_no_tf32" or
            report.get("planning_context") != 2):
        raise ValueError("Wrong exact-candidate fit-only transfer evidence")
    if (contract.get("policy") != TRANSFER_POLICY or
            contract.get("profiled_rank_arm") != arm or
            contract.get("diagnostic_probe_batching_only") is not False or
            contract.get("source_protocol_sha256") != rank_binding["protocol_sha256"] or
            contract.get("source_bank_sha256") != rank_binding["bank_sha256"] or
            contract.get("combined_coupling_binding") != coupling_binding):
        raise ValueError("Source operator or frozen transfer policy changed")
    if not any(row.get("candidate_count") == 300 and
               row.get("full_population_forecast_bitwise_matches_source_hook_compilation") is True
               for row in report["checks"]):
        raise ValueError("Missing full-population exact source parity")


def verify_call_schedule(planning_calls, unroll_calls):
    expected = []
    if not planning_calls:
        raise ValueError("No planning calls")
    for plan in planning_calls:
        if plan["iterations"] != 15:
            raise ValueError("Official CEM iteration budget changed")
        horizon = min(6, plan["steps_left"])
        expected.extend([(horizon, 300), (horizon, 1)] * 15)
    actual = [(row["horizon"], row["candidates"]) for row in unroll_calls]
    if actual != expected:
        raise ValueError("Missing/reordered/shortened official CEM population or mean forecasts")


class ObservedSupport:
    """Record liveness and actual CEM work without changing its optimizer."""
    def __init__(self, adapter, output):
        self.adapter, self.output, self.calls = adapter, output, []

    def __call__(self, context, act_suffix=None, **kwargs):
        horizon, candidates, _ = act_suffix.shape
        before = time.monotonic()
        write_json(self.output / "progress.json", {"completed_unroll_calls": len(self.calls),
            "running_horizon": horizon, "running_candidate_count": candidates,
            "state": "inside_official_cem", "fresh_confirmation": False})
        result = self.adapter(context, act_suffix, **kwargs)
        if any(not torch.isfinite(result[key]).all() for key in ("visual", "proprio")):
            raise ValueError("Nonfinite selected-candidate forecast")
        torch.cuda.synchronize()
        row = {"horizon": horizon, "candidates": candidates,
               "seconds": time.monotonic() - before, "energy": self.adapter.energy[-1]}
        self.calls.append(row)
        write_json(self.output / "unroll_calls.json", self.calls)
        write_json(self.output / "progress.json", {"completed_unroll_calls": len(self.calls),
            "last_call": row, "state": "unroll_returned_to_official_cem", "fresh_confirmation": False})
        print(json.dumps({"completed_unroll_calls": len(self.calls), **row}), flush=True)
        return result


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vendor", "original-root", "runtime-root", "output", "transfer-parity", "simulator-smoke"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--task", choices=("reach", "reach-wall"), required=True)
    parser.add_argument("--arm", choices=("rank4", "matched_random_rank4"), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        parity_hash = TRANSFER_RECEIPTS[args.task, args.arm]
        parity = verified_report(args.transfer_parity, parity_hash)
        if sha256(args.transfer_parity / "transfer_contract.json") != parity["transfer_contract_sha256"]:
            raise ValueError("Transfer contract changed")
        transfer = json.loads((args.transfer_parity / "transfer_contract.json").read_text())
        cohort_path = args.original_root / "cohorts" / args.task / "cohort.json"
        cohort = json.loads(cohort_path.read_text())
        validate_cohort(cohort)
        if cohort["task"] != "mw-" + args.task or parity["fit_trajectory"] not in {
                row["trajectory_id"] for row in cohort["fit"]}:
            raise ValueError("Transfer evidence belongs to a different fitting population")
        fit = args.original_root / "fits-v1/bfloat16" / args.task / "operator_rank"
        _, protocol = verify_fit(fit, sha256(cohort_path), CHECKPOINTS["metaworld"], "bfloat16")
        rank_binding = {"protocol_sha256": sha256(fit / "protocol.json"),
                        "bank_sha256": sha256(fit / "operator_bank.pt")}
        coupling, coupling_binding = None, None
        if args.task == "reach":
            coupling_fit = fit.parent / "vision_action_coupling"
            _, coupling_protocol = verify_fit(coupling_fit, sha256(cohort_path), CHECKPOINTS["metaworld"], "bfloat16")
            coupling_binding = {"protocol_sha256": sha256(coupling_fit / "protocol.json"),
                                "bank_sha256": sha256(coupling_fit / "operator_bank.pt")}
            coupling_arm = ("matched_random_equal_standardized_energy" if args.arm.startswith("matched_random")
                            else "joint_equal_standardized_energy")
            coupling = (coupling_protocol, torch.load(coupling_fit / "operator_bank.pt", weights_only=True,
                                                    map_location="cpu"), coupling_arm)
        validate_transfer(parity, transfer, args.task, args.arm, rank_binding, coupling_binding)
        simulator = verified_report(args.simulator_smoke, SIMULATOR_RECEIPT)
        use_vendor(args.vendor)
        from omegaconf import OmegaConf
        from evals.simu_env_planning.envs.init import make_env
        from evals.simu_env_planning.planning.gc_agent import GC_Agent
        prepared = prepare(args.vendor, args.task)
        cfg = OmegaConf.create(prepared["config"])
        cfg.meta.seed = cfg.local_seed = SMOKE_SEED
        write_json(args.output / "protocol.json", {
            "role": "full_official_cem_simulator_integration_on_reused_excluded_smoke_seed",
            "task": args.task, "arm": args.arm, "combined": coupling is not None,
            "seed": SMOKE_SEED, "source_sha256": source_hash(),
            "parity_report_sha256": parity_hash, "simulator_report_sha256": SIMULATOR_RECEIPT,
            "rank_binding": rank_binding, "coupling_binding": coupling_binding,
            "transfer_policy": TRANSFER_POLICY, "planning_contract": prepared,
            "executed_seed_override": SMOKE_SEED, "episodes_in_this_engineering_check": 1,
            "not_the_96_episode_confirmation": True,
            "candidate_selection_from_smoke_forbidden": True, "fresh_confirmation": False,
            "protected_outcomes_access_authorized": False,
            "no_tuning_or_shortening_after_smoke_results": True})
        random.seed(SMOKE_SEED)
        np.random.seed(SMOKE_SEED)
        torch.manual_seed(SMOKE_SEED)
        backend = JepaBackend(args.vendor, args.runtime_root / "checkpoints/jepa_wm_metaworld.pth.tar",
                              CHECKPOINTS["metaworld"], "metaworld", "cuda:0", "float32")
        versions = _model_versions(backend.model)
        torch.cuda.reset_peak_memory_stats()
        bank = torch.load(fit / "operator_bank.pt", weights_only=True, map_location="cpu")
        agent = GC_Agent(cfg, backend.model, preprocessor=backend.preprocessor)
        adapter = PlanningSupportIntervention(backend, protocol, bank, args.arm, coupling)
        observed = ObservedSupport(adapter, args.output)
        agent.planner.unroll = observed
        env = make_env(cfg)
        try:
            result = run_episode(cfg, backend, agent, env, SMOKE_SEED)
        finally:
            env.close()
        for key in ("initial_sha256", "goal_sha256"):
            if result[key] != simulator["tasks"][args.task][key]:
                raise ValueError("Selected integration changed the verified initial/goal scenario")
        verify_call_schedule(result["planning_calls"], observed.calls)
        if result["elementary_steps"] != 100 or _model_versions(backend.model) != versions:
            raise ValueError("Incomplete official episode or altered frozen parameters")
        write_json(args.output / "report.json", {
            "status": "selected_primary_recipe_full_cem_simulator_smoke_complete",
            "task": args.task, "arm": args.arm, "combined": coupling is not None,
            "protocol_sha256": sha256(args.output / "protocol.json"),
            "unroll_calls_sha256": sha256(args.output / "unroll_calls.json"),
            "actual_cem_integration_validated": True, "full_simulator_episode_validated": True,
            "adapter_calls": len(observed.calls), "parameters_unchanged": True,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(), "seconds": time.monotonic() - started,
            "result": {k.replace("native_", "policy_"): v for k, v in result.items()},
            "candidate_selection_from_smoke_forbidden": True, "scientific_efficacy_measurement": False,
            "fresh_confirmation": False, "protected_outcomes_accessed": False})
        write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "scientific_efficacy_measurement": False})
        raise


if __name__ == "__main__":
    main()
