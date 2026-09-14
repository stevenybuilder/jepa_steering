"""Integrate an unchanged BF16-fitted static arm into the official FP32 planner.

Only the already-used, excluded simulator smoke seed is exercised. This is an
integration/throughput check, not a fresh policy-efficacy trial or candidate selector.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from offline_study.evaluation.checkpoint.author_evaluate import verify_fit
from offline_study.models.backends import JepaBackend
from offline_study.planning.planning_contract import prepare
from offline_study.planning.planning_intervention import StaticPlanningIntervention
from offline_study.planning.planning_native_smoke import CHECKPOINTS, SMOKE_SEED, run_episode
from offline_study.core.protocol import sha256, write_json
from offline_study.models.vendor import use_vendor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vendor", "original-root", "runtime-root", "output", "transfer-parity", "simulator-smoke"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--task", choices=("reach", "reach-wall"), required=True)
    parser.add_argument("--arm", choices=("joint", "matched_random"), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        parity_done = json.loads((args.transfer_parity / "DONE.json").read_text())
        if sha256(args.transfer_parity / "report.json") != parity_done["report_sha256"]:
            raise ValueError("Transfer parity receipt changed")
        parity = json.loads((args.transfer_parity / "report.json").read_text())
        if (not parity.get("primary_bfloat16_bank_on_float32_planner_verified") or
                parity["development_or_protected_outcomes_accessed"]):
            raise ValueError("Need fit-only parity of the PRIMARY bank on the FP32 planner")
        fit = args.original_root / "fits-v1/bfloat16" / args.task / "vision_action_coupling"
        cohort = args.original_root / "cohorts" / args.task / "cohort.json"
        _, protocol = verify_fit(fit, sha256(cohort), CHECKPOINTS["metaworld"], "bfloat16")
        bindings = parity["source_bindings"]["bfloat16_bank_to_float32"]
        if (sha256(fit / "protocol.json") != bindings["protocol_sha256"] or
                sha256(fit / "operator_bank.pt") != bindings["bank_sha256"]):
            raise ValueError("Static integration no longer uses the verified primary operator")
        simulator_done = json.loads((args.simulator_smoke / "DONE.json").read_text())
        if sha256(args.simulator_smoke / "report.json") != simulator_done["report_sha256"]:
            raise ValueError("Simulator receipt changed")
        simulator = json.loads((args.simulator_smoke / "report.json").read_text())
        use_vendor(args.vendor)
        from omegaconf import OmegaConf
        from evals.simu_env_planning.envs.init import make_env
        from evals.simu_env_planning.planning.gc_agent import GC_Agent
        random.seed(SMOKE_SEED)
        np.random.seed(SMOKE_SEED)
        torch.manual_seed(SMOKE_SEED)
        contract = prepare(args.vendor, args.task)
        cfg = OmegaConf.create(contract["config"])
        cfg.meta.seed = cfg.local_seed = SMOKE_SEED
        backend = JepaBackend(args.vendor, args.runtime_root / "checkpoints/jepa_wm_metaworld.pth.tar",
                              CHECKPOINTS["metaworld"], "metaworld", "cuda:0", "float32")
        bank = torch.load(fit / "operator_bank.pt", map_location="cpu", weights_only=True)
        agent = GC_Agent(cfg, backend.model, preprocessor=backend.preprocessor)
        adapter = StaticPlanningIntervention(backend, protocol, bank, args.arm)
        agent.planner.unroll = adapter
        env = make_env(cfg)
        try:
            result = run_episode(cfg, backend, agent, env, SMOKE_SEED)
        finally:
            env.close()
        for key in ("initial_sha256", "goal_sha256"):
            if result[key] != simulator["tasks"][args.task][key]:
                raise ValueError("Integration changed the verified initial/goal setup")
        if not adapter.calls:
            raise ValueError("Official CEM never called the intervention")
        write_json(args.output / "report.json", {"status": "static_primary_bank_cem_integration_smoke_complete",
            "task": args.task, "arm": args.arm, "seed": SMOKE_SEED,
            "bank_precision": "bfloat16", "planner_precision": "float32_strict_no_tf32",
            "operator_binding": bindings, "parity_report_sha256": parity_done["report_sha256"],
            "scientific_efficacy_measurement": False, "fresh_confirmation": False,
            "candidate_selection_from_smoke_forbidden": True, "protected_outcomes_accessed": False,
            "adapter_calls": adapter.calls, "energy": adapter.energy,
            "seconds": time.monotonic() - started,
            "result": {k.replace("native_", "policy_"): v for k, v in result.items()}})
        write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "scientific_efficacy_measurement": False})
        raise


if __name__ == "__main__":
    main()
