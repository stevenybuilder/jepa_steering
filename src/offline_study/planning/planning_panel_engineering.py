"""Engineer the frozen equal-budget drop-one coupling panel, with common H6 scope.

Fits and doses are unchanged. Native at shortened horizons matches the already
frozen combined/drop-one comparison schedule. Only excluded engineering stimuli.
"""
from offline_study._paths import source_path
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from offline_study.evaluation.checkpoint.author_evaluate import verify_fit
from offline_study.evaluation.checkpoint.author_runtime import validate_cohort
from offline_study.models.backends import JepaBackend
from offline_study.runtime.intervention_runner import _model_versions
from offline_study.interventions.interventions import PredictorIntervention
from offline_study.planning.planning_contract import prepare
from offline_study.planning.planning_intervention import StaticPlanningIntervention
from offline_study.planning.planning_native_smoke import CHECKPOINTS, SMOKE_SEED, run_episode
from offline_study.planning.planning_scenarios import close_expert_environments
from offline_study.planning.planning_support_check import fit_candidate_actions, source_coupling_fields
from offline_study.planning.planning_support_smoke import ObservedSupport, verify_call_schedule
from offline_study.core.protocol import sha256, write_json
from offline_study.models.vendor import use_vendor


COUPLING_ARMS = {"coupling_only": "joint_equal_standardized_energy",
                 "matched_random_coupling": "matched_random_equal_standardized_energy"}


class H6StaticPlanningIntervention(StaticPlanningIntervention):
    """The frozen seven-condition panel applies every component only at full H6."""
    @torch.no_grad()
    def __call__(self, context, act_suffix=None, **kwargs):
        if act_suffix is None or kwargs or not 1 <= act_suffix.shape[0] <= 6:
            raise ValueError("Explicit native-default planning actions required")
        horizon, batch, _ = act_suffix.shape
        if horizon < 6:
            result = self.backend.predict(context, act_suffix)
            self.calls += 1
            self.energy.append({"horizon": horizon, "candidates": batch,
                "requested_squared_l2_mean": 0., "realized_squared_l2_mean": 0.})
            return result
        return super().__call__(context, act_suffix=act_suffix)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vendor", "runtime-root", "original-root", "fixture", "simulator-smoke", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--task", required=True, choices=("reach", "reach-wall"))
    parser.add_argument("--arm", required=True, choices=tuple(COUPLING_ARMS))
    args = parser.parse_args()
    use_vendor(args.vendor)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        fixture = json.loads((args.fixture / "report.json").read_text())
        if sha256(args.fixture / "report.json") != json.loads((args.fixture / "DONE.json").read_text())["report_sha256"]:
            raise ValueError("Unverified fit fixture")
        for name, suffix in (("cohort", ".json"), ("inputs", ".pt")):
            if sha256(args.fixture / (name + suffix)) != fixture[name + "_sha256"]:
                raise ValueError("Fit fixture inputs changed")
        cohort = json.loads((args.fixture / "cohort.json").read_text())
        validate_cohort(cohort)
        if (cohort["task"] != "mw-" + args.task or fixture["selected"] != cohort["fit"][:1] or
                fixture["development_or_protected_outcomes_accessed"]):
            raise ValueError("Wrong or outcome-exposed fixture")
        fit = args.original_root / "fits-v1/bfloat16" / args.task / "vision_action_coupling"
        _, protocol = verify_fit(fit, fixture["source_cohort_sha256"], CHECKPOINTS["metaworld"], "bfloat16")
        bank = torch.load(fit / "operator_bank.pt", map_location="cpu", weights_only=True)
        simulator = json.loads((args.simulator_smoke / "report.json").read_text())
        simulator_hash = sha256(args.simulator_smoke / "report.json")
        if simulator_hash != json.loads((args.simulator_smoke / "DONE.json").read_text())["report_sha256"]:
            raise ValueError("Simulator proof changed")
        contract = prepare(args.vendor, args.task)
        write_json(args.output / "protocol.json", {"role": "frozen_panel_coupling_engineering",
            "task": args.task, "arm": args.arm, "source_arm": COUPLING_ARMS[args.arm],
            "source_sha256": {name: sha256(source_path(name)) for name in
                ("planning_panel_engineering.py", "planning_intervention.py", "planning_native_smoke.py")},
            "fit_protocol_sha256": sha256(fit / "protocol.json"),
            "bank_sha256": sha256(fit / "operator_bank.pt"),
            "fixture_report_sha256": sha256(args.fixture / "report.json"),
            "simulator_report_sha256": simulator_hash, "planning": contract,
            "common_scope": "all edits only in full H6; true native below H6",
            "candidate_counts": [8, 19, 300], "parity": "bitwise_source_fields_and_true_native",
            "smoke_seed": SMOKE_SEED, "precision": "float32_no_tf32",
            "outcome_selection": False, "fresh_confirmation": False})
        torch.cuda.set_device(0)
        backend = JepaBackend(args.vendor, args.runtime_root / "checkpoints/jepa_wm_metaworld.pth.tar",
                              CHECKPOINTS["metaworld"], "metaworld", "cuda:0", "float32")
        versions = _model_versions(backend.model)
        inputs = torch.load(args.fixture / "inputs.pt", map_location="cpu", weights_only=True)
        visual, proprio, _ = backend.model.model.encode(
            {k: inputs[k].to(backend.device) for k in ("visual", "proprio")}, inputs["action"].to(backend.device))
        from tensordict import TensorDict
        context = TensorDict({"visual": visual[:1, :1], "proprio": proprio[:1, :1]}, batch_size=[])
        checks = []
        for count in (8, 19, 300):
            actions = fit_candidate_actions(inputs["action"], count).to(backend.device)
            for horizon in (2, 5, 6):
                native = backend.predict(context, actions[:horizon])
                for name in ("native", "zero_dose", COUPLING_ARMS[args.arm]):
                    adapter = H6StaticPlanningIntervention(backend, protocol, bank, name)
                    actual = adapter(context, actions[:horizon])
                    if horizon < 6 or name in ("native", "zero_dose"):
                        expected = native
                    else:
                        fields = source_coupling_fields((protocol, bank, name), count, backend.device)
                        with PredictorIntervention(backend.predictor, fields):
                            expected = backend.predict(context, actions)
                    if any(not torch.equal(actual[k], expected[k]) for k in ("visual", "proprio")):
                        raise ValueError("H6 common-scope adapter changed source/native predictions")
                    checks.append({"candidates": count, "horizon": horizon, "arm": name, "bitwise_equal": True})
        write_json(args.output / "fit_parity.json", {"checks": checks, "development_outcomes_accessed": False})
        del inputs, visual, proprio, context, actions, native, actual, expected
        from omegaconf import OmegaConf
        from evals.simu_env_planning.envs.init import make_env
        from evals.simu_env_planning.planning.gc_agent import GC_Agent
        from evals.simu_env_planning.planning import plan_evaluator
        random.seed(SMOKE_SEED)
        np.random.seed(SMOKE_SEED)
        torch.manual_seed(SMOKE_SEED)
        cfg = OmegaConf.create(contract["config"])
        cfg.local_seed = cfg.meta.seed = SMOKE_SEED
        agent = GC_Agent(cfg, backend.model, preprocessor=backend.preprocessor)
        adapter = H6StaticPlanningIntervention(backend, protocol, bank, COUPLING_ARMS[args.arm])
        observed = ObservedSupport(adapter, args.output)
        agent.planner.unroll = observed
        env = make_env(cfg)
        try:
            with close_expert_environments(plan_evaluator):
                result = run_episode(cfg, backend, agent, env, SMOKE_SEED)
        finally:
            env.close()
        verify_call_schedule(result["planning_calls"], observed.calls)
        if result["elementary_steps"] != 100 or _model_versions(backend.model) != versions:
            raise ValueError("Incomplete native episode or changed weights")
        if any(result[k] != simulator["tasks"][args.task][k] for k in ("initial_sha256", "goal_sha256")):
            raise ValueError("Different excluded engineering initial/goal scenario")
        write_json(args.output / "report.json", {"status": "common_h6_coupling_panel_full_cem_engineering_passed",
            "task": args.task, "arm": args.arm, "protocol_sha256": sha256(args.output / "protocol.json"),
            "fit_parity_sha256": sha256(args.output / "fit_parity.json"),
            "unroll_calls_sha256": sha256(args.output / "unroll_calls.json"),
            "result": result, "seconds": time.monotonic() - started,
            "parameters_unchanged": True, "fresh_confirmation": False,
            "scientific_efficacy_measurement": False, "candidate_selection_from_smoke_forbidden": True})
        write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "fresh_confirmation": False})
        raise


if __name__ == "__main__":
    main()
