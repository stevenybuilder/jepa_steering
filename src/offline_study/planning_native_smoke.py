"""Run the unmodified official CEM and simulator loop on non-confirmatory seeds.

The published candidate/iteration counts are retained. Only post-episode plots,
video encoding and decoded diagnostics are omitted; no policy outcomes select an edit.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .backends import JepaBackend
from .author_runtime import open_normalized_dataset, validate_cohort
from .planning_contract import prepare
from .planning_env_smoke import SingleFitTrajectory, observation_digest
from .protocol import sha256, write_json
from .vendor import use_vendor


CHECKPOINTS = {
    "metaworld": "c3297772c7af4e84c28bd0c0937e22398a888848305014bb3ab3f1508a310bd8",
    "pusht": "9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb",
}
SMOKE_SEED = 2026090719


def run_episode(cfg, backend, agent, env, seed):
    """Same setup, action callback and unroll_agent as PlanEvaluator.eval."""
    from evals.simu_env_planning.planning.plan_evaluator import PlanEvaluator
    from evals.utils import prepare_obs

    evaluator = PlanEvaluator(cfg, agent)
    _, info = env.reset(seed=seed, task_idx=0)
    unwrapped = env.proprio_env.unwrapped
    unwrapped._freeze_rand_vec = False
    unwrapped.seeded_rand_vec = True
    env.seed(seed)
    init, goal, expert, expert_success = evaluator.set_episode(cfg, agent, env, seed, task_idx=0)
    fingerprints = {"initial_sha256": observation_digest(init), "goal_sha256": observation_digest(goal)}
    agent.set_goal(prepare_obs(cfg.task_specification.obs, goal))
    evaluator.prev_losses, evaluator.prev_elite_losses_mean, evaluator.prev_elite_losses_std = [], [], []
    evaluator.prev_pred_frames_over_iterations, evaluator.predicted_best_encs_over_iterations = [], []
    planning_calls = []

    def actor(obs, steps_left):
        before = time.monotonic()
        action = agent.act(prepare_obs(cfg.task_specification.obs, obs), steps_left=steps_left)
        if not torch.isfinite(action).all() or not torch.isfinite(agent._prev_losses).all():
            raise ValueError("Nonfinite native CEM outputs")
        if len(agent._prev_losses) != cfg.planner.iterations:
            raise ValueError("CEM did not perform the published iteration count")
        planning_calls.append({"steps_left": int(steps_left), "returned_model_actions": len(action),
                               "iterations": len(agent._prev_losses), "seconds": time.monotonic() - before})
        evaluator.prev_losses.append(agent._prev_losses)
        evaluator.prev_elite_losses_mean.append(agent._prev_elite_losses_mean)
        evaluator.prev_elite_losses_std.append(agent._prev_elite_losses_std)
        evaluator.prev_pred_frames_over_iterations.append(agent._prev_pred_frames_over_iterations)
        evaluator.predicted_best_encs_over_iterations.append(agent._predicted_best_encs_over_iterations)
        return action

    with torch.no_grad(), backend.autocast():
        observations, reward, actions, infos, success, distance = evaluator.unroll_agent(
            env, init, info, actor, preprocessor=agent.preprocessor)
    return {**fingerprints, "expert_success": float(expert_success), "native_success": bool(success),
            "native_state_distance": float(distance), "native_reward": float(reward),
            "elementary_steps": sum(len(a) for a in actions), "planning_calls": planning_calls,
            "observed_frames": len(observations), "published_candidate_count": agent.planner.num_samples}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vendor", "original-root", "runtime-root", "output", "simulator-smoke"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--task", choices=["reach", "reach-wall", "pusht"], required=True)
    parser.add_argument("--memory-fraction", type=float, default=1.)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        use_vendor(args.vendor)
        from omegaconf import OmegaConf
        from evals.simu_env_planning.envs.init import make_env
        from evals.simu_env_planning.planning.gc_agent import GC_Agent

        simulator = json.loads((args.simulator_smoke / "report.json").read_text())
        done = json.loads((args.simulator_smoke / "DONE.json").read_text())
        if sha256(args.simulator_smoke / "report.json") != done["report_sha256"]:
            raise ValueError("Simulator setup receipt changed")
        if not 0 < args.memory_fraction <= 1:
            raise ValueError("Invalid bounded GPU memory allocation")
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
        torch.cuda.reset_peak_memory_stats()
        random.seed(SMOKE_SEED)
        np.random.seed(SMOKE_SEED)
        torch.manual_seed(SMOKE_SEED)
        contract = prepare(args.vendor, args.task)
        cfg = OmegaConf.create(contract["config"])
        cfg.meta.seed = SMOKE_SEED
        cfg.local_seed = SMOKE_SEED
        kind = "pusht" if args.task == "pusht" else "metaworld"
        # Upstream standalone planning does not enable an outer autocast context.
        # Keep strict FP32 here; offline BF16 is not silently imposed on planning.
        backend = JepaBackend(args.vendor, args.runtime_root / "checkpoints" / ("jepa_wm_" + kind + ".pth.tar"),
                              CHECKPOINTS[kind], kind, "cuda:0", "float32")
        if backend.model.ctxt_window != 2:
            raise ValueError("Planning wrapper context must be two")
        dataset, fit_id = None, None
        if args.task == "pusht":
            cohort = json.loads((args.original_root / "cohorts/pusht/cohort.json").read_text())
            validate_cohort(cohort)
            full = open_normalized_dataset("pusht", args.runtime_root / "data/pusht_noise", cohort["reference_config"], True)
            dataset = SingleFitTrajectory(full, cohort["fit"][0]["index"])
            fit_id = cohort["fit"][0]["trajectory_id"]
        agent = GC_Agent(cfg, backend.model, dset=dataset, preprocessor=backend.preprocessor)
        env = make_env(cfg)
        try:
            result = run_episode(cfg, backend, agent, env, SMOKE_SEED)
        finally:
            env.close()
        expected = simulator["tasks"][args.task]
        for key in ("initial_sha256", "goal_sha256"):
            if result[key] != expected[key]:
                raise ValueError("Planning setup differs from the verified simulator smoke: " + key)
        write_json(args.output / "report.json", {
            "status": "official_native_cem_smoke_complete", "task": args.task,
            "scientific_efficacy_measurement": False, "fresh_confirmation": False,
            "protected_outcomes_accessed": False, "smoke_seed_excluded_from_confirmation": SMOKE_SEED,
            "fit_trajectory_id": fit_id, "precision": "float32_strict_no_tf32",
            "checkpoint_sha256": CHECKPOINTS[kind], "planning_context": 2,
            "simulator_receipt_sha256": done["report_sha256"],
            "source_config_sha256": contract["source_config_sha256"],
            "native_planner_implementation": "unmodified upstream GC_Agent + CEMPlanner + PlanEvaluator.unroll_agent",
            "omitted_work": "post-episode plots/videos/decoded diagnostics only",
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(), "seconds": time.monotonic() - started,
            "result": result})
        write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "scientific_efficacy_measurement": False})
        raise


if __name__ == "__main__":
    main()
