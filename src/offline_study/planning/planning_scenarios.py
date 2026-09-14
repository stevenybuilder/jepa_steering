"""Prepare candidate-independent MetaWorld inputs with the official goal helper.

This runs the expert used to DEFINE goals, never the learned planning policy. It
does not freeze a candidate, open confirmation outcomes, or authorize their reveal.
Keep all scenarios, including expert failures; do not filter by observed difficulty.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from offline_study.planning.planning_contract import prepare, seed_schedule
from offline_study.planning.planning_env_smoke import observation_digest
from offline_study.core.protocol import sha256, write_json
from offline_study.models.vendor import use_vendor


def validate_schedule(contract):
    if contract["task"] not in ("reach", "reach-wall") or contract["goal_source"] != "expert":
        raise ValueError("This preparation does not define fresh Push-T generation")
    expected = seed_schedule(contract["seed_policy"]["base_seed"])
    if contract["episodes"] != expected or contract["planned_replication_episodes_per_condition"] != 96:
        raise ValueError("The paper-scale schedule changed")
    if any(row["environment_seed"] == 2026090719 for row in expected):
        raise ValueError("Smoke seed cannot enter the prospective schedule")
    return expected


@contextlib.contextmanager
def close_expert_environments(module):
    """Close helpers AFTER the unchanged upstream expert has returned its outputs.

    Upstream get_goal_state_mw creates an auxiliary renderer without closing it.
    Repeated preparations need deterministic resource cleanup, not a new expert.
    """
    original, created = module.make_env, []

    def tracked(*args, **kwargs):
        env = original(*args, **kwargs)
        created.append(env)
        return env

    module.make_env = tracked
    try:
        yield
    finally:
        module.make_env = original
        for env in created:
            env.close()


def prepare_episode(cfg, agent, env, episode):
    from evals.simu_env_planning.planning import plan_evaluator

    seed = episode["environment_seed"]
    env.reset(seed=seed, task_idx=0)
    unwrapped = env.proprio_env.unwrapped
    unwrapped._freeze_rand_vec = False
    unwrapped.seeded_rand_vec = True
    env.seed(seed)
    evaluator = plan_evaluator.PlanEvaluator(cfg, agent)
    with close_expert_environments(plan_evaluator):
        init, goal, expert, expert_success = evaluator.set_episode(cfg, agent, env, seed, task_idx=0)
    rand_vec = np.asarray(unwrapped._last_rand_vec).copy()
    if not np.isfinite(rand_vec).all() or not np.isfinite(np.asarray(evaluator.state_g)).all():
        raise ValueError("Nonfinite prospective input")
    record = {**episode, "initial_sha256": observation_digest(init),
        "goal_sha256": observation_digest(goal), "rand_vec": rand_vec.tolist(),
        "goal_state": np.asarray(evaluator.state_g).tolist(),
        "expert_frames": len(expert), "expert_goal_success": float(expert_success),
        "included_regardless_of_expert_goal_success": True}
    tensors = {"initial": {k: init[k].detach().cpu() for k in ("visual", "proprio")},
               "goal": {k: goal[k].detach().cpu() for k in ("visual", "proprio")},
               "expert_actions": evaluator.expert_actions.detach().cpu()}
    return record, tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=("reach", "reach-wall"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    use_vendor(args.vendor)
    from omegaconf import OmegaConf
    from evals.simu_env_planning.envs.init import make_env

    contract = prepare(args.vendor, args.task)
    schedule = validate_schedule(contract)
    args.output.mkdir(parents=True, exist_ok=False)
    # Persist the exact input policy BEFORE touching any new simulator state.
    write_json(args.output / "input_contract.json", {"contract": contract,
        "role": "candidate_independent_initial_goal_preparation",
        "learned_policy_execution": False, "outcome_filtering": False,
        "fresh_confirmation_claim": False,
        "duplicate_policy": "fail closed and preserve every prepared record; never silently replace episodes"})
    started, records, tensors, hashes = time.monotonic(), [], {}, {}
    try:
        for rank in range(8):
            rows = [r for r in schedule if r["logical_rank"] == rank]
            cfg = OmegaConf.create(contract["config"])
            cfg.local_seed = rows[0]["local_seed"]
            # Released configs use local RNG samplers: numpy/torch retain the
            # entrypoint's global seed 0; only dedicated streams get local_seed.
            global_seed = 0 if cfg.distributed.local_rng_samplers else cfg.local_seed
            random.seed(global_seed)  # deterministic Python RNG; no sampler uses it
            np.random.seed(global_seed)
            torch.manual_seed(global_seed)
            agent = SimpleNamespace(local_generator=torch.Generator().manual_seed(cfg.local_seed))
            env = make_env(cfg)
            try:
                for episode in rows:
                    record, values = prepare_episode(cfg, agent, env, episode)
                    records.append(record)
                    tensors[episode["episode"]] = values
                    write_json(args.output / "progress.json", {"prepared": len(records), "target": 96,
                        "learned_policy_episodes": 0, "seconds": time.monotonic() - started})
            finally:
                env.close()
            path = args.output / f"logical-rank-{rank:02d}.pt"
            torch.save(tensors, path)
            hashes[path.name] = sha256(path)
            tensors.clear()
            print(json.dumps({"task": args.task, "prepared": len(records), "seconds": time.monotonic() - started}), flush=True)
        write_json(args.output / "scenarios.json", records)
        if len({tuple(r["rand_vec"]) for r in records}) != 96:
            raise ValueError("Repeated initial-state family in candidate-independent preparation")
        write_json(args.output / "report.json", {"status": "paper_scale_metaworld_inputs_prepared",
            "task": args.task, "prepared_scenarios": len(records), "unique_initial_state_vectors": 96,
            "input_contract_sha256": sha256(args.output / "input_contract.json"),
            "scenarios_sha256": sha256(args.output / "scenarios.json"), "tensor_files_sha256": hashes,
            "versions": {name: importlib.metadata.version(name) for name in ("metaworld", "mujoco", "torch", "numpy")},
            "seconds": time.monotonic() - started, "learned_policy_episodes": 0,
            "candidate_frozen": False, "fresh_confirmation_claim": False,
            "next_gate": "Bind candidate, controls, inference precision, analysis, and exposure audit before policy outcomes; verify regenerated inputs against these hashes."})
        write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "learned_policy_episodes": 0})
        raise


if __name__ == "__main__":
    main()
