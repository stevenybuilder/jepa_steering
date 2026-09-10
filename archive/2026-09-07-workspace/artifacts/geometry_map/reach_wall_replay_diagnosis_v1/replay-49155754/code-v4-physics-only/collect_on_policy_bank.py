#!/usr/bin/env python3
"""Collect reproducible unsteered JEPA-WM episodes and replan-state caches.

This uses the released headless JEPA-WM and the official Reach-Wall environment,
expert-goal construction, CEM planner, and action stepping.  Each episode receives
an explicit environment seed and an explicit planner seed so it can later be paired
exactly across unsteered, sham, and steering arms.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from einops import rearrange
from omegaconf import OmegaConf

from model_loader import load_headless_metaworld
from protocol import file_sha256, write_json_atomic


def first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return first_tensor(item)
            except TypeError:
                pass
    if isinstance(value, dict):
        for item in value.values():
            try:
                return first_tensor(item)
            except TypeError:
                pass
    raise TypeError(f"No tensor in hook output of type {type(value)!r}")


def block_list(
    root: torch.nn.Module, paths: Iterable[str], expected: int, label: str
) -> tuple[str, list[torch.nn.Module]]:
    for path in paths:
        value: Any = root
        try:
            for part in path.split("."):
                value = getattr(value, part)
        except AttributeError:
            continue
        if isinstance(value, (torch.nn.ModuleList, list, tuple)) and len(value) == expected:
            return path, list(value)
    candidates = [
        (name, len(module))
        for name, module in root.named_modules()
        if isinstance(module, torch.nn.ModuleList)
    ]
    raise RuntimeError(f"Could not identify {expected} {label} blocks; ModuleLists={candidates}")


def tensor_cpu(value: torch.Tensor, *, half: bool = False) -> torch.Tensor:
    value = value.detach().cpu()
    return value.half() if half and value.is_floating_point() else value


def physics_snapshot(env: Any) -> dict[str, Any]:
    """Capture MuJoCo physics plus MetaWorld task variables needed for restoration."""
    base = env.proprio_env.unwrapped
    data = base.data
    arrays = {}
    for name in ("qpos", "qvel", "act", "mocap_pos", "mocap_quat", "userdata"):
        value = getattr(data, name, None)
        if value is not None:
            arrays[name] = np.asarray(value).copy()
    task_state = {}
    for name in (
        "_last_rand_vec",
        "_target_pos",
        "obj_init_pos",
        "hand_init_pos",
        "seeded_rand_vec",
        "_freeze_rand_vec",
    ):
        value = getattr(base, name, None)
        if isinstance(value, np.ndarray):
            task_state[name] = value.copy()
        elif isinstance(value, (bool, int, float, str)):
            task_state[name] = value
    return {
        "elapsed_steps": int(env.elapsed_steps()),
        "time": float(data.time),
        "physics": arrays,
        "task_state": task_state,
    }


@torch.no_grad()
def chosen_plan_trace(model: torch.nn.Module, z: Any, actions: torch.Tensor, block: int) -> dict[str, Any]:
    """Rerun only the chosen plan and cache pooled block/output trajectories."""
    _path, blocks = block_list(model.model.predictor, ("predictor_blocks", "blocks"), 6, "predictor")
    captured = []

    def record(_module: torch.nn.Module, _args: tuple[Any, ...], output: Any) -> None:
        value = first_tensor(output).detach()
        # The current predicted frame occupies the last 16x16 spatial tokens.
        pooled = value.reshape(value.shape[0], -1, value.shape[-1])[:, -256:].mean(dim=1)
        captured.append(tensor_cpu(pooled, half=True))

    handle = blocks[block].register_forward_hook(record)
    try:
        rollout = model.unroll(z, act_suffix=actions.unsqueeze(1))
    finally:
        handle.remove()
    visual = rollout["visual"] if isinstance(rollout, (dict,)) or hasattr(rollout, "keys") else rollout
    visual = visual.reshape(visual.shape[0], visual.shape[1], -1, visual.shape[-1]).mean(dim=2)
    return {
        "block": block,
        "block_pooled": torch.cat(captured, dim=0),
        "predicted_visual_pooled": tensor_cpu(visual, half=True),
    }


def initialize_episode(cfg: Any, agent: Any, env: Any, evaluator: Any, episode: int, env_seed: int) -> tuple:
    # Mirror the official PlanEvaluator.eval initialization exactly through set_episode.
    _initial, info = env.reset(seed=env_seed, task_idx=0)
    unwrapped = env.proprio_env.unwrapped
    unwrapped._freeze_rand_vec = False
    unwrapped.seeded_rand_vec = True
    env.seed(env_seed)
    init_obs, goal_obs, expert_obses, expert_success = evaluator.set_episode(
        cfg, agent, env, env_seed, task_idx=0
    )
    if goal_obs["visual"].shape[0] > agent.model.tubelet_size_enc:
        goal_obs["visual"] = goal_obs["visual"][-agent.model.tubelet_size_enc :]
    agent.set_goal(goal_obs)
    return init_obs, info, goal_obs, expert_success, expert_obses


@torch.no_grad()
def run_episode(
    cfg: Any,
    agent: Any,
    env: Any,
    evaluator: Any,
    episode: int,
    env_seed: int,
    planner_seed: int,
    causal_block: int,
) -> dict[str, Any]:
    agent.local_gpu_generator.manual_seed(planner_seed)
    init_obs, info, goal_obs, expert_success, expert_obses = initialize_episode(
        cfg, agent, env, evaluator, episode, env_seed
    )
    goal_encoding = agent.goal_state_enc
    td = init_obs
    done = False
    total_reward = 0.0
    replans = []
    all_actions = []
    all_infos = []
    ever_success = False
    started = time.monotonic()
    while not done:
        steps_left = max((env.steps_left() + 1) * agent.model.action_skip // cfg.frameskip, 1)
        rng_before = agent.local_gpu_generator.get_state().cpu()
        model_obs = td.to(agent.device, non_blocking=True).unsqueeze(0)
        z = agent.model.encode(model_obs, act=True)
        planned = agent.plan(z, steps_left=steps_left)
        trace = chosen_plan_trace(agent.model, z, planned, causal_block)
        raw_actions = rearrange(planned.cpu(), "t (f d) -> (t f) d", d=env.action_dim)
        raw_actions = agent.preprocessor.denormalize_actions(raw_actions)
        replan = {
            "replan_index": len(replans),
            "elapsed_steps": int(env.elapsed_steps()),
            "steps_left_model": int(steps_left),
            "observation_visual": tensor_cpu(td["visual"]).to(torch.uint8),
            "observation_proprio": tensor_cpu(td["proprio"]),
            "encoded_visual": tensor_cpu(z["visual"], half=True),
            "encoded_proprio": tensor_cpu(z["proprio"], half=True),
            "planner_rng_before": rng_before,
            "planned_actions_normalized": tensor_cpu(planned),
            "planned_actions_raw": tensor_cpu(raw_actions),
            "planner_losses": tensor_cpu(agent._prev_losses),
            "planner_elite_mean": tensor_cpu(agent._prev_elite_losses_mean),
            "planner_elite_std": tensor_cpu(agent._prev_elite_losses_std),
            "chosen_plan_trace": trace,
            "simulator": physics_snapshot(env),
            "state": np.asarray(info["state"]).copy(),
        }
        observations, rewards, dones, infos = env.step_multiple(raw_actions)
        replan["executed_action_count"] = len(observations)
        replans.append(replan)
        all_actions.append(raw_actions[: len(observations)].clone())
        all_infos.extend(infos)
        total_reward += sum(float(reward) for reward in rewards)
        ever_success = ever_success or any(bool(step_info["success"]) for step_info in infos)
        done = bool(dones[-1])
        print(
            json.dumps(
                {
                    "event": "replan",
                    "episode": episode,
                    "replan": len(replans),
                    "environment_steps": int(env.elapsed_steps()),
                    "ever_success": ever_success,
                    "elapsed_seconds": time.monotonic() - started,
                }
            ),
            flush=True,
        )
        td = __import__("evals.simu_env_planning.planning.utils", fromlist=["make_td"]).make_td(
            observations[-1], infos[-1]
        )
        info = infos[-1]
    final_success = bool(all_infos[-1]["success"])
    final_state = np.asarray(all_infos[-1]["state"]).copy()
    state_goal = np.asarray(evaluator.state_g).copy()
    return {
        "schema_version": 1,
        "episode": episode,
        "environment_seed": env_seed,
        "planner_seed": planner_seed,
        "task": cfg.task_specification.task,
        "arm": "unsteered_frozen_jepa_wm",
        "expert_success": bool(expert_success),
        "final_success": final_success,
        "ever_success": ever_success,
        "total_reward": total_reward,
        "final_state_distance_to_expert_goal": float(np.linalg.norm(final_state - state_goal)),
        "environment_steps": int(env.elapsed_steps()),
        "replan_count": len(replans),
        "duration_seconds": time.monotonic() - started,
        "goal": {
            "visual": tensor_cpu(goal_obs["visual"]).to(torch.uint8),
            "proprio": tensor_cpu(goal_obs["proprio"]),
            "encoded_visual": tensor_cpu(goal_encoding["visual"], half=True),
            "encoded_proprio": tensor_cpu(goal_encoding["proprio"], half=True),
            "state": state_goal,
        },
        "actions_raw": torch.cat(all_actions, dim=0),
        "replans": replans,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--environment-seed-base", type=int, default=2_026_090_500)
    parser.add_argument("--planner-seed-base", type=int, default=90_500)
    parser.add_argument("--causal-block", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    wm, preprocessor, provenance = load_headless_metaworld(args.repo, device=args.device)
    sys.path.insert(0, str(args.repo.resolve()))
    from evals.simu_env_planning.envs.init import make_env
    from evals.simu_env_planning.planning.common.parser import parse_cfg
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    from evals.simu_env_planning.planning.plan_evaluator import PlanEvaluator
    from evals.simu_env_planning.planning.utils import set_seed

    raw_cfg = yaml.safe_load(args.config.read_text())
    raw_cfg["logging"]["optional_plots"] = False
    raw_cfg["logging"]["tqdm_silent"] = True
    raw_cfg["planner"]["decode_each_iteration"] = False
    raw_cfg["work_dir"] = str(args.output_dir)
    cfg = parse_cfg(OmegaConf.create(raw_cfg))
    cfg.device = args.device
    cfg.rank = 0
    cfg.world_size = 1
    cfg.num_active_gpus = 1
    cfg.active_ranks = [0]
    cfg.local_seed = int(raw_cfg["meta"]["seed"])
    cfg.frameskip = int(raw_cfg["model_kwargs"]["data"]["custom"]["frameskip"])
    cfg.action_ratio = cfg.frameskip // wm.action_skip
    cfg.work_dir = args.output_dir
    cfg.planner.distribute_planner = False
    set_seed(cfg.local_seed)
    env = make_env(cfg)
    agent = GC_Agent(cfg, wm, dset=None, preprocessor=preprocessor)
    evaluator = PlanEvaluator(cfg, agent)
    outputs = []
    started = time.monotonic()
    for index, episode in enumerate(args.episodes):
        env_seed = args.environment_seed_base + episode
        planner_seed = args.planner_seed_base + episode
        result = run_episode(
            cfg, agent, env, evaluator, episode, env_seed, planner_seed, args.causal_block
        )
        path = args.output_dir / f"episode-{episode:03d}.pt"
        torch.save(result, path)
        outputs.append(
            {
                "path": path.name,
                "sha256": file_sha256(path),
                "episode": episode,
                "environment_seed": env_seed,
                "planner_seed": planner_seed,
                "final_success": result["final_success"],
                "ever_success": result["ever_success"],
                "replan_count": result["replan_count"],
                "duration_seconds": result["duration_seconds"],
            }
        )
        print(json.dumps({"event": "episode", "completed": index + 1, **outputs[-1]}), flush=True)
    receipt = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete": len(outputs) == len(args.episodes),
        "scope": "unsteered frozen JEPA-WM on-policy cache",
        "task": "mw-reach-wall",
        "model_provenance": provenance,
        "config": str(args.config),
        "config_sha256": file_sha256(args.config),
        "causal_block": args.causal_block,
        "episode_count": len(outputs),
        "episodes": args.episodes,
        "duration_seconds": time.monotonic() - started,
        "outputs": outputs,
    }
    write_json_atomic(args.output_dir / "DONE.json", receipt)
    print(json.dumps({key: value for key, value in receipt.items() if key != "outputs"}, indent=2))


if __name__ == "__main__":
    main()
