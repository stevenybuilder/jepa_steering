#!/usr/bin/env python3
"""Prepare real Push-T replay trajectories, then collect frozen-model baselines.

Preparation runs without model inference. It reproduces the released benchmark's
30-action dataset replay goal construction and verifies deterministic replay.
Collection caches every simulated frame/state plus planner inputs and activations.
No intervention is applied. Source trajectories, not frames, define the split.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from einops import rearrange
from omegaconf import OmegaConf

from protocol import file_sha256, write_json_atomic

CONFIG = "configs/evals/simu_env_planning/pt/jepa-wm/pt_L2_cem_sourcedset_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml"


def config(repo, output):
    sys.path.insert(0, str(repo.resolve()))
    from evals.simu_env_planning.planning.common.parser import parse_cfg
    cfg = yaml.safe_load((repo / CONFIG).read_text())
    cfg["work_dir"] = str(output)
    cfg["logging"]["optional_plots"] = False
    cfg["logging"]["tqdm_silent"] = True
    cfg["planner"]["decode_each_iteration"] = False
    cfg["planner"]["distribute_planner"] = False
    cfg = parse_cfg(OmegaConf.create(cfg))
    cfg.device = "cuda:0"
    cfg.rank, cfg.world_size, cfg.num_active_gpus = 0, 1, 1
    cfg.active_ranks = [0]
    cfg.local_seed = 90505
    cfg.frameskip = int(cfg.model_kwargs.data.custom.frameskip)
    cfg.action_ratio = cfg.frameskip
    return cfg


def states_of(infos):
    return np.stack([np.asarray(i["state"]).copy() for i in infos])


def save_record(path, value):
    if path.exists():
        raise FileExistsError(f"Preserve existing episode: {path}")
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    tmp.replace(path)
    return {"path": path.name, "sha256": file_sha256(path)}


def prepare(args):
    cfg = config(args.repo, args.output_dir)
    from evals.simu_env_planning.envs.init import make_env
    data = args.dataset
    raw_states = torch.load(data / "states.pth", weights_only=False).float()
    velocities = torch.load(data / "velocities.pth", weights_only=False).float()
    states = torch.cat([raw_states, velocities], dim=-1)
    actions = torch.load(data / "rel_actions.pth", weights_only=False).float() / 100.0
    with (data / "seq_lengths.pkl").open("rb") as f:
        lengths = pickle.load(f)
    shapes = ["T"] * len(states)
    if (data / "shapes.pkl").exists():
        with (data / "shapes.pkl").open("rb") as f:
            shapes = pickle.load(f)
    # Fixed source-ID ordering, no screening by outcome or trajectory difficulty.
    eligible = [i for i, n in enumerate(lengths) if int(n) >= 31]
    split_at = len(eligible) // 2
    manifest = {"stage": "real simulator replay and baseline inputs", "source_dataset": str(data),
                "source_count": len(eligible), "split_rule": "first half of eligible source IDs development; remainder evaluation",
                "config_sha256": file_sha256(args.repo / CONFIG), "episodes": []}
    write_json_atomic(args.output_dir / "manifest.json", manifest)
    env = make_env(cfg)
    for episode, source in enumerate(eligible):
        seed = 2026090600 + episode
        generator = torch.Generator().manual_seed(90600 + episode)
        offset = int(torch.randint(0, int(lengths[source]) - 30, (1,), generator=generator).item())
        init_state = states[source, offset].numpy().copy()
        raw_actions = actions[source, offset:offset + 30].clone()
        env_info = {"shape": shapes[source]}
        env.update_env(env_info)
        obs, infos = env.rollout(seed, init_state, raw_actions, env_info)
        repeat_obs, repeat_infos = env.rollout(seed, init_state, raw_actions, env_info)
        replay_error = float(np.max(np.abs(states_of(infos) - states_of(repeat_infos))))
        if replay_error > 1e-6 or not torch.equal(obs, repeat_obs):
            raise RuntimeError(f"Source {source}: replay mismatch {replay_error}")
        value = {"schema_version": 1, "episode": episode, "source_trajectory": source,
                 "source_offset": offset, "source_length": int(lengths[source]),
                 "split": "development" if episode < split_at else "evaluation",
                 "environment_seed": seed, "planner_seed": 91600 + episode,
                 "initial_state": init_state, "env_info": env_info,
                 "expert_actions_raw": raw_actions, "expert_observations": obs.cpu().to(torch.uint8),
                 "expert_states": states_of(infos), "expert_proprios": torch.stack([torch.as_tensor(i["proprio"]) for i in infos]),
                 "goal_state": np.asarray(infos[-1]["state"]).copy(),
                 "deterministic_replay_max_state_error": replay_error,
                 "deterministic_replay_pixels_equal": True}
        receipt = save_record(args.output_dir / f"input-{episode:03d}.pt", value)
        manifest["episodes"].append({**receipt, "episode": episode, "source_trajectory": source,
                                     "split": value["split"]})
        write_json_atomic(args.output_dir / "manifest.json", manifest)
        print(json.dumps({"event": "replay_prepared", "episode": episode, "source": source, "state_error": replay_error}), flush=True)
    env.close()
    write_json_atomic(args.output_dir / "DONE.json", {**manifest, "complete": True})


def snapshot(env):
    base = env.unwrapped
    result = {"elapsed_steps": int(env.elapsed_steps()), "bodies": {}}
    for name in ("agent", "block"):
        body = getattr(base, name)
        result["bodies"][name] = {"position": list(body.position), "velocity": list(body.velocity),
                                    "angle": float(body.angle), "angular_velocity": float(body.angular_velocity),
                                    "force": list(body.force), "torque": float(body.torque)}
    result["restoration_contract"] = "reset initial state and replay cached actions; direct body restore not certified"
    return result


@torch.no_grad()
def collect(args):
    cfg = config(args.repo, args.output_dir)
    from evals.simu_env_planning.envs.init import make_env
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    from evals.simu_env_planning.planning.utils import make_td, set_seed
    from model_loader import load_headless
    from collect_on_policy_bank import chosen_plan_trace
    set_seed(cfg.local_seed)
    model, preprocessor, provenance = load_headless(args.repo, model_name="jepa_wm_pusht", checkpoint_override=args.checkpoint)
    env = make_env(cfg)
    agent = GC_Agent(cfg, model, dset=None, preprocessor=preprocessor)
    receipts = []
    for episode in args.episodes:
        path = args.inputs / f"input-{episode:03d}.pt"
        bank = torch.load(path, map_location="cpu", weights_only=False)
        start = time.monotonic()
        env.update_env(bank["env_info"])
        obs, info = env.prepare(bank["environment_seed"], bank["initial_state"], bank["env_info"])
        if not torch.equal(obs.cpu(), bank["expert_observations"][0]):
            raise RuntimeError("Initial image does not match prepared replay")
        if not np.allclose(np.asarray(info["state"]), bank["expert_states"][0], atol=1e-6, rtol=0):
            raise RuntimeError("Initial state does not match prepared replay")
        td = make_td(obs, info)
        goal = make_td(bank["expert_observations"][-1], {"proprio": bank["expert_proprios"][-1]})
        agent.set_goal(goal)
        agent.local_gpu_generator.manual_seed(bank["planner_seed"])
        observations, physical_states, proprios = [obs.cpu()], [np.asarray(info["state"]).copy()], [torch.as_tensor(info["proprio"]).cpu()]
        all_actions, replans, rewards = [], [], []
        print(json.dumps({"event": "episode_start", "episode": episode, "split": bank["split"]}), flush=True)
        while env.steps_left() > 0:
            rng = agent.local_gpu_generator.get_state().cpu()
            z = model.encode(td.to(agent.device).unsqueeze(0), act=True)
            physics = snapshot(env)
            planned = agent.plan(z, steps_left=max((env.steps_left() + 1) * model.action_skip // cfg.frameskip, 1))
            raw = preprocessor.denormalize_actions(rearrange(planned.cpu(), "t (f d) -> (t f) d", d=env.action_dim))
            trace = chosen_plan_trace(model, z, planned, 3)
            step_obs, step_rewards, dones, infos = env.step_multiple(raw)
            replans.append({"elapsed_steps": physics["elapsed_steps"], "simulator": physics,
                            "planner_rng_before": rng, "encoded_visual": z["visual"].detach().cpu().half(),
                            "encoded_proprio": z["proprio"].detach().cpu().half(),
                            "planned_actions_normalized": planned.cpu(), "planned_actions_raw": raw,
                            "planner_losses": agent._prev_losses.cpu(), "chosen_plan_trace": trace,
                            "executed_action_count": len(step_obs)})
            observations.extend([o.cpu() for o in step_obs])
            physical_states.extend([np.asarray(i["state"]).copy() for i in infos])
            proprios.extend([torch.as_tensor(i["proprio"]).cpu() for i in infos])
            rewards.extend([float(r) for r in step_rewards])
            all_actions.append(raw[:len(step_obs)])
            td = make_td(step_obs[-1], infos[-1])
            print(json.dumps({"event": "replan", "episode": episode, "steps": int(env.elapsed_steps()), "seconds": time.monotonic() - start}), flush=True)
            if bool(dones[-1]):
                break
        # Evaluate against the replayed goal, not the fixed painted target.
        metrics = env.eval_state(bank["goal_state"], physical_states[-1])
        step_metrics = [env.eval_state(bank["goal_state"], s) for s in physical_states]
        action_tensor = torch.cat(all_actions)
        replay_obs, replay_infos = env.rollout(bank["environment_seed"], bank["initial_state"], action_tensor, bank["env_info"])
        state_error = float(np.max(np.abs(states_of(replay_infos) - np.stack(physical_states))))
        pixels_equal = torch.equal(replay_obs.cpu(), torch.stack(observations))
        if state_error > 1e-6 or not pixels_equal:
            raise RuntimeError(f"Baseline replay verification failed: {state_error}, pixels={pixels_equal}")
        result = {"schema_version": 1, "task": "pusht-base", "arm": "unsteered_frozen_jepa_wm",
                  "episode": episode, "split": bank["split"], "source_trajectory": bank["source_trajectory"],
                  "input_sha256": file_sha256(path), "environment_seed": bank["environment_seed"],
                  "planner_seed": bank["planner_seed"], "observations": torch.stack(observations).to(torch.uint8),
                  "states": np.stack(physical_states), "proprios": torch.stack(proprios),
                  "actions_raw": action_tensor, "rewards": rewards, "replans": replans,
                  "goal_state": bank["goal_state"], "initial_state": bank["initial_state"], "env_info": bank["env_info"],
                  "final_success": bool(metrics["success"]), "final_state_distance": float(metrics["state_dist"]),
                  "step_goal_distance": np.asarray([m["state_dist"] for m in step_metrics]),
                  "step_goal_success": np.asarray([m["success"] for m in step_metrics]),
                  "baseline_replay_max_state_error": state_error, "baseline_replay_pixels_equal": pixels_equal,
                  "duration_seconds": time.monotonic() - start, "model_provenance": provenance}
        receipt = save_record(args.output_dir / f"episode-{episode:03d}.pt", result)
        receipts.append({**receipt, "episode": episode, "split": bank["split"]})
        print(json.dumps({"event": "episode_saved", "episode": episode, "seconds": result["duration_seconds"], "replay_state_error": state_error}), flush=True)
    env.close()
    write_json_atomic(args.output_dir / "DONE.json", {"complete": True, "episodes": args.episodes, "outputs": receipts, "model_provenance": provenance})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "collect"))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, nargs="+")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare":
        if args.dataset is None:
            parser.error("prepare requires --dataset")
        prepare(args)
    else:
        if args.inputs is None or not args.episodes:
            parser.error("collect requires --inputs and --episodes")
        collect(args)


if __name__ == "__main__":
    main()
