#!/usr/bin/env python3
"""Development-only subspace -> native CEM -> actual simulator fork diagnostic.

Existing banks are immutable. Reconstruct each start with its original reset and
saved action prefix, then require matching pixels, proprioception, and physics.
Apply a fitted subspace counterfactual at only the first imagined predictor step.
Compare identical initial CEM candidates, selected plans, and 15-step simulator
consequences. These are short causal forks, NOT full-episode efficacy results.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from einops import rearrange
from omegaconf import OmegaConf

from collect_on_policy_bank import block_list, initialize_episode, physics_snapshot
from model_loader import load_headless_metaworld
from protocol import file_sha256, write_json_atomic
from subspace_control import FirstStepSubspacePatch


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}), flush=True)


def checked_max_error(a, b, name, tolerance=1e-6):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise RuntimeError(f"{name} shape mismatch: {a.shape} vs {b.shape}")
    error = float(np.max(np.abs(a.astype(float) - b.astype(float)))) if a.size else 0.0
    if not np.isfinite(error) or error > tolerance:
        raise RuntimeError(f"{name} mismatch {error} > {tolerance}; refusing paired interpretation")
    return error


def load_development_bank(path):
    """Reject a held-out receipt before any tensor file is opened."""
    receipt = json.loads((path.parent / "DONE.json").read_text())
    if not receipt.get("complete"):
        raise RuntimeError("Baseline receipt is incomplete")
    matches = [row for row in receipt["outputs"] if row["path"] == path.name]
    if len(matches) != 1:
        raise RuntimeError("Baseline must have exactly one matching receipt entry")
    entry = matches[0]
    if not 0 <= int(entry["episode"]) < 12:
        raise RuntimeError("Development runner cannot open on-policy confirmation episodes 12-23")
    if file_sha256(path) != entry["sha256"]:
        raise RuntimeError("Baseline cache hash mismatch")
    bank = torch.load(path, map_location="cpu", weights_only=False)
    if int(bank["episode"]) != int(entry["episode"]):
        raise RuntimeError("Baseline episode disagrees with receipt")
    return bank, entry["sha256"]


def counterfactual_actions(snapshot):
    raw = torch.zeros(6, 5, 4)
    baseline = snapshot["planned_actions_raw"][:5]
    if baseline.shape != (5, 4):
        raise RuntimeError("Expected five raw four-dimensional actions")
    raw[:, :, 3] = baseline[:, 3]  # keep the gripper matched for every candidate
    raw[0] = baseline
    raw[2, :, 0], raw[3, :, 0] = 1, -1
    raw[4, :, 2], raw[5, :, 2] = 1, -1
    return raw


def visual_pooled(prediction, actions):
    visual = prediction["visual"]
    # The released wrapper prepends the one observed context frame to its futures.
    expected = (actions.shape[0] + 1, actions.shape[1])
    if visual.ndim < 4 or visual.shape[:2] != expected:
        raise RuntimeError(f"Expected [time,batch,...,dim] predictions, got {visual.shape} for {actions.shape}")
    return visual[1:].reshape(actions.shape[0], actions.shape[1], -1, visual.shape[-1]).mean(2)


def replay_action_count(bank, replan_index):
    # elapsed_steps includes the reset_warmup step, absent from actions_raw.
    return sum(int(row["executed_action_count"]) for row in bank["replans"][:replan_index])


def restore_cached_goal(goal, bank_goal):
    """Copy the original raw fixed goal stimulus without aliasing its cache."""
    restored = goal.clone() if hasattr(goal, "clone") else dict(goal)
    for name in ("visual", "proprio"):
        restored[name] = bank_goal[name].to(goal[name]).clone()
    return restored


def setup_cfg(path, output, wm):
    from evals.simu_env_planning.planning.common.parser import parse_cfg
    raw = yaml.safe_load(path.read_text())
    raw["logging"]["optional_plots"] = False
    raw["logging"]["tqdm_silent"] = True
    raw["planner"]["decode_each_iteration"] = False
    raw["work_dir"] = str(output)
    cfg = parse_cfg(OmegaConf.create(raw))
    cfg.device, cfg.rank, cfg.world_size, cfg.num_active_gpus = "cuda:0", 0, 1, 1
    cfg.active_ranks = [0]
    cfg.local_seed = int(raw["meta"]["seed"])
    cfg.frameskip = int(raw["model_kwargs"]["data"]["custom"]["frameskip"])
    cfg.action_ratio = cfg.frameskip // wm.action_skip
    cfg.work_dir = output
    cfg.planner.distribute_planner = False
    if cfg.planner.iterations != 15 or cfg.planner.num_samples != 300 or cfg.planner.horizon != 6:
        raise RuntimeError("Expected released Reach-Wall H6/300/15 CEM configuration")
    return cfg


def reconstruct(cfg, wm, preprocessor, bank, replan_index, cached_goal=False):
    from evals.simu_env_planning.envs.init import make_env
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    from evals.simu_env_planning.planning.plan_evaluator import PlanEvaluator
    from evals.simu_env_planning.planning.utils import make_td, set_seed
    set_seed(cfg.local_seed)
    env = make_env(cfg)
    agent = GC_Agent(cfg, wm, dset=None, preprocessor=preprocessor)
    evaluator = PlanEvaluator(cfg, agent)
    td, _, goal, _, _ = initialize_episode(cfg, agent, env, evaluator,
                                         bank["episode"], bank["environment_seed"])
    generated_goal_error = float(np.max(np.abs(goal["visual"].cpu().numpy().astype(float) - np.asarray(bank["goal"]["visual"]).astype(float))))
    agent.replay_goal_metadata = {
        "mode": "immutable_cached_raw" if cached_goal else "regenerate_expert_goal",
        "regenerated_goal_pixels_max_error": generated_goal_error,
    }
    if cached_goal:
        goal = restore_cached_goal(goal, bank["goal"])
        agent.set_goal(goal)
        for name in ("visual", "proprio"):
            cached_encoding = bank["goal"]["encoded_" + name]
            encoded = agent.goal_state_enc[name].detach().cpu().to(cached_encoding.dtype)
            checked_max_error(encoded, cached_encoding, f"cached goal encoded {name} at saved dtype", 0)
        agent.replay_goal_metadata["encoded_goal_check"] = "exact at saved cache dtype (float16)"
        emit("cached_goal_restored", **agent.replay_goal_metadata)
    snapshot = bank["replans"][replan_index]
    elapsed = int(snapshot["elapsed_steps"])
    prefix_steps = replay_action_count(bank, replan_index)
    if prefix_steps:
        observations, _, _, infos = env.step_multiple(bank["actions_raw"][:prefix_steps])
        if len(observations) != prefix_steps:
            raise RuntimeError("Action-prefix replay ended before the target snapshot")
        td = make_td(observations[-1], infos[-1])
    now = physics_snapshot(env)
    emit("replay_start_errors",
         pixels=float(np.max(np.abs(td["visual"].cpu().numpy().astype(float) - np.asarray(snapshot["observation_visual"]).astype(float)))),
         proprio=float(np.max(np.abs(td["proprio"].cpu().numpy() - np.asarray(snapshot["observation_proprio"])))),
         qpos=float(np.max(np.abs(now["physics"]["qpos"] - snapshot["simulator"]["physics"]["qpos"]))),
         goal_pixels=float(np.max(np.abs(goal["visual"].cpu().numpy().astype(float) - np.asarray(bank["goal"]["visual"]).astype(float)))))
    checks = {
        "pixels": checked_max_error(td["visual"].cpu(), snapshot["observation_visual"], "start pixels", 0),
        "proprio": checked_max_error(td["proprio"].cpu(), snapshot["observation_proprio"], "start proprio"),
        "goal_pixels": checked_max_error(goal["visual"].cpu(), bank["goal"]["visual"], "goal pixels", 0),
        "goal_proprio": checked_max_error(goal["proprio"].cpu(), bank["goal"]["proprio"], "goal proprio"),
    }
    if now["elapsed_steps"] != elapsed:
        raise RuntimeError("Elapsed-step mismatch")
    for name, value in snapshot["simulator"]["physics"].items():
        checks[name] = checked_max_error(now["physics"][name], value, name)
    checks["time"] = checked_max_error(now["time"], snapshot["simulator"]["time"], "physics time")
    for name, value in snapshot["simulator"]["task_state"].items():
        other = now["task_state"].get(name)
        if isinstance(value, str):
            if other != value:
                raise RuntimeError(f"Task state mismatch {name}")
        else:
            checks[f"task.{name}"] = checked_max_error(other, value, name)
    agent.local_gpu_generator.set_state(snapshot["planner_rng_before"])
    z = wm.encode(td.to(agent.device).unsqueeze(0), act=True)
    return env, agent, z, checks


def close_env(env):
    if hasattr(env, "close"):
        env.close()


def execute_fork(env, raw_actions):
    observations, rewards, dones, infos = env.step_multiple(raw_actions.cpu())
    if not observations or not all(isinstance(frame, torch.Tensor) for frame in observations):
        raise RuntimeError("Expected a nonempty list of simulator image tensors")
    states = np.stack([np.asarray(info["state"]) for info in infos])
    return {
        "states": states, "frames": torch.stack(observations).cpu().to(torch.uint8),
        "actions": raw_actions[:len(observations)].cpu(), "rewards": np.asarray(rewards),
        "steps": len(observations), "done": bool(dones[-1]),
        "end_success": bool(infos[-1]["success"]),
        "end_goal_distance": float(np.linalg.norm(states[-1, :3] - states[-1, -3:])),
        "end_hand_xyz": states[-1, :3].tolist(),
    }


@torch.no_grad()
def run(args):
    if args.replan != 0:
        raise RuntimeError("This diagnostic supports replan 0 only; later replans require restoring CEM warm-start state")
    bank, expected = load_development_bank(args.bank)
    candidate_receipt = json.loads((args.candidate.parent / "DONE.json").read_text())
    if not candidate_receipt.get("complete") or file_sha256(args.candidate) != candidate_receipt["sha256"]:
        raise RuntimeError("Candidate hash mismatch")
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=False)
    wm, preprocessor, provenance = load_headless_metaworld(args.repo)
    for parameter in wm.parameters():
        parameter.requires_grad_(False)
    cfg = setup_cfg(args.config, args.output_dir, wm)
    _, blocks = block_list(wm.model.predictor, ("predictor_blocks", "blocks"), 6, "predictor")
    block = blocks[int(candidate["block"])]
    basis, scale = candidate["basis"].cuda(), candidate["scale"].cuda()
    snapshot = bank["replans"][args.replan]
    protocol = {
        "status": "development_causal_diagnostic_not_steering_efficacy",
        "bank_sha256": expected, "candidate_sha256": file_sha256(args.candidate),
        "script_sha256": file_sha256(Path(__file__)), "model": provenance,
        "episode": bank["episode"], "replan": args.replan, "beta": args.beta,
        "block": int(candidate["block"]), "rank": basis.shape[1], "imagined_step": 0,
        "counterfactual_model_steps": 1, "counterfactual_raw_steps": 5,
        "native_cem": {"horizon": 6, "samples": 300, "iterations": 15},
        "activation_support_scope": "first 300 candidates for identity and edited arms; unsteered reference has no hook",
        "arms": ["unsteered", "identity", "subspace_up", "sham_up", "subspace_down", "sham_down"],
        "same_start_rule": "original reset + saved action prefix; exact pixels and checked physics",
        "not_yet_tested": ["unrelated-site and time-shift controls", "held-out confirmation", "full-episode task success"],
    }
    write_json_atomic(args.output_dir / "protocol.json", protocol)
    env, agent, z, checks = reconstruct(cfg, wm, preprocessor, bank, args.replan, cached_goal=args.cached_goal)
    protocol["goal_replay"] = agent.replay_goal_metadata
    write_json_atomic(args.output_dir / "protocol.json", protocol)
    emit("replay_verified", checks=checks)
    # Six physical counterfactuals at one identical state. No outcome screening.
    raw = counterfactual_actions(snapshot)
    names = ["baseline_first_chunk", "hold", "plus_x", "minus_x", "plus_z", "minus_z"]
    normalized = preprocessor.normalize_actions(raw.reshape(-1, 4)).reshape(6, 20).cuda()
    action_plans = normalized.unsqueeze(0)  # [one model step, six candidates, 20]
    donor_outputs = []

    def capture_donor(_module, _args, output):
        if output.ndim != 3 or output.shape[:2] != (6, 256):
            raise RuntimeError(f"Expected donor block [6,256,dim], got {output.shape}")
        donor_outputs.append(output.detach().float().mean(1))

    handle = block.register_forward_hook(capture_donor)
    try:
        predicted = wm.unroll(z, act_suffix=action_plans)
    finally:
        handle.remove()
    if len(donor_outputs) != 1:
        raise RuntimeError(f"Expected one imagined step, got {len(donor_outputs)}")
    visual_pooled(predicted, action_plans)  # enforce latent layout before comparing costs
    baseline_costs = agent.objective(predicted, action_plans).detach().cpu()
    donors = {"up": donor_outputs[0][4:5], "down": donor_outputs[0][5:6]}
    del donor_outputs
    close_env(env)
    counterfactuals = []
    for index, name in enumerate(names):
        env, _, _, replay_checks = reconstruct(cfg, wm, preprocessor, bank, args.replan, cached_goal=args.cached_goal)
        result = execute_fork(env, raw[index])
        counterfactuals.append({"name": name, "replay_checks": replay_checks, **result})
        close_env(env)
    torch.save({"candidates": counterfactuals, "native_costs": baseline_costs,
                "prediction": predicted.detach().cpu(), "normalized_actions": action_plans.cpu(),
                "donors": {k:v.cpu() for k,v in donors.items()}}, args.output_dir / "counterfactuals.pt")
    emit("six_same_state_counterfactuals_complete")

    results, summary = {}, []
    for arm in protocol["arms"]:
        started = time.monotonic()
        env, agent, z, replay_checks = reconstruct(cfg, wm, preprocessor, bank, args.replan, cached_goal=args.cached_goal)
        donor = donors["down" if arm.endswith("down") else "up"]
        beta = 0.0 if arm in ("unsteered", "identity") else args.beta
        native_unroll = agent.planner.unroll
        patch = FirstStepSubspacePatch(block, native_unroll, basis, scale, donor,
                                      beta=beta, sham=arm.startswith("sham"))
        first_candidates = {}

        def tracked_unroll(*positional, **kwargs):
            actions = kwargs.get("act_suffix", positional[1] if len(positional) > 1 else None)
            prediction = native_unroll(*positional, **kwargs) if arm == "unsteered" else patch.unroll(*positional, **kwargs)
            if not first_candidates and actions is not None and actions.shape[1] == 300:
                first_candidates["actions"] = actions.detach().cpu()
                first_candidates["costs"] = agent.objective(prediction, actions).detach().cpu()
                first_candidates["predicted_visual_pooled"] = visual_pooled(prediction, actions).detach().cpu()
                first_candidates["predicted_proprio"] = prediction["proprio"].detach().cpu()
            return prediction

        agent.planner.unroll = tracked_unroll
        emit("native_cem_started", arm=arm)
        if arm == "unsteered":
            planned = agent.plan(z, steps_left=snapshot["steps_left_model"])
        else:
            with patch:
                planned = agent.plan(z, steps_left=snapshot["steps_left_model"])
        if not first_candidates:
            raise RuntimeError("No 300-candidate CEM bank captured")
        if arm == "unsteered":
            checked_max_error(planned.cpu(), snapshot["planned_actions_normalized"], "cached native plan", 1e-5)
        else:
            checked_max_error(first_candidates["actions"], results["unsteered"]["first_candidates"]["actions"], "paired initial candidates", 0)
        if arm == "identity":
            checked_max_error(planned.cpu(), results["unsteered"]["plan"], "identity plan", 0)
            checked_max_error(first_candidates["costs"], results["unsteered"]["first_candidates"]["costs"], "identity costs", 0)
        raw_actions = preprocessor.denormalize_actions(rearrange(planned.cpu(), "t (f d) -> (t f) d", d=4))
        fork = execute_fork(env, raw_actions)
        close_env(env)
        result = {"arm": arm, "plan": planned.cpu(), "first_candidates": first_candidates,
                  "selected_prediction": agent._predicted_best_encs_over_iterations[-1].detach().cpu(),
                  "replay_checks": replay_checks, "fork": fork, "edit": patch.first_edit,
                  "activation_support": patch.first_activation,
                  "goal_replay": agent.replay_goal_metadata,
                  "hook_calls": patch.calls, "seconds": time.monotonic() - started}
        if arm == "identity":
            checked_max_error(fork["states"], results["unsteered"]["fork"]["states"], "identity behavior", 0)
            checked_max_error(fork["frames"], results["unsteered"]["fork"]["frames"], "identity pixels", 0)
        baseline = result if arm == "unsteered" else results["unsteered"]
        costs, base_costs = first_candidates["costs"], baseline["first_candidates"]["costs"]
        row = {"arm": arm, "episode": bank["episode"], "replan": args.replan,
               "first_candidate_argmin": int(costs.argmin()),
               "candidate_rank_changes": int((costs.argsort().argsort() != base_costs.argsort().argsort()).sum()),
               "candidate_cost_l2_change": float(torch.linalg.vector_norm(costs - base_costs)),
               "predicted_visual_pooled_l2_change": float(torch.linalg.vector_norm(
                   first_candidates["predicted_visual_pooled"] - baseline["first_candidates"]["predicted_visual_pooled"])),
               "chosen_raw_action_l2_change": float(torch.linalg.vector_norm(raw_actions - baseline["fork"]["actions"])),
               "end_hand_l2_change": float(np.linalg.norm(np.asarray(fork["end_hand_xyz"]) - baseline["fork"]["end_hand_xyz"])),
               "end_goal_distance": fork["end_goal_distance"], "short_fork_success": fork["end_success"],
               "steps": fork["steps"], "seconds": result["seconds"]}
        results[arm] = result
        summary.append(row)
        torch.save(result, args.output_dir / f"{arm}.pt")
        write_json_atomic(args.output_dir / "progress.json", {"complete": False, "rows": summary})
        emit("arm_complete", **row)
    write_json_atomic(args.output_dir / "DONE.json", {
        **protocol, "complete": True, "rows": summary,
        "outputs": [{"path": p.name, "sha256": file_sha256(p)} for p in sorted(args.output_dir.glob("*.pt"))],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("repo", "config", "bank", "candidate", "output-dir"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--replan", type=int, default=0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--cached-goal", action="store_true", help="Restore the immutable raw benchmark goal for every arm")
    args = parser.parse_args()
    # Exclusive creation is outside the handler, so an existing run is never modified.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except Exception as exc:
        write_json_atomic(args.output_dir / "FAILED.json", {"complete": False, "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
