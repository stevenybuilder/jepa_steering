#!/usr/bin/env python3
"""Fixed developmental candidates in fresh paired native 99-step episodes.

No candidate selection occurs here. Original confirmation episodes are refused.
Replans use native CEM (maximumH6, shortened near episode end); only the
executed prefix has physical truth.
"""
from __future__ import annotations
import argparse
import contextlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch


def validate_design(episodes, candidates):
    if not episodes or len(set(episodes)) != len(episodes) or any(e not in (0, 4, 7) for e in episodes):
        raise ValueError("Only unique predetermined DEVELOPMENT starts0/4/7")
    if not 1 <= len(candidates) <= 2 or len({c["id"] for c in candidates}) != len(candidates):
        raise ValueError("Freeze one or two distinct candidates before any full episode")
    if any(c["family"] not in ("branch_scale", "pca_gain", "action_gain") for c in candidates):
        raise ValueError("Full follow-up is for broad candidates, not superseded coordinate pilot")


def native_steps_left(env, wm, cfg):
    return max((env.steps_left()+1)*wm.action_skip//cfg.frameskip, 1)


def full_patch_unroll(patch, *args, **kwargs):
    """Full-episode adapter; leave the frozen fixed-H6 search implementation alone."""
    from run_residual_coordinate_time import native_unroll_actions
    actions = native_unroll_actions(args, kwargs)
    horizon = len(actions)
    if not 1 <= horizon <= 6:
        raise RuntimeError("Native remaining-episode horizon outside1..6")
    if patch.candidate["family"] == "action_gain":
        patch.raw = patch.wm.preprocessor.denormalize_actions(actions.reshape(-1, 4).cpu()).to(actions).reshape(*actions.shape[:2], 5, 4)
    patch.component.norm_targets = patch.norm_targets
    patch.component.reset()
    before = len(patch.component.records)
    result = patch.native(*args, **kwargs)
    if patch.component.step != horizon:
        raise RuntimeError("Actual block calls differ from native proposed horizon")
    patch.records.extend({**r, "native_horizon": horizon,
                          "requested_norm": r["requested_residual_norm"],
                          "rounded_norm": r["actual_rounded_residual_norm"]}
                         for r in patch.component.records[before:])
    return result


def performance(start_hand, states, rewards, native_info):
    from run_residual_coordinate_time import hand_goal_metrics
    if len(states) != 99 or len(rewards) != 99 or len(native_info) != 99:
        raise RuntimeError("Incomplete native99 episode cannot be reported as complete")
    if not all("success" in row for row in native_info):
        raise RuntimeError("Native success missing; no hand-distance label substitution")
    return {**hand_goal_metrics(start_hand, states), "raw_steps": 99,
            "native_ever_success": any(r["success"] > 0 for r in native_info),
            "native_final_success": native_info[-1]["success"] > 0,
            "native_reward_sum": float(sum(rewards)), "native_reward_max": float(max(rewards))}


@torch.no_grad()
def run_arm(cfg, wm, preprocessor, source_bank, candidate, donors, basis, sham,
            baseline, deadline, output_dir):
    from capture_horizon_coordinates import reconstruct_fresh, same_physics, exact
    from causal_planner_forks import close_env
    from collect_on_policy_bank import physics_snapshot
    from precompute_native_replay import ReachControlCapture
    from residual_search_v1 import make_patch
    from run_residual_coordinate_time import native_unroll_actions
    from protocol import file_sha256, write_json_atomic
    from evals.simu_env_planning.planning.utils import make_td
    episode = source_bank["episode"]
    arm = "unsteered" if candidate is None else candidate["id"]+("-sham" if sham else "-edit")
    env, agent, z, checks = reconstruct_fresh(cfg, wm, preprocessor, source_bank)
    native = agent.planner.unroll
    start = physics_snapshot(env)
    initial_context = {k: z[k].detach().cpu().clone() for k in ("visual", "proprio")}
    initial_hand = source_bank["replans"][0]["observation_proprio"].reshape(-1)[:3].numpy()
    patch = None if candidate is None else make_patch(wm, candidate, donors, basis, sham=sham)
    if patch is not None:
        patch.native = native
    if baseline is not None:
        same_physics(start, baseline["initial_physics"], "full99 paired initial physics")
        if checks["fresh_goal_sha256"] != baseline["fresh_goal_sha256"]:
            raise RuntimeError("Fresh paired full99 goal changed")
        for key in initial_context:
            exact(initial_context[key], baseline["initial_context"][key], "full99 initial "+key)
    states, frames, commands, rewards, infos, controls, snapshots = [], [], [], [], [], [], []
    replan_receipts, first_candidates = [], None
    began, done, iteration_records = time.monotonic(), False, []

    def tracked(*a, **kw):
        actions = native_unroll_actions(a, kw)
        if not 1 <= actions.shape[0] <= 6:
            raise RuntimeError("Native full99 horizon outside1..6")
        pred = native(*a, **kw) if patch is None else full_patch_unroll(patch, *a, **kw)
        costs = agent.objective(pred, actions).detach().cpu().reshape(-1)
        iteration_records.append({"native_horizon": int(actions.shape[0]), "actions_normalized": actions.detach().cpu().clone(),
                                  "costs": costs, "ranks": costs.argsort().argsort()})
        return pred

    agent.planner.unroll = tracked
    base = env.proprio_env.unwrapped
    try:
        print(json.dumps({"event": "full99_arm_started", "episode": episode, "arm": arm}), flush=True)
        while not done:
            if time.monotonic() > deadline:
                raise RuntimeError("Worker budget expired before next replan; partial episode retained")
            iteration_records.clear()
            if patch is not None:
                patch.records.clear()
                if hasattr(patch, "component"):
                    patch.component.records.clear()
            before = physics_snapshot(env)
            rng = agent.local_gpu_generator.get_state().cpu().clone()
            steps_left = native_steps_left(env, wm, cfg)
            # Hooks exist only during prediction, never real-observation encoding.
            with patch if patch is not None else contextlib.nullcontext():
                prefix = agent.plan(z.clone(), steps_left=steps_left)
            populations = [r for r in iteration_records if r["actions_normalized"].shape[1] == 300]
            if not populations:
                raise RuntimeError("Native CEM population300 missing")
            if first_candidates is None:
                first_candidates = populations[0]["actions_normalized"]
                if baseline is not None:
                    exact(first_candidates, baseline["first_candidates"], "full99 same first300 proposals")
            raw = preprocessor.denormalize_actions(prefix.detach().cpu().reshape(-1, 4))
            if not 1 <= len(raw) <= 15:
                raise RuntimeError("Native receding prefix outside1..15 commands")
            chunk_states, chunk_frames, chunk_infos, chunk_rewards = [], [], [], []
            chunk_controls, chunk_physics, source_infos = [], [], []
            with ReachControlCapture(base) as capture:
                for action in raw:
                    capture.records.clear()
                    observations, physical_rewards, dones, physical_infos = env.step_multiple(action[None])
                    if len(observations) != 1:
                        raise RuntimeError("Expected exactly one actual transition per raw command")
                    native_reward, native_info = base.evaluate_state(base._get_obs(), action.numpy())
                    clean_info = {k: float(v) for k, v in native_info.items() if np.asarray(v).shape == ()}
                    chunk_infos.append(clean_info)
                    chunk_rewards.append(float(physical_rewards[0]))
                    source_infos.append(physical_infos[0])
                    chunk_states.append(np.asarray(physical_infos[0]["state"]).copy())
                    chunk_frames.append(observations[0].detach().cpu().clone())
                    chunk_controls.append({"xyz_setter": list(capture.records), "actuator_ctrl": np.asarray(base.data.ctrl).copy(),
                                           "native_evaluate_reward": float(native_reward)})
                    chunk_physics.append(physics_snapshot(env))
                    done = bool(dones[-1])
                    if done:
                        break
            count = len(chunk_states)
            selected = agent._predicted_best_encs_over_iterations[-1].detach().cpu()
            # Only H1..H3 of the chosen plan are actually executed before replanning.
            truth, prediction_errors = [], []
            for offset in range(4, count, 5):
                physical_td = make_td(chunk_frames[offset], source_infos[offset])
                encoded = wm.encode(physical_td.to(agent.device).unsqueeze(0), act=True)
                h = (offset+1)//5
                visual_truth = encoded["visual"].detach().cpu()
                proprio_truth = encoded["proprio"].detach().cpu()
                vp = selected["visual"][h].reshape(-1)
                pp = selected["proprio"][h].reshape(-1)
                vt, pt = visual_truth.reshape(-1), proprio_truth.reshape(-1)
                if vp.shape != vt.shape or pp.shape != pt.shape:
                    raise RuntimeError("Physical/predicted prefix tensor axes differ")
                prediction_errors.append({"imagined_horizon": h,
                    "visual_mse": float((vp.float()-vt.float()).square().mean()),
                    "proprio_mse": float((pp.float()-pt.float()).square().mean())})
                truth.append({"horizon": h, "visual": visual_truth, "proprio": proprio_truth})
            index = len(replan_receipts)
            value = {"episode": episode, "arm": arm, "replan": index, "candidate": candidate, "sham": sham,
                     "initial_physics": before, "planner_rng_before": rng, "steps_left_model": steps_left,
                     "encoded_context": {k: z[k].detach().cpu() for k in ("visual", "proprio")},
                     "candidate_iterations": list(iteration_records), "selected_full_plan": agent.planner._prev_mean.detach().cpu().clone(),
                     "selected_prediction": selected, "raw_prefix_proposed": raw, "raw_commands_executed": raw[:count],
                     "states": np.stack(chunk_states), "frames": torch.stack(chunk_frames), "native_info": chunk_infos,
                     "rewards": chunk_rewards, "controls": chunk_controls, "physics": chunk_physics,
                     "actual_executed_prefix_encodings": truth, "prediction_errors": prediction_errors,
                     "edit_records": [] if patch is None else list(patch.records)}
            path = output_dir/f"episode-{episode:03d}-{arm}-replan-{index:02d}.pt"
            torch.save(value, path)
            receipt = {"path": path.name, "sha256": file_sha256(path), "bytes": path.stat().st_size,
                       "raw_steps": count, "prediction_errors": prediction_errors}
            write_json_atomic(path.with_suffix(".DONE.json"), {"complete": True, "outputs": [receipt]})
            replan_receipts.append(receipt)
            states.extend(chunk_states); frames.extend(chunk_frames); commands.append(raw[:count])
            rewards.extend(chunk_rewards); infos.extend(chunk_infos); controls.extend(chunk_controls); snapshots.extend(chunk_physics)
            if len(states) > 99 or (len(states) == 99 and not done):
                raise RuntimeError("Native termination differs from frozen99-step protocol")
            if not done:
                td = make_td(chunk_frames[-1], source_infos[-1])
                z = wm.encode(td.to(agent.device).unsqueeze(0), act=True)
            print(json.dumps({"event": "full99_replan_complete", "episode": episode, "arm": arm,
                              "replan": index, "raw_steps": len(states), "seconds": time.monotonic()-began}), flush=True)
        metrics = performance(initial_hand, states, rewards, infos)
        result = {"episode": episode, "arm": arm, "candidate": candidate, "sham": sham, **metrics,
                  "replans": replan_receipts, "seconds": time.monotonic()-began,
                  "same_initial_physics_and_first300": True,
                  "progress_vs_baseline_m": 0. if baseline is None else metrics["hand_goal_progress_m"]-baseline["metrics"]["hand_goal_progress_m"]}
        return {"metrics": result, "initial_physics": start, "initial_context": initial_context,
                "first_candidates": first_candidates, "fresh_goal_sha256": checks["fresh_goal_sha256"],
                "states": np.stack(states), "actions_raw": torch.cat(commands), "native_info": infos,
                "rewards": rewards, "checks": checks}
    finally:
        agent.planner.unroll = native
        close_env(env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "config", "project-root", "inventory", "candidates", "bases", "output-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--seconds-limit", type=float, default=7200)
    args = parser.parse_args()
    candidates = json.loads(args.candidates.read_text())["candidates"]
    validate_design(args.episodes, candidates)
    sys.path.insert(0, str(args.repo))
    from protocol import file_sha256, write_json_atomic
    from model_loader import load_headless_metaworld
    from capture_specificity_controls import parameters_sha
    from causal_planner_forks import setup_cfg
    from precompute_native_replay import safe_rows
    from package_steering_banks import load_development
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); deadline = started+args.seconds_limit
    outputs, results = [], []
    protocol = {"phase": "adaptive_development_full99", "episodes": args.episodes, "candidates": candidates,
                "native_cem": {"maximum_horizon": 6, "near_end_horizon": "Released planner shortens toremainingsteps; no override", "population": 300, "iterations": 15, "max_raw_prefix": 15},
                "raw_episode_steps": 99, "held_data_opened": False,
                "primary": "Native ever-success; final-success/reward/final-hand-goal-distance all reported, no endpoint swapping",
                "interpretation": "Exploratory selected candidates, not independent confirmation",
                "sham": "Same-state post-projection/gate norm and fixed permutation; adaptive states/populations differ, no cumulative cross-arm dose equality claim",
                "prediction_truth": "Compare only physically executed5-step prefix horizons, not counterfactual future after replan",
                "timing_policy": "Pulse indexes imagined steps within each native unroll; a pulse beyond a shortened horizon does not fire",
                "source_sha256": {p.name: file_sha256(p) for p in (Path(__file__), args.config, args.candidates, args.bases)},
                "seconds_limit": args.seconds_limit}
    write_json_atomic(args.output_dir/"protocol.json", protocol)
    try:
        torch.set_num_threads(2)
        rows = safe_rows(json.loads(args.inventory.read_text()), "reach_wall", args.episodes)
        wm, preprocessor, model = load_headless_metaworld(args.repo)
        wm.eval().requires_grad_(False); before = parameters_sha(wm)
        checkpoint_sha = file_sha256(Path(model["checkpoint"]))
        if any(row["model"]["checkpoint_sha256"] != checkpoint_sha for row in rows):
            raise RuntimeError("Frozen development checkpoint differs")
        cfg = setup_cfg(args.config, args.output_dir, wm)
        basis = {k: torch.as_tensor(v, device="cuda", dtype=torch.float32) for k, v in np.load(args.bases, allow_pickle=False).items()}
        for episode in args.episodes:
            bank = load_development(args.project_root, next(r for r in rows if r["episode"] == episode))
            baseline = None
            for candidate, sham in [(None, False)]+[(c, s) for c in candidates for s in (False, True)]:
                value = run_arm(cfg, wm, preprocessor, bank, candidate, None, basis, sham, baseline, deadline, args.output_dir)
                metrics = value["metrics"]
                path = args.output_dir/f"episode-{episode:03d}-{metrics['arm']}.pt"
                torch.save(value, path)
                receipt = {"path": path.name, "sha256": file_sha256(path), "bytes": path.stat().st_size}
                write_json_atomic(path.with_suffix(".DONE.json"), {"complete": True, "outputs": [receipt]})
                outputs.extend([receipt]+metrics["replans"]); results.append(metrics)
                write_json_atomic(args.output_dir/"progress.json", {"complete": False, "rows": results, "outputs": outputs})
                print(json.dumps({"event": "full99_arm_complete", "metrics": metrics}), flush=True)
                if baseline is None:
                    baseline = value
        if parameters_sha(wm) != before:
            raise RuntimeError("Frozen weights changed during full episodes")
        report = args.output_dir/"full_episodes.json"
        write_json_atomic(report, {"complete": True, "protocol": protocol, "rows": results,
                                  "seconds": time.monotonic()-started, "model": model, "parameters_sha256": before})
        outputs.extend({"path": p.name, "sha256": file_sha256(p), "bytes": p.stat().st_size} for p in (report, args.output_dir/"protocol.json"))
        write_json_atomic(args.output_dir/"DONE.json", {"complete": True, "outputs": outputs})
    except Exception as exc:
        write_json_atomic(args.output_dir/"FAILED.json", {"complete": False, "error": str(exc), "outputs": outputs, "rows": results})
        raise


if __name__ == "__main__":
    main()
