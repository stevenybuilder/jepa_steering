#!/usr/bin/env python3
"""Replay immutable development baseline commands; capture physical data, no CEM.

Reach uses the exact cached goal stimulus, checks all original replan pixels and
physics, and encodes sampled observed states. Push captures actual collision
callbacks while verifying every existing state and image. Held banks stay sealed.
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from package_steering_banks import load_development, replay_prefixes, sha
from protocol import write_json_atomic


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}), flush=True)


def exact_array(actual, expected, name, tolerance=1e-6):
    actual = actual.detach().cpu().numpy() if isinstance(actual, torch.Tensor) else np.asarray(actual)
    expected = expected.detach().cpu().numpy() if isinstance(expected, torch.Tensor) else np.asarray(expected)
    if actual.shape != expected.shape:
        raise RuntimeError(f"{name}: layout {actual.shape} != {expected.shape}")
    error = float(np.max(np.abs(actual.astype(float)-expected.astype(float)))) if actual.size else 0.0
    if not np.isfinite(error) or error > tolerance:
        raise RuntimeError(f"{name}: error {error} > {tolerance}")
    return error


def safe_rows(inventory, task, episodes):
    rows = {int(row["episode"]): row for row in inventory["banks"] if row["task"] == task}
    selected = []
    for episode in episodes:
        if episode not in rows or rows[episode]["sealed"] or rows[episode]["split"] != "development":
            raise RuntimeError("Refusing non-development replay before tensor access")
        selected.append(rows[episode])
    if len(set(episodes)) != len(episodes):
        raise RuntimeError("Duplicate replay episode")
    return selected


def checkpoint_record(provenance, expected):
    path = Path(provenance["checkpoint"])
    digest = sha(path)
    if digest != expected:
        raise RuntimeError("Runtime checkpoint differs from frozen bank provenance")
    return {**provenance, "checkpoint_sha256": digest}


def diagnose_physics_without_model(repo, bank_path, config_path, output_dir):
    """Bounded diagnostic, never a replay-ready product: no encoder or renderer.

    Compare native command chunks and instrumented single steps on this CPU.
    The rendered production replay still must pass its untouched pixel guards.
    """
    import platform
    import sys
    import mujoco
    sys.path.insert(0, str(repo))
    from causal_planner_forks import load_development_bank, setup_cfg, close_env
    from collect_on_policy_bank import initialize_episode, physics_snapshot
    from evals.simu_env_planning.envs.init import make_env
    from evals.simu_env_planning.envs.metaworld import MetaWorldWrapper
    from evals.simu_env_planning.planning.plan_evaluator import PlanEvaluator
    from evals.simu_env_planning.planning.utils import set_seed

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    bank, digest = load_development_bank(Path(bank_path))
    cfg = setup_cfg(Path(config_path), output_dir, SimpleNamespace(action_skip=1))
    cfg.device = "cpu"
    prefixes = replay_prefixes(bank["replans"], len(bank["actions_raw"]), "reach_wall")
    saved_render = MetaWorldWrapper.render
    # Explicitly no pixel inference: constant buffers only exercise the exact
    # official task/reset/control wrapper sequence without making an EGL context.
    MetaWorldWrapper.render = lambda self, *a, **k: np.zeros(
        (self.cfg.task_specification.img_size, self.cfg.task_specification.img_size, 3), dtype=np.uint8)
    modes = {}
    try:
        for mode in ("native_chunks", "instrumented_single_steps"):
            set_seed(cfg.local_seed)
            env = make_env(cfg)
            agent = SimpleNamespace(model=SimpleNamespace(tubelet_size_enc=1), set_goal=lambda goal: None)
            evaluator = PlanEvaluator(cfg, agent)
            initialize_episode(cfg, agent, env, evaluator, bank["episode"], bank["environment_seed"])
            base = env.proprio_env.unwrapped
            if mode == "instrumented_single_steps":
                base._get_obs()  # same additional observation read as production capture
                mujoco_contacts(base)
            physical = [physics_snapshot(env)]
            native_step = env.step

            def recorded_step(action, *a, **k):
                result = native_step(action, *a, **k)
                physical.append(physics_snapshot(env))
                return result

            env.step = recorded_step
            controls = []
            with ReachControlCapture(base) as capture:
                # Native mode bypasses even the control observer wrapper.
                if mode == "native_chunks":
                    base.set_xyz_action = capture.original
                for index, (prefix, count, _) in enumerate(prefixes):
                    commands = bank["actions_raw"][prefix:prefix+count]
                    exact_array(commands, bank["replans"][index]["planned_actions_raw"][:count], "cached action chunk", 0)
                    if mode == "native_chunks":
                        env.step_multiple(commands)
                    else:
                        for action in commands:
                            capture.records.clear()
                            env.step_multiple(action.unsqueeze(0))
                            mujoco_contacts(base)
                            controls.extend(capture.records)
            comparisons = []
            for index, (prefix, _, _) in enumerate(prefixes):
                actual, expected = physical[prefix], bank["replans"][index]["simulator"]
                errors = {key: float(np.max(np.abs(np.asarray(value)-expected["physics"][key]))) if np.asarray(value).size else 0.
                          for key, value in actual["physics"].items()}
                errors["time"] = abs(actual["time"]-expected["time"])
                comparisons.append({"replan": index, "prefix": prefix, "max_absolute_errors": errors})
            modes[mode] = {"physics": physical, "controls": controls, "waypoints": comparisons}
            close_env(env)
        differences = []
        for step, (a, b) in enumerate(zip(modes["native_chunks"]["physics"], modes["instrumented_single_steps"]["physics"])):
            errors = {key: float(np.max(np.abs(np.asarray(a["physics"][key])-b["physics"][key]))) if np.asarray(a["physics"][key]).size else 0.
                      for key in a["physics"]}
            if any(errors.values()):
                differences.append({"raw_prefix": step, "errors": errors})
        trace = output_dir / "physics_traces.pt"
        torch.save(modes, trace)
        write_json_atomic(output_dir / "DIAGNOSTIC.json", {
            "diagnostic_complete": True, "replay_ready": False, "episode": bank["episode"],
            "bank_sha256": digest, "script_sha256": sha(Path(__file__)), "mujoco": mujoco.__version__,
            "numpy": np.__version__, "platform": platform.platform(), "model_execution": False,
            "render_execution": False, "pixels_verified": False,
            "native_vs_instrumented_first_differences": differences[:5],
            "modes": {key: {"waypoints": value["waypoints"]} for key, value in modes.items()},
            "outputs": [{"path": trace.name, "sha256": sha(trace)}],
        })
        emit("physics_only_diagnostic_complete", episode=bank["episode"], differences=len(differences), replay_ready=False)
    finally:
        MetaWorldWrapper.render = saved_render


def check_reach_waypoint(env, observation, proprio, snapshot):
    from collect_on_policy_bank import physics_snapshot
    now = physics_snapshot(env)
    checks, failures = {}, {}

    def check(name, actual, expected, tolerance=1e-6):
        try:
            checks[name] = exact_array(actual, expected, "waypoint " + name, tolerance)
        except RuntimeError as exc:
            failures[name] = str(exc)

    # Compute physics even when pixels fail, so render-only and physical failures
    # can be distinguished without weakening any original equality threshold.
    check("pixels", observation, snapshot["observation_visual"], 0)
    check("proprio", proprio, snapshot["observation_proprio"])
    check("time", now["time"], snapshot["simulator"]["time"])
    if now["elapsed_steps"] != snapshot["elapsed_steps"]:
        failures["elapsed_steps"] = "Waypoint elapsed steps differ"
    for key, expected in snapshot["simulator"]["physics"].items():
        check(key, now["physics"][key], expected)
    for key, expected in snapshot["simulator"]["task_state"].items():
        if isinstance(expected, str):
            if now["task_state"][key] != expected:
                failures["task."+key] = f"Waypoint task string differs: {key}"
        else:
            check("task."+key, now["task_state"][key], expected)
    if failures:
        raise RuntimeError("Waypoint checks failed: " + json.dumps({"passed_max_errors": checks, "failures": failures}, sort_keys=True))
    return checks, now


def mujoco_contacts(base):
    import mujoco
    result = []
    for index in range(int(base.data.ncon)):
        contact = base.data.contact[index]
        force = np.zeros(6)
        mujoco.mj_contactForce(base.model, base.data, index, force)
        result.append({"geom1": int(contact.geom1), "geom2": int(contact.geom2),
                       "geom1_name": mujoco.mj_id2name(base.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)),
                       "geom2_name": mujoco.mj_id2name(base.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)),
                       "position": np.asarray(contact.pos).copy(), "distance": float(contact.dist),
                       "constraint_address": int(contact.efc_address), "solver_active": int(contact.efc_address) >= 0,
                       "contact_force_torque": force, "normal_force_positive": bool(force[0] > 0)})
    return result


class ReachControlCapture:
    """Observe native XYZ action handling and resulting actual mocap controls."""
    def __init__(self, base):
        self.base, self.original, self.records = base, base.set_xyz_action, []
        self.source = inspect.getsource(type(base).set_xyz_action)

    def __enter__(self):
        def observed(action):
            before = np.asarray(self.base.data.mocap_pos).copy()
            raw = np.asarray(action).copy()
            value = self.original(action)
            self.records.append({"xyz_argument_to_native_setter": raw,
                                 "mocap_position_before": before,
                                 "mocap_position_after": np.asarray(self.base.data.mocap_pos).copy()})
            return value
        self.base.set_xyz_action = observed
        return self

    def __exit__(self, *_):
        self.base.set_xyz_action = self.original


class PushContactCapture:
    """Observe the existing Pymunk post-solve callback without replacing physics."""
    def __init__(self, base):
        self.base, self.records = base, []
        self.handler = getattr(base, "collision_handeler", None)
        self.available = self.handler is not None and hasattr(self.handler, "post_solve")

    def __enter__(self):
        if not self.available:
            return self
        self.original = self.handler.post_solve
        def observed(arbiter, space, data):
            def body_name(body):
                if body is self.base.agent:
                    return "agent"
                if body is self.base.block:
                    return "block"
                if body is self.base.space.static_body:
                    return "static_wall"
                return "other_simulator_body"
            bodies = [body_name(shape.body) for shape in arbiter.shapes]
            points = arbiter.contact_point_set
            self.records.append({"bodies": bodies, "agent_block_contact": set(bodies) == {"agent", "block"},
                                 "normal": list(points.normal), "total_impulse": list(arbiter.total_impulse),
                                 "points": [{"point_a": list(point.point_a), "point_b": list(point.point_b),
                                             "distance": float(point.distance)} for point in points.points]})
            if self.original is not None:
                return self.original(arbiter, space, data)
        self.handler.post_solve = observed
        return self

    def __exit__(self, *_):
        if self.available:
            self.handler.post_solve = self.original


def restore_goal_encodings(agent, cached_goal):
    differences = {}
    for name in ("visual", "proprio"):
        cached = cached_goal["encoded_"+name]
        fresh = agent.goal_state_enc[name].detach().cpu().to(cached.dtype)
        if fresh.shape != cached.shape or not torch.isfinite(fresh).all():
            raise RuntimeError("Fresh goal encoding is malformed")
        differences[name] = float((fresh.float()-cached.float()).abs().max())
        agent.goal_state_enc[name] = cached.to(agent.device)
        exact_array(agent.goal_state_enc[name].detach().cpu(), cached, "restored cached goal encoding", 0)
    return differences


def reconstruct_physical(cfg, wm, preprocessor, bank):
    """Use immutable cached goal encodings for physical replay, without CEM.

    Fresh cross-hardware encoding differences are recorded; physical checks remain
    exact. Cached encodings are stored at their original (lossy) cache dtype and
    do not certify bitwise equivalence to the original full-precision planner.
    """
    from causal_planner_forks import restore_cached_goal
    from collect_on_policy_bank import initialize_episode
    from evals.simu_env_planning.envs.init import make_env
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    from evals.simu_env_planning.planning.plan_evaluator import PlanEvaluator
    from evals.simu_env_planning.planning.utils import set_seed
    set_seed(cfg.local_seed)
    env = make_env(cfg)
    agent = GC_Agent(cfg, wm, dset=None, preprocessor=preprocessor)
    evaluator = PlanEvaluator(cfg, agent)
    td, _, generated, _, _ = initialize_episode(cfg, agent, env, evaluator, bank["episode"], bank["environment_seed"])
    generated_error = float(np.max(np.abs(generated["visual"].cpu().numpy().astype(float)-np.asarray(bank["goal"]["visual"]).astype(float))))
    goal = restore_cached_goal(generated, bank["goal"])
    agent.set_goal(goal)
    differences = restore_goal_encodings(agent, bank["goal"])
    agent.replay_goal_metadata = {"mode": "immutable_cached_raw_and_saved_encoded_stimulus",
                                 "regenerated_goal_pixels_max_error": generated_error,
                                 "fresh_goal_encoding_max_error_at_saved_dtype": differences,
                                 "restored_saved_encodings_exact": True,
                                 "exact_planner_continuation_claimed": False}
    checks, _ = check_reach_waypoint(env, td["visual"].cpu(), td["proprio"].cpu(), bank["replans"][0])
    checks["goal_pixels"] = exact_array(goal["visual"].cpu(), bank["goal"]["visual"], "cached goal pixels", 0)
    checks["goal_proprio"] = exact_array(goal["proprio"].cpu(), bank["goal"]["proprio"], "cached goal proprio", 0)
    agent.local_gpu_generator.set_state(bank["replans"][0]["planner_rng_before"])
    z = wm.encode(td.to(agent.device).unsqueeze(0), act=True)
    emit("physical_reconstruction_verified", **agent.replay_goal_metadata)
    return env, agent, z, checks


@torch.no_grad()
def reach_episode(args, row, wm, preprocessor, cfg, provenance):
    from causal_planner_forks import close_env
    from collect_on_policy_bank import physics_snapshot
    from evals.simu_env_planning.planning.utils import make_td
    bank = load_development(args.project_root, row)
    prefixes = replay_prefixes(bank["replans"], len(bank["actions_raw"]), "reach_wall")
    started = time.monotonic()
    env, agent, z, initial_checks = reconstruct_physical(cfg, wm, preprocessor, bank)
    base = env.proprio_env.unwrapped
    first = bank["replans"][0]
    observation, proprio = first["observation_visual"], first["observation_proprio"]
    state = np.asarray(base._get_obs()).copy()
    exact_array(state[:3], np.asarray(proprio).reshape(-1)[:3], "initial observed hand")
    observations, states, proprios = [observation], [state], [proprio]
    physical, contacts, controls, waypoints = [physics_snapshot(env)], [mujoco_contacts(base)], [], []
    encoded = [{"raw_prefix_count": 0, "visual_pooled": z["visual"].detach().float().cpu().reshape(-1, 384).mean(0),
                "proprio": z["proprio"].detach().cpu()}]
    boundary = {prefix: (index, count) for index, (prefix, count, _elapsed) in enumerate(prefixes)}
    total_encoder_seconds = 0.0
    try:
        with ReachControlCapture(base) as capture:
            for raw_index, action in enumerate(bank["actions_raw"]):
                if raw_index in boundary:
                    replan, count = boundary[raw_index]
                    exact_array(bank["actions_raw"][raw_index:raw_index+count],
                                bank["replans"][replan]["planned_actions_raw"][:count],
                                "saved command prefix matches native planned chunk", 0)
                    checks, now = check_reach_waypoint(env, observation, proprio, bank["replans"][replan])
                    waypoints.append({"replan": replan, "raw_prefix_count": raw_index, "executed_action_count": count,
                                      "elapsed_steps": now["elapsed_steps"], "checks": checks, "physics": now})
                    emit("waypoint_verified", task="reach_wall", episode=row["episode"], replan=replan, prefix=raw_index)
                capture.records.clear()
                obs, _rewards, _dones, infos = env.step_multiple(action.unsqueeze(0))
                if len(obs) != 1:
                    raise RuntimeError("Expected exactly one raw simulator step")
                observation, info = obs[0].cpu(), infos[0]
                proprio = torch.as_tensor(info["proprio"]).cpu()
                states.append(np.asarray(info["state"]).copy()); observations.append(observation); proprios.append(proprio)
                physical.append(physics_snapshot(env)); contacts.append(mujoco_contacts(base))
                controls.append({"native_xyz_setter_calls": list(capture.records),
                                 "actuator_ctrl_after_step": np.asarray(base.data.ctrl).copy(),
                                 "actuator_force_after_step": np.asarray(base.data.actuator_force).copy(),
                                 "mocap_position_after_step": np.asarray(base.data.mocap_pos).copy()})
                if (raw_index+1) % 5 == 0 or raw_index+1 == len(bank["actions_raw"]):
                    torch.cuda.synchronize()
                    encode_start = time.monotonic()
                    now_z = wm.encode(make_td(observation, info).to(agent.device).unsqueeze(0), act=True)
                    torch.cuda.synchronize()
                    total_encoder_seconds += time.monotonic()-encode_start
                    encoded.append({"raw_prefix_count": raw_index+1,
                                    "visual_pooled": now_z["visual"].detach().float().cpu().reshape(-1, 384).mean(0),
                                    "proprio": now_z["proprio"].detach().cpu()})
            native_setter_source = capture.source
        if len(waypoints) != len(bank["replans"]):
            raise RuntimeError("Not all cached replan waypoints were checked")
        hand = np.asarray(states)[:, :3]
        five_steps = [{"raw_start": i, "raw_end": i+5, "hand_delta_xyz": hand[i+5]-hand[i]}
                      for i in range(0, len(hand)-5, 5)]
        return {"complete": True, "task": "reach_wall", "episode": row["episode"], "split": "development",
                "scope": "same-command physical replay and observed-state encoding; no CEM or steering",
                "bank_path": row["path"], "bank_sha256": row["sha256"], "environment_seed": row["environment_seed"],
                "planner_seed": row["planner_seed"], "model_provenance": provenance,
                "goal": bank["goal"], "goal_origin": "immutable original baseline raw visual/proprio/state and cached encodings",
                "goal_replay_metadata": agent.replay_goal_metadata, "initial_checks": initial_checks,
                "raw_actions_submitted": bank["actions_raw"], "states_including_start": np.stack(states),
                "observations_including_start": torch.stack(observations).to(torch.uint8),
                "proprios_including_start": torch.stack(proprios), "physics_including_start": physical,
                "contact_source": "MuJoCo data.contact and mj_contactForce at each raw-step boundary; intervening integration contacts may be missed",
                "contacts_including_start": contacts, "applied_control_source": "actual actuator ctrl/force and mocap arrays plus native XYZ setter arguments",
                "applied_controls": controls, "native_xyz_setter_source": native_setter_source,
                "five_raw_step_displacements": five_steps, "waypoints": waypoints,
                "final_chunk_endpoint_hand_xyz": hand[-1], "observed_state_encodings": encoded,
                "later_planner_warmstart": "not reconstructed; unnecessary for paired full episodes initialized at original start",
                "seconds": time.monotonic()-started, "sampled_encoder_gpu_seconds": total_encoder_seconds}
    except Exception:
        # Keep the diagnostic actual trace, without issuing a readiness receipt.
        torch.save({"complete": False, "episode": row["episode"],
                    "states_including_start": states, "observations_including_start": observations,
                    "physics_including_start": physical, "applied_controls": controls,
                    "waypoints": waypoints}, args.output_dir / f"episode-{row['episode']:03d}-FAILED-trace.pt")
        raise
    finally:
        close_env(env)


def push_episode(args, row, cfg):
    from evals.simu_env_planning.envs.init import make_env
    bank = load_development(args.project_root, row)
    started = time.monotonic()
    env = make_env(cfg)
    env.update_env(bank["env_info"])
    observation, info = env.prepare(bank["environment_seed"], bank["initial_state"], bank["env_info"])
    initial_checks = {"pixels": exact_array(observation.cpu(), bank["observations"][0], "Push initial pixels", 0),
                      "state": exact_array(info["state"], bank["states"][0], "Push initial state")}
    rows, max_state_error, max_pixel_error = [], 0.0, 0.0
    try:
        with PushContactCapture(env.unwrapped) as capture:
            for index, action in enumerate(bank["actions_raw"]):
                capture.records.clear()
                obs, _rewards, _dones, infos = env.step_multiple(action.unsqueeze(0))
                if len(obs) != 1:
                    raise RuntimeError("Expected one Push raw step")
                state_error = exact_array(infos[0]["state"], bank["states"][index+1], "Push replay state")
                pixel_error = exact_array(obs[0].cpu(), bank["observations"][index+1], "Push replay pixels", 0)
                max_state_error, max_pixel_error = max(max_state_error, state_error), max(max_pixel_error, pixel_error)
                base = env.unwrapped
                rows.append({"raw_step": index+1, "contacts": list(capture.records),
                             "agent_block_contact": any(x["agent_block_contact"] for x in capture.records) if capture.available else None,
                             "native_n_contacts": infos[0].get("n_contacts"),
                             "native_action_target_xy": np.asarray(base.latest_action).copy(),
                             "agent_velocity_after_step": list(base.agent.velocity),
                             "block_velocity_after_step": list(base.block.velocity)})
            available = capture.available
        return {"complete": True, "task": "pusht", "episode": row["episode"], "split": "development",
                "scope": "same-command physical contact replay; existing state/frame tensors referenced, not duplicated",
                "bank_path": row["path"], "bank_sha256": row["sha256"], "input_sha256": bank["input_sha256"],
                "prepared_input": row["prepared_input"], "environment_seed": bank["environment_seed"], "planner_seed": bank["planner_seed"],
                "model_provenance": row["model"], "model_execution": False, "contact_api_available": available,
                "contact_source": "existing Pymunk post_solve callback, all internal physics substeps of each raw action" if available else "native contact callback unavailable; no distance proxy substituted",
                "initial_checks": initial_checks, "max_state_error": max_state_error, "max_pixel_error": max_pixel_error,
                "raw_step_records": rows, "seconds": time.monotonic()-started, "sampled_encoder_gpu_seconds": 0.0}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "project-root", "inventory", "output-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--task", choices=("reach_wall", "pusht"), required=True)
    parser.add_argument("--episodes", nargs="+", type=int, required=True)
    parser.add_argument("--initial-candidates", type=Path)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    selected = safe_rows(inventory, args.task, args.episodes)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    from model_loader import load_headless_metaworld
    if args.task == "reach_wall":
        from causal_planner_forks import setup_cfg
        wm, preprocessor, provenance = load_headless_metaworld(args.repo)
        provenance = checkpoint_record(provenance, selected[0]["model"]["checkpoint_sha256"])
        cfg = setup_cfg(Path(provenance["config"]), args.output_dir, wm)
    else:
        from collect_pusht_bank import config
        cfg = config(args.repo, args.output_dir)
        cfg.device = "cpu"
    initial_candidates = {"status": "existing first300 bank reusable by hash/pointer; not recomputed",
                          "bank_path": "artifacts/geometry_map/reach_wall_causal_recovery_v2/diagnostic-v2/unsteered.pt",
                          "keys": ["first_candidates"], "episode": 0, "replan": 0,
                          "later_CEM_warmstart_reconstructed": False}
    if args.initial_candidates:
        value = json.loads(args.initial_candidates.read_text())
        entry = next(row for row in value["outputs"] if row["path"] == "unsteered.pt")
        initial_candidates["bank_sha256"] = entry["sha256"]
        initial_candidates["receipt_sha256"] = sha(args.initial_candidates)
    outputs = []
    started = time.monotonic()
    try:
        for row in selected:
            emit("episode_started", task=args.task, episode=row["episode"])
            value = reach_episode(args, row, wm, preprocessor, cfg, provenance) if args.task == "reach_wall" else push_episode(args, row, cfg)
            path = args.output_dir / f"episode-{row['episode']:03d}.pt"
            torch.save(value, path)
            output = {"path": path.name, "sha256": sha(path), "episode": row["episode"], "task": args.task,
                      "raw_actions": len(value["raw_actions_submitted"]) if args.task == "reach_wall" else len(value["raw_step_records"]),
                      "waypoints_verified": len(value["waypoints"]) if args.task == "reach_wall" else len(load_development(args.project_root, row)["replans"]),
                      "seconds": value["seconds"], "sampled_encoder_gpu_seconds": value["sampled_encoder_gpu_seconds"]}
            outputs.append(output)
            write_json_atomic(args.output_dir / "progress.json", {"complete": False, "outputs": outputs})
            emit("episode_complete", **output)
        write_json_atomic(args.output_dir / "DONE.json", {"complete": True, "task": args.task, "episodes": args.episodes,
            "outputs": outputs, "scope": "development baseline physical replay preprocessing; no steering/CEM/evaluation outcomes",
            "seconds": time.monotonic()-started, "initial_candidates": initial_candidates,
            "script_sha256": sha(Path(__file__)), "inventory_sha256": sha(args.inventory)})
    except Exception as exc:
        write_json_atomic(args.output_dir / "FAILED.json", {"complete": False, "error": str(exc), "outputs": outputs})
        raise


if __name__ == "__main__":
    main()
