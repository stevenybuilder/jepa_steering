#!/usr/bin/env python3
"""Fresh development fixed-H6 forecasts and exact-action physical truth.

Never reuse later replans as fixed-action truth. The old cache contains only the
first three selected action chunks; recover a NEW full native CEM mean, freeze
it, and record the old-prefix discrepancy without selecting on it.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import hashlib
import numpy as np
import torch


def horizon_labels(start_step, start_time, future_times, frameskip=5):
    times = np.asarray(future_times, dtype=np.float64)
    if times.shape != (6,) or np.any(np.diff(np.r_[start_time, times]) <= 0):
        raise ValueError("Six strictly increasing actual simulator times required")
    horizon = np.arange(1, 7, dtype=np.int64)
    return {"imagined_step": horizon, "raw_action_offset": horizon * frameskip,
            "real_raw_step": start_step + horizon * frameskip,
            "context_real_raw_step": np.full(6, start_step, dtype=np.int64),
            "physics_time_seconds": times, "context_physics_time_seconds": np.full(6, start_time)}


def exact(actual, expected, label):
    a = actual.detach().cpu().numpy() if torch.is_tensor(actual) else np.asarray(actual)
    b = expected.detach().cpu().numpy() if torch.is_tensor(expected) else np.asarray(expected)
    if a.shape != b.shape or not np.array_equal(a, b):
        error = float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.shape == b.shape and a.size else None
        raise RuntimeError(f"New paired identity failed: {label}, maxABS={error}")


def pool_visual(value):
    if value.shape[-1] != 384:
        raise ValueError("Expected native384 visual channels")
    return value.reshape(value.shape[0], -1, 384).mean(1)


def decode_physical(features, readout):
    return ((np.asarray(features)-readout["mean"])/readout["scale"]) @ readout["coef"].T + readout["intercept"]


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    header = json.dumps({"shape": list(value.shape), "dtype": str(value.dtype)}, sort_keys=True).encode()
    return hashlib.sha256(header+value.numpy().tobytes()).hexdigest()


def reconstruct_fresh(cfg, wm, preprocessor, bank):
    """Original raw stimulus, fresh consistent runtime encoding, no old-dtype gate.

    This creates a NEW paired reference. It does not claim bitwise replay of an
    old float16 encoder cache. Exact fresh goal hashes are checked across arms.
    """
    from evals.simu_env_planning.envs.init import make_env
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    from evals.simu_env_planning.planning.plan_evaluator import PlanEvaluator
    from evals.simu_env_planning.planning.utils import set_seed
    from collect_on_policy_bank import initialize_episode
    from causal_planner_forks import restore_cached_goal
    from precompute_native_replay import check_reach_waypoint
    set_seed(cfg.local_seed)
    env = make_env(cfg)
    agent = GC_Agent(cfg, wm, dset=None, preprocessor=preprocessor)
    evaluator = PlanEvaluator(cfg, agent)
    td, _, goal, _, _ = initialize_episode(cfg, agent, env, evaluator, bank["episode"], bank["environment_seed"])
    goal = restore_cached_goal(goal, bank["goal"])
    agent.set_goal(goal)
    checks, _ = check_reach_waypoint(env, td["visual"].cpu(), td["proprio"].cpu(), bank["replans"][0])
    fresh, original_raw, differences = {}, {}, {}
    for key in ("visual", "proprio"):
        exact(goal[key], bank["goal"][key], "original RAW goal " + key)
        original_raw[key] = tensor_hash(goal[key])
        fresh[key] = tensor_hash(agent.goal_state_enc[key])
        cached = bank["goal"]["encoded_"+key]
        differences[key] = float((agent.goal_state_enc[key].cpu().to(cached.dtype).float()-cached.float()).abs().max())
    checks["fresh_goal_sha256"] = fresh
    checks["original_raw_goal_sha256"] = original_raw
    checks["old_float16_goal_encoding_discrepancy_descriptive_only"] = differences
    agent.replay_goal_metadata = {"mode": "original_raw_goal_fresh_consistent_fullprecision_encoding", **checks}
    agent.local_gpu_generator.set_state(bank["replans"][0]["planner_rng_before"])
    z = wm.encode(td.to(agent.device).unsqueeze(0), act=True)
    agent.capture_raw_context = {key: td[key].detach().cpu().clone() for key in ("visual", "proprio")}
    print(json.dumps({"event": "fresh_goal_reference_frozen", "episode": bank["episode"],
                      "fresh_goal_sha256": fresh, "old_encoding_difference": differences}), flush=True)
    return env, agent, z, checks


def load_late_reference(directories, episode):
    from package_steering_banks import sha
    if not 0 <= episode < 12:
        raise RuntimeError("Held late-context reference refused before tensor access")
    paths = [directory/f"episode-{episode:03d}.pt" for directory in directories
             if (directory/f"episode-{episode:03d}.DONE.json").exists()]
    if len(paths) != 1:
        raise RuntimeError("Exactly one completed initial-context reference required")
    path = paths[0]
    receipt = json.loads(path.with_suffix(".DONE.json").read_text())
    entry = next(row for row in receipt["outputs"] if row["path"] == path.name)
    if not receipt["complete"] or entry["episode"] != episode or sha(path) != entry["sha256"]:
        raise RuntimeError("Initial-context reference checksum mismatch")
    reference = torch.load(path, map_location="cpu", weights_only=False)
    if not reference["fresh30action_physics_and_pixels_repeat_exact"]:
        raise RuntimeError("Initial reference is not verified fixed-action physics")
    return reference, sha(path)


@torch.no_grad()
def reconstruct_capture_context(args, cfg, wm, preprocessor, bank):
    env, agent, z, checks = reconstruct_fresh(cfg, wm, preprocessor, bank)
    if not args.late_reference_dirs:
        return env, agent, z, checks
    from collect_on_policy_bank import physics_snapshot
    from evals.simu_env_planning.planning.utils import make_td
    reference, digest = load_late_reference(args.late_reference_dirs, bank["episode"])
    observations, _, _, infos = env.step_multiple(reference["raw_actions"].cpu())
    if len(observations) != 30:
        raise RuntimeError("Late-context incoming fixed30-action prefix ended early")
    exact(torch.stack(observations), reference["truth"]["frames"], "late incoming prefix all30 images")
    exact(np.stack([np.asarray(info["state"]) for info in infos]), reference["truth"]["states"][1:], "late incoming prefix all30 states")
    same_physics(physics_snapshot(env), reference["truth"]["physics"][-1], "late actual snapshot")
    td = make_td(observations[-1], infos[-1])
    z = wm.encode(td.to(agent.device).unsqueeze(0), act=True)
    if z["visual"].shape[1] != 1:
        raise RuntimeError("Late comparison must use native single current context frame")
    agent.capture_raw_context = {key: td[key].detach().cpu().clone() for key in ("visual", "proprio")}
    checks["late_initial_reference_sha256"] = digest
    checks["late_incoming_raw_actions_sha256"] = tensor_hash(reference["raw_actions"])
    checks["late_incoming_prefix_exact"] = True
    checks["context_history_policy"] = "native single currentframe/proprio; no action-history or architectural adapter"
    return env, agent, z, checks


class CaptureP3Horizons:
    def __init__(self, block, expected=6):
        self.block, self.expected, self.values = block, expected, []

    def capture(self, _module, _inputs, output):
        if not torch.is_tensor(output) or output.ndim != 3 or output.shape[-1] != 400 or output.shape[1] % 256:
            raise RuntimeError("Expected P3 residual [B,T*256,400]")
        self.values.append(output[:, -256:].detach().float().cpu().clone())

    def __enter__(self):
        self.handle = self.block.register_forward_hook(self.capture)
        return self

    def __exit__(self, kind, *_):
        self.handle.remove()
        if kind is None and len(self.values) != self.expected:
            raise RuntimeError(f"P3 calls {len(self.values)} != imagined horizons {self.expected}")


def geometry(base):
    import mujoco
    wall = []
    for index in range(base.model.ngeom):
        body = mujoco.mj_id2name(base.model, mujoco.mjtObj.mjOBJ_BODY, int(base.model.geom_bodyid[index]))
        name = mujoco.mj_id2name(base.model, mujoco.mjtObj.mjOBJ_GEOM, index)
        if "wall" in ((body or "") + " " + (name or "")).lower():
            wall.append({"geom_id": index, "geom_name": name, "body_name": body,
                         "type": int(base.model.geom_type[index]), "size": base.model.geom_size[index].copy(),
                         "world_position": base.data.geom_xpos[index].copy(),
                         "world_rotation": base.data.geom_xmat[index].reshape(3, 3).copy()})
    if not wall:
        raise RuntimeError("Native wall geometry unavailable")
    return {"wall_geometries": wall, "task_goal_xyz": np.asarray(base._target_pos).copy(),
            "object_xyz": np.asarray(base._get_pos_objects()).reshape(-1, 3)[0].copy(),
            "object_semantics": "Native Reach object body, not a manipulated-object efficacy endpoint"}


def same_physics(a, b, label):
    exact(a["elapsed_steps"], b["elapsed_steps"], label + " elapsed")
    exact(a["time"], b["time"], label + " time")
    for key in a["physics"]:
        exact(a["physics"][key], b["physics"][key], label + " " + key)
    for key, value in a["task_state"].items():
        exact(value, b["task_state"][key], label + " task " + key)


@torch.no_grad()
def physical_fork(env, raw_actions, wm, agent, encode=True):
    from collect_on_policy_bank import physics_snapshot
    from precompute_native_replay import ReachControlCapture
    from evals.simu_env_planning.planning.utils import make_td
    if raw_actions.shape != (30, 4):
        raise RuntimeError("Expected exact frozen30 raw actions")
    base = env.proprio_env.unwrapped
    result = {"states": [np.asarray(base._get_obs()).astype(np.float32)],
              "physics": [physics_snapshot(env)], "frames": [], "proprios": [],
              "controls": [], "encoded_visual": [], "encoded_proprio": [], "geometry": geometry(base)}
    with ReachControlCapture(base) as capture:
        for index, action in enumerate(raw_actions):
            capture.records.clear()
            frames, rewards, dones, infos = env.step_multiple(action[None].cpu())
            if len(frames) != 1:
                raise RuntimeError("Fixed-action fork ended before requested raw step")
            info = infos[0]
            result["states"].append(np.asarray(info["state"]).copy())
            result["physics"].append(physics_snapshot(env))
            result["frames"].append(frames[0].cpu().clone())
            result["proprios"].append(torch.as_tensor(info["proprio"]).cpu().clone())
            result["controls"].append({"native_xyz_setter_calls": list(capture.records),
                                       "actuator_ctrl": np.asarray(base.data.ctrl).copy(),
                                       "mocap_pos": np.asarray(base.data.mocap_pos).copy()})
            if encode and (index+1) % 5 == 0:
                encoded = wm.encode(make_td(frames[0], info).to(agent.device).unsqueeze(0), act=True)
                result["encoded_visual"].append(encoded["visual"].detach().float().cpu().clone())
                result["encoded_proprio"].append(encoded["proprio"].detach().float().cpu().clone())
        result["native_xyz_setter_source"] = capture.source
    result["states"] = np.stack(result["states"])
    result["frames"] = torch.stack(result["frames"]).to(torch.uint8)
    result["proprios"] = torch.stack(result["proprios"])
    return result


@torch.no_grad()
def capture_episode(args, row, wm, preprocessor, cfg, readout):
    from package_steering_banks import load_development, sha
    from causal_planner_forks import reconstruct, close_env
    from collect_on_policy_bank import physics_snapshot
    bank = load_development(args.project_root, row)
    start = time.monotonic()
    env, agent, z, checks = reconstruct_capture_context(args, cfg, wm, preprocessor, bank)
    initial = physics_snapshot(env)
    context = {key: z[key].detach().cpu().clone() for key in z.keys()}
    raw_context = agent.capture_raw_context
    try:
        print(json.dumps({"event": "horizon_native_cem_started", "episode": bank["episode"]}), flush=True)
        steps_left = max((env.steps_left()+1)*wm.action_skip//cfg.frameskip, 1)
        prefix = agent.plan(z.clone(), steps_left=steps_left)
        actions = agent.planner._prev_mean.detach().clone()
        if actions.shape != (6, 20) or prefix.shape != (3, 20):
            raise RuntimeError("Expected native selectedH6 mean and returnedH3 prefix")
        exact(prefix, actions[:3], "native selected prefix")
        raw = preprocessor.denormalize_actions(actions.cpu().reshape(30, 4))
        old = bank["replans"][0]["planned_actions_normalized"]
        old_error = None if args.late_reference_dirs else float((prefix.cpu().double()-old.double()).abs().max())
        baseline = wm.unroll(z.clone(), act_suffix=actions[:, None])
        with CaptureP3Horizons(wm.model.predictor.predictor_blocks[3]) as capture:
            repeated = wm.unroll(z.clone(), act_suffix=actions[:, None])
        for key in ("visual", "proprio"):
            exact(repeated[key], baseline[key], "capture hook identity " + key)
        for key in context:
            exact(z[key], context[key], "unchanged encoded context " + key)
        p3 = torch.stack(capture.values)[:, 0]
        predictions = {key: baseline[key][1:, 0].detach().float().cpu().clone() for key in ("visual", "proprio")}
        first_loss = agent.objective(baseline, actions[:, None], keepdims=True).cpu()
        truth = physical_fork(env, raw, wm, agent)
    finally:
        close_env(env)
    env, repeat_agent, repeat_z, repeat_checks = reconstruct_capture_context(args, cfg, wm, preprocessor, bank)
    try:
        if checks["fresh_goal_sha256"] != repeat_checks["fresh_goal_sha256"]:
            raise RuntimeError("Fresh paired goal encoding changed")
        same_physics(physics_snapshot(env), initial, "repeat fresh initial")
        for key in context:
            exact(repeat_z[key], context[key], "repeat encoded context " + key)
        repeat = physical_fork(env, raw, wm, repeat_agent, encode=False)
        exact(repeat["states"], truth["states"], "all30 states")
        exact(repeat["frames"], truth["frames"], "all30 image buffers")
        exact(repeat["proprios"], truth["proprios"], "all30 proprio buffers")
        for index, (a, b) in enumerate(zip(repeat["physics"], truth["physics"])):
            same_physics(a, b, f"rawstep{index}")
        for a, b in zip(repeat["controls"], truth["controls"]):
            exact(a["actuator_ctrl"], b["actuator_ctrl"], "native actuator controls")
            exact(a["mocap_pos"], b["mocap_pos"], "native mocap controls")
    finally:
        close_env(env)
    actual_visual = torch.cat(truth["encoded_visual"], 0)
    predicted_flat = predictions["visual"].reshape(6, -1, 384)
    actual_flat = actual_visual.reshape(6, -1, 384)
    if predicted_flat.shape != actual_flat.shape:
        raise RuntimeError("Predicted/actual encoded spatial layout mismatch")
    pp, ap = predicted_flat.mean(1), actual_flat.mean(1)
    states = truth["states"]
    next_hand, current_hand = states[5::5, :3], states[0:30:5, :3]
    pred_xyz, actual_xyz = decode_physical(pp, readout), decode_physical(ap, readout)
    frame = truth["geometry"]
    labels = horizon_labels(initial["elapsed_steps"], initial["time"], [x["time"] for x in truth["physics"][5::5]])
    arrays = {**labels, "episode": np.full(6, bank["episode"], dtype=np.int64),
              "context_index": np.full(6, int(bool(args.late_reference_dirs)), dtype=np.int64),
              "environment_seed": np.full(6, bank["environment_seed"], dtype=np.int64),
              "p3_pooled": p3.mean(1).numpy(), "p3_pooled_raw_norm": p3.mean(1).norm(dim=-1).numpy(),
              "predicted_visual_pooled": pp.numpy(), "actual_visual_pooled": ap.numpy(),
              "current_hand_xyz": current_hand, "next_hand_xyz": next_hand,
              "start_hand_xyz": np.repeat(states[0:1, :3], 6, axis=0),
              "goal_xyz": np.repeat(frame["task_goal_xyz"][None], 6, axis=0),
              "task_goal_xyz": np.repeat(frame["task_goal_xyz"][None], 6, axis=0),
              "expert_goal_hand_xyz": np.repeat(np.asarray(bank["goal"]["state"])[None, :3], 6, axis=0),
              "object_xyz": np.repeat(frame["object_xyz"][None], 6, axis=0),
              "wall_position": np.repeat(frame["wall_geometries"][0]["world_position"][None], 6, axis=0),
              "wall_rotation": np.repeat(frame["wall_geometries"][0]["world_rotation"][None], 6, axis=0),
              "frozen_physical_readout_predicted_xyz": pred_xyz, "frozen_physical_readout_actual_xyz": actual_xyz,
              "predicted_xyz_error": pred_xyz-next_hand, "actual_encoder_xyz_error": actual_xyz-next_hand,
              "latent_mse_to_actual": (predicted_flat-actual_flat).square().mean((1, 2)).numpy(),
              "normalized_actions": actions.cpu().numpy(), "raw_actions": raw.numpy()}
    tensor_path = args.output_dir / f"episode-{bank['episode']:03d}.pt"
    array_path = tensor_path.with_suffix(".npz")
    value = {"complete": True, "episode": bank["episode"], "split": "development", "replan": 0,
             "bank_sha256": row["sha256"], "bank_path": row["path"], "context": context,
             "raw_context_visual": raw_context["visual"],
             "raw_context_proprio": raw_context["proprio"], "goal": bank["goal"],
             "normalized_actions": actions.cpu(), "raw_actions": raw, "p3_spatial": p3,
             "predictions": predictions, "native_goal_losses": first_loss, "truth": truth,
             "arrays": arrays, "initial_checks": checks, "repeat_initial_checks": repeat_checks,
             "new_within_run_identity_exact": True, "fresh30action_physics_and_pixels_repeat_exact": True,
             "old_cached3step_prefix_max_absolute_error": old_error,
             "plan_origin": "New nativeCEM fullH6 selected mean, original saved initial RNG; old cache held onlyH3 prefix",
             "future_labels_used_by_model": False, "seconds": time.monotonic()-start}
    torch.save(value, tensor_path)
    np.savez_compressed(array_path, **arrays)
    entries = [{"path": p.name, "episode": bank["episode"], "sha256": sha(p), "bytes": p.stat().st_size} for p in (tensor_path, array_path)]
    from protocol import write_json_atomic
    write_json_atomic(tensor_path.with_suffix(".DONE.json"), {"complete": True, "outputs": entries})
    print(json.dumps({"event": "horizon_episode_complete", "episode": bank["episode"], "seconds": value["seconds"],
                      "old_prefix_error": old_error, "outputs": entries}), flush=True)
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "config", "project-root", "inventory", "readout", "output-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--episodes", nargs="+", type=int, required=True)
    parser.add_argument("--late-reference-dirs", nargs="+", type=Path)
    args = parser.parse_args()
    from package_steering_banks import sha
    from precompute_native_replay import safe_rows
    from protocol import write_json_atomic
    from model_loader import load_headless_metaworld
    from capture_specificity_controls import parameters_sha
    from causal_planner_forks import setup_cfg
    args.output_dir.mkdir(parents=True, exist_ok=False)
    inventory = json.loads(args.inventory.read_text())
    rows = safe_rows(inventory, "reach_wall", args.episodes)
    readout = dict(np.load(args.readout, allow_pickle=False))
    protocol = {"schema_version": 1, "scope": "12 predeclared Reach development first planning states; no held outcomes",
                "context_index": int(bool(args.late_reference_dirs)),
                "late_context_policy": "after exact original fresh30-action reference prefix, native single currentframe atelapsed31; state alsochanges, notclockcausality" if args.late_reference_dirs else None,
                "episodes": args.episodes, "imagined_horizons": [1, 2, 3, 4, 5, 6], "raw_actions_per_horizon": 5,
                "plan_origin": "fresh nativeH6/300/15 CEM mean from original initial RNG, not a cached six-step plan",
                "old_prefix_policy": "record discrepancy, no selection or exact-old-action prerequisite",
                "goal_encoding_policy": "original raw goal pixels/proprio exact; fresh fullprecision encoding hashed and identical across newpairedarms; oldfloat16 discrepancy descriptive only",
                "truth_origin": "two newly reconstructed identical initial states executing same frozen30rawactions, no replans",
                "time_semantics": "real_raw_step is future simulator step; context_real_raw_step remains start time; imagined_step is distinct",
                "frame_semantics": "world axes; task_goal_xyz distinct from expert_goal_hand_xyz; current_hand is true preceding horizon, start_hand is horizon0; future labels never supplied to unroll",
                "batch_semantics": "selected plan batch1 baseline/capture identity; no comparison to candidate batch300 exactness",
                "readout_sha256": sha(args.readout), "inventory_sha256": sha(args.inventory),
                "script_sha256": sha(Path(__file__)), "source_rows": rows}
    write_json_atomic(args.output_dir / "protocol.json", protocol)
    try:
        wm, preprocessor, provenance = load_headless_metaworld(args.repo)
        wm.eval().requires_grad_(False)
        parameter_sha = parameters_sha(wm)
        checkpoint_sha = sha(Path(provenance["checkpoint"]))
        if any(row["model"]["checkpoint_sha256"] != checkpoint_sha for row in rows):
            raise RuntimeError("Checkpoint differs from original development source")
        cfg = setup_cfg(args.config, args.output_dir, wm)
        outputs = []
        for row in rows:
            outputs.extend(capture_episode(args, row, wm, preprocessor, cfg, readout))
        if parameters_sha(wm) != parameter_sha:
            raise RuntimeError("Frozen parameters changed")
        report = args.output_dir / "horizon_coordinates.json"
        write_json_atomic(report, {**protocol, "complete": True, "model": provenance, "checkpoint_sha256": checkpoint_sha,
                                  "parameters_unchanged": True, "parameters_sha256": parameter_sha, "outputs": outputs})
        outputs += [{"path": p.name, "sha256": sha(p), "bytes": p.stat().st_size} for p in (report, args.output_dir/"protocol.json")]
        write_json_atomic(args.output_dir / "DONE.json", {"complete": True, "outputs": outputs})
    except Exception as exc:
        write_json_atomic(args.output_dir / "FAILED.json", {"complete": False, "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
