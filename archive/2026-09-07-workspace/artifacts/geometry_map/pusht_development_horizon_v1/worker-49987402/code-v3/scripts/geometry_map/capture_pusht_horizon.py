#!/usr/bin/env python3
"""Fixed cached nativeH6 Push development plans, forecasts and paired physical truth."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from capture_horizon_coordinates import CaptureP3Horizons, exact, pool_visual
from capture_pusht_calibration import parameter_sha
from collect_pusht_bank import config, snapshot
from precompute_native_replay import PushContactCapture
from complete_cached_geometry import sha256, write_json

CHECKPOINT = "9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb"


def select_rows(manifest, episodes):
    if manifest.get("task") != "pusht_original_development" or manifest.get("checkpoint_sha256") != CHECKPOINT:
        raise ValueError("Wrong task/checkpoint")
    rows = {int(r["episode"]): r for r in manifest["rows"]}
    if len(rows) != 10 or sorted(rows) != list(range(10)): raise ValueError("Must retain all original10 development sources")
    if not episodes or len(set(episodes)) != len(episodes): raise ValueError("Invalid shard")
    selected = []
    for episode in episodes:
        if episode not in rows or rows[episode]["split"] != "development" or rows[episode]["sealed"]:
            raise ValueError("Held tensor access forbidden")
        selected.append(rows[episode])
    return selected


def validate_plan(bank, preprocessor):
    if len(bank["replans"]) != 1: raise ValueError("Expected one native full episode plan; no automatic CEM reconstruction")
    row = bank["replans"][0]
    plan = row["planned_actions_normalized"].float().cpu()
    actions = bank["actions_raw"].float().cpu()
    if plan.shape != (6, 10) or actions.shape != (30, 2) or row["executed_action_count"] != 30:
        raise ValueError("Saved nativeH6 plan incomplete; no automatic new optimization")
    exact(preprocessor.denormalize_actions(plan.reshape(30, 2)), actions, "cached normalized-to-raw action conversion")
    exact(row["planned_actions_raw"], actions, "saved full action sequence")
    return plan, actions


def temporal_labels():
    return {"imagined_step": np.arange(1, 7), "raw_action_offset": np.arange(5, 31, 5),
            "real_raw_step": np.arange(5, 31, 5), "context_real_raw_step": np.zeros(6, dtype=np.int64),
            "preceding_actual_raw_step": np.arange(0, 26, 5)}


def encoder_inputs(frames, proprios):
    if frames.shape[0] != 7 or frames.shape[-3] != 3 or proprios.numel() != 7*4:
        raise ValueError("Expected seven RGB observations and seven4D proprio states")
    height,width=frames.shape[-2:]
    if frames.numel()!=7*3*height*width:raise ValueError("Unexpected non-singleton observation axes")
    return {"visual":frames.reshape(1,7,3,height,width),"proprio":proprios.reshape(1,7,4)}


def error(actual, expected):
    a = actual.detach().cpu().numpy() if torch.is_tensor(actual) else np.asarray(actual)
    b = expected.detach().cpu().numpy() if torch.is_tensor(expected) else np.asarray(expected)
    return float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.shape == b.shape else None


@torch.no_grad()
def physical_fork(env, observation, info, actions):
    frames, states, proprios, physics, controls, contacts = [observation.cpu().clone()], [np.asarray(info["state"]).copy()], [torch.as_tensor(info["proprio"]).cpu().clone()], [snapshot(env)], [], []
    with PushContactCapture(env.unwrapped) as capture:
        for step, command in enumerate(actions):
            capture.records.clear()
            obs, _reward, _done, infos = env.step_multiple(command[None])
            if len(obs) != 1 or env.elapsed_steps() != step+1: raise RuntimeError("Unexpected early stop/raw elapsed step")
            frames.append(obs[0].cpu().clone()); states.append(np.asarray(infos[0]["state"]).copy())
            proprios.append(torch.as_tensor(infos[0]["proprio"]).cpu().clone())
            physics.append(snapshot(env))
            controls.append(np.asarray(env.unwrapped.latest_action).copy())
            contacts.append({"raw_step": step+1, "records": list(capture.records),
                             "agent_block_contact": any(r["agent_block_contact"] for r in capture.records) if capture.available else None})
        available = capture.available
    return {"frames": torch.stack(frames).to(torch.uint8), "states": np.stack(states),
            "proprios": torch.stack(proprios), "physics": physics, "native_applied_targets_xy": np.stack(controls),
            "contacts": contacts, "contact_api_available": available}


def compare_physics(first, second):
    if len(first) != len(second): raise RuntimeError("Different snapshot count")
    # The native snapshot uses only finite Python scalar/list fields.
    if json.dumps(first, sort_keys=True) != json.dumps(second, sort_keys=True): raise RuntimeError("Fresh paired full-body physics mismatch")


@torch.no_grad()
def capture_episode(args, row, wm, preprocessor, cfg):
    from evals.simu_env_planning.envs.init import make_env
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    from evals.simu_env_planning.planning.utils import make_td
    # Manifest split is checked before checksum/tensor load; all sources immutable.
    for key in ("baseline", "input"):
        if sha256(args.inputs/row[key+"_file"]) != row[key+"_sha256"]: raise RuntimeError("Input checksum mismatch")
    bank = torch.load(args.inputs/row["baseline_file"], map_location="cpu", weights_only=False)
    stimulus = torch.load(args.inputs/row["input_file"], map_location="cpu", weights_only=False)
    if bank["split"] != "development" or stimulus["split"] != "development": raise ValueError("Tensor split mismatch")
    if bank["input_sha256"] != row["input_sha256"] or bank["episode"] != row["episode"]: raise ValueError("Wrong paired input")
    plan, actions = validate_plan(bank, preprocessor)
    started = time.monotonic()
    def initialize():
        env = make_env(cfg)
        env.update_env(stimulus["env_info"])
        obs, info = env.prepare(stimulus["environment_seed"], stimulus["initial_state"], stimulus["env_info"])
        return env, obs, info
    env, obs, info = initialize()
    agent = GC_Agent(cfg, wm, dset=None, preprocessor=preprocessor)
    goal = make_td(stimulus["expert_observations"][-1], {"proprio": stimulus["expert_proprios"][-1]})
    agent.set_goal(goal)
    context = wm.encode(make_td(obs, info).to(agent.device).unsqueeze(0), act=True)
    z_copy = {k: context[k].detach().cpu().clone() for k in context.keys()}
    try:
        forecast = wm.unroll(context.clone(), act_suffix=plan[:, None].to(agent.device))
        with CaptureP3Horizons(wm.model.predictor.predictor_blocks[3]) as captured:
            repeated = wm.unroll(context.clone(), act_suffix=plan[:, None].to(agent.device))
        for key in ("visual", "proprio"):
            exact(forecast[key], repeated[key], "native/captured forecast identity "+key)
            exact(context[key], z_copy[key], "unchanged context "+key)
        p3 = torch.stack(captured.values)[:, 0]
        predictions = {k: forecast[k][1:, 0].float().cpu().clone() for k in ("visual", "proprio")}
        if predictions["visual"].shape[0] != 6: raise RuntimeError("Initial context not removed correctly")
        costs = agent.objective(forecast, plan[:, None].to(agent.device), keepdims=True).cpu()
        truth = physical_fork(env, obs, info, actions)
    finally: env.close()
    second, second_obs, second_info = initialize()
    try:
        repeated_truth = physical_fork(second, second_obs, second_info, actions)
        for key in ("frames", "states", "proprios", "native_applied_targets_xy"):
            exact(truth[key], repeated_truth[key], "fresh paired30-step "+key)
        compare_physics(truth["physics"], repeated_truth["physics"])
    finally: second.close()
    indices = np.arange(0, 31, 5)
    # Native buffers are [T,1,C,H,W]/[T,1,P]. Preserve raw records and
    # canonicalize only encoder inputs to the documented [B,T,C,H,W]/[B,T,P].
    actual = wm.encode(encoder_inputs(truth["frames"][indices],truth["proprios"][indices]))
    actual_visual = actual["visual"][0, 1:].float().cpu()
    pred_visual = predictions["visual"]
    if actual_visual.shape != pred_visual.shape: raise RuntimeError("Actual/predicted spatial layouts differ")
    pp, ap = pool_visual(pred_visual), pool_visual(actual_visual)
    states = truth["states"]
    actual_next = states[5::5]
    actual_current = states[:30:5]
    goal_state = np.asarray(stimulus["goal_state"])
    old = {"raw_images_max_abs": error(truth["frames"], bank["observations"]),
           "state_max_abs": error(states, bank["states"]), "proprio_max_abs": error(truth["proprios"], bank["proprios"]),
           "encoded_start_half_max_abs": error(z_copy["visual"].half(), bank["replans"][0]["encoded_visual"]),
           "comparison_is_not_selection_or_new_pair_identity_gate": True}
    arrays = {**temporal_labels(), "episode": np.full(6, row["episode"]),
        "environment_seed": np.full(6, stimulus["environment_seed"]), "p3_pooled": p3.mean(1).numpy(),
        "predicted_visual_pooled": pp.numpy(), "actual_visual_pooled": ap.numpy(),
        "current_agent_xy": actual_current[:, :2], "current_block_xy": actual_current[:, 2:4],
        "next_agent_xy": actual_next[:, :2], "next_block_xy": actual_next[:, 2:4],
        "next_block_sin_cos": np.column_stack([np.sin(actual_next[:, 4]), np.cos(actual_next[:, 4])]),
        "current_block_sin_cos": np.column_stack([np.sin(actual_current[:, 4]), np.cos(actual_current[:, 4])]),
        "start_agent_xy": np.repeat(states[:1, :2], 6, axis=0), "start_block_xy": np.repeat(states[:1, 2:4], 6, axis=0),
        "goal_agent_xy": np.repeat(goal_state[None, :2], 6, axis=0), "goal_block_xy": np.repeat(goal_state[None, 2:4], 6, axis=0),
        "goal_block_sin_cos": np.repeat(np.array([[np.sin(goal_state[4]),np.cos(goal_state[4])]]),6,axis=0),
        "latent_mse_to_actual": (pred_visual-actual_visual).square().reshape(6,-1).mean(1).numpy(),
        "normalized_actions": plan.numpy(), "raw_actions": actions.numpy()}
    value = {"complete": True, "episode": row["episode"], "split": "development", "replan": 0,
        "baseline_sha256": row["baseline_sha256"], "input_sha256": row["input_sha256"], "context": z_copy,
        "goal": {"raw_visual": goal["visual"].cpu(), "raw_proprio": goal["proprio"].cpu(),
                 "encoded_visual": agent.goal_state_enc["visual"].float().cpu(), "encoded_proprio": agent.goal_state_enc["proprio"].float().cpu(),
                 "state": goal_state, "origin": "immutable original development expert replay endpoint, not painted target"},
        "normalized_actions": plan, "raw_actions": actions, "predictions": predictions, "p3_spatial": p3,
        "actual_visual_spatial": actual_visual, "native_goal_losses": costs, "truth": truth, "arrays": arrays,
        "old_cache_comparison": old, "new_within_run_identity_exact": True, "fresh30action_physics_and_pixels_repeat_exact": True,
        "plan_origin": "existing complete nativeH6/30-action selected plan; zero CEM calls", "future_labels_used_by_model": False,
        "physical_readout_error_status": "not computed; no validated Push physical readout supplied; actual physical truth+encoded error retained",
        "seconds": time.monotonic()-started}
    tensor = args.output_dir/f"episode-{row['episode']:03d}.pt"
    array = tensor.with_suffix(".npz")
    torch.save(value, tensor); np.savez_compressed(array, **arrays)
    outputs = [{"path": p.name,"episode":row["episode"],"sha256":sha256(p),"bytes":p.stat().st_size} for p in (tensor,array)]
    write_json(tensor.with_suffix(".DONE.json"), {"complete": True,"outputs":outputs})
    print(json.dumps({"event":"push_horizon_complete","episode":row["episode"],"seconds":value["seconds"],"old_cache":old}),flush=True)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("manifest","inputs","repo","checkpoint","output-dir"): parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--episodes",nargs="+",type=int,required=True)
    args = parser.parse_args()
    rows = select_rows(json.loads(args.manifest.read_text()),args.episodes)
    if sha256(args.checkpoint) != CHECKPOINT: raise RuntimeError("Wrong native checkpoint")
    args.output_dir.mkdir(parents=True,exist_ok=False)
    write_json(args.output_dir/"protocol.json", {"scope":"original10development fixedcachedH6 selected actions; no heldscripted outcomes",
        "source_rows":rows,"manifest_sha256":sha256(args.manifest),"script_sha256":sha256(__file__),
        "helper_sha256":sha256(Path(__file__).with_name("capture_horizon_coordinates.py")),
        "horizons":6,"modelsteps_rawactions":5,"cem_calls":0,"new_pair_exact_required":True,"gpu_hour_cap":.5,
        "physical_time_seconds":"not asserted; raw environment step indices saved separately from imaginedhorizon"})
    from model_loader import load_headless
    cfg = config(args.repo,args.output_dir)
    from evals.simu_env_planning.planning.utils import set_seed
    set_seed(cfg.local_seed)
    wm,preprocessor,provenance = load_headless(args.repo,model_name="jepa_wm_pusht",checkpoint_override=args.checkpoint)
    wm.eval().requires_grad_(False)
    before = parameter_sha(wm)
    started=time.monotonic(); outputs=[]
    for row in rows:
        if time.monotonic()-started>850: raise RuntimeError("Perworker bounded processing time exhausted")
        outputs.extend(capture_episode(args,row,wm,preprocessor,cfg))
    if parameter_sha(wm)!=before:raise RuntimeError("Frozen model parameters changed")
    write_json(args.output_dir/"DONE.json", {"complete":True,"outputs":outputs,"episodes":args.episodes,
        "seconds":time.monotonic()-started,"checkpoint_sha256":CHECKPOINT,"parameter_sha256":before,
        "provenance":provenance,"cem_calls":0,"held_scripted_outcomes_accessed":False})


if __name__ == "__main__": main()
