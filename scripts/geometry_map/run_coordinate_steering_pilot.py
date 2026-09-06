#!/usr/bin/env python3
"""Fresh full-native Reach development episodes: baseline/calibration/sham.

No old trajectory continuation, validation tuning, or held performance-bank use.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time

import numpy as np
import torch
from einops import rearrange

from causal_planner_forks import checked_max_error, close_env, load_development_bank, restore_cached_goal, setup_cfg
from collect_on_policy_bank import block_list, initialize_episode, physics_snapshot
from coordinate_calibration_operator import CoordinateCalibration, CoordinateCalibrationHook
from model_loader import load_headless_metaworld
from protocol import file_sha256, write_json_atomic

ARMS = ("unsteered", "static_candidate", "matched_sham")
DEVELOPMENT_EPISODES = (0,1,4,7,10)


def emit(event, **values):
    print(json.dumps({"event":event, **values}), flush=True)


def paired_start_checks(actual, expected):
    checks = {}
    for name in ("visual", "proprio", "goal_visual", "goal_proprio", "goal_encoded_visual", "goal_encoded_proprio", "planner_rng"):
        checks[name] = checked_max_error(actual[name], expected[name], "fresh paired " + name, 0)
    for name in ("qpos", "qvel", "act", "mocap_pos", "mocap_quat", "userdata"):
        checks[name] = checked_max_error(actual["physics"]["physics"][name], expected["physics"]["physics"][name], "fresh paired physics " + name, 0)
    checks["time"] = checked_max_error(actual["physics"]["time"], expected["physics"]["time"], "fresh paired time", 0)
    if actual["physics"]["elapsed_steps"] != expected["physics"]["elapsed_steps"]:
        raise RuntimeError("Fresh pair elapsed steps differ")
    for name,value in expected["physics"]["task_state"].items():
        if isinstance(value,str):
            if actual["physics"]["task_state"][name] != value:
                raise RuntimeError("Fresh pair task string differs")
        else:
            checks["task."+name] = checked_max_error(actual["physics"]["task_state"][name],value,"fresh paired task " + name,0)
    return checks


def fresh_start(cfg, wm, preprocessor, bank):
    from evals.simu_env_planning.envs.init import make_env
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    from evals.simu_env_planning.planning.plan_evaluator import PlanEvaluator
    from evals.simu_env_planning.planning.utils import set_seed
    set_seed(cfg.local_seed)
    env=make_env(cfg)
    agent=GC_Agent(cfg,wm,dset=None,preprocessor=preprocessor)
    evaluator=PlanEvaluator(cfg,agent)
    td, _, generated_goal, _, _ = initialize_episode(cfg,agent,env,evaluator,bank["episode"],bank["environment_seed"])
    generated_error=float(np.abs(generated_goal["visual"].cpu().numpy().astype(float)-np.asarray(bank["goal"]["visual"]).astype(float)).max())
    goal=restore_cached_goal(generated_goal,bank["goal"])
    agent.set_goal(goal)  # Fresh full-precision same-worker encoding, never old f16 restoration.
    agent.local_gpu_generator.set_state(bank["replans"][0]["planner_rng_before"])
    snapshot={"visual":td["visual"].detach().cpu().clone(),"proprio":td["proprio"].detach().cpu().clone(),
              "goal_visual":goal["visual"].detach().cpu().clone(),"goal_proprio":goal["proprio"].detach().cpu().clone(),
              "goal_encoded_visual":agent.goal_state_enc["visual"].detach().cpu().clone(),
              "goal_encoded_proprio":agent.goal_state_enc["proprio"].detach().cpu().clone(),
              "planner_rng":agent.local_gpu_generator.get_state().cpu().clone(), "physics":physics_snapshot(env)}
    # Fixed-start identity, not an old plan/trajectory-continuation requirement.
    original=bank["replans"][0]
    checked_max_error(snapshot["visual"],original["observation_visual"],"original fixed-start pixels",0)
    checked_max_error(snapshot["proprio"],original["observation_proprio"],"original fixed-start proprio")
    for name,value in original["simulator"]["physics"].items():
        checked_max_error(snapshot["physics"]["physics"][name],value,"original initial physics " + name)
    return env,agent,td,snapshot,{"mode":"original cached raw goal; fresh same-worker full-precision encoding",
                                 "regenerated_goal_pixel_max_error":generated_error,"old_cached_plan_match_required":False}


def native_identity_canary(wm, block, calibration, z, bank, agent):
    action=bank["replans"][0]["planned_actions_normalized"][:1]
    actions=action[:,None].expand(6,3,-1).contiguous().to(agent.device)
    base=wm.unroll(z,act_suffix=actions)
    with CoordinateCalibrationHook(wm,block,calibration,beta=0):
        identical=wm.unroll(z,act_suffix=actions)
    for key in ("visual","proprio"):
        if not torch.equal(base[key],identical[key]):
            raise RuntimeError("Native H6 beta-zero complete trace identity failed: " + key)
    if not torch.equal(agent.objective(base,actions),agent.objective(identical,actions)):
        raise RuntimeError("Native beta-zero objective identity failed")
    # Actual native nonzero tuple/layout exercise before paid episode optimization.
    native_one=wm.unroll(z,act_suffix=actions[:1])
    with CoordinateCalibrationHook(wm,block,calibration) as patch:
        changed=wm.unroll(z,act_suffix=actions[:1])
    if patch.calls!=1 or not torch.isfinite(changed["visual"]).all():
        raise RuntimeError("Native one-step correction canary failed")
    if not torch.equal(changed["proprio"],native_one["proprio"]):
        raise RuntimeError("Immediate predicted proprio was altered by visual-only hook")
    return {"beta_zero_H6_visual_proprio_objective_exact":True,"native_layout_nonzero_passed":True,
            "first_step_proprio_unchanged":True,"calibration":patch.statistics()}


@torch.no_grad()
def run_arm(cfg, wm, preprocessor, bank, calibration, block, arm, baseline_start, deadline, canary=False):
    from evals.simu_env_planning.planning.utils import make_td
    env,agent,td,start,goal_mode=fresh_start(cfg,wm,preprocessor,bank)
    paired_checks=paired_start_checks(start,baseline_start) if baseline_start is not None else {}
    initial_hand=start["proprio"].numpy().reshape(-1,4)[-1,:3]
    base_env=env.proprio_env.unwrapped
    goal_xyz=np.asarray(base_env._target_pos).copy()
    z=wm.encode(td.to(agent.device).unsqueeze(0),act=True)
    technical_canary=native_identity_canary(wm,block,calibration,z,bank,agent) if canary else None
    patch=CoordinateCalibrationHook(wm,block,calibration,sham=arm=="matched_sham")
    context=nullcontext() if arm=="unsteered" else patch
    plans,actions,states,frames,rewards,successes=[],[],[],[],[],[]
    first_candidates={}
    native_unroll=agent.planner.unroll

    def tracked(*a,**kw):
        candidate_actions=kw.get("act_suffix",a[1] if len(a)>1 else None)
        prediction=native_unroll(*a,**kw)
        if not first_candidates and candidate_actions is not None and candidate_actions.shape[1]==300:
            first_candidates.update(actions=candidate_actions.detach().cpu().clone(),
                                    costs=agent.objective(prediction,candidate_actions).detach().cpu().clone())
        return prediction

    agent.planner.unroll=tracked
    began=time.monotonic()
    try:
        with context:
            done=False
            while not done:
                if time.monotonic()>deadline:
                    raise RuntimeError("Frozen worker budget elapsed; no automatic extra episodes")
                steps_left=max((env.steps_left()+1)*wm.action_skip//cfg.frameskip,1)
                z=wm.encode(td.to(agent.device).unsqueeze(0),act=True)
                planned=agent.plan(z,steps_left=steps_left)
                raw=preprocessor.denormalize_actions(rearrange(planned.cpu(),"t (f d) -> (t f) d",d=4))
                observations,chunk_rewards,dones,infos=env.step_multiple(raw)
                if not observations:
                    raise RuntimeError("Native planner produced no physical transition")
                count=len(observations)
                actions.append(raw[:count].detach().cpu().clone()); plans.append(planned.detach().cpu().clone())
                states.extend(np.asarray(info["state"]).copy() for info in infos)
                frames.extend(o.detach().cpu().to(torch.uint8) for o in observations)
                rewards.extend(float(x) for x in chunk_rewards); successes.extend(bool(info["success"]) for info in infos)
                td=make_td(observations[-1],infos[-1]); done=bool(dones[-1])
                emit("pilot_replan_complete",episode=bank["episode"],arm=arm,replan=len(plans),raw_steps=len(states),seconds=time.monotonic()-began)
        if len(states)!=99:
            raise RuntimeError(f"Expected full native99raw-step episode, got {len(states)}")
        state_array=np.stack(states)
        final_distance=float(np.linalg.norm(state_array[-1,:3]-goal_xyz))
        return {"complete":True,"episode":bank["episode"],"arm":arm,"initial_snapshot":start,"paired_start_checks":paired_checks,
                "goal_replay":goal_mode,"technical_canary":technical_canary,"initial_hand_xyz":initial_hand,
                "physical_task_goal_xyz":goal_xyz,"states":state_array,"frames":torch.stack(frames),
                "actions_raw":torch.cat(actions),"plans":plans,"first_candidates":first_candidates,
                "final_goal_distance_m":final_distance,"final_success":successes[-1],"ever_success":any(successes),
                "total_reward":sum(rewards),"raw_steps":len(states),"seconds":time.monotonic()-began,
                "operator_statistics":patch.statistics(),"ground_truth_performance_source":"actual simulator hand/goal and native success; not calibration readouts"}
    finally:
        close_env(env)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("repo","candidate-dir","output-dir"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--banks",type=Path,nargs="+",required=True)
    parser.add_argument("--time-limit-seconds",type=float,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    began=time.monotonic();deadline=began+args.time_limit_seconds
    torch.set_num_threads(2)
    frozen=json.loads((args.candidate_dir/"FROZEN.json").read_text())
    fit_done=json.loads((args.candidate_dir/"DONE.json").read_text())
    candidate_path=args.candidate_dir/"coordinate_candidate.npz"
    if not frozen.get("frozen") or not fit_done.get("complete") or file_sha256(candidate_path)!=frozen["candidate_sha256"]:
        raise RuntimeError("Frozen completed candidate fit/validation receipt required")
    if file_sha256(Path(__file__).with_name("coordinate_calibration_operator.py"))!=frozen["operator_script_sha256"]:
        raise RuntimeError("Operator source changed after fitting; refuse silent change")
    banks=[load_development_bank(path) for path in args.banks]
    if any(bank["episode"] not in DEVELOPMENT_EPISODES for bank,_ in banks) or len({bank["episode"] for bank,_ in banks})!=len(banks):
        raise RuntimeError("Only predetermined unique development starts allowed")
    wm,preprocessor,provenance=load_headless_metaworld(args.repo)
    for parameter in wm.parameters(): parameter.requires_grad_(False)
    cfg=setup_cfg(Path(provenance["config"]),args.output_dir,wm)
    _,blocks=block_list(wm.model.predictor,("predictor_blocks","blocks"),6,"predictor")
    calibration=CoordinateCalibration(dict(np.load(candidate_path)),device="cuda:0")
    protocol={"pilot":"coordinate-calibrated Sonar; exploratory development performance",
              "episodes":[bank["episode"] for bank,_ in banks],"arms":list(ARMS),"native_cem":{"H":6,"samples":300,"iterations":15},
              "candidate_sha256":frozen["candidate_sha256"],"frozen_receipt_sha256":file_sha256(args.candidate_dir/"FROZEN.json"),
              "model":provenance,"checkpoint_sha256":file_sha256(Path(provenance["checkpoint"])),
              "script_sha256":file_sha256(Path(__file__)),"time_limit_seconds":args.time_limit_seconds,
              "full_raw_steps":99,"fresh_unsteered_baseline":True,"held_performance_data_opened":False,
              "operator":frozen,"sham":"fixed orthogonal det+1 alternating-sign rotation, matched instantaneous raw delta norm",
              "sham_limit":"adaptive candidate populations can diverge; no equal cumulative cross-arm dose claim"}
    write_json_atomic(args.output_dir/"protocol.json",protocol)
    outputs=[]
    try:
        for bank,bank_sha in banks:
            episode_root=args.output_dir/f"episode-{bank['episode']:03d}"
            episode_root.mkdir(exist_ok=False)
            baseline_start,baseline_candidates=None,None
            for arm in ARMS:
                emit("pilot_arm_started",episode=bank["episode"],arm=arm)
                result=run_arm(cfg,wm,preprocessor,bank,calibration,blocks[3],arm,baseline_start,deadline,canary=arm=="unsteered")
                if arm=="unsteered":
                    baseline_start=result["initial_snapshot"];baseline_candidates=result["first_candidates"]["actions"]
                elif not torch.equal(result["first_candidates"]["actions"],baseline_candidates):
                    raise RuntimeError("Fresh paired first300 CEM actions differ")
                result["bank_sha256"]=bank_sha
                path=episode_root/f"{arm}.pt";torch.save(result,path)
                row={"path":str(path.relative_to(args.output_dir)),"sha256":file_sha256(path),"episode":bank["episode"],"arm":arm,
                     **{key:result[key] for key in ("final_goal_distance_m","final_success","ever_success","raw_steps","seconds","operator_statistics")}}
                outputs.append(row)
                write_json_atomic(args.output_dir/"progress.json",{"complete":False,"outputs":outputs})
                emit("pilot_arm_complete",**row)
                del result
        write_json_atomic(args.output_dir/"DONE.json",{**protocol,"complete":True,"outputs":outputs,"seconds":time.monotonic()-began})
    except Exception as exc:
        write_json_atomic(args.output_dir/"FAILED.json",{"complete":False,"error":str(exc),"outputs":outputs})
        raise


if __name__=="__main__":
    main()
