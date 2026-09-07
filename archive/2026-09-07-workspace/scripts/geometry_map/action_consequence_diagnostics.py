#!/usr/bin/env python3
"""Same-state Push action bank and trained residual-subspace interventions."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from capture_pusht_horizon import physical_fork,compare_physics,encoder_inputs
from capture_horizon_coordinates import CaptureP3Horizons,exact,pool_visual
from collect_pusht_bank import config
from capture_pusht_calibration import parameter_sha,physical_q
from complete_cached_geometry import sha256,write_json

CHECKPOINT="9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb"
NAMES=("reference","zero","pulse_x_plus","pulse_x_minus","pulse_y_plus","pulse_y_minus","reference_reversed")

def action_bank(reference):
    if reference.shape!=(30,2):raise ValueError("Expected30 native relativeXY actions")
    bank=torch.zeros((7,30,2),dtype=torch.float32)
    bank[0]=reference.float();bank[6]=-reference.float()
    for i,(axis,sign) in enumerate(((0,1),(0,-1),(1,1),(1,-1)),2):bank[i,:5,axis]=.1*sign
    return bank

def physical_metrics(states,goal):
    states=np.asarray(states);goal=np.asarray(goal)
    xy=np.linalg.norm(states[...,:4]-goal[:4],axis=-1)
    block=np.linalg.norm(states[...,2:4]-goal[2:4],axis=-1)
    delta=states[...,4]-goal[4]
    angle=np.abs(np.arctan2(np.sin(delta),np.cos(delta)))
    native_angle=np.minimum(np.abs(delta),2*np.pi-np.abs(delta))
    return {"goal_xy_distance":xy,"goal_block_distance":block,"wrapped_angle_error":angle,
        "robust_native_success":(xy<20)&(angle<np.pi/9),"unchanged_native_success":(xy<20)&(native_angle<np.pi/9),
        "native_angle_error":native_angle}

def actual_cost_state(actual,forecast):
    result=forecast.clone()
    for key in ("visual","proprio"):
        value=actual[key].transpose(0,1)
        if value.shape!=forecast[key].shape:raise ValueError("Actual/native cost tensor layout differs")
        result[key]=value
    return result

def select_inputs(args):
    receipt=json.loads((args.inputs/"DONE.json").read_text())
    if not receipt.get("complete"):raise ValueError("Incomplete source inputs")
    if args.panel=="diverse":
        available={r["source_id"]:r for r in receipt["rows"]}
        expected=list(range(202,2122,101))
        if sorted(available)!=expected:raise ValueError("Wrong frozen20 sources")
        rows=[]
        for source in args.source_ids:
            r=available[source]
            if r["split"] not in ("fit","development_validation"):raise ValueError("Forbidden source split")
            rows.append({**r,"key":f"train-{source:04d}","initial_seed":90505,"panel":"diverse"})
    else:
        if any(i not in range(4) for i in args.source_ids):raise ValueError("Only originaldevelopment0..3 in this frozen extension")
        raw=receipt.get("episodes",receipt.get("outputs",[]));available={r["episode"]:r for r in raw}
        rows=[]
        for source in args.source_ids:
            r=available[source]
            if r.get("split")!="development":raise ValueError("Held original source forbidden before hash/load")
            rows.append({**r,"source_id":source,"key":f"official-dev-{source:03d}","panel":"original","split":"development_external"})
    if len({r["key"] for r in rows})!=len(rows):raise ValueError("Duplicate inputs")
    return rows

def load_stimulus(args,row):
    path=args.inputs/row["path"]
    if sha256(path)!=row["sha256"]:raise ValueError("Input checksum mismatch before loading")
    bank=torch.load(path,map_location="cpu",weights_only=args.panel=="diverse")
    if args.panel=="diverse":return bank["states"][0].numpy(),bank["raw_actions"],90505,bank["states"][-1].numpy()
    if bank["split"]!="development" or bank["episode"]!=row["source_id"]:raise ValueError("Wrong tensor development split")
    return np.asarray(bank["initial_state"]),bank["expert_actions_raw"],bank["environment_seed"],np.asarray(bank["goal_state"])

@torch.no_grad()
def capture_source(args,row,wm,preprocessor,cfg):
    from evals.simu_env_planning.envs.init import make_env
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    from evals.simu_env_planning.planning.utils import make_td
    initial,reference,seed,old_goal=load_stimulus(args,row);actions=action_bank(reference);start=time.monotonic()
    def initialize():
        env=make_env(cfg);env.update_env({"shape":"T"});obs,info=env.prepare(seed,initial,{"shape":"T"})
        if env.unwrapped.window_size!=512 or not env.unwrapped.relative or env.unwrapped.action_scale!=100:raise ValueError("Native control domain changed")
        return env,obs,info
    truths=[]
    for candidate in range(7):
        env,obs,info=initialize()
        try:truth=physical_fork(env,obs,info,actions[candidate])
        finally:env.close()
        env,obs,info=initialize()
        try:repeat=physical_fork(env,obs,info,actions[candidate])
        finally:env.close()
        for key in ("frames","states","proprios","native_applied_targets_xy"):exact(truth[key],repeat[key],"candidate paired truth "+key)
        compare_physics(truth["physics"],repeat["physics"])
        if truths:
            for key in ("frames","states","proprios"):exact(truth[key][0],truths[0][key][0],"same candidate initial "+key)
        truths.append(truth)
    agent=GC_Agent(cfg,wm,dset=None,preprocessor=preprocessor)
    goal=make_td(truths[0]["frames"][-1],{"proprio":truths[0]["proprios"][-1]})
    agent.set_goal(goal)
    z=wm.encode(make_td(truths[0]["frames"][0],{"proprio":truths[0]["proprios"][0]}).to(agent.device).unsqueeze(0),act=True)
    context={k:z[k].detach().cpu().clone() for k in z.keys()}
    normalized=preprocessor.normalize_actions(actions.reshape(7,6,5,2)).reshape(7,6,10)
    records=[]
    for candidate,truth in enumerate(truths):
        plan=normalized[candidate,:,None].to(agent.device)
        native=wm.unroll(z.clone(),act_suffix=plan)
        with CaptureP3Horizons(wm.model.predictor.predictor_blocks[3]) as cap:forecast=wm.unroll(z.clone(),act_suffix=plan)
        for key in ("visual","proprio"):exact(native[key],forecast[key],"native captured forecast "+key)
        p3=torch.stack(cap.values)[:,0].float().cpu()
        actual=wm.encode(encoder_inputs(truth["frames"][::5],truth["proprios"][::5]))
        actual_for_cost=actual_cost_state(actual,forecast)
        pred=forecast["visual"][1:,0].float().cpu();av=actual["visual"][0,1:].float().cpu()
        if pred.shape!=av.shape:raise ValueError("Forecast truth layout differs")
        cost=agent.objective(forecast,plan,keepdims=True).detach().cpu()
        actual_cost=agent.objective(actual_for_cost,plan,keepdims=True).detach().cpu()
        states=truth["states"];metrics=physical_metrics(states[5::5],truths[0]["states"][-1])
        record={"name":NAMES[candidate],"p3_pooled":p3.mean(1),"predicted_visual":pred,"actual_visual":av,
            "predicted_visual_pooled":pool_visual(pred),"actual_visual_pooled":pool_visual(av),
            "native_cost_by_horizon":cost,"actual_encoded_native_cost_by_horizon":actual_cost,
            "latent_mse_to_actual":(pred-av).square().reshape(6,-1).mean(1),"metrics":metrics,
            "states":states,"actual_proprio":truth["proprios"][::5],"contacts":truth["contacts"],"physics":truth["physics"],
            "contact_api_available":truth["contact_api_available"],"native_applied_targets_xy":truth["native_applied_targets_xy"],
            "frames_modelsteps":truth["frames"][::5],"predicted_proprio":forecast["proprio"][1:,0].float().cpu()}
        records.append(record)
    value={"complete":True,"row":row,"context":context,"raw_goal":{k:goal[k].cpu() for k in goal.keys()},
        "goal_encoded":{k:agent.goal_state_enc[k].cpu() for k in ("visual","proprio")},"goal_state":truths[0]["states"][-1],
        "goal_origin":"Frozen actual endpoint of this source's prescribed30reference actions; no outcomescreening",
        "old_source_goal_state_difference":float(np.max(np.abs(old_goal-truths[0]["states"][-1]))),
        "raw_actions":actions,"normalized_actions":normalized,"candidates":records,"seconds":time.monotonic()-start,
        "same_state_actions_and_two_physical_replays_exact":True,"future_ground_truth_used_by_model":False}
    path=args.output/f"{row['key']}.pt";torch.save(value,path)
    arrays={key:np.stack([np.asarray(r[key]) for r in records]) for key in ("p3_pooled","predicted_visual_pooled","actual_visual_pooled","native_cost_by_horizon","actual_encoded_native_cost_by_horizon","latent_mse_to_actual","states")}
    arrays.update({key:np.stack([r["metrics"][key] for r in records]) for key in records[0]["metrics"]})
    arrays.update(raw_actions=actions.numpy(),normalized_actions=normalized.numpy(),goal_state=value["goal_state"],
        q_next=np.stack([physical_q(torch.from_numpy(r["states"][5::5])).numpy() for r in records]),
        q_current=np.stack([physical_q(torch.from_numpy(r["states"][:30:5])).numpy() for r in records]),
        contact=np.asarray([[any(c["agent_block_contact"] is True for c in r["contacts"][:5*h]) for h in range(1,7)] for r in records]))
    np.savez_compressed(path.with_suffix(".npz"),**arrays)
    outputs=[{"path":p.name,"sha256":sha256(p),"bytes":p.stat().st_size,"source_id":row["source_id"],"split":row["split"],"key":row["key"]} for p in (path,path.with_suffix(".npz"))]
    write_json(path.with_suffix(".DONE.json"),{"complete":True,"outputs":outputs,"row":row,"seconds":value["seconds"]})
    print(json.dumps({"event":"action_consequence_source_complete","key":row["key"],"seconds":value["seconds"]}),flush=True)
    return outputs

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("inputs","repo","checkpoint","output"):p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--source-ids",type=int,nargs="+",required=True);p.add_argument("--panel",choices=["diverse","original"],default="diverse")
    args=p.parse_args();rows=select_inputs(args)
    if sha256(args.checkpoint)!=CHECKPOINT:raise ValueError("Wrong native frozen checkpoint")
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/"protocol.json",{"rows":rows,"candidate_names":NAMES,"pulse_first5_steps_relative_xy":.1,"no_environment_clipping_claimed":True,
        "input_done_sha256":sha256(args.inputs/"DONE.json"),"script_sha256":sha256(__file__),"horizons":6,"raw_steps":30,
        "primary_error":"Full native forecast visual versus actual future encoded visual","ranking_reference":"Same native goal objective on actual encoded futures, plus physical block/angle goalprogress",
        "painted_target_reward_not_replayed_goal":True,"confirmation_or_held_panel_opened":False,"cem_calls":0})
    from model_loader import load_headless
    cfg=config(args.repo,args.output);torch.set_num_threads(2);torch.manual_seed(90505)
    wm,preprocessor,provenance=load_headless(args.repo,model_name="jepa_wm_pusht",checkpoint_override=args.checkpoint)
    wm.eval().requires_grad_(False);before=parameter_sha(wm);start=time.monotonic();outputs=[]
    for row in rows:
        if time.monotonic()-start>3500:raise RuntimeError("Bounded shard runtime exceeded")
        outputs.extend(capture_source(args,row,wm,preprocessor,cfg))
    if parameter_sha(wm)!=before:raise RuntimeError("Weights changed")
    write_json(args.output/"DONE.json",{"complete":True,"outputs":outputs,"seconds":time.monotonic()-start,"parameter_sha256":before,
        "checkpoint_sha256":CHECKPOINT,"provenance":provenance,"cem_calls":0,"future_truth_supplied_to_model":False})

if __name__=="__main__":main()
