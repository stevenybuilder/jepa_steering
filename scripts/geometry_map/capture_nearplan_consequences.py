#!/usr/bin/env python3
"""Eight frozen antithetic local action perturbations around native selected H6 plans."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from action_consequence_diagnostics import physical_metrics,actual_cost_state,CHECKPOINT
from capture_pusht_horizon import physical_fork,compare_physics,encoder_inputs
from capture_horizon_coordinates import CaptureP3Horizons,exact,pool_visual
from collect_pusht_bank import config
from capture_pusht_calibration import parameter_sha,physical_q
from complete_cached_geometry import sha256,write_json

def local_actions(center,episode):
    if center.shape!=(30,2) or episode not in range(4):raise ValueError("Fixed fourdevelopment H6 plans only")
    rng=np.random.default_rng(2026090800+episode);result=[center.clone().float()]
    for _ in range(4):
        v=rng.normal(size=(30,2));v*=.01*np.sqrt(60)/np.linalg.norm(v)
        delta=torch.from_numpy(v).float();result.extend([center.float()+delta,center.float()-delta])
    return torch.stack(result)

@torch.no_grad()
def source(args,episode,wm,prep,cfg):
    from evals.simu_env_planning.envs.init import make_env
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    from evals.simu_env_planning.planning.utils import make_td
    path=args.inputs/f'episode-{episode:03d}.pt';done=json.loads(path.with_suffix('.DONE.json').read_text())
    entry=next(r for r in done['outputs'] if r['path']==path.name)
    if not done['complete'] or entry['episode']!=episode or sha256(path)!=entry['sha256']:raise ValueError("H6 source receipt/hash mismatch")
    original=torch.load(path,map_location='cpu',weights_only=False)
    if original['split']!='development' or original['episode']!=episode:raise ValueError("Held source forbidden")
    prepared=json.loads((args.prepared_inputs/'DONE.json').read_text())
    reset_row=next(r for r in prepared['episodes'] if r['episode']==episode)
    if not prepared['complete'] or reset_row['split']!='development':raise ValueError("Forbidden reset source before load")
    reset_path=args.prepared_inputs/reset_row['path']
    if sha256(reset_path)!=reset_row['sha256']:raise ValueError("Reset-input hash mismatch")
    stimulus=torch.load(reset_path,map_location='cpu',weights_only=False)
    if stimulus['split']!='development' or stimulus['episode']!=episode:raise ValueError("Reset input split mismatch")
    # Native _set_state advances the simulator by0.01s. Reusing observed state0
    # as reset input would incorrectly advance nonzero agent velocity twice.
    raw=local_actions(original['raw_actions'],episode);seed=stimulus['environment_seed'];initial=stimulus['initial_state']
    agent=GC_Agent(cfg,wm,dset=None,preprocessor=prep)
    raw_goal=make_td(original['goal']['raw_visual'],{'proprio':original['goal']['raw_proprio']});agent.set_goal(raw_goal)
    env=make_env(cfg)
    def initialize():
        env.update_env({'shape':'T'});obs,info=env.prepare(seed,initial,{'shape':'T'})
        exact(obs,original['truth']['frames'][0],"originalnative start pixels")
        exact(np.asarray(info['state']),original['truth']['states'][0],"originalnative observed start state")
        return obs,info
    obs,info=initialize();z=wm.encode(make_td(obs,info).to(agent.device).unsqueeze(0),act=True)
    normalized=prep.normalize_actions(raw.reshape(9,6,5,2)).reshape(9,6,10);records=[];start=time.monotonic()
    try:
        for i in range(9):
            obs,info=initialize();truth=physical_fork(env,obs,info,raw[i])
            obs,info=initialize();repeat=physical_fork(env,obs,info,raw[i])
            for key in ('frames','states','proprios','native_applied_targets_xy'):exact(truth[key],repeat[key],"nearplan fresh paired "+key)
            compare_physics(truth['physics'],repeat['physics'])
            plan=normalized[i,:,None].to(agent.device)
            native=wm.unroll(z.clone(),act_suffix=plan)
            with CaptureP3Horizons(wm.model.predictor.predictor_blocks[3]) as captured:forecast=wm.unroll(z.clone(),act_suffix=plan)
            for key in ('visual','proprio'):exact(native[key],forecast[key],"native/observed nearplan forecast")
            actual=wm.encode(encoder_inputs(truth['frames'][::5],truth['proprios'][::5]));actual_cost=actual_cost_state(actual,forecast)
            pred=forecast['visual'][1:,0].float().cpu();av=actual['visual'][0,1:].float().cpu()
            metrics=physical_metrics(truth['states'][5::5],original['goal']['state'])
            records.append({'name':'native_selected' if i==0 else f'antithetic_{i}',
                'p3_pooled':torch.stack(captured.values)[:,0].float().mean(1).cpu(),
                'predicted_visual':pred,'actual_visual':av,'predicted_visual_pooled':pool_visual(pred),'actual_visual_pooled':pool_visual(av),
                'predicted_proprio':forecast['proprio'][1:,0].float().cpu(),'native_cost_by_horizon':agent.objective(forecast,plan,keepdims=True).cpu(),
                'actual_encoded_native_cost_by_horizon':agent.objective(actual_cost,plan,keepdims=True).cpu(),
                'latent_mse_to_actual':(pred-av).square().reshape(6,-1).mean(1),'metrics':metrics,'states':truth['states'],
                'contacts':truth['contacts'],'contact_api_available':truth['contact_api_available'],'physics':truth['physics'],
                'native_applied_targets_xy':truth['native_applied_targets_xy'],'frames_modelsteps':truth['frames'][::5]})
    finally:env.close()
    key=f'near-dev-{episode:03d}';row={'key':key,'source_id':episode,'split':'development_external','center_origin':'immutable original nativeCEM selectedH6 plan'}
    value={'complete':True,'row':row,'context':{k:z[k].cpu() for k in z.keys()},'raw_goal':{k:raw_goal[k].cpu() for k in raw_goal.keys()},
        'goal_encoded':{k:agent.goal_state_enc[k].cpu() for k in ('visual','proprio')},'goal_state':original['goal']['state'],
        'raw_actions':raw,'normalized_actions':normalized,'candidates':records,'source_horizon_sha256':entry['sha256'],
        'same_state_actions_and_two_physical_replays_exact':True,'future_ground_truth_used_by_model':False,'seconds':time.monotonic()-start,
        'reset_input_sha256':reset_row['sha256'],'reset_contract':'Original raw prepared initialstate; native_set_state advances0.01s before observation'}
    pt=args.output/(key+'.pt');torch.save(value,pt)
    arrays={k:np.stack([np.asarray(r[k]) for r in records]) for k in ('p3_pooled','predicted_visual_pooled','actual_visual_pooled','native_cost_by_horizon','actual_encoded_native_cost_by_horizon','latent_mse_to_actual','states')}
    arrays.update({k:np.stack([r['metrics'][k] for r in records]) for k in records[0]['metrics']})
    arrays.update(raw_actions=raw.numpy(),normalized_actions=normalized.numpy(),goal_state=original['goal']['state'],
        q_next=np.stack([physical_q(torch.from_numpy(r['states'][5::5])).numpy() for r in records]),
        contact=np.asarray([[any(c['agent_block_contact'] is True for c in r['contacts'][:h*5]) for h in range(1,7)] for r in records]))
    np.savez_compressed(pt.with_suffix('.npz'),**arrays)
    outputs=[{**row,'path':p.name,'sha256':sha256(p),'bytes':p.stat().st_size} for p in (pt,pt.with_suffix('.npz'))]
    write_json(pt.with_suffix('.DONE.json'),{'complete':True,'outputs':outputs,'seconds':value['seconds']})
    print(json.dumps({'event':'nearplan_consequences_complete','episode':episode,'seconds':value['seconds']}),flush=True)
    return outputs

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('inputs','prepared-inputs','repo','checkpoint','output'):p.add_argument('--'+k,type=Path,required=True)
    args=p.parse_args()
    if sha256(args.checkpoint)!=CHECKPOINT:raise ValueError("Wrong pinned nativecheckpoint")
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'protocol.json',{'episodes':[0,1,2,3],'candidates':9,'newperturbations':8,'raw_command_rms_epsilon':.01,
        'perturbation_seed_rule':'2026090800+episode','antithetic_pairs':4,'action_domain':'native relativeXY commands×100, no newclip',
        'center':'cached nativeH6 CEM selected plan','comparison':'coarse7 vs local9 per sameinitialstart, N=4starts not36candidates',
        'goal':'original fixed expert-replay goal, not perturbed endpoint','futuretruthonlyoffline':True,'cem_calls':0,'script_sha256':sha256(__file__)})
    from model_loader import load_headless
    torch.set_num_threads(2);torch.manual_seed(90505);cfg=config(args.repo,args.output)
    wm,prep,provenance=load_headless(args.repo,model_name='jepa_wm_pusht',checkpoint_override=args.checkpoint)
    wm.eval().requires_grad_(False);before=parameter_sha(wm);start=time.monotonic();outputs=[]
    for episode in range(4):outputs.extend(source(args,episode,wm,prep,cfg))
    if parameter_sha(wm)!=before:raise ValueError("Native weights changed")
    write_json(args.output/'DONE.json',{'complete':True,'outputs':outputs,'seconds':time.monotonic()-start,'parameter_sha256':before,'provenance':provenance,'cem_calls':0})

if __name__=='__main__':main()
