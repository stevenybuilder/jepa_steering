#!/usr/bin/env python3
"""Frozen Reach dose/timing diagnostic: one native CEM and15 raw steps perarm."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from coordinate_calibration_operator import CoordinateCalibration, CoordinateCalibrationHook
from protocol import file_sha256,write_json_atomic

SETTINGS=(('reference_all',1.,False,6101),('quarter_all',.25,False,6102),('quarter_first',.25,True,6103))
ARMS=[('unsteered',0.,False,None)]+[(name+('_sham' if sham else ''),mult,first,seed if sham else None) for name,mult,first,seed in SETTINGS for sham in (False,True)]


def shuffled_signs(seed,device='cpu',dtype=torch.float32):
    """Independent balanced sign orientations; exact coordinatewise norm match."""
    signs=torch.ones(384,dtype=dtype)
    signs[torch.randperm(384,generator=torch.Generator().manual_seed(seed))[:192]]=-1
    return signs.to(device)


class DoseTimingHook(CoordinateCalibrationHook):
    def __init__(self,wm,block,calibration,multiplier,first_only=False,sham_seed=None):
        super().__init__(wm,block,calibration)
        self.multiplier=multiplier;self.first_only=first_only
        self.signs=None if sham_seed is None else shuffled_signs(sham_seed,calibration.values['p3_mean'].device)
        self.step=0;self.records=[];self.record=False
        self.horizons=[dict(candidate_count=0,active_count=0,original_cap_saturated=0,
                            standardized_residual_sum=0.,p3_standardized_rms_sum=0.,
                            original_requested_norm_sum=0.,original_capped_norm_sum=0.,delivered_norm_sum=0.) for _ in range(6)]
    def reset(self,record=False):
        self.step=0;self.records=[];self.record=record
    def forward(self,*args,**kwargs):
        self.pending=None;result=self.original(*args,**kwargs);self.step+=1
        if not 1<=self.step<=6 or self.pending is None:raise RuntimeError('Unreset or invalid native six-step calibration')
        visual=result[0]
        if visual.ndim!=6 or visual.shape[2:]!=(1,16,16,384):raise RuntimeError('Native visual layout changed')
        flat=visual[:,-1].reshape(len(self.pending),256,384);pooled=flat.float().mean(1)
        capped,saturated,residual=self.calibration.delta(self.pending,pooled)
        requested=self.calibration.beta*(residual@self.calibration.values['correction_matrix'].T)
        active=self.multiplier>0 and (not self.first_only or self.step==1)
        delta=capped*(self.multiplier if active else 0.)
        if self.signs is not None:delta=delta*self.signs
        stats=self.horizons[self.step-1];stats['candidate_count']+=len(delta);stats['active_count']+=len(delta) if active else 0
        stats['original_cap_saturated']+=int(saturated.sum())
        stats['standardized_residual_sum']+=float(residual.norm(dim=-1).sum())
        standardized=(self.pending-self.calibration.values['p3_mean'])/self.calibration.values['p3_scale']
        stats['p3_standardized_rms_sum']+=float(standardized.square().mean(-1).sqrt().sum())
        stats['original_requested_norm_sum']+=float(requested.norm(dim=-1).sum())
        stats['original_capped_norm_sum']+=float(capped.norm(dim=-1).sum())
        stats['delivered_norm_sum']+=float(delta.norm(dim=-1).sum())
        if active:
            edited=visual.clone();edited[:,-1]=(flat+delta[:,None].to(flat)).reshape_as(visual[:,-1])
            answer=(edited,result[1],result[2])
        else:answer=result
        if self.record:
            self.records.append({k:v.detach().cpu().clone() for k,v in {'p3_pooled':self.pending,'before_visual_pooled':pooled,
                'after_visual_pooled':answer[0][:,-1].reshape(len(delta),256,384).float().mean(1),'original_requested_delta':requested,
                'original_capped_delta':capped,'delivered_delta':delta,'cap_saturated':saturated,'standardized_residual':residual}.items()})
        return answer
    def summary(self):
        return [dict(imagined_step=i+1,candidate_count=v['candidate_count'],active_count=v['active_count'],
                     original_cap_fraction=v['original_cap_saturated']/max(v['candidate_count'],1),
                     **{k.replace('_sum','_mean'):x/max(v['candidate_count'],1) for k,x in v.items() if k.endswith('_sum')}) for i,v in enumerate(self.horizons)]


def exact(a,b,label):
    if not torch.equal(torch.as_tensor(a).cpu(),torch.as_tensor(b).cpu()):raise RuntimeError('Exact paired guard failed: '+label)


def clone_prediction(pred):
    return {k:pred[k].detach().float().cpu().clone() for k in ('visual','proprio')}


@torch.no_grad()
def encode_actual(wm,frames,proprios):
    from evals.simu_env_planning.planning.utils import make_td
    values=[]
    for image,proprio in zip(frames,proprios):
        td=make_td(torch.as_tensor(image),{'proprio':torch.as_tensor(proprio)})
        value=wm.encode(td.cuda().unsqueeze(0),act=True)['visual']
        values.append(value.reshape(256,384).float().cpu())
    return torch.stack(values)


def latent_errors(prediction,actual):
    predicted=prediction['visual'][1:1+len(actual),0].reshape(len(actual),256,384).cpu()
    return (predicted-actual).square().mean((1,2)).tolist()


@torch.no_grad()
def fixed_reference(args,bank,wm,preprocessor,cfg,calibration,block,output):
    from run_coordinate_steering_pilot import fresh_start,close_env
    path=args.references/f"episode-{bank['episode']:03d}.pt"
    receipt=json.loads(path.with_suffix('.DONE.json').read_text());entry=next(x for x in receipt['outputs'] if x['path']==path.name)
    if not receipt['complete'] or file_sha256(path)!=entry['sha256']:raise RuntimeError('Fixed-reference hash mismatch')
    reference=torch.load(path,map_location='cpu',weights_only=False)
    env,agent,td,start,_=fresh_start(cfg,wm,preprocessor,bank)
    try:
        exact(td['visual'],reference['raw_context_visual'],'fixed-reference initial pixels')
        exact(td['proprio'],reference['raw_context_proprio'],'fixed-reference initial proprio')
        z=wm.encode(td.cuda().unsqueeze(0),act=True);actions=reference['normalized_actions'][:,None].cuda()
        truth=reference['truth'];actual=encode_actual(wm,truth['frames'][4::5],truth['proprios'][4::5])
        results={};baseline=wm.unroll(z.clone(),act_suffix=actions)
        for name,mult,first,seed in ARMS:
            with DoseTimingHook(wm,block,calibration,mult,first,seed) as hook:
                hook.reset(record=True);prediction=wm.unroll(z.clone(),act_suffix=actions)
            if mult==0:
                for k in ('visual','proprio'):exact(prediction[k],baseline[k],'fixedH6 zero identity '+k)
            results[name]={'prediction':clone_prediction(prediction),'per_horizon_latent_mse_to_same_action_actual':latent_errors(prediction,actual),
                           'per_horizon_cap_and_scale':hook.summary(),'activation_records':hook.records}
        path_out=output/'fixed-action-reference.pt'
        torch.save({'complete':True,'source_sha256':entry['sha256'],'actions_normalized':actions.cpu(),'actual_visual':actual,
                    'results':results,'truth_scope':'All6 horizons from saved exactsame30action simulator reference; actualpixels reencoded on currentworker'},path_out)
        return {'path':path_out.name,'sha256':file_sha256(path_out),'errors':{k:v['per_horizon_latent_mse_to_same_action_actual'] for k,v in results.items()}}
    finally:close_env(env)


@torch.no_grad()
def run_arm(cfg,wm,preprocessor,bank,calibration,block,arm,baseline_start,baseline_actions):
    from run_coordinate_steering_pilot import fresh_start,paired_start_checks,close_env
    from scipy.stats import rankdata
    name,mult,first,seed=arm;started=time.monotonic()
    env,agent,td,start,goal_mode=fresh_start(cfg,wm,preprocessor,bank)
    checks=paired_start_checks(start,baseline_start) if baseline_start is not None else {}
    z=wm.encode(td.cuda().unsqueeze(0),act=True)
    goal=np.asarray(env.proprio_env.unwrapped._target_pos).copy();initial_hand=start['proprio'].numpy().reshape(-1,4)[-1,:3]
    hook=DoseTimingHook(wm,block,calibration,mult,first,seed)
    native_unroll=agent.planner.unroll;first_candidates={}
    def tracked(*a,**kw):
        actions=kw.get('act_suffix',a[1] if len(a)>1 else None)
        if actions is None:raise RuntimeError('Native action binding absent')
        capture=not first_candidates and actions.shape[1]==300;hook.reset(record=capture)
        pred=native_unroll(*a,**kw)
        if capture:
            if baseline_actions is not None:exact(actions,baseline_actions,'first300 native sampled actions')
            costs=agent.objective(pred,actions).cpu();by_step=agent.objective(pred,actions,keepdims=True).cpu()
            first_candidates.update(actions=actions.cpu().clone(),costs=costs,ranks=torch.from_numpy(rankdata(costs.numpy())),
                                    costs_by_step=by_step,visual_pooled=pred['visual'][1:].reshape(6,300,256,384).mean(2).cpu(),
                                    proprio=pred['proprio'].cpu(),activation_records=copy.deepcopy(hook.records))
        return pred
    agent.planner.unroll=tracked
    try:
        with hook:
            planned=agent.plan(z.clone(),steps_left=bank['replans'][0]['steps_left_model'])
            full=agent.planner._prev_mean.detach().clone()
            exact(planned,full[:3],'fullH6 selectedprefix')
            planning_statistics=hook.summary();hook.reset(record=True)
            selected=wm.unroll(z.clone(),act_suffix=full[:,None]);selected_records=hook.records
        raw=preprocessor.denormalize_actions(planned.cpu().reshape(15,4))
        observations,rewards,dones,infos=env.step_multiple(raw)
        if len(observations)!=15:raise RuntimeError('Expected exact15 raw physicalsteps')
        states=np.stack([np.asarray(x['state']) for x in infos]);frames=torch.stack(observations).cpu()
        proprios=torch.stack([torch.as_tensor(x['proprio']) for x in infos]).cpu()
        actual=encode_actual(wm,frames[4::5],proprios[4::5])
        distance=float(np.linalg.norm(states[-1,:3]-goal));initial_distance=float(np.linalg.norm(initial_hand-goal))
        return {'complete':True,'episode':bank['episode'],'arm':name,'multiplier':mult,'first_only':first,'sham_seed':seed,
                'initial_snapshot':start,'paired_start_checks':checks,'goal_replay':goal_mode,'physical_goal_xyz':goal,
                'first_candidates':first_candidates,'selected_plan_fullH6':full.cpu(),'selected_prefix_H3':planned.cpu(),
                'selected_prediction':clone_prediction(selected),'selected_activation_records':selected_records,
                'planning_horizon_statistics':planning_statistics,'actual_visual_horizons1to3':actual,
                'selected_per_horizon_latent_mse_to_actual':latent_errors(selected,actual),
                'unobserved_selected_future_horizons':[4,5,6],'states':states,'frames':frames,'proprios':proprios,'raw_actions':raw,
                'raw_steps':15,'initial_goal_distance_m':initial_distance,'final_goal_distance_m':distance,'goal_progress_m':initial_distance-distance,
                'ever_success':any(bool(x['success']) for x in infos),'final_success':bool(infos[-1]['success']),'seconds':time.monotonic()-started}
    finally:
        agent.planner.unroll=native_unroll;close_env(env)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('repo','candidate-dir','references','output-dir'):parser.add_argument('--'+arg,type=Path,required=True)
    parser.add_argument('--banks',nargs='+',type=Path,required=True);parser.add_argument('--time-limit-seconds',type=float,required=True)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False);started=time.monotonic();torch.set_num_threads(2)
    from causal_planner_forks import load_development_bank,setup_cfg
    from model_loader import load_headless_metaworld
    frozen=json.loads((args.candidate_dir/'FROZEN.json').read_text());candidate=args.candidate_dir/'coordinate_candidate.npz'
    if file_sha256(candidate)!=frozen['candidate_sha256'] or file_sha256(Path(__file__).with_name('coordinate_calibration_operator.py'))!=frozen['operator_script_sha256']:
        raise RuntimeError('Frozen original candidate/operator hash changed')
    banks=[load_development_bank(p) for p in args.banks]
    if any(x['episode'] not in (0,1,4,7,10) for x,_ in banks):raise RuntimeError('Only frozen5 developmentstarts')
    protocol={'episodes':[b['episode'] for b,_ in banks],'arms':ARMS,'candidate_sha256':frozen['candidate_sha256'],'original_beta':.5,
              'dose_policy':'multiplier AFTER original beta0.5 radialcap; quarter norm exactly1/4 sameinput original; uncapped coefficient0.125',
              'cap_radius':frozen['cap_radius'],'native_cem':{'H':6,'samples':300,'iterations':15},'raw_steps_per_arm':15,
              'sham':'Independent fixedseed balanced192negative-sign orientations for eachdose/timing setting; det+1, bitexactsameinput raw norm',
              'sham_scope':'No claim cumulative cross-armnorms match after adaptivecandidatepopulations diverge',
              'time_limit_seconds':args.time_limit_seconds,'script_sha256':file_sha256(Path(__file__)),'no_refit':True,'held_sources_opened':False}
    write_json_atomic(args.output_dir/'protocol.json',protocol);outputs=[]
    try:
        wm,preprocessor,provenance=load_headless_metaworld(args.repo);wm.eval().requires_grad_(False)
        cfg=setup_cfg(Path(provenance['config']),args.output_dir,wm);block=wm.model.predictor.predictor_blocks[3]
        calibration=CoordinateCalibration(dict(np.load(candidate)),device='cuda:0')
        for bank,sha in banks:
            output=args.output_dir/f"episode-{bank['episode']:03d}";output.mkdir()
            try:
                fixed=fixed_reference(args,bank,wm,preprocessor,cfg,calibration,block,output)
                print(json.dumps({'event':'fixed_reference_complete','episode':bank['episode'],'errors':fixed['errors']}),flush=True)
            except Exception as exc:
                if 'identity' in str(exc).lower() or 'guard' in str(exc).lower():raise
                write_json_atomic(output/'fixed-reference-FAILED.json',{'error':repr(exc),'native_CEM_proceeds':True})
                print(json.dumps({'event':'fixed_reference_unavailable','episode':bank['episode'],'error':repr(exc)}),flush=True)
            baseline_start=baseline_actions=None
            for arm in ARMS:
                if time.monotonic()-started>args.time_limit_seconds:raise RuntimeError('Frozenworker budget elapsed')
                print(json.dumps({'event':'short_native_cem_started','episode':bank['episode'],'arm':arm[0]}),flush=True)
                result=run_arm(cfg,wm,preprocessor,bank,calibration,block,arm,baseline_start,baseline_actions)
                if arm[0]=='unsteered':baseline_start=result['initial_snapshot'];baseline_actions=result['first_candidates']['actions']
                result['bank_sha256']=sha;path=output/(arm[0]+'.pt');torch.save(result,path)
                row={'path':str(path.relative_to(args.output_dir)),'sha256':file_sha256(path),'episode':bank['episode'],'arm':arm[0],
                     **{k:result[k] for k in ('goal_progress_m','final_goal_distance_m','ever_success','final_success','seconds','planning_horizon_statistics','selected_per_horizon_latent_mse_to_actual')}}
                outputs.append(row);write_json_atomic(args.output_dir/'progress.json',{'complete':False,'outputs':outputs})
                print(json.dumps({'event':'short_arm_complete',**row}),flush=True)
                del result
        write_json_atomic(args.output_dir/'DONE.json',protocol|{'complete':True,'outputs':outputs,'seconds':time.monotonic()-started,'model':provenance})
    except Exception as exc:
        write_json_atomic(args.output_dir/'FAILED.json',{'error':repr(exc),'outputs':outputs});raise


if __name__=='__main__':main()
