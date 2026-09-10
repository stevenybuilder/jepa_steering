#!/usr/bin/env python3
"""Project native model action inputs onto confirmed simulator XYZ commands.

This changes neither semantic activations nor CEM proposals/executed commands.
Three development starts only; projected15 then fresh paired full99 episodes.
"""
from __future__ import annotations
import argparse
import inspect
import json
from pathlib import Path
import time
import numpy as np
import torch
from protocol import file_sha256,write_json_atomic
from run_coordinate_dose_timing import exact,clone_prediction,encode_actual,latent_errors
from run_planner_compensation_v1 import restore_fixed_encoded_goal,compare_replay,CHECKPOINT_SHA
from run_native_restart_control_v1 import checked_baseline


def project_actions(actions,preprocessor):
    """Official CPU affine coordinates; only raw XYZ clipped, gripper unchanged."""
    if actions.ndim!=3 or actions.shape[-1]!=20 or not torch.isfinite(actions).all():
        raise ValueError('Finite native [horizon,batch,5*4] actions required')
    cpu=actions.detach().cpu()
    raw=preprocessor.denormalize_actions(cpu.reshape(-1,4))
    effective=raw.clone();effective[:,:3]=effective[:,:3].clamp(-1,1)
    exact(raw[:,3],effective[:,3],'raw gripper never clipped')
    projected=preprocessor.normalize_actions(effective).reshape_as(cpu)
    return projected.to(actions),{'xyz_clipped':int((raw[:,:3].abs()>1).sum()),'xyz_total':raw[:,:3].numel(),
                                  'raw_projection_l2':float((raw-effective).norm())}


def unroll_with_projection(native,preprocessor,enabled,*args,**kwargs):
    """No-op branch is exact and bypasses every affine operation."""
    if not enabled:return native(*args,**kwargs)
    args=list(args);kwargs=dict(kwargs)
    if 'act_suffix' in kwargs:
        kwargs['act_suffix'],_=project_actions(kwargs['act_suffix'],preprocessor)
    elif len(args)>1:
        args[1],_=project_actions(args[1],preprocessor)
    else:raise ValueError('Native action suffix binding absent')
    return native(*args,**kwargs)


@torch.no_grad()
def run_episode(cfg,wm,preprocessor,bank,baseline,projected,raw_limit,output,deadline,short=None):
    from run_coordinate_steering_pilot import fresh_start,paired_start_checks,close_env
    from evals.simu_env_planning.planning.utils import make_td
    from scipy.stats import rankdata
    env,agent,td,snapshot,mode=fresh_start(cfg,wm,preprocessor,bank)
    mode=mode|restore_fixed_encoded_goal(agent,snapshot,baseline['initial_snapshot'])
    checks=paired_start_checks(snapshot,baseline['initial_snapshot'])
    method=env.proprio_env.unwrapped.set_xyz_action
    if 'np.clip(action, -1, 1)' not in inspect.getsource(method):raise RuntimeError('Actual XYZ clip rule changed')
    if agent.objective.alpha!=.1 or agent.objective.sum_all_diffs:raise RuntimeError('Native terminal goal objective changed')
    initial=snapshot['proprio'].numpy().reshape(-1,4)[-1,:3];goal=np.asarray(env.proprio_env.unwrapped._target_pos).copy()
    initial_distance=float(np.linalg.norm(initial-goal));native=agent.planner.unroll
    first={};plans=[];states=[];frames=[];proprios=[];actions=[];rewards=[];successes=[];chunks=[]
    stats={'xyz_clipped':0,'xyz_total':0};began=time.monotonic()
    def tracked(*a,**kw):
        proposed=kw.get('act_suffix',a[1] if len(a)>1 else None)
        if proposed is None:raise RuntimeError('Native proposed actions missing')
        capture=not first and proposed.shape[1]==300
        if capture:exact(proposed,baseline['first_candidates']['actions'],'initial300 proposals unchanged')
        if projected:
            _,s=project_actions(proposed,preprocessor)
            for k in stats:stats[k]+=s[k]
        result=unroll_with_projection(native,preprocessor,projected,*a,**kw)
        if capture:
            costs=agent.objective(result,proposed).cpu()
            if not torch.equal(costs,agent.objective(result,-proposed).cpu()):raise RuntimeError('Objective unexpectedly depends on action argument')
            unedited=native(*a,**kw) if projected else result
            native_costs=agent.objective(unedited,proposed).cpu()
            replay=compare_replay(native_costs,baseline['first_candidates']['costs'],'same300 native costs')
            exact(torch.as_tensor(rankdata(native_costs.numpy())),baseline['first_candidates']['ranks'],'same300 native rank identity')
            first.update(actions=proposed.cpu().clone(),costs=costs,native_costs=native_costs,
                         ranks=torch.as_tensor(rankdata(costs.numpy())),native_ranks=torch.as_tensor(rankdata(native_costs.numpy())),
                         replay=replay,objective_actions_unused_exact=True,
                         predicted_visual_pooled=result['visual'][1:].reshape(len(result['visual'])-1,300,256,384).mean(2).cpu(),
                         predicted_proprio=result['proprio'].cpu())
        return result
    agent.planner.unroll=tracked
    try:
        while len(states)<raw_limit:
            if time.monotonic()>deadline:raise RuntimeError('Frozen projected-planner budget elapsed')
            z=wm.encode(td.cuda().unsqueeze(0),act=True)
            steps_left=max((env.steps_left()+1)*wm.action_skip//cfg.frameskip,1)
            planned=agent.plan(z.clone(),steps_left=steps_left)
            full=agent.planner._prev_mean.detach().clone()
            raw=preprocessor.denormalize_actions(planned.cpu().reshape(-1,4))
            f0=wm.unroll(z.clone(),act_suffix=full[:,None])
            fe=unroll_with_projection(wm.unroll,preprocessor,True,z.clone(),act_suffix=full[:,None])
            observations,rs,ds,infos=env.step_multiple(raw)
            count=len(observations)
            if not count or len(states)+count>raw_limit:raise RuntimeError('Native execution length differs from frozen limit')
            cf=torch.stack(observations).cpu();cp=torch.stack([torch.as_tensor(i['proprio']) for i in infos]).cpu()
            cs=np.stack([np.asarray(i['state']) for i in infos]);actual=encode_actual(wm,cf[4::5],cp[4::5])
            selected=fe if projected else f0
            row={'replan':len(plans),'raw_steps':len(states)+count,'goal_distance_m':float(np.linalg.norm(cs[-1,:3]-goal)),
                 'native_selected_cost':float(agent.objective(f0,full[:,None])[-1]),
                 'projected_selected_cost':float(agent.objective(fe,full[:,None])[-1]),
                 'unedited_forecast_mse_to_actual':latent_errors(f0,actual),
                 'projected_forecast_mse_to_actual':latent_errors(fe,actual),
                 'ever_success_so_far':any(successes) or any(bool(i['success']) for i in infos),
                 'seconds':time.monotonic()-began}
            if not plans:
                if not projected:
                    exact(full,baseline['selected_plan_fullH6'],'fresh native first H6 plan')
                    exact(cf,baseline['frames'],'fresh native original15 pixels');exact(cs,baseline['states'],'fresh native original15 states')
                if short is not None:
                    exact(full,short['plans'][0],'projected short/full same first H6 plan')
                    exact(cf,short['frames'],'projected short/full same15 pixels');exact(cs,short['states'],'projected short/full same15 states')
            frames.extend(cf);proprios.extend(cp);states.extend(cs);actions.append(raw[:count].cpu());rewards.extend(float(r) for r in rs);successes.extend(bool(i['success']) for i in infos)
            plans.append(full.cpu());chunks.append({'row':row,'prediction_native':clone_prediction(f0),'prediction_projected':clone_prediction(fe),'actual_visual':actual})
            write_json_atomic(output/'live-progress.json',{'projected':projected,'raw_limit':raw_limit,'episode':bank['episode'],'rows':[c['row'] for c in chunks]})
            print(json.dumps({'event':'projected_planner_replan_complete','episode':bank['episode'],'projected':projected,'raw_limit':raw_limit,**row}),flush=True)
            td=make_td(observations[-1],infos[-1])
            if ds[-1] and len(states)!=raw_limit:raise RuntimeError('Unexpected early native termination')
        return {'complete':True,'episode':bank['episode'],'projected':projected,'raw_steps':raw_limit,
                'initial_snapshot':snapshot,'paired_start_checks':checks,'goal_mode':mode,
                'plans':plans,'states':np.stack(states),'frames':torch.stack(frames),'proprios':torch.stack(proprios),
                'raw_actions':torch.cat(actions),'rewards':rewards,'successes':successes,'chunks':chunks,'first_candidates':first,
                'final_goal_distance_m':chunks[-1]['row']['goal_distance_m'],'goal_progress_m':initial_distance-chunks[-1]['row']['goal_distance_m'],
                'initial_goal_distance_m':initial_distance,'final_success':successes[-1],'ever_success':any(successes),
                'total_reward':sum(rewards),'projection_stats':stats,'seconds':time.monotonic()-began}
    finally:agent.planner.unroll=native;close_env(env)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('repo','bank','baseline-dir','output-dir'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);began=time.monotonic();outputs=[]
    from causal_planner_forks import load_development_bank,setup_cfg
    from model_loader import load_headless_metaworld
    bank,banksha=load_development_bank(a.bank)
    if bank['episode'] not in (0,4,7):raise RuntimeError('Only fixed starts0/4/7')
    baseline,source=checked_baseline(a.baseline_dir,bank['episode'])
    protocol={'episode':bank['episode'],'bank_sha256':banksha,'source_baseline':source,
              'intervention':'Model action suffix only: official CPU denorm -> XYZ clip[-1,1], raw gripper unchanged -> official renorm',
              'native_cem':{'H':6,'samples':300,'iterations':15},'unchanged':'CEM proposals, executed original raw commands, model weights, objective and raw/encoded goal',
              'stages':['projected15','fresh_native99','fresh_projected99'],'first300_proposals_exact':True,'held_sources_opened':False,
              'no_semantic_intervention':True,'budget_seconds':2100,'script_sha256':file_sha256(Path(__file__)),'checkpoint_sha256':CHECKPOINT_SHA}
    write_json_atomic(a.output_dir/'protocol.json',protocol)
    try:
        wm,preprocessor,provenance=load_headless_metaworld(a.repo);wm.eval().requires_grad_(False)
        if file_sha256(Path(provenance['checkpoint']))!=CHECKPOINT_SHA:raise RuntimeError('Frozen model checkpoint changed')
        cfg=setup_cfg(Path(provenance['config']),a.output_dir,wm)
        # Exact disabled wrapper and direct projected-action equivalence on native H6.
        from run_coordinate_steering_pilot import fresh_start,close_env
        env,agent,td,snapshot,_=fresh_start(cfg,wm,preprocessor,bank)
        try:
            restore_fixed_encoded_goal(agent,snapshot,baseline['initial_snapshot'])
            with torch.no_grad():
                z=wm.encode(td.cuda().unsqueeze(0),act=True);actions=baseline['selected_plan_fullH6'][:,None].cuda()
                f0=wm.unroll(z.clone(),act_suffix=actions);identity=unroll_with_projection(wm.unroll,preprocessor,False,z.clone(),act_suffix=actions)
                projected,_=project_actions(actions,preprocessor)
                direct=wm.unroll(z.clone(),act_suffix=projected);wrapped=unroll_with_projection(wm.unroll,preprocessor,True,z.clone(),act_suffix=actions)
                for key in ('visual','proprio'):
                    exact(f0[key],identity[key],'disabled projection exact native '+key)
                    exact(direct[key],wrapped[key],'direct/wrapped projection exact '+key)
        finally:close_env(env)
        print(json.dumps({'event':'projection_native_identity_passed','episode':bank['episode']}),flush=True)
        short=None
        for name,projected,limit in (('projected15',True,15),('native99',False,99),('projected99',True,99)):
            print(json.dumps({'event':'projected_planner_episode_started','stage':name,'episode':bank['episode']}),flush=True)
            value=run_episode(cfg,wm,preprocessor,bank,baseline,projected,limit,a.output_dir,began+2100,short if name=='projected99' else None)
            if name=='projected15':short=value
            path=a.output_dir/(name+'.pt');torch.save(value,path)
            row={'path':path.name,'sha256':file_sha256(path),'bytes':path.stat().st_size,'stage':name,'episode':bank['episode'],
                 **{k:value[k] for k in ('raw_steps','goal_progress_m','final_goal_distance_m','ever_success','final_success','total_reward','seconds')},
                 'first300_rank_change_fraction':float((value['first_candidates']['ranks']!=value['first_candidates']['native_ranks']).float().mean()),
                 'same300_native_replay':value['first_candidates']['replay'],'chunks':[c['row'] for c in value['chunks']]}
            outputs.append(row);write_json_atomic(a.output_dir/'progress.json',{'outputs':outputs})
            print(json.dumps({'event':'projected_planner_episode_complete',**row}),flush=True)
        write_json_atomic(a.output_dir/'DONE.json',protocol|{'complete':True,'outputs':outputs,'seconds':time.monotonic()-began,'model':provenance})
    except Exception as exc:
        write_json_atomic(a.output_dir/'FAILED.json',{'error':repr(exc),'outputs':outputs});raise


if __name__=='__main__':main()
