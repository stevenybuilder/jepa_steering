#!/usr/bin/env python3
"""Cross fixed plans and signed forecast edits; no held data or refitting.

Negative calibration reverses the ORIGINAL capped update, not its dose/cap.
Fixed-command simulator controls block action selection but cannot establish
intentional compensation. Local Jacobians and crossed forecasts are descriptive.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from coordinate_calibration_operator import CoordinateCalibration
from run_coordinate_dose_timing import DoseTimingHook, exact, clone_prediction, run_arm, ARMS
from protocol import file_sha256, write_json_atomic

EPISODES=(0,1,4,7,10)
CONDITIONS=(('unsteered',0,None),('positive',1,None),('negative',-1,None),
            ('positive_sham',1,6101),('negative_sham',-1,6101))
NEW_ARMS=(('negative_all',1.,False,None),('negative_all_sham',1.,False,6101))
EPSILONS=(.01,.005)
CHECKPOINT_SHA='c3297772c7af4e84c28bd0c0937e22398a888848305014bb3ab3f1508a310bd8'


class SignedCalibration:
    """Exact sign reversal after the unchanged beta0.5/radial cap operation."""
    def __init__(self, original, sign):
        if sign not in (-1,1): raise ValueError('Only frozen signed directions')
        self.original,self.sign=original,sign
        self.values=dict(original.values)
        self.values['correction_matrix']=sign*original.values['correction_matrix']
        self.beta,self.radius=original.beta,original.radius
    def delta(self,*args,**kwargs):
        delta,saturated,residual=self.original.delta(*args,**kwargs)
        return self.sign*delta,saturated,residual


def perturbation_batch(actions, epsilon):
    """All120 native normalized coordinates; no clipping or gripper exception."""
    if actions.shape!=(6,20) or epsilon<=0 or not torch.isfinite(actions).all():
        raise ValueError('Finite normalized nativeH6 plan and positive epsilon required')
    eye=torch.eye(120,device=actions.device,dtype=actions.dtype).reshape(120,6,20)
    return torch.cat((actions[None]+epsilon*eye,actions[None]-epsilon*eye),0).permute(1,0,2).contiguous()


def central_jacobian(values,epsilon):
    if values.shape[1]!=240 or epsilon<=0: raise ValueError('Expected plus120 then minus120 candidates')
    return (values[:,:120].double()-values[:,120:].double())/(2*epsilon)


def compare_replay(actual,expected,label,bound=1e-6):
    """Previously approved float32 cross-process policy, never identity policy."""
    if actual.shape!=expected.shape: raise RuntimeError('Replay shape mismatch: '+label)
    error=float((actual.double().cpu()-expected.double().cpu()).abs().max())
    if not np.isfinite(error) or error>bound: raise RuntimeError(f'{label}: {error} > {bound}')
    return {'max_absolute_error':error,'bound':bound,'actual_dtype':str(actual.dtype),'cached_dtype':str(expected.dtype)}


def restore_fixed_encoded_goal(agent,snapshot,reference):
    """Restore the immutable goal stimulus, not a tolerance on paired identity."""
    for key in ('goal_visual','goal_proprio'):exact(snapshot[key],reference[key],'fixed raw goal '+key)
    differences={}
    for key in ('visual','proprio'):
        name='goal_encoded_'+key
        differences[key]=float((snapshot[name].double()-reference[name].double()).abs().max())
        # Native target may be an expanded zero-stride view; replace, never write into its aliases.
        agent.goal_state_enc[key]=reference[name].to(agent.goal_state_enc[key]).clone()
        snapshot[name]=agent.goal_state_enc[key].detach().cpu().clone()
        exact(snapshot[name],reference[name],'restored encoded goal '+key)
    agent.objective.target_enc=agent.goal_state_enc
    agent.planner.set_objective(agent.objective)
    for key in ('visual','proprio'):exact(agent.objective.target_enc[key],reference['goal_encoded_'+key],'effective objective goal '+key)
    return {'mode':'immutable baseline full-precision encoded goal restored for every arm',
            'fresh_encoding_difference':differences,'fresh_encoding_identity_claimed':False,'effective_objective_target_exact':True}


def load_previous(root,episode):
    done=json.loads((root/'DONE.json').read_text())
    if not done['complete'] or done['held_sources_opened'] or done['candidate_sha256']!='b80aebccc641da5913b6e1b0fbb315323e4745876547fec1eb8f9e88f8471e4c':
        raise RuntimeError('Expected frozen completed development dose/timing source')
    values={};sources=[]
    for entry in done['outputs']:
        if entry['episode']!=episode: continue
        path=root/entry['path']
        if not path.resolve().is_relative_to(root.resolve()) or file_sha256(path)!=entry['sha256']:
            raise RuntimeError('Previous arm SHA/path mismatch')
        values[entry['arm']]=torch.load(path,map_location='cpu',weights_only=False)
        sources.append({'path':str(path),'sha256':entry['sha256']})
    if set(values)!={x[0] for x in ARMS}: raise RuntimeError('Missing frozen seven plans')
    return values,sources


@torch.no_grad()
def fixed_command_controls(cfg,wm,preprocessor,bank,calibration,block,baseline):
    from run_coordinate_steering_pilot import fresh_start,paired_start_checks,close_env
    from collect_on_policy_bank import physics_snapshot
    rows={}
    for label,sign,seed in CONDITIONS:
        env,agent,td,start,goal_mode=fresh_start(cfg,wm,preprocessor,bank)
        checks=paired_start_checks(start,baseline['initial_snapshot'])
        signed=SignedCalibration(calibration,sign or 1)
        try:
            if label=='unsteered':
                z=wm.encode(td.cuda().unsqueeze(0),act=True)
                actions=baseline['selected_plan_fullH6'][:,None].cuda()
                native=wm.unroll(z.clone(),act_suffix=actions)
                with DoseTimingHook(wm,block,calibration,0.) as identity:
                    identity.reset();observed=wm.unroll(z.clone(),act_suffix=actions)
                for key in ('visual','proprio'):exact(native[key],observed[key],'pre-CEM native identity '+key)
                print(json.dumps({'event':'native_identity_passed','episode':bank['episode']}),flush=True)
            with DoseTimingHook(wm,block,signed,float(sign!=0),False,seed) as hook:
                observations,rewards,dones,infos=env.step_multiple(baseline['raw_actions'].clone())
            if hook.step!=0: raise RuntimeError('Fixed-command control unexpectedly called predictor')
            frames=torch.stack(observations).cpu();states=np.stack([np.asarray(i['state']) for i in infos])
            exact(frames,baseline['frames'],'fixed-command rawpixels '+label)
            exact(states,baseline['states'],'fixed-command observedstates '+label)
            rows[label]={'states':states,'frames':frames,'final_physics':physics_snapshot(env),
                         'paired_start_checks':checks,'model_prediction_calls':hook.step,
                         'raw_actions':baseline['raw_actions'].clone(),'source_states_pixels_exact':True}
            if label!='unsteered':
                for k,v in rows['unsteered']['final_physics']['physics'].items():
                    exact(rows[label]['final_physics']['physics'][k],v,'blocked finalphysics '+k)
        finally: close_env(env)
    return rows


@torch.no_grad()
def native_cross_evaluate(cfg,wm,preprocessor,bank,calibration,block,plans,baseline):
    from run_coordinate_steering_pilot import fresh_start,paired_start_checks,close_env
    from scipy.stats import rankdata
    env,agent,td,start,goal_mode=fresh_start(cfg,wm,preprocessor,bank)
    paired_start_checks(start,baseline['initial_snapshot'])
    z=wm.encode(td.cuda().unsqueeze(0),act=True);original_z={k:z[k].clone() for k in z.keys()}
    names=list(plans);actions=torch.stack([plans[n]['selected_plan_fullH6'] for n in names],dim=1).cuda()
    first=baseline['first_candidates']['actions'].cuda();selected={};candidate_rows={}
    try:
        native=wm.unroll(z.clone(),act_suffix=actions)
        for label,sign,seed in CONDITIONS:
            signed=SignedCalibration(calibration,sign or 1)
            with DoseTimingHook(wm,block,signed,float(sign!=0),False,seed) as hook:
                hook.reset(record=True);pred=wm.unroll(z.clone(),act_suffix=actions)
                records=copy.deepcopy(hook.records)
                if sign==0:
                    for k in ('visual','proprio'):exact(pred[k],native[k],'new native fulltrace identity '+k)
                selected[label]={'prediction':clone_prediction(pred),'costs':agent.objective(pred,actions).cpu(),
                                 'costs_by_step':agent.objective(pred,actions,keepdims=True).cpu(),
                                 'activation_records':records}
                hook.reset(record=False);p=wm.unroll(z.clone(),act_suffix=first)
                costs=agent.objective(p,first).cpu()
                candidate_rows[label]={'costs':costs,'ranks':torch.from_numpy(rankdata(costs.numpy())),
                                      'costs_by_step':agent.objective(p,first,keepdims=True).cpu(),
                                      'visual_pooled':p['visual'][1:].reshape(6,300,256,384).mean(2).cpu(),
                                      'proprio':p['proprio'].cpu()}
        replay={}
        for label,source in (('unsteered','unsteered'),('positive','reference_all'),('positive_sham','reference_all_sham'),
                             ('negative','negative_all'),('negative_sham','negative_all_sham')):
            actual=candidate_rows[label];old=plans[source]['first_candidates']
            replay[label]={key:compare_replay(actual[key],old[key],label+' '+key) for key in ('costs','visual_pooled','proprio')}
            exact(actual['ranks'],old['ranks'],'same300 rank replay '+label)
        jacobians={}
        for epsilon in EPSILONS:
            perturbed=perturbation_batch(baseline['selected_plan_fullH6'].cuda(),epsilon)
            p=wm.unroll(z.clone(),act_suffix=perturbed)
            pooled=p['visual'][1:].reshape(6,240,256,384).mean(2)
            q=calibration.observer_physical(pooled)
            jacobians[str(epsilon)]={'visual_pooled':central_jacobian(pooled,epsilon).cpu(),
                                    'proprio':central_jacobian(p['proprio'][1:],epsilon).cpu(),
                                    'observer_xyz':central_jacobian(q,epsilon).cpu(),
                                    'native_cost_by_step':central_jacobian(agent.objective(p,perturbed,keepdims=True),epsilon).cpu()}
        for k,v in original_z.items():exact(z[k],v,'encoded observation unchanged '+k)
        return {'plan_names':names,'normalized_actions':actions.cpu(),'first300_actions':first.cpu(),
                'selected_crossed':selected,'first300_crossed':candidate_rows,'jacobians':jacobians,
                'jacobian_layout':'[future6, normalized_action_coordinate120, output...]; cost includes context+6',
                'jacobian_coordinate_order':'flatten [H6,20]; within20 is5rawsteps*4actionchannels; no clipping',
                'goal_visual':start['goal_visual'],'goal_proprio':start['goal_proprio'],
                'goal_encoded_visual':start['goal_encoded_visual'],'goal_encoded_proprio':start['goal_encoded_proprio'],
                'physical_goal_xyz':baseline['physical_goal_xyz'],'initial_hand_xyz':start['proprio'].reshape(-1,4)[-1,:3],
                'cross_process_replay_checks':replay,'within_run_native_identity_exact':True,'goal_mode':goal_mode}
    finally: close_env(env)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('repo','candidate-dir','previous-dir','output-dir'):parser.add_argument('--'+arg,type=Path,required=True)
    parser.add_argument('--banks',nargs='+',type=Path,required=True)
    parser.add_argument('--time-limit-seconds',type=float,default=2400)
    parser.add_argument('--restore-baseline-encoded-goal',action='store_true')
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();torch.set_num_threads(2);outputs=[]
    from causal_planner_forks import load_development_bank,setup_cfg
    from model_loader import load_headless_metaworld
    candidate=args.candidate_dir/'coordinate_candidate.npz';frozen=json.loads((args.candidate_dir/'FROZEN.json').read_text())
    if file_sha256(candidate)!=frozen['candidate_sha256']:raise RuntimeError('Frozen candidate SHA mismatch')
    banks=[load_development_bank(p) for p in args.banks]
    if any(b['episode'] not in EPISODES for b,_ in banks):raise RuntimeError('Only five predetermined development starts')
    protocol={'episodes':[b['episode'] for b,_ in banks],'conditions':CONDITIONS,'new_cem_arms':NEW_ARMS,
              'candidate_sha256':frozen['candidate_sha256'],'original_beta':.5,'cap_radius':frozen['cap_radius'],
              'negative_policy':'Exact negation AFTER original beta0.5 capped update; no refit or dose search',
              'native_cem':{'H':6,'samples':300,'iterations':15},'raw_steps_new_plan':15,
              'finite_difference_epsilons_normalized':EPSILONS,'time_limit_seconds':args.time_limit_seconds,
              'cross_process_float32_bound':1e-6,'cross_process_ranks_exact':True,'within_run_identity':'bit exact',
              'script_sha256':file_sha256(Path(__file__)),'held_sources_opened':False,'no_refit':True,
              'restore_baseline_encoded_goal':args.restore_baseline_encoded_goal,'checkpoint_sha256':CHECKPOINT_SHA,
              'limitations':['Action blocking alone is not proof of compensation','Sign/Jacobian agreement is local evidence, not intentionality or formal mediation','Fixed first15steps, not closed-loop full-episode efficacy','Old native commands may exceed action bounds; never silently clipped by this runner']}
    write_json_atomic(args.output_dir/'protocol.json',protocol)
    def save(name,value,**meta):
        path=args.output_dir/name;path.parent.mkdir(parents=True,exist_ok=True);torch.save(value,path)
        row={'path':name,'sha256':file_sha256(path),'bytes':path.stat().st_size,**meta};outputs.append(row)
        write_json_atomic(args.output_dir/'progress.json',{'outputs':outputs});print(json.dumps({'event':'saved',**row}),flush=True)
    try:
        wm,preprocessor,provenance=load_headless_metaworld(args.repo);wm.eval().requires_grad_(False)
        if file_sha256(Path(provenance['checkpoint']))!=CHECKPOINT_SHA:raise RuntimeError('Original paired checkpoint SHA changed')
        cfg=setup_cfg(Path(provenance['config']),args.output_dir,wm);block=wm.model.predictor.predictor_blocks[3]
        calibration=CoordinateCalibration(dict(np.load(candidate)),device='cuda:0');negative=SignedCalibration(calibration,-1)
        import run_coordinate_steering_pilot as pilot
        original_start=pilot.fresh_start;references={}
        if args.restore_baseline_encoded_goal:
            def fixed_goal_start(cfg,wm,preprocessor,bank):
                env,agent,td,snapshot,mode=original_start(cfg,wm,preprocessor,bank)
                mode=mode|restore_fixed_encoded_goal(agent,snapshot,references[bank['episode']])
                return env,agent,td,snapshot,mode
            pilot.fresh_start=fixed_goal_start
        for bank,banksha in banks:
            ep=bank['episode'];plans,sources=load_previous(args.previous_dir,ep);base=plans['unsteered']
            references[ep]=base['initial_snapshot']
            print(json.dumps({'event':'fixed_command_controls_started','episode':ep}),flush=True)
            blocked=fixed_command_controls(cfg,wm,preprocessor,bank,calibration,block,base)
            save(f'episode-{ep:03}/fixed_command_controls.pt',blocked,episode=ep,stage='blocked_control')
            for arm in NEW_ARMS:
                if time.monotonic()-started>args.time_limit_seconds:raise RuntimeError('Frozen worker budget elapsed')
                print(json.dumps({'event':'negative_native_cem_started','episode':ep,'arm':arm[0]}),flush=True)
                result=run_arm(cfg,wm,preprocessor,bank,negative,block,arm,base['initial_snapshot'],base['first_candidates']['actions'])
                result['bank_sha256']=banksha;plans[arm[0]]=result
                save(f'episode-{ep:03}/{arm[0]}.pt',result,episode=ep,stage='new_native_plan',arm=arm[0],goal_progress_m=result['goal_progress_m'])
            crossed=native_cross_evaluate(cfg,wm,preprocessor,bank,calibration,block,plans,base)
            crossed.update(complete=True,episode=ep,bank_sha256=banksha,previous_sources=sources,
                           actual_by_plan={name:{k:p[k] for k in ('states','raw_actions','actual_visual_horizons1to3','initial_goal_distance_m','final_goal_distance_m','goal_progress_m','ever_success','final_success')} for name,p in plans.items()})
            save(f'episode-{ep:03}/crossed_forecasts_and_jacobians.pt',crossed,episode=ep,stage='crossed_native_forecasts')
            del plans,blocked,crossed
        write_json_atomic(args.output_dir/'DONE.json',protocol|{'complete':True,'outputs':outputs,'seconds':time.monotonic()-started,'model':provenance})
    except Exception as exc:
        write_json_atomic(args.output_dir/'FAILED.json',{'error':repr(exc),'outputs':outputs});raise


if __name__=='__main__':main()
