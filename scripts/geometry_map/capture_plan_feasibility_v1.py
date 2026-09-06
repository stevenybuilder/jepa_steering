#!/usr/bin/env python3
"""Frozen nine-plan H6 physical extension and confirmed XYZ-clip falsification.

No CEM, refit, held data, or gripper clipping. RawXYZ commands are explicitly
clipped by the installed Metaworld set_xyz_action, independent of action Box.
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

MEAN=(.005723577458411455,.15735651552677155,-.1396457850933075,.1998193860054016)
STD=(.7359239459037781,.73408043384552,.7182977199554443,.743205726146698)
EPISODES=(0,4,7)


def xyz_clip(raw):
    if raw.shape!=(30,4) or not torch.isfinite(raw).all():raise ValueError('Finite raw30x4 actions required')
    answer=raw.clone();answer[:,:3]=answer[:,:3].clamp(-1,1)
    return answer


def clip_metrics(normalized):
    raw=normalized.reshape(30,4).float()*torch.tensor(STD)+torch.tensor(MEAN)
    clipped=xyz_clip(raw);renormalized=(clipped-torch.tensor(MEAN))/torch.tensor(STD)
    return raw,clipped,{'raw_xyz_clipped_fraction':float((raw[:,:3].abs()>1).float().mean()),
                        'raw_xyz_clipped_count':int((raw[:,:3].abs()>1).sum()),
                        'raw_gripper_out_of_unit_box_count_not_modified':int((raw[:,3].abs()>1).sum()),
                        'raw_translation_l2_change':float((raw[:,:3]-clipped[:,:3]).norm()),
                        'normalized_projection_l2':float((renormalized-normalized.reshape(30,4)).norm()),
                        'normalized_changed_translation_coordinate_indices':torch.nonzero((raw-clipped).reshape(-1)!=0).flatten().tolist(),
                        'raw_gripper_unchanged':bool(torch.equal(raw[:,3],clipped[:,3]))}


def pack(args):
    args.output_dir.mkdir(parents=True,exist_ok=False);outputs=[];all_metrics=[]
    for episode,worker,version in ((0,49982193,'v1'),(1,49982193,'v1'),(4,49982195,'v1'),(7,49982195,'v1'),(10,49987405,'v3')):
        root=args.compensation_root/f'worker-{worker}'/f'results-{version}';receipt=json.loads((root/'DONE.json').read_text())
        old=args.previous_root/f'worker-{worker}'/'results-v1';old_done=json.loads((old/'DONE.json').read_text())
        if not receipt['complete'] or not old_done['complete'] or receipt['held_sources_opened']:raise RuntimeError('Frozen completed development sources required')
        plans={};sources=[];snapshot=None
        for directory,entries in ((old,old_done['outputs']),(root,receipt['outputs'])):
            for row in entries:
                if row.get('episode')!=episode or 'arm' not in row:continue
                path=directory/row['path']
                if file_sha256(path)!=row['sha256']:raise RuntimeError('Source SHA mismatch')
                value=torch.load(path,map_location='cpu',weights_only=False)
                if row['arm']=='unsteered':snapshot=value['initial_snapshot']
                raw,clipped,metrics=clip_metrics(value['selected_plan_fullH6'])
                if not torch.equal(raw[:15],value['raw_actions']):raise RuntimeError('Official affine action stats disagree with actual saved raw prefix')
                plans[row['arm']]={k:value[k] for k in ('selected_plan_fullH6','frames','states','proprios','raw_actions')}
                plans[row['arm']]['raw_fullH6']=raw;plans[row['arm']]['clipping']=metrics
                all_metrics.append({'episode':episode,'arm':row['arm'],**metrics});sources.append({'path':str(path),'sha256':row['sha256']})
        if len(plans)!=9 or snapshot is None:raise RuntimeError('Expected allnine saved plans and observed start')
        if episode in EPISODES:
            value={'episode':episode,'plans':plans,'initial_snapshot':snapshot,'sources':sources,'action_mean':MEAN,'action_std':STD}
            path=args.output_dir/f'episode-{episode:03}.pt';torch.save(value,path)
            outputs.append({'path':path.name,'sha256':file_sha256(path),'bytes':path.stat().st_size,'episode':episode})
    write_json_atomic(args.output_dir/'all45_clipping_metrics.json',{'complete':True,'rows':all_metrics,'action_mean':MEAN,'action_std':STD,
                      'source':'Installed official get_data_stats(metaworld); affine normalize/denormalize source inspected; all45 saved15-action raw prefixes exact',
                      'clipping_scope':'Confirmed XYZ only; gripper unchanged, workspace mocap saturation and actuator limits not approximated'})
    outputs.append({'path':'all45_clipping_metrics.json','sha256':file_sha256(args.output_dir/'all45_clipping_metrics.json')})
    write_json_atomic(args.output_dir/'DONE.json',{'complete':True,'outputs':outputs,'script_sha256':file_sha256(Path(__file__))})
    print(json.dumps({'event':'pack_complete','outputs':outputs}),flush=True)


@torch.no_grad()
def physical_plan(cfg,wm,preprocessor,bank,reference,plan,clip=False):
    from run_coordinate_steering_pilot import fresh_start,paired_start_checks,close_env
    from run_planner_compensation_v1 import restore_fixed_encoded_goal
    from run_coordinate_dose_timing import exact
    from collect_on_policy_bank import physics_snapshot
    from evals.simu_env_planning.planning.utils import make_td
    env,agent,td,snapshot,goal_mode=fresh_start(cfg,wm,preprocessor,bank)
    try:
        goal_mode=goal_mode|restore_fixed_encoded_goal(agent,snapshot,reference['initial_snapshot'])
        paired_start_checks(snapshot,reference['initial_snapshot'])
        method=env.proprio_env.unwrapped.set_xyz_action;source=inspect.getsource(method)
        if 'np.clip(action, -1, 1)' not in source:raise RuntimeError('Current environment does not confirm the frozen XYZ clip rule')
        normalized=plan['selected_plan_fullH6'].clone();raw=preprocessor.denormalize_actions(normalized.reshape(30,4))
        exact(raw,plan['raw_fullH6'],'official raw H6 conversion')
        if clip:
            raw=xyz_clip(raw);normalized=preprocessor.normalize_actions(raw).reshape(6,20)
        z=wm.encode(td.cuda().unsqueeze(0),act=True);before={k:z[k].clone() for k in z.keys()}
        prediction=wm.unroll(z.clone(),act_suffix=normalized[:,None].cuda())
        for k,v in before.items():exact(z[k],v,'unedited observation '+k)
        costs=agent.objective(prediction,normalized[:,None].cuda(),keepdims=True).cpu()
        observations,rewards,dones,infos=env.step_multiple(raw)
        if len(observations)!=30:raise RuntimeError('Expected exact30 raw simulator steps')
        frames=torch.stack(observations).cpu();states=np.stack([np.asarray(i['state']) for i in infos])
        proprios=torch.stack([torch.as_tensor(i['proprio']) for i in infos]).cpu()
        exact(frames[:15],plan['frames'],'saved15 prefix pixels')
        exact(states[:15],plan['states'],'saved15 prefix observed states')
        exact(proprios[:15],plan['proprios'],'saved15 prefix proprio')
        encoded={'visual':[],'proprio':[]}
        for image,proprio in zip(frames[4::5],proprios[4::5]):
            obs=make_td(image,{'proprio':proprio});features=wm.encode(obs.cuda().unsqueeze(0),act=True)
            for key in encoded:encoded[key].append(features[key].float().cpu().reshape(-1))
        encoded={k:torch.stack(v) for k,v in encoded.items()}
        true_cost=0
        for key,weight in (('visual',1.),('proprio',.1)):
            target=snapshot['goal_encoded_'+key].float().reshape(1,-1)
            if target.shape[-1]!=encoded[key].shape[-1]:raise RuntimeError('Actual/goal encoded width disagreement: '+key)
            true_cost=true_cost+weight*(encoded[key]-target).square().mean(-1)
        goal=np.asarray(env.proprio_env.unwrapped._target_pos).copy();initial=snapshot['proprio'].numpy().reshape(-1,4)[-1,:3]
        distances=np.linalg.norm(states[4::5,:3]-goal,axis=-1)
        return {'complete':True,'episode':bank['episode'],'xyz_clipped':clip,'goal_mode':goal_mode,
                'normalized_actions':normalized.cpu(),'raw_actions':raw.cpu(),'states':states,'frames':frames,'proprios':proprios,
                'rewards':np.asarray(rewards),'ever_success':any(bool(i['success']) for i in infos),'final_success':bool(infos[-1]['success']),
                'predicted_visual':prediction['visual'].float().cpu(),'predicted_proprio':prediction['proprio'].float().cpu(),
                'predicted_native_goal_cost_by_step':costs,'actual_encoded':encoded,'actual_native_goal_cost_h1_to_h6':true_cost,
                'physical_goal_xyz':goal,'actual_goal_distance_h1_to_h6_m':distances,'initial_goal_distance_m':float(np.linalg.norm(initial-goal)),
                'goal_progress_h1_to_h6_m':float(np.linalg.norm(initial-goal))-distances,
                'final_physics':physics_snapshot(env),'prefix15_exact':True,'environment_clip_source':source,
                'environment_source_path':inspect.getfile(method),'environment_source_sha256':file_sha256(Path(inspect.getfile(method)))}
    finally:close_env(env)


def run(args):
    from causal_planner_forks import load_development_bank,setup_cfg
    from model_loader import load_headless_metaworld
    from run_planner_compensation_v1 import CHECKPOINT_SHA
    from run_coordinate_dose_timing import exact
    args.output_dir.mkdir(parents=True,exist_ok=False);started=time.monotonic();outputs=[]
    receipt=json.loads((args.plan_input.parent/'DONE.json').read_text());entry=next(x for x in receipt['outputs'] if x['path']==args.plan_input.name)
    if not receipt['complete'] or file_sha256(args.plan_input)!=entry['sha256']:raise RuntimeError('Thin plan input SHA mismatch')
    reference=torch.load(args.plan_input,map_location='cpu',weights_only=False);bank,banksha=load_development_bank(args.bank)
    if bank['episode'] not in EPISODES or reference['episode']!=bank['episode']:raise RuntimeError('Frozen three development starts only')
    protocol={'episode':bank['episode'],'plans':list(reference['plans']),'plan_input_sha256':entry['sha256'],'bank_sha256':banksha,
              'raw_steps':30,'new_cem_calls':0,'model_interventions':0,'xyz_clip_falsification':'baseline only, gripper unchanged',
              'required_guards':'Exact saved15-step prefix for all9 raw plans; clipped baseline must reproduce entire30-step raw trajectory',
              'native_goal_cost':'Full-spatial visual MSE +0.1 proprio MSE at actual observed H6, not a decoded physical proxy',
              'held_sources_opened':False,'checkpoint_sha256':CHECKPOINT_SHA,'script_sha256':file_sha256(Path(__file__)),'budget_seconds':900}
    write_json_atomic(args.output_dir/'protocol.json',protocol)
    try:
        wm,preprocessor,provenance=load_headless_metaworld(args.repo);wm.eval().requires_grad_(False)
        if file_sha256(Path(provenance['checkpoint']))!=CHECKPOINT_SHA:raise RuntimeError('Frozen checkpoint changed')
        cfg=setup_cfg(Path(provenance['config']),args.output_dir,wm);base=None
        for name,plan in reference['plans'].items():
            if time.monotonic()-started>900:raise RuntimeError('Frozen physical extension budget elapsed')
            print(json.dumps({'event':'physical30_started','episode':bank['episode'],'plan':name}),flush=True)
            result=physical_plan(cfg,wm,preprocessor,bank,reference,plan)
            if name=='unsteered':base=result
            path=args.output_dir/(name+'.pt');torch.save(result,path)
            row={'path':path.name,'sha256':file_sha256(path),'bytes':path.stat().st_size,'plan':name,'episode':bank['episode'],
                 'physical_progress_h6_m':float(result['goal_progress_h1_to_h6_m'][-1]),'actual_native_cost_h6':float(result['actual_native_goal_cost_h1_to_h6'][-1]),
                 'predicted_native_cost_h6':float(result['predicted_native_goal_cost_by_step'][-1,0]),'prefix15_exact':True,'ever_success':result['ever_success']}
            outputs.append(row);write_json_atomic(args.output_dir/'progress.json',{'outputs':outputs});print(json.dumps({'event':'physical30_complete',**row}),flush=True)
        if base is None:raise RuntimeError('Baseline missing')
        clipped=physical_plan(cfg,wm,preprocessor,bank,reference,reference['plans']['unsteered'],clip=True)
        for key in ('states','frames','proprios','rewards'):exact(clipped[key],base[key],'full30 XYZ-projection equivalence '+key)
        for key,value in base['final_physics']['physics'].items():exact(clipped['final_physics']['physics'][key],value,'full30 clipped finalphysics '+key)
        clipped['full30_raw_clipped_equivalence_exact']=True
        path=args.output_dir/'baseline_xyz_clipped.pt';torch.save(clipped,path)
        outputs.append({'path':path.name,'sha256':file_sha256(path),'bytes':path.stat().st_size,'episode':bank['episode'],'plan':'baseline_xyz_clipped',
                        'full30_equivalence_exact':True,'predicted_native_cost_h6':float(clipped['predicted_native_goal_cost_by_step'][-1,0])})
        write_json_atomic(args.output_dir/'DONE.json',protocol|{'complete':True,'outputs':outputs,'seconds':time.monotonic()-started,'model':provenance})
    except Exception as exc:
        write_json_atomic(args.output_dir/'FAILED.json',{'error':repr(exc),'outputs':outputs});raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('pack','run'))
    for key in ('compensation-root','previous-root','repo','bank','plan-input'):p.add_argument('--'+key,type=Path)
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
    (pack if a.mode=='pack' else run)(a)


if __name__=='__main__':main()
