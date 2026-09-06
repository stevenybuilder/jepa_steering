#!/usr/bin/env python3
"""Frozen eleven-arm replication on all64 existing actions in each seen DEV state."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from run_patch_policy_action_spatial import CHECKPOINT, sha, write, exact, scratch, cpu

import shutil
from run_joint_visual_action_factorial import FACTORS, Factorial, vector_interaction, metric_decomposition

@torch.no_grad()
def run_source(args,row,wm,prep):
    from tensordict import TensorDict
    from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
    from diagnose_pusht_goal_coverage import coverage,native_helper
    path=args.bank/row['path']
    if sha(path)!=row['sha256']:raise ValueError('Bank hash mismatch before tensor load')
    bank=torch.load(path,map_location='cpu',weights_only=False)
    if bank['row']['split']!='development_external' or len(bank['candidates'])!=64 or not bank['original9_actions_goal_physics_pixels_exact']:raise ValueError('Only verified64 original development sources')
    z=TensorDict({k:v.repeat_interleave(64,dim=0) for k,v in bank['context'].items()},batch_size=[]).cuda()
    z_before=z.clone()
    normalized=prep.normalize_actions(bank['raw_actions'].reshape(64,6,5,2)).reshape(64,6,10)
    exact(normalized,bank['normalized_actions'],'official normalized plans unchanged')
    actions=normalized.transpose(0,1).contiguous().cuda();saved_actions=actions.clone()
    goal=TensorDict(bank['goal_encoded'],batch_size=[]).cuda()
    objective=ReprTargetDistMPCObjective({},goal,sum_all_diffs=False,alpha=.1)
    start=time.monotonic();native=wm.unroll(z.clone(),act_suffix=actions)
    with Factorial(wm.model.predictor) as capture:repeat=wm.unroll(z.clone(),act_suffix=actions)
    for k in ('visual','proprio'):exact(native[k],repeat[k],'native repeat capture floor '+k)
    reference=capture.record
    expected=torch.nn.functional.linear((normalized[:,2]-normalized[0:1,2]).cuda(),wm.model.predictor.action_encoder.weight,bias=None)
    torch.testing.assert_close(reference['delta_z'],expected,rtol=2e-6,atol=2e-6)
    if not (reference['delta_z'][1:].norm(dim=-1)>0).any() or not (reference['delta_visual'][1:].flatten(1).norm(dim=-1)>0).any():
        raise ValueError('Required nonzero components absent from entire bank')
    with Factorial(wm.model.predictor,reference,'joint',0.) as zero:identity=wm.unroll(z.clone(),act_suffix=actions)
    for k in ('visual','proprio'):exact(native[k],identity[k],'joint zero identity '+k)
    predictions={'native':cpu({k:native[k][1:] for k in ('visual','proprio')})}
    costs={'native':objective(native,actions,keepdims=True)[1:].cpu()}
    records={'native':cpu(reference)}
    for sign in (-1,1):
        for mode in FACTORS:
            name=mode+('_minus' if sign<0 else '_plus')
            with Factorial(wm.model.predictor,reference,mode,sign*.1) as hook:
                out=wm.unroll(z.clone(),act_suffix=actions)
            for k in ('visual','proprio'):
                exact(out[k][:3],native[k][:3],'pre-H3 prediction identity '+name+'/'+k)
                exact(out[k][:,0],native[k][:,0],'central trajectory identity '+name+'/'+k)
            predictions[name]=cpu({k:out[k][1:] for k in ('visual','proprio')})
            costs[name]=objective(out,actions,keepdims=True)[1:].cpu();records[name]=cpu(hook.record)
    for sign in ('minus','plus'):
        exact(records['action_'+sign]['block_x_before'],records['native']['block_x_before'],'condition-only block x unchanged')
        for prefix in ('','permuted_'):
            exact(records[prefix+'visual_'+sign]['block_x_before'],records[prefix+'joint_'+sign]['block_x_before'],'joint and visual-only same P3 x')
    # Actual outcomes are consulted only AFTER all model interventions complete.
    actual={'visual':torch.stack([c['actual_visual'] for c in bank['candidates']],1).float()}
    states=np.stack([c['states'] for c in bank['candidates']]);raw_prop=torch.tensor(states[:,::5][:,:,[0,1,5,6]],dtype=torch.float32)
    actual['proprio']=wm.model.encode_proprio(prep.normalize_proprios(raw_prop).cuda()).cpu()[:,1:].transpose(0,1).contiguous()
    future=states[:,5::5];gs=np.asarray(bank['goal_state'])
    goal_coverage=np.array([[coverage(s[2:5],gs[2:5]) for s in c] for c in future]).T
    painted=np.array([[coverage(s[2:5],[256.,256.,np.pi/4]) for s in c] for c in future]).T
    xy=np.linalg.norm(future[:,:,:4]-gs[:4],axis=-1).T
    polygon=native_helper(args.repo/'evals/simu_env_planning/envs/pusht_env/pusht_env.py');target=polygon(gs[2:5])
    checked=np.array([polygon(s[2:5]).intersection(target).area/target.area for s in future[:,-1]])
    if np.max(np.abs(checked-goal_coverage[-1]))>1e-12:raise ValueError('Official requested-goal polygon parity')
    crossbatch={}
    for field in ('visual','proprio'):
        old=torch.stack([c['predicted_'+field] for c in bank['candidates']],1).float()
        difference=(predictions['native'][field]-old).abs()
        crossbatch[field]=dict(maxabs=float(difference.max()),original9_maxabs=float(difference[:,:9].max()),
            dtype=str(old.dtype),diagnostic_bound=1e-4,within_diagnostic_bound=bool(difference.max()<=1e-4))
    oldcost=torch.cat([c['native_cost_by_horizon'] for c in bank['candidates']],1)[1:]
    crossbatch['cost']=dict(maxabs=float((costs['native']-oldcost).abs().max()),diagnostic_bound=1e-5,
        within_diagnostic_bound=bool((costs['native']-oldcost).abs().max()<=1e-5))
    baseline_choice=int(costs['native'][-1].argmin());score=[];interactions=[];vectors={}
    for name,p in predictions.items():
        error={k:(p[k].double()-actual[k].double()).square().flatten(2).mean(-1) for k in actual}
        choice=int(costs[name][-1].argmin())
        score.append(dict(arm=name,choice=choice,cost_by_horizon=costs[name].tolist(),rank_by_horizon=costs[name].argsort(dim=1,stable=True).tolist(),
            mse_by_horizon_candidate={k:v.tolist() for k,v in error.items()},equal_candidate_mse_by_horizon={k:v.mean(1).tolist() for k,v in error.items()},
            selected_goal_coverage=float(goal_coverage[-1,choice]),goal_coverage_delta=float(goal_coverage[-1,choice]-goal_coverage[-1,baseline_choice]),
            selected_xy_distance_px=float(xy[-1,choice]),xy_delta_px=float(xy[-1,choice]-xy[-1,baseline_choice]),
            selected_native_painted_coverage=float(painted[-1,choice]),
            visual_delivered_l2=records[name].get('visual_delivered_l2',torch.zeros(64)).tolist(),
            condition_delivered_l2=records[name].get('condition_delivered_l2',torch.zeros(64)).tolist()))
    for sign in ('minus','plus'):
        for prefix in ('','permuted_'):
            names=('native',prefix+'visual_'+sign,'action_'+sign,prefix+'joint_'+sign)
            key=(prefix or 'dense_')+sign;vectors[key]={};report=dict(sign=sign,visual_kind=prefix or 'dense',output={},metric_to_actual={},metric_to_goal={})
            for field in ('visual','proprio'):
                values=[predictions[n][field] for n in names]
                interaction,stats,additive=vector_interaction(*values)
                vectors[key][field]=interaction.float();report['output'][field]=stats
                report['metric_to_actual'][field]=metric_decomposition(*values,actual[field])
                target=bank['goal_encoded'][field].cpu().expand_as(values[0])
                report['metric_to_goal'][field]=metric_decomposition(*values,target)
            direct=[records[n]['block_output'][None] for n in names]
            interaction,stats,_=vector_interaction(*direct);vectors[key]['block_output']=interaction.float();report['direct_block']=stats
            interactions.append(report)
    exact(actions,saved_actions,'external model action plans unchanged')
    for key in z.keys():exact(z[key],z_before[key],'initial context unchanged '+key)
    strata={}
    for i,metadata in enumerate(bank['metadata']):
        label=metadata['family']
        if 'radius' in metadata:label+=':radius='+str(metadata['radius'])
        if 'normalized_rms' in metadata:label+=':rms='+str(metadata['normalized_rms'])
        if 'normalized_rms_radius' in metadata:label+=':rms='+str(metadata['normalized_rms_radius'])
        strata.setdefault(label,[]).append(i)
    for item in score:
        item['radius_strata']={label:dict(candidate_indices=ids,count=len(ids),
            visual_mse_by_horizon=np.asarray(item['mse_by_horizon_candidate']['visual'])[:,ids].mean(1).tolist(),
            proprio_mse_by_horizon=np.asarray(item['mse_by_horizon_candidate']['proprio'])[:,ids].mean(1).tolist())
            for label,ids in strata.items()}
    file=args.output/(row['key']+'.pt')
    torch.save(dict(complete=True,source_sha256=row['sha256'],raw_actions=bank['raw_actions'],normalized_actions=normalized,
        initial_context=bank['context'],goal_encoded=bank['goal_encoded'],goal_state=gs,predictions=predictions,costs=costs,
        records=records,interaction_vectors=vectors,actual=actual,states=states,score=score,interactions=interactions,metadata=bank['metadata'],crossbatch=crossbatch),file)
    torch.cuda.synchronize();seconds=time.monotonic()-start
    report=dict(complete=True,key=row['key'],seconds=seconds,candidate_metadata=bank['metadata'],crossbatch=crossbatch,original9_actions_goal_physics_pixels_exact=True,source_sha256=row['sha256'],score=score,interactions=interactions,
        action_embedding_range_maxabs=float((reference['delta_z']-expected).abs().max()),
        visual_delta_l2=reference['delta_visual'].double().flatten(1).norm(dim=-1).cpu().tolist(),
        condition_delta_l2=reference['delta_z'].double().norm(dim=-1).cpu().tolist(),
        exact_repeat_output_floor=0.,all_identity_guards_exact=True,
        goal_coverage_headroom=float(goal_coverage[-1].max()-goal_coverage[-1,baseline_choice]),xy_headroom_px=float(xy[-1,baseline_choice]-xy[-1].min()),
        goal_coverage_by_horizon=goal_coverage.tolist(),full_tensor_sha256=sha(file),full_tensor_bytes=file.stat().st_size)
    write(file.with_suffix('.json'),report)
    print(json.dumps(dict(event='joint_factorial_state_complete',key=row['key'],seconds=seconds,choices={r['arm']:r['choice'] for r in score})),flush=True)
    return report,[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in (file,file.with_suffix('.json'))]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('bank','repo','checkpoint','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--episodes',type=int,nargs='+',required=True)
    p.add_argument('--max-seconds',type=float,default=150)
    args=p.parse_args();started=time.monotonic()
    rows=[]
    if not args.episodes or len(set(args.episodes))!=len(args.episodes) or any(e not in range(4) for e in args.episodes):raise ValueError('Fixed seen DEV shard only')
    for episode in args.episodes:
        root=args.bank/f'episode-{episode:03d}'
        done=json.loads((root/'DONE.json').read_text())
        if not done['complete'] or not done['original9_actions_goal_physics_pixels_exact']:raise ValueError('Input adapter incomplete')
        for r in done['outputs']:
            rows.append(dict(r,path=f'episode-{episode:03d}/'+r['path']))
    if [r['key'] for r in rows]!=[f'expanded-dev-{i:03d}' for i in args.episodes] or any(r['split']!='development_external' for r in rows):raise ValueError('Fixed shard source IDs before load')
    if sha(args.checkpoint)!=CHECKPOINT:raise ValueError('Pinned checkpoint mismatch')
    args.output.mkdir(parents=True,exist_ok=False)
    try:
        write(args.output/'protocol.json',dict(rows=rows,block_condition_site=3,imagined_horizon=3,arms=FACTORS,doses=[-.1,.1],
            visual_input='Newest imagined visual input frame[B,1,16,16,384] BEFORE predictor projection/proprio concatenation; previous visual frame and entire proprio field unchanged',
            visual_delta='recipient natural imagined input minus same-state central-plan natural imagined input, freshly captured; not actual camera pixels',
            action_condition='Fixed recipient-minus-central z from nativeLinear10to400; newest P3condition slot only, past slots unchanged; external plans NEVER changed',
            interaction='Vector yjoint-yvisual-yaction+ynative and direct block-output contrast; squared-error/cost interaction decomposed into additive cross-term plus output-coupling contribution',
            sham='Fixed256patch permutation of visual delta, equal raw Frobenius norm; both shamvisual-only and shamjoint arms',
            component_budget='Joint retains each single-component dose; x/z norms have different units, no equal-total-budget or confirmed joint-superiority claim',
            motivation_primary_sources={'conditional_action_identification':'https://arxiv.org/abs/2607.22430v2','same_action_visual_consistency':'https://arxiv.org/abs/2608.12939v1','dense_trained_policy':'https://arxiv.org/abs/2607.18236'},
            literature_scope='Parent-reviewed methodological motivation only; not a robotics guarantee or adaptation of a trained action/state identification method',
            limitations='Same-state donors differ in imagined action/history; visual/condition edits may be inconsistent with external plan; n4states not256independent plans; no significance or efficacy gate',
            candidate_count=64,candidate_breadth='Current chunk rank10 versus original9 rank4; added radii5–20x old, breadth/radius/bank all change; no additional independent states',crossbatch_numerical_policy='Saved1/9 versus64 batch forecasts descriptive1e-4 andcost1e-5; within64 native/zero/repeat EXACT',max_process_seconds=args.max_seconds,checkpoint_sha256=CHECKPOINT,script_sha256=sha(__file__),held_access=False,cem_calls=0,simulator_calls=0))
        from model_loader import load_headless
        torch.set_num_threads(2);torch.manual_seed(90505)
        wm,prep,provenance=load_headless(args.repo,model_name='jepa_wm_pusht',checkpoint_override=args.checkpoint);wm.eval().requires_grad_(False)
        def params_sha():
            h=hashlib.sha256()
            for name,v in wm.state_dict().items():h.update(name.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
            return h.hexdigest()
        before=params_sha();init=time.monotonic()-started;reports=[];outputs=[]
        for i,row in enumerate(rows):
            if time.monotonic()-started>args.max_seconds:raise ValueError('Process budget reached before next state')
            projected_bytes=3_350_000_000 if not reports else reports[-1]['full_tensor_bytes']+20_000_000
            if shutil.disk_usage(args.output).free<1_000_000_000+projected_bytes:raise ValueError('Pre-state disk reserve insufficient')
            report,entries=run_source(args,row,wm,prep);reports.append(report);outputs.extend(entries)
            if i==0:
                estimate=time.monotonic()-started+(len(rows)-1)*report['seconds'];write(args.output/'CANARY.json',dict(complete=True,estimated_seconds=estimate,max_seconds=args.max_seconds,first_state_bytes=report['full_tensor_bytes'],disk_free_bytes=shutil.disk_usage(args.output).free))
                print(json.dumps(dict(event='joint_factorial_canary',estimated_seconds=estimate)),flush=True)
                if estimate>args.max_seconds:raise ValueError('First-state estimate exceeds shard cap')
        if params_sha()!=before:raise ValueError('Frozen parameter hash changed')
        aggregate=[]
        for i,first in enumerate(reports[0]['score']):
            values=[r['score'][i] for r in reports]
            aggregate.append(dict(arm=first['arm'],choices=[v['choice'] for v in values],
                mean_goal_coverage_delta=float(np.mean([v['goal_coverage_delta'] for v in values])),mean_xy_delta_px=float(np.mean([v['xy_delta_px'] for v in values])),
                equal_state_mse_by_horizon={k:np.mean([v['equal_candidate_mse_by_horizon'][k] for v in values],axis=0).tolist() for k in ('visual','proprio')}))
        write(args.output/'summary.json',dict(complete=True,independent_states=len(rows),candidate_count=64,episodes=args.episodes,aggregate=aggregate,seconds=time.monotonic()-started,
            native_vs_zero_repeat_exact=True,no_new_simulator_or_cem=True,per_state_reports=[r['key']+'.json' for r in reports]))
        for path in (args.output/'protocol.json',args.output/'CANARY.json',args.output/'summary.json'):outputs.append(dict(path=path.name,sha256=sha(path),bytes=path.stat().st_size))
        write(args.output/'DONE.json',dict(complete=True,outputs=outputs,seconds=time.monotonic()-started,parameter_sha256=before,provenance=provenance,all_identity_guards_exact=True,held_access=False,cem_calls=0,simulator_calls=0))
    except Exception:
        write(args.output/'FAILED.json',dict(complete=False,exception=traceback.format_exc()));raise


if __name__=='__main__':main()
