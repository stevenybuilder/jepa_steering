#!/usr/bin/env python3
"""True predictor-visual input x current P3 AdaLN-condition factorial at H3."""
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

FACTORS = ('visual', 'action', 'joint', 'permuted_visual', 'permuted_joint')


def vector_interaction(native, visual, action, joint):
    """Double arithmetic distinguishes output coupling from quadratic scoring."""
    b,v,a,j=(x.double() for x in (native,visual,action,joint))
    dv,da=v-b,a-b; additive=v+a-b; interaction=j-additive
    axes=tuple(range(2,b.ndim))
    norm=lambda x:x.square().sum(axes).sqrt()
    denominator=norm(dv)+norm(da)
    floor=(additive.float().double()-additive).abs()
    return interaction,dict(raw_l2=norm(interaction).cpu().tolist(),
        relative_to_sum_individual_norms=(norm(interaction)/denominator.clamp_min(1e-30)).cpu().tolist(),
        individual_visual_l2=norm(dv).cpu().tolist(),individual_action_l2=norm(da).cpu().tolist(),
        additive_float32_rounding_l2=norm(floor).cpu().tolist(),
        interaction_storage_float32_maxabs=float((interaction.float().double()-interaction).abs().max())),additive


def metric_decomposition(native, visual, action, joint, truth):
    b,v,a,j,t=(x.double() for x in (native,visual,action,joint,truth))
    dv,da=v-b,a-b;r=j-v-a+b;additive=v+a-b
    axes=tuple(range(2,b.ndim));mse=lambda y:(y-t).square().mean(axes)
    contrast=mse(j)-mse(v)-mse(a)+mse(b)
    additive_cross=2*(dv*da).mean(axes)
    coupling=2*((additive-t)*r).mean(axes)+r.square().mean(axes)
    torch.testing.assert_close(contrast,additive_cross+coupling,rtol=1e-7,atol=1e-10)
    return dict(metric_interaction=contrast.cpu().tolist(),pure_quadratic_additive_cross_term=additive_cross.cpu().tolist(),
        nonlinear_output_coupling_term=coupling.cpu().tolist(),additive_forecast_mse=mse(additive).cpu().tolist(),
        joint_forecast_mse=mse(j).cpu().tolist(),identity_decomposition_maxabs=float((contrast-additive_cross-coupling).abs().max()))


def input_edits(args, delta_visual, mode, dose):
    visual,actions,proprio=args
    if mode not in FACTORS and mode!='native':raise ValueError('Unknown fixed factorial arm')
    if visual.shape[1:]!=(2,1,16,16,384):raise ValueError('H3 true visual field must retain native2frame geometry')
    if mode in ('action','native') or dose==0:return args
    change=delta_visual
    if mode.startswith('permuted'):
        gen=torch.Generator(device='cpu').manual_seed(2026090623)
        perm=torch.randperm(256,generator=gen).to(change.device)
        change=change.reshape(len(change),256,384)[:,perm].reshape_as(change)
    edited=scratch(visual);edited[:,-1]+=dose*change
    exact(edited[:,:-1],visual[:,:-1],'past visual frame unchanged')
    exact(edited[0],visual[0],'central visual input exact')
    return (edited,actions,proprio)


class Factorial:
    def __init__(self,predictor,record=None,mode='native',dose=0.):
        self.predictor=predictor;self.block=predictor.predictor_blocks[3]
        self.reference=record;self.mode=mode;self.dose=dose;self.calls=[0,0,0];self.record={}
    def __enter__(self):
        self.handles=[self.predictor.register_forward_pre_hook(self.visual),
            self.block.register_forward_pre_hook(self.condition,with_kwargs=True),
            self.block.register_forward_hook(self.output)]
        return self
    def visual(self,module,args):
        self.calls[0]+=1
        if self.calls[0]!=3:return None
        if len(args)!=3:raise ValueError('Native predictor visual/action/proprio signature required')
        if self.reference is None:
            self.record['predictor_inputs']=tuple(cpu(x) for x in args)
            self.record['delta_visual']=(args[0][:,-1]-args[0][0:1,-1]).detach().clone()
            exact(self.record['delta_visual'][0],torch.zeros_like(self.record['delta_visual'][0]),'central visual donor null')
            return None
        for i,value in enumerate(args):exact(value.cpu(),self.reference['predictor_inputs'][i],f'same pre-edit predictor input{i}')
        changed=input_edits(args,self.reference['delta_visual'],self.mode,self.dose)
        exact(changed[1],args[1],'all official action inputs unchanged')
        exact(changed[2],args[2],'entire true proprio input unchanged')
        self.record['visual_delivered_l2']=(changed[0][:,-1]-args[0][:,-1]).double().flatten(1).norm(dim=-1).cpu()
        return changed
    def condition(self,module,args,kwargs):
        self.calls[1]+=1
        if self.calls[1]!=3:return None
        x,z=args
        if x.shape[1:]!=(512,400) or z.shape[1:]!=(2,400):raise ValueError('Pinned P3 H3 shapes differ')
        self.record['block_x_before']=x[:,-256:].detach().cpu().clone()
        if self.reference is None:
            self.record['block_z']=z.detach().cpu().clone()
            self.record['delta_z']=(z[:,-1]-z[0:1,-1]).detach().clone()
            return None
        exact(z.cpu(),self.reference['block_z'],'fixed unedited action condition across ALL arms')
        if self.mode in ('native','visual','permuted_visual') or self.dose==0:
            self.record['condition_delivered_l2']=torch.zeros(len(z));return None
        changed=scratch(z);changed[:,-1]+=self.dose*self.reference['delta_z']
        exact(changed[:,:-1],z[:,:-1],'all past action conditions unchanged')
        exact(changed[0],z[0],'central action-condition input exact')
        self.record['condition_delivered_l2']=(changed[:,-1]-z[:,-1]).double().norm(dim=-1).cpu()
        return (x,changed),kwargs
    def output(self,module,args,out):
        self.calls[2]+=1
        if self.calls[2]==3:self.record['block_output']=out[:,-256:].detach().cpu().clone()
    def __exit__(self,*exc):
        for h in self.handles:h.remove()
        if exc[0] is None and self.calls!=[6,6,6]:raise ValueError('Expected six calls per native hook')


@torch.no_grad()
def run_source(args,row,wm,prep):
    from tensordict import TensorDict
    from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
    from diagnose_pusht_goal_coverage import coverage,native_helper
    path=args.bank/row['path']
    if sha(path)!=row['sha256']:raise ValueError('Bank hash mismatch before tensor load')
    bank=torch.load(path,map_location='cpu',weights_only=False)
    if bank['row']['split']!='development_external' or len(bank['candidates'])!=9:raise ValueError('Only near9 original development sources')
    z=TensorDict({k:v.repeat_interleave(9,dim=0) for k,v in bank['context'].items()},batch_size=[]).cuda()
    z_before=z.clone()
    normalized=prep.normalize_actions(bank['raw_actions'].reshape(9,6,5,2)).reshape(9,6,10)
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
    if not (reference['delta_z'][1:].norm(dim=-1)>0).all() or not (reference['delta_visual'][1:].flatten(1).norm(dim=-1)>0).all():
        raise ValueError('Required visual/action components are nonzero for every noncentral candidate')
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
    baseline_choice=int(costs['native'][-1].argmin());score=[];interactions=[];vectors={}
    for name,p in predictions.items():
        error={k:(p[k].double()-actual[k].double()).square().flatten(2).mean(-1) for k in actual}
        choice=int(costs[name][-1].argmin())
        score.append(dict(arm=name,choice=choice,cost_by_horizon=costs[name].tolist(),rank_by_horizon=costs[name].argsort(dim=1,stable=True).tolist(),
            mse_by_horizon_candidate={k:v.tolist() for k,v in error.items()},equal_candidate_mse_by_horizon={k:v.mean(1).tolist() for k,v in error.items()},
            selected_goal_coverage=float(goal_coverage[-1,choice]),goal_coverage_delta=float(goal_coverage[-1,choice]-goal_coverage[-1,baseline_choice]),
            selected_xy_distance_px=float(xy[-1,choice]),xy_delta_px=float(xy[-1,choice]-xy[-1,baseline_choice]),
            selected_native_painted_coverage=float(painted[-1,choice]),
            visual_delivered_l2=records[name].get('visual_delivered_l2',torch.zeros(9)).tolist(),
            condition_delivered_l2=records[name].get('condition_delivered_l2',torch.zeros(9)).tolist()))
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
    file=args.output/(row['key']+'.pt')
    torch.save(dict(complete=True,source_sha256=row['sha256'],raw_actions=bank['raw_actions'],normalized_actions=normalized,
        initial_context=bank['context'],goal_encoded=bank['goal_encoded'],goal_state=gs,predictions=predictions,costs=costs,
        records=records,interaction_vectors=vectors,actual=actual,states=states,score=score,interactions=interactions),file)
    torch.cuda.synchronize();seconds=time.monotonic()-start
    report=dict(complete=True,key=row['key'],seconds=seconds,source_sha256=row['sha256'],score=score,interactions=interactions,
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
    args=p.parse_args();started=time.monotonic()
    done=json.loads((args.bank/'DONE.json').read_text());rows=sorted([r for r in done['outputs'] if r['path'].endswith('.pt')],key=lambda r:r['key'])
    if not done['complete'] or [r['key'] for r in rows]!=[f'near-dev-{i:03d}' for i in range(4)] or any(r['split']!='development_external' for r in rows):raise ValueError('Only four original nearDEV sources before load')
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
            limitations='Same-state donors differ in imagined action/history; visual/condition edits may be inconsistent with external plan; n4states not36independent plans; no significance or efficacy gate',
            max_process_seconds=300,checkpoint_sha256=CHECKPOINT,script_sha256=sha(__file__),held_access=False,cem_calls=0,simulator_calls=0))
        from model_loader import load_headless
        torch.set_num_threads(2);torch.manual_seed(90505)
        wm,prep,provenance=load_headless(args.repo,model_name='jepa_wm_pusht',checkpoint_override=args.checkpoint);wm.eval().requires_grad_(False)
        def params_sha():
            h=hashlib.sha256()
            for name,v in wm.state_dict().items():h.update(name.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
            return h.hexdigest()
        before=params_sha();init=time.monotonic()-started;reports=[];outputs=[]
        for i,row in enumerate(rows):
            if time.monotonic()-started>300:raise ValueError('Process budget reached before next state')
            report,entries=run_source(args,row,wm,prep);reports.append(report);outputs.extend(entries)
            if i==0:
                estimate=init+4*report['seconds'];write(args.output/'CANARY.json',dict(complete=True,estimated_seconds=estimate,max_seconds=300))
                print(json.dumps(dict(event='joint_factorial_canary',estimated_seconds=estimate)),flush=True)
                if estimate>300:raise ValueError('First-state estimate exceeds300second cap')
        if params_sha()!=before:raise ValueError('Frozen parameter hash changed')
        aggregate=[]
        for i,first in enumerate(reports[0]['score']):
            values=[r['score'][i] for r in reports]
            aggregate.append(dict(arm=first['arm'],choices=[v['choice'] for v in values],
                mean_goal_coverage_delta=float(np.mean([v['goal_coverage_delta'] for v in values])),mean_xy_delta_px=float(np.mean([v['xy_delta_px'] for v in values])),
                equal_state_mse_by_horizon={k:np.mean([v['equal_candidate_mse_by_horizon'][k] for v in values],axis=0).tolist() for k in ('visual','proprio')}))
        write(args.output/'summary.json',dict(complete=True,independent_states=4,aggregate=aggregate,seconds=time.monotonic()-started,
            native_vs_zero_repeat_exact=True,no_new_simulator_or_cem=True,per_state_reports=[r['key']+'.json' for r in reports]))
        for path in (args.output/'protocol.json',args.output/'CANARY.json',args.output/'summary.json'):outputs.append(dict(path=path.name,sha256=sha(path),bytes=path.stat().st_size))
        write(args.output/'DONE.json',dict(complete=True,outputs=outputs,seconds=time.monotonic()-started,parameter_sha256=before,provenance=provenance,all_identity_guards_exact=True,held_access=False,cem_calls=0,simulator_calls=0))
    except Exception:
        write(args.output/'FAILED.json',dict(complete=False,exception=traceback.format_exc()));raise


if __name__=='__main__':main()
