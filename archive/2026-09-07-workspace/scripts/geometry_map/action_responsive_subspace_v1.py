#!/usr/bin/env python3
"""Response-only rank4/patch localization at true predictor visual INPUT H3."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from run_patch_policy_action_spatial import CHECKPOINT,sha,write,exact,scratch,cpu

KINDS=('targeted','random_basis','random_patches','random_both')


def norm(x):return x.double().flatten(1).norm(dim=-1)


def fit_geometry(delta,seed,rank=4,count=64):
    if delta.ndim!=3 or delta.shape[1:]!=(256,384):raise ValueError('True visual384/native256 patches required')
    if rank!=4 or count!=64:raise ValueError('Frozen rank and patch count')
    x=delta[1:].double().reshape(-1,384);mean=x.mean(0);x=x-mean
    eig,u=torch.linalg.eigh(x.T@x/max(len(x)-1,1));u=u[:,-rank:].contiguous()
    if float(eig[-rank])<=float(eig[-1])*1e-10:raise ValueError('Fewer than four response covariance directions')
    projected=delta.double()@u@u.T
    salience=projected[1:].square().sum(-1).mean(0)
    selected=salience.argsort(descending=True,stable=True)[:count]
    gen=torch.Generator(device='cpu').manual_seed(seed)
    random_u=torch.linalg.qr(torch.randn(384,rank,generator=gen,dtype=torch.float64),mode='reduced').Q.to(delta.device)
    random_selected=torch.randperm(256,generator=gen)[:count].to(delta.device)
    mask=torch.zeros(256,device=delta.device,dtype=torch.bool);mask[selected]=True
    random_mask=torch.zeros_like(mask);random_mask[random_selected]=True
    variants={};bases={};masks={}
    for kind in KINDS:
        basis=random_u if kind in ('random_basis','random_both') else u
        chosen=random_mask if kind in ('random_patches','random_both') else mask
        variants[kind]=((delta.double()@basis@basis.T)*chosen[None,:,None]).float()
        bases[kind]=basis;masks[kind]=chosen
    energy=float(delta[1:].double().square().sum())
    stats=dict(rank=rank,patch_count=count,seed=seed,eigenvalues=eig.cpu().tolist(),
        retained_centered_variance=float(eig[-rank:].sum()/eig.clamp_min(0).sum()),
        projected_contrast_energy_fraction=float(projected[1:].square().sum())/energy,
        selected_projected_contrast_energy_fraction=float(variants['targeted'][1:].double().square().sum())/energy,
        selected_patches=selected.cpu().tolist(),random_patches=random_selected.cpu().tolist(),patch_salience=salience.cpu().tolist(),
        random_basis_overlap=float((u.T@random_u).square().sum()/rank),
        basis_mean=mean.cpu(),fit_excludes_outcomes=True,fit_observations='63 dependent action-history contrasts times256patches; not independent episodes')
    return variants,bases,masks,stats


def match_delivered(base,direction,target_norm,steps=25):
    """Match actual float32-addition norm, not just ideal requested norm."""
    scale=target_norm/norm(direction).clamp_min(1e-30)
    if ((target_norm>0)&(norm(direction)==0)).any():raise ValueError('Nonzero target with zero control direction')
    low=torch.zeros_like(scale);high=2*scale
    best=scale.clone();best_error=torch.full_like(scale,float('inf'))
    for _ in range(steps):
        mid=(low+high)/2
        value=base+(direction*mid[:,None,None]).float()
        delivered=norm(value-base);error=(delivered-target_norm).abs()
        improve=error<best_error;best=torch.where(improve,mid,best);best_error=torch.minimum(best_error,error)
        low=torch.where(delivered<target_norm,mid,low);high=torch.where(delivered>=target_norm,mid,high)
    requested=(direction*best[:,None,None]).float();edited=base+requested
    actual_norm=norm(edited-base);torch.testing.assert_close(actual_norm,target_norm,rtol=2e-6,atol=2e-6)
    return edited,requested,best


def prepare_edits(base,variants,bases,masks):
    edits={};stats={}
    for sign in (-1,1):
        target_req=sign*.1*variants['targeted'];target_edit=base+target_req;target_norm=norm(target_edit-base)
        for kind in KINDS:
            name=kind+('_minus' if sign<0 else '_plus')
            if kind=='targeted':edited,requested,scale=target_edit,target_req,torch.ones(len(base),device=base.device,dtype=torch.float64)*.1
            else:edited,requested,scale=match_delivered(base,sign*variants[kind],target_norm)
            delta=edited-base;basis=bases[kind];off=delta.double()-delta.double()@basis@basis.T
            roundoff=delta.double()-requested.double()
            ideal_off=requested.double()-requested.double()@basis@basis.T
            # Only arithmetic rounding may create a component outside the selected channel subspace.
            if bool((norm(off)>norm(roundoff)+norm(ideal_off)+1e-7).any()):raise ValueError('Unexpected off-subspace edit')
            exact(edited[:,~masks[kind]],base[:,~masks[kind]],'unselected patches preserved')
            exact(edited[0],base[0],'central exact null')
            edits[name]=edited
            stats[name]=dict(delivered_l2=norm(delta).cpu().tolist(),matched_target_l2=target_norm.cpu().tolist(),
                delivered_norm_mismatch=(norm(delta)-target_norm).abs().cpu().tolist(),requested_l2=norm(requested).cpu().tolist(),
                scale=scale.cpu().tolist(),off_subspace_l2=norm(off).cpu().tolist(),rounding_l2=norm(roundoff).cpu().tolist(),
                ideal_projection_rounding_l2=norm(ideal_off).cpu().tolist(),raw_relative_l2=(norm(delta)/norm(base).clamp_min(1e-30)).cpu().tolist())
    return edits,stats


class InputPatch:
    def __init__(self,predictor,reference=None,edited=None):self.predictor=predictor;self.reference=reference;self.edited=edited;self.calls=0;self.block_calls=0;self.record={}
    def __enter__(self):
        self.handles=[self.predictor.register_forward_pre_hook(self.hook),self.predictor.predictor_blocks[3].register_forward_hook(self.block)]
        return self
    def hook(self,module,args):
        self.calls+=1
        if self.calls!=3:return None
        v,a,p=args
        if v.shape[1:]!=(2,1,16,16,384):raise ValueError('Pinned H3 visual input shape')
        if self.reference is None:self.record['inputs']=tuple(x.detach().clone() for x in args);return None
        for x,y in zip(args,self.reference):exact(x,y,'same pre-edit visual/action/proprio inputs')
        if self.edited is None:return None
        changed=scratch(v);changed[:,-1].copy_(self.edited.reshape_as(v[:,-1]))
        exact(changed[:,:-1],v[:,:-1],'past visual frame unchanged');exact(changed[0],v[0],'center input exact')
        return changed,a,p
    def block(self,module,args,out):
        self.block_calls+=1
        if self.block_calls==3:self.record['p3_output']=out[:,-256:].detach().cpu().clone()
    def __exit__(self,*exc):
        for h in self.handles:h.remove()
        if exc[0] is None and (self.calls,self.block_calls)!=(6,6):raise ValueError('Six native calls required')


@torch.no_grad()
def run_state(args,row,wm,prep):
    from tensordict import TensorDict
    from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
    from diagnose_pusht_goal_coverage import coverage,native_helper
    started=time.monotonic();path=Path(row['bank']);receipt=json.loads(Path(row['receipt']).read_text());e=row['episode']
    entry=next(r for r in receipt['outputs'] if r['path']==path.name)
    if not receipt['complete'] or not receipt['original9_actions_goal_physics_pixels_exact'] or entry['source_id']!=e or entry['split']!='development_external':raise ValueError('DEV source receipt before load')
    if sha(path)!=entry['sha256']:raise ValueError('Bank SHA before load')
    bank=torch.load(path,map_location='cpu',weights_only=False)
    if len(bank['candidates'])!=64 or bank['row']['source_id']!=e:raise ValueError('Fixed64 source')
    z=TensorDict({k:v.repeat_interleave(64,0) for k,v in bank['context'].items()},batch_size=[]).cuda();z_before=z.clone()
    normalized=prep.normalize_actions(bank['raw_actions'].reshape(64,6,5,2)).reshape(64,6,10)
    exact(normalized,bank['normalized_actions'],'official normalized actions');actions=normalized.transpose(0,1).contiguous().cuda();saved_actions=actions.clone()
    goal=TensorDict(bank['goal_encoded'],batch_size=[]).cuda();objective=ReprTargetDistMPCObjective({},goal,sum_all_diffs=False,alpha=.1)
    native=wm.unroll(z.clone(),act_suffix=actions)
    with InputPatch(wm.model.predictor) as cap:repeat=wm.unroll(z.clone(),act_suffix=actions)
    for k in ('visual','proprio'):exact(native[k],repeat[k],'read-only capture identity')
    reference=cap.record['inputs'];base=reference[0][:,-1].reshape(64,256,384);contrast=base-base[0:1]
    variants,bases,masks,geometry=fit_geometry(contrast,2026090649+e)
    edits,edit_stats=prepare_edits(base,variants,bases,masks)
    geometry_cpu=cpu(geometry);basis_file=args.output/f'episode-{e:03d}-FROZEN_GEOMETRY.pt'
    torch.save(dict(geometry=geometry_cpu,bases=cpu(bases),masks=cpu(masks),contrast=cpu(contrast),source_sha256=entry['sha256'],outcome_fields_accessed_by_fit=False),basis_file)
    with InputPatch(wm.model.predictor,reference,base) as zero:identity=wm.unroll(z.clone(),act_suffix=actions)
    for k in ('visual','proprio'):exact(native[k],identity[k],'zero edit identity')
    predictions={'native':cpu({k:native[k][1:] for k in ('visual','proprio')})};costs={'native':objective(native,actions,keepdims=True)[1:].cpu()};p3={'native':cap.record['p3_output']}
    for name,edited in edits.items():
        with InputPatch(wm.model.predictor,reference,edited) as hook:out=wm.unroll(z.clone(),act_suffix=actions)
        for k in ('visual','proprio'):
            exact(out[k][:3],native[k][:3],'preH3 forecast exact');exact(out[k][:,0],native[k][:,0],'central trajectory exact')
        predictions[name]=cpu({k:out[k][1:] for k in ('visual','proprio')});costs[name]=objective(out,actions,keepdims=True)[1:].cpu();p3[name]=hook.record['p3_output']
    torch.cuda.synchronize();model_seconds=time.monotonic()-started
    write(args.output/f'episode-{e:03d}-MODEL_COMPLETE.json',dict(complete=True,seconds=model_seconds,all_forward_guards_exact=True,geometry_sha256=sha(basis_file)))
    print(json.dumps(dict(event='model_complete',episode=e,seconds=model_seconds)),flush=True)
    # Actual future fields are consulted only after basis, patches, dose and all edited forwards are frozen.
    states=np.stack([c['states'] for c in bank['candidates']]);actual={'visual':torch.stack([c['actual_visual'] for c in bank['candidates']],1).float()}
    prop=torch.tensor(states[:,::5][:,:,[0,1,5,6]],dtype=torch.float32)
    actual['proprio']=wm.model.encode_proprio(prep.normalize_proprios(prop).cuda()).cpu()[:,1:].transpose(0,1).contiguous()
    future=states[:,5::5];gs=np.asarray(bank['goal_state']);covered=np.array([[coverage(s[2:5],gs[2:5]) for s in c] for c in future]).T
    polygon=native_helper(args.repo/'evals/simu_env_planning/envs/pusht_env/pusht_env.py');target=polygon(gs[2:5])
    checked=np.array([polygon(s[2:5]).intersection(target).area/target.area for s in future[:,-1]])
    if np.max(np.abs(checked-covered[-1]))>1e-12:raise ValueError('Requested-goal native polygon parity')
    xy=np.linalg.norm(future[:,:,:4]-gs[:4],axis=-1).T;basechoice=int(costs['native'][-1].argmin());rows=[]
    for name,pred in predictions.items():
        errors={k:(pred[k].double()-actual[k].double()).square().flatten(2).mean(-1) for k in actual};choice=int(costs[name][-1].argmin())
        rows.append(dict(arm=name,choice=choice,cost_by_horizon=costs[name].tolist(),rank_by_horizon=costs[name].argsort(dim=1,stable=True).tolist(),
            mse_by_horizon_candidate={k:v.tolist() for k,v in errors.items()},equal_candidate_mse_by_horizon={k:v.mean(1).tolist() for k,v in errors.items()},
            selected_goal_coverage=float(covered[-1,choice]),goal_coverage_delta=float(covered[-1,choice]-covered[-1,basechoice]),
            selected_xy_distance_px=float(xy[-1,choice]),xy_delta_px=float(xy[-1,choice]-xy[-1,basechoice]),edit=edit_stats.get(name)))
    # Support is only distance to these same-state natural input fields, not manifold membership.
    flat=base.double().flatten(1);distance=torch.cdist(flat,flat,compute_mode='donot_use_mm_for_euclid_dist');distance.fill_diagonal_(float('inf'));support={'native_nearest_other':distance.min(1).values.cpu().tolist()}
    for name,edited in edits.items():
        distance=torch.cdist(edited.double().flatten(1),flat,compute_mode='donot_use_mm_for_euclid_dist');distance.fill_diagonal_(float('inf'))
        support[name]=distance.min(1).values.cpu().tolist()
    exact(actions,saved_actions,'external actions unchanged')
    for k in z.keys():exact(z[k],z_before[k],'original context unchanged')
    output=args.output/f'episode-{e:03d}.pt'
    torch.save(dict(complete=True,episode=e,source_sha256=entry['sha256'],raw_actions=bank['raw_actions'],normalized_actions=normalized,
        initial_context=bank['context'],goal_encoded=bank['goal_encoded'],goal_state=gs,metadata=bank['metadata'],base_visual=cpu(base),edited_visual=cpu(edits),
        predictions=predictions,costs=costs,p3=p3,actual=actual,states=states,score=rows,geometry=geometry_cpu,edit_stats=edit_stats,support=support),output)
    stats={k:v for k,v in geometry_cpu.items() if k!='basis_mean'}
    report=dict(complete=True,episode=e,seconds=time.monotonic()-started,model_stage_seconds=model_seconds,score=rows,geometry=stats,support=support,
        all_identity_guards_exact=True,original9_source_guard_exact=True,full_tensor_sha256=sha(output),full_tensor_bytes=output.stat().st_size,
        geometry_sha256=sha(basis_file),source_sha256=entry['sha256'],goal_coverage_headroom=float(covered[-1].max()-covered[-1,basechoice]),independent_states=1)
    write(output.with_suffix('.json'),report)
    print(json.dumps(dict(event='state_done',episode=e,seconds=time.monotonic()-started,choices={r['arm']:r['choice'] for r in rows})),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('inputs','output','repo','checkpoint'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--max-seconds',type=float,default=300);a=p.parse_args();started=time.monotonic();rows=json.loads(a.inputs.read_text())
    if len(rows)!=2 or any(r['episode'] not in range(4) for r in rows) or len({r['episode'] for r in rows})!=2:raise ValueError('Frozen two-state shard')
    if sha(a.checkpoint)!=CHECKPOINT:raise ValueError('Frozen checkpoint mismatch')
    a.output.mkdir(parents=True,exist_ok=False)
    try:
        write(a.output/'protocol.json',dict(inputs=rows,site='true384-channel newest predictor visual INPUT H3, before400mixing; P3 downstream trace observed',rank=4,patches=64,doses=[-.1,.1],
            arms=KINDS,random_seed_rule='2026090649+episode',basis='Centered channel PCA of63 natural action-history contrasts times256patches; no labels/outcomes',
            controls='random orthonormalrank4 and/or fixed random64patches, per-candidate float32 DELIVERED norm matched2e-6 rtol/atol',
            conservation='unchanged actions,proprio,pastframes,unselectedpatches; off-subspace change bounded by measured arithmetic rounding',
            practical_lead='Exploratory only: mean selected goalcoverage gain>=.01,positive>=3/4states,and mean exceeding every corresponding matchedcontrol; all4fixedconfig required',
            limitations='n4seenstates,64dependentplans; action-HISTORY/consequence response notisolatedcurrentaction; rank4 localization notmanifold orDAS learnedcausalalignment; finitebank notfullCEM',
            primary_sources=['https://arxiv.org/html/2308.10248v5#S3','https://arxiv.org/html/2308.10248v5#A7','https://proceedings.mlr.press/v236/geiger24a/geiger24a.pdf'],
            checkpoint_sha256=CHECKPOINT,script_sha256=sha(__file__),max_seconds=a.max_seconds,held_access=False,simulator_calls=0,cem_calls=0))
        from model_loader import load_headless
        torch.set_num_threads(2);torch.manual_seed(90505);wm,prep,provenance=load_headless(a.repo,model_name='jepa_wm_pusht',checkpoint_override=a.checkpoint);wm.eval().requires_grad_(False)
        def params():
            h=hashlib.sha256()
            for name,v in wm.state_dict().items():h.update(name.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
            return h.hexdigest()
        before=params();reports=[]
        for i,row in enumerate(rows):
            if time.monotonic()-started>a.max_seconds:raise ValueError('Shard budget before next state')
            report=run_state(a,row,wm,prep);reports.append(report)
            if i==0:
                estimate=time.monotonic()-started+report['seconds'];write(a.output/'CANARY.json',dict(complete=True,estimate_seconds=estimate,max_seconds=a.max_seconds))
                if estimate>a.max_seconds:raise ValueError('Firststate canary exceeds shard cap')
        if params()!=before:raise ValueError('Frozen parameter hash changed')
        write(a.output/'summary.json',dict(complete=True,independent_states=2,reports=[f"episode-{r['episode']:03d}.json" for r in reports],seconds=time.monotonic()-started))
        outputs=[dict(path=q.name,bytes=q.stat().st_size,sha256=sha(q)) for q in sorted(a.output.iterdir()) if q.is_file()]
        write(a.output/'DONE.json',dict(complete=True,seconds=time.monotonic()-started,outputs=outputs,parameter_sha256=before,provenance=provenance,held_access=False))
    except Exception:
        write(a.output/'FAILED.json',dict(complete=False,exception=traceback.format_exc()));raise


if __name__=='__main__':main()
