#!/usr/bin/env python3
"""Bounded local rank4 P5 edits with native nonlinear suffix-dose calibration."""
import argparse
import json
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
import torch
try:
    from .run_acpc_image_nuisance import CHECKPOINT,DINO_SHA,sha,write,exact,parameter_sha
    from .run_patch_policy_action_spatial import CurrentConditionResponse,cpu
except ImportError:
    from run_acpc_image_nuisance import CHECKPOINT,DINO_SHA,sha,write,exact,parameter_sha
    from run_patch_policy_action_spatial import CurrentConditionResponse,cpu

RANK=4
RAW_DOSE=.1
TARGET_FRACTION=.005
RAW_RADIUS=.02


def l2(x):return x.double().flatten(1).norm(dim=1)


def fit_action_subspaces(delta,seed):
    """Uncentered channel contrast span; same coefficients in orthogonal random span."""
    x=delta.double().reshape(-1,delta.shape[-1]);eig,v=torch.linalg.eigh(x.T@x)
    basis=v[:,-RANK:];coeff=delta.double()@basis;projected=coeff@basis.T
    g=torch.Generator(device='cpu').manual_seed(seed)
    random=torch.randn(delta.shape[-1],RANK,generator=g,dtype=torch.float64).to(delta.device)
    random=random-basis@(basis.T@random);random=torch.linalg.qr(random,mode='reduced').Q
    sham=coeff@random.T
    stats=dict(eigenvalues_descending=eig.flip(0).cpu().tolist(),retained_contrast_energy=float(eig[-RANK:].sum()/eig.sum().clamp_min(1e-30)),
        random_vs_action_subspace_maxabs=float((basis.T@random).abs().max()),
        action_orthonormal_maxabs=float((basis.T@basis-torch.eye(RANK,device=basis.device)).abs().max()),
        random_orthonormal_maxabs=float((random.T@random-torch.eye(RANK,device=basis.device)).abs().max()),
        paired_norm_maxabs=float((l2(projected)-l2(sham)).abs().max()))
    return basis,random,projected.to(delta),sham.to(delta),stats


def weighted_rms(visual,proprio):
    return (visual.double().flatten(1).square().mean(1)+.1*proprio.double().flatten(1).square().mean(1)).sqrt()


def first_crossing_match(response,target,max_radius,steps=16,bisections=24):
    """Label-free first bracket; do not assume globally monotone native response."""
    grid=torch.linspace(0,1,steps+1,device=target.device,dtype=torch.float64)
    samples=torch.stack([response(max_radius*t) for t in grid])
    crossing=(samples[:-1]<target)&(samples[1:]>=target)
    valid=(target>0)&crossing.any(0);first=crossing.to(torch.int64).argmax(0)
    lo=grid[first]*max_radius;hi=grid[first+1]*max_radius
    for _ in range(bisections):
        mid=(lo+hi)/2;value=response(mid);low=value<target
        lo=torch.where(low,mid,lo);hi=torch.where(low,hi,mid)
    amplitude=(lo+hi)/2;achieved=response(amplitude)
    tolerance=torch.maximum(target*.002,torch.full_like(target,1e-6))
    valid=valid&((achieved-target).abs()<=tolerance)
    amplitude=torch.where(valid,amplitude,torch.zeros_like(amplitude))
    return amplitude,valid,dict(scan_fractions=grid.cpu().tolist(),scan_response_RMS=samples.cpu().tolist(),
        upward_crossings=crossing.sum(0).cpu().tolist(),downward_crossings=((samples[:-1]>=target)&(samples[1:]<target)).sum(0).cpu().tolist(),
        first_bracket_low=lo.cpu().tolist(),first_bracket_high=hi.cpu().tolist(),target=target.cpu().tolist(),
        achieved_at_final_bisection=achieved.cpu().tolist(),feasible=valid.cpu().tolist(),relative_tolerance=.002,absolute_tolerance=1e-6)


def suffix(predictor,full):
    b,tokens,_=full.shape;t=tokens//256
    norm=predictor.predictor_norm(full).view(b,t,256,400)
    visual=predictor.predictor_proj(norm[...,:384])
    return visual[:,-1],norm[:,-1,:,384:]


class SubspacePatch:
    def __init__(self,block,reference,delta):self.block,self.reference,self.delta=block,reference,delta;self.calls=0;self.delivered=None
    def __enter__(self):self.handle=self.block.register_forward_hook(self.hook,with_kwargs=True);return self
    def hook(self,module,args,kwargs,output):
        self.calls+=1
        if self.calls!=3:return None
        exact(args[0],self.reference['x'],'Same native x before edit');exact(args[1],self.reference['z'],'Same action condition before edit')
        exact(output,self.reference['original'],'Same native P5 output')
        changed=output.clone();changed[:,-256:]+=self.delta
        exact(changed[:,:-256],output[:,:-256],'History tokens unchanged');exact(changed[0],output[0],'Central null exact')
        self.delivered=changed[:,-256:]-output[:,-256:]
        return changed
    def __exit__(self,*exc):
        self.handle.remove()
        if exc[0] is None and self.calls!=6:raise ValueError('Six native prediction horizons required')


def actual_error(prediction,actual):
    return torch.stack([(x.double()-y.double()).square().flatten(1).mean(1) for x,y in zip(prediction,actual)])


@torch.no_grad()
def one_state(args,root,row,wm,prep):
    from tensordict import TensorDict
    from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
    from diagnose_pusht_goal_coverage import coverage,native_helper
    started=time.monotonic();path=root/row['path']
    if sha(path)!=row['sha256']:raise ValueError('Frozen bank SHA before load')
    bank=torch.load(path,map_location='cpu',weights_only=False);episode=bank['row']['source_id']
    if bank['row']['split']!='development_external' or len(bank['candidates'])!=64 or not bank['original9_actions_goal_physics_pixels_exact']:raise ValueError('Existing64 DEV only')
    normalized=prep.normalize_actions(bank['raw_actions'].reshape(64,6,5,2)).reshape(64,6,10)
    exact(normalized,bank['normalized_actions'],'Native normalization unchanged')
    actions=normalized.transpose(0,1).contiguous().cuda();frozen_actions=actions.clone()
    context=TensorDict({k:v.repeat_interleave(64,dim=0).cuda() for k,v in bank['context'].items()},batch_size=[]);frozen_context=context.clone()
    goal=TensorDict(bank['goal_encoded'],batch_size=[]).cuda();frozen_goal=goal.clone()
    objective=ReprTargetDistMPCObjective({},goal,sum_all_diffs=False,alpha=.1)
    def unroll():return wm.unroll(context.clone(),act_suffix=actions)
    native=unroll();predictor=wm.model.predictor;block=predictor.predictor_blocks[5]
    with CurrentConditionResponse(block,horizon=3) as capture:repeat=unroll()
    for k in ('visual','proprio'):exact(native[k],repeat[k],'Read-only capture identity '+k)
    record=capture.record;original=record['original'];x=original[:,-256:];delta=record['delta'];base_v,base_p=suffix(predictor,original)
    expected_v=native['visual'][3].reshape(64,256,384);expected_p=native['proprio'][3].reshape(64,256,16)
    parity=dict(visual_maxabs=float((base_v-expected_v).abs().max()),proprio_maxabs=float((base_p-expected_p).abs().max()))
    # Same GPU/precision/layout suffix is independently measured, not assumed exact.
    if max(parity.values())>1e-5:raise ValueError('Native suffix reconstruction exceeds predeclared1e-5 numerical floor')
    b,q,semantic,random,basis_stats=fit_action_subspaces(delta,2026090641+episode)
    exact(semantic[0],torch.zeros_like(semantic[0]),'Central rank4 null');exact(random[0],torch.zeros_like(random[0]),'Central sham null')
    with SubspacePatch(block,record,torch.zeros_like(delta)):zero=unroll()
    for k in ('visual','proprio'):exact(native[k],zero[k],'Exact zero patch '+k)
    del zero,repeat
    directions={'semantic':semantic,'random':random};unit={k:v/l2(v).clamp_min(1e-30).to(v)[:,None,None] for k,v in directions.items()}
    target=TARGET_FRACTION*weighted_rms(base_v,base_p);target[0]=0
    max_radius=RAW_RADIUS*l2(x);max_radius[0]=0
    calibration={};amplitudes={};valid={}
    for kind in directions:
        for sign in (-1,1):
            key=kind+('_minus' if sign<0 else '_plus')
            def response(amplitude):
                changed=original.clone();changed[:,-256:]+=unit[kind]*(sign*amplitude).to(x)[:,None,None]
                v,p=suffix(predictor,changed)
                return weighted_rms(v-base_v,p-base_p)
            amplitudes[key],valid[key],calibration[key]=first_crossing_match(response,target,max_radius)
    common=torch.stack(list(valid.values())).all(0);common[0]=False
    frozen_edits={};edit_basis={}
    for sign in (-1,1):
        tail='_minus' if sign<0 else '_plus'
        for kind in directions:
            name='raw_'+kind+tail;frozen_edits[name]=sign*RAW_DOSE*directions[kind];edit_basis[name]=b if kind=='semantic' else q
            name='calibrated_'+kind+tail
            amp=torch.where(common,amplitudes[kind+tail],torch.zeros_like(target))
            frozen_edits[name]=sign*unit[kind]*amp.to(x)[:,None,None];edit_basis[name]=b if kind=='semantic' else q
        name='calibrated_rawnorm_random'+tail
        amp=torch.where(common,amplitudes['semantic'+tail],torch.zeros_like(target))
        frozen_edits[name]=sign*unit['random']*amp.to(x)[:,None,None];edit_basis[name]=q
    print(json.dumps(dict(event='normalization_calibrated',episode=episode,seconds=time.monotonic()-started,common_feasible=int(common.sum()),retained_rank4_energy=basis_stats['retained_contrast_energy'])),flush=True)
    # All edits frozen before accessing actual futures or selecting on physical labels.
    actual={'visual':torch.stack([c['actual_visual'] for c in bank['candidates']],1).float()}
    states=np.stack([c['states'] for c in bank['candidates']]);gs=np.asarray(bank['goal_state']);future=states[:,5::5]
    props=torch.tensor(states[:,::5][:,:,[0,1,5,6]],dtype=torch.float32)
    actual['proprio']=wm.model.encode_proprio(prep.normalize_proprios(props).cuda()).cpu()[:,1:].transpose(0,1).contiguous()
    cover=np.array([[coverage(s[2:5],gs[2:5]) for s in plan] for plan in future]).T
    xy=np.linalg.norm(future[:,:,:4]-gs[:4],axis=-1).T
    poly=native_helper(args.repo/'evals/simu_env_planning/envs/pusht_env/pusht_env.py');targetpoly=poly(gs[2:5])
    if np.max(np.abs([poly(s[2:5]).intersection(targetpoly).area/targetpoly.area for s in future[:,-1]]-cover[-1]))>1e-12:raise ValueError('Native requested-goal polygon parity')
    native_cost=objective(native,actions,keepdims=True)[1:].cpu();choice0=int(native_cost[-1].argmin())
    old_cost=torch.cat([c['native_cost_by_horizon'] for c in bank['candidates']],1)[1:]
    if int(old_cost[-1].argmin())!=choice0:raise ValueError('Native64 choice differs before editing')
    rows=[];forecasts={};norms={};all_costs={}
    for name in ['native']+list(frozen_edits):
        if name=='native':out=native
        else:
            with SubspacePatch(block,record,frozen_edits[name]) as patch:out=unroll()
            for k in ('visual','proprio'):
                exact(out[k][:3],native[k][:3],'Pre-edit forecast prefix '+name)
                exact(out[k][:,0],native[k][:,0],'Central forecast '+name)
            delivered=patch.delivered;basis=edit_basis[name];off=delivered.double()-(delivered.double()@basis)@basis.T
            direct_v=out['visual'][3].reshape(64,256,384)-base_v;direct_p=out['proprio'][3].reshape(64,256,16)-base_p
            wanted=frozen_edits[name];requested_off=wanted.double()-(wanted.double()@basis)@basis.T
            norms[name]=dict(requested_raw_l2=l2(wanted).cpu().tolist(),actual_delivered_raw_l2=l2(delivered).cpu().tolist(),
                raw_fraction_native=l2(delivered).div(l2(x)).cpu().tolist(),rounding_l2=l2(delivered-wanted).cpu().tolist(),
                requested_off_subspace_l2=l2(requested_off).cpu().tolist(),actual_off_subspace_rounding_l2=l2(off).cpu().tolist(),
                delivered_suffix_weighted_RMS=weighted_rms(direct_v,direct_p).cpu().tolist(),
                visual_output_RMS=direct_v.double().square().flatten(1).mean(1).sqrt().cpu().tolist(),
                proprio_output_RMS=direct_p.double().square().flatten(1).mean(1).sqrt().cpu().tolist())
            if name.startswith('calibrated') and bool((l2(delivered)>max_radius+1e-2).any()):raise ValueError('Raw radius cap violated')
        prediction=cpu({k:out[k][1:] for k in ('visual','proprio')});cost=objective(out,actions,keepdims=True)[1:].cpu();chosen=int(cost[-1].argmin())
        error={k:actual_error(prediction[k],actual[k]) for k in actual}
        rows.append(dict(arm=name,choice=chosen,cost_by_horizon=cost.tolist(),rank_by_horizon=cost.argsort(dim=1,stable=True).tolist(),
            actual_MSE_by_horizon_candidate={k:v.tolist() for k,v in error.items()},actual_mean_MSE_by_horizon={k:v.mean(1).tolist() for k,v in error.items()},
            selected_requested_goal_coverage=float(cover[-1,chosen]),selected_goal_coverage_delta_vs_native=float(cover[-1,chosen]-cover[-1,choice0]),
            selected_combined_XY_distance_px=float(xy[-1,chosen]),selected_XY_delta_vs_native_px=float(xy[-1,chosen]-xy[-1,choice0])))
        # Final-arm H3/H6 tensors only; calibration iterates are NOT saved.
        forecasts[name]={k:v[[2,5]].clone() for k,v in prediction.items()};all_costs[name]=cost
        del prediction
    exact(actions,frozen_actions,'External action plans preserved')
    for k in ('visual','proprio'):exact(context[k],frozen_context[k],'Initial state preserved');exact(goal[k],frozen_goal[k],'Fullprecision goal preserved')
    report=dict(complete=True,episode=episode,seconds=time.monotonic()-started,source_sha256=row['sha256'],rows=rows,basis_stats=basis_stats,
        calibration=calibration,common_feasible=common.cpu().tolist(),common_feasible_count=int(common.sum()),suffix_parity=parity,edit_norms=norms,
        goal_coverage_by_horizon_candidate=cover.tolist(),combined_XY_by_horizon_candidate=xy.tolist(),native_zero_prefix_center_exact=True,
        old_native_cost_maxabs=float((native_cost-old_cost).abs().max()),fit_scope='Unlabelled per-state model-only action contrasts; no physical labels; extra runtime local adaptation, not free inference')
    path=args.output/f'episode-{episode:03d}.pt'
    torch.save(dict(report=report,source_root=str(root),source_row=row,original_p5=cpu(original),raw_action_delta=cpu(delta),
        action_basis=cpu(b),random_basis=cpu(q),frozen_edits=cpu(frozen_edits),final_arm_H3_H6_forecasts=forecasts,costs=all_costs,
        full_native=cpu({k:native[k][1:] for k in ('visual','proprio')}),raw_actions=bank['raw_actions'],normalized_actions=normalized,states=states,goal_encoded=bank['goal_encoded']),path)
    report['seconds']=time.monotonic()-started;write(path.with_suffix('.json'),report)
    print(json.dumps(dict(event='normalization_state_DONE',episode=episode,seconds=report['seconds'],bytes=path.stat().st_size,choices={r['arm']:r['choice'] for r in rows})),flush=True)
    return report,[dict(path=f.name,sha256=sha(f),bytes=f.stat().st_size) for f in (path,path.with_suffix('.json'))]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('repo','checkpoint','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--banks',type=Path,nargs=4,required=True);p.add_argument('--max-seconds',type=float,default=600.)
    args=p.parse_args();start=time.monotonic();args.output.mkdir(parents=True,exist_ok=False)
    try:
        indexed=[]
        for episode,root in enumerate(args.banks):
            done=json.loads((root/'DONE.json').read_text())
            if not done['complete'] or done.get('held_access',True):raise ValueError('Completed development receipt')
            rows=[r for r in done['outputs'] if r['path']==f'expanded-dev-{episode:03d}.pt']
            if len(rows)!=1 or rows[0]['split']!='development_external':raise ValueError('Fixed four states/split before load')
            indexed.append((root,rows[0]))
        if sha(args.checkpoint)!=CHECKPOINT:raise ValueError('Frozen checkpoint SHA')
        if sha(Path('/root/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth'))!=DINO_SHA:raise ValueError('Frozen external visual encoder')
        write(args.output/'protocol.json',dict(exploratory=True,states=list(range(4)),candidates=64,rank=4,block=5,horizon=3,
            site='Newest256 P5 residual400 channels; preserve all history and orthogonal component except measured FP32 rounding',
            basis='Uncentered channel PCA top4 of same-x model current-action contrasts; local model-only perstate adaptation',
            arms='native + signs± for rawsemantic/rawrandom/calibratedsemantic/calibrated_outputrandom/calibrated_rawnormrandom =11',
            raw_dose=RAW_DOSE,output_target_fraction_native_suffix_signal=TARGET_FRACTION,weighted_RMS_squared='mean visual384 squared +.1 mean proprio16 squared',
            raw_radius_fraction_native_P5=RAW_RADIUS,calibration='16 fixed uniform fractions first upward bracket +24 bisections; target tolerance max1e-6,.002target; all crossing counts exposed',
            feasibility='All four semantic/random/sign calibrations intersection per candidate. Nonfeasible candidates retain native in ALL calibrated arms; entire64 pool retained',
            controls='Raw matches inputnorm andrank; calibratedrandom matches outputnorm andrank; extra calibratedrawnormrandom matches semantic inputnorm; neither falsely claimed to match both',
            primary='Selected cached requested-goal coverage vsnative and correspondingrandom, fourstate units',
            continuation_rule='Mean coveragegain>=.01, positive>=3/4states, exceeds matchedcontrolsmean; exploratory allocation only, not significance',
            suffix_floor_maxabs=1e-5,seed_formula='2026090641+state',code_sha256=sha(__file__),checkpoint_sha256=CHECKPOINT,
            inputs=[dict(root=str(root),row=row) for root,row in indexed],max_process_seconds=args.max_seconds,held_access=False,model_training_steps=0,cem_calls=0,simulator_calls=0,
            limitations=['Rank4 support is not a discovered manifold','Outputnorm calibration is not optimizing goals or task labels','Delivered effects can have different directions despite equal norms','Four states reused extensively, no confirmation claim']))
        from model_loader import load_headless
        torch.set_num_threads(2);torch.manual_seed(90505)
        wm,prep,provenance=load_headless(args.repo,model_name='jepa_wm_pusht',checkpoint_override=args.checkpoint);wm.eval().requires_grad_(False)
        before=parameter_sha(wm);init=time.monotonic()-start;reports=[];outputs=[]
        for i,(root,row) in enumerate(indexed):
            if shutil.disk_usage(args.output).free<2_300_000_000:raise ValueError('At least1GB reserve plus final state output required')
            if time.monotonic()-start>args.max_seconds-60:raise ValueError('Do not start state near frozen cap')
            report,files=one_state(args,root,row,wm,prep);reports.append(report);outputs.extend(files)
            if i==0:
                estimate=init+report['seconds']*4
                write(args.output/'CANARY.json',dict(complete=True,first_state_seconds=report['seconds'],estimated_seconds=estimate,max_seconds=args.max_seconds,common_feasible=report['common_feasible_count'],suffix_parity=report['suffix_parity']))
                if estimate>args.max_seconds:raise ValueError('Firststate estimate exceeds cap')
        if parameter_sha(wm)!=before:raise ValueError('Frozen parameter hash changed')
        aggregate=[]
        for index,item in enumerate(reports[0]['rows']):
            values=[r['rows'][index] for r in reports];gains=[r['selected_goal_coverage_delta_vs_native'] for r in values]
            aggregate.append(dict(arm=item['arm'],choices=[r['choice'] for r in values],coverage_deltas=gains,mean_coverage_delta=float(np.mean(gains)),positive_states=sum(g>0 for g in gains),
                XY_deltas_px=[r['selected_XY_delta_vs_native_px'] for r in values],visual_H6_MSE=[r['actual_mean_MSE_by_horizon']['visual'][-1] for r in values]))
        seconds=time.monotonic()-start
        if seconds>args.max_seconds:raise ValueError('Total process cap exceeded')
        write(args.output/'summary.json',dict(complete=True,states=4,seconds=seconds,aggregate=aggregate,common_feasible_counts=[r['common_feasible_count'] for r in reports],parameter_sha256=before,provenance=provenance))
        for f in (args.output/'protocol.json',args.output/'CANARY.json',args.output/'summary.json'):outputs.append(dict(path=f.name,sha256=sha(f),bytes=f.stat().st_size))
        write(args.output/'DONE.json',dict(complete=True,outputs=outputs,seconds=time.monotonic()-start,parameters_sha256=before,held_access=False,cem_calls=0,simulator_calls=0))
        print(json.dumps(dict(event='normalization_ALL_DONE',seconds=time.monotonic()-start)),flush=True)
    except Exception:
        write(args.output/'FAILED.json',dict(complete=False,seconds=time.monotonic()-start,exception=traceback.format_exc()));raise


if __name__=='__main__':main()
