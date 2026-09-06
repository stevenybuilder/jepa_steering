#!/usr/bin/env python3
"""Bounded residual Gaussian-energy contrast, never a returned-output edit."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from complete_cached_geometry import sha256,write_json
from capture_horizon_coordinates import exact,tensor_hash
from run_reach_native_coordinates import P3,rank_metrics,CHECKPOINT
from public_panel_sonar_math import fit_shrinkage_gaussian,outcome_energy_contrast_gradient

EPISODES=(0,4,7)
DOSES=(-1.,-.5,.5,1.)


def high_progress_labels(progress):
    x=np.asarray(progress,dtype=np.float64)
    labels=x>=np.median(x)
    if not np.isfinite(x).all() or labels.all() or not labels.any():raise ValueError('High/low local progress not identified')
    return labels


def fit_energy(features,progress_groups):
    x=np.asarray(features,dtype=np.float64);mean=x.mean(0)
    _,singular,basis=np.linalg.svd(x-mean,full_matrices=False)
    rank=min(3,int(np.sum(singular>max(singular[0]*1e-8,1e-10))))
    if rank<1:raise ValueError('No supported TRAIN residual coordinate')
    basis=basis[:rank];coord=(x-mean)@basis.T;scale=np.maximum(coord.std(0),1e-8);coord/=scale
    labels=np.concatenate([high_progress_labels(v) for v in progress_groups])
    groups={}
    for name,label in (('good',True),('bad',False)):
        m,c,ess=fit_shrinkage_gaussian(coord[labels==label],shrinkage=.2,floor=1e-4)
        groups[name]=dict(weights=np.ones(1),means=m[None],covariances=c[None],effective_n=ess)
    return dict(mean=mean,basis=basis,scale=scale,groups=groups,rank=rank,labels=labels,singular=singular)


def raw_energy_gradient(features,fit):
    coordinates=(np.asarray(features,dtype=np.float64)-fit['mean'])@fit['basis'].T/fit['scale']
    energy,gradient,_=outcome_energy_contrast_gradient(coordinates,fit['groups']['good'],fit['groups']['bad'])
    return energy,gradient/fit['scale']


def capped_fields(gradient,gram,radius,normalize_direction=False):
    """Base cap then signed dose at call site; trace-normalized local metric."""
    g=np.asarray(gradient,dtype=np.float64);G=np.asarray(gram,dtype=np.float64);r=np.asarray(radius,dtype=np.float64)
    eig=np.linalg.eigvalsh((G+G.transpose(0,2,1))/2);mean=eig.sum(1)/g.shape[1]
    damping=np.maximum(.01*mean,1e-12);metric=(G+damping[:,None,None]*np.eye(g.shape[1]))/(mean+damping)[:,None,None]
    raw={'static':-g,'local_metric':-np.linalg.solve(metric,g[...,None])[...,0]};out={};records={}
    for name,d in raw.items():
        length=np.linalg.norm(d,axis=1)
        factor=np.minimum(1,r/np.maximum(length,1e-30))
        if normalize_direction:
            factor=np.where(length>1e-12,r/np.maximum(length,1e-30),0.)
        out[name]=d*factor[:,None]
        records[name]=dict(unconstrained_norm=length.tolist(),actual_base_norm=np.linalg.norm(out[name],axis=1).tolist(),
            cap_saturated=(length>r).tolist(),cap_scale=factor.tolist(),
            direction_normalized=normalize_direction,zero_direction=(length<=1e-12).tolist())
    return out,dict(fields=records,metric_eigenvalues=eig.tolist(),metric_damping=damping.tolist(),radius=r.tolist())


def prepare(root,output):
    """One-thread CPU readback of verified completed development sources only."""
    torch.set_num_threads(1);output.mkdir(parents=True,exist_ok=False)
    receipt=json.loads((root/'results-v1/DONE.json').read_text());entries={r['path']:r for r in receipt['outputs']}
    if not receipt['complete'] or receipt['states']!=3:raise ValueError('Need completed three-state source')
    outputs=[]
    for ep in EPISODES:
        path=root/'results-v1'/f'episode-{ep:03}.pt';entry=entries[path.name]
        if sha256(path)!=entry['sha256']:raise ValueError('Coordinate source SHA before load')
        c=torch.load(path,map_location='cpu',weights_only=False);source=c['source']
        if source['episode']!=ep or source['split']!='development':raise ValueError('Wrong split')
        baseline=source['plans'][0];goal=source['goal_path']
        if sha256(baseline['path'])!=baseline['sha256'] or sha256(goal)!=source['goal_sha256']:raise ValueError('Original inputs SHA')
        b=torch.load(baseline['path'],map_location='cpu',weights_only=False)
        goal_data=torch.load(goal,map_location='cpu',weights_only=False)
        distance=np.asarray(c['report']['actual_H3_goal_distance_m']);initial=np.asarray(goal_data['arrays']['start_hand_xyz'])[0]
        taskgoal=np.asarray(goal_data['arrays']['task_goal_xyz'])[0];initial_distance=float(np.linalg.norm(initial-taskgoal))
        value=dict(episode=ep,split='development',context=b['encoded_context'],raw_goal={k:goal_data['goal'][k] for k in ('visual','proprio')},
            normalized_actions=c['selected_plans'],actual_H3_visual=c['actual_H3_visual'],endpoint_distance=distance,
            progress=initial_distance-distance,initial_hand=initial,task_goal=taskgoal,initial_distance=initial_distance,
            source=source,coordinate_source_sha256=entry['sha256'],actual_horizon=3,physical_raw_steps=15)
        target=output/f'episode-{ep:03}.pt';torch.save(value,target)
        outputs.append(dict(path=target.name,episode=ep,split='development',sha256=sha256(target),bytes=target.stat().st_size))
    write_json(output/'DONE.json',dict(complete=True,outputs=outputs,source_done_sha256=sha256(root/'results-v1/DONE.json'),
        no_model_or_simulator_calls=True,CUDA_invisible=True,script_sha256=sha256(__file__)))


@torch.no_grad()
def run(a):
    from model_loader import load_headless_metaworld
    from capture_specificity_controls import parameters_sha
    from causal_planner_forks import setup_cfg
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    from tensordict import TensorDict
    from intervene_head_spatial_mean import batch_context
    a.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((a.inputs/'DONE.json').read_text());rows=manifest['outputs']
    if not manifest['complete'] or [v['episode'] for v in rows]!=list(EPISODES) or any(v['split']!='development' for v in rows):
        raise ValueError('Only frozen development0/4/7 before tensor access')
    protocol=dict(sources=rows,script_sha256=sha256(__file__),checkpoint_sha256=CHECKPOINT,
        archived_math_sha256=sha256(Path(__file__).with_name('public_panel_sonar_math.py')),
        method='Sonar-style static shrinkage Gaussian E_high_progress - E_low_progress contrast, not output-calibration or external exact-paper replication',
        labels='TRAIN-only within-state median actual15-action hand-goal progress; >=median high, <median low; not episode success',
        folds='Leave one of three already-seen development states out; all five historical native/067/sham/126/sham plans retained',
        PCA='TRAIN-only maximum3 full-token raw Euclidean PCs, standardized before Gaussian fit',shrinkage=.2,covariance_floor=1e-4,
        site='P3 residual newest256x400 at imaginedH3 only',doses=DOSES,base_step_size=1.,
        cap='.005 recipient raw fullspatial L2; base unconstrained gradient cap first, signed dose afterward',
        metric='Native H3 returnedvisual/proprio finite-difference J^TJ in same TRAIN-PC raw coordinates; trace-normalized +.01 damping',
        no_eval_labels_in_fit=True,exactself=True,sham='Same PCA subspace private random coordinate direction, matched separately to each family/action/dose',
        primary='H3 native cost rank vs actual matching15-action progress and selected action physical endpoint',
        secondary='Actual encoded visual H3 error; H6 objective only prefix-associated, not H6 physical truth',
        no_held=True,cem_calls=0,simulator_calls=0,max_seconds=900,
        direction_only_ablation=bool(getattr(a,'normalize_direction',False)),
        direction_only_note='If enabled, normalize nonzero gradient fields to the same .005 raw residual L2 before signed doses; this discards energy magnitude and tests directions only, not the original energy-step rule.')
    if protocol['direction_only_ablation']:
        protocol['method']+='; direction-only normalized-dose ablation'
        protocol['base_step_size']=None
        protocol['cap']='Normalize each nonzero field to .005 recipient raw fullspatial L2, then apply signed dose; zero directions remain zero'
    write_json(a.output/'protocol.json',protocol);torch.set_num_threads(2);torch.manual_seed(905081);began=time.monotonic()
    wm,prep,provenance=load_headless_metaworld(a.repo);wm.eval().requires_grad_(False)
    if sha256(provenance['checkpoint'])!=CHECKPOINT:raise ValueError('Wrong checkpoint')
    before=parameters_sha(wm);cfg=setup_cfg(a.config,a.output,wm);block=wm.model.predictor.predictor_blocks[3];data=[];outputs=[]
    for row in rows:
        path=a.inputs/row['path']
        if sha256(path)!=row['sha256']:raise ValueError('Input SHA before load')
        b=torch.load(path,map_location='cpu',weights_only=False);agent=GC_Agent(cfg,wm,dset=None,preprocessor=prep)
        agent.set_goal(TensorDict(b['raw_goal'],batch_size=[]))
        z=batch_context(b['context'],5).to(wm.device);plan=b['normalized_actions'].contiguous().to(wm.device)
        native=wm.unroll(z.clone(),act_suffix=plan)
        with P3(block) as cap:repeat=wm.unroll(z.clone(),act_suffix=plan)
        for k in ('visual','proprio'):exact(native[k],repeat[k],'capture identity')
        data.append(dict(bank=b,source=row,agent=agent,z=z,plan=plan,native=native,residual=cap.value.flatten(1),
            input_hashes={k:tensor_hash(z[k]) for k in z.keys()},goal_hashes={k:tensor_hash(agent.goal_state_enc[k]) for k in ('visual','proprio')}))
    for index,d in enumerate(data):
        train=[v for j,v in enumerate(data) if j!=index];fit=fit_energy(torch.cat([v['residual'] for v in train]).numpy(),[v['bank']['progress'] for v in train])
        ep=d['bank']['episode'];energy,gradient=raw_energy_gradient(d['residual'].numpy(),fit);basis=torch.from_numpy(fit['basis'])
        fit_path=a.output/f'episode-{ep:03}-frozen-fit.pt';torch.save(dict(fit=fit,training_episodes=[v['bank']['episode'] for v in train]),fit_path)
        fit_sha=sha256(fit_path);outputs.append(dict(path=fit_path.name,sha256=fit_sha,bytes=fit_path.stat().st_size))
        raw_norm=d['residual'].double().norm(dim=1);epsilon=.001*raw_norm;responses=[]
        for axis in range(fit['rank']):
            values=[]
            for sign in (-1,1):
                delta=(basis[axis][None]*epsilon[:,None]*sign).float()
                with P3(block,delta):pred=wm.unroll(d['z'].clone(),act_suffix=d['plan'])
                for k in ('visual','proprio'):exact(pred[k][:3],d['native'][k][:3],'Jacobian unchanged prefix')
                parts=[]
                for k,weight in (('visual',1.),('proprio',float(d['agent'].objective.alpha))):
                    value=pred[k][3].double().cpu().flatten(1);parts.append(value*(weight/value.shape[1])**.5)
                values.append(torch.cat(parts,1))
            responses.append((values[1]-values[0])/(2*epsilon[:,None]))
        jac=torch.stack(responses,2);gram=(jac.transpose(1,2)@jac).numpy();coefficients,cap_records=capped_fields(gradient,gram,.005*raw_norm.numpy(),bool(getattr(a,'normalize_direction',False)))
        generator=np.random.default_rng(9050800+ep);random=generator.normal(size=(5,fit['rank']));random/=np.linalg.norm(random,axis=1,keepdims=True)
        fields={name:torch.from_numpy(value)@basis for name,value in coefficients.items()}
        for name in list(fields):fields[name+'_norm_sham']=torch.from_numpy(random*np.linalg.norm(coefficients[name],axis=1,keepdims=True))@basis
        conditions=[('exact_self',torch.zeros_like(d['residual']))]+[(name+f'/dose{dose:+g}',(field*dose).float()) for name,field in fields.items() for dose in DOSES]
        native_cost=d['agent'].objective(d['native'],d['plan'],keepdims=True).cpu().reshape(7,5)
        rows_out=[];predictions={};rounded={}
        for name,delta in conditions:
            if time.monotonic()-began>900:raise RuntimeError('Bounded Sonar budget exceeded')
            with P3(block,delta) as patch:changed=wm.unroll(d['z'].clone(),act_suffix=d['plan'])
            for k in ('visual','proprio'):
                exact(changed[k][:3],d['native'][k][:3],'unchanged preH3')
                if name=='exact_self':exact(changed[k],d['native'][k],'exact self')
            actual=(patch.value.flatten(1)+delta)-patch.value.flatten(1);actual_norm=actual.double().norm(dim=1);requested=delta.double().norm(dim=1)
            if float((actual_norm-requested).abs().max())>1e-3:raise ValueError('Rounded norm differs from requested budget')
            if bool((actual_norm>.005*raw_norm+1e-3).any()):raise ValueError('Raw trust region exceeded')
            rounded[name]=actual_norm.tolist();cost=d['agent'].objective(changed,d['plan'],keepdims=True).cpu().reshape(7,5)
            visual=changed['visual'][3].cpu();error=(visual-d['bank']['actual_H3_visual']).double().square().flatten(1).mean(1)
            metric=rank_metrics(cost[3],torch.from_numpy(d['bank']['endpoint_distance']))
            rows_out.append(dict(arm=name,**metric,selected_actual_progress_m=float(d['bank']['progress'][metric['selected_candidate']]),
                H3_cost=cost[3].tolist(),H6_cost_secondary=cost[6].tolist(),actual_H3_visual_mse=error.tolist(),
                raw_requested_norm=requested.tolist(),actual_rounded_norm=actual_norm.tolist()))
            predictions[name]=visual
        for name in ('static','local_metric'):
            for dose in DOSES:
                torch.testing.assert_close(torch.tensor(rounded[name+f'/dose{dose:+g}']),torch.tensor(rounded[name+'_norm_sham'+f'/dose{dose:+g}']),rtol=1e-3,atol=1e-3)
        if d['input_hashes']!={k:tensor_hash(d['z'][k]) for k in d['z'].keys()} or d['goal_hashes']!={k:tensor_hash(d['agent'].goal_state_enc[k]) for k in d['goal_hashes']}:
            raise ValueError('Inputs/goal modified')
        native_metrics=rank_metrics(native_cost[3],torch.from_numpy(d['bank']['endpoint_distance']))
        pt=a.output/f'episode-{ep:03}.pt';torch.save(dict(source=d['source'],residual=d['residual'],fields=fields,energy=energy,gradient=gradient,
            metric=gram,native_response_jacobian=jac.float(),predicted_H3_visual=predictions,native_H3_visual=d['native']['visual'][3].cpu(),
            actual_H3_visual=d['bank']['actual_H3_visual'],plans=d['plan'].cpu(),fit_sha256=fit_sha),pt)
        report=pt.with_suffix('.json');write_json(report,dict(complete=True,episode=ep,training_episodes=[v['bank']['episode'] for v in train],rows=rows_out,
            native_H3_ranking=native_metrics,native_selected_actual_progress_m=float(d['bank']['progress'][native_metrics['selected_candidate']]),
            native_H3_cost=native_cost[3].tolist(),actual_H3_progress_by_candidate=d['bank']['progress'].tolist(),
            native_actual_H3_visual_mse=(d['native']['visual'][3].cpu()-d['bank']['actual_H3_visual']).double().square().flatten(1).mean(1).tolist(),
            cap_records=cap_records,PCA_rank=fit['rank'],fit_sha256=fit_sha,fit_saved_before_test_edits=True,identity_exact=True,
            good_training_count=int(fit['labels'].sum()),bad_training_count=int((~fit['labels']).sum()),
            tensor_sha256=sha256(pt),tensor_bytes=pt.stat().st_size,elapsed_seconds=time.monotonic()-began))
        outputs.extend(dict(path=f.name,sha256=sha256(f),bytes=f.stat().st_size) for f in (pt,report))
        print(json.dumps(dict(event='sonar_state_complete',episode=ep,seconds=time.monotonic()-began)),flush=True)
        if index==0:write_json(a.output/'CANARY.json',dict(complete=True,identity_and_norms_passed=True,no_outcome_gate=True,seconds=time.monotonic()-began))
    if parameters_sha(wm)!=before:raise ValueError('Weights changed')
    for name in ('protocol.json','CANARY.json'):
        path=a.output/name;outputs.append(dict(path=name,sha256=sha256(path),bytes=path.stat().st_size))
    write_json(a.output/'DONE.json',dict(complete=True,outputs=outputs,states=3,conditions_per_state=17,seconds=time.monotonic()-began,
        weights_unchanged=True,parameter_sha256=before,provenance=provenance,held_data=False,cem_calls=0,simulator_calls=0))


def main():
    p=argparse.ArgumentParser()
    for key in ('inputs','repo','config','prepare_coordinates'):p.add_argument('--'+key.replace('_','-'),type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--normalize-direction',action='store_true',help='Explicit direction-only dose ablation, not original energy-step rule')
    a=p.parse_args()
    try:
        if a.prepare_coordinates:prepare(a.prepare_coordinates,a.output)
        else:run(a)
    except Exception as exc:
        if a.output.exists():write_json(a.output/'FAILED.json',dict(complete=False,error=repr(exc)))
        raise


if __name__=='__main__':main()
