#!/usr/bin/env python3
"""Exploratory TRAIN-only two-state routing of a fixed action-tangent field."""
import argparse
import json
from pathlib import Path
import time
import torch
from run_reach_native_coordinates import sha, write, P3, rank_metrics, EPISODES, CHECKPOINT


def emission(x, mean, variance):
    return -.5*(((x[...,None,:]-mean)**2/variance)+variance.log()+torch.log(torch.tensor(2*torch.pi))).sum(-1)


def filter_beliefs(x, model):
    emit=emission(x,model['means'],model['variance']);state=model['initial'].log()+emit[:,0]
    states=[state.softmax(-1)]
    for t in range(1,x.shape[1]):
        state=torch.logsumexp(state[:,:,None]+model['transition'].log()[None],1)+emit[:,t]
        states.append(state.softmax(-1))
    return torch.stack(states,1)


def fit_hmm(features, iterations=20):
    """Only model features accepted; no physical labels or development endpoints."""
    n,t,d=features.shape;flat=features.reshape(-1,d).double();center=flat.mean(0)
    _,_,components=torch.linalg.svd(flat-center,full_matrices=False);basis=components[:3]
    scale=((flat-center)@basis.T).std(0,unbiased=False).clamp_min(1e-8)
    x=((features.double()-center)@basis.T)/scale
    order=torch.argsort(x.reshape(-1,3)[:,0]);half=len(order)//2
    means=torch.stack([x.reshape(-1,3)[order[:half]].mean(0),x.reshape(-1,3)[order[half:]].mean(0)])
    variance=torch.ones(2,3,dtype=torch.float64);initial=torch.ones(2,dtype=torch.float64)/2
    transition=torch.tensor([[.9,.1],[.1,.9]],dtype=torch.float64);history=[]
    for _ in range(iterations):
        emit=emission(x,means,variance);forward=[initial.log()+emit[:,0]]
        for j in range(1,t):forward.append(torch.logsumexp(forward[-1][:,:,None]+transition.log()[None],1)+emit[:,j])
        alpha=torch.stack(forward,1);normalizer=torch.logsumexp(alpha[:,-1],-1)
        beta=torch.zeros_like(alpha)
        for j in range(t-2,-1,-1):beta[:,j]=torch.logsumexp(transition.log()[None]+emit[:,j+1,None,:]+beta[:,j+1,None,:],-1)
        gamma=(alpha+beta-normalizer[:,None,None]).exp()
        xi=(alpha[:,:-1,:,None]+transition.log()[None,None]+emit[:,1:,None,:]+beta[:,1:,None,:]-normalizer[:,None,None,None]).exp().sum((0,1))
        initial=(gamma[:,0].sum(0)+1)/(n+2);transition=(xi+1)/(xi.sum(1,keepdim=True)+2)
        count=gamma.sum((0,1));means=(gamma[...,None]*x[:,:,None,:]).sum((0,1))/count[:,None]
        variance=(gamma[...,None]*(x[:,:,None,:]-means).square()).sum((0,1))/count[:,None];variance=variance.clamp_min(.05)
        history.append(float(normalizer.sum()))
    model=dict(center=center,basis=basis,scale=scale,means=means,variance=variance,initial=initial,transition=transition)
    beliefs=filter_beliefs(x,model);drift=torch.cat([torch.zeros(n,1), (x[:,1:]-x[:,:-1]).norm(dim=-1)],1)
    regime_drift=(beliefs*drift[:,:,None]).sum((0,1))/beliefs.sum((0,1))
    gates=torch.ones(2,dtype=torch.float64);gates[regime_drift.argmax()]=.5
    model.update(gates=gates,occupancy=beliefs.mean((0,1)),regime_drift=regime_drift,log_likelihood_history=history,
                 training_sequences=n,training_length=t)
    return model


def route(features, model):
    x=((features.double()-model['center'])@model['basis'].T)/model['scale']
    # Only native predicted H1/H2 are available to the routing decision at H3.
    posterior=filter_beliefs(x[:,:2],model)[:,-1]
    memoryless=(emission(x[:,1],model['means'],model['variance'])+model['occupancy'].log()).softmax(-1)
    static=float(model['occupancy']@model['gates'])
    return dict(static=torch.full((len(x),),static,dtype=torch.float64),hmm=posterior@model['gates'],
                memoryless=memoryless@model['gates'],posterior=posterior)


def equal_energy(weights, base_norms, static):
    desired=(base_norms.square()*static.square()).sum()
    actual=(base_norms.square()*weights.square()).sum()
    return weights*(desired/actual.clamp_min(1e-30)).sqrt()


def prepare(a):
    a.output.mkdir(parents=True,exist_ok=False);d=json.loads((a.coordinates/'results-v1/DONE.json').read_text())
    if not d['complete'] or d['states']!=3 or d['held_data']:raise ValueError('Only completed development coordinate source')
    outputs=[]
    for episode in EPISODES:
        row=next(r for r in d['outputs'] if r['path']==f'episode-{episode:03}.pt');path=a.coordinates/'results-v1'/row['path']
        if sha(path)!=row['sha256']:raise ValueError('Coordinate SHA before load')
        value=torch.load(path,map_location='cpu',weights_only=False);source=value['source']
        if source['episode']!=episode or source['split']!='development':raise ValueError('Wrong development source')
        plan=source['plans'][0]
        if sha(plan['path'])!=plan['sha256'] or sha(source['goal_path'])!=source['goal_sha256']:raise ValueError('Context/goal input changed')
        bank=torch.load(plan['path'],map_location='cpu',weights_only=False)
        goal=torch.load(source['goal_path'],map_location='cpu',weights_only=False)
        actual=value['actual_H3_visual'];field=value['fields']['local_action_jacobian'];sham=value['fields']['rank3_random_sham']
        bundle=dict(episode=episode,split='development',source_sha256=row['sha256'],context=bank['encoded_context'],
            goal={k:goal['goal'][k] for k in ('visual','proprio')},actions=value['selected_plans'],actual=actual,
            distances=torch.tensor(value['report']['actual_H3_goal_distance_m'],dtype=torch.float64),
            p3_history=value['native_p3_pooled_history'].transpose(0,1).contiguous(),field=field,sham=sham,
            coordinate_training_episodes=value['report']['training_episodes'])
        out=a.output/f'episode-{episode:03}.pt';torch.save(bundle,out)
        outputs.append(dict(path=out.name,episode=episode,split='development',sha256=sha(out),bytes=out.stat().st_size))
    write(a.output/'DONE.json',dict(complete=True,outputs=outputs,source_done_sha256=sha(a.coordinates/'results-v1/DONE.json'),held_data=False))
    print(json.dumps(dict(complete=True,prepared=3,bytes=sum(r['bytes'] for r in outputs))),flush=True)


@torch.no_grad()
def run(a):
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);began=time.monotonic()
    done=json.loads((a.inputs/'DONE.json').read_text())
    if not done['complete'] or done['held_data'] or [r['episode'] for r in done['outputs']]!=list(EPISODES):raise ValueError('Only original three development folds')
    data=[]
    for row in done['outputs']:
        if row['split']!='development' or sha(a.inputs/row['path'])!=row['sha256']:raise ValueError('Split/SHA before load')
        data.append(torch.load(a.inputs/row['path'],map_location='cpu',weights_only=False))
    protocol=dict(episodes=list(EPISODES),held_data=False,candidate='Frozen local_action_jacobian +1 from coordinate pilot; not chosen by HMM outcomes',
        fit='LOO initial state; two diagonal Gaussian states, PCA3,20deterministic EM steps, Dirichlet1 transitions, variancefloor.05',
        emission_sequence='Native imagined P3 pooled H1..H6 TRAIN; test routing uses H1/H2 only, physical and imagined time not conflated',
        regime_rule='Higher TRAIN PC step-change norm gate.5; lower gate1; no labels fit this mapping',
        static='TRAIN occupancy weighted gate; all treatments equal aggregate raw patch energy across5plans',
        success_rule='HMM improves mean H3 rank correlation and forecast MSE plus TRAIN-action-subspace selectivity over static, with positive differences in at least2/3 excluded states; otherwise reject',
        controls=['unsteered','exact_self','static','hmm','memoryless','time_only_equals_static','static_sham','hmm_sham'],
        block=3,horizon=3,rank=3,simulator_calls=0,cem_calls=0,script_sha256=sha(__file__),max_process_seconds=300)
    write(a.output/'protocol.json',protocol)
    from model_loader import load_headless_metaworld
    from capture_specificity_controls import parameters_sha
    from causal_planner_forks import setup_cfg
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    from intervene_head_spatial_mean import batch_context
    from tensordict import TensorDict
    wm,prep,provenance=load_headless_metaworld(a.repo);wm.eval().requires_grad_(False)
    if sha(provenance['checkpoint'])!=CHECKPOINT:raise ValueError('Wrong native weights')
    before=parameters_sha(wm);cfg=setup_cfg(a.config,a.output,wm);outputs=[];reports=[]
    for i,d in enumerate(data):
        train=[v for j,v in enumerate(data) if j!=i]
        if set(d['coordinate_training_episodes'])!={v['episode'] for v in train}:raise ValueError('Coordinate target fold leaked')
        hmm=fit_hmm(torch.cat([v['p3_history'] for v in train]));gates=route(d['p3_history'],hmm)
        # Save fitted routing parameters before evaluating any excluded-state physical endpoint.
        frozen={k:v.tolist() if isinstance(v,torch.Tensor) else v for k,v in hmm.items()}
        write(a.output/f'episode-{d["episode"]:03}-FROZEN.json',dict(training_episodes=[v['episode'] for v in train],held_development_state=d['episode'],model=frozen))
        effect=torch.cat([(v['actual']-v['actual'][0:1]).flatten(1)[1:] for v in train]).double()
        _,_,behavior_basis=torch.linalg.svd(effect,full_matrices=False);behavior_basis=behavior_basis[:3]
        agent=GC_Agent(cfg,wm,dset=None,preprocessor=prep);agent.set_goal(TensorDict(d['goal'],batch_size=[]))
        z=batch_context(d['context'],5).to(wm.device);actions=d['actions'].contiguous().to(wm.device)
        native=wm.unroll(z.clone(),act_suffix=actions);norm=d['field'].double().norm(dim=1)
        hm=equal_energy(gates['hmm'],norm,gates['static']);mem=equal_energy(gates['memoryless'],norm,gates['static'])
        fields={name:d['field']*gate[:,None].float() for name,gate in [('static',gates['static']),('hmm',hm),('memoryless',mem)]}
        fields.update(exact_self=torch.zeros_like(d['field']),time_only_equals_static=fields['static'].clone(),
            static_sham=d['sham']*gates['static'][:,None].float(),hmm_sham=d['sham']*hm[:,None].float())
        rows=[];predictions={}
        for name,delta in [('unsteered',None)]+list(fields.items()):
            if time.monotonic()-began>300:raise RuntimeError('Bounded HMM process budget exceeded')
            if delta is None:changed=native
            else:
                with P3(wm.model.predictor.predictor_blocks[3],delta):changed=wm.unroll(z.clone(),act_suffix=actions)
            for key in ('visual','proprio'):
                if not torch.equal(changed[key][:3],native[key][:3]):raise ValueError('H1/H2 changed')
                if name=='exact_self' and not torch.equal(changed[key],native[key]):raise ValueError('Identity failed')
            prediction=changed['visual'][3].float().cpu();change=(prediction-native['visual'][3].cpu()).double().flatten(1)
            length=change.norm(dim=1);projected=(change@behavior_basis.T).norm(dim=1)
            selectivity=[float(p/n) if n>1e-10 else None for p,n in zip(projected,length)]
            error=(prediction-d['actual']).square().flatten(1).mean(1)
            cost=agent.objective(changed,actions,keepdims=True).float().cpu().reshape(7,5)[3]
            rows.append(dict(condition=name,ranking=rank_metrics(cost,d['distances']),visual_MSE=error.tolist(),
                mean_visual_MSE=float(error.mean()),training_action_subspace_fraction=selectivity,
                output_change_norm=length.tolist(),raw_patch_norm=None if delta is None else delta.double().norm(dim=1).tolist()))
            predictions[name]=prediction
        if not torch.equal(predictions['static'],predictions['time_only_equals_static']):raise ValueError('Time-only/static same-energy null failed')
        report=dict(complete=True,episode=d['episode'],rows=rows,gates={k:v.tolist() for k,v in gates.items()},
            energy_normalized_hmm_gates=hm.tolist(),energy_normalized_memoryless_gates=mem.tolist(),
            training_episodes=[v['episode'] for v in train],held_physical_labels_used_by_routing=False)
        out=a.output/f'episode-{d["episode"]:03}.pt';torch.save(dict(report=report,predictions=predictions,model=hmm),out)
        write(out.with_suffix('.json'),report);reports.append(report)
        outputs.extend(dict(path=p.name,sha256=sha(p),bytes=p.stat().st_size) for p in (out,out.with_suffix('.json'),a.output/f'episode-{d["episode"]:03}-FROZEN.json'))
        print(json.dumps(dict(event='hmm_state_complete',episode=d['episode'],seconds=time.monotonic()-began)),flush=True)
    if parameters_sha(wm)!=before:raise ValueError('Weights changed')
    write(a.output/'DONE.json',dict(complete=True,outputs=outputs,seconds=time.monotonic()-began,model=provenance,states=3,
        simulator_calls=0,cem_calls=0,held_data=False,limitations=['New exploratory HMM on imagined sequences, not validated physical regimes',
            'Routing uses native pre-intervention H1/H2; static candidate already failed ranking pilot',
            'Three reused development initial states; no confirmation or autonomous policy efficacy',
            'Selectivity is in TRAIN true-future encoded action-difference span, not proven physical specificity',
            'No fullH6 physical truth; H3 native cost and actual15-step outcome only']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--output',type=Path,required=True)
    for key in ('coordinates','inputs','repo','config'):p.add_argument('--'+key,type=Path)
    a=p.parse_args()
    try:prepare(a) if a.mode=='prepare' else run(a)
    except Exception:
        import traceback
        if a.output.exists() and not (a.output/'FAILED.json').exists():write(a.output/'FAILED.json',dict(complete=False,exception=traceback.format_exc()))
        raise
