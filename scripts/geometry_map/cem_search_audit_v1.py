#!/usr/bin/env python3
"""Read-only native CEM trace, then paired mean-versus-best physical forks."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch

STAGES=(1,15,30)
CHECKPOINT='9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2);f.write('\n')


def exact(a,b,label):
    if torch.is_tensor(a):a=a.detach().cpu().numpy()
    if torch.is_tensor(b):b=b.detach().cpu().numpy()
    if not np.array_equal(a,b):raise ValueError('Exact guard: '+label)


def prediction_cpu(out):return {k:out[k].detach().cpu().clone() for k in ('visual','proprio')}


class CostObserver:
    def __init__(self,objective):self.objective=objective;self.costs=[]
    def __call__(self,out,actions,**kwargs):
        cost=self.objective(out,actions,**kwargs)
        if actions.shape[1]==300 and not kwargs.get('keepdims',False):self.costs.append(cost.detach().cpu().clone())
        return cost


def elite_statistics(actions,cost,num_elites):
    if actions.ndim!=3 or cost.shape!=(actions.shape[1],):raise ValueError('Native candidate axes required')
    idx=torch.topk(-cost,num_elites,dim=0).indices
    elite=actions[:,idx]
    return idx,elite.mean(1),elite.std(1),int(cost.argmin())


class NativeTrace:
    """Wraps instance methods only, never changes a proposed action or cost."""
    def __init__(self,planner):
        self.p=planner;self.native_cost=planner.cost_function;self.native_unroll=planner.unroll
        self.rows=[];self.stages={};self.in_cost=False;self.samples=None
    def __enter__(self):self.p.cost_function=self.cost;self.p.unroll=self.unroll;return self
    def __exit__(self,*args):self.p.cost_function=self.native_cost;self.p.unroll=self.native_unroll
    def unroll(self,*args,**kwargs):
        actions=kwargs.get('act_suffix',args[1] if len(args)>1 else None)
        out=self.native_unroll(*args,**kwargs)
        if self.in_cost:self.samples=out
        else:
            row=self.rows[-1];exact(actions[:,0],row['mean_after'],'native updated mean')
            row['mean_cost']=float(self.p.objective(out,actions)[0])
            row['mean_cost_by_horizon']=self.p.objective(out,actions,keepdims=True).detach().cpu()
            if row['iteration'] in STAGES:self.stages[row['iteration']]['mean_forecast']=prediction_cpu(out)
        return out
    def cost(self,actions,z):
        before=actions.detach().clone();self.in_cost=True
        try:cost=self.native_cost(actions,z)
        finally:self.in_cost=False
        exact(actions,before,'read-only proposed samples')
        idx,newmean,newstd,best=elite_statistics(actions,cost,self.p.num_elites)
        previous=actions[:,0]
        if self.rows:exact(previous,self.rows[-1]['mean_after'],'candidate0 is previous mean')
        mean=newmean*(1-self.p.momentum_mean)+previous*self.p.momentum_mean
        oldstd=torch.ones_like(newstd)*self.p.var_scale if not self.rows else self.rows[-1]['std_after'].to(newstd)
        std=newstd*(1-self.p.momentum_std)+oldstd*self.p.momentum_std
        iteration=len(self.rows)+1
        row=dict(iteration=iteration,actions=actions.detach().cpu().clone(),costs=cost.detach().cpu().clone(),elite_indices=idx.cpu(),
            mean_before=previous.detach().cpu().clone(),mean_after=mean.cpu(),std_after=std.cpu(),best_index=best,best_cost=float(cost[best]),
            elite_cost_mean=float(cost[idx].mean()),elite_cost_std=float(cost[idx].std()),rng_after_draw=self.p.local_generator.get_state().cpu())
        self.rows.append(row)
        if iteration in STAGES:
            self.stages[iteration]=dict(best_action=actions[:,best].cpu().clone(),mean_action=mean.cpu(),
                best_forecast={k:self.samples[k][:,best:best+1].detach().cpu().clone() for k in ('visual','proprio')})
        self.samples=None
        print(json.dumps(dict(event='cem_iteration',iteration=iteration,best_cost=row['best_cost'])),flush=True)
        return cost


def compare_plans(native,traced):
    for key in ('actions','losses','prev_elite_losses_mean','prev_elite_losses_std'):exact(getattr(native,key),getattr(traced,key),'full CEM '+key)
    if len(native.predicted_best_encs_over_iterations)!=30 or len(traced.predicted_best_encs_over_iterations)!=30:raise ValueError('Thirty native means')
    for a,b in zip(native.predicted_best_encs_over_iterations,traced.predicted_best_encs_over_iterations):
        for k in ('visual','proprio'):exact(a[k],b[k],'all thirty native mean forecasts '+k)


def select_source(row,prepared):
    episode=row['episode'];path=Path(row['bank']);receipt=json.loads(Path(row['receipt']).read_text())
    source=next(r for r in receipt['outputs'] if r['path']==path.name)
    if episode not in range(4) or not receipt['complete'] or not receipt['original9_actions_goal_physics_pixels_exact'] or source['source_id']!=episode or source['split']!='development_external':raise ValueError('Fixed DEV before tensor load')
    resetrow=next(r for r in prepared['episodes'] if r['episode']==episode)
    if not prepared['complete'] or resetrow['split']!='development':raise ValueError('Prepared split before load')
    if sha(path)!=source['sha256']:raise ValueError('Source bank SHA')
    return path,source,resetrow


@torch.no_grad()
def run_state(args,row,wm,prep,cfg):
    from tensordict import TensorDict
    from evals.simu_env_planning.envs.init import make_env
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
    from collect_curvature_interior_futures import replay,compare_replays
    from capture_pusht_horizon import encoder_inputs
    from diagnose_pusht_goal_coverage import coverage,native_helper
    started=time.monotonic();episode=row['episode']
    prepared=json.loads((args.prepared_inputs/'DONE.json').read_text());path,source,rr=select_source(row,prepared)
    resetpath=args.prepared_inputs/rr['path']
    if sha(resetpath)!=rr['sha256']:raise ValueError('Original reset SHA')
    bank=torch.load(path,map_location='cpu',weights_only=False);reset=torch.load(resetpath,map_location='cpu',weights_only=False)
    if bank['row']['source_id']!=episode or reset['episode']!=episode or reset['split']!='development':raise ValueError('Tensor identity')
    exact(bank['goal_state'],reset['goal_state'],'original requested goal')
    exact(bank['raw_goal']['visual'],reset['expert_observations'][-1],'original requested goal pixels')
    exact(bank['raw_goal']['proprio'],reset['expert_proprios'][-1],'original requested goal proprio')
    goal=np.asarray(bank['goal_state']);env=make_env(cfg);envinfo=reset.get('env_info',{'shape':'T'})
    def initialize():
        env.update_env(envinfo);obs,info=env.prepare(reset['environment_seed'],np.asarray(reset['initial_state']).copy(),envinfo)
        exact(obs,reset['expert_observations'][0],'original raw prepared pixels')
        exact(info['state'],reset['expert_states'][0],'prepared observed state')
        exact(info['state'],bank['candidates'][0]['states'][0],'same seen bank state')
        return obs,info
    initialize()
    agent=GC_Agent(cfg,wm,dset=None,preprocessor=prep);planner=agent.planner
    if (planner.iterations,planner.num_samples,planner.num_elites,planner.horizon,planner.num_act_stepped)!=(30,300,10,6,6):raise ValueError('Native Push CEM configuration')
    if planner.max_norms is not None or planner.distribute_planner or planner.momentum_mean or planner.momentum_std:raise ValueError('Unexpected CEM update policy')
    target=TensorDict({k:v.clone().cuda() for k,v in bank['goal_encoded'].items()},batch_size=[])
    if any(v.dtype!=torch.float32 for v in target.values()):raise ValueError('Original fullprecision fixed goal')
    objective=ReprTargetDistMPCObjective(cfg,target,**cfg.planner.planning_objective);planner.set_objective(objective)
    z=TensorDict({k:v.clone().cuda() for k,v in bank['context'].items()},batch_size=[]);original=z.clone()
    seed=int(reset['planner_seed']);agent.local_gpu_generator.manual_seed(seed);rng=agent.local_gpu_generator.get_state().clone()
    native=None;identity_seconds=0.
    resume=args.resume_cem is not None and episode==2
    if resume:
        if sha(args.resume_cem)!=args.resume_cem_sha256:raise ValueError('Immutable prior CEM hash')
        saved=torch.load(args.resume_cem,map_location='cpu',weights_only=False)
        if saved['episode']!=episode or saved['planner_seed']!=seed or not saved['identity_full_cem_exact']:raise ValueError('Resume original exact canary only')
        trace=NativeTrace(planner);trace.rows=saved['rows'];trace.stages=saved['stages'];native_exec=saved['native_returned_actions']
        if len(trace.rows)!=30 or set(trace.stages)!=set(STAGES):raise ValueError('Complete original search required')
        cem_seconds=0.
    elif args.identity:
        observer=CostObserver(objective);planner.set_objective(observer)
        t=time.monotonic();native=planner.plan(z.clone());native_rng=agent.local_gpu_generator.get_state().clone();identity_seconds=time.monotonic()-t
        planner.set_objective(objective)
        print(json.dumps(dict(event='native_identity_reference_complete',episode=episode,seconds=identity_seconds)),flush=True)
    if not resume:
        agent.local_gpu_generator.set_state(rng);planner._prev_mean=None
        t=time.monotonic()
        with NativeTrace(planner) as trace:traced=planner.plan(z.clone())
        cem_seconds=time.monotonic()-t
        if native is not None:
            compare_plans(native,traced);exact(agent.local_gpu_generator.get_state(),native_rng,'planner RNG exact')
            if len(observer.costs)!=30:raise ValueError('Native cost observer missed an iteration')
            for a,b in zip(observer.costs,trace.rows):exact(a,b['costs'],'all native candidate costs')
        if len(trace.rows)!=30:raise ValueError('Full native search required')
        exact(traced.actions,trace.rows[-1]['mean_after'][:planner.num_act_stepped],'returned executed prefix is final elite mean')
        native_exec=traced.actions.detach().cpu().clone();del native,traced
    torch.save(dict(episode=episode,rows=trace.rows,stages=trace.stages,planner_rng_before=rng.cpu(),planner_seed=seed,
        native_returned_actions=native_exec,identity_full_cem_exact=args.identity or resume),args.output/f'episode-{episode:03d}-CEM.pt')
    write(args.output/f'episode-{episode:03d}-MODEL_COMPLETE.json',dict(complete=True,cem_seconds=cem_seconds,identity_seconds=identity_seconds,identity_full_cem_exact=args.identity or resume,resumed_prior_cem=resume))
    print(json.dumps(dict(event='search_complete',episode=episode,cem_seconds=cem_seconds,identity_seconds=identity_seconds)),flush=True)
    records=[];metrics=[];polygon=native_helper(args.repo/'evals/simu_env_planning/envs/pusht_env/pusht_env.py');targetpolygon=polygon(goal[2:5])
    try:
        for stage in STAGES:
            for kind in ('best','mean'):
                plan=trace.stages[stage][kind+'_action'];raw=prep.denormalize_actions(plan.reshape(30,2)).float()
                obs,info=initialize();truth=replay(env,obs,info,raw,goal)
                obs,info=initialize();repeat=replay(env,obs,info,raw,goal);compare_replays(truth,repeat)
                forecast=trace.stages[stage][kind+'_forecast']
                # Use the ACTUAL native population-best/mean forecasts; no reranking or fresh cross-host model replay.
                reforecast=TensorDict({k:v.cuda() for k,v in forecast.items()},batch_size=[])
                actual=wm.encode(encoder_inputs(truth['frames'][::5],truth['proprios'][::5]))
                predicted={k:reforecast[k][1:,0].float().cpu() for k in ('visual','proprio')}
                av={k:actual[k][0,1:].float().cpu() for k in ('visual','proprio')}
                mse={k:(predicted[k].double()-av[k].double()).square().flatten(1).mean(1).tolist() for k in predicted}
                states=truth['states'];covered=[coverage(s[2:5],goal[2:5]) for s in states[5::5]]
                check=polygon(states[-1,2:5]).intersection(targetpolygon).area/targetpolygon.area
                if abs(check-covered[-1])>1e-12:raise ValueError('Native goalpolygon parity')
                predcost=objective(reforecast,plan[:,None].cuda(),keepdims=True)[1:,0].cpu()
                actualtd=reforecast.clone()
                for k in ('visual','proprio'):actualtd[k]=actual[k].transpose(0,1)
                actualcost=objective(actualtd,plan[:,None].cuda(),keepdims=True)[1:,0].cpu()
                rec=dict(stage=stage,kind=kind,normalized_actions=plan,raw_actions=raw,truth=truth,forecast=predicted,actual=av,predicted_cost=predcost,actual_encoded_cost=actualcost)
                records.append(rec)
                item=dict(stage=stage,kind=kind,requested_coverage_by_horizon=covered,requested_coverage_final=covered[-1],
                    goal_xy_distance_by_horizon=np.linalg.norm(states[5::5,:4]-goal[:4],axis=-1).tolist(),goal_state7=goal.tolist(),states7_by_horizon=states[5::5].tolist(),
                    native_reward_final=float(truth['native_rewards'][-1]),native_goal_success_final=bool(truth['native_goal_success'][-1]),native_goal_ever_success=bool(truth['native_goal_success'].any()),
                    forecast_mse_by_horizon=mse,predicted_cost_by_horizon=predcost.tolist(),actual_encoded_cost_by_horizon=actualcost.tolist(),
                    original_selection_cost=trace.rows[stage-1]['best_cost' if kind=='best' else 'mean_cost'],
                    crossbatch_cost_difference=float(predcost[-1])-trace.rows[stage-1]['best_cost' if kind=='best' else 'mean_cost'],
                    executed_prefix_raw_steps=planner.num_act_stepped*5,executed_prefix_equals_H6=True,exact_replay=True)
                metrics.append(item);print(json.dumps(dict(event='fork_complete',episode=episode,stage=stage,kind=kind,coverage=covered[-1])),flush=True)
    finally:env.close()
    for k in z.keys():exact(z[k],original[k],'fixed native context unchanged')
    tensor=args.output/f'episode-{episode:03d}-FORKS.pt';torch.save(dict(episode=episode,source_sha256=source['sha256'],reset_sha256=rr['sha256'],original_raw_reset=reset['initial_state'],
        environment_seed=reset['environment_seed'],planner_seed=seed,context=bank['context'],raw_goal=bank['raw_goal'],goal_encoded=bank['goal_encoded'],goal_state=goal,records=records),tensor)
    iterations=[{k:(v.tolist() if torch.is_tensor(v) else v) for k,v in r.items() if k not in ('actions','rng_after_draw')} for r in trace.rows]
    differences=[]
    for stage in STAGES:
        best=next(r for r in metrics if r['stage']==stage and r['kind']=='best');mean=next(r for r in metrics if r['stage']==stage and r['kind']=='mean')
        differences.append(dict(stage=stage,mean_minus_best_requested_coverage=mean['requested_coverage_final']-best['requested_coverage_final'],
            mean_minus_best_model_cost=mean['predicted_cost_by_horizon'][-1]-best['predicted_cost_by_horizon'][-1],mean_minus_best_xy_px=mean['goal_xy_distance_by_horizon'][-1]-best['goal_xy_distance_by_horizon'][-1]))
    report=dict(complete=True,episode=episode,seconds=time.monotonic()-started,cem_seconds=cem_seconds,identity_seconds=identity_seconds,identity_full_cem_exact=args.identity or resume,
        metric_rows=metrics,stage_differences=differences,iterations=iterations,all_physics_replays_exact=True,native_num_act_stepped=6,
        resumed_prior_cem=resume,resumed_cem_sha256=args.resume_cem_sha256 if resume else None,
        forecast_semantics='Actual original native best-candidate B300 row and updated-mean B1 outputs; no fresh reforecast or reranking',
        native_executed_raw_steps=30,full_H6_equals_native_first_plan_execution=True,independent_states=1,held_access=False,
        source_sha256=source['sha256'],reset_sha256=rr['sha256'],frozen_planner_seed=seed)
    write(args.output/f'episode-{episode:03d}.json',report)
    print(json.dumps(dict(event='state_complete',episode=episode,seconds=report['seconds'],differences=differences)),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('inputs','prepared-inputs','repo','checkpoint','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--episodes',type=int,nargs='+',required=True);p.add_argument('--identity',action='store_true');p.add_argument('--max-seconds',type=float,default=300)
    p.add_argument('--resume-cem',type=Path);p.add_argument('--resume-cem-sha256')
    a=p.parse_args();started=time.monotonic();rows=json.loads(a.inputs.read_text())
    if len(set(a.episodes))!=len(a.episodes) or any(e not in range(4) for e in a.episodes):raise ValueError('Predetermined four seen states only')
    if a.identity and len(a.episodes)!=1:raise ValueError('Identity firststate only')
    if sha(a.checkpoint)!=CHECKPOINT:raise ValueError('Frozen checkpoint hash')
    a.output.mkdir(parents=True,exist_ok=False)
    try:
        write(a.output/'PROTOCOL.json',dict(episodes=a.episodes,all_predetermined_states=[0,1,2,3],horizon=6,samples=300,iterations=30,elites=10,stages=STAGES,
            comparison='CURRENT lowest model-cost candidate vs updated elite MEAN; no physical outcome used in selection',
            primary='Stage30 fullH6 requested-goalcoverage mean-minus-best, paired by four seen states',secondary='Stages1/15; native cost,4DXY,7Dstate,forecast errors,native success/reward',
            native_execution='num_act_stepped6,5rawactions/modelstep =30raw steps; equals H6 here, not closed-loop efficacy',
            identity_full_cem=a.identity,resume_cem_sha256=a.resume_cem_sha256,budget_seconds=a.max_seconds,script_sha256=sha(__file__),checkpoint_sha256=CHECKPOINT,held_access=False,
            paper='https://proceedings.mlr.press/v120/bharadhwaj20a/bharadhwaj20a.pdf; Algorithm1 returns highest-model-return plan, not evidence mean is bad in this model'))
        from collect_pusht_bank import config
        from model_loader import load_headless
        from capture_pusht_calibration import parameter_sha
        torch.set_num_threads(2);torch.manual_seed(90505);cfg=config(a.repo,a.output)
        wm,prep,provenance=load_headless(a.repo,model_name='jepa_wm_pusht',checkpoint_override=a.checkpoint);wm.eval().requires_grad_(False);before=parameter_sha(wm)
        reports=[]
        for e in a.episodes:
            if time.monotonic()-started>a.max_seconds:raise ValueError('Aggregate process cap before next state')
            report=run_state(a,next(r for r in rows if r['episode']==e),wm,prep,cfg);reports.append(report)
        if parameter_sha(wm)!=before:raise ValueError('Frozen parameters changed')
        outputs=[dict(path=q.name,sha256=sha(q),bytes=q.stat().st_size) for q in sorted(a.output.iterdir()) if q.is_file()]
        write(a.output/'DONE.json',dict(complete=True,episodes=a.episodes,seconds=time.monotonic()-started,outputs=outputs,parameter_sha256=before,provenance=provenance,
            full_cem_runs=sum(0 if r['resumed_prior_cem'] else (2 if a.identity else 1) for r in reports),held_access=False))
    except Exception:
        write(a.output/'FAILED.json',dict(complete=False,seconds=time.monotonic()-started,exception=traceback.format_exc()));raise


if __name__=='__main__':main()
