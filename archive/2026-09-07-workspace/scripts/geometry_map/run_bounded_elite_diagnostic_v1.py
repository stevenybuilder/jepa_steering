#!/usr/bin/env python3
"""Six short native CEM runs: projected model, nominal versus bounded elite fit."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from protocol import file_sha256,write_json_atomic
from run_xyz_projected_planner_v1 import project_actions,unroll_with_projection
from run_coordinate_dose_timing import exact,clone_prediction,encode_actual,latent_errors
from run_planner_compensation_v1 import restore_fixed_encoded_goal,compare_replay,CHECKPOINT_SHA


def canonicalize_scratch(actions,effective,enabled):
    if not enabled:return actions
    if actions.requires_grad or actions._base is not None or actions.storage_offset()!=0 or not actions.is_contiguous():
        raise ValueError('Only owned contiguous native CEM scratch may be modified')
    if actions.shape!=effective.shape:raise ValueError('Scratch/effective shape mismatch')
    actions.copy_(effective)
    return actions


class CEMTrace:
    def __init__(self,planner,preprocessor,bounded):
        self.planner,self.preprocessor,self.bounded=planner,preprocessor,bounded
        self.native_cost,self.native_unroll=planner.cost_function,planner.unroll
        self.rows=[];self.means=[];self.in_cost=False;self.last_prediction=None
        self.expected_mean=None;self.parameter_std=None
    def __enter__(self):
        self.planner.cost_function=self.cost;self.planner.unroll=self.unroll
        return self
    def __exit__(self,*args):
        self.planner.cost_function=self.native_cost;self.planner.unroll=self.native_unroll
    def unroll(self,*args,**kwargs):
        actions=kwargs.get('act_suffix',args[1] if len(args)>1 else None)
        if actions is None:raise RuntimeError('Native unroll action binding absent')
        # Bounded samples are already projected in cost(); elite means remain
        # in the convex box. Model-only arm projects a copy, never fit scratch.
        answer=unroll_with_projection(self.native_unroll,self.preprocessor,not self.bounded,*args,**kwargs)
        if not self.in_cost:
            if actions.shape[1]!=1:raise RuntimeError('Expected one native updated mean')
            self.means.append({'actions':actions.cpu().clone(),'projected_objective_cost':float(self.planner.objective(answer,actions)[0])})
            self.last_prediction=clone_prediction(answer)
        return answer
    def cost(self,actions,z):
        if actions.shape[1]!=300:raise RuntimeError('Native300 sample budget changed')
        nominal=actions.clone();effective,stats=project_actions(nominal,self.preprocessor)
        before_mean=nominal[:,0].clone()
        if self.expected_mean is not None:exact(before_mean,self.expected_mean,'native mean evolution reconstruction')
        if self.parameter_std is None:self.parameter_std=torch.ones_like(before_mean)*self.planner.var_scale
        std_before=self.parameter_std.clone()
        canonicalize_scratch(actions,effective,self.bounded)
        self.in_cost=True
        try:cost=self.native_cost(actions,z)
        finally:self.in_cost=False
        elites=torch.topk(-cost,self.planner.num_elites).indices
        fit_actions=actions[:,elites]
        self.expected_mean=fit_actions.mean(1)*(1-self.planner.momentum_mean)+before_mean*self.planner.momentum_mean
        self.parameter_std=fit_actions.std(1)*(1-self.planner.momentum_std)+std_before*self.planner.momentum_std
        raw=self.preprocessor.denormalize_actions(nominal.cpu().reshape(-1,4))
        eraw=self.preprocessor.denormalize_actions(nominal[:,elites].cpu().reshape(-1,4))
        flat=effective.cpu().permute(1,0,2).reshape(300,-1).numpy()
        record={'iteration':len(self.rows),'nominal_actions':nominal.cpu(),'effective_model_actions':effective.cpu(),
                'costs':cost.cpu(),'elite_indices':elites.cpu(),'mean_before':before_mean.cpu(),'std_before':std_before.cpu(),
                'mean_after':self.expected_mean.cpu(),'std_after':self.parameter_std.cpu(),
                'generator_state_after_draw':self.planner.local_generator.get_state().cpu().clone(),
                'nominal_clipped_xyz_fraction':stats['xyz_clipped']/stats['xyz_total'],
                'elite_nominal_clipped_xyz_fraction':float((eraw[:,:3].abs()>1).float().mean()),
                'unique_effective_proposals':int(len(np.unique(flat,axis=0))),
                'mean_unique_effective_values_per_coordinate':float(np.mean([len(np.unique(flat[:,i])) for i in range(flat.shape[1])])),
                'mean_std_before':float(std_before.mean()),'mean_std_after':float(self.parameter_std.mean()),
                'nominal_mean_l2':float(before_mean.norm()),'new_mean_l2':float(self.expected_mean.norm()),
                'minimum_candidate_cost':float(cost.min()),'mean_elite_cost':float(cost[elites].mean())}
        self.rows.append(record)
        return cost
    def summary(self):
        exclude={'nominal_actions','effective_model_actions','costs','elite_indices','mean_before','std_before','mean_after','std_after','generator_state_after_draw'}
        if len(self.rows)!=15 or len(self.means)!=15:raise RuntimeError('Native15 CEM iterations required')
        return [{k:v for k,v in row.items() if k not in exclude}|{'updated_mean_cost':mean['projected_objective_cost']} for row,mean in zip(self.rows,self.means)]


@torch.no_grad()
def run_arm(cfg,wm,preprocessor,bank,prior,bounded,paired):
    from run_coordinate_steering_pilot import fresh_start,paired_start_checks,close_env
    from scipy.stats import rankdata
    started=time.monotonic();env,agent,td,snapshot,mode=fresh_start(cfg,wm,preprocessor,bank)
    try:
        mode=mode|restore_fixed_encoded_goal(agent,snapshot,prior['initial_snapshot'])
        checks=paired_start_checks(snapshot,prior['initial_snapshot'])
        planner=agent.planner
        if planner.max_norms is not None or planner.distribute_planner or planner.num_elites!=10 or planner.iterations!=15:raise RuntimeError('Native CEM configuration differs')
        z=wm.encode(td.cuda().unsqueeze(0),act=True)
        with CEMTrace(planner,preprocessor,bounded) as trace:
            planned=agent.plan(z.clone(),steps_left=bank['replans'][0]['steps_left_model'])
            full=planner._prev_mean.cpu().clone()
        summaries=trace.summary()
        exact(trace.rows[0]['nominal_actions'],prior['first_candidates']['actions'],'original first nominal300')
        if paired is None:
            compare_replay(trace.rows[0]['costs'],prior['first_candidates']['costs'],'uninstrumented projected first300 costs')
            exact(torch.as_tensor(rankdata(trace.rows[0]['costs'].numpy())),prior['first_candidates']['ranks'],'uninstrumented first300 ranks')
            exact(full,prior['plans'][0],'uninstrumented projected selected H6 plan')
        else:
            exact(trace.rows[0]['effective_model_actions'],paired['iterations'][0]['effective_model_actions'],'same first300 effective model inputs')
            exact(trace.rows[0]['costs'],paired['iterations'][0]['costs'],'same first300 projected costs')
            for row,old in zip(trace.rows,paired['iterations']):exact(row['generator_state_after_draw'],old['generator_state_after_draw'],'common underlying Gaussian RNG each iteration')
        raw=preprocessor.denormalize_actions(planned.cpu().reshape(15,4))
        observations,rewards,dones,infos=env.step_multiple(raw)
        if len(observations)!=15:raise RuntimeError('Exact15 physical steps required')
        frames=torch.stack(observations).cpu();states=np.stack([np.asarray(i['state']) for i in infos]);proprios=torch.stack([torch.as_tensor(i['proprio']) for i in infos]).cpu()
        if paired is None:
            exact(frames,prior['frames'],'instrumentation zero-op projected15 pixels');exact(states,prior['states'],'instrumentation zero-op projected15 states')
        actual=encode_actual(wm,frames[4::5],proprios[4::5])
        goal=np.asarray(env.proprio_env.unwrapped._target_pos).copy();initial=snapshot['proprio'].numpy().reshape(-1,4)[-1,:3]
        finalcost=trace.means[-1]['projected_objective_cost'];best=min(r['minimum_candidate_cost'] for r in trace.rows)
        bestrow=min(trace.rows,key=lambda r:r['minimum_candidate_cost']);bestindex=int(bestrow['costs'].argmin())
        return {'complete':True,'episode':bank['episode'],'bounded_elite_fit':bounded,'initial_snapshot':snapshot,'paired_start_checks':checks,'goal_mode':mode,
                'iterations':trace.rows,'updated_means':trace.means,'iteration_summary':summaries,
                'selected_plan_fullH6':full,'selected_prediction':trace.last_prediction,
                'best_candidate_nominal_plan':bestrow['nominal_actions'][:,bestindex],'best_candidate_effective_plan':bestrow['effective_model_actions'][:,bestindex],
                'best_candidate_scope':'Cost-only diagnostic; no best-candidate physical execution or outcome selection',
                'selected_projected_goal_cost_h6':finalcost,'best_seen_projected_candidate_cost':best,'returned_mean_minus_best_seen_cost':finalcost-best,
                'states':states,'frames':frames,'proprios':proprios,'raw_actions':raw,'rewards':np.asarray(rewards),
                'goal_progress_m':float(np.linalg.norm(initial-goal)-np.linalg.norm(states[-1,:3]-goal)),
                'final_goal_distance_m':float(np.linalg.norm(states[-1,:3]-goal)),
                'ever_success':any(bool(i['success']) for i in infos),'final_success':bool(infos[-1]['success']),
                'selected_visual_mse_to_actual_h1_to_h3':latent_errors(trace.last_prediction,actual),'seconds':time.monotonic()-started}
    finally:close_env(env)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('repo','bank','projection-dir','output-dir'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);started=time.monotonic();outputs=[]
    from causal_planner_forks import load_development_bank,setup_cfg
    from model_loader import load_headless_metaworld
    d=json.loads((a.projection_dir/'DONE.json').read_text());entry=next(r for r in d['outputs'] if r['stage']=='projected15')
    source=a.projection_dir/entry['path']
    if not d['complete'] or d['held_sources_opened'] or file_sha256(source)!=entry['sha256']:raise RuntimeError('Completed frozen projected source required')
    prior=torch.load(source,map_location='cpu',weights_only=False);bank,banksha=load_development_bank(a.bank)
    if bank['episode'] not in (0,4,7) or bank['episode']!=prior['episode']:raise RuntimeError('Fixed three development starts only')
    protocol={'episode':bank['episode'],'bank_sha256':banksha,'source_projected15':entry,'source_root':str(a.projection_dir),
              'arms':['nominal_elite_fit','bounded_elite_fit'],'native_cem':{'H':6,'samples':300,'iterations':15,'elites':10},'raw_steps':15,
              'change':'Only scratch candidate action coordinates passed to native elite mean/std fit, exact official XYZ projection; gripper unchanged',
              'common_randomness':'Same original initial nominal300 and effective model inputs; same native Gaussian generator state after every iteration',
              'metric_separation':'Projected objective gain/regret is separate from actual physical goal progress; no full episode efficacy',
              'held_sources_opened':False,'budget_seconds':600,'script_sha256':file_sha256(Path(__file__)),'checkpoint_sha256':CHECKPOINT_SHA}
    write_json_atomic(a.output_dir/'protocol.json',protocol)
    try:
        wm,preprocessor,provenance=load_headless_metaworld(a.repo);wm.eval().requires_grad_(False)
        if file_sha256(Path(provenance['checkpoint']))!=CHECKPOINT_SHA:raise RuntimeError('Frozen model checkpoint changed')
        cfg=setup_cfg(Path(provenance['config']),a.output_dir,wm);paired=None
        for bounded in (False,True):
            if time.monotonic()-started>600:raise RuntimeError('Frozen short diagnostic budget elapsed')
            name='bounded_elite_fit' if bounded else 'nominal_elite_fit';print(json.dumps({'event':'bounded_elite_cem_started','episode':bank['episode'],'arm':name}),flush=True)
            value=run_arm(cfg,wm,preprocessor,bank,prior,bounded,paired)
            if not bounded:paired=value
            path=a.output_dir/(name+'.pt');torch.save(value,path)
            row={'path':path.name,'sha256':file_sha256(path),'bytes':path.stat().st_size,'episode':bank['episode'],'arm':name,
                 **{k:value[k] for k in ('selected_projected_goal_cost_h6','best_seen_projected_candidate_cost','returned_mean_minus_best_seen_cost','goal_progress_m','final_goal_distance_m','ever_success','final_success','selected_visual_mse_to_actual_h1_to_h3','seconds','iteration_summary')},
                 'selected_normalized_H6_plan':value['selected_plan_fullH6'].tolist(),'selected_raw15_actions':value['raw_actions'].tolist()}
            outputs.append(row);write_json_atomic(a.output_dir/'progress.json',{'outputs':outputs});print(json.dumps({'event':'bounded_elite_arm_complete',**row}),flush=True)
        write_json_atomic(a.output_dir/'DONE.json',protocol|{'complete':True,'outputs':outputs,'seconds':time.monotonic()-started,'model':provenance})
    except Exception as exc:
        write_json_atomic(a.output_dir/'FAILED.json',{'error':repr(exc),'outputs':outputs});raise


if __name__=='__main__':main()
