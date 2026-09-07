#!/usr/bin/env python3
"""Read-only native crossed-forecast and local-Jacobian analysis on development."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_crossed_compensation import native_metric_vectors,crossed_decomposition
from protocol import file_sha256,write_json_atomic

PAIRS=(('positive','reference_all'),('positive_sham','reference_all_sham'),
       ('negative','negative_all'),('negative_sham','negative_all_sham'))


def jacobian_alignment(jacobian,bias,selected_action_change,actual_forecast_change):
    """Local inverse-response compatibility; not formal causal mediation."""
    j,b,a,f=[np.asarray(x,dtype=np.float64) for x in (jacobian,bias,selected_action_change,actual_forecast_change)]
    if j.ndim!=2 or j.shape!=(b.size,a.size) or f.shape!=b.shape:raise ValueError('Aligned output-by-action Jacobian required')
    inverse=np.linalg.pinv(j,rcond=1e-6);desired=-inverse@b
    linear=j@a;rowspace=inverse@linear;nullspace=a-rowspace
    s=np.linalg.svd(j,compute_uv=False);b2=float(b@b)
    def cosine(x,y):
        n=float(np.linalg.norm(x)*np.linalg.norm(y));return float(x@y/n) if n>1e-14 else None
    return {'observed_action_vs_minimum_norm_inverse_bias_cosine':cosine(a,desired),
            'rowspace_action_vs_inverse_bias_cosine':cosine(rowspace,desired),
            'linearized_forecast_change_vs_negative_bias_cosine':cosine(linear,-b),
            'actual_forecast_change_vs_negative_bias_cosine':cosine(f,-b),
            'action_nullspace_norm_fraction':float(np.linalg.norm(nullspace)/max(np.linalg.norm(a),1e-14)),
            'rowspace_action_change':rowspace.tolist(),
            'minimum_norm_inverse_bias_action_l2':float(np.linalg.norm(desired)),
            'observed_action_change_l2':float(np.linalg.norm(a)),
            'linearized_bias_cancellation_projection':float(-b@linear/b2) if b2>1e-16 else None,
            'observed_bias_cancellation_projection':float(-b@f/b2) if b2>1e-16 else None,
            'linearization_error_relative_to_actual_action_effect':float(np.linalg.norm(linear-f)/max(np.linalg.norm(f),1e-14)),
            'jacobian_rank_relative_cutoff_1e6':int(np.sum(s>s[0]*1e-6)) if len(s) and s[0]>0 else 0,
            'jacobian_singular_values':s.tolist(),
            'minimum_norm_inverse_bias_action_change':desired.tolist(),
            'linearized_selected_action_forecast_change':linear.tolist(),
            'scope':'One local finite-difference derivative at baseline plan; full-action inverse cosine alone cannot exclude compensation because a large Jacobian nullspace may dominate action norm.'}


def summarize_tensor(value,artifact):
    def array(v):return v.detach().cpu().double().numpy() if hasattr(v,'detach') else np.asarray(v,dtype=float)
    names=value['plan_names'];bi=names.index('unsteered');preds=value['selected_crossed']
    def metric(label,i):
        p=preds[label]['prediction'];return native_metric_vectors(array(p['visual'])[1:,i],array(p['proprio'])[1:,i],.1)
    def observer(label,i):
        pooled=array(preds[label]['prediction']['visual'])[1:,i].reshape(6,-1,384).mean(1)
        return ((pooled-artifact['observer_mean'])/artifact['observer_scale'])@artifact['observer_coef'].T+artifact['observer_intercept']
    goalv=array(value['goal_encoded_visual']).reshape(1,-1)
    goalp=array(value['goal_encoded_proprio']).reshape(1,-1)
    goal=native_metric_vectors(goalv,goalp,.1)[0]
    physical_goal=array(value['physical_goal_xyz']);base_actual=value['actual_by_plan']['unsteered']
    base=metric('unsteered',bi);baseq=observer('unsteered',bi)
    actions=array(value['normalized_actions']);jac={}
    for eps,values in value['jacobians'].items():
        jac[eps]=array(values['observer_xyz']).transpose(0,2,1).reshape(18,120)
    derivative_disagreement=float(np.linalg.norm(jac['0.01']-jac['0.005'])/max(np.linalg.norm(jac['0.005']),1e-14))
    rows=[]
    for label,plan_name in PAIRS:
        i=names.index(plan_name);eb=metric(label,bi);be=metric('unsteered',i);ee=metric(label,i)
        native=crossed_decomposition(base,eb,be,ee,goal)
        qb=observer(label,bi);qn=observer('unsteered',i);qe=observer(label,i)
        decoded=crossed_decomposition(baseq,qb,qn,qe,physical_goal)
        action_delta=(actions[:,i]-actions[:,bi]).reshape(-1)
        align=jacobian_alignment(jac['0.005'],(qb-baseq).reshape(-1),action_delta,(qn-baseq).reshape(-1))
        actual=value['actual_by_plan'][plan_name];actual_xyz=array(actual['states'])[4::5,:3]
        actual_distance=np.linalg.norm(actual_xyz-physical_goal,axis=-1)
        selected_distance=np.linalg.norm(qe-physical_goal,axis=-1)
        unedited_distance=np.linalg.norm(qn-physical_goal,axis=-1)
        rows.append({'episode':value['episode'],'condition':label,'selected_plan':plan_name,
                     'native_crossed':native,'decoded_coordinate_crossed':decoded,'local_action_response':align,
                     'jacobian_relative_epsilon_disagreement':derivative_disagreement,
                     'goal_progress_m':actual['goal_progress_m'],
                     'goal_progress_change_vs_baseline_mm':1000*(actual['goal_progress_m']-base_actual['goal_progress_m']),
                     'ever_success':bool(actual['ever_success']),
                     'selected_edited_observer_goal_error_h1_to_h6_m':selected_distance.tolist(),
                     'selected_unedited_observer_goal_error_h1_to_h6_m':unedited_distance.tolist(),
                     'actual_goal_error_h1_to_h3_m':actual_distance.tolist(),
                     'selected_edited_observer_optimism_h1_to_h3_m':(actual_distance-selected_distance[:3]).tolist(),
                     'selected_unedited_observer_optimism_h1_to_h3_m':(actual_distance-unedited_distance[:3]).tolist(),
                     'raw_command_delta':(array(actual['raw_actions'])-array(base_actual['raw_actions'])).tolist(),
                     'normalized_plan_delta':action_delta.tolist(),
                     'initial_native_cost_rank_change_fraction':float(np.mean(array(value['first300_crossed'][label]['ranks'])!=array(value['first300_crossed']['unsteered']['ranks']))),
                     'goal_mode':value['goal_mode'],
                     'native_metric_cost_reconstruction_max_error':float(np.max(np.abs(np.array([x['edited_model_edited_plan_cost'] for x in native['rows']])-array(preds[label]['costs_by_step'])[1:,i]))),
                     'interpretation':'Crossed forecasts separate representation bias, action selection, and nonlinear interaction; observer error and Jacobian cancellation do not alone establish compensation.'})
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input-dirs',nargs='+',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=False)
    import torch
    torch.set_num_threads(2);artifact=dict(np.load(a.candidate));rows=[];sources=[];episode_ids=[]
    for directory in a.input_dirs:
        receipt=json.loads((directory/'DONE.json').read_text())
        if not receipt['complete'] or receipt['held_sources_opened'] or file_sha256(a.candidate)!=receipt['candidate_sha256']:raise RuntimeError('Frozen completed source required')
        for entry in receipt['outputs']:
            if entry.get('stage')!='crossed_native_forecasts':continue
            path=directory/entry['path']
            if file_sha256(path)!=entry['sha256']:raise RuntimeError('Crossed tensor SHA mismatch')
            value=torch.load(path,map_location='cpu',weights_only=False)
            if value['episode'] not in (0,1,4,7,10):raise RuntimeError('Unexpected development source')
            rows.extend(summarize_tensor(value,artifact));episode_ids.append(value['episode']);sources.append(entry|{'root':str(directory)})
    if len(set(episode_ids))!=len(episode_ids):raise RuntimeError('Duplicate start')
    aggregate={}
    for condition,_ in PAIRS:
        selected=[r for r in rows if r['condition']==condition]
        if not selected:continue
        terminal=[r['native_crossed']['rows'][-1] for r in selected]
        aggregate[condition]={'n_starts':len(selected),'episodes':[r['episode'] for r in selected],
                             'mean_progress_change_vs_baseline_mm':float(np.mean([r['goal_progress_change_vs_baseline_mm'] for r in selected])),
                             'mean_terminal_edited_model_selection_gain':float(np.mean([r['edited_model_selection_gain'] for r in terminal])),
                             'mean_terminal_unedited_model_selection_gain':float(np.mean([r['base_model_selection_gain'] for r in terminal])),
                             'terminal_opposition_count':sum(r['edit_action_dot']<0 for r in terminal),
                             'local_inverse_action_alignment_cosines':[r['local_action_response']['observed_action_vs_minimum_norm_inverse_bias_cosine'] for r in selected],
                             'jacobian_linearization_relative_errors':[r['local_action_response']['linearization_error_relative_to_actual_action_effect'] for r in selected]}
    out={'complete':True,'all_five_starts':set(episode_ids)=={0,1,4,7,10},'episodes':sorted(episode_ids),'rows':rows,'aggregate':aggregate,'sources':sources,
         'interpretation':'No inference of formal mediation or intentional compensation from opposition alone. Full held performance not evaluated; candidates are not independent starts.'}
    write_json_atomic(a.output_dir/'compensation.json',out)
    write_json_atomic(a.output_dir/'DONE.json',{'complete':True,'outputs':[{'path':'compensation.json','sha256':file_sha256(a.output_dir/'compensation.json')}],'script_sha256':file_sha256(Path(__file__))})
    print(json.dumps({'episodes':out['episodes'],'aggregate':aggregate}),flush=True)


if __name__=='__main__':main()
