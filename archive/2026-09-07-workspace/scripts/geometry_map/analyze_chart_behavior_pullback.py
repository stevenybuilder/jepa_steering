#!/usr/bin/env python3
"""CPU-only, post-hoc decomposition of immutable chart pilot outputs."""
import argparse
import json
from pathlib import Path
import torch
from run_behavior_pullback import sha,write


def coordinate_drift(delta64,control,basis4,scale4):
    requested=.5*scale4*torch.tanh(control)
    actual=delta64@basis4.T
    residual=actual-requested[:,None]
    return dict(mean_actual_shift_norm=float(actual.norm(dim=-1).mean()),mean_requested_shift_norm=float(requested.norm(dim=-1).mean()),
        mean_coordinate_drift_norm=float(residual.norm(dim=-1).mean()),max_coordinate_drift_norm=float(residual.norm(dim=-1).max()),
        relative_coordinate_drift=float(residual.norm()/requested[:,None].expand_as(actual).norm().clamp_min(1e-20)),
        mean_standardized_coordinate_drift=float((residual/scale4).norm(dim=-1).mean()))


def analyze(a):
    torch.set_num_threads(1);a.output.mkdir(parents=True,exist_ok=False)
    receipt=json.loads((a.results/'DONE.json').read_text())
    if not receipt['complete'] or receipt['episode']!=a.episode:raise ValueError('Complete assigned source required')
    for row in receipt['outputs']:
        if sha(a.results/row['path'])!=row['sha256']:raise ValueError('Source SHA before read')
    data=torch.load(a.results/'results.pt',map_location='cpu',weights_only=False)
    report=data['report'];central={r['method']:r for r in report['rows'] if r['waypoint']=='central'};native=central['native']
    inputs=json.loads((a.inputs/'DONE.json').read_text());row=inputs['outputs'][0]
    if not inputs['complete'] or inputs['episode']!=a.episode or inputs['split']!='development_external':raise ValueError('Development before load')
    if sha(a.inputs/row['path'])!=row['sha256']:raise ValueError('Source input SHA')
    b=torch.load(a.inputs/row['path'],map_location='cpu',weights_only=False)
    states=b['physical_states'].double();goal=torch.as_tensor(b['goal_state']).double()
    block_distance=(states[:,-1,2:4]-goal[2:4]).norm(dim=-1)
    angle=states[:,-1,4]-goal[4];angle=torch.atan2(angle.sin(),angle.cos()).abs()
    maps={}
    for method,record in data['maps'].items():
        path_rows=[r for r in report['rows'] if r['method']==method and isinstance(r['waypoint'],int)]
        primary=central[method];sham=central[method+'_norm_sham']
        drift=coordinate_drift(record['delta64'].double(),record['path'].double(),data['chart_state']['basis'],data['chart_state']['scale'])
        maps[method]=dict(mean_target_hellinger2=sum(r['target_hellinger2'] for r in path_rows)/len(path_rows),
            target_improvement_fraction=1-sum(r['target_hellinger2'] for r in path_rows)/len(path_rows)/report['baseline_target_hellinger2'],
            selected_progress_delta_px=primary['selected_actual_progress_px']-native['selected_actual_progress_px'],
            sham_selected_progress_delta_px=sham['selected_actual_progress_px']-native['selected_actual_progress_px'],
            selected_action=primary['selected_action_index'],sham_selected_action=sham['selected_action_index'],
            cost_vs_physical_spearman=primary['cost_vs_physical_distance_spearman'],sham_spearman=sham['cost_vs_physical_distance_spearman'],
            actual_visual_MSE_H1_H6=primary['actual_visual_MSE_H1_H6'],sham_actual_visual_MSE_H1_H6=sham['actual_visual_MSE_H1_H6'],
            selected_block_distance_px=float(block_distance[primary['selected_action_index']]),selected_wrapped_angle_error_rad=float(angle[primary['selected_action_index']]),
            coordinate_drift=drift,central_raw_norm=primary['realized_raw_norm'],sham_raw_norm=sham['realized_raw_norm'],
            requested_sham_norm_max_error=record['requested_sham_norm_max_error'],realized_sham_norm_max_error=record['realized_sham_norm_max_error'],
            baseline_support=report['baseline_support'],central_actual_edited_support=primary['actual_edited_coordinate_support'],
            central_requested_support=primary['requested_coordinate_support'],endpoint_raw_shift_norm=record['endpoints_delta64_norm'],
            stop=record['stop'],seconds=record['seconds'],accepted_outersteps=len(record['history']),canary=report['canaries'][method])
    result=dict(complete=True,episode=a.episode,source_done_sha256=sha(a.results/'DONE.json'),source_files=receipt['outputs'],
        native=dict(selected_action=native['selected_action_index'],cost_vs_physical_spearman=native['cost_vs_physical_distance_spearman'],
            actual_visual_MSE_H1_H6=native['actual_visual_MSE_H1_H6'],target_hellinger2=report['baseline_target_hellinger2'],
            selected_block_distance_px=float(block_distance[native['selected_action_index']]),selected_wrapped_angle_error_rad=float(angle[native['selected_action_index']])),
        maps=maps,seconds=report['seconds'],candidate_promoted=False,limitations=report['limitations']+[
        'Combined agent+blockXY progress is diagnostic, not native Push success or goal-aligned polygon coverage',
        'Coordinate drift calculated as delta64@PCA4basis minus requested shift; no source-recipient numerical approximation',
        'Charts fit separate model-only donor samples within each seen state; no out-of-state generalization claim'])
    write(a.output/'analysis.json',result)
    write(a.output/'DONE.json',dict(complete=True,episode=a.episode,outputs=[dict(path='analysis.json',sha256=sha(a.output/'analysis.json'),bytes=(a.output/'analysis.json').stat().st_size)]))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episode',type=int,choices=range(4),required=True)
    for key in ('results','inputs','output'):p.add_argument('--'+key,type=Path,required=True)
    analyze(p.parse_args())
