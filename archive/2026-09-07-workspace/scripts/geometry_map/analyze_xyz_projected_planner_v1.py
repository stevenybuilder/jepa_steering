#!/usr/bin/env python3
"""Compact CPU extraction from immutable projected-planner full tensors."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from protocol import file_sha256,write_json_atomic


def action_metrics(raw):
    a=torch.as_tensor(raw).double().cpu().reshape(-1,4)
    return {'raw_xyz_out_of_bounds_fraction':float((a[:,:3].abs()>1).double().mean()),
            'raw_xyz_l2':float(a[:,:3].norm()),'effective_xyz_l2':float(a[:,:3].clamp(-1,1).norm()),
            'mean_raw_xyz':a[:,:3].mean(0).tolist(),'mean_effective_xyz':a[:,:3].clamp(-1,1).mean(0).tolist(),
            'raw_gripper_out_of_bounds_fraction_not_projected':float((a[:,3].abs()>1).double().mean())}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    receipt=json.loads((a.input_dir/'DONE.json').read_text())
    if not receipt['complete'] or receipt['held_sources_opened'] or receipt['episode'] not in (0,4,7):raise ValueError('Completed fixed development source required')
    values={};rows=[]
    for entry in receipt['outputs']:
        path=a.input_dir/entry['path']
        if file_sha256(path)!=entry['sha256']:raise ValueError('Source SHA mismatch')
        v=torch.load(path,map_location='cpu',weights_only=False);values[entry['stage']]=v
        row=dict(entry)
        row['executed_action_metrics']=action_metrics(v['raw_actions'])
        row['first_selected_normalized_H6_plan']=v['plans'][0].tolist()
        row['first15_selected_raw_actions']=v['raw_actions'][:15].tolist()
        row['per_replan_selected_normalized_plans']=[plan.tolist() for plan in v['plans']]
        row['actual_hand_xyz_trajectory']=np.asarray(v['states'])[:,:3].tolist()
        row['actual_rewards']=v['rewards'];row['actual_successes']=v['successes']
        row['first300_native_argmin']=int(v['first_candidates']['native_costs'].argmin())
        row['first300_active_argmin']=int(v['first_candidates']['costs'].argmin())
        row['first300_native_costs']=v['first_candidates']['native_costs'].tolist()
        row['first300_active_costs']=v['first_candidates']['costs'].tolist()
        rows.append(row)
    if set(values)!={'projected15','native99','projected99'}:raise ValueError('Every frozen stage required')
    native=values['native99'];projected=values['projected99'];short=values['projected15']
    if not torch.equal(native['first_candidates']['actions'],projected['first_candidates']['actions']):raise ValueError('Paired first300 differs')
    if not torch.equal(short['plans'][0],projected['plans'][0]):raise ValueError('Short/full projected first plan differs')
    delta=projected['raw_actions'].double()-native['raw_actions'].double()
    comparison={'final_goal_error_change_projected_minus_native_mm':1000*(projected['final_goal_distance_m']-native['final_goal_distance_m']),
                'total_reward_change_projected_minus_native':projected['total_reward']-native['total_reward'],
                'native_final_success':native['final_success'],'projected_final_success':projected['final_success'],
                'native_ever_success':native['ever_success'],'projected_ever_success':projected['ever_success'],
                'first15_progress_change_projected_minus_original_native_mm':1000*(short['goal_progress_m']-receipt['source_baseline']['goal_progress_m']),
                'all99_raw_action_difference_l2':float(delta.norm()),
                'first15_raw_action_difference_l2':float(delta[:15].norm()),
                'first300_proposals_bitexact':True,'short_full_first_plan_bitexact':True}
    out={'complete':True,'episode':receipt['episode'],'rows':rows,'comparison':comparison,
         'source_receipt_sha256':file_sha256(a.input_dir/'DONE.json'),'source_root':str(a.input_dir),'sources':receipt['outputs'],
         'scope':'Projection input-fidelity control, no semantic intervention. Three fixed development starts, no held evidence; candidates/replans are not independent episodes.'}
    write_json_atomic(a.output_dir/'summary.json',out)
    write_json_atomic(a.output_dir/'DONE.json',{'complete':True,'outputs':[{'path':'summary.json','sha256':file_sha256(a.output_dir/'summary.json')}],
                      'script_sha256':file_sha256(Path(__file__))})
    print(json.dumps({'episode':receipt['episode'],'comparison':comparison}),flush=True)


if __name__=='__main__':main()
