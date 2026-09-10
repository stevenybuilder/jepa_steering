#!/usr/bin/env python3
"""CPU-only paired optimizer diagnostics with source hash verification."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from protocol import file_sha256,write_json_atomic
from capture_plan_feasibility_v1 import MEAN,STD


def roundtrip_gripper_error(nominal,effective):
    mean=torch.tensor(MEAN);std=torch.tensor(STD)
    raw=nominal.float().cpu().reshape(-1,4)*std+mean
    effective_raw=effective.float().cpu().reshape(-1,4)*std+mean
    return float((raw[:,3]-effective_raw[:,3]).abs().max())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    receipt=json.loads((a.input_dir/'DONE.json').read_text());values={};rows=[]
    if not receipt['complete'] or receipt['held_sources_opened'] or receipt['episode'] not in (0,4,7):raise ValueError('Frozen completed development source required')
    for entry in receipt['outputs']:
        path=a.input_dir/entry['path']
        if file_sha256(path)!=entry['sha256']:raise ValueError('Source hash mismatch')
        v=torch.load(path,map_location='cpu',weights_only=False);values[entry['arm']]=v
        rows.append(entry|{'gripper_affine_roundtrip_raw_maxabs_each_iteration':[roundtrip_gripper_error(r['nominal_actions'],r['effective_model_actions']) for r in v['iterations']],
                           'raw_hand_xyz_trajectory':v['states'][:,:3].tolist(),'actual_rewards':np.asarray(v['rewards']).tolist(),
                           'gaussian_draw_state_hashes':[__import__('hashlib').sha256(r['generator_state_after_draw'].numpy().tobytes()).hexdigest() for r in v['iterations']]})
    n=values['nominal_elite_fit'];b=values['bounded_elite_fit']
    for key in ('nominal_actions','effective_model_actions','costs'):
        if not torch.equal(n['iterations'][0][key],b['iterations'][0][key]):raise ValueError('First input/cost parity lost: '+key)
    for nr,br in zip(n['iterations'],b['iterations']):
        if not torch.equal(nr['generator_state_after_draw'],br['generator_state_after_draw']):raise ValueError('Common Gaussian RNG parity lost')
    out={'complete':True,'episode':receipt['episode'],'rows':rows,'sources':receipt['outputs'],'source_root':str(a.input_dir),
         'source_receipt_sha256':file_sha256(a.input_dir/'DONE.json'),'common_gaussian_states_all15_exact':True,'first_nominal_effective_costs_exact':True,
         'comparison':{'projected_objective_change_bounded_minus_nominal':b['selected_projected_goal_cost_h6']-n['selected_projected_goal_cost_h6'],
                       'physical_progress_change_bounded_minus_nominal_mm':1000*(b['goal_progress_m']-n['goal_progress_m']),
                       'best_seen_projected_candidate_cost_change_bounded_minus_nominal':b['best_seen_projected_candidate_cost']-n['best_seen_projected_candidate_cost'],
                       'nominal_returned_mean_minus_best_seen_cost':n['returned_mean_minus_best_seen_cost'],
                       'bounded_returned_mean_minus_best_seen_cost':b['returned_mean_minus_best_seen_cost']},
         'scope':'Two native300x15 optimizer variants with shared Gaussian draws. Only15 actual steps observed; native cost terminalH6. No formal objective misranking claim from this horizon mismatch, no full99 extension, semantic edit, held access, best-candidate physical execution or outcome selection. Many clipped coordinates do not imply duplicate entire120D candidate plans.'}
    write_json_atomic(a.output_dir/'summary.json',out)
    write_json_atomic(a.output_dir/'DONE.json',{'complete':True,'outputs':[{'path':'summary.json','sha256':file_sha256(a.output_dir/'summary.json')}],
                      'script_sha256':file_sha256(Path(__file__))})
    print(json.dumps({'episode':out['episode'],'comparison':out['comparison'],'gripper_max':[max(r['gripper_affine_roundtrip_raw_maxabs_each_iteration']) for r in rows],
                      'iteration0_and14':[{'arm':r['arm'],'first':r['iteration_summary'][0],'last':r['iteration_summary'][-1]} for r in rows]}),flush=True)


if __name__=='__main__':main()
