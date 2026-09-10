#!/usr/bin/env python3
"""Equal-state readback of the frozen four-state response-localization pilot."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, median


def aggregate(reports):
    reports=sorted(reports,key=lambda x:x['episode'])
    if [r['episode'] for r in reports]!=[0,1,2,3]:raise ValueError('All four predetermined states required')
    if not all(r['complete'] and r['all_identity_guards_exact'] for r in reports):raise ValueError('Incomplete/failed guard')
    indexed=[{a['arm']:a for a in r['score']} for r in reports]
    arms={}
    for name in indexed[0]:
        selected=[a[name] for a in indexed]
        gains=[a['goal_coverage_delta'] for a in selected]
        errors={k:[[100*(b['native']['equal_candidate_mse_by_horizon'][k][h]-b[name]['equal_candidate_mse_by_horizon'][k][h])/b['native']['equal_candidate_mse_by_horizon'][k][h] for h in range(6)] for b in indexed] for k in ('visual','proprio')}
        entry=dict(choices=[a['choice'] for a in selected],coverage_gain_by_state=gains,mean_coverage_gain=mean(gains),positive_states=sum(x>0 for x in gains),
            selected_coverage_by_state=[a['selected_goal_coverage'] for a in selected],selected_xy_change_px_by_state=[a['xy_delta_px'] for a in selected],
            forecast_error_reduction_percent_by_state_horizon=errors,
            equal_state_forecast_error_reduction_percent_by_horizon={k:[mean(e[h] for e in v) for h in range(6)] for k,v in errors.items()})
        if name!='native':
            entry['edit_by_state']=[dict(delivered_l2_median_noncentral=median(a['edit']['delivered_l2'][1:]),relative_l2_median_noncentral=median(a['edit']['raw_relative_l2'][1:]),
                max_delivered_norm_mismatch=max(a['edit']['delivered_norm_mismatch']),max_off_subspace_l2=max(a['edit']['off_subspace_l2']),max_rounding_l2=max(a['edit']['rounding_l2']),
                nearest_other_natural_distance_ratio=mean(x/y for x,y in zip(r['support'][name][1:],r['support']['native_nearest_other'][1:]))) for a,r in zip(selected,reports)]
        arms[name]=entry
    gate={}
    for sign in ('minus','plus'):
        target=arms['targeted_'+sign];controls=['random_basis_'+sign,'random_patches_'+sign,'random_both_'+sign]
        gate[sign]=dict(mean_gain_at_least_one_percentage_point=target['mean_coverage_gain']>=.01,positive_at_least_three_states=target['positive_states']>=3,
            beats_every_matched_control_mean=all(target['mean_coverage_gain']>arms[c]['mean_coverage_gain'] for c in controls))
        gate[sign]['exploratory_lead']=all(gate[sign].values())
    return dict(complete=True,independent_seen_development_states=4,candidates_per_state=64,arms=arms,exploratory_gate=gate,
        geometry_by_state=[dict(episode=r['episode'],retained_centered_variance=r['geometry']['retained_centered_variance'],projected_contrast_energy_fraction=r['geometry']['projected_contrast_energy_fraction'],
            selected_projected_contrast_energy_fraction=r['geometry']['selected_projected_contrast_energy_fraction'],random_basis_overlap=r['geometry']['random_basis_overlap']) for r in reports],
        original_native_coverage_headroom_by_state=[r['goal_coverage_headroom'] for r in reports],
        source_full_tensors=[dict(episode=r['episode'],sha256=r['full_tensor_sha256'],bytes=r['full_tensor_bytes']) for r in reports],
        limitations='Same four seen states and dependent fixed64 candidate banks; unchanged choice is not a continuous-CEM impossibility result. Rank4 response localization, not intrinsic-manifold steering. Action-history/consequence input, not isolated current action. No held data or new simulator calls.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    paths=sorted(a.root.glob('worker-*/episode-???.json'));reports=[json.loads(q.read_text()) for q in paths]
    # The compact source receipts bind each report to the just-completed remote source.
    for q in paths:
        receipt=json.loads((q.parent/'DONE.json').read_text());entry=next(r for r in receipt['outputs'] if r['path']==q.name)
        if hashlib.sha256(q.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Compact report SHA mismatch')
    result=aggregate(reports);done=[json.loads(q.read_text()) for q in a.root.glob('worker-*/DONE.json')]
    result['aggregate_process_seconds']=sum(d['seconds'] for d in done);result['aggregate_process_gpu_hours']=result['aggregate_process_seconds']/3600
    result['all_final_parameter_hashes_match']=len(set(d['parameter_sha256'] for d in done))==1
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(output=str(a.output),sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),seconds=result['aggregate_process_seconds'],gate=result['exploratory_gate'])))


if __name__=='__main__':main()
