#!/usr/bin/env python3
"""Episode-equal descriptive aggregation of the frozen four-state density pilot."""
import argparse,hashlib,json,statistics
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    source=[]
    for path in a.root.glob('worker-*/verified-v2*/near-dev-*.json'):
        d=json.loads(path.read_text());verified=json.loads((path.parent/'REMOTE_BACKUP_VERIFIED.json').read_text())
        if not d['complete'] or not verified['source_and_backup_sha_verified']:raise ValueError('Unverified source')
        source.append((path,d,verified))
    source.sort(key=lambda item:item[1]['key'])
    if [d['key'] for _,d,_ in source]!=[f'near-dev-{i:03}' for i in range(4)]:raise ValueError('Exactly four original states')
    states=[];arms={};traversal=[]
    for path,d,verified in source:
        by={r['arm']:r for r in d['rows']};base=by['native_unhooked'];local=by['pca64_local_chord']
        state={'key':d['key'],'source':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'backup_receipt':str(path.parent/'REMOTE_BACKUP_VERIFIED.json'),'seconds':d['seconds'],
            'native_choice':base['selected_action_index'],'arc_over_chord_range':[min(sum(d['arc_over_chord'],[])),max(sum(d['arc_over_chord'],[]))]}
        states.append(state)
        for name,r in by.items():
            item=dict(state=d['key'],spearman=r['cost_vs_physical_distance_spearman'],choice=r['selected_action_index'],
                progress_delta_native_px=r['selected_actual_progress_px']-base['selected_actual_progress_px'],
                H6_actual_mse=r['mean_actual_visual_mse_by_horizon'][-1],
                H6_actual_gain_native_percent=100*(base['mean_actual_visual_mse_by_horizon'][-1]-r['mean_actual_visual_mse_by_horizon'][-1])/base['mean_actual_visual_mse_by_horizon'][-1],
                H6_actual_gain_local_percent=100*(local['mean_actual_visual_mse_by_horizon'][-1]-r['mean_actual_visual_mse_by_horizon'][-1])/base['mean_actual_visual_mse_by_horizon'][-1])
            arms.setdefault(name,[]).append(item)
        for coordinate in (-3.,-1.,0.,1.,3.):
            for method in ('endpoint_chord','local_linear','natural_spline','spline_norm_sham'):
                rows=[r for r in d['traversal'] if r['coordinate']==coordinate and r['method']==method]
                if len(rows)!=4:raise ValueError('Missing fixed direction traversal')
                traversal.append(dict(state=d['key'],coordinate=coordinate,method=method,
                    mean_direction_spearman=statistics.mean(r['cost_vs_physical_distance_spearman'] for r in rows),
                    mean_direction_progress_delta_native_px=statistics.mean(r['selected_actual_progress_px']-base['selected_actual_progress_px'] for r in rows),
                    mean_direction_H6_actual_mse=statistics.mean(r['mean_actual_visual_mse_by_horizon'][-1] for r in rows)))
    result=dict(complete=True,independent_development_states=4,physical_plans_per_state=9,interior_forward_conditions=320,
        full_outputs_source_and_distinct_backup_verified=True,source_output_bytes=sum(v['bytes'] for _,_,v in source),
        source_output_files=sum(v['file_count'] for _,_,v in source),states=states,midpoint_arms=arms,
        episode_equal_means={name:{'spearman':statistics.mean(r['spearman'] for r in rows),
            'selected_progress_delta_native_px':statistics.mean(r['progress_delta_native_px'] for r in rows),
            'H6_actual_gain_native_percent':statistics.mean(r['H6_actual_gain_native_percent'] for r in rows),
            'H6_actual_gain_local_percent':statistics.mean(r['H6_actual_gain_local_percent'] for r in rows)} for name,rows in arms.items()},
        interior_by_state_coordinate=traversal,
        interpretation='No midpoint spline physical-selection improvement; natural spline and local-linear differ geometrically without robust superiority on actual forecast accuracy.',
        limitations=['Known local raw-action amplitude, not learned global physical intrinsic coordinates',
            '150-interval numerical arc length on each fitted 1D curve, not exact multidimensional geodesic',
            'PCA64 and off-subspace preservation in BOTH spline and chord arms; float32 reconstruction rounding recorded',
            'No formal behavior-space pullback in this pilot; separately owned follow-up',
            'Interior edits retain original actions, so physical truth remains that original action sequence',
            'Four adaptively explored development states, not unseen confirmation or full episode task efficacy'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as file:json.dump(result,file,indent=2);file.write('\n')
    print(json.dumps(dict(output=str(a.output),sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),states=4,
        output_bytes=result['source_output_bytes'],midpoint=result['episode_equal_means'])))


if __name__=='__main__':main()
