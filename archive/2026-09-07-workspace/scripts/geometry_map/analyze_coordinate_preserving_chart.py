#!/usr/bin/env python3
"""CPU-only, source-verified summary of the frozen graph decoder ablation."""
import argparse
import json
from pathlib import Path
from run_behavior_pullback import sha,write


def mean(values):return sum(values)/len(values)


def summarize(report):
    native=next(r for r in report['rows'] if r['method']=='native')
    parents={}
    for parent in ('linear_rank4','nonlinear_RBF_rank4'):
        rows=[r for r in report['rows'] if r['control_parent']==parent]
        methods={}
        for name in ('rbf_same_coordinate','graph_same_coordinate','rbf_matched_dose','graph_matched_dose','common_norm_sham'):
            source=[r for r in rows if r['method']==name];central=next(r for r in source if r['waypoint']=='central');path=[r for r in source if isinstance(r['waypoint'],int)]
            methods[name]=dict(mean_target_hellinger2=mean([r['target_hellinger2'] for r in path]),
                mean_path_forecast_H3_MSE=mean([r['actual_visual_MSE_H1_H6'][2] for r in path]),mean_path_forecast_H6_MSE=mean([r['actual_visual_MSE_H1_H6'][5] for r in path]),
                central_forecast_H1_H6_MSE=central['actual_visual_MSE_H1_H6'],central_native_cost=central['native_cost'],
                central_spearman=central['cost_vs_physical_distance_spearman'],central_selected_action=central['selected_action_index'],
                central_selected_progress_delta_px=central['selected_actual_progress_px']-native['selected_actual_progress_px'],
                central_selected_block_distance_px=central['selected_block_distance_px'],central_selected_wrapped_angle_error_rad=central['selected_wrapped_angle_error_rad'],
                central_realized_raw_norm=central['realized_raw_norm'],central_requested_raw_norm=central['requested_raw_norm'],
                central_outside_box_count=sum(central['actual_support']['outside_train_box']),central_nearest_donor_standardized_distance=central['actual_support']['nearest_train_distance'],
                coordinate_error_max=max([max(r.get('coordinate_shift_error_norm',[0])) for r in source]),
                realized_coordinate_error_max=max([max(r.get('native_realized_coordinate_shift_error_norm',[0])) for r in source]),
                maximum_PCA64_complement_error=max(r['off_PCA64_maxabs'] for r in source))
            if name in ('rbf_matched_dose','graph_matched_dose'):
                alphas=[x for r in source for x in r['ray_match']['alpha']]
                methods[name]['ray']=dict(alpha_min=min(alphas),alpha_max=max(alphas),alpha_mean=mean(alphas),
                    observed_multiple_crossings=sum(x>1 for r in source for x in r['ray_match']['observed_grid_crossings']),
                    crossings_max=max(x for r in source for x in r['ray_match']['observed_grid_crossings']),
                    coordinate_norm_match_max_error=max(x for r in source for x in r['ray_match']['coordinate_norm_error']),
                    central_alpha=central['ray_match']['alpha'],recipient_points=len(alphas))
        pair_norm=[]
        for index in list(range(20))+['central']:
            paired={r['method']:r for r in rows if r['waypoint']==index}
            for k in range(9):pair_norm.append(abs(paired['rbf_matched_dose']['realized_raw_norm'][k]-paired['graph_matched_dose']['realized_raw_norm'][k]))
        parents[parent]=dict(methods=methods,matched_pair_realized_norm_error_max=max(pair_norm),
            matched_graph_minus_RBF_H3_MSE=methods['graph_matched_dose']['central_forecast_H1_H6_MSE'][2]-methods['rbf_matched_dose']['central_forecast_H1_H6_MSE'][2],
            matched_graph_minus_RBF_H6_MSE=methods['graph_matched_dose']['central_forecast_H1_H6_MSE'][5]-methods['rbf_matched_dose']['central_forecast_H1_H6_MSE'][5],
            matched_graph_minus_RBF_progress_px=methods['graph_matched_dose']['central_selected_progress_delta_px']-methods['rbf_matched_dose']['central_selected_progress_delta_px'])
    return dict(complete=True,episode=report['episode'],seconds=report['seconds'],native=native,parents=parents,canary=report['canary'],
        old_rbf_central_exact=report['old_rbf_central_exact'],candidate_promoted=False,limitations=report['limitations'])


def run(a):
    a.output.mkdir(parents=True,exist_ok=False)
    receipt=json.loads((a.results/'DONE.json').read_text())
    if not receipt['complete'] or receipt['episode']!=a.episode:raise ValueError('Completed assigned state required')
    for row in receipt['outputs']:
        if sha(a.results/row['path'])!=row['sha256']:raise ValueError('Source SHA before summary read')
    report=json.loads((a.results/'summary.json').read_text());result=summarize(report)
    result.update(source_done_sha256=sha(a.results/'DONE.json'),source_files=receipt['outputs'],script_sha256=sha(__file__))
    write(a.output/'analysis.json',result)
    write(a.output/'DONE.json',dict(complete=True,episode=a.episode,outputs=[dict(path='analysis.json',sha256=sha(a.output/'analysis.json'),bytes=(a.output/'analysis.json').stat().st_size)]))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episode',type=int,choices=range(4),required=True)
    for key in ('results','output'):p.add_argument('--'+key,type=Path,required=True)
    run(p.parse_args())
