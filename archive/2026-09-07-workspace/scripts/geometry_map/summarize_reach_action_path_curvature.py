#!/usr/bin/env python3
"""Episode-balanced, raw-support Reach path diagnostics; no steering claims."""
import argparse
import hashlib
import json
from pathlib import Path
from summarize_action_path_curvature import average,episode_interval

EPISODES=(0,1,4,7)


def summarize(reports):
    expected={(e,p,r) for e in EPISODES for p in range(4) for r in (1,4)}
    actual={(r['source']['episode'],r['path']['pair'],r['path']['radius']) for r in reports}
    if len(reports)!=32 or actual!=expected or not all(r['complete'] for r in reports):
        raise ValueError('Require exactly32 fixed paths on development0/1/4/7')
    views=[]
    for radius in (1,4):
        for block in (1,3,5):
            for horizon in (1,3,6):
                selected=[(r,next(s for s in r['rows'] if s['block']==block and s['intervention_horizon']==horizon))
                          for r in reports if r['path']['radius']==radius]
                def grouped(fn):
                    return [average(fn(r,s) for r,s in selected if r['source']['episode']==e) for e in EPISODES]
                def metric(s,name,key):return next(m for m in s['methods'] if m['method']==name)[key]
                def native_error(r):return average(r['baseline_actual_fullspatial_mse_by_horizon'][horizon-1:])
                geometry={k:episode_interval(grouped(lambda r,s,k=k:s['geometry'][k])) for k in
                    ('path_length','chord_length','path_chord_ratio','max_orthogonal_fraction_of_chord')}
                methods={}
                for name in [m['method'] for m in selected[0][1]['methods']]:
                    methods[name]={k:episode_interval(grouped(lambda r,s,k=k,name=name:metric(s,name,k))) for k in
                        ('midpoint_l2_error','downstream_native_fullspatial_mse_after_patch','actual_future_mse_delta_after_patch')}
                    methods[name]['actual_future_error_reduction_percent_vs_native']=episode_interval(grouped(
                        lambda r,s,name=name:-100*metric(s,name,'actual_future_mse_delta_after_patch')/native_error(r)))
                contrasts={}
                for comparator in ('linear_equal_data','near_chord','reparameterized_chord','reflected_curvature'):
                    contrasts['cubic_minus_'+comparator]={k:episode_interval(grouped(
                        lambda r,s,k=k,comparator=comparator:metric(s,'cubic_equal_data',k)-metric(s,comparator,k)))
                        for k in ('midpoint_l2_error','downstream_native_fullspatial_mse_after_patch','actual_future_mse_delta_after_patch')}
                    contrasts['cubic_minus_'+comparator]['actual_future_error_reduction_percent_native_denominator']=episode_interval(grouped(
                        lambda r,s,comparator=comparator:100*(metric(s,comparator,'actual_future_mse_delta_after_patch')-
                            metric(s,'cubic_equal_data','actual_future_mse_delta_after_patch'))/native_error(r)))
                views.append(dict(radius=radius,raw_xyz_rms=.05*radius,block=block,horizon=horizon,
                    geometry=geometry,methods=methods,contrasts=contrasts))
    return dict(complete=True,episodes=EPISODES,independent_initial_states=4,model_paths=32,views=views,
        all_self_patch_exact=all(s['self_patch_exact'] for r in reports for s in r['rows']),
        maximum_historical_single_vs_batch_visual_maxabs=max(r['source']['historical_single_vs_batch_visual_maxabs'] for r in reports),
        maximum_action_roundtrip_absolute=max(r['path']['normalization_raw_roundtrip_maxabs'] for r in reports),
        raw_xyz_bound_exceedance_by_state=[dict(episode=e,central_xyz_count=next(r['path']['raw_xyz_outside_unit_box_counts'][2]
            for r in reports if r['source']['episode']==e),maximum_path_xyz_count=max(max(r['path']['raw_xyz_outside_unit_box_counts'])
            for r in reports if r['source']['episode']==e)) for e in EPISODES],
        primary_views=[dict(block=3,horizon=3,radius=r) for r in (1,4)],
        limitations=['Four already-seen development states, not confirmation or32 independent robot episodes',
            'All four directions averaged within state before descriptive n4 bootstrap;18 views exploratory',
            'New raw XYZ action paths are model-only and unprojected, not guaranteed feasible simulator controls',
            'Radius units differ from Push; qualitative cross-task comparison only',
            'Native restoration differs from accuracy against the existing central actual encoded frames',
            'Cubic midpoint equals symmetric four-point quadratic LS, not a cubic-specific mechanism',
            'No manifold/density or physical steering efficacy inference; no new simulator/CEM'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();done=json.loads((a.input/'DONE.json').read_text());reports=[]
    if not done['complete']:raise ValueError('Incomplete run')
    for row in done['outputs']:
        if not row['path'].startswith('reach-dev-') or not row['path'].endswith('.json'):continue
        path=a.input/row['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Compact report SHA mismatch')
        reports.append(json.loads(path.read_text()))
    result=summarize(reports);result.update(source_done_sha256=hashlib.sha256((a.input/'DONE.json').read_bytes()).hexdigest(),
        process_seconds=done['seconds'],source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(complete=True,path=str(a.output),sha256=hashlib.sha256(a.output.read_bytes()).hexdigest())))


if __name__=='__main__':main()
