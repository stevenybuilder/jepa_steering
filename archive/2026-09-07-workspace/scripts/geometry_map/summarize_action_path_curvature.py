#!/usr/bin/env python3
"""Compact, initial-state-grouped summary of immutable action-path experiments."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics


def average(values):
    return statistics.fmean(values)


def episode_interval(values):
    values=list(values)
    if not values: raise ValueError('No initial-state values')
    rng=random.Random(2026090601)
    boot=sorted(average(rng.choices(values,k=len(values))) for _ in range(10000))
    return dict(mean=average(values), descriptive_episode_bootstrap_95=[boot[249],boot[9749]],
                initial_state_values=values, n_initial_states=len(values))


def summarize(reports):
    if len(reports)!=32 or any(not r['complete'] for r in reports): raise ValueError('All32 fixed paths required')
    expected={(f'near-dev-{e:03}',p,r) for e in range(4) for p in range(4) for r in (1,4)}
    if {(r['source']['key'],r['path']['pair'],r['path']['radius']) for r in reports}!=expected:
        raise ValueError('Missing, duplicated, or substituted source/direction/radius')
    views=[]
    for radius in (1,4):
        for block in (1,3,5):
            for horizon in (1,3,6):
                selected=[]
                for report in reports:
                    if report['path']['radius']!=radius: continue
                    site=next(r for r in report['rows'] if r['block']==block and r['intervention_horizon']==horizon)
                    selected.append((report['source']['key'],site))
                keys=sorted({key for key,_ in selected})
                def grouped(fn): return [average(fn(row) for key2,row in selected if key2==key) for key in keys]
                geom={k:episode_interval(grouped(lambda row,k=k:row['geometry'][k])) for k in
                    ('path_length','chord_length','path_chord_ratio','max_orthogonal_fraction_of_chord')}
                geom['mean_tangent_rotation_radians']=episode_interval(grouped(lambda row:average(row['geometry']['successive_segment_angle_radians'])))
                geom['backtracking_segments_total']=sum(row['geometry']['backtracking_segments'] for _,row in selected)
                names=[r['method'] for r in selected[0][1]['methods']]
                def method(row,name): return next(r for r in row['methods'] if r['method']==name)
                metrics=('midpoint_l2_error','midpoint_relative_l2_error','downstream_native_fullspatial_mse_after_patch',
                    'actual_future_mse_delta_after_patch','coordinate_box_overshoot_l2','coordinate_box_overshoot_fraction',
                    'estimate_offset_over_sample_radius')
                methods={name:{k:episode_interval(grouped(lambda row,k=k,name=name:method(row,name)[k])) for k in metrics} for name in names}
                contrasts={}
                for comparator in ('linear_equal_data','reparameterized_chord','reflected_curvature','near_chord'):
                    contrast={}
                    for metric in ('midpoint_l2_error','downstream_native_fullspatial_mse_after_patch','actual_future_mse_delta_after_patch'):
                        contrast[metric]=episode_interval(grouped(lambda row,metric=metric,comparator=comparator:
                            method(row,'cubic_equal_data')[metric]-method(row,comparator)[metric]))
                    contrasts['cubic_minus_'+comparator]=contrast
                views.append(dict(radius=radius,block=block,horizon=horizon,geometry=geom,methods=methods,contrasts=contrasts))
    return dict(complete=True,paths=32,initial_states=4,action_directions_per_state=4,views=views,
        maximum_historical_batch_difference=max(r['source']['historical_single_vs_batch_visual_maxabs'] for r in reports),
        maximum_native_action_conversion_roundoff=max(r['path']['native_denormalized_raw_maxabs'] for r in reports),
        maximum_normalized_path_affine_roundoff=max(r['path']['normalized_affine_path_maxabs'] for r in reports),
        sum_path_stage_seconds=sum(r['seconds'] for r in reports),
        all_self_patch_exact=all(row['self_patch_exact'] for r in reports for row in r['rows']),
        primary_exploratory_views=[dict(block=3,horizon=3,radius=r) for r in (1,4)],
        limitations=['Four development starts, not confirmatory or independent candidate counts',
            'Directions averaged within each initial state before intervals; only descriptive bootstrap with n4',
            '18 block/horizon/radius views are not multiple-comparison-adjusted confirmatory findings',
            'Cubic central estimate is also symmetric four-point quadratic LS; not a uniquely cubic mechanism',
            'Action-response path bending does not establish a semantic manifold, density, or useful steering',
            'Actual future score compares predicted versus actual ENCODED images; no new physical intervention rollout',
            'Only central action physical futures are observed; other sampled action futures remain unobserved',
            'Actual-future and native-fidelity errors summarized only at and after intervention horizon'])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); done=json.loads((a.input/'DONE.json').read_text())
    rows=[r for r in done['outputs'] if r['path'].endswith('.json')]
    if not done['complete']: raise ValueError('Incomplete run')
    reports=[]
    for row in rows:
        path=a.input/row['path']
        if sha(path)!=row['sha256']: raise ValueError('Compact report SHA mismatch')
        reports.append(json.loads(path.read_text()))
    result=summarize(reports);result['source_done_sha256']=sha(a.input/'DONE.json');result['total_stage_seconds']=done['seconds']
    if a.output.exists(): raise FileExistsError('Immutable summary output already exists')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(complete=True,summary=str(a.output),sha256=sha(a.output),views=len(result['views']))))


if __name__=='__main__': main()
