#!/usr/bin/env python3
"""Initial-state paired summaries for all predeclared head decomposition ablations."""
import argparse
import json
from pathlib import Path
from summarize_action_consequence import load,ci,sha

def summary(rows):
    if len(rows)!=4 or {r['key'] for r in rows}!={f'near-dev-{i:03d}' for i in range(4)}:raise ValueError('All four predeclared starts required')
    cases={r['key']:{c['name']:c for c in r['conditions']} for r in rows};first=rows[0]['conditions'];table=[]
    fields={'full_spatial_forecast_mse':lambda r:r['forecast_visual_mse'],
        'actual_encoded_regret':lambda r:r['ranking']['actual_encoded_regret'],
        'actual_4coordinate_distance_px':lambda r:r['ranking']['selected_physical_xy_distance'],
        'actual_block_distance_px':lambda r:r['selected_actual_block_distance_px'],
        'actual_wrapped_angle_rad':lambda r:r['selected_actual_wrapped_angle_rad'],
        'rank_actual_encoded':lambda r:r['ranking']['rank_correlation_actual_encoded']}
    for setting in first:
        if setting['name']=='unsteered' or setting['sham']:continue
        name=setting['name'];sham=name.removesuffix('_semantic')+'_sham'
        semantic=[cases[r['key']][name] for r in rows];control=[cases[r['key']][sham] for r in rows];base=[cases[r['key']]['unsteered'] for r in rows]
        record={k:setting[k] for k in ('name','head','horizon','kind')};record['metrics']={}
        for metric,get in fields.items():
            record['metrics'][metric]=dict(semantic_vs_baseline=ci([get(s)-get(b) for s,b in zip(semantic,base)]),
                sham_vs_baseline=ci([get(s)-get(b) for s,b in zip(control,base)]),semantic_vs_sham=ci([get(s)-get(c) for s,c in zip(semantic,control)]))
        record['semantic_changed_selection']=sum(s['ranking']['selected_candidate']!=b['ranking']['selected_candidate'] for s,b in zip(semantic,base))
        record['sham_changed_selection']=sum(s['ranking']['selected_candidate']!=b['ranking']['selected_candidate'] for s,b in zip(control,base))
        record['max_matched_actual_norm_error']=max(abs(x-y) for s,c in zip(semantic,control) for x,y in zip(s['norm_diagnostics']['actual_rounded_residual_norm'],c['norm_diagnostics']['actual_rounded_residual_norm']))
        record['max_suppression_fraction']=max(v for s in semantic for v in s['norm_diagnostics']['suppression_fraction'])
        record['episode_outcomes']=[dict(key=r['key'],semantic_selected=s['ranking']['selected_candidate'],sham_selected=c['ranking']['selected_candidate'],
            baseline_selected=b['ranking']['selected_candidate'],actual_xy_delta=s['ranking']['selected_physical_xy_distance']-b['ranking']['selected_physical_xy_distance'],
            angle_delta=s['selected_actual_wrapped_angle_rad']-b['selected_actual_wrapped_angle_rad']) for r,s,c,b in zip(rows,semantic,control,base)]
        table.append(record)
    return dict(complete=True,initialstate_groups=4,semantic_specifications=len(table),rows=table,
        posthoc_best_by_actual_xy=min(table,key=lambda r:r['metrics']['actual_4coordinate_distance_px']['semantic_vs_baseline']['mean'])['name'],
        successful_stage_seconds=sum(r['seconds'] for r in rows),old_single_vs_batch=[dict(key=r['key'],**r['old_single_vs_fresh_batch']) for r in rows],
        limitations=['All four development starts repeatedly inspected; this is exploratory screening, not held efficacy',
            'Bootstrap intervals per initialstate are descriptive and unadjusted for the fixed multi-comparison grid',
            'Posthoc best is an index for follow-up, not a validated selected operator',
            'Same site/head/residual norm, but spatial and mean components have different token degrees of freedom',
            'Suppression fraction above1 is over-removal/sign inversion rather than pure attenuation, reported explicitly'])

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report=summary(load([a.input]));a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(dict(complete=True,sha256=sha(a.output),specifications=report['semantic_specifications'],posthoc_best=report['posthoc_best_by_actual_xy'])))

if __name__=='__main__':main()
