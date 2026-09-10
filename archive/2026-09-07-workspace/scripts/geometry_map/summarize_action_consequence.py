#!/usr/bin/env python3
"""Episode-group paired summaries of fixed-grid action-consequence interventions."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ci(values):
    rng=random.Random(17);n=len(values)
    if not n:return {'n_initialstates':0,'mean':None,'interval95':None}
    samples=sorted(statistics.mean(values[rng.randrange(n)] for _ in range(n)) for _ in range(2000))
    return {'n_initialstates':n,'mean':statistics.mean(values),'interval95':[samples[49],samples[1949]]}

def load(dirs):
    rows=[];seen=set()
    for directory in dirs:
        d=json.loads((directory/'DONE.json').read_text())
        if not d['complete']:raise ValueError('Incomplete intervention shard')
        for r in d['outputs']:
            if not r['path'].endswith('.json'):continue
            if r['split'] not in ('fit','development_validation','development_external'):raise ValueError('Forbidden heldoutcome')
            p=directory/r['path']
            if sha(p)!=r['sha256']:raise ValueError('Hash mismatch')
            v=json.loads(p.read_text())
            if v['key'] in seen:raise ValueError('Duplicateinitialstate')
            seen.add(v['key']);rows.append(v)
    return rows

def summarize(rows):
    conditions={r['key']:{a['name']:a for a in r['conditions']} for r in rows};names=list(conditions[rows[0]['key']])
    result={}
    groups={'fit':[r for r in rows if r['split']=='fit'],'seen_train_development':[r for r in rows if r['split']=='development_validation'],
        'original_external_development':[r for r in rows if r['split']=='development_external'],'all_development':[r for r in rows if r['split']!='fit']}
    for split,sources in groups.items():
        if not sources:continue
        result[split]={}
        for name in names:
            selected=[conditions[r['key']][name] for r in sources];base=[conditions[r['key']]['unsteered'] for r in sources]
            msekey='forecast_visual_mse' if 'forecast_visual_mse' in selected[0] else 'visual_mse'
            delta=[a[msekey]-b[msekey] for a,b in zip(selected,base)]
            r={'full_native_forecast_mse_delta':ci(delta),'relative_mse_delta_percent':ci([100*d/b[msekey] for d,b in zip(delta,base)]),
                'actual_encoded_selected_action_regret_delta':ci([a['ranking']['actual_encoded_regret']-b['ranking']['actual_encoded_regret'] for a,b in zip(selected,base)]),
                'selected_physical_xy_distance_delta':ci([a['ranking']['selected_physical_xy_distance']-b['ranking']['selected_physical_xy_distance'] for a,b in zip(selected,base)]),
                'selected_candidate_changed':sum(a['ranking']['selected_candidate']!=b['ranking']['selected_candidate'] for a,b in zip(selected,base))}
            if 'first_horizon_mse' in selected[0]:r['first_horizon_mse_delta']=ci([a['first_horizon_mse']-b['first_horizon_mse'] for a,b in zip(selected,base)])
            if name.startswith(('readable_','action_effect_')) and 'negative' not in name:
                other='random_'+name.removeprefix('readable_').removeprefix('action_effect_')
                if other in names:
                    r['matched_random_mse_difference']=ci([a[msekey]-conditions[s['key']][other][msekey] for a,s in zip(selected,sources)])
                    if 'delivered_pooled_norm_by_horizon' in selected[0]:r['max_matched_norm_error']=max(abs(x-y) for a,s in zip(selected,sources) for x,y in zip(a['delivered_pooled_norm_by_horizon'],conditions[s['key']][other]['delivered_pooled_norm_by_horizon']))
            result[split][name]=r
    train=result.get('fit',{})
    winner=min((name for name in train if name not in ('unsteered','identity')),key=lambda name:train[name]['full_native_forecast_mse_delta']['mean']) if train else None
    return {'complete':True,'n_initialstates':len(rows),'source_keys':[r['key'] for r in rows],'groups':result,'train_score_selected_arm':winner,
        'selection_semantics':'Derived fromfixedgrid training scores only afterexecution; notapredeclaredwinner orheldperformanceclaim',
        'native_model_seconds':sum(r['seconds'] for r in rows),'future_ground_truth_used_by_edit':False,
        'limitations':['Developmentepisodesalreadyseen; exploratorypairedbootstrap, notheldstudystatisticalconfirmation',
            'Coarsebank referenceactiongeneratesgoal, so native top1 mayalreadyatceiling; localnearplandiagnosticseparate',
            'Improved decodedmovement is not success; fullnative forecast error and actualselectedactionconsequences reported',
            'Traininglinearoutputresponse isassociational; finite-responsediagnostic separatelytestsactual modelresponse']}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dirs',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rows=load(a.dirs);report=summarize(rows)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({'complete':True,'n_initialstates':len(rows),'train_score_selected_arm':report['train_score_selected_arm'],'output_sha256':sha(a.output)}))

if __name__=='__main__':main()
