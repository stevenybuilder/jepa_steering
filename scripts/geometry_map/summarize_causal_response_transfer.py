#!/usr/bin/env python3
"""Matched four-development comparison of observational versus causal response transfer."""
import argparse
import json
from pathlib import Path
import numpy as np
from complete_cached_geometry import sha256,write_json
from summarize_action_consequence import load,summarize,ci

def response_errors(pred,actual):
    p=pred.reshape(len(actual),-1);a=actual.reshape(len(actual),-1)
    norms=np.linalg.norm(a,axis=1).clip(1e-12)
    return dict(relative_error=float(np.mean(np.linalg.norm(p-a,axis=1)/norms)),
        cosine=float(np.mean(np.sum(p*a,axis=1)/(np.linalg.norm(p,axis=1).clip(1e-12)*norms))))

def arrays(dirs):
    result={}
    for root in dirs:
        d=json.loads((root/'DONE.json').read_text())
        if not d['complete']:raise ValueError('Incomplete result')
        for r in d['outputs']:
            if not r['path'].endswith('.npz'):continue
            if r['split']!='development_validation':raise ValueError('Only fixed four TRAIN-development groups')
            p=root/r['path']
            if sha256(p)!=r['sha256']:raise ValueError('Result hash mismatch')
            if r['key'] in result:raise ValueError('Duplicate source')
            result[r['key']]=dict(np.load(p,allow_pickle=False))
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--finite',type=Path,nargs='+',required=True);p.add_argument('--transfer',type=Path,nargs='+',required=True)
    p.add_argument('--models',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    old=load(a.finite);new=load(a.transfer);oo=arrays(a.finite);nn=arrays(a.transfer)
    f=json.loads((a.models/'FROZEN.json').read_text())
    if not f['complete'] or sha256(a.models/'response_models.npz')!=f['response_models_sha256']:raise ValueError('Bad frozen TRAIN response')
    models=dict(np.load(a.models/'response_models.npz',allow_pickle=False))
    if {r['key'] for r in new} != set(oo) or len(new)!=4:raise ValueError('Exactly same fixed four development states')
    merged=[];response=[];old_by={r['key']:r for r in old}
    for row in new:
        key=row['key'];native_error=float(np.max(np.abs(nn[key]['unsteered_native_cost']-oo[key]['unsteered_native_cost'])))
        if not np.array_equal(nn[key]['unsteered_visual_mse'],oo[key]['unsteered_visual_mse']):raise ValueError('Native baseline forecast mismatch across response methods')
        merged.append({**row,'conditions':old_by[key]['conditions']+[r for r in row['conditions'] if r['name']!='unsteered'],
            'seconds':row['seconds']+old_by[key]['seconds']})
        for basis in ('readable','action_effect','random'):
            actual=oo[key][basis+'_actual_finite_response']
            methods={'observational':oo[key][basis+'_surrogate_response'],
                'causal_global_mean':np.broadcast_to(models[basis+'_global_mean'],actual.shape),
                'causal_PCA16_ridge':nn[key][basis+'_response']}
            for method,pred in methods.items():response.append(dict(key=key,basis=basis,method=method,
                native_baseline_cost_maxabs=native_error,**response_errors(pred,actual)))
    report=summarize(merged);report['response_transfer_errors']={}
    for basis in ('readable','action_effect','random'):
        report['response_transfer_errors'][basis]={method:{metric:ci([r[metric] for r in response if r['basis']==basis and r['method']==method])
            for metric in ('relative_error','cosine')} for method in ('observational','causal_global_mean','causal_PCA16_ridge')}
    report.update(response_rows=response,train_initialstate_groups=16,training_response_action_rows=48,
        response_model_sha256=f['response_models_sha256'],same_native_baseline_visual_error_exact=True,
        native_baseline_cost_maxabs=max(r['native_baseline_cost_maxabs'] for r in response),
        no_development_response_used_to_fit=True,script_sha256=sha256(__file__))
    report['limitations'].extend(['Only three predeclared action types sampled for causal-response training, seven evaluated',
        'Global mean and conditioned ridge fixed before development scoring; no four-state refit or held performance claim',
        'Per-state finite-response benchmark requires12 extra native unrolls per basis/action set and is not free runtime steering'])
    write_json(a.output,report)

if __name__=='__main__':main()
