#!/usr/bin/env python3
"""Immutable dense-view packaging and episode-grouped TRAIN-only patch charts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from complete_cached_geometry import sha256,write_json

VIEWS=tuple((b,h) for b in (1,3,5) for h in (1,3,6))

def verified_rows(directory,suffix):
    d=json.loads((directory/'DONE.json').read_text())
    if not d['complete']:raise ValueError('Incomplete source')
    rows=[r for r in d['outputs'] if r['path'].endswith(suffix)]
    for r in rows:
        if r['split'] not in ('fit','development_validation','development_external'):raise ValueError('Held source forbidden before loading')
        if sha256(directory/r['path'])!=r['sha256']:raise ValueError('Source SHA mismatch')
    return rows

def pack(args):
    args.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    rows=verified_rows(args.patch_dir,'.pt');bank_rows={r['key']:r for r in verified_rows(args.bank_dir,'.npz')}
    tensors=[];labels=[];contexts=[];groups=[];splits=[];sources=[]
    for r in rows:
        data=torch.load(args.patch_dir/r['path'],map_location='cpu',weights_only=False)
        bank=dict(np.load(args.bank_dir/bank_rows[r['key']]['path'],allow_pickle=False))
        if data['meta']['split']!=r['split']:raise ValueError('Payload split mismatch')
        tensors.append(data['residual_patches']);labels.append(bank['q_next'][...,:4]);contexts.append(bank['states'][:,0,:4])
        groups.extend([r['key']]*7);splits.extend([r['split']]*7);sources.append(dict(patches=r,bank=bank_rows[r['key']]))
    dense=torch.cat(tensors,dim=0);target=np.concatenate(labels);outputs=[]
    # Primary P3/H3 written first for transfer/analysis while fixed secondary views follow.
    for block,horizon in ((3,3),)+tuple(v for v in VIEWS if v!=(3,3)):
        p=args.output/f'P{block}-H{horizon}.npz'
        value=dense[:,horizon-1,(1,3,5).index(block)].numpy()
        np.savez_compressed(p,dense=value,targets=target[:,horizon-1],context_targets=np.concatenate(contexts),groups=np.asarray(groups),splits=np.asarray(splits))
        outputs.append(dict(path=p.name,sha256=sha256(p),bytes=p.stat().st_size,block=block,horizon=horizon))
        write_json(p.with_suffix('.DONE.json'),dict(complete=True,outputs=[outputs[-1]],sources=sources,
            axes='actionrow,patch256,channel400',independent_sources=len(rows)))
        print(json.dumps(dict(event='patch_view_packed',block=block,horizon=horizon,bytes=p.stat().st_size)),flush=True)
    write_json(args.output/'DONE.json',dict(complete=True,outputs=outputs,sources=sources,seconds=time.monotonic()-start,
        script_sha256=sha256(__file__)))

def bundle(path):
    d=json.loads(path.with_suffix('.DONE.json').read_text());r=d['outputs'][0]
    if not d['complete'] or sha256(path)!=r['sha256']:raise ValueError('Bad view bundle SHA')
    b=dict(np.load(path,allow_pickle=False))
    if any(s not in ('fit','development_validation','development_external') for s in b['splits']):raise ValueError('Held bundle forbidden')
    return b

def chart(args):
    from local_patch_geometry import analyze_chart
    start=time.monotonic();loaded=[bundle(p) for p in args.bundles]
    b={k:np.concatenate([v[k] for v in loaded]) for k in loaded[0]}
    train=b['splits']=='fit';dev=~train
    if len(set(b['groups'][train]))!=16 or len(set(b['groups'][dev]))!=8:raise ValueError('Require unchanged16fit/8development groups')
    groups=b['groups'];results={}
    for name,x in (('pooled',b['dense'].mean(1)),('dense',b['dense'].reshape(len(groups),-1))):
        if (args.rank,args.ambient_rank)!=(8,32) and (args.block,args.horizon)!=(3,3):raise ValueError('Sensitivity restricted to P3/H3')
        result=analyze_chart(torch.from_numpy(x[train]),torch.from_numpy(x[dev]),groups[train],groups[dev],
            rank=args.rank,ambient_rank=args.ambient_rank,neighbors=32,max_per_group=4,train_targets=b['targets'][train],test_targets=b['targets'][dev])
        result['physical_target']='actual candidate future [agentX,agentY,blockX,blockY], pixel units; no future in neighborhood selection'
        result['physical_target_rmse_px']={k:float(np.sqrt(result['equal_episode_means'][k]/4))
            for k in ('global_target_squared_error','local_target_squared_error','random_neighborhood_target_squared_error')}
        result['mean_prediction_rmse_px']=float(np.sqrt(np.square(b['targets'][dev]-b['targets'][train].mean(0)).mean()))
        if 'context_targets' in b:
            result['context_physical_persistence_rmse_px']=float(np.sqrt(np.square(b['targets'][dev]-b['context_targets'][dev]).mean()))
        subgroup={str(g):str(s) for g,s in zip(groups[dev],b['splits'][dev])}
        result['development_subgroups']={}
        for label in ('development_validation','development_external'):
            selected=[r for r in result['episode_summary'] if subgroup[r['episode']]==label]
            result['development_subgroups'][label]=dict(initialgroups=len(selected),
                equal_episode_means={k:float(np.mean([r[k] for r in selected])) for k in result['equal_episode_means']})
        results[name]=result
    write_json(args.output,dict(complete=True,block=args.block,horizon=args.horizon,primary_exploratory=(args.block==3 and args.horizon==3),
        charts=results,seconds=time.monotonic()-start,script_sha256=sha256(__file__),helper_sha256=sha256(Path(__file__).with_name('local_patch_geometry.py')),
        bundles=[dict(path=str(p),sha256=sha256(p)) for p in args.bundles],no_test_refit=True,
        future_frame_offset_raw_steps=5*args.horizon,context_raw_step=0,limitations=['Charts describe local approximation, not proof of curved causal manifold',
        'Action alternatives correlated within16TRAIN/8development initialstates','Physical position readability is not action-ranking improvement or steering success']))
    print(json.dumps(dict(event='patch_chart_complete',block=args.block,horizon=args.horizon,seconds=time.monotonic()-start)),flush=True)

def coverage(args):
    from local_patch_geometry import gram_pca
    loaded=[bundle(p) for p in args.bundles];b={k:np.concatenate([v[k] for v in loaded]) for k in loaded[0]}
    fit=b['splits']=='fit';dev=~fit;results={}
    if len(set(b['groups'][fit]))!=16 or len(set(b['groups'][dev]))!=8:raise ValueError('Unchanged16/8 initialgroups required')
    for name,x in (('pooled',b['dense'].mean(1)),('dense',b['dense'].reshape(len(fit),-1))):
        tr=torch.from_numpy(x[fit]);te=torch.from_numpy(x[dev]);mean,basis,eigen=gram_pca(tr,len(tr)-1)
        centered=te-mean;total=centered.square().sum(1);scores=centered@basis.T
        train_energy=(tr-tr.mean(0)).square().sum(1).mean();rows=[]
        for requested in (8,16,32,64,96):
            actual=min(requested,len(basis));omitted=(total-scores[:,:actual].square().sum(1)).clamp_min(0)
            ep=[]
            for group in sorted(set(b['groups'][dev])):
                take=b['groups'][dev]==group
                ep.append(dict(key=str(group),split=str(b['splits'][dev][take][0]),
                    omitted_to_train_total=float((omitted[take]/train_energy).mean()),omitted_fraction_test_energy=float((omitted[take]/total[take].clamp_min(1e-20)).mean())))
            rows.append(dict(requested_rank=requested,actual_identifiable_rank=actual,episodes=ep,
                equal_episode_omitted_to_train_total=float(np.mean([r['omitted_to_train_total'] for r in ep])),
                equal_episode_omitted_fraction_test_energy=float(np.mean([r['omitted_fraction_test_energy'] for r in ep]))))
        results[name]=dict(train_nonzero_eigenvalues=eigen.tolist(),identifiable_rank=len(basis),curves=rows)
    write_json(args.output,dict(complete=True,block=3,horizon=3,coverage=results,train_groups=16,development_groups=8,
        no_target_fit=True,no_test_refit=True,script_sha256=sha256(__file__),
        limitation='Outside-TRAIN-linear-span energy measures coverage, not manifold curvature or causal utility'))

def head_metrics(b):
    n=b['head_norm'].astype(float);e=n*n;s=e.sum(-1).clip(1e-30)
    cos=b['head_cosine'].astype(float);pn=b['head_pooled_norm'].astype(float);pc=b['head_pooled_cosine'].astype(float)
    return dict(effective_head_count=s*s/np.square(e).sum(-1).clip(1e-30),max_head_energy_fraction=e.max(-1)/s,
        total_head_sum_energy_to_individual_energy=(n[...,None]*n[...,None,:]*cos).sum((-1,-2))/s,
        pooled_head_sum_energy_to_individual_energy=(pn[...,None]*pn[...,None,:]*pc).sum((-1,-2))/np.square(pn).sum(-1).clip(1e-30),
        pooling_retained_energy_fraction=256*np.square(pn).sum(-1)/s,
        mean_abs_head_cosine=np.abs(cos[...,np.triu_indices(16,1)[0],np.triu_indices(16,1)[1]]).mean(-1),
        mean_abs_pooled_head_cosine=np.abs(pc[...,np.triu_indices(16,1)[0],np.triu_indices(16,1)[1]]).mean(-1))

def heads(args):
    rows=[]
    for directory in args.patch_dirs:
        for r in verified_rows(directory,'.npz'):
            b=dict(np.load(directory/r['path'],allow_pickle=False));rows.append((r,head_metrics(b)))
    if sum(r['split']=='fit' for r,_ in rows)!=16 or len(rows)!=24:raise ValueError('Require fixed24 sources')
    result=[]
    for block in range(6):
        for h in range(6):
            fields={}
            for field in rows[0][1]:
                tr=np.concatenate([v[field][:,h,block] for r,v in rows if r['split']=='fit'])
                threshold=float(np.quantile(tr,.9));group=[]
                for r,v in rows:
                    values=v[field][:,h,block]
                    group.append(dict(key=r['key'],split=r['split'],mean=float(values.mean()),within_context_action_variance=float(values.var()),
                        fraction_above_TRAIN90=float((values>threshold).mean())))
                fields[field]=dict(train90_threshold=threshold,episode_rows=group,train_context_mean_variance=float(np.var([r['mean'] for r in group if r['split']=='fit'])),
                    development_context_mean_variance=float(np.var([r['mean'] for r in group if r['split']!='fit'])))
            result.append(dict(block=block,imagined_horizon=h+1,metrics=fields))
    write_json(args.output,dict(complete=True,rows=result,train_initialgroups=16,development_initialgroups=8,
        script_sha256=sha256(__file__),interpretation='Head energy/cosine activity, not causal importance or firing. Coactivation does not establish superposition.',
        physical_time='All contexts fixed real rawstep0; imagined horizons analyzed separately'))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('pack','chart','heads','coverage'))
    p.add_argument('--patch-dir',type=Path);p.add_argument('--bank-dir',type=Path);p.add_argument('--patch-dirs',type=Path,nargs='+')
    p.add_argument('--bundles',type=Path,nargs='+');p.add_argument('--block',type=int);p.add_argument('--horizon',type=int)
    p.add_argument('--rank',type=int,default=8);p.add_argument('--ambient-rank',type=int,default=32)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
    {'pack':pack,'chart':chart,'heads':heads,'coverage':coverage}[a.mode](a)

if __name__=='__main__':main()
