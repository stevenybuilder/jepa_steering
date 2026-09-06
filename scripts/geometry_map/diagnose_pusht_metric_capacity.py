#!/usr/bin/env python3
"""Read-only decomposition of the completed PSD task-metric treatment."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from fit_pusht_task_metric import digest
from run_expand_pusht_candidates import spearman


def distribution(x):
    x=np.asarray(x,dtype=float)
    return dict(mean=float(x.mean()),median=float(np.median(x)),minimum=float(x.min()),maximum=float(x.max()))


def scaled_labels(y,scale):
    # The frozen fit_weights explicitly casts labels to float64 before scaling.
    return np.asarray(y,dtype=np.float64)*scale


def energy_parts(x,basis):
    x=np.asarray(x,dtype=float);coordinates=x@basis.T;native=np.sum(x*x,axis=1)
    span=np.sum(coordinates**2,axis=1);complement=native-span
    centered=x-x.mean(0);centered_span=np.sum((centered@basis.T)**2)
    return native,span,complement,coordinates,dict(
        fraction_native_cost_in_span=distribution(span/native),fraction_native_cost_in_complement=distribution(complement/native),
        within_state_activation_variation_fraction_in_span=float(centered_span/np.sum(centered**2)),
        native_cost_std=float(native.std()),span_cost_std=float(span.std()),complement_cost_std=float(complement.std()),
        variance_decomposition=dict(native=float(native.var()),span=float(span.var()),complement=float(complement.var()),
            twice_covariance=float(2*np.mean((span-span.mean())*(complement-complement.mean())))))


def margin_diagnostics(native,span,complement,coordinates,weights):
    native=np.asarray(native);order=np.argsort(native,kind='stable');winner,runner=map(int,order[:2])
    adjustment=coordinates**2@(weights-1)
    rows=[]
    for beta in (0.,.25,.5,1.):
        cost=native+beta*adjustment;others=np.arange(len(cost))!=winner
        challenger=int(np.arange(len(cost))[others][np.argmin(cost[others])])
        rows.append(dict(blend=beta,selected_index=int(np.argmin(cost)),
            native_winner_cost=float(cost[winner]),native_runner_cost=float(cost[runner]),
            original_pair_margin=float(cost[runner]-cost[winner]),
            original_pair_margin_ratio=float((cost[runner]-cost[winner])/(native[runner]-native[winner])),
            strongest_challenger_index=challenger,margin_to_strongest_challenger=float(cost[challenger]-cost[winner]),
            winner_cost_change=float(beta*adjustment[winner]),runner_cost_change=float(beta*adjustment[runner]),
            absolute_cost_change_relative_to_native=distribution(np.abs(beta*adjustment)/native)))
    return dict(native_winner=winner,native_runner=runner,native_margin=float(native[runner]-native[winner]),
        native_pair_span_margin=float(span[runner]-span[winner]),
        native_pair_complement_margin=float(complement[runner]-complement[winner]),
        fitted_pair_span_margin=float((coordinates[runner]**2-coordinates[winner]**2)@weights),
        native_winner_span_fraction=float(span[winner]/native[winner]),
        native_runner_span_fraction=float(span[runner]/native[runner]),blends=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fit',type=Path,required=True)
    p.add_argument('--inputs',nargs='+',type=Path,required=True);p.add_argument('--coverage',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();started=time.monotonic()
    receipt=json.loads((a.fit/'DONE.json').read_text());entries={r['path']:r for r in receipt['outputs']}
    if not receipt['complete'] or receipt['states']!=4:raise ValueError('Completed fourfold metric required')
    def verified(name):
        path=a.fit/name
        if digest(path)!=entries[name]['sha256']:raise ValueError('Fit source SHA differs')
        return path
    protocol=json.loads(verified('protocol.json').read_text())
    if digest(a.coverage)!=protocol['coverage_sha256']:raise ValueError('Original coverage labels changed')
    coverage=json.loads(a.coverage.read_text());cover={r['episode']:np.asarray(r['goal_aligned_coverage_by_candidate']) for r in coverage['rows']}
    banks={};sources=[]
    for directory in a.inputs:
        d=json.loads((directory/'DONE.json').read_text());r=d['outputs'][0]
        if not d['complete'] or d['episode'] not in range(4) or digest(directory/r['path'])!=r['sha256']:raise ValueError('Input source SHA/group failure')
        banks[d['episode']]=dict(np.load(directory/r['path'],allow_pickle=False));sources.append(d)
    if sorted(banks)!=list(range(4)):raise ValueError('Retain all four development groups')
    reports=[]
    for held in range(4):
        fit=dict(np.load(verified(f'fold-{held}-FROZEN.npz'),allow_pickle=False));meta=json.loads(verified(f'fold-{held}-FROZEN.json').read_text())
        old=json.loads(verified(f'fold-{held}.json').read_text());train=meta['fit_states'];basis=fit['basis']
        if train!=[i for i in range(4) if i!=held]:raise ValueError('LOSO grouping changed')
        parts={key:energy_parts(banks[held][key].astype(float),basis) for key in ('predicted','actual')}
        train_x=np.concatenate([banks[i]['actual'] for i in train]).astype(float)
        train_native=np.sum(train_x**2,axis=1);train_coords=train_x@basis.T
        for label in ('joint_xy_squared_proxy','one_minus_requested_goal_coverage'):
            weights=fit[label+'_weights'];m=meta['models'][label]
            def labels(i):
                return np.sum((banks[i]['states'][:,:4]-banks[i]['goal_state'][:4])**2,axis=1) if label=='joint_xy_squared_proxy' else 1-cover[i]
            train_y=scaled_labels(np.concatenate([labels(i) for i in train]),m['label_scale_train_only'])
            fitted_train=train_native+(train_coords**2)@(weights-1)
            fit_mse=float(np.mean((fitted_train-train_y)**2));baseline_mse=float(np.mean((train_native-train_y)**2))
            if not np.isclose(fit_mse,m['train_fit_mse'],rtol=1e-12,atol=1e-12):raise ValueError('Saved training fit not reproduced')
            train_rank=[]
            for k,episode in enumerate(train):
                sl=slice(64*k,64*(k+1));train_rank.append(dict(episode=episode,native_rho=spearman(train_native[sl],train_y[sl]),fitted_rho=spearman(fitted_train[sl],train_y[sl])))
            channels={}
            for key,channel in [('predicted','predicted'),('actual','actual_oracle_encoding')]:
                native,span,complement,coords,energy=parts[key]
                margins=margin_diagnostics(native,span,complement,coords,weights)
                fitted=native+(coords**2)@(weights-1);y=scaled_labels(labels(held),m['label_scale_train_only'])
                oldrow=next(r for r in old['rows'] if r['fit_label']==label and r['channel']==channel and r['family']=='learned_psd' and r['blend']==1.)
                if not np.allclose(fitted,oldrow['cost_by_candidate'],rtol=1e-12,atol=1e-12):raise ValueError('Existing candidate costs changed')
                channels[channel]=dict(energy=energy,margins=margins,held_native_rho=spearman(native,y),held_fitted_rho=spearman(fitted,y),
                    held_native_scaled_label_mse=float(np.mean((native-y)**2)),held_fitted_scaled_label_mse=float(np.mean((fitted-y)**2)))
            reports.append(dict(held_state=held,label=label,weights={**distribution(weights),'zero_count':int(np.sum(weights==0)),
                'rms_change_from_native':float(np.sqrt(np.mean((weights-1)**2))),'values':weights.tolist()},
                training=dict(native_mse=baseline_mse,fitted_mse=fit_mse,mse_reduction_fraction=1-fit_mse/baseline_mse,
                    per_state_ranking=train_rank,mean_native_within_state_rho=float(np.mean([r['native_rho'] for r in train_rank])),
                    mean_fitted_within_state_rho=float(np.mean([r['fitted_rho'] for r in train_rank]))),channels=channels))
        if time.monotonic()-started>600:raise RuntimeError('Bounded CPU budget exceeded')
    a.output.mkdir(parents=True,exist_ok=False)
    result=dict(complete=True,rows=reports,seconds=time.monotonic()-started,fit_source_done_sha256=digest(a.fit/'DONE.json'),
        sources=sources,script_sha256=digest(__file__),new_optimization_calls=0,model_calls=0,simulator_calls=0,
        interpretation_constraints=['Fraction of cost in complement alone is not a capacity proof; compare within-state variation and pair margins',
            'Fitted weights and all originally frozen blends reused, no new doses/selection',
            'Four previously seen development states, no untouched confirmation'])
    path=a.output/'summary.json';path.write_text(json.dumps(result,indent=2))
    (a.output/'DONE.json').write_text(json.dumps(dict(complete=True,seconds=result['seconds'],outputs=[dict(path=path.name,sha256=digest(path),bytes=path.stat().st_size)]),indent=2))
    print(json.dumps(dict(event='metric_capacity_complete',seconds=result['seconds'],rows=len(reports))),flush=True)


if __name__=='__main__':main()
