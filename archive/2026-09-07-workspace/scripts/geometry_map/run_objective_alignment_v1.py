#!/usr/bin/env python3
"""Frozen-cost/physical-objective diagnostic on four cached DEVELOPMENT banks."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import time
import numpy as np

PROTOCOL = {
    'version': 'objective-alignment-v1', 'episodes': [0,1,2,3], 'plans_per_state': 64,
    'independent_units': 'four previously seen DEVELOPMENT states; not 256 independent samples',
    'primary_horizon': 6, 'all_horizons': [1,2,3,4,5,6],
    'objective': 'mean visual squared error + 0.1 * mean proprio squared error to exact frozen goal',
    'sum_all_diffs': False, 'cached_H0': 'excluded from future diagnostics',
    'physical_metrics': ['requested-goal polygon coverage', 'combined agentXY/blockXY distance'],
    'coverage_goal': 'FROZEN_INPUTS goal_state[2:5], not native painted default goal',
    'tie_policy': 'exact cost ties; physical ties abs<=1e-12; smallest index plus tied outcome ranges',
    'margin_bin_edges_relative_mean_cost': [0, .001, .01, .1, None],
    'reliability': 'descriptive pairwise ordering agreement, not calibrated probability; pairs dependent',
    'cost_recompute_atol': 2e-6, 'cost_recompute_rtol': 2e-6,
    'float_policy': 'record CPU vs cached GPU discrepancy as measurement floor, not bit identity',
    'physical_oracles': 'nondeployable within-bank upper bounds, not policies or independent confirmations',
    'actual_encoding_oracle': 'unavailable online; perfect-forecast cost selection, NOT physical utility upper bound',
    'new_model_or_simulator_calls': 0, 'fit_or_weight_search': False, 'held_access': False,
    'cpu_seconds_per_shard_limit': 45, 'aggregate_cpu_seconds_limit': 180,
}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def ranks(x):
    x=np.asarray(x,dtype=float); order=np.argsort(x,kind='stable'); out=np.empty(len(x),float)
    i=0
    while i<len(x):
        j=i+1
        while j<len(x) and x[order[j]]==x[order[i]]: j+=1
        out[order[i:j]]=(i+j-1)/2; i=j
    return out


def rho(x,y):
    a,b=ranks(x),ranks(y)
    return None if np.std(a)==0 or np.std(b)==0 else float(np.corrcoef(a,b)[0,1])


def paired_order(cost, benefit, tolerance=1e-12):
    """cost lower is better; benefit higher is better; dependent pairs retained."""
    cost,benefit=np.asarray(cost,float),np.asarray(benefit,float)
    i,j=np.triu_indices(len(cost),1); dc=cost[j]-cost[i]; dy=benefit[i]-benefit[j]
    ct=dc==0; yt=np.abs(dy)<=tolerance; usable=~ct & ~yt
    correct=dc*dy>0
    denom=max(float(np.mean(np.abs(cost))),1e-12)
    relative=np.abs(dc)/denom; bins=[]
    edges=[0,.001,.01,.1,np.inf]
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask=(relative>=lo)&(relative<hi); selected=mask&usable
        bins.append(dict(lower=lo,upper=None if np.isinf(hi) else hi,
            all_pair_count=int(mask.sum()),informative_pair_count=int(selected.sum()),
            agreement=None if not selected.any() else float(correct[selected].mean()),
            physical_ties=int((mask&yt).sum()),cost_ties=int((mask&ct).sum())))
    return dict(pair_count=len(i),informative_pair_count=int(usable.sum()),
        agreement=None if not usable.any() else float(correct[usable].mean()),
        cost_ties=int(ct.sum()),benefit_ties=int(yt.sum()),
        spearman_cost_negative_vs_benefit=rho(-cost,benefit),margin_scale=denom,bins=bins)


def selection(cost, coverage, xy, tolerance=0.):
    cost,coverage,xy=map(lambda x:np.asarray(x,float),(cost,coverage,xy))
    pick=int(np.argmin(cost)); tied=np.flatnonzero(np.abs(cost-cost[pick])<=tolerance)
    order=np.argsort(cost,kind='stable')
    return dict(index=pick,tied_indices=tied.tolist(),tie_count=len(tied),
        winner_runner_up_margin=float(cost[order[1]]-cost[pick]),
        requested_goal_coverage=float(coverage[pick]),xy_distance=float(xy[pick]),
        coverage_regret=float(coverage.max()-coverage[pick]),xy_regret=float(xy[pick]-xy.min()),
        tied_coverage_range=[float(coverage[tied].min()),float(coverage[tied].max())],
        tied_xy_range=[float(xy[tied].min()),float(xy[tied].max())])


def horizon_summary(pred, actual, cov, xy):
    pred,actual,cov,xy=map(lambda x:np.asarray(x,float),(pred,actual,cov,xy))
    i,j=np.triu_indices(len(pred),1); eps=np.abs(pred-actual)
    certified=np.abs(pred[i]-pred[j])>eps[i]+eps[j]
    agreement=np.sign(pred[i]-pred[j])==np.sign(actual[i]-actual[j])
    if not np.all(agreement[certified]): raise AssertionError('Cost error margin certificate violated')
    winner=int(pred.argmin()); competitors=np.arange(len(pred))!=winner
    safe=(pred-pred[winner])>eps+eps[winner]
    return dict(native_predicted=selection(pred,cov,xy),
        actual_encoding_cost_oracle=selection(actual,cov,xy),
        requested_coverage_oracle=selection(-cov,cov,xy,1e-12),
        xy_oracle=selection(xy,cov,xy,1e-12),
        predicted_vs_coverage=paired_order(pred,cov),actual_encoded_vs_coverage=paired_order(actual,cov),
        predicted_vs_xy=paired_order(pred,-xy),actual_encoded_vs_xy=paired_order(actual,-xy),
        predicted_vs_actual_cost=paired_order(pred,-actual,0.),
        cost_error_mae=float(eps.mean()),cost_error_rmse=float(np.sqrt(np.mean((pred-actual)**2))),
        cost_error_max=float(eps.max()),
        sufficient_pair_cost_rank_certificate_count=int(certified.sum()),
        sufficient_pair_cost_rank_certificate_fraction=float(certified.mean()),
        sufficient_native_winner_certificate=bool(np.all(safe[competitors])),
        certificate_scope='actual-encoded cost ordering only; no physical reward certificate')


def native_cost(visual, proprio, goal_visual, goal_proprio):
    """CPU float32 arithmetic; same MSE reductions as native objective."""
    import torch
    v=visual.float().reshape(6,-1); p=proprio.float().reshape(6,-1)
    gv=goal_visual.float().reshape(1,-1); gp=goal_proprio.float().reshape(1,-1)
    if v.shape[1]!=gv.shape[1] or p.shape[1]!=gp.shape[1]: raise ValueError('Goal feature shape mismatch')
    return ((v-gv).square().mean(1)+.1*(p-gp).square().mean(1)).numpy()


def write_json(path,data):
    with Path(path).open('x') as f: json.dump(data,f,indent=2,allow_nan=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',type=Path,required=True);p.add_argument('--episode',type=int,choices=range(4),required=True)
    p.add_argument('--done-sha',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--native-objective-source',type=Path,required=True)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'protocol.json',PROTOCOL)
    started=time.monotonic(); cpu0=time.process_time()
    try:
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='': raise ValueError('Must explicitly disable CUDA')
        resource.setrlimit(resource.RLIMIT_CPU,(45,46))
        import torch
        torch.set_num_threads(1)
        from diagnose_pusht_goal_coverage import coverage
        marker=a.bank/'DONE.json'
        if sha(marker)!=a.done_sha: raise ValueError('Source DONE SHA mismatch')
        done=json.loads(marker.read_text())
        if not done['complete']: raise ValueError('Source incomplete')
        entries={r['path']:r for r in done['outputs']}
        def checked(name):
            f=a.bank/name; expected=entries[name]['sha256']
            if sha(f)!=expected: raise ValueError('Source SHA mismatch: '+name)
            return f
        summary=json.loads(checked('summary.json').read_text())
        if summary['episode']!=a.episode or summary['plan_count']!=64: raise ValueError('Source split/count')
        if summary['native_objective_sum_all_diffs'] or summary['native_objective_alpha']!=.1: raise ValueError('Objective contract')
        frozen=torch.load(checked('FROZEN_INPUTS.pt'),map_location='cpu',weights_only=False)
        if frozen['episode']!=a.episode or frozen['split']!='development_external': raise ValueError('Frozen split')
        goal=np.asarray(frozen['goal_state'],dtype=float).reshape(-1); enc=frozen['goal_encoded']
        arrays={k:[] for k in ['predicted_cost','actual_encoded_cost','coverage','default_coverage','xy_distance',
            'coverage_progress','xy_progress','weighted_forecast_mse','goal_cost_bound','recomputed_predicted_cost','recomputed_actual_cost']}
        discrepancies=[]; provenance=[]; shapes=None
        for k in range(64):
            path=checked(f'candidate-{k:03d}.pt'); row=json.loads(checked(f'candidate-{k:03d}.json').read_text())
            x=torch.load(path,map_location='cpu',weights_only=False)
            if x['episode']!=a.episode or x['candidate']!=k or x['split']!='development_external': raise ValueError('Candidate split/id')
            if not torch.equal(x['raw_actions'],frozen['raw_actions'][k]): raise ValueError('Frozen raw actions differ')
            pred=np.asarray(row['native_cost_by_horizon'],float); actual=np.asarray(row['actual_encoded_native_cost_by_horizon'],float)
            if pred.shape!=(7,) or actual.shape!=(7,): raise ValueError('Expected cached H0 plus H1..6')
            pred,actual=pred[1:],actual[1:]
            rp=native_cost(x['predicted_visual'],x['predicted_proprio'],enc['visual'],enc['proprio'])
            ra=native_cost(x['actual_visual'],x['actual_proprio'],enc['visual'],enc['proprio'])
            for stored,recomputed in [(pred,rp),(actual,ra)]:
                if not np.allclose(stored,recomputed,atol=2e-6,rtol=2e-6): raise ValueError('Native cost remeasurement floor exceeded')
                discrepancies.append(float(np.max(np.abs(stored-recomputed))))
            states=np.asarray(x['truth']['states'],dtype=float)
            if states.shape!=(31,7): raise ValueError('Exact30-step physical truth required')
            cov=np.array([coverage(s[2:5],goal[2:5]) for s in states[5::5]])
            default=np.array([coverage(s[2:5],[256.,256.,np.pi/4]) for s in states[5::5]])
            xy=np.linalg.norm(states[5::5,:4]-goal[:4],axis=1)
            ev=(x['predicted_visual'].double()-x['actual_visual'].double()).square().reshape(6,-1).mean(1)
            ep=(x['predicted_proprio'].double()-x['actual_proprio'].double()).square().reshape(6,-1).mean(1)
            err=(ev+.1*ep).numpy()
            bound=np.sqrt(err)*(np.sqrt(pred)+np.sqrt(actual))
            if np.any(np.abs(pred-actual)>bound+4e-6): raise ValueError('Weighted goal cost bound failed')
            values=[pred,actual,cov,default,xy,cov-coverage(states[0,2:5],goal[2:5]),
                np.linalg.norm(states[0,:4]-goal[:4])-xy,err,bound,rp,ra]
            for key,value in zip(arrays,values): arrays[key].append(value)
            provenance.append(dict(candidate=k,sha256=entries[path.name]['sha256'],family=row['family'],
                normalized_rms=row.get('normalized_rms'),raw_actions_sha256=hashlib.sha256(x['raw_actions'].numpy().tobytes()).hexdigest()))
            if k==0:
                shapes={key:list(x[key].shape) for key in ['predicted_visual','predicted_proprio','actual_visual','actual_proprio']}
                print(json.dumps(dict(stage='canary_passed',episode=a.episode,cost_floor=max(discrepancies),shapes=shapes)),flush=True)
            del x
        arrays={k:np.asarray(v,float) for k,v in arrays.items()}
        horizons=[dict(horizon=h+1,**horizon_summary(arrays['predicted_cost'][:,h],arrays['actual_encoded_cost'][:,h],
            arrays['coverage'][:,h],arrays['xy_distance'][:,h])) for h in range(6)]
        result=dict(complete=True,episode=a.episode,plan_count=64,independent_states=1,horizons=horizons,
            arrays={k:v.tolist() for k,v in arrays.items()},goal_state=goal.tolist(),shapes=shapes,
            cpu_seconds=time.process_time()-cpu0,wall_seconds=time.monotonic()-started,
            native_cost_recompute_maxabs=max(discrepancies),source_bank=str(a.bank),source_DONE_sha256=a.done_sha,
            frozen_inputs_sha256=entries['FROZEN_INPUTS.pt']['sha256'],candidate_provenance=provenance,
            native_objective_source_sha256=sha(a.native_objective_source),code_sha256=sha(__file__),
            protocol=PROTOCOL)
        write_json(a.output/'summary.json',result)
        for source in [Path(__file__),Path(__file__).with_name('diagnose_pusht_goal_coverage.py'),a.native_objective_source,
                Path(__file__).with_name('test_objective_alignment_v1.py')]:
            shutil.copy2(source,a.output/source.name)
        outputs=[dict(path=f.name,sha256=sha(f),bytes=f.stat().st_size) for f in sorted(a.output.iterdir()) if f.is_file()]
        write_json(a.output/'DONE.json',dict(complete=True,episode=a.episode,outputs=outputs))
        print(json.dumps(dict(stage='complete',episode=a.episode,cpu_seconds=result['cpu_seconds'],
            max_cost_floor=result['native_cost_recompute_maxabs'],H6=horizons[-1])),flush=True)
    except BaseException as e:
        write_json(a.output/'FAILED.json',dict(complete=False,error=repr(e),cpu_seconds=time.process_time()-cpu0))
        raise


if __name__=='__main__':main()
