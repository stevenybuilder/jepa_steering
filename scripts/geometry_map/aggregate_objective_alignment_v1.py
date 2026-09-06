#!/usr/bin/env python3
"""Equal-state aggregate of immutable objective-alignment receipts, no refitting."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def summarize(values):
    valid=[float(v) for v in values if v is not None]
    return dict(per_state=values,defined_states=len(valid),
        mean=None if not valid else float(np.mean(valid)),range=None if not valid else [min(valid),max(valid)])


def aggregate(paths):
    states=[];sources=[]
    for path in paths:
        marker=path.parent/'DONE.json';d=json.loads(marker.read_text())
        expected=next(r['sha256'] for r in d['outputs'] if r['path']==path.name)
        if not d['complete'] or sha(path)!=expected:raise ValueError('Source receipt fails')
        report=json.loads(path.read_text());states.append(report)
        sources.append(dict(path=str(path),sha256=sha(path),DONE_sha256=sha(marker)))
    states.sort(key=lambda d:d['episode'])
    if [s['episode'] for s in states]!=[0,1,2,3]:raise ValueError('Exactly four seen development states required')
    rows=[]
    for h in range(6):
        hs=[s['horizons'][h] for s in states]
        row=dict(horizon=h+1,per_state=hs)
        for key in ['native_predicted','actual_encoding_cost_oracle','requested_coverage_oracle','xy_oracle']:
            row[key]={f:summarize([x[key][f] for x in hs]) for f in
                ['requested_goal_coverage','coverage_regret','xy_distance','xy_regret','winner_runner_up_margin']}
            row[key]['selected_indices']=[x[key]['index'] for x in hs]
        for key in ['predicted_vs_coverage','actual_encoded_vs_coverage','predicted_vs_xy','actual_encoded_vs_xy','predicted_vs_actual_cost']:
            row[key]={f:summarize([x[key][f] for x in hs]) for f in ['spearman_cost_negative_vs_benefit','agreement']}
            row[key]['margin_bins_equal_state']=[dict(lower=hs[0][key]['bins'][b]['lower'],upper=hs[0][key]['bins'][b]['upper'],
                agreement=summarize([x[key]['bins'][b]['agreement'] for x in hs]),
                per_state_informative_pairs=[x[key]['bins'][b]['informative_pair_count'] for x in hs]) for b in range(4)]
        gains=[x['actual_encoding_cost_oracle']['requested_goal_coverage']-x['native_predicted']['requested_goal_coverage'] for x in hs]
        remaining=[x['actual_encoding_cost_oracle']['coverage_regret'] for x in hs]
        native=[x['native_predicted']['coverage_regret'] for x in hs]
        err=np.asarray(native)-np.asarray(gains)-np.asarray(remaining)
        if np.max(np.abs(err))>1e-12:raise AssertionError('Regret accounting')
        row['coverage_regret_accounting']=dict(forecast_cost_correction_effect=summarize(gains),
            remaining_metric_selection_regret=summarize(remaining),native_regret=summarize(native),
            identity_maxabs=float(np.max(np.abs(err))),
            scope='descriptive exact accounting, not causal mediation; correction effect may be negative')
        rows.append(row)
    return dict(complete=True,independent_states=4,plans_per_state=64,primary_horizon=6,horizons=rows,sources=sources,
        total_cpu_seconds=sum(s['cpu_seconds'] for s in states),GPU_model_simulator_calls=0,
        timing_scope='sum of timed analyzer regions; excludes initial NumPy import and final artifact serialization; entire processes each limited to45 CPU seconds',
        novelty_scope='H6 terminal objective alignment repeats prior same-bank diagnostics; new contribution is H1..H6 plus pairwise margins/ties and regret accounting, not a new independent discovery',
        max_native_cost_remeasurement_floor=max(s['native_cost_recompute_maxabs'] for s in states),
        no_confirmatory_claim=True,no_new_metric=True,
        literature=[dict(url='https://arxiv.org/pdf/2608.18746v1',
            relevance='Plan-Real and CEM-stage rank diagnostics are prior art. Ours reuses fixed multiscale banks, not stagewise CEM samples; requested coverage and XY are distinct from their full7D L2 proxy.'),
            dict(url='https://arxiv.org/pdf/2608.12939v1',relevance='Cost-gap bounds protect model-cost choices, not environment utility. Our actual-encoding comparison needs known futures and is not online ACPC.'),
            dict(url='https://arxiv.org/pdf/2605.05115v1',relevance='MountainCar behavior probabilities are encoder-centroid distance softmax, not an independent control objective; response geometry does not establish robot-goal metric validity.')],
        protocol=states[0]['protocol'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',nargs=4,type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=aggregate(a.inputs)
    with a.output.open('x') as f:json.dump(r,f,indent=2,allow_nan=False)
    print(json.dumps(dict(path=str(a.output),sha256=sha(a.output),cpu_seconds=r['total_cpu_seconds'],
        H6=r['horizons'][-1]['coverage_regret_accounting']),indent=2))


if __name__=='__main__':main()
