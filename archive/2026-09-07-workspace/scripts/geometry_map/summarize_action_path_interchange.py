#!/usr/bin/env python3
"""Initial-state-grouped fit-withheld interior interchange, with effect denominators."""
import argparse
import hashlib
import json
from pathlib import Path
from summarize_action_path_curvature import average,episode_interval


def effect_fraction(error,effect,threshold=1e-12):
    return None if effect<=threshold else 1-error/effect


def summarize(reports,denominator_threshold=1e-12):
    expected={(f'near-dev-{i:03}',p,r,t) for i in range(4) for p in range(4) for r in (1,4) for t in (-.25,.25)}
    actual={(r['meta']['key'],r['meta']['pair'],r['meta']['radius'],r['meta']['target_t']) for r in reports}
    if len(reports)!=64 or actual!=expected or any(not r['complete'] for r in reports):raise ValueError('All64 fixed targets required')
    views=[]
    for radius in (1,4):
        for h in (1,3,6):
            selected=[(r['meta']['key'],next(v for v in r['rows'] if v['intervention_horizon']==h)) for r in reports if r['meta']['radius']==radius]
            keys=sorted({k for k,_ in selected})
            def grouped(fn):return [average(fn(row) for key2,row in selected if key2==key) for key in keys]
            names=[r['method'] for r in selected[0][1]['methods']]
            def get(row,name):return next(m for m in row['methods'] if m['method']==name)
            methods={}
            for name in names:
                methods[name]={m:episode_interval(grouped(lambda row,m=m:get(row,name)[m])) for m in
                    ('midpoint_l2_error','downstream_oracle_patch_mse_after_patch')}
                fractions=[effect_fraction(get(row,name)['downstream_oracle_patch_mse_after_patch'],row['actual_donor_patch_vs_native_mse_after_patch'],denominator_threshold) for _,row in selected]
                methods[name]['tiny_or_zero_donor_effect_rows']=sum(f is None for f in fractions)
                if all(f is not None for f in fractions):
                    methods[name]['fraction_of_nonzero_causal_effect_reproduced']=episode_interval(grouped(lambda row:effect_fraction(get(row,name)['downstream_oracle_patch_mse_after_patch'],row['actual_donor_patch_vs_native_mse_after_patch'],denominator_threshold)))
                else:methods[name]['fraction_of_nonzero_causal_effect_reproduced']=None
            contrasts={}
            for other in ('linear_equal_data','reparameterized_chord','reflected_curvature','near_chord'):
                contrasts['cubic_minus_'+other]=episode_interval(grouped(lambda row,other=other:get(row,'cubic_equal_data')['downstream_oracle_patch_mse_after_patch']-get(row,other)['downstream_oracle_patch_mse_after_patch']))
            views.append(dict(block=3,horizon=h,radius=radius,
                actual_donor_causal_effect_mse=episode_interval(grouped(lambda row:row['actual_donor_patch_vs_native_mse_after_patch'])),
                methods=methods,contrasts=contrasts))
    return dict(complete=True,targets=64,initial_states=4,views=views,denominator_threshold=denominator_threshold,
        all_self_patch_exact=all(v['exact_self_patch'] for r in reports for v in r['rows']),
        limitations=['Four original development states; average8 directional/interior targets perstate before intervals',
            'Reported intervals are descriptive and exploratory, not confirmation or multiple-comparison corrected',
            'Donor-patch causal reference uses central-action recipient, not native donor-action rollout',
            'Fraction1-MSE(method,donorpatch)/MSE(native,donorpatch) is a fidelity ratio, not physical success',
            'Tiny donor-effect denominators flagged against max(1e-12,100 times measured propagated numerical floor)',
            'Targets are withheld from each curve fit, not globally unseen: radius4 times±.25 reuses prior radius1 endpoints±1',
            'Radii are dependent sensitivity conditions, not independent replications; only1x±.25 is globally new versus prior five-point paths',
            'The near_chord local-linear estimator is a strong low-data comparator and is reported alongside equal-data OLS',
            'Actual donor-action physical accuracy requires separate replay truth, not cached central-action truth'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--floor',type=Path);a=p.parse_args()
    done=json.loads((a.input/'DONE.json').read_text());reports=[]
    if not done['complete']:raise ValueError('Incomplete interchange')
    for row in done['outputs']:
        if not row['path'].endswith('.json'):continue
        data=(a.input/row['path']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('Report hash failed')
        reports.append(json.loads(data))
    threshold=1e-12
    if a.floor:
        floor=json.loads(a.floor.read_text())
        if not floor['complete'] or len(floor['rows'])!=12:raise ValueError('All12 measured propagated floor cases required')
        threshold=max(threshold,100*max(r['propagated_fullspatial_mse_after_patch'] for r in floor['rows']))
    out=summarize(reports,threshold);out['seconds']=done['seconds'];out['source_done_sha256']=hashlib.sha256((a.input/'DONE.json').read_bytes()).hexdigest()
    if a.floor:out['propagated_floor_sha256']=hashlib.sha256(a.floor.read_bytes()).hexdigest()
    if a.output.exists():raise FileExistsError('Immutable summary exists')
    a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(complete=True,path=str(a.output),views=len(out['views']))))


if __name__=='__main__':main()
