#!/usr/bin/env python3
"""State-grouped natural versus observed-past geometry and prediction errors."""
import argparse
import hashlib
import json
from pathlib import Path
from summarize_action_path_curvature import average,episode_interval


def summarize(reports):
    expected={(f'near-dev-{e:03}',p,r) for e in range(4) for p in range(4) for r in (1,4)}
    if len(reports)!=32 or {(r['key'],r['pair'],r['radius']) for r in reports}!=expected or any(not r['complete'] for r in reports):raise ValueError('All32fixed paths required')
    views=[];forecast_views=[]
    for radius in (1,4):
        panel=[r for r in reports if r['radius']==radius];keys=sorted({r['key'] for r in panel})
        def grouped(fn):return [average(fn(r) for r in panel if r['key']==key) for key in keys]
        for h in range(1,7):
            forecast_views.append(dict(radius=radius,horizon=h,
                native_actual_mse=episode_interval(grouped(lambda r:r['native_actual_future_mse_by_horizon'][h-1])),
                actual_past_actual_mse=episode_interval(grouped(lambda r:r['actual_past_actual_future_mse_by_horizon'][h-1])),
                actual_past_minus_native_mse=episode_interval(grouped(lambda r:r['actual_past_actual_future_mse_by_horizon'][h-1]-r['native_actual_future_mse_by_horizon'][h-1])),
                native_history_layout_floor_mse=episode_interval(grouped(lambda r:r['native_history_replay_visual_mse_by_horizon'][h-1]))))
        for block in (1,3,5):
            for h in (1,3,6):
                def site(r):return next(v for v in r['rows'] if v['block']==block and v['horizon']==h)
                modes={}
                metrics=('path_length','chord_length','path_chord_ratio','max_orthogonal_fraction_of_chord')
                for mode in ('native','actual_past_context','native_history_replay'):
                    modes[mode]={m:episode_interval(grouped(lambda r,m=m,mode=mode:site(r)[mode][m])) for m in metrics}
                contrasts={m:episode_interval(grouped(lambda r,m=m:site(r)['actual_past_context'][m]-site(r)['native'][m])) for m in metrics}
                views.append(dict(radius=radius,block=block,horizon=h,modes=modes,actual_past_minus_native=contrasts,
                    maximum_native_history_replay_residual_floor=max(site(r)['native_history_replay_residual_maxabs'] for r in panel)))
    return dict(complete=True,paths=32,initial_states=4,geometry_views=views,forecast_views=forecast_views,
        all_H1_exact=all(r['H1_exact'] for r in reports),all_actions_exact=all(r['actions_exact'] for r in reports),
        maximum_native_history_replay_visual_maxabs=max(r['native_history_replay_visual_maxabs'] for r in reports),
        no_future_leakage=all(v['largest_actual_frame']<v['target_frame'] for r in reports for v in r['temporal_records']),
        limitations=['Actual past encoded observations are privileged offline information, not prospective runtime input',
            'Both actual visual and proprio throughH−1 replaced; action-history/current-action conditioning unchanged',
            'Actual-context curvature may reflect real state variation, encoder geometry, contact or one-step dynamics; not proof of a representation manifold',
            'A reduction versus free imagination is consistent with feedback-error contribution, not proof that all curvature is error',
            'Four initial states, dependent directions/horizons/radii; descriptive episode bootstrap only',
            'Lower actual encoded-frame prediction error is not demonstrated improvement in selected actions or robot success'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    done=json.loads((a.input/'DONE.json').read_text());reports=[]
    if not done['complete']:raise ValueError('Incomplete run')
    for row in done['outputs']:
        if not row['path'].endswith('.json'):continue
        data=(a.input/row['path']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('Compact source checksum failed')
        reports.append(json.loads(data))
    result=summarize(reports);result['seconds']=done['seconds'];result['source_done_sha256']=hashlib.sha256((a.input/'DONE.json').read_bytes()).hexdigest()
    if a.output.exists():raise FileExistsError('Immutable output exists')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(complete=True,views=18,paths=32)))


if __name__=='__main__':main()
