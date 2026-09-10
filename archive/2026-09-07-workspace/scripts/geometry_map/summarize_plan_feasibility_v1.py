#!/usr/bin/env python3
"""Same-horizon fixed-plan feasibility and native restart summaries, CPU only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from protocol import file_sha256, write_json_atomic


def ranks(x):
    x = np.asarray(x, dtype=float)
    return np.array([np.sum(x < v) + .5 * (np.sum(x == v) - 1) for v in x])


def rank_agreement(predicted, observed):
    p, o = (np.asarray(x, dtype=float) for x in (predicted, observed))
    if p.ndim != 1 or p.shape != o.shape or len(p) < 2:
        raise ValueError('Aligned candidate vectors required')
    a, b = ranks(p), ranks(o)
    pairs = [(i, j) for i in range(len(p)) for j in range(i) if p[i] != p[j] and o[i] != o[j]]
    return {'spearman': float(np.corrcoef(a, b)[0, 1]) if np.std(a) and np.std(b) else None,
            'discordant_pair_fraction': float(np.mean([(p[i]-p[j])*(o[i]-o[j]) < 0 for i,j in pairs])) if pairs else None,
            'non_tied_pairs': len(pairs), 'predicted_argmin_index': int(p.argmin()),
            'observed_argmin_index': int(o.argmin())}


def array(x):
    return x.detach().cpu().double().numpy() if hasattr(x, 'detach') else np.asarray(x, dtype=float)


def forecast_error(value):
    """Native feature MSE at each observed future, excluding context."""
    channels=forecast_error_by_channel(value)
    return channels['visual']+.1*channels['proprio']


def forecast_error_by_channel(value):
    result = {}
    for key in ('visual', 'proprio'):
        pred = array(value['predicted_'+key])[1:].reshape(6, -1)
        actual = array(value['actual_encoded'][key]).reshape(6, -1)
        if pred.shape != actual.shape:
            raise ValueError('Unaligned native future shapes: '+key)
        result[key] = np.mean((pred-actual)**2, axis=1)
    return result


def summarize(directory):
    import torch
    receipt = json.loads((directory/'DONE.json').read_text())
    if not receipt['complete'] or receipt['held_sources_opened'] or receipt['new_cem_calls']:
        raise ValueError('Completed frozen fixed-plan development run required')
    values, sources, rows = {}, [], []
    for entry in receipt['outputs']:
        path = directory/entry['path']
        if file_sha256(path) != entry['sha256']:
            raise ValueError('Source hash mismatch: '+str(path))
        v = torch.load(path, map_location='cpu', weights_only=False)
        values[entry['plan']] = v
        sources.append(entry | {'root': str(directory)})
        if entry['plan'] == 'baseline_xyz_clipped':
            continue
        rows.append({'episode': receipt['episode'], 'plan': entry['plan'],
                     'predicted_native_goal_cost_h1_to_h6': array(v['predicted_native_goal_cost_by_step'])[1:].reshape(6).tolist(),
                     'actual_native_goal_cost_h1_to_h6': array(v['actual_native_goal_cost_h1_to_h6']).tolist(),
                     'native_forecast_mse_h1_to_h6': forecast_error(v).tolist(),
                     'physical_goal_progress_h1_to_h6_mm': (1000*array(v['goal_progress_h1_to_h6_m'])).tolist(),
                     'actual_goal_distance_h1_to_h6_m': array(v['actual_goal_distance_h1_to_h6_m']).tolist(),
                     'ever_success': v['ever_success'], 'final_success': v['final_success'], 'prefix15_exact': v['prefix15_exact']})
    if len(rows) != 9 or len(values) != 10:
        raise ValueError('All nine plans and clipping control required')
    base = values['unsteered']; clip = values['baseline_xyz_clipped']
    if not clip['full30_raw_clipped_equivalence_exact']:
        raise ValueError('Exact full30 physical equivalence missing')
    for key in ('states', 'frames', 'proprios', 'rewards'):
        if not np.array_equal(array(base[key]), array(clip[key])):
            raise ValueError('Copied raw/clipped equivalence disagrees: '+key)
    bcost = array(base['predicted_native_goal_cost_by_step'])[1:].reshape(6)
    ccost = array(clip['predicted_native_goal_cost_by_step'])[1:].reshape(6)
    berr, cerr = forecast_error(base), forecast_error(clip)
    bchannels,cchannels=forecast_error_by_channel(base),forecast_error_by_channel(clip)
    clip_summary = {'episode': receipt['episode'], 'full30_physics_pixels_exact': True,
                    'raw_predicted_native_goal_cost_h1_to_h6': bcost.tolist(),
                    'clipped_predicted_native_goal_cost_h1_to_h6': ccost.tolist(),
                    'raw_forecast_mse_h1_to_h6': berr.tolist(), 'clipped_forecast_mse_h1_to_h6': cerr.tolist(),
                    'clipped_to_raw_forecast_mse_ratio_h1_to_h6': (cerr/np.maximum(berr,1e-15)).tolist(),
                    'per_channel_forecast_error':{k:{'raw_mse_h1_to_h6':bchannels[k].tolist(),
                                                   'clipped_mse_h1_to_h6':cchannels[k].tolist(),
                                                   'clipped_to_raw_ratio_h1_to_h6':(cchannels[k]/np.maximum(bchannels[k],1e-15)).tolist()} for k in bchannels},
                    'forecast_channel_maxabs_raw_vs_clipped': {k:float(np.max(np.abs(array(base['predicted_'+k])-array(clip['predicted_'+k])))) for k in ('visual','proprio')},
                    'raw_xyz_clipped_fraction': float(np.mean(np.abs(array(base['raw_actions'])[:,:3]) > 1)),
                    'gripper_unchanged_exact': bool(np.array_equal(array(base['raw_actions'])[:,3], array(clip['raw_actions'])[:,3]))}
    names = [r['plan'] for r in rows]
    comparisons = []
    for h in range(6):
        predicted = [r['predicted_native_goal_cost_h1_to_h6'][h] for r in rows]
        observed = [r['actual_native_goal_cost_h1_to_h6'][h] for r in rows]
        physical = [r['actual_goal_distance_h1_to_h6_m'][h] for r in rows]
        comparisons.append({'horizon':h+1, 'raw_step':5*(h+1),
                            'native_goal_rank':rank_agreement(predicted,observed),
                            'physical_goal_rank':rank_agreement(predicted,physical)})
    base_row = next(r for r in rows if r['plan']=='unsteered')
    for r in rows:
        r['physical_progress_vs_baseline_h6_mm'] = r['physical_goal_progress_h1_to_h6_mm'][-1]-base_row['physical_goal_progress_h1_to_h6_mm'][-1]
    return {'episode':receipt['episode'], 'plan_order':names, 'rows':rows, 'ranks_by_horizon':comparisons,
            'clipping':clip_summary, 'sources':sources, 'seconds':receipt['seconds']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dirs',nargs='+',type=Path,required=True)
    p.add_argument('--restart-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    states=[summarize(d) for d in a.input_dirs]
    if sorted(x['episode'] for x in states) != [0,4,7]:raise ValueError('Fixed starts0/4/7 required')
    restarts=[]
    for path in sorted(a.restart_root.glob('worker-*/results-v1/DONE.json')):
        d=json.loads(path.read_text())
        if not d['complete'] or d['restart_seeds'] != [817,1801] or not d['all_seeds_retained']:raise ValueError('Frozen restart receipt required')
        for r in d['outputs']:
            if file_sha256(path.parent/r['path'])!=r['sha256']:raise ValueError('Restart hash mismatch')
            restarts.append(r|{'root':str(path.parent)})
    if {(r['episode'],r['seed']) for r in restarts}!={(e,s) for e in (0,4,7) for s in (817,1801)}:raise ValueError('All six restart controls required')
    arms=states[0]['plan_order'];aggregate={}
    for arm in arms:
        rows=[next(r for r in s['rows'] if r['plan']==arm) for s in states]
        aggregate[arm]={'n_starts':3,'episodes':[s['episode'] for s in states],
                        'mean_h6_progress_vs_baseline_mm':float(np.mean([r['physical_progress_vs_baseline_h6_mm'] for r in rows])),
                        'per_start_h6_progress_vs_baseline_mm':[r['physical_progress_vs_baseline_h6_mm'] for r in rows],
                        'ever_success_count':sum(r['ever_success'] for r in rows)}
    out={'complete':True,'states':states,'h6_equal_start_aggregate':aggregate,'restart_controls':restarts,
         'scope':'Three fixed development starts; nine related generated plans are not independent episodes. Native restarts have unmatched initial populations and only15-step physical outcomes. H6 plan extensions are30-step outcomes, not full closed-loop efficacy. XYZ projection keeps gripper unchanged; altered fixed forecasts with identical physics identify model feasibility mismatch, not proof a clipped CEM planner improves performance.'}
    write_json_atomic(a.output_dir/'feasibility.json',out)
    write_json_atomic(a.output_dir/'DONE.json',{'complete':True,'script_sha256':file_sha256(Path(__file__)),
                      'outputs':[{'path':'feasibility.json','sha256':file_sha256(a.output_dir/'feasibility.json')}]})
    print(json.dumps({'complete':True,'aggregate':aggregate,'clipping':[s['clipping'] for s in states]}),flush=True)


if __name__=='__main__':main()
