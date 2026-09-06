#!/usr/bin/env python3
"""CPU-only exact-current-input comparison of frozen imagined-history clamps.

No world-model, simulator, fitted readout, or checkpoint construction occurs.
Effective arguments are reconstructed using the archived producer's exact hook;
the old files did not directly save every arm's post-hook input arguments.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

import numpy as np
import torch

ARMS = ('native', 'central_state_context', 'central_state_and_past_actions')
NONCENTRAL = [0, 1, 3, 4]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def exact(a, b, label):
    if not torch.equal(a, b):
        raise ValueError('Exact guard: '+label)


def check_nulls(predictions):
    for key in ('visual', 'proprio'):
        base = predictions[ARMS[0]][key]
        if base.shape[:2] != (7, 5):
            raise ValueError('Expected initial plus six horizons and five path points')
        for arm in ARMS[1:]:
            value = predictions[arm][key]
            exact(value[:2], base[:2], 'initial/H1 '+key)
            exact(value[:, 2], base[:, 2], 'central path '+key)
    return True


def reconstructed_arguments(payload, horizon, clamp):
    """Apply actual archived hook to replayed pre-hook tensors, preserving strides."""
    h = horizon
    reference = payload['central_predictor_inputs'][h-1]
    width = reference[1].shape[1]
    actions = payload['normalized_actions'][:, h-width:h]
    exact(reference[1][0], actions[2], 'saved central action chronology')
    results = []
    for arm in ARMS[1:]:
        predictions = payload['predictions'][arm]
        values = (predictions['visual'][h-width:h].transpose(0, 1), actions,
                  predictions['proprio'][h-width:h].transpose(0, 1))
        arguments = []
        for value, spec in zip(values, payload['context_shapes'][arm][h-1]):
            padded = torch.cat((value, value[2:3].repeat(2, *([1]*(value.ndim-1)))), 0)
            scratch = torch.empty_strided(spec['shape'], spec['stride'], dtype=value.dtype)
            scratch.copy_(padded)
            arguments.append(scratch)
        changed = clamp(tuple(arguments), reference, past_actions=arm == ARMS[2])
        if h == 1:
            for original, replay in zip(arguments, changed):
                exact(original, replay, 'H1 hook is identity')
        results.append(changed)
    left, right = results
    for index in (0, 2):
        exact(left[index], right[index], 'full state context same')
        exact(left[index], reference[index].expand_as(left[index]), 'full frozen central state')
    exact(left[1][:, -1], right[1][:, -1], 'current action same')
    exact(left[1][:5, -1], payload['normalized_actions'][:, h-1], 'current original action')
    exact(left[1][2], right[1][2], 'central past action unchanged')
    delta = left[1][:5, :-1]-right[1][:5, :-1]
    return dict(full_state_context_exact=True, current_action_exact=True,
                context_width=width, previous_action_changed=bool(torch.count_nonzero(delta)),
                past_action_maxabs=float(delta.abs().max()) if delta.numel() else 0.,
                past_action_mse=float(delta.double().square().mean()) if delta.numel() else 0.)


def contrast(left, right, native):
    delta = (right.double()-left.double()).flatten(1)
    native_delta = (native.double()-native[2:3].double()).flatten(1)
    per_t = delta.square().mean(1)
    spread = native_delta[NONCENTRAL].square().mean()
    effect = per_t[NONCENTRAL].mean()
    return dict(mse_by_t=per_t.tolist(), noncentral_mse=float(effect),
                maxabs=float(delta.abs().max()), native_noncentral_variation_mse=float(spread),
                effect_over_native_variation=float(effect/spread) if float(spread)>0 else None,
                central_exact=bool(torch.count_nonzero(delta[2]) == 0))


def costs(encodings, goal, alpha=.1):
    parts = {}
    for key in ('visual', 'proprio'):
        value = encodings[key]
        parts[key] = (value-goal[key]).square().mean(tuple(range(2, value.ndim)))
    return parts['visual']+alpha*parts['proprio']


def rank_comparison(left, right):
    a, b = np.asarray(left), np.asarray(right)
    if a.shape != (5,) or b.shape != (5,):
        raise ValueError('Five predeclared path points required')
    pairs = [(i, j) for i in range(5) for j in range(i)]
    return dict(left_cost=a.tolist(), right_cost=b.tolist(), left_order=np.argsort(a, kind='stable').tolist(),
                right_order=np.argsort(b, kind='stable').tolist(), left_choice=int(a.argmin()),
                right_choice=int(b.argmin()), choice_changed=bool(a.argmin()!=b.argmin()),
                rank_order_changed=bool(not np.array_equal(np.argsort(a, kind='stable'), np.argsort(b, kind='stable'))),
                pair_order_changes=int(sum(np.sign(a[i]-a[j]) != np.sign(b[i]-b[j]) for i, j in pairs)),
                left_min_ties=int(np.sum(a==a.min())), right_min_ties=int(np.sum(b==b.min())),
                maxabs_cost_difference=float(np.abs(a-b).max()))


def grouped_summary(rows):
    groups = []
    for radius in (1, 4):
        for h in range(1, 7):
            states = []
            for ep in range(4):
                subset = [r for r in rows if (r['episode'], r['radius'], r['horizon']) == (ep, radius, h)]
                if len(subset) != 4:
                    raise ValueError('Four dependent directions per source state required')
                item = dict(episode=ep, dependent_paths=4,
                            changed_choices=sum(r['cost']['choice_changed'] for r in subset),
                            changed_rank_orders=sum(r['cost']['rank_order_changed'] for r in subset),
                            maxabs_cost_difference=max(r['cost']['maxabs_cost_difference'] for r in subset))
                for key in ('visual', 'proprio', 'weighted'):
                    effect = statistics.mean(r[key]['noncentral_mse'] for r in subset)
                    scale = statistics.mean(r[key]['native_noncentral_variation_mse'] for r in subset)
                    item[key] = dict(mse=effect, native_variation_mse=scale,
                                     fraction=effect/scale if scale>0 else None)
                states.append(item)
            aggregate = {key:dict(mean_mse=statistics.mean(s[key]['mse'] for s in states),
                                 state_fractions=[s[key]['fraction'] for s in states],
                                 mean_fraction=statistics.mean(s[key]['fraction'] for s in states)
                                 if all(s[key]['fraction'] is not None for s in states) else None)
                         for key in ('visual', 'proprio', 'weighted')}
            groups.append(dict(radius=radius, horizon=h, n_states=4, states=states,
                               states_with_choice_changes=sum(s['changed_choices']>0 for s in states),
                               states_with_rank_changes=sum(s['changed_rank_orders']>0 for s in states), **aggregate))
    return groups


def validate_sources(done, protocol):
    if not done['complete'] or done['paths'] != 32 or done['held_access'] is not False:
        raise ValueError('Need complete 32-development-path receipt before loading')
    if [r['key'] for r in protocol['rows']] != [f'near-dev-{ep:03}' for ep in range(4)]:
        raise ValueError('Only four frozen original development states')
    if any(r['split'] != 'development_external' for r in protocol['rows']):
        raise ValueError('Held source forbidden before tensor loading')
    expected = {f'near-dev-{ep:03}-pair{pair}-radius{radius}.pt'
                for ep in range(4) for pair in range(4) for radius in (1, 4)}
    entries = {r['path']:r for r in done['outputs'] if r['path'].endswith('.pt')}
    if set(entries) != expected:
        raise ValueError('Frozen path coverage changed')
    return entries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'producer_code', 'bank', 'repo', 'output'):
        p.add_argument('--'+name.replace('_', '-'), type=Path, required=True)
    args = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Explicit CUDA invisibility required')
    torch.set_num_threads(1)
    started = time.monotonic()
    done = json.loads((args.source/'DONE.json').read_text())
    protocol = json.loads((args.source/'protocol.json').read_text())
    entries = validate_sources(done, protocol)
    producer = args.producer_code/'run_action_history_clamp.py'
    if sha(producer) != protocol['script_sha256']:
        raise ValueError('Archived clamp producer hash changed')
    sys.path.insert(0, str(args.producer_code))
    from run_action_history_clamp import clamp_arguments
    import yaml
    from tensordict import TensorDict
    config_path = Path(done['provenance']['config'])
    config = yaml.safe_load(config_path.read_text())
    setting = config['planner']['planning_objective']
    if setting != dict(objective_type='L2', sum_all_diffs=False, alpha=.1):
        raise ValueError('Native goal objective changed')
    if not config['model_kwargs']['pretrain_kwargs']['action_encoder']['action_encoder_inpred']:
        raise ValueError('Stored normalized action features no longer native predictor inputs')
    objective_file = args.repo/'evals/simu_env_planning/planning/planning/objectives.py'
    spec = importlib.util.spec_from_file_location('cached_native_objective', objective_file)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        write(args.output/'protocol.json', dict(complete=False,source=str(args.source), source_done_sha256=sha(args.source/'DONE.json'),
              producer_sha256=sha(producer), analysis_sha256=sha(__file__), native_objective_sha256=sha(objective_file),
              config_sha256=sha(config_path), n_states=4, dependent_paths=32, horizons=list(range(1,7)),
              comparison=list(ARMS[1:]), no_model_calls=True, no_simulator=True, no_CEM=True, no_held=True,
              input_verification='Reconstructed effective inputs using saved central inputs, every arm prediction history, original normalized actions, recorded strides and exact archived hook; original post-hook full arguments were not directly saved.',
              interpretation='Native finite-history conditional sufficiency, not physical belief-state identifiability or steering utility',
              max_seconds=600, process_pid=os.getpid()))
        rows=[]; source_hashes=[]; outputs=[]
        for bank_row in protocol['rows']:
            bank_path=args.bank/bank_row['path']
            if sha(bank_path)!=bank_row['sha256']:raise ValueError('Original goal bank hash before load')
            bank=torch.load(bank_path,map_location='cpu',weights_only=False)
            if bank['row']['split']!='development_external':raise ValueError('Wrong goal split')
            goal=bank['goal_encoded'];objective=module.ReprTargetDistMPCObjective({},TensorDict(goal,batch_size=[]),alpha=.1)
            ep=bank_row['source_id']
            for pair in range(4):
                for radius in (1,4):
                    if time.monotonic()-started>590:raise RuntimeError('CPU budget before next source')
                    name=f'near-dev-{ep:03}-pair{pair}-radius{radius}.pt';source=args.source/name
                    if sha(source)!=entries[name]['sha256']:raise ValueError('Path SHA before tensor load')
                    value=torch.load(source,map_location='cpu',weights_only=False)
                    if value['source_sha256']!=bank_row['sha256'] or value['key']!=bank_row['key']:raise ValueError('Source provenance mismatch')
                    if value['wm_ctxt_window']!=2 or value['sample_t']!=(-1.,-.5,0.,.5,1.):raise ValueError('Fixed window/path changed')
                    exact(value['raw_actions'][2],bank['raw_actions'][0],'original central raw actions')
                    exact(value['normalized_actions'][2],bank['normalized_actions'][0],'original central normalized actions')
                    check_nulls(value['predictions'])
                    for site in ('P1_H1','P3_H1','P5_H1'):
                        for arm in ARMS[1:]:exact(value['residuals'][arm][site],value['residuals']['native'][site],'H1 residual '+site)
                    cost={}
                    for arm in ARMS:
                        cost[arm]=objective(TensorDict(value['predictions'][arm],batch_size=[]),None,keepdims=True)
                        exact(cost[arm],costs(value['predictions'][arm],goal),'same native objective numerical implementation')
                    for h in range(1,7):
                        inputs=reconstructed_arguments(value,h,clamp_arguments)
                        current=dict(episode=ep,pair=pair,radius=radius,horizon=h,inputs=inputs,
                                     cost=rank_comparison(cost[ARMS[1]][h],cost[ARMS[2]][h]),native_cost=cost['native'][h].tolist())
                        for key in ('visual','proprio'):
                            current[key]=contrast(*(value['predictions'][arm][key][h] for arm in (*ARMS[1:],ARMS[0])))
                        v,q=current['visual'],current['proprio'];eff=v['noncentral_mse']+.1*q['noncentral_mse'];var=v['native_noncentral_variation_mse']+.1*q['native_noncentral_variation_mse']
                        current['weighted']=dict(noncentral_mse=eff,native_noncentral_variation_mse=var,effect_over_native_variation=eff/var if var>0 else None)
                        rows.append(current)
                    source_hashes.append(dict(path=str(source),sha256=entries[name]['sha256']))
                    print(json.dumps(dict(event='cached_path_analyzed',episode=ep,pair=pair,radius=radius,seconds=time.monotonic()-started)),flush=True)
                    del value
            del bank
        report=dict(complete=True,rows=rows,groups=grouped_summary(rows),n_states=4,paths=32,dependent_horizon_comparisons=192,
                    all_H1_and_central_exact=True,same_full_state_and_current_action_reconstructed_exact=True,
                    native_cost_function_parity_exact=True,source_paths=source_hashes,seconds=time.monotonic()-started,
                    cuda_initialized=torch.cuda.is_initialized(),no_held=True,no_model_calls=True,
                    limitations=['Finite native previous-action conditioning only; no claim about a true POMDP belief state.',
                                 'Both arms clamp full state histories identically; cannot isolate prior-state memory.',
                                 'Original per-arm post-hook inputs were reconstructed, not directly recorded.',
                                 'P3 may already carry history; this does not disprove sufficiently rich residual-state charts.',
                                 'Ranks/choices refer to five fixed action-path points, not a CEM search or physical task outcome.',
                                 'Four source states; directions/radii/horizons are dependent. No significance or power claim.'])
        if report['cuda_initialized']:raise RuntimeError('Forbidden CUDA initialization')
        write(args.output/'summary.json',report)
        for name in ('summary.json','protocol.json'):
            target=args.output/name;outputs.append(dict(path=name,sha256=sha(target),bytes=target.stat().st_size))
        write(args.output/'DONE.json',dict(complete=True,outputs=outputs,n_states=4,paths=32,seconds=time.monotonic()-started,
                                         no_model_calls=True,cuda_initialized=False,source_done_sha256=sha(args.source/'DONE.json')))
    except Exception:
        write(args.output/'FAILED.json',dict(complete=False,error=traceback.format_exc()));raise


if __name__=='__main__':main()
