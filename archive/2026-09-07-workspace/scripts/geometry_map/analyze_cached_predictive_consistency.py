#!/usr/bin/env python3
"""Cached same-action predictive consistency; no trained ACPC objective or utility claim."""
import argparse
import json
from pathlib import Path
import time
import torch
from decompose_actual_context_paths import checked, sha
from summarize_action_path_curvature import average, episode_interval


def average_ranks(values):
    values = torch.as_tensor(values).double().flatten()
    return torch.tensor([float((values < value).sum())+(float((values == value).sum())-1)/2
                         for value in values], dtype=torch.float64)


def rank_correlation(a, b):
    a = average_ranks(a); b = average_ranks(b)
    a -= a.mean(); b -= b.mean(); denominator = a.norm()*b.norm()
    return float((a@b)/denominator) if denominator > 0 else None


def path_metrics(native, conditioned, actual, goal):
    if any(x.shape != (5, 1, 16, 16, 384) for x in (native, conditioned, actual)):
        raise ValueError('Five full spatial action-conditioned futures required')
    vectors = {k: v.double().reshape(5, -1) for k, v in
               [('native', native), ('actual_past', conditioned), ('actual', actual)]}
    goal = goal.double().reshape(-1)
    if goal.numel() != vectors['native'].shape[1]: raise ValueError('Goal must be one full spatial visual feature')
    spread = {k: float((v-v.mean(0)).square().mean()) for k, v in vectors.items()}
    costs = {k: (v-goal).square().mean(1) for k, v in vectors.items()}
    action_effect = {k: v-v[2:3] for k, v in vectors.items()}
    n, c, truth = vectors['native'], vectors['actual_past'], vectors['actual']
    return dict(latent_spread_mse=spread, paired_same_action_visual_mse=float((n-c).square().mean()),
        native_actual_visual_mse=float((n-truth).square().mean()),
        actual_past_actual_visual_mse=float((c-truth).square().mean()),
        paired_action_effect_visual_mse=float((action_effect['native']-action_effect['actual_past']).square().mean()),
        native_action_effect_actual_mse=float((action_effect['native']-action_effect['actual']).square().mean()),
        actual_past_action_effect_actual_mse=float((action_effect['actual_past']-action_effect['actual']).square().mean()),
        visual_goal_costs={k: v.tolist() for k, v in costs.items()},
        visual_goal_cost_rms_divergence=float((costs['native']-costs['actual_past']).square().mean().sqrt()),
        visual_goal_cost_signed_divergence=float((costs['actual_past']-costs['native']).mean()),
        visual_goal_cost_rank_correlation=rank_correlation(costs['native'], costs['actual_past']),
        native_actual_visual_cost_rank_correlation=rank_correlation(costs['native'], costs['actual']),
        actual_past_actual_visual_cost_rank_correlation=rank_correlation(costs['actual_past'], costs['actual']),
        visual_only_argmin_changed=int(costs['native'].argmin() != costs['actual_past'].argmin()),
        visual_only_exact_minimum_ties={k: int((v == v.min()).sum()) for k, v in costs.items()})


def main():
    p = argparse.ArgumentParser()
    for name in ('input', 'bank', 'output'): p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); torch.set_num_threads(1); started = time.monotonic()
    done = json.loads((a.input/'DONE.json').read_text()); bank_done = json.loads((a.bank/'DONE.json').read_text())
    banks = {r['key']: r for r in bank_done['outputs'] if r['path'].endswith('.pt')}
    if not done['complete'] or not bank_done['complete'] or set(banks) != {f'near-dev-{e:03}' for e in range(4)}:
        raise ValueError('Only complete original four development sources')
    if any(r['split'] != 'development_external' for r in banks.values()): raise ValueError('Split check before loading')
    sources = {r['path']: r for r in done['outputs']}
    expected = {f'near-dev-{e:03}-pair{pair}-radius{radius}.pt'
                for e in range(4) for pair in range(4) for radius in (1, 4)}
    if {key for key in sources if key.endswith('.pt')} != expected: raise ValueError('Frozen path coverage changed')
    a.output.mkdir(parents=True, exist_ok=False); rows = []
    for episode in range(4):
        entry = banks[f'near-dev-{episode:03}']; bank = checked(a.bank/entry['path'], entry['sha256'])
        goal = bank['goal_encoded']['visual']
        if goal.dtype != torch.float32: raise ValueError('Original full-precision visual goal required')
        for pair in range(4):
            for radius in (1, 4):
                name = f'near-dev-{episode:03}-pair{pair}-radius{radius}.pt'
                payload = checked(a.input/name, sources[name]['sha256'])
                if not torch.equal(payload['native_predicted_visual'][0], payload['actual_past_predicted_visual'][0]):
                    raise ValueError('H1 same-context identity failed')
                for h in range(1, 7):
                    metrics = path_metrics(*(payload[k][h-1] for k in
                        ('native_predicted_visual', 'actual_past_predicted_visual', 'actual_visual')), goal)
                    rows.append(dict(episode=episode, pair=pair, radius=radius, horizon=h, **metrics))
    views = []
    scalar_fields = ('paired_same_action_visual_mse', 'paired_action_effect_visual_mse',
        'native_actual_visual_mse', 'actual_past_actual_visual_mse', 'native_action_effect_actual_mse',
        'actual_past_action_effect_actual_mse', 'visual_goal_cost_rms_divergence',
        'visual_goal_cost_signed_divergence', 'visual_only_argmin_changed', 'visual_goal_cost_rank_correlation',
        'native_actual_visual_cost_rank_correlation', 'actual_past_actual_visual_cost_rank_correlation')
    for radius in (1, 4):
        for horizon in range(1, 7):
            panel = [r for r in rows if r['radius'] == radius and r['horizon'] == horizon]
            def grouped(fn): return [average(fn(r) for r in panel if r['episode'] == e) for e in range(4)]
            stats = {}
            for field in scalar_fields:
                values = [average(valid) if (valid := [r[field] for r in panel
                    if r['episode'] == e and r[field] is not None]) else None for e in range(4)]
                nonempty = [v for v in values if v is not None]
                stats[field] = episode_interval(nonempty) if nonempty else {'mean': None, 'n_initial_states': 0}
                stats[field]['values_by_initial_state'] = values
            spread = {name: episode_interval(grouped(lambda r, name=name: r['latent_spread_mse'][name]))
                      for name in ('native', 'actual_past', 'actual')}
            views.append(dict(radius=radius, horizon=horizon, metrics=stats, latent_spread_mse=spread))
    result = dict(complete=True, initial_states=4, paths=32, rows=rows, views=views,
        goal_cost_scope='Unweighted native full-spatial VISUAL MSE component only; predicted proprio was not cached',
        limitations=['ACPC-inspired descriptive consistency diagnostic, not the ACPC training loss or a bisimulation claim',
            'Conditioned predictions use privileged actual past states; not an available prospective intervention',
            'Visual-only cost/argmin is NOT full weighted native visual+proprio planner cost or an executed decision',
            'Five fixed action points per path, not independent starts; average four directions within each of four states',
            'Latent spread and cost disagreement are not safety, utility, physical success or operator readiness',
            'Action-effect error subtracts each path central action forecast; the central point then contributes zero by definition'])
    target = a.output/'CONSISTENCY.json'; target.write_text(json.dumps(result, indent=2)+'\n')
    (a.output/'DONE.json').write_text(json.dumps(dict(complete=True, seconds=time.monotonic()-started,
        outputs=[dict(path=target.name, sha256=sha(target), bytes=target.stat().st_size)], gpu_calls=0, simulator_calls=0,
        source_done_sha256=sha(a.input/'DONE.json'), bank_done_sha256=sha(a.bank/'DONE.json')), indent=2)+'\n')
    print(json.dumps(dict(complete=True, seconds=time.monotonic()-started, paths=32)), flush=True)


if __name__ == '__main__': main()
