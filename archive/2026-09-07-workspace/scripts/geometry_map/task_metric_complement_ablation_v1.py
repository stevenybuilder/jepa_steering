#!/usr/bin/env python3
"""Immutable, no-refit decomposition of four saved development PSD metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

POINTWISE_DONE = '105cb5a3e12cfc9bf84e44a6e7592bfc0fdd0082c548cd49b3de89c22fbbe873'
PAIRWISE_DONE = 'df99f0e08bb55cea8ced6e2f2d8752126a49f02901d1b5d3d7051f96feb9e9ae'
LABELS = ('joint_xy_squared_proxy', 'one_minus_requested_goal_coverage')
INPUT_NAMES = ('terminal-input-0-v1', 'terminal-input-1-v1',
               'terminal-input-2-copy2-v1', 'terminal-input-3-v1')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def checked_manifest(root, expected_sha):
    root = Path(root)
    if digest(root / 'DONE.json') != expected_sha:
        raise ValueError('Pinned original DONE differs')
    d = json.loads((root / 'DONE.json').read_text())
    if not d['complete'] or d['states'] != 4:
        raise ValueError('Require complete original four-state fit')
    entries = {r['path']: r for r in d['outputs']}

    def checked(name):
        p = root / name
        if digest(p) != entries[name]['sha256']:
            raise ValueError('Original source checksum differs: ' + name)
        return p

    return d, checked


def decompose(x, basis, rotation, weights):
    """No mean subtraction: metric input is the already weighted goal difference."""
    x, basis, rotation = [np.asarray(v, dtype=np.float64) for v in (x, basis, rotation)]
    if x.ndim != 2 or basis.ndim != 2 or x.shape[1] != basis.shape[1]:
        raise ValueError('Native full-dimensional feature support differs')
    rank = len(basis)
    if not np.allclose(basis @ basis.T, np.eye(rank), atol=2e-10, rtol=0):
        raise ValueError('Basis not orthonormal')
    if not np.allclose(rotation.T @ rotation, np.eye(rank), atol=2e-10, rtol=0):
        raise ValueError('Saved control not orthogonal')
    native = np.sum(x * x, axis=1)
    coords = x @ basis.T
    span = np.sum(coords * coords, axis=1)
    complement = native - span
    if complement.min() < -1e-9 * max(1., float(native.max())):
        raise ValueError('Negative complement energy')
    rotated = coords @ rotation
    costs = dict(native_full=native, unweighted_span_only=span, complement_only=complement)
    for name, w in weights.items():
        w = np.asarray(w, dtype=float)
        if w.shape != (rank,) or not np.isfinite(w).all() or np.any(w < 0):
            raise ValueError('Frozen weights must remain finite PSD')
        for control, c in [('learned', coords), ('rotated_control', rotated)]:
            learned = (c * c) @ w
            costs[f'{name}/{control}/span_only'] = learned
            costs[f'{name}/{control}/with_complement'] = learned + complement
    return costs, coords


def score(cost, states, goal, coverage, baseline):
    from scipy.stats import spearmanr
    cost, baseline, coverage = [np.asarray(v, dtype=float) for v in (cost, baseline, coverage)]
    xy = np.linalg.norm(np.asarray(states)[:, :4] - np.asarray(goal)[:4], axis=1)
    selected, native = int(np.argmin(cost)), int(np.argmin(baseline))
    def corr(a, b):
        if np.ptp(a) == 0 or np.ptp(b) == 0:
            return None
        return float(spearmanr(a, b).statistic)
    return dict(selected_index=selected, baseline_selected_index=native,
                selected_goal_coverage=float(coverage[selected]),
                goal_coverage_delta_vs_baseline=float(coverage[selected] - coverage[native]),
                goal_coverage_regret=float(coverage.max() - coverage[selected]),
                selected_xy_distance_px=float(xy[selected]),
                xy_delta_vs_baseline_px=float(xy[selected] - xy[native]),
                cost_vs_xy_squared_spearman=corr(cost, xy ** 2),
                negative_cost_vs_goal_coverage_spearman=corr(-cost, coverage),
                cost_by_candidate=cost.tolist(),
                margins_relative_to_native_winner=(cost - cost[native]).tolist())


def gate(gains, controls):
    gains, controls = np.asarray(gains), np.asarray(controls)
    if gains.shape != (4,) or controls.shape != (4,):
        raise ValueError('Four equal-weight state units required')
    return dict(mean_gain=float(gains.mean()), positive_states=int(np.sum(gains > 0)),
                mean_matched_control_gain=float(controls.mean()),
                pass_exploratory_gate=bool(gains.mean() >= .01 and np.sum(gains > 0) >= 3
                                           and gains.mean() > controls.mean()))


def component_diagnostics(costs):
    native, span, outside = (costs[n] for n in ('native_full', 'unweighted_span_only', 'complement_only'))
    winner = int(np.argmin(native))
    i, j = np.triu_indices(len(native), 1)
    result = dict(complement_energy_fraction_mean=float(np.mean(outside / native)),
                  span_energy_fraction_mean=float(np.mean(span / native)),
                  native_winner=winner, components={})
    for name, c in costs.items():
        result['components'][name] = dict(mean=float(c.mean()), candidate_std=float(c.std()),
            pair_difference_rms=float(np.sqrt(np.mean((c[i] - c[j]) ** 2))),
            strongest_challenger_margin=float(np.min(np.delete(c - c[winner], winner))))
    result['native_decomposition_maxabs'] = float(np.max(np.abs(native - span - outside)))
    result['interpretation'] = 'Mean energy is not ranking dominance; candidate and pair contrasts are separate.'
    return result


def assert_fold(frozen, held):
    if int(frozen['held_state']) != held or list(frozen['train_states']) != [i for i in range(4) if i != held]:
        raise ValueError('Original whole-state exclusion changed')


def run(args):
    started, cpu = time.monotonic(), time.process_time()
    args.output.mkdir(parents=True, exist_ok=False)
    pd, pfile = checked_manifest(args.pointwise, POINTWISE_DONE)
    qd, qfile = checked_manifest(args.pairwise, PAIRWISE_DONE)
    original = json.loads(pfile('protocol.json').read_text())
    qp = json.loads(qfile('protocol.json').read_text())
    if qp['source_pointwise_done_sha256'] != POINTWISE_DONE or digest(args.coverage) != original['coverage_sha256']:
        raise ValueError('Original basis family or goal-coverage labels differ')
    protocol = dict(complete=True, version='task_metric_complement_ablation_v1', frozen_before_scoring=True,
        state_ids=list(range(4)), candidate_count_per_state=64,
        statistical_units='4 repeatedly seen DEVELOPMENT states, not 256 independent plans',
        weights='Reuse original pointwise and pairwise LOSO weights and Haar rotations exactly; no refit or scaling',
        dimensions=102400, pca_rank=16, centering='NONE at scoring; original TRAIN covariance mean not subtracted',
        input_weighting='Saved full spatial visual /sqrt98304 and proprio sqrt(.1/4096); terminal H6 only',
        source_pointwise_done_sha256=POINTWISE_DONE, source_pairwise_done_sha256=PAIRWISE_DONE,
        coverage_sha256=digest(args.coverage), labels=list(LABELS), blend=1,
        arms='native, unweighted span, complement, 4 frozen learned metrics each span-only/full and same old rotated controls',
        channels=['predicted', 'actual_oracle_encoding'],
        primary='Predicted-channel selected requested-goal polygon coverage vs native predicted cost',
        actual_channel='Offline true-future encoder oracle diagnosis, never runtime-deployable',
        gate=dict(mean_coverage_gain_min=.01, positive_states_min=3, beat_corresponding_rotated_control_mean=True),
        no_outcome_selection=True, model_calls=0, simulator_calls=0, fit_calls=0, sealed_data_access=False,
        cpu_seconds_cap=180, code_sha256=digest(__file__),
        limitations=['All states adaptive development, no confirmatory claim or full rollout efficacy.',
                     'Old weights were fitted with complement included; span-only is a fixed ablation, not a refitted optimum.',
                     'Dropping complement changes metric support, deliberately; no residual-stream steering occurs.',
                     'No inference that high mean complement energy alone dominates cost rankings.'])
    write_json(args.output / 'PROTOCOL.json', protocol)
    print(json.dumps(dict(event='protocol_frozen', sha256=digest(args.output / 'PROTOCOL.json'))), flush=True)
    coverage_json = json.loads(args.coverage.read_text())
    if not coverage_json['complete'] or sorted(r['episode'] for r in coverage_json['rows']) != list(range(4)):
        raise ValueError('Require all four original coverage groups')
    coverage = {r['episode']: np.asarray(r['goal_aligned_coverage_by_candidate']) for r in coverage_json['rows']}
    reports = []
    for held, dirname in enumerate(INPUT_NAMES):
        root = args.inputs_root / dirname
        receipt = json.loads((root / 'DONE.json').read_text())
        expected = next(r for r in original['sources'] if r['episode'] == held)
        row = receipt['outputs'][0]
        if not receipt['complete'] or receipt['episode'] != held or row != expected['outputs'][0]:
            raise ValueError('Original cached array grouping/receipt differs')
        if digest(root / row['path']) != row['sha256']:
            raise ValueError('Cached array checksum failed before load')
        ppath, qpath = pfile(f'fold-{held}-FROZEN.npz'), qfile(f'fold-{held}-FROZEN.npz')
        pf, qf = [dict(np.load(p, allow_pickle=False)) for p in (ppath, qpath)]
        for f in (pf, qf):
            assert_fold(f, held)
        if str(qf['basis_source_sha256']) != digest(ppath):
            raise ValueError('Pairwise frozen basis link differs')
        weights = {f'{family}/{label}': frozen[label + '_weights']
                   for family, frozen in [('pointwise', pf), ('pairwise', qf)] for label in LABELS}
        # Weights and original exclusion are fixed and SHA-verified before reading evaluation labels.
        bank = dict(np.load(root / row['path'], allow_pickle=False))
        if int(bank['episode']) != held or not np.array_equal(bank['candidate_ids'], np.arange(64)):
            raise ValueError('Candidate order changed')
        old = json.loads(qfile(f'fold-{held}.json').read_text())
        report = dict(complete=True, state=held, source_array_sha256=row['sha256'],
            frozen_pointwise_sha256=digest(ppath), frozen_pairwise_sha256=digest(qpath),
            channels={}, goal_coverage_by_candidate=coverage[held].tolist(),
            joint_xy_distance_by_candidate_px=np.linalg.norm(bank['states'][:, :4] - bank['goal_state'][:4], axis=1).tolist())
        for channel, key in [('predicted', 'predicted'), ('actual_oracle_encoding', 'actual')]:
            x = bank[key].astype(float)
            if x.shape != (64, 102400) or pf['basis'].shape != (16, 102400):
                raise ValueError('Full native feature or PCA rank differs')
            costs, coords = decompose(x, pf['basis'], pf['random_rotation'], weights)
            native = costs['native_full']; cached_native = bank['native_' + key + '_cost']
            if not np.allclose(native, cached_native, rtol=2e-6, atol=1e-8) or np.argmin(native) != np.argmin(cached_native):
                raise ValueError('Native reconstruction/choice parity failed')
            parity = []
            for name, w in weights.items():
                family, label = name.split('/')
                for control in ('learned', 'rotated_control'):
                    target = next(r for r in old['rows'] if r['channel'] == channel and r['fit_label'] == label
                        and r['algorithm'] == ('saved_pointwise' if family == 'pointwise' else 'pairwise_logistic')
                        and r['family'] == ('learned_psd' if control == 'learned' else 'spectrum_matched_rotation')
                        and r['blend'] == 1)
                    current = costs[f'{name}/{control}/with_complement']
                    error = float(np.max(np.abs(current - np.asarray(target['cost_by_candidate']))))
                    if error > 1e-12 or np.argmin(current) != target['selected_index']:
                        raise ValueError('Old learned full-cost/choice parity failed')
                    parity.append(error)
            report['channels'][channel] = dict(
                native_cost_maxabs=float(np.max(np.abs(native - cached_native))),
                native_argmin_exact=True, native_rank_exact=bool(np.array_equal(np.argsort(native), np.argsort(cached_native))),
                old_full_cost_maxabs=max(parity), old_full_choices_exact=True,
                diagnostics=component_diagnostics(costs),
                rows=[dict(arm=name, **score(c, bank['states'], bank['goal_state'], coverage[held], native))
                      for name, c in costs.items()])
        write_json(args.output / f'state-{held}.json', report)
        reports.append(report)
        print(json.dumps(dict(event='state_complete', state=held, wall_seconds=time.monotonic()-started,
                              cpu_seconds=time.process_time()-cpu)), flush=True)
        if time.process_time() - cpu > 170:
            raise RuntimeError('CPU budget guard')
    aggregate = dict(complete=True, states=4, channels={}, protocol_sha256=digest(args.output / 'PROTOCOL.json'))
    for channel in ('predicted', 'actual_oracle_encoding'):
        by_state = [{r['arm']: r for r in report['channels'][channel]['rows']} for report in reports]
        summaries = []
        for arm in by_state[0]:
            rows = [state[arm] for state in by_state]
            gains = [r['goal_coverage_delta_vs_baseline'] for r in rows]
            summary = dict(arm=arm, selected_indices=[r['selected_index'] for r in rows],
                coverage_gain_by_state=gains, mean_coverage_gain=float(np.mean(gains)),
                positive_states=sum(v > 0 for v in gains),
                selected_coverage_by_state=[r['selected_goal_coverage'] for r in rows],
                xy_delta_by_state_px=[r['xy_delta_vs_baseline_px'] for r in rows])
            if '/learned/' in arm:
                control = arm.replace('/learned/', '/rotated_control/')
                summary['gate'] = gate(gains, [s[control]['goal_coverage_delta_vs_baseline'] for s in by_state])
            summaries.append(summary)
        aggregate['channels'][channel] = summaries
    aggregate['cpu_seconds'] = time.process_time() - cpu
    aggregate['wall_seconds'] = time.monotonic() - started
    aggregate['model_calls'] = aggregate['simulator_calls'] = aggregate['fit_calls'] = 0
    aggregate['claim_scope'] = protocol['limitations']
    write_json(args.output / 'AGGREGATE.json', aggregate)
    paths = sorted(args.output.glob('*.json'))
    write_json(args.output / 'DONE.json', dict(complete=True, states=4, cpu_seconds=aggregate['cpu_seconds'],
        wall_seconds=aggregate['wall_seconds'], outputs=[dict(path=p.name, sha256=digest(p), bytes=p.stat().st_size) for p in paths]))
    print(json.dumps(dict(event='DONE', cpu_seconds=aggregate['cpu_seconds'], wall_seconds=aggregate['wall_seconds'])), flush=True)


def recompute(output):
    """Independent scalar recompute from saved candidate costs, no original large arrays."""
    done = json.loads((output / 'DONE.json').read_text())
    for row in done['outputs']:
        if digest(output / row['path']) != row['sha256']:
            raise ValueError('Output checksum mismatch')
    checks = 0
    for state in range(4):
        report = json.loads((output / f'state-{state}.json').read_text())
        coverage = np.asarray(report['goal_coverage_by_candidate'])
        for channel in report['channels'].values():
            rows = {r['arm']: r for r in channel['rows']}
            n, s, o = [np.asarray(rows[k]['cost_by_candidate']) for k in ('native_full', 'unweighted_span_only', 'complement_only')]
            if not np.allclose(n, s + o, rtol=0, atol=1e-12):
                raise ValueError('Saved native decomposition mismatch')
            for arm, row in rows.items():
                cost = np.asarray(row['cost_by_candidate'])
                if row['selected_index'] != int(cost.argmin()) or row['selected_goal_coverage'] != coverage[cost.argmin()]:
                    raise ValueError('Saved cost choice or coverage join differs')
                if arm.endswith('/with_complement'):
                    span = np.asarray(rows[arm.replace('/with_complement', '/span_only')]['cost_by_candidate'])
                    if not np.allclose(cost, span + o, rtol=0, atol=1e-12):
                        raise ValueError('Saved learned decomposition mismatch')
                checks += 1
    return dict(complete=True, independently_recomputed_rows=checks, checksum_verified=True,
                native_and_learned_decompositions_exact_to_1e12=True, model_calls=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('pointwise', 'pairwise', 'inputs-root', 'coverage', 'output', 'verify'):
        p.add_argument('--' + name, type=Path)
    a = p.parse_args()
    if a.verify:
        print(json.dumps(recompute(a.verify)), flush=True)
    else:
        if any(getattr(a, k) is None for k in ('pointwise', 'pairwise', 'inputs_root', 'coverage', 'output')):
            p.error('All input paths and output are required')
        run(a)


if __name__ == '__main__':
    main()
