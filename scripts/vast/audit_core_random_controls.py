"""Read-only, explicitly post-hoc audit of the preserved five-arm core panel.

No simulation, fit, training, GPU work, or changes to the frozen primary analysis.
Requires the full restored archive to have passed part/archive/member hashes.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path('/workspace/table-completion-20260911-v1')
RAW = ROOT / 'control-audit-restored'
EVIDENCE = RAW / 'fixed-response-behavior-evidence-20260908-v1/artifacts/offline_study'
ARMS = ('native', 'fixed_rank4', 'matched_random_fixed_rank4', 'coupling_only', 'matched_random_coupling')
TASKS = ('reach', 'reach-wall')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(test, message):
    if not test:
        raise ValueError(message)


def exact(wins, losses):
    n = wins + losses
    return min(1., 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n) if n else 1.


def holm(values):
    result, previous = [None] * len(values), 0.
    for rank, index in enumerate(sorted(range(len(values)), key=lambda i: (values[i], i))):
        previous = max(previous, min(1., (len(values) - rank) * values[index]))
        result[index] = previous
    return result


def summarize(values):
    return {'count': len(values), 'min': float(min(values)), 'max': float(max(values)),
            'mean': float(np.mean(values))} if values else None


def main():
    verified = read(ROOT / 'CONTROL_AUDIT_RESTORED.json')
    spec = read(ROOT / 'control-audit-inputs/PARTS.json')
    require(verified['archive_sha256'] == spec['archive_sha256'], 'Wrong restored archive')
    freeze = EVIDENCE / 'fixed-response-20260908-v1/behavioral-freeze-v1/protocol.json'
    require(digest(freeze) == '853e8bf6b71c65c1e61d249ba84d8d57d6172ba1bd3a15e0424dc2d9e32cfa1f', 'Wrong frozen panel')
    protocol = read(freeze)
    original = RAW / 'fixed-response-behavior-analysis-20260908-v1/report.json'
    require(digest(original) == '0bcfbae89587430782451429f8cad950fc5f8ecc3c0e4ae5a9a2acf88eae977a', 'Wrong primary report')
    original_report = read(original)
    panels, energy, reports = {}, {}, {}
    for task in TASKS:
        panels[task], energy[task] = {}, {}
        for arm in ARMS:
            rows, requests, actuals, errors, active = [], [], [], [], []
            counts = {'calls': 0, 'full_h6_calls': 0, 'shortened_calls': 0}
            for shard in sorted((RAW / 'fixed-response-behavior-20260908-v1' / task / arm).glob('shard-*')):
                report = read(shard / 'report.json')
                require(digest(shard / 'report.json') == read(shard / 'DONE.json')['report_sha256'], 'Changed report')
                launch = read(shard / 'protocol.json')
                require(digest(shard / 'protocol.json') == report['protocol_sha256'], 'Changed launch')
                require(launch['freeze_sha256'] == digest(freeze) and launch['source_sha256'] == protocol['source_sha256'], 'Wrong frozen source')
                require(report['task'] == task and report['arm'] == arm and report['parameters_unchanged'] is True, 'Wrong shard')
                reports[str(shard.relative_to(RAW))] = digest(shard / 'report.json')
                for name, expected_hash in report['episode_files_sha256'].items():
                    require(Path(name).name == name and digest(shard / name) == expected_hash, 'Changed episode')
                    row = read(shard / name)
                    calls_root = shard / f"calls-{row['episode']:03d}"
                    for filename, key in (('unroll_calls.json', 'unroll_calls_sha256'), ('action_trace.json', 'action_trace_sha256')):
                        require(digest(calls_root / filename) == row[key], 'Changed trace')
                    require(row['result']['elementary_steps'] == 100 and row['result']['published_candidate_count'] == 300, 'Shortened planner budget')
                    for call in read(calls_root / 'unroll_calls.json'):
                        counts['calls'] += 1
                        require(call['backend_calls'] == 1, 'Extra backend calls')
                        e = call['energy']
                        if call['horizon'] < 6:
                            counts['shortened_calls'] += 1
                            require('requested_l2' not in e and e.get('requested_squared_l2_mean', 0.) == 0., 'Edit on shortened horizon')
                            continue
                        counts['full_h6_calls'] += 1
                        if 'requested_l2' in e:
                            req, got = np.asarray(e['requested_l2']), np.asarray(e['realized_l2'])
                            require(len(req) == call['candidates'] and req.shape == got.shape, 'Candidate energy length')
                            active.extend(e['active'])
                            requests.append(float(req.mean())); actuals.append(float(got.mean()))
                            errors.append(float(np.max(np.abs(req-got) / np.maximum(req, 1e-20))))
                        elif 'requested_squared_l2_mean' in e:
                            req, got = math.sqrt(e['requested_squared_l2_mean']), math.sqrt(e['realized_squared_l2_mean'])
                            requests.append(req); actuals.append(got); errors.append(abs(req-got)/max(req, 1e-20))
                    rows.append(row)
            rows.sort(key=lambda row: row['episode'])
            require(len(rows) == 96, 'Incomplete or duplicate panel')
            for row, expected in zip(rows, protocol['episodes']):
                require(all(row[k] == v for k, v in expected.items()) and row['arm'] == arm, 'Wrong scenario/seed assignment')
                if arm != 'native':
                    reference = panels[task]['native'][row['episode']]
                    require(row['initial_state_vector'] == reference['initial_state_vector'], 'Unpaired state')
                    require(all(row['result'][k] == reference['result'][k] for k in ('initial_sha256', 'goal_sha256')), 'Unpaired images')
            require(sum(r['result']['native_success'] for r in rows) == original_report['tasks'][task]['arms'][arm]['successes'], 'Changed success counts')
            panels[task][arm] = rows
            energy[task][arm] = {**counts, 'requested_l2_or_root_summed_site_l2_squared': summarize(requests),
                'realized_l2_or_root_summed_site_l2_squared': summarize(actuals), 'max_relative_delivery_error': max(errors, default=0.),
                'active_candidates': sum(active), 'candidate_activation_records': len(active)}
        print(json.dumps({'validated_task': task, 'episodes': 480}), flush=True)
    contrasts, rng = [], np.random.default_rng(2026091101)
    for task in TASKS:
        native = panels[task]['native']
        keys = {json.dumps([r['initial_state_vector'],r['result']['goal_sha256']]) for r in native}
        require(len(keys) == 96, 'This supplemental calculation assumes 96 distinct paired scenarios')
        draws = rng.integers(96, size=(20000, 96))
        for arm in ('matched_random_fixed_rank4', 'matched_random_coupling'):
            candidate = panels[task][arm]
            delta = np.array([int(c['result']['native_success'])-int(n['result']['native_success']) for c,n in zip(candidate,native)])
            wins = [r['episode'] for r,d in zip(native,delta) if d > 0]
            losses = [r['episode'] for r,d in zip(native,delta) if d < 0]
            by_stream = {str(rank): {'wins': sum(d > 0 for r,d in zip(native,delta) if r['logical_rank']==rank),
                                    'losses': sum(d < 0 for r,d in zip(native,delta) if r['logical_rank']==rank)} for rank in range(8)}
            samples = delta[draws].mean(1)*100
            contrasts.append({'task':task, 'arm':arm, 'control':'native', 'episodes':96,
                'gain_percentage_points':float(delta.mean()*100), 'wins':len(wins), 'losses':len(losses),
                'both_succeed':sum(c['result']['native_success'] and n['result']['native_success'] for c,n in zip(candidate,native)),
                'both_fail':sum(not c['result']['native_success'] and not n['result']['native_success'] for c,n in zip(candidate,native)),
                'win_episode_ids':wins, 'loss_episode_ids':losses, 'by_logical_stream':by_stream,
                'unadjusted_exact_p':exact(len(wins),len(losses)),
                'exploratory_simultaneous_95_interval_four_comparisons_pp':np.quantile(samples,[.05/8,1-.05/8]).tolist()})
    for c,p in zip(contrasts,holm([c['unadjusted_exact_p'] for c in contrasts])):
        c['exploratory_holm_p_four_comparisons'] = p
    combined = [c['exact_discordance_p'] for c in original_report['contrasts']] + [c['unadjusted_exact_p'] for c in contrasts]
    for c,p in zip(contrasts,holm(combined)[8:]):
        c['sensitivity_holm_p_original_eight_plus_four_posthoc'] = p
    result = {'status':'saved_core_pairing_counts_and_delivered_energy_audited', 'posthoc':True,
        'fresh_confirmation':False, 'random_seed_robustness_established':False,
        'original_primary_analysis_unchanged':True, 'new_gpu_calls':0, 'verified_episodes':960,
        'source_sha256':digest(Path(__file__)), 'archive_sha256':verified['archive_sha256'],
        'report_hashes':reports, 'energy':energy, 'exploratory_contrasts':contrasts,
        'interpretation':'Numerical/setup audit, not proof of repeatable random-control benefit; exploratory tests were selected after seeing outcomes.'}
    # numpy scalar stream counts need conversion for portable JSON serialization.
    with (ROOT / 'CONTROL_AUDIT_REPORT.json').open('x') as output:
        json.dump(result,output,indent=2,default=lambda x:x.item())
    print(json.dumps({'status':result['status'],'contrasts':contrasts},default=lambda x:x.item()),flush=True)


if __name__ == '__main__':
    main()
