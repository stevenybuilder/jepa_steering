"""Audit and summarize the fixed 200-scenario action-history replication.

Case extraction audits completed records without aggregating partial results.
Aggregation requires every registered scientific scenario, each cloud verified.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from .action_counterfactual_summary import (
    ARMS, BANKS, MODALITIES, audit_scores, finite, score_audit, validate_runtime,
)

PROTOCOL_SHA = '967aebfd743fe2bfb226bef2915dbebbac31f4c816f84f987ed514a3356c9204'
TASKS = ('reach', 'reach-wall')
EXPECTED = {(task, episode) for task in TASKS for episode in range(100)}
NAMES = ('STARTED.json', 'scores.json', 'actions.pt', 'report.json', 'DONE.json', 'REPLICATION_CASE.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def wilson(k, n, confidence):
    if not (0 <= k <= n and n > 0 and 0 < confidence < 1):
        raise ValueError('Invalid binomial count or confidence')
    z = NormalDist().inv_cdf((1+confidence)/2)
    p = k/n; denominator = 1+z*z/n
    center = (p+z*z/(2*n))/denominator
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return [max(0., float(center-half)), min(1., float(center+half))]


def selection_metrics(reference, current):
    a, b = finite(reference['costs'], (300,)), finite(current['costs'], (300,))
    agreement = score_audit(reference, current)
    best, selected = int(np.argmin(a)), int(np.argmin(b))
    gap = float(a[selected]-a[best])
    def cutoff_ties(values):
        cutoff = np.sort(values)[9]
        return int(np.sum(values == cutoff))
    return dict(reference_winner=best, selected_winner=selected,
        winner_changed=int(best != selected), selected_in_reference_minimum_set=int(a[selected] == a[best]),
        reference_minimum_ties=int(np.sum(a == a[best])), selected_minimum_ties=int(np.sum(b == b[selected])),
        reference_top10_cutoff_ties=cutoff_ties(a), selected_top10_cutoff_ties=cutoff_ties(b),
        reference_spearman=agreement['native_spearman'], reference_top10_overlap=agreement['native_top10_overlap'],
        reference_minimum_cost=float(a[best]), excess_reference_cost=gap,
        excess_reference_cost_percent=100*gap/float(a[best]) if a[best] > 0 else None)


def extract_case(directory, manifest_path, protocol_path):
    directory = Path(directory); manifest_sha = sha(manifest_path)
    manifest, protocol = read(manifest_path), read(protocol_path)
    if sha(protocol_path) != PROTOCOL_SHA or manifest['protocol_sha256'] != PROTOCOL_SHA:
        raise ValueError('Frozen protocol differs')
    if manifest['exposure_audit']['passed'] is not True or manifest['exposure_audit']['scientific_scenarios'] != 200:
        raise ValueError('Input exposure audit missing')
    report, started, payload, done, receipt, cloud = [read(directory/name) for name in
        ('report.json', 'STARTED.json', 'scores.json', 'DONE.json', 'REPLICATION_CASE.json', 'CLOUD_VERIFIED.json')]
    binding = report['input_binding']; key = (binding['task'], binding['episode'])
    if key not in EXPECTED or binding['role'] != 'scientific_replication':
        raise ValueError('Require a registered scientific scenario; engineering is excluded')
    records = [r for r in manifest['records'] if (r['task'], r['episode']) == key]
    if records != [binding]:
        raise ValueError('Delivered input is not the unique frozen manifest record')
    registered = next(r for r in protocol['scenarios'] if (r['task'], r['episode']) == key)
    if any(binding[k] != v for k, v in registered.items()):
        raise ValueError('Scenario differs from the pre-execution registry')
    hashes = {name: sha(directory/name) for name in NAMES}
    files = {name: hashes[name] for name in ('STARTED.json', 'scores.json', 'actions.pt')}
    if done['report_sha256'] != hashes['report.json'] or done['files'] != files or report['files'] != files:
        raise ValueError('Case member hashes differ')
    if (cloud.get('compact_sha256') != hashes or cloud.get('gcs_download_sha256_verified') is not True
            or cloud.get('all_report_files_hash_verified') is not True or cloud.get('raw_preservation_pending', False)):
        raise ValueError('Full cloud readback and member hashes required')
    if not str(cloud.get('cloud_uri', '')).startswith('gs://') or not cloud.get('generation') or len(cloud.get('sha256', '')) != 64:
        raise ValueError('Cloud object identity missing')
    for item in (report, started, payload):
        if (item['input_binding'] != binding or item['execution_manifest_sha256'] != manifest_sha
                or item['protocol_sha256'] != PROTOCOL_SHA):
            raise ValueError('Source/input/protocol identity differs')
    if (receipt.get('manifest_sha256') != manifest_sha or receipt.get('protocol_sha256') != PROTOCOL_SHA
            or receipt.get('kernel_done_sha256') != hashes['DONE.json'] or receipt.get('complete') is not True
            or receipt.get('forward_count') != 70 or receipt.get('role') != 'scientific_replication'
            or receipt.get('fresh_input_replication') is not True or receipt.get('protected_behavioral_panel') is not False):
        raise ValueError('Replication completion receipt differs')
    if (report['status'] != 'complete_action_counterfactual_development_case' or done['status'] != report['status']
            or report['global_rng_unchanged'] is not True or report['physical_outcomes_measured'] is not False
            or report['fresh_confirmation'] is not False or report['total_forward_count'] != 70
            or started['arms'] != list(ARMS)):
        raise ValueError('Frozen kernel execution checks failed')
    runtime = report['runtime_provenance']
    if runtime != started['runtime_provenance'] or receipt['gpu_uuid'] != runtime['gpu_uuid']:
        raise ValueError('Runtime identity differs')
    validate_runtime(runtime, {'receivers': {str(i): {'gpu_uuid': u} for i, u in enumerate(manifest['gpu_uuids'])}})
    if payload['range_audit'] != report['range_audit'] or payload['range_audit'] != started['range_audit']:
        raise ValueError('Action-encoder range audit differs')
    for bank in BANKS:
        bank_receipt = report['bank_receipts'][bank]
        if (bank_receipt != payload['bank_receipts'][bank] or bank_receipt['forecast_count'] != 35
                or any(bank_receipt[k] is not True for k in ('zero_full_horizon_byte_parity',
                    'persistent_allblock_raw_full_horizon_byte_parity', 'h1_h2_unchanged_all_arms'))):
            raise ValueError('Full-horizon parity failed')
    reconstruction, norms = audit_scores(payload, key)
    ranks = []
    for bank in BANKS:
        arms = payload['banks'][bank]
        for arm in ARMS:
            for modality in MODALITIES:
                ranks.append(dict(task=key[0], episode=key[1], bank=bank, arm=arm, modality=modality,
                    **selection_metrics(arms['coherent_raw_h3']['scores'][modality], arms[arm]['scores'][modality])))
    return dict(task=key[0], episode=key[1], environment_seed=binding['environment_seed'],
        protocol_sha256=PROTOCOL_SHA, manifest_sha256=manifest_sha, hashes=hashes,
        cloud=cloud, extractor_sha256=sha(__file__), rank_metrics=ranks,
        reconstruction_metrics=reconstruction, norm_audit_rows=len(norms),
        range_audit=report['range_audit'], gpu_uuid=runtime['gpu_uuid'], forward_count=70)


def estimate(values, weights):
    values = np.asarray(values, dtype=float)
    if len(values) != 100:
        raise ValueError('Require 100 independent scenarios per summary cell')
    result = dict(n=100, n_defined=int(np.isfinite(values).sum()), mean=None, marginal_95_low=None, marginal_95_high=None)
    if np.isfinite(values).all():
        low, high = np.quantile(weights@values, [.025, .975])
        result.update(mean=float(values.mean()), marginal_95_low=float(low), marginal_95_high=float(high))
    else:
        result['undefined_reason'] = 'At least one registered scenario is undefined; no cases dropped'
    return result


def aggregate(cases, output):
    keys = [(r['task'], r['episode']) for r in cases]
    if len(keys) != 200 or set(keys) != EXPECTED:
        raise ValueError('Complete 200 distinct registered scenarios required before aggregation')
    if any(r['protocol_sha256'] != PROTOCOL_SHA or r['extractor_sha256'] != sha(__file__)
           or r['forward_count'] != 70 or not r['cloud']['gcs_download_sha256_verified'] for r in cases):
        raise ValueError('Case audit/protocol differs')
    if len({r['environment_seed'] for r in cases}) != 200:
        raise ValueError('Repeated environment seed')
    if len({r['range_audit']['weight_sha256'] for r in cases}) != 1:
        raise ValueError('Frozen encoder weights differ')
    ranks = pd.DataFrame([r for c in cases for r in c['rank_metrics']])
    recon = pd.DataFrame([r for c in cases for r in c['reconstruction_metrics']])
    if len(ranks) != 200*2*35*3 or len(recon) != 200*2*35*3*3:
        raise ValueError('Full registered metric grid required')
    primary = []
    for task in TASKS:
        group = ranks[(ranks.task == task) & (ranks.bank == 'original') &
                      (ranks.arm == 'all_blocks_h3_donor') & (ranks.modality == 'official')].sort_values('episode')
        if list(group.episode) != list(range(100)):
            raise ValueError('Primary registry incomplete')
        k = int(group.winner_changed.sum())
        primary.append(dict(task=task, changed=k, n=100, fraction=k/100,
            family_95_interval=wilson(k, 100, .975), marginal_95_interval=wilson(k, 100, .95)))
    # One shared resampling schedule preserves pairing across banks and arms.
    weights = np.random.default_rng(2026091421).multinomial(100, np.full(100, .01), size=20000)/100
    secondary = []
    for frame, columns, metrics in (
        (ranks, ['task', 'bank', 'arm', 'modality'],
         ['reference_spearman', 'reference_top10_overlap', 'winner_changed', 'selected_in_reference_minimum_set',
          'excess_reference_cost', 'excess_reference_cost_percent', 'reference_minimum_ties',
          'selected_minimum_ties', 'reference_top10_cutoff_ties', 'selected_top10_cutoff_ties']),
        (recon, ['task', 'bank', 'arm', 'modality', 'horizon'], ['donor_reconstruction'])):
        for key, group in frame.groupby(columns, sort=True):
            group = group.sort_values('episode')
            if list(group.episode) != list(range(100)):
                raise ValueError('Summary cell contains missing or repeated scenarios')
            for metric in metrics:
                secondary.append(dict(zip(columns, key), metric=metric, **estimate(group[metric].to_numpy(float), weights)))
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    ranks.to_csv(output/'lcfm_replication_rank_cases.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    recon.to_csv(output/'lcfm_replication_reconstruction_cases.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    pd.DataFrame(secondary).to_csv(output/'lcfm_replication_secondary.csv', index=False)
    sources = [{k: v for k, v in c.items() if k not in ('rank_metrics', 'reconstruction_metrics')} for c in cases]
    receipt = dict(role='complete_fresh_action_history_replication', protocol_sha256=PROTOCOL_SHA,
        analysis_sha256=sha(__file__), independent_scenarios=200, per_task=100, h6_forecasts=14000,
        old_16_pooled=False, physical_outcomes_measured=False, all_cloud_verified=True,
        primary=primary, secondary_interval='Marginal 95% paired context bootstrap; descriptive, not simultaneous',
        sources=sources, outputs={p.name: sha(p) for p in output.iterdir()})
    write(output/'lcfm_replication_summary.json', receipt)
    return primary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    case = sub.add_parser('extract')
    for name in ('case', 'manifest', 'protocol', 'output'):
        case.add_argument('--'+name, type=Path, required=True)
    total = sub.add_parser('aggregate')
    total.add_argument('--cases', type=Path, required=True)
    total.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'extract':
        data = extract_case(args.case, args.manifest, args.protocol)
        with args.output.open('xb') as stream:
            with gzip.GzipFile(fileobj=stream, mode='wb', mtime=0) as compressed:
                compressed.write(json.dumps(data, allow_nan=False).encode())
        print(json.dumps({'task': data['task'], 'episode': data['episode'], 'audit_passed': True}))
    else:
        cases = []
        for path in sorted(args.cases.glob('*.json.gz')):
            with gzip.open(path, 'rt') as stream: cases.append(json.load(stream))
        print(json.dumps(aggregate(cases, args.output), indent=2))


if __name__ == '__main__':
    main()
