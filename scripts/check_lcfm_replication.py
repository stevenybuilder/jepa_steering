"""Recount the complete replication's published tables and provenance on CPU."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data = ROOT/'paper/data'
    report = json.loads((data/'lcfm_replication_summary.json').read_text())
    protocol_path = data/'lcfm_replication_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    assert report['protocol_sha256'] == sha(protocol_path)
    assert report['independent_scenarios'] == 200 and report['per_task'] == 100
    assert report['h6_forecasts'] == 14000 and report['all_cloud_verified']
    assert report['old_16_pooled'] is False and report['physical_outcomes_measured'] is False
    assert report['analysis_sha256'] == sha(ROOT/'analysis/mechanism/lcfm_replication_summary.py')
    for name, digest in report['outputs'].items():
        assert sha(data/name) == digest, name

    expected = {(r['task'], r['episode']) for r in protocol['scenarios']}
    sources = report['sources']
    assert len(sources) == 200 and {(r['task'], r['episode']) for r in sources} == expected
    assert len({r['environment_seed'] for r in sources}) == 200
    for item in sources:
        cloud = item['cloud']
        assert item['forward_count'] == 70
        assert item['protocol_sha256'] == report['protocol_sha256']
        assert item['extractor_sha256'] == report['analysis_sha256']
        assert cloud['gcs_download_sha256_verified'] and cloud['all_report_files_hash_verified']
        assert cloud['raw_preservation_pending'] is False
        assert cloud['compact_sha256'] == item['hashes'] and len(item['hashes']) == 6
        assert cloud['manifest_sha256'] == item['manifest_sha256']

    ranks = pd.read_csv(data/'lcfm_replication_rank_cases.csv.gz')
    recon = pd.read_csv(data/'lcfm_replication_reconstruction_cases.csv.gz')
    secondary = pd.read_csv(data/'lcfm_replication_secondary.csv')
    assert len(ranks) == 42000 and len(recon) == 126000
    for frame, keys in [(ranks, ['task', 'episode', 'bank', 'arm', 'modality']),
                        (recon, ['task', 'episode', 'bank', 'arm', 'modality', 'horizon'])]:
        assert not frame.duplicated(keys).any()
        assert set(frame.arm) == set(protocol['arms'])
        assert set(frame.bank) == set(protocol['banks'])
        assert set(zip(frame.task, frame.episode)) == expected
    assert (ranks.excess_reference_cost >= 0).all()
    stable = ranks.winner_changed == 0
    assert (ranks.loc[stable, 'excess_reference_cost'] == 0).all()
    assert (ranks.loc[stable, 'selected_in_reference_minimum_set'] == 1).all()

    for item in report['primary']:
        rows = ranks[(ranks.task == item['task']) & (ranks.bank == 'original') &
                     (ranks.arm == 'all_blocks_h3_donor') & (ranks.modality == 'official')]
        assert len(rows) == 100 and set(rows.episode) == set(range(100))
        k = int(rows.winner_changed.sum())
        assert item['changed'] == k and item['fraction'] == k/100
        for confidence, field in [(.975, 'family_95_interval'), (.95, 'marginal_95_interval')]:
            independent = binomtest(k, 100).proportion_ci(confidence_level=confidence, method='wilson')
            np.testing.assert_allclose(item[field], independent, rtol=0, atol=1e-14)

    exact = ranks.arm.isin(['coherent_raw_h3', 'all_blocks_persistent_donor'])
    assert (ranks.loc[exact, 'winner_changed'] == 0).all()
    assert (ranks.loc[exact, 'excess_reference_cost'] == 0).all()
    assert (ranks.loc[exact, 'reference_top10_overlap'] == 10).all()
    np.testing.assert_allclose(ranks.loc[exact, 'reference_spearman'], 1, atol=1e-14)
    for arms, value in [(['native', 'zero_capture'], 0),
                        (['coherent_raw_h3', 'all_blocks_persistent_donor'], 1)]:
        assert (recon.loc[recon.arm.isin(arms), 'donor_reconstruction'] == value).all()
    assert (recon.loc[(recon.arm == 'all_blocks_h3_donor') & (recon.horizon == 3),
                      'donor_reconstruction'] == 1).all()

    # Recount every public secondary cell, including undefined measurements.
    for _, row in secondary.iterrows():
        frame = recon if row.metric == 'donor_reconstruction' else ranks
        mask = ((frame.task == row.task) & (frame.bank == row.bank) &
                (frame.arm == row.arm) & (frame.modality == row.modality))
        if frame is recon:
            mask &= frame.horizon == row.horizon
        values = frame.loc[mask, row.metric].to_numpy(float)
        assert len(values) == row.n == 100
        assert np.isfinite(values).sum() == row.n_defined
        if row.n_defined == 100:
            np.testing.assert_allclose(row['mean'], values.mean(), rtol=1e-12, atol=1e-13)
            assert np.isfinite([row.marginal_95_low, row.marginal_95_high]).all()
            assert row.marginal_95_low <= row.marginal_95_high
        else:
            assert pd.isna(row['mean']) and pd.isna(row.marginal_95_low) and pd.isna(row.marginal_95_high)
    print(json.dumps({'passed': True, 'independent_states': 200,
                      'selection_rows': len(ranks), 'reconstruction_rows': len(recon),
                      'secondary_cells': len(secondary), 'primary_intervals_checked_with': 'SciPy Wilson'}, indent=2))


if __name__ == '__main__':
    main()
