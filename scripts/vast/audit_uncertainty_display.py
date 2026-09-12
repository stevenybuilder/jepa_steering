"""Supplementary display audit, never a replacement for frozen inference.

Enumerates the empirical bootstrap distribution of the 96 distinct scenario
differences from their sufficient win/loss counts. Exact enumeration removes
Monte Carlo error; it does NOT make a percentile interval an exact-coverage CI.
"""
import hashlib
import json
import math
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT/'artifacts/offline_study/core-completion-preservation-20260908-v1/ANALYSIS_REPORT.json'
OUTPUT = PROJECT/'artifacts/offline_study/table-completion-20260911-v1/UNCERTAINTY_DISPLAY_AUDIT.json'


def bootstrap_interval(n, wins, losses, alpha):
    counts = (wins, losses, n-wins-losses)
    probabilities = [x/n for x in counts]
    pmf = {}
    for w in range(n+1):
        for l in range(n-w+1):
            draws = (w, l, n-w-l)
            if any(k and not p for k,p in zip(draws, probabilities)):
                continue
            logp = math.lgamma(n+1)-sum(math.lgamma(x+1) for x in draws)
            logp += sum(k*math.log(p) for k,p in zip(draws, probabilities) if k)
            pmf[w-l] = pmf.get(w-l, 0.)+math.exp(logp)
    assert math.isclose(sum(pmf.values()), 1., abs_tol=1e-10)
    def quantile(q):
        cumulative = 0.
        for difference, mass in sorted(pmf.items()):
            cumulative += mass
            if cumulative >= q:
                return 100*difference/n
        raise ValueError('Bad bootstrap CDF')
    return [quantile(alpha/2), quantile(1-alpha/2)]


def main():
    source = json.loads(SOURCE.read_text())
    assert source['analysis']['family'] == 8
    result = {'role': 'supplementary_uncertainty_display_audit',
        'source_report_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'original_frozen_analysis_unchanged': True, 'new_gpu_calls': 0,
        'new_simulator_episodes': 0, 'training_seed_variability_estimated': False,
        'method': 'exact enumeration of empirical percentile bootstrap; nominal coverage not guaranteed',
        'contrasts': []}
    for c in source['contrasts']:
        n = c['episodes']; w = c['discordant_win_clusters']; l = c['discordant_loss_clusters']
        assert n == c['scenario_clusters'] == 96
        delta = (w-l)/n
        assert math.isclose(100*delta, c['gain_percentage_points'])
        result['contrasts'].append({'task': c['task'], 'arm': c['arm'], 'control': c['control'],
            'success_gain_pp': 100*delta, 'wins': w, 'losses': l, 'ties': n-w-l,
            'paired_bootstrap_se_pp': 100*math.sqrt(((w+l)/n-delta**2)/n),
            'supplementary_unadjusted_95_interval_pp': bootstrap_interval(n,w,l,.05),
            'enumerated_eight_comparison_interval_pp': bootstrap_interval(n,w,l,.05/8),
            'original_frozen_simultaneous_interval_pp': c['simultaneous_95_interval_percentage_points'],
            'original_unadjusted_exact_p': c['exact_discordance_p'], 'original_holm_p': c['holm_p']})
    assert all(c['original_unadjusted_exact_p'] > .05 for c in result['contrasts'])
    text = json.dumps(result, indent=2)+'\n'
    if OUTPUT.exists():
        assert OUTPUT.read_text() == text
    else:
        with OUTPUT.open('x') as f:
            f.write(text)
    print(text)


if __name__ == '__main__':
    main()
