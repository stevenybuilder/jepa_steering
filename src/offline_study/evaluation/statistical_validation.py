"""Post-result statistical audit; never changes a frozen test or launches work.

Exact binomial bounds concern choice-change frequency, NOT mean physical gain.
Power calculations are normal-approximation planning sensitivities for a future
paired trial, not observed power, a sample-size freeze, or a promise of efficacy.
"""
import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist

from offline_study.experiments.decision_diagnostic import CONTRASTS, TASKS, validate_records
from offline_study.core.protocol import sha256, write_json


def binomial_cdf(k, n, p):
    if k < 0:
        return 0.0
    if k >= n or p == 0:
        return 1.0
    if p == 1:
        return 0.0
    terms = [math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1)
             + j * math.log(p) + (n - j) * math.log1p(-p) for j in range(k + 1)]
    largest = max(terms)
    return min(1.0, math.exp(largest) * math.fsum(math.exp(t - largest) for t in terms))


def exact_binomial_interval(k, n, alpha=.05):
    """Two-sided equal-tail Clopper-Pearson, via monotone binomial inversion."""
    if (type(k) is not int or type(n) is not int or n <= 0 or not 0 <= k <= n
            or not 0 < alpha < 1):
        raise ValueError("Require integer 0 <= count <= n and 0 < alpha < 1")
    def root(index, target):
        left, right = 0., 1.
        for _ in range(70):
            middle = (left + right) / 2
            if binomial_cdf(index, n, middle) > target:
                left = middle
            else:
                right = middle
        return (left + right) / 2
    return [0. if k == 0 else root(k - 1, 1 - alpha / 2),
            1. if k == n else root(k, alpha / 2)]


def paired_exact_p(wins, losses):
    if any(type(x) is not int or x < 0 for x in (wins, losses)):
        raise ValueError("Require nonnegative integer paired counts")
    total = wins + losses
    return min(1., 2 * sum(math.comb(total, j) for j in range(min(wins, losses) + 1)) / 2**total)


def holm(values):
    if not values or any(not 0 <= p <= 1 for p in values):
        raise ValueError("Invalid p-value family")
    output, previous = [0.] * len(values), 0.
    for position, index in enumerate(sorted(range(len(values)), key=lambda i: values[i])):
        previous = max(previous, min(1., (len(values) - position) * values[index]))
        output[index] = previous
    return output


def paired_sample_size(q, gain=.05, power=.8, family=8):
    """Approximate per-contrast n; q=P(win)+P(loss), gain=P(win)-P(loss).

    n = (z_(1-alpha/(2m))*sqrt(q) + z_power*sqrt(q-gain**2))**2/gain**2.
    This omits continuity/exact-test adjustments and training-seed variation.
    Per-contrast power is NOT joint power to pass both reference comparisons.
    """
    if (not 0 < gain <= q <= 1 or not .5 < power < 1
            or type(family) is not int or family < 1):
        raise ValueError("Invalid paired planning assumptions")
    normal = NormalDist()
    numerator = (normal.inv_cdf(1 - .05 / (2 * family)) * math.sqrt(q)
                 + normal.inv_cdf(power) * math.sqrt(q - gain**2))
    return math.ceil((numerator / gain)**2)


def audit_core(report):
    expected = {(t, a, b) for t in TASKS for a, b in CONTRASTS}
    rows = report['contrasts']
    if len(rows) != 8 or {(r['task'], r['arm'], r['control']) for r in rows} != expected:
        raise ValueError("Require exactly the original eight comparisons")
    output, pvalues = [], []
    for row in rows:
        n = row['scenario_clusters']
        wins, losses = row['discordant_win_clusters'], row['discordant_loss_clusters']
        if n != 96 or row['episodes'] != n or wins + losses > n:
            raise ValueError("Unexpected paired population")
        gain = (wins - losses) / n
        p = paired_exact_p(wins, losses)
        if (not math.isclose(100 * gain, row['gain_percentage_points'], abs_tol=1e-12)
                or not math.isclose(p, row['exact_discordance_p'], abs_tol=1e-12)):
            raise ValueError("Original counts/effect/exact test disagree")
        pvalues.append(p)
        q = (wins + losses) / n
        output.append({'task': row['task'], 'arm': row['arm'], 'control': row['control'],
            'n': n, 'wins': wins, 'losses': losses, 'unchanged': n - wins - losses,
            'gain_percentage_points': gain * 100, 'exact_p': p,
            'estimated_standard_error_percentage_points': 100 * math.sqrt((q - gain**2) / (n - 1)),
            'original_simultaneous_interval_percentage_points': row['simultaneous_95_interval_percentage_points'],
            'planning_discordance_estimate': q,
            'illustrative_n_for_5pp_80pct_per_contrast_power': paired_sample_size(q)})
    for original, row, adjusted in zip(rows, output, holm(pvalues)):
        if not math.isclose(adjusted, original['holm_p'], abs_tol=1e-12):
            raise ValueError("Original Holm family differs")
        row['holm_p'] = adjusted
    return output


def audit_decisions(root):
    protocol_path = root / 'protocol.json'
    digest = sha256(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    if digest != json.loads((root / 'FROZEN.json').read_text())['protocol_sha256']:
        raise ValueError("Changed diagnostic protocol")
    records = []
    for task in TASKS:
        for path in sorted((root / task).glob('episode-*/record.json')):
            done = json.loads((path.parent / 'DONE.json').read_text())
            for name, expected in done['files'].items():
                if Path(name).name != name or sha256(path.parent / name) != expected:
                    raise ValueError("Changed diagnostic raw file")
            records.append(json.loads(path.read_text()))
    found = validate_records(records, protocol['episodes'], digest)
    output = []
    for task in TASKS:
        rows = [r for (t, _), r in found.items() if t == task]
        for treatment, reference in CONTRASTS:
            count = sum(r['arms'][treatment]['selected'] != r['arms'][reference]['selected'] for r in rows)
            output.append({'task': task, 'treatment': treatment, 'reference': reference,
                'changes': count, 'n': len(rows),
                'marginal_95_choice_change_probability_interval': exact_binomial_interval(count, len(rows)),
                'bonferroni_8_choice_change_probability_interval': exact_binomial_interval(count, len(rows), .05 / 8)})
    return output, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', type=Path, required=True)
    parser.add_argument('--diagnostic', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    core = audit_core(json.loads(args.core.read_text()))
    decisions, protocol = audit_decisions(args.diagnostic)
    report = {'status': 'completed_post_result_supplement_not_independent_confirmation',
        'core_source_sha256': sha256(args.core), 'diagnostic_protocol_sha256': protocol,
        'analysis_source_sha256': sha256(Path(__file__)),
        'frozen_analyses_changed': False, 'new_model_or_simulator_calls': 0,
        'core_count_effect_exact_p_holm_rechecks': core, 'diagnostic_choice_change_bounds': decisions,
        'planning_sensitivity': [{'discordance_assumption': q, 'target_gain_pp': 100 * gain,
            'power_per_contrast': .8, 'family': 8,
            'approximate_paired_scenarios_per_task': paired_sample_size(q, gain)}
            for q in (.2, .3, .4, .5) for gain in (.05, .1)],
        'limits': ['Not fresh confirmation or an author three-seed history replication.',
            'Core recheck uses previously verified paired counts, not a new raw-rollout loader.',
            'Diagnostic bounds assume independent exchangeable scenarios within task and the fixed candidate-seed policy.',
            'Choice-change bounds are post-result supplemental estimates, not replacements for the frozen physical-progress endpoint.',
            'Zero empirical bootstrap width does not establish population equivalence.',
            'Power is an approximate prospective sensitivity, not observed power or a finalized launch sample size.',
            'Joint power, seed/checkpoint variance, eligibility and total cost need a separate future freeze.']}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'REPORT.json', report)
    write_json(args.output / 'DONE.json', {'report_sha256': sha256(args.output / 'REPORT.json'),
        'original_inputs_unchanged': True, 'paid_compute_started': False})
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
