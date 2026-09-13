"""CPU consistency checks of published aggregates; not a raw-episode replay."""
import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
TASKS = ('reach', 'reach-wall', 'pointmaze', 'wall')
ARMS = ('native', 'fixed_rank4', 'matched_random_fixed_rank4', 'coupling_only',
        'matched_random_coupling', 'joint', 'visual_only', 'action_condition_only')
REPORT_SHA = '8123d71497835fc164f09f5094c430308647a3ee2462c66746baea32633a0b15'


def best_observed_edit(report, task):
    """Descriptive post-hoc maximum over all seven edits; preserve every tie."""
    rates = report['results'][task]['success_percent']
    assert set(rates) == set(ARMS), 'Incomplete best-edit selection pool'
    pool = tuple(arm for arm in ARMS if arm != 'native')
    best = max(rates[arm] for arm in pool)
    return {'success_percent': best,
            'arms': tuple(arm for arm in pool if abs(rates[arm] - best) < 1e-10)}


def check_headline(root, readme, report):
    section = readme.split('### Robot success and published benchmark context')[1].split('\n## ')[0]
    assert 'post hoc' in section.lower() and 'seven' in section.lower(), 'Missing best-edit selection disclosure'
    delta_text = section.split('Against concurrent unsteered JEPA-WM')[1].split('percentage points')[0]
    actual = [Decimal(n) for n in re.findall(r'[+−-]?\d+\.\d+', delta_text.replace('−', '-'))]
    deltas = [(Decimal(str(best_observed_edit(report, t)['success_percent'])) -
               Decimal(str(report['results'][t]['success_percent']['native']))).quantize(
                   Decimal('.01'), rounding=ROUND_HALF_UP) for t in TASKS]
    assert actual == deltas, 'Headline paired delta mismatch'


def check_full_table(root, readme, report):
    sources = json.loads((root / 'paper/data/benchmark_comparison_sources.json').read_text())
    section = readme.split('## Final protected results and earlier benchmarks')[1].split('## What')[0]
    rows = [[v.strip() for v in line.split('|')[1:-1]] for line in section.splitlines()
            if re.match(r'^\| (Published|Development|Protected) \|', line)]
    author_by_id = {r['id']: r for r in sources['author_rows'] + sources['additional_author_rows']}
    expected = [('Published', label, author_by_id[key]['values']) for key, label in (
        ('dino_wm', 'DINO-WM'), ('jepa_improved', 'JEPA-WM recipe, CEM-L2'),
        ('jepa_cem_l1', 'JEPA-WM recipe, CEM-L1'), ('jepa_final', 'JEPA-WM final, CEM-L2'))]
    names = ('Unsteered', 'Refined four-direction', 'Calibrated random subspace',
             'Equal-budget coupling', 'Dose-matched random directions', 'Unscaled joint',
             'Visual only', 'Action-conditioning only')
    assert [r['id'] for r in sources['development_rows']] == list(ARMS)
    expected.extend(('Development', label, row['values']) for label, row in zip(names, sources['development_rows']))
    expected.extend(('Protected', label, [report['results'][task]['success_percent'][arm] if task in TASKS else None
                                         for task in sources['task_order']]) for label, arm in zip(names, ARMS))
    assert len(rows) == len(expected) == 20, 'Missing full comparison row'
    for actual, (stage, label, values) in zip(rows, expected):
        assert actual[:2] == [stage, label], 'Table stage/label mismatch'
        display = ['—' if v is None else str(Decimal(str(v)).quantize(Decimal('.01'), rounding=ROUND_HALF_UP)) for v in values]
        assert actual[2:] == display, f'Table value mismatch: {stage}/{label}'
    return {arm: {task: float(rows[12+ai][2+sources['task_order'].index(task)]) for task in TASKS}
            for ai, arm in enumerate(ARMS)}


def check(root=ROOT):
    raw = (root / 'reports/fresh-confirmation/report.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == REPORT_SHA, 'Final report changed'
    report = json.loads(raw)
    audit = json.loads((root / 'reports/fresh-confirmation/mechanism_audit.json').read_text())
    assert report['scientific_evaluations'] == 3072
    assert set(report['results']) == set(TASKS)
    assert report['analysis']['family_size'] == 48
    assert report['analysis']['bootstrap_draws'] == 20000
    assert report['analysis']['historical_results_pooled'] is False
    full_readme = (root / 'README.md').read_text()
    check_headline(root, full_readme, report)
    rows = check_full_table(root, full_readme, report)
    for ti, task in enumerate(TASKS):
        result = report['results'][task]
        assert result['n'] == 96 and set(result['success_percent']) == set(ARMS)
        assert len(result['contrasts']) == 12
        for ai, arm in enumerate(ARMS):
            count = audit[task][arm]['success']
            pct = result['success_percent'][arm]
            assert audit[task][arm]['n'] == 96 and 0 <= count <= 96
            assert abs(pct - count / 96 * 100) < 1e-9
            # Display exact count-based percentages with conventional half-up rounding.
            display = float((Decimal(count) * 100 / 96).quantize(Decimal('.01'), rounding=ROUND_HALF_UP))
            assert rows[arm][task] == display, (task, arm, 'README mismatch')
            if arm != 'native':
                a = audit[task][arm]
                assert count - audit[task]['native']['success'] == a['rescue'] - a['regress']
        for value in result['contrasts'].values():
            lo, hi = value['simultaneous_95_interval_pp']
            assert lo <= 0 <= hi
    return report, audit


if __name__ == '__main__':
    check()
    print('PASS: report SHA256, 104 provenance-separated table values, paired deltas, 48 intervals, and rescue/regression identities')
