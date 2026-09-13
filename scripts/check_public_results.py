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


def check_headline(root, readme, report):
    sources = json.loads((root / 'paper/data/benchmark_comparison_sources.json').read_text())
    section = readme.split('### Robot success and published benchmark context')[1].split('## Architecture')[0]
    rows = [line.split('|')[1:-1] for line in section.splitlines() if re.match(r'^\|[^|]+\|[^|]+\|\s*\d', line)]
    expected = []
    for author in sources['author_rows']:
        expected.append((author['label'], 'Published reference',
                         [author['values'][sources['task_order'].index(t)] for t in TASKS]))
    for label, arm in (('Unsteered JEPA-WM', 'native'), ('Refined four-direction edit', 'fixed_rank4')):
        expected.append((label, 'Our protected evaluation',
                         [report['results'][t]['success_percent'][arm] for t in TASKS]))
    assert len(rows) == len(expected), 'Missing headline comparison row'
    for row, (label, source, values) in zip(rows, expected):
        assert [cell.strip() for cell in row[:2]] == [label, source], 'Headline source/label mismatch'
        displayed = [Decimal(str(v)).quantize(Decimal('.01'), rounding=ROUND_HALF_UP) for v in values]
        assert [Decimal(cell.strip()) for cell in row[2:]] == displayed, 'Headline value mismatch'
    delta_text = section.split('Against concurrent unsteered JEPA-WM')[1].split('percentage points')[0]
    actual = [Decimal(n) for n in re.findall(r'[+−-]?\d+\.\d+', delta_text.replace('−', '-'))]
    deltas = [(Decimal(str(report['results'][t]['success_percent']['fixed_rank4'])) -
               Decimal(str(report['results'][t]['success_percent']['native']))).quantize(
                   Decimal('.01'), rounding=ROUND_HALF_UP) for t in TASKS]
    assert actual == deltas, 'Headline paired delta mismatch'


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
    rows = []
    full_readme = (root / 'README.md').read_text()
    check_headline(root, full_readme, report)
    readme = full_readme.split('## Final protected results')[1].split('## What')[0]
    for line in readme.splitlines():
        if re.match(r'^\|[^|]+\|\s*\d', line):
            rows.append([float(v.strip()) for v in line.split('|')[2:-1]])
    assert len(rows) == len(ARMS), 'Missing README result row'
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
            assert rows[ai][ti] == display, (task, arm, 'README mismatch')
            if arm != 'native':
                a = audit[task][arm]
                assert count - audit[task]['native']['success'] == a['rescue'] - a['regress']
        for value in result['contrasts'].values():
            lo, hi = value['simultaneous_95_interval_pp']
            assert lo <= 0 <= hi
    return report, audit


if __name__ == '__main__':
    check()
    print('PASS: report SHA256, 32 full-panel + 20 headline rates, paired deltas, 48 intervals, and rescue/regression identities')
