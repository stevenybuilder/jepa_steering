"""Validate the focused LCFM paper against its complete public replication."""
import csv
import hashlib
import json
from pathlib import Path
import re

from check_manuscript import CheckError, check_citations, check_figures, without_comments

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_SHA = '3769fa71bad0f6e954a6a801a7c23df9ed6abc96ff53958caf5e4a2dc628883e'


def expected_numbers(root=ROOT):
    source = root/'paper/data/lcfm_replication_summary.json'
    if hashlib.sha256(source.read_bytes()).hexdigest() != SUMMARY_SHA:
        raise CheckError('The complete replication summary changed')
    summary = json.loads(source.read_text())
    table = root/'paper/data/lcfm_replication_secondary.csv'
    if hashlib.sha256(table.read_bytes()).hexdigest() != summary['outputs'][table.name]:
        raise CheckError('Replication table hash differs from summary')
    with table.open() as f:
        rows = list(csv.DictReader(f))
    def cell(task, bank, metric, horizon=None):
        matches = [r for r in rows if r['task'] == task and r['bank'] == bank
                   and r['arm'] == 'all_blocks_h3_donor' and r['modality'] == 'official'
                   and r['metric'] == metric and (horizon is None or float(r['horizon']) == horizon)]
        if len(matches) != 1:
            raise CheckError('Missing or duplicate manuscript source cell')
        return matches[0]
    expected = {}
    for r in summary['primary']:
        prefix = 'LCFMReach' if r['task'] == 'reach' else 'LCFMWall'
        expected[prefix+'Changed'] = str(r['changed'])
        lo, hi = r['family_95_interval']
        expected[prefix+'CI'] = f'{100*lo:.1f}--{100*hi:.1f}\\%'
        c = cell(r['task'], 'original', 'excess_reference_cost_percent')
        expected[prefix+'Cost'] = f'{float(c["mean"]):.2f}\\%'
        expected[prefix+'CostCI'] = f'{float(c["marginal_95_low"]):.2f}--{float(c["marginal_95_high"]):.2f}\\%'
    for metric, key, digits, suffix, h in [
        ('excess_reference_cost_percent', 'LCFMCostRange', 2, r'\%', None),
        ('donor_reconstruction', 'LCFMReconRange', 4, '', 6),
        ('reference_spearman', 'LCFMRankRange', 3, '', None),
        ('reference_top10_overlap', 'LCFMEliteRange', 2, '', None)]:
        values = [float(cell(t, b, metric, h)['mean']) for t in ('reach','reach-wall') for b in ('original','fresh')]
        expected[key] = f'{min(values):.{digits}f}--{max(values):.{digits}f}'+suffix
    return expected


def check_numbers(content, expected):
    definitions = re.findall(r'\\newcommand\{\\(LCFM\w+)\}\{([^{}]*)\}', content)
    if len(definitions) != len(dict(definitions)) or dict(definitions) != expected:
        raise CheckError('Manuscript numbers differ from complete replication')
    if SUMMARY_SHA not in content:
        raise CheckError('Missing replication source identity')


def check(root=ROOT, tex=None, numbers=None, checklist=None):
    base = root/'paper/lcfm'
    tex = without_comments((base/'main.tex').read_text() if tex is None else tex)
    numbers = (base/'replication_numbers.tex').read_text() if numbers is None else numbers
    checklist = (base/'checklist.tex').read_text() if checklist is None else checklist
    expected = expected_numbers(root)
    check_numbers(numbers, expected)
    if r'\input{replication_numbers.tex}' not in tex:
        raise CheckError('Manuscript must use generated replication numbers')
    if set(re.findall(r'\\(LCFM\w+)', tex)) != set(expected):
        raise CheckError('Missing or unknown replication result macro')
    labels = set(re.findall(r'\\label\{([^}]+)\}', tex))
    refs = set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}', tex+'\n'+checklist))
    if refs-labels:
        raise CheckError('Undefined section/equation references: '+str(sorted(refs-labels)))
    figures = check_figures(tex, base)
    citations = check_citations(tex, (base/'references.bib').read_text())
    return {'passed': True, 'replication_macros': len(expected),
            'figure_includes': figures, 'citation_keys': citations}


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
