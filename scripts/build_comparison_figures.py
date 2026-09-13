"""Reproduce benchmark bars from provenance-separated, hash-checked aggregates.

No model calls, inference, best-arm selection, cross-source effect calculation,
or synthesized uncertainty. JSON is the curated source; CSV is generated output.
"""
import csv
import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from check_public_results import ARMS, REPORT_SHA, ROOT, TASKS
from build_readme_figures import save

SOURCE = ROOT / 'paper/data/benchmark_comparison_sources.json'
TASK_LABELS = {'reach': 'Reach', 'reach-wall': 'Reach-Wall', 'pusht': 'Push-T',
               'pointmaze': 'PointMaze', 'wall': 'Wall', 'droid': 'DROID† score'}
SHORT = ['Native', 'Refined', 'Random R', 'Coupling', 'Random V/A', 'Joint', 'Visual', 'Action']
COLORS = ['#626a67', '#4e6fa5', '#a9bbd6', '#24796c', '#98bfb2', '#b67930', '#cfa86d', '#b05443']
INK, MUTED, RULE = '#242a26', '#73776f', '#dedfd6'


def load_data():
    source = json.loads(SOURCE.read_text())
    assert source['fresh_source']['sha256'] == REPORT_SHA
    raw = (ROOT / source['fresh_source']['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == REPORT_SHA, 'Frozen report changed'
    report = json.loads(raw)
    assert report['scientific_evaluations'] == 3072
    assert report['analysis']['historical_results_pooled'] is False
    assert set(report['results']) == set(TASKS)
    assert [r['id'] for r in source['development_rows']] == list(ARMS)
    assert len(source['author_rows']) == 3
    for row in source['author_rows'] + source['development_rows']:
        assert len(row['values']) == 6 and all(0 <= v <= 100 for v in row['values'])
    for result in report['results'].values():
        assert result['n'] == 96 and set(result['success_percent']) == set(ARMS)
        assert len(result['contrasts']) == 12
        for contrast in result['contrasts'].values():
            lo, hi = contrast['simultaneous_95_interval_pp']
            assert lo <= 0 <= hi
    return source, report


def display(value, places=1):
    return str(Decimal(str(value)).quantize(Decimal('0.'+'0'*places), rounding=ROUND_HALF_UP))


def bars(ax, values, labels, colors, task, *, historical=False, size=8):
    positions = np.arange(len(values))
    ax.bar(positions, values, width=.73, color=colors, edgecolor='white', linewidth=.4, zorder=3)
    # Labels are absolute rates, not deltas against an incomparable baseline.
    for x, value in zip(positions, values):
        ax.text(x, value+2, display(value), ha='center', va='bottom', fontsize=size,
                color=INK, rotation=90 if len(values)>3 else 0, weight='medium')
    ax.set(ylim=(0, 106), xlim=(-.7, len(values)-.3))
    ax.set_yticks([0, 50, 100])
    ax.set_xticks(positions, labels, rotation=52, ha='right', rotation_mode='anchor')
    ax.tick_params(axis='x', labelsize=size, length=0, pad=3)
    ax.tick_params(axis='y', labelsize=size, length=0, pad=3, colors=MUTED)
    ax.spines[['top', 'right']].set_visible(False)
    for key in ('left', 'bottom'):
        ax.spines[key].set_color(RULE)
    ax.grid(axis='y', color='#edf0eb', zorder=0, linewidth=.6)
    ax.set_axisbelow(True)
    ax.set_title(TASK_LABELS[task], fontsize=12, weight='bold', color=INK, pad=12)
    if task == 'droid':
        ax.set_facecolor('#faf7f1')
    if historical and task in ('reach', 'reach-wall'):
        # Mark the three component arms rather than implying a canonical pairing.
        ax.set_xticks(positions, [x + ('*' if i >= 5 else '') for i, x in enumerate(labels)],
                      rotation=52, ha='right', rotation_mode='anchor')


def overview(source, report):
    """Aligned task columns, three explicitly separate evidence populations."""
    tasks = source['task_order']
    fig, axes = plt.subplots(3, 6, figsize=(16.8, 11.2))
    fig.subplots_adjust(left=.055, right=.985, bottom=.105, top=.867, wspace=.29, hspace=1.30)
    fig.text(.035,.967,'Benchmark context, without mixing the evidence',size=23,weight='bold',color=INK)
    fig.text(.035,.931,'Separate populations, checkpoints and planner RNG. These bars are descriptive, not cross-study treatment effects.',size=11,color=MUTED)
    author_labels = ['DINO-WM', 'JEPA improved', 'JEPA final‡']
    headings = [
        (.898, 'A  AUTHOR-REPORTED', 'CEM L2 · Tables 2/11; ‡final checkpoint from Table 12'),
        (.598, 'B  HISTORICAL DEVELOPMENT', 'Earlier six-task measurements · not the protected panel'),
        (.302, 'C  FRESH PROTECTED', '96 new scenarios/task × all 8 arms · concurrent native within each scenario')]
    for y,title,subtitle in headings:
        fig.text(.035,y,title,size=12,weight='bold',color=INK)
        fig.text(.265,y,subtitle,size=10.5,color=MUTED)
    for col, task in enumerate(tasks):
        bars(axes[0,col], [row['values'][col] for row in source['author_rows']],
             author_labels, ['#a6aaa6','#4e6fa5','#91a6c7'], task, size=8.8)
        bars(axes[1,col], [row['values'][col] for row in source['development_rows']],
             SHORT, COLORS, task, historical=True, size=8)
        if task in report['results']:
            bars(axes[2,col], [report['results'][task]['success_percent'][a] for a in ARMS],
                 SHORT, COLORS, task, size=8)
        else:
            axes[2,col].set_axis_off()
            axes[2,col].text(.5,.53,'Not rerun\nin protected panel',ha='center',va='center',
                            transform=axes[2,col].transAxes,fontsize=11,color=MUTED,linespacing=1.5)
        for row in range(3):
            if col == 0:
                axes[row,col].set_ylabel('Success (%)', fontsize=10, color=INK)
    fig.text(.035,.023,'* Historical MetaWorld Joint / Visual / Action used native 45.83 / 29.17, not the canonical 44.79 / 30.21 bars.',fontsize=9.5,color=MUTED)
    fig.text(.035,.000,'† DROID is recorded-action agreement, not physical success. Point estimates only; no comparable error bars. All 48 fresh paired intervals include zero.',fontsize=9.5,color=MUTED)
    save(fig, 'benchmark_comparison',pdf=True)


def protected(report):
    fig, axes = plt.subplots(1,4,figsize=(13.5,4.9))
    fig.subplots_adjust(left=.06,right=.985,bottom=.25,top=.73,wspace=.23)
    fig.text(.035,.93,'Fresh protected evaluation: all eight arms',fontsize=21,weight='bold',color=INK)
    fig.text(.035,.86,'4 tasks × 96 scenarios × 8 paired arms. Frozen checkpoint, inputs and analysis; no historical observations pooled.',fontsize=10.5,color=MUTED)
    for ax, task in zip(axes,TASKS):
        bars(ax,[report['results'][task]['success_percent'][a] for a in ARMS],SHORT,COLORS,task,size=9)
    axes[0].set_ylabel('Simulator success (%)',fontsize=10,color=INK)
    fig.text(.035,.035,'Point estimates, not evidence of a reliable gain: all 48 simultaneous 95% paired contrast intervals include zero. See the paired-effect figure.',fontsize=10,color=MUTED)
    save(fig,'protected_benchmark',pdf=True)


def headline(source, report):
    """Published references are visually separated from within-cohort comparisons."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 5.4))
    fig.subplots_adjust(left=.045, right=.987, bottom=.28, top=.76, wspace=.22)
    fig.text(.035, .94, 'Steering a frozen world model: benchmark context and paired outcomes',
             fontsize=22, weight='bold', color=INK)
    fig.text(.035, .875, 'Published DINO-WM / JEPA-WM references are separate from our fresh 96-scenario evaluations. No cross-study pairing.',
             fontsize=11, color=MUTED)
    labels = ['DINO-WM', 'JEPA improved', 'JEPA final', *SHORT]
    for ax, task in zip(axes, TASKS):
        ti = source['task_order'].index(task)
        values = [row['values'][ti] for row in source['author_rows']]
        values += [report['results'][task]['success_percent'][arm] for arm in ARMS]
        bars(ax, values, labels, ['#c4c7c3', '#92998f', '#a7b09e', *COLORS], task, size=8)
        ax.axvspan(-.65, 2.45, color='#eef0ea', zorder=0)
        ax.axvline(2.5, color='#959c92', linewidth=.7, linestyle='--')
        ax.text(1, 105, 'Published', ha='center', va='bottom', fontsize=8, color=MUTED)
        ax.text(6.5, 105, 'Ours: fresh scenarios', ha='center', va='bottom', fontsize=8, color=MUTED)
        ax.set_ylim(0, 116)
    axes[0].set_ylabel('Success (%)', fontsize=11, color=INK)
    fig.text(.035, .04, 'All eight local arms shown; compare edits with their concurrent Native and randomized-edit references. Point estimates, not inferred cross-paper gains.', fontsize=10, color=MUTED)
    save(fig, 'headline_benchmark', pdf=True)


def export(source, report):
    rows=[]
    for population, source_key in [('author','author_rows'),('development','development_rows')]:
        for arm in source[source_key]:
            for task, value in zip(source['task_order'],arm['values']):
                component_native = (source['development_source']['component_native'].get(task)
                                    if population == 'development' and arm['id'] in source['development_source']['component_native_applies_to'] else None)
                rows.append({'population':population,'task':task,'arm':arm['id'],'value_percent':value,
                    'metric':source['metric_by_task'][task],'n_per_arm':None,'concurrent_component_native_percent':component_native,
                    'source':source['author_source']['url']+('#A7.T12' if arm['table']=='12' else '#S5.T2') if population=='author' else 'user-supplied historical table; docs/RESULTS.md',
                    'source_sha256':None})
    for task in TASKS:
        for arm in ARMS:
            rows.append({'population':'fresh_protected','task':task,'arm':arm,
                'value_percent':report['results'][task]['success_percent'][arm],
                'metric':source['metric_by_task'][task],'n_per_arm':96,'concurrent_component_native_percent':None,
                'source':source['fresh_source']['path'],'source_sha256':REPORT_SHA})
    assert len(rows)==98 and len({(r['population'],r['task'],r['arm']) for r in rows})==98
    with (ROOT/'paper/data/benchmark_comparison.csv').open('w', newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    captions={
        'ablation_architecture':'Inference schematic for the frozen six-block JEPA-WM checkpoints used here. The visual-input (V), B3 action-conditioning (A), and rank-four B3-output (R) markers identify alternative hook sites at H3 within H6. Actions normally condition all blocks. CEM scores 300 candidate plans by encoded-goal L2 cost, executes a prefix, and replans. No R+V/A arm is tested.',
        'ablation_table':'The complete eight-arm design. R-only corrections and V/A interventions are alternatives. Equal-budget learned and random V/A components share scaled standardized doses; unscaled joint and drop-one arms retain the original per-component dose.',
        'benchmark_comparison':'Descriptive comparison across three separate evidence sources: author-reported JEPA-WM paper aggregates, historical development measurements, and fresh protected confirmation. Rows are not pooled or paired across sources. DROID is recorded-action agreement rather than physical robot success. Historical MetaWorld component arms have their own concurrent native 45.83/29.17. All fresh arms are shown in protocol order; the full 48-comparison simultaneous interval family includes zero.',
        'protected_benchmark':'Success rates on the complete protected four-task, eight-arm, 96-scenario panel. These are descriptive bars without synthesized error bars; inferential comparisons are paired against the fresh concurrent native. All 48 simultaneous 95% contrast intervals include zero.',
        'headline_benchmark':'Published DINO-WM, improved JEPA-WM and final-checkpoint references in a shaded band, alongside all eight fresh arms in fixed protocol order. Published and fresh values differ in checkpoint/aggregation/cohort and are not paired treatment comparisons. No error bars are synthesized; paired uncertainty remains in the frozen result report.'}
    provenance={'schema_version':1,'generator':'scripts/build_comparison_figures.py','curated_source':str(SOURCE.relative_to(ROOT)),
                'curated_source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'fresh_report_sha256':REPORT_SHA,
                'numeric_rows':len(rows),'captions':captions,'rendering':'Matplotlib; vector text retained; point estimates only; no outcome-sorted arms'}
    (ROOT/'paper/data/benchmark_figure_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')


if __name__ == '__main__':
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','svg.hashsalt':'jepa-comparison-v1'})
    source, report=load_data()
    overview(source,report);protected(report);headline(source,report);export(source,report)
    print('PASS: 98 provenance-separated numeric cells, frozen report SHA, all eight arms, 48 intervals. Wrote two SVG/PNG pairs and CSV/provenance.')
