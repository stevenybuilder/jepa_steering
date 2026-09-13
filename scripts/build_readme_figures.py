"""Generate README figures from checked aggregates. No model or GPU calls."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

from check_public_results import ROOT, TASKS, check

OUT = ROOT / 'docs/figures'
BLUE, TEAL, RED, INK = '#3264a8', '#087f8c', '#c85c48', '#223248'
LABELS = dict(zip(TASKS, ('Reach', 'Reach-Wall', 'PointMaze', 'Wall')))


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{name}.png', dpi=180, facecolor='white', bbox_inches='tight')
    fig.savefig(OUT / f'{name}.svg', facecolor='white', bbox_inches='tight')
    svg = OUT / f'{name}.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)


def overview():
    fig, ax = plt.subplots(figsize=(12, 4.3))
    ax.set(xlim=(0, 12), ylim=(0, 4.3))
    ax.axis('off')
    ax.text(.05, 4.05, 'A forecast correction is not yet a planning improvement',
            fontsize=18, weight='bold', color=INK)
    boxes = [
        (.05, '01  FIT + DEVELOP', 'Recorded trajectories\nFit directions and response maps\nEvaluate forecast error'),
        (4.05, '02  FREEZE', 'Task-specific interventions\nEight arms + inputs + RNG\nPrespecified paired analysis'),
        (8.05, '03  CONFIRM', 'New simulator scenarios\n4 tasks × 96 scenarios × 8 arms\n3,072 completed evaluations')]
    for x, title, body in boxes:
        ax.add_patch(FancyBboxPatch((x, 1.5), 3.65, 1.9, boxstyle='round,pad=.04',
                                   facecolor='#eef3f8', edgecolor='#d3dde9'))
        ax.text(x+.18, 3.02, title, color=BLUE, weight='bold', fontsize=12)
        ax.text(x+.18, 2.55, body, color=INK, fontsize=11, va='top', linespacing=1.7)
    for x in (3.78, 7.78):
        ax.annotate('', (x+.25, 2.42), (x-.04, 2.42), arrowprops={'arrowstyle':'->', 'color':INK})
    ax.text(.05, .91, 'Frozen predictor + activation edit → latent forecasts → CEM planner → simulator',
            fontsize=13, color=TEAL, weight='bold')
    ax.text(.05, .39, 'Base weights unchanged. Offline and protected evidence remain separate. No held-out-task transfer claim.',
            fontsize=10, color='#526174')
    save(fig, 'study_overview')


def outcomes(report, audit):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios':[1.35, 1]})
    labels = []
    for ti, task in enumerate(TASKS):
        for ai, (arm, short, color) in enumerate((('fixed_rank4','Refined',BLUE), ('coupling_only','Coupling',TEAL))):
            y = ti*2+ai
            labels.append(f'{LABELS[task]} · {short}')
            row = report['results'][task]['contrasts'][arm+'-native']
            value = row['difference_pp']
            lo, hi = row['simultaneous_95_interval_pp']
            ax.errorbar(value, y, xerr=[[value-lo], [hi-value]], fmt='o', color=color, capsize=4, lw=2)
            a = audit[task][arm]
            bx.barh(y, -a['regress'], color=RED, height=.55)
            bx.barh(y, a['rescue'], color=TEAL, height=.55)
            bx.text(-a['regress']-.7, y, str(a['regress']), ha='right', va='center', fontsize=10)
            bx.text(a['rescue']+.7, y, str(a['rescue']), ha='left', va='center', fontsize=10)
    ax.set_yticks(range(8), labels)
    bx.set_yticks(range(8), ['']*8)
    for a in (ax, bx):
        a.set_ylim(7.7, -.8)
        a.axvline(0, color='#9ba7b4', lw=1)
        a.spines[['top','right','left']].set_visible(False)
        a.tick_params(axis='y', length=0)
        a.grid(axis='x', color='#e8edf2', zorder=0)
        a.set_axisbelow(True)
    ax.set_title('Paired effect vs concurrent native', loc='left', weight='bold', pad=16)
    ax.set_xlabel('Success-rate difference (percentage points)')
    bx.set_title('Regressions  ←  0  →  Rescues', loc='left', weight='bold', pad=16)
    bx.set_xlim(-27, 27)
    bx.set_xticks([-20, -10, 0, 10, 20], ['20','10','0','10','20'])
    bx.set_xlabel('Paired scenarios (96 per task)')
    fig.suptitle('Protected evaluation: effects are uncertain; outcome changes run both ways',
                 fontsize=15, weight='bold', color=INK, x=.02, ha='left')
    fig.text(.02, .01, 'Intervals: simultaneous 95%, Bonferroni family of 48; 20,000 paired bootstrap draws. Eight selected contrasts shown; all 48 include zero.',
             fontsize=9, color='#526174')
    fig.tight_layout(rect=(0, .06, 1, .93))
    save(fig, 'protected_results')


def basis():
    data = json.loads((ROOT/'paper/data/basis_spatial_summary.json').read_text())['tasks']
    tasks = ('reach', 'reach-wall')
    maps = [np.array(data[t]['patch_squared_loadings']).reshape(4,16,16)*100 for t in tasks]
    assert all(np.allclose(m.sum(axis=(1,2)), 100) for m in maps)
    fig, axes = plt.subplots(2, 5, figsize=(11, 4.8), layout='constrained')
    vmax = max(float(m.max()) for m in maps)
    for row, (task, m) in enumerate(zip(tasks, maps)):
        for col in range(5):
            values = m[col] if col < 4 else m.mean(axis=0)
            im = axes[row,col].imshow(values, cmap='viridis', vmin=0, vmax=vmax, interpolation='nearest')
            axes[row,col].set_xticks([]); axes[row,col].set_yticks([])
            if row == 0:
                axes[row,col].set_title(f'Direction {col+1}' if col < 4 else 'Subspace average', fontsize=10)
        axes[row,0].set_ylabel(LABELS[task], weight='bold')
    fig.colorbar(im, ax=axes, shrink=.65, label='Squared loading per patch (%)')
    fig.suptitle('Fitted basis geometry — fixed weights, not attention or saliency', color=INK, weight='bold')
    save(fig, 'fitted_basis')


if __name__ == '__main__':
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'svg.fonttype':'none'})
    report, audit = check()
    overview(); outcomes(report, audit); basis()
    print(f'Wrote three PNG/SVG figures to {OUT}')
