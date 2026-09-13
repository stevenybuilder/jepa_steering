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


def architecture():
    """Schematic of actual hook sites, not a measurement or simultaneous edit."""
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set(xlim=(0, 16), ylim=(0, 9.6))
    ax.axis('off')

    def box(x, y, w, h, text, color='#eef3f8', edge='#bdcbdc', size=10):
        ax.add_patch(FancyBboxPatch((x,y), w,h, boxstyle='round,pad=.035',
                                   facecolor=color, edgecolor=edge, linewidth=1.2))
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=size, color=INK)

    def arrow(start, end, color='#64748b', style='-'):
        ax.annotate('', end, start, arrowprops={'arrowstyle':'->','color':color,
                                               'lw':1.5,'linestyle':style})

    ax.text(.1,9.23,'Where we intervene in JEPA-WM',fontsize=22,weight='bold',color=INK)
    ax.text(.1,8.82,'Frozen model • batched candidate forecasts • alternative ablation arms',
            fontsize=12,color='#526174')
    box(.1,6.9,1.8,.95,'Observation\nimage + state')
    box(2.25,6.9,2.05,.95,'Frozen encoders\nvisual + proprio')
    arrow((1.94,7.38),(2.2,7.38)); arrow((4.35,7.38),(4.53,7.38))
    # The V hook changes only the visual stream at the predictor input.
    box(4.57,7.14,.55,.48,'V',color='#dceef0',edge=TEAL,size=12)
    arrow((5.17,7.38),(5.48,7.38))
    box(5.55,6.45,5.25,1.9,'',color='#f8fafc')
    ax.text(8.17,8.48,'Dynamics predictor at imagined step H3',ha='center',size=11,weight='bold',color=INK)
    for i in range(6):
        x=5.76+i*.82
        box(x,7.0,.6,.77,f'B{i}',color='#dce7f5' if i==3 else 'white',edge=BLUE if i==3 else '#bdcbdc')
        if i<5: arrow((x+.64,7.38),(x+.78,7.38))
    # R sits on the output edge of B3, not on the action-conditioning input.
    ax.text(8.94,7.97,'R',ha='center',weight='bold',size=13,color=BLUE)
    arrow((8.94,7.82),(8.94,7.4),BLUE)
    ax.text(5.82,6.65,'Unrolled to H6',ha='left',size=10,color='#526174')
    box(11.15,6.9,1.55,.95,'Latent\nforecasts')
    box(13.08,6.9,2.72,.95,'CEM planner\nscore → elites → action')
    arrow((10.84,7.38),(11.1,7.38)); arrow((12.74,7.38),(13.02,7.38))
    box(13.57,8.14,1.75,.38,'Encoded goal',size=9)
    arrow((14.44,8.12),(14.44,7.91))
    box(.1,5.05,2.8,.87,'CEM proposal batch\n300 action trajectories')
    box(3.4,5.05,3.2,.87,'Action-conditioning pathway\nconditions all predictor blocks')
    arrow((2.96,5.48),(3.35,5.48))
    # Dashed line shows the specific edited conditioning connection at B3.
    arrow((6.65,5.48),(8.52,6.12),TEAL,'--')
    box(8.25,6.14,.55,.42,'A',color='#dceef0',edge=TEAL,size=12)
    arrow((8.52,6.59),(8.52,6.95),TEAL,'--')
    box(11.95,5.05,3.85,.87,'Execute selected actions\nin the simulator')
    arrow((14.44,6.85),(14.44,5.97))
    ax.text(8.1,4.53,'V: visual input     A: B3 conditioning     R: B3 output',
            ha='center',fontsize=12,weight='bold',color=INK)
    ax.text(.1,3.99,'Eight paired arms',fontsize=15,weight='bold',color=INK)
    rows=[
        ['Native','—','—','—'],
        ['Refined four-direction','learned','—','—'],
        ['Calibrated random subspace','random','—','—'],
        ['Equal-budget coupling','—','scaled','scaled'],
        ['Dose-matched random directions','—','random','random'],
        ['Unscaled joint','—','original','original'],
        ['Visual only','—','original','—'],
        ['Action-conditioning only','—','—','original'],
    ]
    table=ax.table(cellText=rows,colLabels=['Arm','R','V','A'],cellLoc='center',
                   colWidths=[.46,.13,.13,.13],bbox=[.006,.025,.66,.36])
    table.auto_set_font_size(False); table.set_fontsize(10)
    for (row,col),cell in table.get_celld().items():
        cell.set_edgecolor('#d3dde9')
        cell.set_facecolor('#e8eef6' if row==0 else ('#f6f8fb' if row%2==0 else 'white'))
        if row==0: cell.set_text_props(weight='bold',color=INK)
        if col==0: cell.set_text_props(ha='left')
    ax.text(11.05,3.45,'Controlled comparisons',fontsize=12,weight='bold',color=INK)
    ax.text(11.05,2.99,'R: learned basis vs calibrated\nrandom subspace.\n\nV + A: joint vs drop-one arms;\nequal-budget vs random directions.\n\nRefined R is not combined with V/A\nin this eight-arm panel.',
            va='top',fontsize=11,linespacing=1.45,color='#526174')
    save(fig,'ablation_architecture')


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
    overview(); architecture(); outcomes(report, audit); basis()
    print(f'Wrote four PNG/SVG figures to {OUT}')
