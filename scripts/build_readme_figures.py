"""Generate README figures from checked aggregates. No model or GPU calls."""
import argparse
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


def save(fig, name, *, pdf=False):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{name}.png', dpi=180, facecolor='white', bbox_inches='tight')
    fig.savefig(OUT / f'{name}.svg', facecolor='white', bbox_inches='tight')
    svg = OUT / f'{name}.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    if pdf:
        fig.savefig(OUT / f'{name}.pdf', facecolor='white', bbox_inches='tight',
                    metadata={'CreationDate': None, 'ModDate': None})
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
    """Inference-only six-block checkpoint; colored sites are alternatives."""
    fig, ax = plt.subplots(figsize=(15, 6.3))
    ax.set(xlim=(0, 16), ylim=(0, 6.7))
    ax.axis('off')
    ink, muted, blue, green, amber = '#242a26', '#73776f', '#4e6fa5', '#24796c', '#b67930'

    def box(x, y, w, h, text, color='white', edge='#cdd3cc', size=10.5, zorder=3):
        ax.add_patch(FancyBboxPatch((x,y), w,h, boxstyle='round,pad=.035',
                                   facecolor=color, edgecolor=edge, linewidth=1.1, zorder=zorder))
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=size, color=ink, zorder=4)

    def arrow(start, end, color=muted, style='-'):
        ax.annotate('', end, start, zorder=2, arrowprops={'arrowstyle':'->','color':color,
                                               'lw':1.3,'linestyle':style})

    ax.text(.1,6.27,'Edit the forecast, then let the planner act',fontsize=23,weight='bold',color=ink)
    ax.text(.1,5.83,'JEPA-WM inference  ·  All base weights frozen  ·  The eight arms use alternative edits',
            fontsize=12,color=muted)
    box(.1,4.30,1.45,.70,'Image')
    box(1.94,4.30,1.80,.70,'DINO visual\nencoder')
    arrow((1.60,4.65),(1.89,4.65)); arrow((3.79,4.65),(4.29,4.65))
    box(4.34,4.43,.48,.44,'V',color='#e4f0ed',edge=green,size=12)
    arrow((4.87,4.65),(5.76,4.65))
    box(.1,3.12,1.45,.70,'Proprioception',size=9.7)
    box(1.94,3.12,1.80,.70,'State encoder')
    arrow((1.60,3.47),(1.89,3.47))
    # Proprioception is not changed by the visual-input edit.
    ax.plot([3.80,4.04,4.04,5.35,5.35],[3.47,3.47,4.05,4.05,4.65],color=muted,lw=1.3)
    arrow((5.35,4.65),(5.76,4.65))
    box(5.72,3.22,5.55,2.30,'',color='#f7f8f5',zorder=0)
    ax.text(8.49,5.26,'Six-block dynamics predictor',ha='center',fontsize=11,weight='bold',color=ink)
    xs=[5.95,6.70,7.45,8.20,9.48,10.23]
    for i,x in enumerate(xs):
        box(x,4.25,.58,.80,f'B{i}',color='#e8eef6' if i==3 else 'white',edge=blue if i==3 else '#cdd3cc')
        if i<5: arrow((x+.61,4.65),(xs[i+1]-.04,4.65))
    box(8.98,4.43,.39,.44,'R',color='#e8eef6',edge=blue,size=11)
    # The unedited action path conditions every block; A modifies B3 only.
    ax.plot([5.15,10.52],[3.50,3.50],color=muted,lw=1.3)
    for i,x in enumerate(xs):
        arrow((x+.29,3.50),(x+.29,4.19),amber if i==3 else muted)
    box(8.29,3.69,.40,.37,'A',color='#f8eddc',edge=amber,size=11)
    ax.text(8.49,3.01,'Hooks active only at H3 of the H6 imagined rollout',ha='center',fontsize=10,color=muted)
    box(11.66,4.25,1.54,.80,'Latent\nforecast')
    box(13.68,4.25,2.12,.80,'Encoded-goal\nL2 cost')
    arrow((11.30,4.65),(11.61,4.65)); arrow((13.25,4.65),(13.62,4.65))
    box(13.92,5.35,1.66,.38,'Goal encoders',size=9.5)
    arrow((14.75,5.31),(14.75,5.11))
    box(.10,1.76,2.47,.85,'300 candidate\naction sequences',color='#f7f8f5')
    box(3.04,1.76,2.05,.85,'Action encoder')
    arrow((2.62,2.18),(2.99,2.18))
    ax.plot([5.14,5.15],[2.18,3.50],color=muted,lw=1.3)
    ax.text(5.48,2.10,'Actions condition every block',fontsize=10.5,color=muted)
    box(11.05,1.76,2.21,.85,'CEM\nrefit to elite plans',color='#f7f8f5')
    box(13.72,1.76,2.08,.85,'Execute prefix\nObserve + replan')
    ax.plot([14.75,14.75,12.15],[4.18,3.45,3.45],color=muted,lw=1.3)
    arrow((12.15,3.45),(12.15,2.66))
    arrow((13.31,2.18),(13.67,2.18))
    # CEM resamples its proposals, not the frozen model weights.
    ax.plot([12.15,12.15,1.33],[1.71,1.21,1.21],color=muted,lw=1.3)
    arrow((1.33,1.21),(1.33,1.71))
    ax.text(7.0,1.29,'Refine the proposal distribution; repeat candidate scoring',ha='center',fontsize=10,color=muted)
    for x,letter,label,color in ((.1,'V','visual input',green),(4.25,'A','B3 action conditioning',amber),(9.40,'R','rank-four B3 output',blue)):
        ax.text(x,.57,letter,color=color,fontsize=14,weight='bold')
        ax.text(x+.30,.57,label,color=ink,fontsize=11)
    ax.text(.1,.08,'Colored markers locate possible edits, not a joint R + V + A arm. The separate ablation table defines each intervention.',fontsize=10,color=muted)
    save(fig,'ablation_architecture',pdf=True)


def ablation_table():
    """One standalone table, with no result-dependent arm selection."""
    fig, ax = plt.subplots(figsize=(12, 5.1))
    ax.axis('off')
    ax.text(.01,1.00,'Eight arms, three intervention sites',transform=ax.transAxes,
            fontsize=21,weight='bold',color=INK)
    ax.text(.01,.91,'Within each task: the same frozen checkpoint and H3 hook timing; native has no edit.',
            transform=ax.transAxes,fontsize=11,color='#73776f')
    rows=[
        ['Native','—','—','—'],
        ['Refined four-direction','learned','—','—'],
        ['Calibrated random subspace','random','—','—'],
        ['Equal-budget coupling','—','scaled learned','scaled learned'],
        ['Dose-matched random directions','—','scaled random','scaled random'],
        ['Unscaled joint','—','original dose','original dose'],
        ['Visual only','—','original dose','—'],
        ['Action-conditioning only','—','—','original dose'],
    ]
    table=ax.table(cellText=rows,colLabels=['Arm','R · B3 output','V · visual input','A · B3 condition'],cellLoc='center',
                   colWidths=[.40,.20,.20,.20],bbox=[.01,.14,.98,.71])
    table.auto_set_font_size(False); table.set_fontsize(11)
    for (row,col),cell in table.get_celld().items():
        cell.set_edgecolor('#dedfd6')
        cell.set_linewidth(.7)
        cell.set_facecolor('#edf0ea' if row==0 else ('#f8f9f6' if row%2==0 else 'white'))
        if row==0: cell.set_text_props(weight='bold',color=INK)
        if col==0: cell.set_text_props(ha='left')
    ax.text(.01,.055,'Scaled: each V/A component uses 1/√2 of its original standardized dose. R arms never include V or A.',
            transform=ax.transAxes,fontsize=10,color='#73776f')
    ax.text(.01,.005,'Compare learned R with its calibrated random subspace; compare learned V/A with dose-matched random directions.',
            transform=ax.transAxes,fontsize=10,color='#73776f')
    save(fig,'ablation_table',pdf=True)


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', nargs='+', choices=['overview', 'architecture', 'ablation', 'outcomes', 'basis'])
    args = parser.parse_args()
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'svg.fonttype':'none'})
    report, audit = check()
    selected = args.only or ['overview', 'architecture', 'ablation', 'outcomes', 'basis']
    for name in selected:
        {'overview': overview, 'architecture': architecture, 'ablation': ablation_table,
         'outcomes': lambda: outcomes(report, audit), 'basis': basis}[name]()
    print(f'Wrote {len(selected)} PNG/SVG figures to {OUT}')
