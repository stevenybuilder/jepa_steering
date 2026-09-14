"""Figures for the complete 200-scenario replication; never plot partial runs."""
import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from build_publication_figures import ROOT, INK, MUTED, TEAL, AMBER, clean, save, sha, style


def load():
    path = ROOT/'paper/data/lcfm_replication_summary.json'
    report = json.loads(path.read_text())
    if (report['role'] != 'complete_fresh_action_history_replication' or report['independent_scenarios'] != 200
            or report['per_task'] != 100 or report['h6_forecasts'] != 14000 or not report['all_cloud_verified']
            or report['old_16_pooled'] or report['physical_outcomes_measured']):
        raise ValueError('Require the complete, independently audited replication')
    table_path = ROOT/'paper/data/lcfm_replication_secondary.csv'
    if sha(table_path) != report['outputs'][table_path.name]:
        raise ValueError('Summary table differs from the analysis receipt')
    table = pd.read_csv(table_path)
    if not (table.n == 100).all():
        raise ValueError('Incorrect statistical unit or sample count')
    provenance = dict(source_summary_sha256=sha(path), source_table_sha256=sha(table_path),
        plot_source_sha256=sha(__file__), independent_scenarios=200, no_new_model_calls=True)
    return report, table, provenance


def cell(table, task, bank, arm, metric, horizon=None):
    rows = table[(table.task == task) & (table.bank == bank) & (table.arm == arm)
                 & (table.modality == 'official') & (table.metric == metric)]
    if horizon is not None:
        rows = rows[rows.horizon == horizon]
    if len(rows) != 1:
        raise ValueError('Expected one complete registered metric cell')
    row = rows.iloc[0]
    if row.n_defined != 100 or not np.isfinite([row['mean'], row.marginal_95_low, row.marginal_95_high]).all():
        raise ValueError('Undefined result needs explicit figure treatment')
    return row


def render():
    report, table, provenance = load()
    style()
    render_lead(report, table, provenance)
    fig, axes = plt.subplots(1, 2, figsize=(8.7, 4.5))
    fig.subplots_adjust(left=.10, right=.96, bottom=.27, top=.70, wspace=.40)
    fig.text(.045, .94, 'What a one-time action patch changes at the planning horizon',
             fontsize=15, weight='bold', color=INK)
    fig.text(.045, .87, 'Reference: change the input action and predict its consequences.', fontsize=11, color=MUTED)
    tasks = [('reach', 'Reach'), ('reach-wall', 'Reach-Wall')]
    primary = {r['task']: r for r in report['primary']}
    for i, (task, label) in enumerate(tasks):
        p = primary[task]
        if p['n'] != 100:
            raise ValueError('Primary sample differs')
        interval = np.array(p['family_95_interval'])*100
        value = p['fraction']*100
        axes[0].errorbar(i, value, yerr=[[value-interval[0]], [interval[1]-value]],
                         fmt='o', color=AMBER, ms=7, capsize=4, lw=1.7)
        axes[0].annotate(f"{p['changed']}/100", (i, interval[1]), xytext=(0, 9),
                         textcoords='offset points', ha='center', color=INK, fontsize=10, weight='bold')
        for bank, offset, marker, color in [('original', -.09, 'o', AMBER), ('fresh', .09, 's', TEAL)]:
            row = cell(table, task, bank, 'all_blocks_h3_donor', 'excess_reference_cost_percent')
            value = row['mean']
            axes[1].errorbar(i+offset, value,
                yerr=[[value-row.marginal_95_low], [row.marginal_95_high-value]],
                fmt=marker, color=color, ms=6, capsize=4, lw=1.7)
            control = cell(table, task, bank, 'all_blocks_persistent_donor', 'excess_reference_cost_percent')
            if control['mean'] != 0:
                raise ValueError('Persistent all-block control must match the reference')
    for ax in axes:
        ax.set_xticks([0, 1], ['Reach', 'Reach-Wall'])
        ax.set_xlim(-.45, 1.45)
        clean(ax)
    axes[0].set_title('How often the choice changes', loc='left', fontsize=11, weight='bold', pad=14)
    axes[0].set_ylabel('Starting states (%)')
    axes[0].set_ylim(0, 110); axes[0].set_yticks([0, 25, 50, 75, 100])
    axes[1].set_title('How much that choice costs', loc='left', fontsize=11, weight='bold', pad=14)
    axes[1].set_ylabel('Mean extra cost under reference (%)')
    axes[1].set_ylim(bottom=0)
    axes[1].axhline(0, color=INK, ls=':', lw=1)
    axes[1].legend([Line2D([0], [0], marker=m, color=c, ls='none') for m, c in [('o', AMBER), ('s', TEAL)]],
                   ['Original bank', 'Second bank'], frameon=False, fontsize=9, loc='best')
    fig.text(.045, .13, '200 independent starting states · 100 per task · patch at all six blocks, at H3 only',
             fontsize=10, color=INK)
    fig.text(.045, .075, 'Left: 97.5% Wilson intervals (two-task correction). Right: 95% bootstrap intervals.',
             fontsize=9, color=MUTED)
    fig.text(.045, .025, 'Repeating the patch at H4 matches the reference exactly. These are model costs, not robot outcomes.',
             fontsize=9, color=MUTED)
    save(fig, 'lcfm_replication_choices', provenance)

    fig, axes = plt.subplots(1, 2, figsize=(8.7, 4.5), sharey=True)
    fig.subplots_adjust(left=.10, right=.97, bottom=.26, top=.72, wspace=.20)
    fig.text(.045, .94, 'A patch can match now and diverge one step later',
             fontsize=16, weight='bold', color=INK)
    fig.text(.045, .87, 'The same action enters the predictor at H3 and again as history at H4.',
             fontsize=11, color=MUTED)
    for ax, (task, label) in zip(axes, tasks):
        for bank, linestyle in [('original', '-'), ('fresh', '--')]:
            for arm, color in [('all_blocks_h3_donor', AMBER), ('all_blocks_persistent_donor', TEAL)]:
                rows = [cell(table, task, bank, arm, 'donor_reconstruction', h) for h in (3, 4, 6)]
                if rows[0]['mean'] != 1 or (arm == 'all_blocks_persistent_donor' and any(r['mean'] != 1 for r in rows)):
                    raise ValueError('Expected all-block computation parity did not hold')
                ax.plot([3, 4, 6], [r['mean'] for r in rows], color=color, ls=linestyle, lw=1.8)
                ax.fill_between([3, 4, 6], [r.marginal_95_low for r in rows],
                                [r.marginal_95_high for r in rows], color=color, alpha=.12)
        ax.set_title(label, loc='left', fontsize=12, weight='bold', pad=14)
        ax.set_xticks([3, 4, 6], ['H3', 'H4', 'H6'])
        ax.set_xlabel('Imagined step')
        ax.set_ylim(0, 1.07)
        ax.set_yticks([0, .25, .5, .75, 1])
        clean(ax)
    axes[0].set_ylabel('Forecast reconstruction, R')
    handles = [Line2D([0], [0], color=AMBER, lw=2), Line2D([0], [0], color=TEAL, lw=2),
               Line2D([0], [0], color=MUTED, lw=1.5), Line2D([0], [0], color=MUTED, lw=1.5, ls='--')]
    fig.legend(handles, ['Patch at H3', 'Patch at H3 and H4', 'Original bank', 'Second bank'],
               loc='lower center', bbox_to_anchor=(.51, .075), ncol=4, frameon=False, fontsize=9)
    fig.text(.045, .025, '100 independent states per task · all six blocks patched · marginal 95% bootstrap intervals',
             fontsize=9, color=MUTED)
    save(fig, 'lcfm_replication_history', provenance)

    # All six layers and every single-layer control family, without layer selection.
    modes = [('donor_h3', 'Patch once'), ('donor_persistent', 'Patch both times'),
             ('random_range', 'Random: input range'), ('random_off_range', 'Random: orthogonal'),
             ('random_isotropic', 'Random: isotropic')]
    arrays = []
    for task, _ in tasks:
        for bank in ('original', 'fresh'):
            arrays.append(np.array([[cell(table, task, bank, f'{mode}_B{layer}',
                'donor_reconstruction', 6)['mean'] for layer in range(6)] for mode, _ in modes]))
    limit = max(1., max(float(np.max(np.abs(a))) for a in arrays))
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.8))
    fig.subplots_adjust(left=.23, right=.86, top=.81, bottom=.14, wspace=.17, hspace=.36)
    fig.text(.045, .95, 'How much of the changed-action forecast each layer reproduces',
             fontsize=15, weight='bold', color=INK)
    fig.text(.045, .90, 'All six blocks and all five single-block intervention families · H6', fontsize=11, color=MUTED)
    for i, (ax, values) in enumerate(zip(axes.flat, arrays)):
        picture = ax.imshow(values, vmin=-limit, vmax=limit, cmap='RdBu', aspect='auto')
        ax.set_xticks(range(6), [f'B{j}' for j in range(6)])
        ax.set_yticks(range(5), [label for _, label in modes] if i % 2 == 0 else [])
        ax.tick_params(length=0, labelsize=9)
        ax.set_title(tasks[i//2][1]+' · '+('original bank' if i % 2 == 0 else 'second bank'),
                     loc='left', fontsize=11, weight='bold', pad=10)
        for row in range(5):
            for col in range(6):
                value = values[row, col]
                label = (f'{value:.0e}'.replace('e-0', 'e−').replace('-', '−')
                         if 0 < abs(value) < .005 else f'{value:.2f}')
                ax.text(col, row, label, ha='center', va='center', fontsize=8.5,
                        color='white' if abs(values[row, col])/limit > .6 else INK)
        for spine in ax.spines.values(): spine.set_visible(False)
    cax = fig.add_axes([.90, .21, .015, .53])
    fig.colorbar(picture, cax=cax, label='Forecast reconstruction, R')
    fig.text(.045, .075, 'R = 1: matches the changed-action forecast. R = 0: the unmodified forecast’s error. Negative values retained.',
             fontsize=9, color=MUTED)
    fig.text(.045, .025, 'Each cell averages 100 independent states. Both banks use the same states; all intervals are in the source table.',
             fontsize=9, color=MUTED)
    save(fig, 'lcfm_replication_layers', provenance)


def render_lead(report, table, provenance):
    """One conceptual panel and two measured panels; no illustrative data."""
    fig = plt.figure(figsize=(10, 7.6))
    fig.text(.045, .951, 'Correct at one step. A different plan three steps later.',
             fontsize=18, weight='bold', color=INK)
    fig.text(.045, .907, 'Frozen JEPA-WM · 200 independent starting states · two-frame context',
             fontsize=11, color=MUTED)
    ax = fig.add_axes([.045, .565, .92, .305])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.text(0, .99, 'A   The edited action is used again as history', color=INK, fontsize=12, weight='bold')
    ax.text(.405, .83, 'H3 · new action', ha='center', color=MUTED, fontsize=10)
    ax.text(.655, .83, 'H4 · same action, older position', ha='center', color=MUTED, fontsize=10)
    rows = [('Change the input', .63, TEAL, True),
            ('Patch at H3 only', .37, AMBER, False),
            ('Patch at H3 and H4', .11, TEAL, True)]
    def token(x, y, text, color=None):
        ax.add_patch(Rectangle((x-.037, y-.085), .074, .17,
                              facecolor=color or '#eef1f2', edgecolor='none'))
        ax.text(x, y, text, ha='center', va='center', fontsize=12,
                color='white' if color else INK)
    for label, y, color, persists in rows:
        ax.text(0, y, label, va='center', color=INK, fontsize=11)
        token(.36, y, '$a_2$'); token(.45, y, '$a_3^*$', TEAL)
        ax.annotate('', (.56, y), (.51, y), arrowprops=dict(arrowstyle='->', color=MUTED, lw=1))
        token(.61, y, '$a_3^*$' if persists else '$a_3$', color)
        token(.70, y, '$a_4$')
        ax.text(.78, y, 'Reference' if label == 'Change the input' else
                ('Exact match' if persists else 'Original action returns'),
                va='center', fontsize=10.5, color=color, weight='bold')

    left = fig.add_axes([.125, .18, .345, .275])
    right = fig.add_axes([.64, .18, .30, .275])
    fig.text(.045, .49, 'B   Forecasts separate after H3', fontsize=12, weight='bold', color=INK)
    fig.text(.56, .49, 'C   The selected candidate changes', fontsize=12, weight='bold', color=INK)
    for task, marker, linestyle in [('reach', 'o', '-'), ('reach-wall', 's', '--')]:
        rows = [cell(table, task, 'original', 'all_blocks_h3_donor', 'donor_reconstruction', h) for h in (3, 4, 6)]
        left.plot([3, 4, 6], [1-r['mean'] for r in rows], color=AMBER,
                  marker=marker, ms=5, ls=linestyle, lw=1.7)
        left.fill_between([3, 4, 6], [1-r.marginal_95_high for r in rows],
                          [1-r.marginal_95_low for r in rows], color=AMBER, alpha=.14)
        for h in (3, 4, 6):
            if cell(table, task, 'original', 'all_blocks_persistent_donor', 'donor_reconstruction', h)['mean'] != 1:
                raise ValueError('The persistent-patch parity control failed')
    left.plot([3, 4, 6], [0, 0, 0], color=TEAL, lw=2)
    left.text(4.16, .57, 'Patch once', color=AMBER, fontsize=10, weight='bold')
    left.text(4.12, .035, 'Patch both appearances', color=TEAL, fontsize=10, weight='bold')
    left.set_ylim(-.035, .65); left.set_xlim(2.9, 6.1)
    left.set_xticks([3, 4, 6], ['H3', 'H4', 'H6'])
    left.set_yticks([0, .25, .5], ['0 · exact', '0.25', '0.50'])
    left.set_ylabel('Forecast distance to reference\n(relative to unmodified forecast)', fontsize=9.5)
    left.set_xlabel('Imagined step', fontsize=10)
    clean(left)
    primary = {r['task']: r for r in report['primary']}
    for y, task, label in [(1, 'reach', 'Reach'), (0, 'reach-wall', 'Reach-Wall')]:
        p = primary[task]; value = p['fraction']*100
        low, high = np.array(p['family_95_interval'])*100
        right.errorbar(value, y, xerr=[[value-low], [high-value]], fmt='o',
                       color=AMBER, ms=8, lw=2, capsize=4)
        right.text(value, y+.24, f"{p['changed']}%", ha='center', fontsize=23, weight='bold', color=INK)
        cost = cell(table, task, 'original', 'all_blocks_h3_donor', 'excess_reference_cost_percent')['mean']
        right.text(0, y-.29, f'{cost:.2f}% mean extra reference cost', fontsize=9.5, color=MUTED)
    right.set_xlim(0, 100); right.set_ylim(-.5, 1.6)
    right.set_xticks([0, 25, 50, 75, 100]); right.set_yticks([1, 0], ['Reach', 'Reach-Wall'])
    right.set_xlabel('Starting states with a different H6 choice (%)', fontsize=9.5)
    clean(right); right.grid(False)
    fig.text(.105, .108, 'Reach: solid / circles   ·   Reach-Wall: dashed / squares', fontsize=9, color=MUTED)
    fig.text(.045, .061, 'Reference: the same model with one input action changed. Patching both appearances restores exact agreement.',
             fontsize=10, color=INK)
    fig.text(.045, .025, 'Original candidate bank · 100 states per task · B: 95% bootstrap bands · C: two-task corrected Wilson intervals · model costs',
             fontsize=8.5, color=MUTED)
    save(fig, 'lcfm_context_lifetime', provenance)


if __name__ == '__main__':
    render()
