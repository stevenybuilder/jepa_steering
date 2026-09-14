"""Figures for the complete 200-scenario replication; never plot partial runs."""
import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from build_publication_figures import ROOT, INK, MUTED, TEAL, AMBER, save, sha


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


TASKS = [('reach', 'Reach'), ('reach-wall', 'Reach-Wall')]


def academic_style():
    """Size figures for a full-width paper column, with ordinary panel labels."""
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'Times New Roman', 'Times'],
        'mathtext.fontset': 'stix', 'font.size': 9,
        'axes.labelsize': 8.5, 'axes.titlesize': 9, 'xtick.labelsize': 8,
        'ytick.labelsize': 8, 'legend.fontsize': 8, 'pdf.fonttype': 42,
        'svg.fonttype': 'none', 'svg.hashsalt': 'jepa-lcfm-figures-v3',
        'axes.labelcolor': INK, 'text.color': INK, 'axes.titlecolor': INK})


def axes_style(ax, grid='y'):
    ax.spines[['top', 'right']].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#8b9195')
        ax.spines[side].set_linewidth(.6)
    ax.tick_params(length=3, width=.6, pad=3, colors=INK)
    ax.grid(axis=grid, color='#e5e7e8', lw=.5)
    ax.set_axisbelow(True)


def decimal(value):
    """Fixed decimals; preserve the sign of small nonzero values without -0.000."""
    if not np.isfinite(value):
        raise ValueError('Nonfinite heatmap value')
    places = 4 if 0 < abs(value) < .001 else 3
    while value != 0 and float(f'{value:.{places}f}') == 0:
        places += 1
        if places > 12:
            raise ValueError('Value requires an explicit smaller-unit label')
    return f'{value:.{places}f}'.replace('-', '−')


def panel(ax, label, title):
    ax.set_title(f'{label}  {title}', loc='left', weight='bold', pad=10)


def render():
    report, table, provenance = load()
    academic_style()
    render_lead(report, table, provenance)
    render_choices(report, table, provenance)
    render_history(table, provenance)
    render_layers(table, provenance)


def render_choices(report, table, provenance):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    fig.subplots_adjust(left=.115, right=.965, bottom=.30, top=.73, wspace=.65)
    primary = {r['task']: r for r in report['primary']}
    for i, (task, _) in enumerate(TASKS):
        p = primary[task]
        if p['n'] != 100:
            raise ValueError('Primary sample differs')
        value = p['fraction'] * 100
        low, high = np.array(p['family_95_interval']) * 100
        axes[0].errorbar(value, 1-i, xerr=[[value-low], [high-value]],
                        fmt='o', color=AMBER, ms=5, capsize=3, lw=1.2)
        axes[0].annotate(f"{p['changed']}/{p['n']}", (value, 1-i),
                         xytext=(0, 9), textcoords='offset points', ha='center', fontsize=8)
        for bank, offset, marker, color in [('original', .13, 'o', AMBER),
                                           ('fresh', -.13, 's', TEAL)]:
            row = cell(table, task, bank, 'all_blocks_h3_donor', 'excess_reference_cost_percent')
            value = row['mean']
            axes[1].errorbar(value, 1-i+offset,
                xerr=[[value-row.marginal_95_low], [row.marginal_95_high-value]],
                fmt=marker, color=color, ms=4.5, capsize=3, lw=1.2)
            control = cell(table, task, bank, 'all_blocks_persistent_donor', 'excess_reference_cost_percent')
            if control['mean'] != 0:
                raise ValueError('Persistent all-block control must match the reference')
    for ax in axes:
        ax.set_yticks([1, 0], ['Reach', 'Reach-Wall'])
        ax.set_ylim(-.5, 1.55)
        axes_style(ax, 'x')
    panel(axes[0], '(a)', 'H6 selection change')
    axes[0].set(xlim=(0, 100), xticks=[0, 25, 50, 75, 100],
                xlabel='States with a different choice (%)')
    panel(axes[1], '(b)', 'H6 excess reference cost')
    axes[1].set_xlim(left=0)
    axes[1].set_xlabel('Mean excess cost (%)')
    axes[1].legend([Line2D([0], [0], marker=m, color=c, ls='none')
                    for m, c in [('o', AMBER), ('s', TEAL)]],
                   ['Original bank', 'Second bank'], loc='lower right',
                   frameon=False, bbox_to_anchor=(1, 1.34), ncol=2, columnspacing=.9)
    fig.text(.115, .09, 'H3-only patch at all six blocks · 100 states per task', fontsize=8)
    fig.text(.115, .015, '(a) Two-task corrected Wilson intervals; (b) marginal 95% bootstrap intervals.',
             fontsize=7.4, color=MUTED)
    save(fig, 'lcfm_replication_choices', provenance)


def render_history(table, provenance):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    fig.subplots_adjust(left=.095, right=.98, bottom=.28, top=.86, wspace=.20)
    for ax, (task, label) in zip(axes, TASKS):
        for bank, linestyle in [('original', '-'), ('fresh', '--')]:
            for arm, color, marker in [('all_blocks_h3_donor', AMBER, 'o'),
                                        ('all_blocks_persistent_donor', TEAL, 's')]:
                rows = [cell(table, task, bank, arm, 'donor_reconstruction', h) for h in (3, 4, 6)]
                if rows[0]['mean'] != 1 or (arm == 'all_blocks_persistent_donor'
                                          and any(r['mean'] != 1 for r in rows)):
                    raise ValueError('Expected all-block computation parity did not hold')
                ax.plot([3, 4, 6], [r['mean'] for r in rows], color=color,
                        ls=linestyle, marker=marker, ms=3.5, lw=1.3)
                ax.fill_between([3, 4, 6], [r.marginal_95_low for r in rows],
                                [r.marginal_95_high for r in rows], color=color, alpha=.13, lw=0)
        panel(ax, '(a)' if task == 'reach' else '(b)', label)
        ax.set(xticks=[3, 4, 6], xticklabels=['H3', 'H4', 'H6'], xlabel='Forecast endpoint',
               ylim=(0, 1.08), yticks=[0, .25, .5, .75, 1])
        axes_style(ax)
    axes[0].set_ylabel('Forecast reconstruction, R')
    handles = [Line2D([0], [0], color=AMBER, marker='o', lw=1.3),
               Line2D([0], [0], color=TEAL, marker='s', lw=1.3),
               Line2D([0], [0], color=INK, lw=1.3),
               Line2D([0], [0], color=INK, lw=1.3, ls='--')]
    fig.legend(handles, ['H3 only', 'H3 + H4', 'Original bank', 'Second bank'],
               loc='lower center', bbox_to_anchor=(.54, .065), ncol=4, frameon=False,
               columnspacing=1.3, handlelength=1.8)
    fig.text(.095, .015, 'All six blocks · 100 states per task · marginal 95% bootstrap bands',
             fontsize=7.5, color=MUTED)
    save(fig, 'lcfm_replication_history', provenance)


def render_layers(table, provenance):
    modes = [('donor_h3', 'H3 only'), ('donor_persistent', 'H3 + H4'),
             ('random_range', 'Random: input range'), ('random_off_range', 'Random: orthogonal'),
             ('random_isotropic', 'Random: isotropic')]
    arrays = [np.array([[cell(table, task, bank, f'{mode}_B{layer}',
        'donor_reconstruction', 6)['mean'] for layer in range(6)] for mode, _ in modes])
        for task, _ in TASKS for bank in ('original', 'fresh')]
    limit = max(1., max(float(np.max(np.abs(a))) for a in arrays))
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.35))
    fig.subplots_adjust(left=.20, right=.98, top=.85, bottom=.27, hspace=.68)
    for i, ax in enumerate(axes):
        values = np.concatenate(arrays[2*i:2*i+2], axis=1)
        picture = ax.imshow(values, vmin=-limit, vmax=limit, cmap='RdBu', aspect='auto')
        ax.set_xticks(range(12), [f'B{j}' for j in range(6)] * 2)
        ax.set_yticks(range(5), [label for _, label in modes])
        ax.tick_params(length=0, labelsize=7.5, pad=4)
        ax.set_title(f'({chr(97+i)})  {TASKS[i][1]}', loc='left', weight='bold', pad=25)
        for x, bank in ((.25, 'Original bank'), (.75, 'Second bank')):
            ax.text(x, 1.035, bank, ha='center', va='bottom', transform=ax.transAxes, fontsize=8)
        for row, col in np.ndindex(values.shape):
            ax.text(col, row, decimal(values[row, col]), ha='center', va='center',
                    fontsize=7.1, color='white' if abs(values[row, col])/limit > .6 else INK)
        ax.axvline(5.5, color='white', lw=2)
        ax.axhline(1.5, color='white', lw=1.3)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cax = fig.add_axes([.47, .13, .34, .022])
    bar = fig.colorbar(picture, cax=cax, orientation='horizontal', ticks=[-1, -.5, 0, .5, 1])
    bar.ax.tick_params(labelsize=7.5, length=2)
    fig.text(.20, .141, 'Forecast reconstruction, R', fontsize=8, va='center')
    fig.text(.20, .044, 'H6 · one block patched · 100 states per task', fontsize=8)
    fig.text(.20, .008, 'R = 1: exact match; R = 0: unmodified forecast discrepancy.', fontsize=7.5, color=MUTED)
    save(fig, 'lcfm_replication_layers', dict(provenance, plotted_mean_cells=120,
        label_format='fixed decimals, >=3 places; extra precision retains small negative signs'))


def render_lead(report, table, provenance):
    """Headline figure: mechanism, forecast drift, decision effect, layer ablation, ranking."""
    REF, ONE = '#1a1a1a', '#a3261c'
    fig = plt.figure(figsize=(7.2, 6.3))
    timeline = fig.add_axes([.055, .745, .92, .23])
    timeline.set(xlim=(0, 1), ylim=(0, 1)); timeline.axis('off')
    timeline.text(0, .99, '(a)  One action is read twice in the rolling context', weight='bold', fontsize=9.5)
    for x, label in ((.40, 'H3: first read'), (.65, 'H4: second read (history)'), (.895, 'H5, H6')):
        timeline.text(x, .78, label, ha='center', fontsize=8.5, color=MUTED)
    rows = [('Changed input (reference)', .58, True),
            ('Patch at H3 only', .34, False),
            ('Patch at H3 and H4', .10, True)]
    for label, y, persists in rows:
        timeline.text(0, y, label, va='center', fontsize=9)
        for x in (.40, .65):
            timeline.add_patch(Rectangle((x-.085, y-.09), .17, .18,
                facecolor='white', edgecolor=REF, lw=.6))
        timeline.text(.358, y, '$a_2$', ha='center', va='center', fontsize=10.5)
        timeline.text(.442, y, '$a_3^*$', ha='center', va='center', fontsize=10.5)
        timeline.text(.608, y, '$a_3^*$' if persists else '$a_3$', ha='center', va='center',
                      fontsize=10.5, color=REF if persists else ONE)
        timeline.text(.692, y, '$a_4$', ha='center', va='center', fontsize=10.5)
        for x0, x1 in ((.49, .56), (.74, .81)):
            timeline.annotate('', (x1, y), (x0, y), arrowprops=dict(arrowstyle='->', color=REF, lw=.6))
        timeline.text(.895, y, '= reference' if persists else '$\\neq$ reference', ha='center',
                      va='center', fontsize=9, color=REF if persists else ONE)
    # row 2: forecast drift, selection change, excess cost
    left = fig.add_axes([.085, .445, .30, .22])
    right = fig.add_axes([.535, .445, .19, .22])
    cost = fig.add_axes([.775, .445, .19, .22])
    for task, marker, linestyle in [('reach', 'o', '-'), ('reach-wall', 's', '--')]:
        rows = [cell(table, task, 'original', 'all_blocks_h3_donor', 'donor_reconstruction', h)
                for h in (3, 4, 6)]
        left.plot([3, 4, 6], [1-r['mean'] for r in rows], color=ONE,
                  marker=marker, ms=3.6, ls=linestyle, lw=1.1, mfc='white' if marker == 's' else ONE)
        left.fill_between([3, 4, 6], [1-r.marginal_95_high for r in rows],
                          [1-r.marginal_95_low for r in rows], color=ONE, alpha=.12, lw=0)
        for h in (3, 4, 6):
            if cell(table, task, 'original', 'all_blocks_persistent_donor',
                    'donor_reconstruction', h)['mean'] != 1:
                raise ValueError('The persistent-patch parity control failed')
    left.plot([3, 4, 6], [0, 0, 0], color=REF, lw=1.2)
    panel(left, '(b)', 'Forecast drift after the second read')
    left.set(ylim=(-.035, .65), xlim=(2.9, 6.1), xticks=[3, 4, 6],
             xticklabels=['H3', 'H4', 'H6'], yticks=[0, .25, .5],
             ylabel='Relative discrepancy $E$', xlabel='Forecast step')
    axes_style(left)
    primary = {r['task']: r for r in report['primary']}
    for y, (task, _) in zip([1, 0], TASKS):
        p = primary[task]; value = p['fraction'] * 100
        low, high = np.array(p['family_95_interval']) * 100
        right.errorbar(value, y, xerr=[[value-low], [high-value]], fmt='o',
                       color=REF, ms=4.5, lw=1, capsize=2.5)
        right.annotate(f"{p['changed']}/{p['n']}", (value, y), xytext=(0, 8),
                       textcoords='offset points', ha='center', fontsize=8.5)
        row = cell(table, task, 'original', 'all_blocks_h3_donor', 'excess_reference_cost_percent')
        control = cell(table, task, 'original', 'all_blocks_persistent_donor', 'excess_reference_cost_percent')
        if control['mean'] != 0:
            raise ValueError('Persistent all-block control must match the reference')
        cost.errorbar(row['mean'], y, xerr=[[row['mean']-row.marginal_95_low],
                      [row.marginal_95_high-row['mean']]], fmt='o', color=REF, ms=4.5, lw=1, capsize=2.5)
        cost.annotate(f"{row['mean']:.2f}%", (row['mean'], y), xytext=(0, 8),
                      textcoords='offset points', ha='center', fontsize=8.5)
    panel(right, '(c)', 'The drift changes the planner\'s choice')
    right.set(xlim=(0, 100), ylim=(-.55, 1.55), xticks=[0, 25, 50, 75, 100],
              yticks=[1, 0], yticklabels=['Reach', 'Reach-Wall'],
              xlabel='States with a different choice (%)')
    axes_style(right, 'x')
    cost.set(xlim=(0, 3), ylim=(-.55, 1.55), xticks=[0, 1, 2, 3], yticks=[1, 0], yticklabels=[],
             xlabel='Mean excess cost (%)')
    axes_style(cost, 'x')
    # row 3: single-block ablation and ranking agreement
    layer = fig.add_axes([.085, .10, .42, .22])
    rank = fig.add_axes([.62, .10, .345, .22])
    blocks = np.arange(6)
    for task, marker, linestyle in [('reach', 'o', '-'), ('reach-wall', 's', '--')]:
        for mode, color in (('donor_h3', ONE), ('donor_persistent', REF)):
            rows = [cell(table, task, 'original', f'{mode}_B{b}', 'donor_reconstruction', 6) for b in blocks]
            layer.errorbar(blocks, [r['mean'] for r in rows],
                yerr=[[r['mean']-r.marginal_95_low for r in rows], [r.marginal_95_high-r['mean'] for r in rows]],
                color=color, marker=marker, ms=3.6, ls=linestyle, lw=1.1, capsize=2,
                mfc='white' if marker == 's' else color)
        randoms = [cell(table, task, 'original', f'{fam}_B{b}', 'donor_reconstruction', 6)['mean']
                   for fam in ('random_range', 'random_off_range', 'random_isotropic') for b in blocks]
        if max(randoms) > 0:
            raise ValueError('Random controls should not exceed the unmodified forecast')
    layer.axhspan(-.02, 0, color=MUTED, alpha=.25, lw=0)
    layer.text(-.15, -.035, 'random controls (all three families)', fontsize=7.5, color=MUTED, ha='left', va='top')
    panel(layer, '(d)', 'Single-block patches: the second read matters most at B1')
    layer.set(xticks=blocks, xticklabels=[f'B{b}' for b in blocks], ylim=(-.1, 1.0), yticks=[0, .25, .5, .75, 1],
              ylabel='Reconstruction $R$ at H6', xlabel='Patched predictor block')
    axes_style(layer)
    for i, (task, label) in enumerate(TASKS):
        for j, (arm, color, hatch) in enumerate((('all_blocks_h3_donor', ONE, None), ('all_blocks_persistent_donor', REF, None))):
            sp = cell(table, task, 'original', arm, 'reference_spearman')
            ov = cell(table, task, 'original', arm, 'reference_top10_overlap')
            x = i*2.6 + j*0.55
            rank.bar(x, sp['mean'], width=.5, color=color, alpha=.9 if j else .75)
            rank.bar(x+1.25, ov['mean']/10, width=.5, color=color, alpha=.9 if j else .75)
            rank.text(x, sp['mean']+.02, f"{sp['mean']:.2f}", ha='center', fontsize=7.2)
            rank.text(x+1.25, ov['mean']/10+.02, f"{ov['mean']:.1f}/10", ha='center', fontsize=7.2)
    rank.set(ylim=(0, 1.28), yticks=[0, .5, 1], xticks=[.275, 1.525, 2.875, 4.125],
             xticklabels=['Spearman', 'Top-10\noverlap', 'Spearman', 'Top-10\noverlap'],
             ylabel='Agreement with reference')
    rank.tick_params(axis='x', labelsize=7.5)
    for x, label in ((.9, 'Reach'), (3.5, 'Reach-Wall')):
        rank.text(x, 1.17, label, ha='center', fontsize=8.5)
    panel(rank, '(e)', 'Whole ranking agrees; the top ten does not')
    axes_style(rank)
    handles = [Line2D([0], [0], color=ONE, lw=1.2),
               Line2D([0], [0], color=REF, lw=1.2),
               Line2D([0], [0], color=REF, marker='o', lw=0, ms=3.6),
               Line2D([0], [0], color=REF, marker='s', lw=0, ms=3.6, mfc='white')]
    fig.legend(handles, ['H3 only', 'H3 + H4 (= reference)', 'Reach', 'Reach-Wall'],
               loc='lower center', bbox_to_anchor=(.53, .005), ncol=4,
               frameon=False, handlelength=1.8, columnspacing=1.6)
    save(fig, 'lcfm_context_lifetime', dict(provenance, discrepancy_metric='E=1-R',
        selection_intervals='97.5% Wilson per task; two-task family coverage >=95%',
        panels='a schematic; b E by step; c selection and cost; d single-block R at H6; e Spearman and top-10 overlap'))


if __name__ == '__main__':
    render()
