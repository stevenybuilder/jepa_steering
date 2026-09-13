"""All-case H3 physical-prefix figures from complete, hash-bound CPU outputs."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_publication_figures import ROOT, INK, MUTED, TEAL, AMBER, BLUE, clean, save, sha, style

TASKS = ('reach', 'reach-wall')
ARMS = ('native', 'fixed_rank4', 'matched_random_fixed_rank4')
NAMES = ('Unsteered', 'Intervention (rank-four)', 'Calibrated random')
COLORS = (BLUE, TEAL, AMBER)
METRICS = ('terminal_ee_distance', 'actual_weighted_goal_cost', 'same_native_prefix_forecast_weighted_mse')
PROTOCOL_SHA = '8d80f98a557c79b6dedad27bb3ef63c2ede4979150ba8373c893da8882a232b3'


def validate_frames(forecasts, primary, summary):
    designs = [(forecasts, ['model_arm','plan_arm'], [(a,b) for a in ARMS for b in ARMS]),
               (primary, ['arm','metric'], [(a,m) for a in ARMS[1:] for m in METRICS])]
    for frame, columns, crosses in designs:
        keys = ['task','episode',*columns]
        expected = {(t,e,*cross) for t in TASKS for e in range(4,32) for cross in crosses}
        if frame.duplicated(keys).any() or set(frame[keys].itertuples(index=False,name=None)) != expected:
            raise ValueError('Require every registered56 case and full crossed grid')
    for key in ('predicted_goal_cost_weighted','actual_goal_cost_weighted'):
        values = forecasts[key].to_numpy()
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError('Invalid goal costs; no clipping or omission')
    if not np.isfinite(primary.effect).all():
        raise ValueError('Nonfinite primary effect')
    expected = {(t,a,m) for t in TASKS for a in ARMS[1:] for m in METRICS}
    keys = ['task','arm','metric']
    if summary.duplicated(keys).any() or set(summary[keys].itertuples(index=False,name=None)) != expected:
        raise ValueError('All twelve registered primary contrasts required')
    for row in summary.itertuples(index=False):
        values = primary[(primary.task==row.task)&(primary.arm==row.arm)&(primary.metric==row.metric)].effect.to_numpy()
        if row.n != 28 or not np.isclose(row.mean,values.mean(),rtol=1e-10,atol=1e-12):
            raise ValueError('Primary mean or sample count changed')
        if row.bonferroni_family != 12 or not np.isfinite([row.bonferroni_95_low,row.bonferroni_95_high]).all():
            raise ValueError('Registered family12 interval missing')
        if row.bonferroni_95_low > row.bonferroni_95_high:
            raise ValueError('Reversed primary interval')


def load(root=ROOT):
    root = Path(root); path = root/'paper/data/planned_prefix_summary.json'
    receipt = json.loads(path.read_text())
    required = dict(status='complete56_development_physical_prefix_analysis',cases=56,n_per_task=28,
        all_full_raw_preservation_verified=True,physical_outcomes_measured=True,
        full_task_success_measured=False,fresh_confirmation=False,forecast_horizon_scored=3,
        CEM_optimized_horizon=6,protocol_sha256=PROTOCOL_SHA)
    if any(receipt.get(k)!=v for k,v in required.items()):
        raise ValueError('Complete source-bound physical56 analysis required')
    for name, info in receipt['outputs'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha(root/name)!=info['sha256']:
            raise ValueError('Public physical table identity changed')
    names = ('forecast_cases','primary_cases','primary_summary')
    for name in names:
        if f'paper/data/planned_prefix_{name}.csv' not in receipt['outputs']:
            raise ValueError('Required plot table is not hash-bound')
    frames = [pd.read_csv(root/f'paper/data/planned_prefix_{name}.csv') for name in names]
    validate_frames(*frames)
    return receipt,frames,sha(path)


def forecast_figure(frame,meta):
    fig, axes = plt.subplots(3,2,figsize=(7.8,8.2),sharex='col',sharey='col')
    fig.subplots_adjust(left=.12,right=.98,top=.82,bottom=.13,hspace=.25,wspace=.20)
    fig.text(.055,.955,'Predicted endpoint versus the endpoint reached',fontsize=17,weight='bold',color=INK)
    fig.text(.055,.911,'Same three plans, evaluated by each model and executed after identical resets.',fontsize=10.5,color=MUTED)
    for col,task in enumerate(TASKS):
        block = frame[frame.task==task]
        maximum = max(block.actual_goal_cost_weighted.max(),block.predicted_goal_cost_weighted.max())
        for row,model in enumerate(ARMS):
            ax = axes[row,col]
            ax.plot([0,maximum*1.04],[0,maximum*1.04],ls='--',lw=.8,color=MUTED,zorder=0)
            for plan,color,label in zip(ARMS,COLORS,NAMES):
                points = block[(block.model_arm==model)&(block.plan_arm==plan)]
                ax.scatter(points.actual_goal_cost_weighted,points.predicted_goal_cost_weighted,
                    s=18,alpha=.62,color=color,label=label+' plan',edgecolors='none')
            ax.set_xlim(0,maximum*1.04); ax.set_ylim(0,maximum*1.04); clean(ax)
            model_label = ('Unsteered','Intervention','Random')[row]
            ax.set_title(('Reach' if task=='reach' else 'Reach-Wall')+' / '+model_label+' model',loc='left',fontsize=10.5)
            if col==0:ax.set_ylabel('Predicted H3 goal cost')
            if row==2:ax.set_xlabel('Actual encoded H3 goal cost')
    handles,labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper left',bbox_to_anchor=(.045,.885),ncol=3,frameon=False,fontsize=10)
    fig.text(.055,.064,'All 56 development states × 3 models × 3 plans. Dashed: equal predicted/actual cost.',fontsize=10,color=INK)
    fig.text(.055,.027,'Visual MSE + 0.1 proprio MSE. H3 is the executed prefix; CEM optimized H6.',fontsize=10,color=MUTED)
    save(fig,'planned_prefix_forecasts',meta|dict(all_crossed_points=504,uncertainty='none; every registered point shown'))


def effects_figure(primary,summary,meta):
    fig, axes = plt.subplots(3,2,figsize=(7.8,8.2),sharey='row')
    fig.subplots_adjust(left=.145,right=.98,top=.84,bottom=.135,hspace=.36,wspace=.20)
    fig.text(.055,.955,'Do changed plans improve their first physical prefix?',fontsize=16,weight='bold',color=INK)
    fig.text(.055,.908,'Every paired state; diamonds and intervals report the registered aggregate.',fontsize=10.5,color=MUTED)
    labels = ('Terminal end-effector distance\nEdited − unsteered plan',
              'Actual encoded goal cost\nEdited − unsteered plan',
              'Same unsteered-prefix forecast MSE\nEdited − unsteered model')
    for row,metric in enumerate(METRICS):
        for col,task in enumerate(TASKS):
            ax = axes[row,col]; ax.axhline(0,color=MUTED,lw=.8,zorder=0)
            for i,(arm,color) in enumerate(zip(ARMS[1:],COLORS[1:])):
                values = primary[(primary.task==task)&(primary.metric==metric)&(primary.arm==arm)].sort_values('episode').effect.to_numpy()
                jitter = np.linspace(-.12,.12,28)
                ax.scatter(i+jitter,values,s=13,color=color,alpha=.58,edgecolors='none')
                s = summary[(summary.task==task)&(summary.metric==metric)&(summary.arm==arm)].iloc[0]
                ax.plot(i+.23,s['mean'],'D',color=INK,ms=4.5)
                ax.vlines(i+.23,s.bonferroni_95_low,s.bonferroni_95_high,color=INK,lw=1.3)
                ax.hlines([s.bonferroni_95_low,s.bonferroni_95_high],i+.18,i+.28,color=INK,lw=1)
            ax.set(xticks=[0,1],xticklabels=['Intervention','Random'],xlim=(-.3,1.43)); clean(ax)
            ax.set_title('Reach' if task=='reach' else 'Reach-Wall',loc='left',fontsize=12)
            if col==0:ax.set_ylabel(labels[row],fontsize=10.5)
    fig.text(.055,.066,'n=28 paired states/task. Intervals: registered simultaneous family12, 95%.',fontsize=10.5,color=INK)
    fig.text(.055,.028,'Lower is better for each endpoint; not full-task success or protected confirmation.',fontsize=10.5,color=MUTED)
    save(fig,'planned_prefix_effects',meta|dict(primary_cells=12,uncertainty='paired-context family12 Bonferroni95 intervals'))


if __name__ == '__main__':
    receipt,(forecasts,primary,summary),receipt_sha = load(); style()
    meta = dict(physical_plot_source_sha256=sha(__file__),receipt_sha256=receipt_sha,
        source_tables=receipt['outputs'],cases=56,n_per_task=28,scored_horizon=3,optimized_horizon=6,
        full_task_success=False,protected_confirmation=False)
    forecast_figure(forecasts,meta); effects_figure(primary,summary,meta)
    print('PASS: all56 cases, all3×3 forecasts, all12 registered primary contrasts')
