"""LCFM history figure from the complete registered context-counterfactual table.

B1 is an explicitly post hoc illustration; all six layers remain in the paper.
No model executions, fits, or new hypothesis tests.
"""
import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from build_action_history_figure import load
from build_publication_figures import ROOT, INK, MUTED, TEAL, AMBER, clean, save, style, sha


def render():
    _, provenance = load()  # Receipt hash, complete16 cohort and all-block checks.
    table = pd.read_csv(ROOT/'paper/data/action_counterfactual_summary.csv')
    arms = ('coherent_raw_h3', 'all_blocks_h3_donor', 'all_blocks_persistent_donor',
            'donor_h3_B1', 'donor_persistent_B1')
    rows = table[(table.modality == 'official') & (table.metric == 'donor_reconstruction') & table.arm.isin(arms)].copy()
    keys = ['task', 'bank', 'arm', 'horizon']
    expected = {(t,b,a,h) for t in ('reach','reach-wall') for b in ('original','fresh') for a in arms for h in (3,4,6)}
    if len(rows) != 60 or rows.duplicated(keys).any() or set(rows[keys].itertuples(index=False,name=None)) != expected:
        raise ValueError('Require all 60 task/bank/arm/endpoint cells')
    if not (rows.n == 8).all() or not (rows.n_defined == 8).all() or not np.isfinite(rows[['mean','marginal_95_low','marginal_95_high']]).all().all():
        raise ValueError('Incomplete or nonfinite source data')
    if not (rows.loc[rows.arm.isin(arms[:1]+arms[2:3]),'mean'] == 1).all():
        raise ValueError('Raw/all-block persistent reference mismatch')
    style()
    fig, axes = plt.subplots(2,2,figsize=(7.5,5.4),sharex=True,sharey=True)
    fig.subplots_adjust(left=.115,right=.975,top=.78,bottom=.17,hspace=.5,wspace=.2)
    fig.text(.055,.955,'An action change has to survive its own history.',fontsize=16,weight='bold',color=INK)
    fig.text(.055,.905,'Replace it at H3, then replace its historical occurrence at H4.',fontsize=10.5,color=MUTED)
    for row,pair in enumerate((arms[1:3],arms[3:5])):
        for col,(task,label) in enumerate((('reach','Reach'),('reach-wall','Reach-Wall'))):
            ax=axes[row,col]
            ax.axhline(1,color=INK,lw=1,ls=':',zorder=1)
            ax.axvline(4,color='#bcc8cc',lw=.8,ls=':')
            for arm,color in zip(pair,(AMBER,TEAL)):
                for bank,ls in (('original','-'),('fresh','--')):
                    d=rows[(rows.task==task)&(rows.arm==arm)&(rows.bank==bank)].sort_values('horizon')
                    ax.plot(d.horizon,d['mean'],color=color,ls=ls,marker='o',ms=3,lw=1.8)
                    ax.fill_between(d.horizon,d.marginal_95_low,d.marginal_95_high,color=color,alpha=.14,lw=0)
            ax.set_title(label+' · '+('all six blocks' if row==0 else 'B1 only'),loc='left',fontsize=10,weight='bold')
            ax.set(xticks=[3,4,6],xticklabels=['H3','H4','H6'],xlim=(2.9,6.1),ylim=(-.035,1.07),yticks=[0,.5,1])
            clean(ax)
    fig.text(.015,.49,'Reconstruction of the changed-action forecast, R',rotation=90,va='center',fontsize=10,color=INK)
    axes[1,0].set_xlabel('Forecast endpoint');axes[1,1].set_xlabel('Forecast endpoint')
    fig.legend([Line2D([0],[0],color=c,lw=2,ls=s) for c,s in ((AMBER,'-'),(TEAL,'-'),(INK,':'))],
               ['Patch H3 only','Patch H3 and H4','Raw-action reference (R = 1)'],loc='upper left',bbox_to_anchor=(.07,.878),ncol=3,frameon=False,fontsize=9)
    fig.text(.055,.075,'B1 is a post hoc illustration · solid / dashed: original / fresh banks · marginal 95% intervals',fontsize=9,color=MUTED)
    fig.text(.055,.025,'R = 1: matches the counterfactual; R = 0: native forecast. No physical outcome measured.',fontsize=9,color=MUTED)
    provenance.update(plot_source_sha256=sha(__file__), plotted_cells=60,
                      layer_selection='B1 post hoc illustration; all six layers retained in appendix',
                      intervals='existing marginal 95% context bootstrap; no new tests',new_model_execution=False)
    save(fig,'lcfm_context_history',provenance)
    print('PASS: all 60 source-bound history cells; no new model execution')

if __name__=='__main__': render()
