"""Complete H6 action-history ranking map from verified existing-data summaries."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from build_publication_figures import ROOT,INK,MUTED,TEAL,AMBER,clean,save,sha,style


def render():
    report_path=ROOT/'paper/data/lcfm_history_ranking_report.json';report=json.loads(report_path.read_text())
    if report['status']!='complete_post_hoc_existing_data_ranking_analysis' or report['n_per_task']!=8 or report['context_banks']!=32 or report['new_model_calls']:
        raise ValueError('Require the complete frozen existing-data analysis')
    path=ROOT/'paper/data/lcfm_history_ranking_summary.csv'
    if sha(path)!=report['outputs'][path.name]['sha256']:raise ValueError('Summary checksum mismatch')
    data=pd.read_csv(path);sites=['all_blocks',*[f'B{i}' for i in range(6)]]
    modes=['h3_vs_counterfactual','persistent_vs_counterfactual'];metrics=['spearman','elite_overlap','winner_agreement']
    d=data[data.comparison.isin(modes)&data.metric.isin(metrics)]
    keys=['task','bank','site','comparison','metric']
    expected={(t,b,s,c,m) for t in report['tasks'] for b in ['original','fresh'] for s in sites for c in modes for m in metrics}
    if len(d)!=168 or d.duplicated(keys).any() or set(d[keys].itertuples(index=False,name=None))!=expected or not (d.n_defined==8).all():raise ValueError('Require all168 defined mean cells')
    style();fig,axes=plt.subplots(2,3,figsize=(9.3,6.7),sharex=True)
    fig.subplots_adjust(left=.085,right=.975,bottom=.22,top=.735,wspace=.34,hspace=.65)
    fig.text(.045,.945,'Action history changes which candidates the model prefers.',fontsize=16,weight='bold',color=INK)
    fig.text(.045,.888,'Agreement with the coherent input-action change at H6 · all blocks and all six single-layer sites',fontsize=10,color=MUTED)
    for row,(task,label) in enumerate((('reach','Reach'),('reach-wall','Reach-Wall'))):
        for col,(metric,title) in enumerate(zip(metrics,['Rank agreement (Spearman ρ)','Shared top-10 candidates','Winner agreement'])):
            ax=axes[row,col]
            for mode,color in zip(modes,(AMBER,TEAL)):
                for bank,ls in [('original','-'),('fresh','--')]:
                    subset=d[(d.task==task)&(d.bank==bank)&(d.comparison==mode)&(d.metric==metric)].set_index('site').loc[sites]
                    ax.plot(range(1,7),subset['mean'].iloc[1:],color=color,ls=ls,marker='o',ms=3,lw=1.6)
                    ax.plot([0],[subset['mean'].iloc[0]],color=color,marker='s' if bank=='original' else 'o',ms=4,linestyle='none')
            ax.set_title(label+'\n'+title,fontsize=10,loc='left',weight='bold',color=INK)
            ax.axvline(.5,color='#cbd4d9',lw=.8)
            ax.set(xticks=range(7),xticklabels=['All\nblocks',*[f'B{i}' for i in range(6)]],xlim=(-.25,6.25))
            if metric=='elite_overlap':ax.set(ylim=(-.2,10.3),yticks=[0,5,10])
            else:ax.set(ylim=(-.02,1.03),yticks=[0,.5,1])
            if row==1:ax.set_xlabel('Patched blocks',fontsize=10)
            clean(ax)
    fig.legend([Line2D([0],[0],color=c,lw=2) for c in (AMBER,TEAL)],['Patch H3 only','Patch both H3 and H4 appearances'],ncol=2,frameon=False,fontsize=10,loc='upper left',bbox_to_anchor=(.035,.85))
    fig.text(.045,.083,'Original: solid / square · fresh: dashed / circle · eight contexts per task · descriptive means',fontsize=9.5,color=MUTED)
    fig.text(.045,.034,'All-block persistent matching is the exact control. Single-block persistence need not preserve the same winner.',fontsize=9.5,color=MUTED)
    save(fig,'lcfm_history_ranking',dict(source_report_sha256=sha(report_path),source_table_sha256=sha(path),plot_source_sha256=sha(__file__),plotted_mean_cells=168,scope='H6 only; fixed banks; no physical outcomes; all sites/tasks/banks retained'))
    print('PASS: all168 ranking/elite/winner means displayed')

if __name__=='__main__':render()
