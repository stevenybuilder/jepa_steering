"""Plot all 72 registered C contrasts from hash-bound, complete public outputs."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_SHA = 'e1ee17c7c2c6d35061aa3cdf77ecdb3d656c619c5dca142be42aa482bc122f51'
CONTRASTS = (
    ('persistence_minus_h3', 'Persistent − transient donor', 'Reconstruction ΔR'),
    ('h3_donor_minus_range_random', 'Donor − range-random', 'Reconstruction ΔR'),
    ('range_minus_offrange_rank_loss', 'Range − off-range random', 'Rank-loss difference Δ(1 − ρ)'),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(root=ROOT):
    data = root/'paper/data'; receipt_path = data/'action_counterfactual_summary.json'
    receipt = json.loads(receipt_path.read_text())
    if (receipt['status'] != 'complete16_action_counterfactual_development'
            or receipt['cohort'] != 16 or not receipt['all_cloud_verified']
            or receipt['analysis_source_sha256'] != ANALYSIS_SHA):
        raise ValueError('Complete frozen C analysis is required')
    path = data/'action_counterfactual_primary_contrasts.csv'
    if sha(path) != receipt['outputs'][path.name]['sha256']:
        raise ValueError('Primary contrast table changed')
    table = pd.read_csv(path)
    expected = {(task, contrast, bank, layer) for task in ('reach','reach-wall')
                for contrast,_,_ in CONTRASTS for bank in ('original','fresh') for layer in range(6)}
    keys = list(zip(table.task,table.contrast,table.bank,table.layer))
    if len(keys) != 72 or set(keys) != expected:
        raise ValueError('Missing, duplicate or unexpected primary contrast cells')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,
        'axes.labelsize':10.5,'axes.titlesize':11,'xtick.labelsize':10.5,
        'ytick.labelsize':10.5,'svg.fonttype':'none','pdf.fonttype':42})
    fig, axes = plt.subplots(3,2,figsize=(6.8,7.2),sharex=True,sharey='row')
    colors = {'original':'#247c9a','fresh':'#ba622b'}
    undefined = []
    for row,(contrast,title,ylabel) in enumerate(CONTRASTS):
        for col,task in enumerate(('reach','reach-wall')):
            ax = axes[row,col]; ax.axhline(0,color='#68717f',linewidth=.8,zorder=0)
            for bank,offset in (('original',-.09),('fresh',.09)):
                subset = table[(table.task==task)&(table.contrast==contrast)&(table.bank==bank)].sort_values('layer')
                x = subset.layer.to_numpy()+offset; y = subset['mean'].to_numpy()
                low = subset.simultaneous_72_primary_95_low.to_numpy()
                high = subset.simultaneous_72_primary_95_high.to_numpy()
                defined = np.isfinite(y)&np.isfinite(low)&np.isfinite(high)
                ax.plot(x,np.where(defined,y,np.nan),color=colors[bank],marker='o',
                    markersize=3.5,linewidth=1,label=bank.capitalize()+' bank')
                # Endpoints are drawn directly; no clipping or zero substitution.
                ax.vlines(x[defined],low[defined],high[defined],color=colors[bank],linewidth=1)
                for xx,ll,hh in zip(x[defined],low[defined],high[defined]):
                    ax.hlines([ll,hh],xx-.045,xx+.045,color=colors[bank],linewidth=.8)
                for layer in subset.layer.to_numpy()[~defined]:
                    undefined.append([task,contrast,bank,int(layer)])
            ax.set_title(('Reach' if task=='reach' else 'Reach-Wall')+'\n'+title,pad=7)
            ax.spines[['top','right']].set_visible(False)
            ax.grid(axis='y',alpha=.16)
            ax.set_xticks(range(6),[f'B{k}' for k in range(6)])
            if col==0:ax.set_ylabel(ylabel)
            if row==2:ax.set_xlabel('Predictor block')
    handles,labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1),ncol=2,frameon=False)
    footer = 'All 72 registered contrasts · simultaneous 95% intervals\nn = 8 paired development contexts/task · no physical outcome'
    if undefined:footer += f'\n{len(undefined)} undefined cells omitted, never plotted as zero'
    fig.text(.5,.012,footer,ha='center',fontsize=10.5)
    fig.tight_layout(rect=(0,.065,1,.95),h_pad=1.45,w_pad=1)
    out = root/'docs/figures'; out.mkdir(parents=True,exist_ok=True); outputs = {}
    for extension in ('png','svg','pdf'):
        path = out/f'paper_action_counterfactual.{extension}'
        fig.savefig(path,dpi=220,facecolor='white')
        outputs[str(path.relative_to(root))] = sha(path)
    plt.close(fig)
    result = {'analysis_receipt_sha256':sha(receipt_path),'plot_source_sha256':sha(__file__),
        'primary_table_sha256':sha(data/'action_counterfactual_primary_contrasts.csv'),
        'registered_cells':72,'undefined_cells':undefined,'outputs':outputs,
        'interpretation':'Predictor-internal counterfactuals; no physical success or manifold claim'}
    (data/'action_counterfactual_figure.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return result


if __name__ == '__main__':
    print(json.dumps(render(),indent=2))
