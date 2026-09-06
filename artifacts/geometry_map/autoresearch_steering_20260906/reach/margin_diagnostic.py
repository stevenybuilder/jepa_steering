from pathlib import Path
import argparse,json,time
import numpy as np
import torch

def run(a):
 torch.set_num_threads(1);rows=[]
 for family in ['pilot-v1','pilot-fullgrid-v1']:
  r=Path(a.root)/family;done=json.load((r/'DONE.json').open());assert done['complete'];entries={x['path']:x for x in done['outputs']}
  from extract_fit import sha
  for p in sorted(r.glob('episode-*-bank.pt')):
   assert sha(p)==entries[p.name]['sha256'];d=torch.load(p,map_location='cpu',weights_only=False);b=np.array(d['costs']['native']);win=int(b.argmin());sortedidx=b.argsort();delta=np.array(d['costs']['edit'])-b;m=b-b[win];adv=delta[win]-delta;mask=(m>1e-9)&(adv>0)
   ranks=b.argsort().argsort();cranks=np.array(d['costs']['edit']).argsort().argsort()
   rows.append(dict(family=family,episode=int(p.name.split('-')[1]),n_candidates=len(b),native_win=win,edit_win=int(np.argmin(d['costs']['edit'])),native_winner_margin=float(b[sortedidx[1]]-b[win]),native_cost_std=float(b.std()),correction_mean=float(delta.mean()),correction_std=float(delta.std()),correction_std_over_native_std=float(delta.std()/max(b.std(),1e-12)),maximum_fraction_of_native_margin_closed=float(np.max(adv[mask]/m[mask]))if mask.any() else 0.,rank_correlation_native_vs_edit=float(np.corrcoef(ranks,cranks)[0,1]),selected_native_mean_progress_is_separate='Bank winner may differ from native planner selected mean; no physicaloracle for301 candidates'))
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False);(out/'summary.json').write_text(json.dumps(dict(complete=True,rows=rows,scope='Read-only saved candidate costs; no new model/simulator/fit or held access'),indent=2));(out/'DONE.json').write_text(json.dumps(dict(complete=True,outputs=[dict(path='summary.json',sha256=sha(out/'summary.json'),bytes=(out/'summary.json').stat().st_size)]),indent=2));print(json.dumps(dict(complete=True,rows=rows)),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);run(p.parse_args())
