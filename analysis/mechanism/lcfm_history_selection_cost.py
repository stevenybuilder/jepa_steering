"""Descriptive cost differences in existing saved candidate banks; no simulator claims."""
from pathlib import Path
import json,hashlib,csv,statistics
import argparse
ap=argparse.ArgumentParser(description='Recount excess model-reference cost behind the history winner changes.')
ap.add_argument('--output',type=Path,required=True)
args=ap.parse_args()
if args.output.exists(): raise ValueError('Use a new output directory')
args.output.mkdir(parents=True)
root=Path(__file__).resolve().parents[2];data=root/'paper/data';p=data/'lcfm_history_ranking_inputs.json';proto=data/'lcfm_history_selection_cost_protocol.json';assert hashlib.sha256(p.read_bytes()).hexdigest()==json.loads(proto.read_text())['input_sha256']
rows=[]
for r in json.loads(p.read_text())['records']:
 ref=r['arms']['coherent_raw_h3']['costs'];best=min(range(300),key=lambda i:ref[i])
 for arm in ['all_blocks_h3_donor','all_blocks_persistent_donor']:
  c=r['arms'][arm]['costs'];selected=min(range(300),key=lambda i:c[i]);gap=ref[selected]-ref[best]
  rows.append(dict(task=r['task'],episode=r['episode'],bank=r['bank'],arm=arm,reference_best=best,selected=selected,changed=int(best!=selected),reference_min_cost=ref[best],excess_reference_cost=gap,excess_reference_cost_percent=100*gap/ref[best] if ref[best]>0 else None))
with (args.output/'lcfm_history_selection_cost_cases.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=[]
for task in ['reach','reach-wall']:
 for bank in ['original','fresh']:
  rs=[r for r in rows if r['task']==task and r['bank']==bank and r['arm']=='all_blocks_h3_donor'];assert len(rs)==8
  summary.append(dict(task=task,bank=bank,n_contexts=8,changed=sum(r['changed'] for r in rs),mean_excess_reference_cost=statistics.mean(r['excess_reference_cost'] for r in rs),mean_excess_reference_cost_percent=statistics.mean(r['excess_reference_cost_percent'] for r in rs),median_excess_reference_cost_percent=statistics.median(r['excess_reference_cost_percent'] for r in rs),max_excess_reference_cost_percent=max(r['excess_reference_cost_percent'] for r in rs)))
assert sum(r['changed'] for r in rows)==17
assert all(r['excess_reference_cost']==0 for r in rows if r['arm']=='all_blocks_persistent_donor')
receipt={'role':'post_hoc_model_reference_cost_reanalysis','physical_outcomes':False,'independent_contexts':16,'banks':32,'protocol_sha256':hashlib.sha256(proto.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'analysis_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'summary':summary}
(args.output/'lcfm_history_selection_cost_summary.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(summary,indent=2))
