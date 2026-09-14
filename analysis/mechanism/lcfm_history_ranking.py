"""Post hoc H6 ranking reanalysis of the complete16 action-history experiment.

Only saved costs; no model imports or executions. Run --export-source once to
verify the original archive members and publish the selected cost vectors.
Public reproduction uses the resulting input file and SHA256 manifest.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from .action_counterfactual_summary import rank, finite, score_audit, ARMS

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'paper/data'
PROTOCOL_SHA='a7db541eac268a467100beb9e23a3298064efbf1cc50a298aea0dd327210a88d'
TASKS=('reach','reach-wall');BANKS=('original','fresh');SITES=('all_blocks',*(f'B{i}' for i in range(6)))
METRICS=('spearman','elite_overlap','winner_agreement')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
def arms(site):
    return ('all_blocks_h3_donor','all_blocks_persistent_donor') if site=='all_blocks' else (f'donor_h3_{site}',f'donor_persistent_{site}')
SELECTED=('native','coherent_raw_h3',*(a for site in SITES for a in arms(site)))


def protocol():
    p=DATA/'lcfm_history_ranking_protocol.json'
    if sha(p)!=PROTOCOL_SHA:raise ValueError('Frozen analysis protocol changed')
    return read(p)


def source_summary():
    p=protocol();path=DATA/'action_counterfactual_summary.json'
    if sha(path)!=p['source_summary_sha256']:raise ValueError('Original analysis receipt changed')
    receipt=read(path)
    expected={(t,e) for t in TASKS for e in range(8)}
    keys=[(x['task'],x['episode']) for x in receipt['sources']]
    if len(keys)!=16 or set(keys)!=expected or receipt['cohort']!=16 or not receipt['all_cloud_verified']:
        raise ValueError('Complete verified16 original contexts required')
    return receipt


def export_inputs(source,output):
    receipt=source_summary();records=[]
    # Verify every required original member before reading any new ranking values.
    for r in receipt['sources']:
        directory=Path(source)/r['task']/f"episode-{r['episode']}"
        for name,digest in r['hashes'].items():
            if sha(directory/name)!=digest:raise ValueError(f'Archive member changed: {directory/name}')
        if sha(directory/'CLOUD_VERIFIED.json')!=r['cloud_receipt_sha256']:raise ValueError('Cloud proof changed')
    for r in receipt['sources']:
        directory=Path(source)/r['task']/f"episode-{r['episode']}"
        payload=read(directory/'scores.json')
        if payload['protocol_sha256']!=receipt['protocol_sha256'] or payload['execution_manifest_sha256']!=receipt['execution_manifest_sha256']:
            raise ValueError('Original protocol or manifest changed')
        if set(payload['banks'])!=set(BANKS):raise ValueError('Missing bank')
        for bank in BANKS:
            values=payload['banks'][bank]
            if set(values)!=set(ARMS):raise ValueError('Original 35-arm registry changed')
            selected={}
            for arm in SELECTED:
                scores=values[arm]['scores']
                official=finite(scores['official']['costs'],(300,))
                expected=(finite(scores['visual']['costs'],(300,)).astype(np.float32)+np.float32(.1)*finite(scores['proprio']['costs'],(300,)).astype(np.float32)).astype(float)
                if not np.array_equal(official,expected):raise ValueError('FP32 official goal-cost identity failed')
                score_audit(scores['official'],scores['official'])
                selected[arm]=scores['official']
            if selected['all_blocks_persistent_donor']!=selected['coherent_raw_h3']:raise ValueError('All-block persistent parity failed')
            records.append(dict(task=r['task'],episode=r['episode'],bank=bank,original_scores_sha256=r['hashes']['scores.json'],arms=selected))
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    path=output/'lcfm_history_ranking_inputs.json'
    write(path,dict(protocol_sha256=PROTOCOL_SHA,source_summary_sha256=protocol()['source_summary_sha256'],horizon=6,modality='official',records=records))
    write(output/'lcfm_history_ranking_inputs_manifest.json',dict(input_sha256=sha(path),source_summary_sha256=protocol()['source_summary_sha256'],protocol_sha256=PROTOCOL_SHA,source_members_verified=80,cloud_receipts_verified=16,selected_cost_vectors=512,costs_per_vector=300,exporter_sha256=sha(__file__)))
    print('PASS: 80 original members +16 cloud receipts; 512 FP32-weighted cost vectors exported',flush=True)


def validate_score(score):
    x=finite(score['costs'],(300,))
    ids=np.asarray(score['elite_indices'])
    if ids.shape!=(10,) or ids.dtype.kind not in 'iu' or len(set(ids))!=10 or ids.min()<0 or ids.max()>=300:
        raise ValueError('Invalid archived elite IDs')
    if x[ids].max()>np.delete(x,ids).min():raise ValueError('Archived elites are not a valid top10 set')
    return x,ids


def agreement(left,right):
    a,ia=validate_score(left);b,ib=validate_score(right)
    ra,rb=rank(a),rank(b);ra-=ra.mean();rb-=rb.mean()
    denominator=np.linalg.norm(ra)*np.linalg.norm(rb)
    rho=float(np.clip(ra@rb/denominator,-1,1)) if denominator else None
    ma=np.flatnonzero(a==a.min());mb=np.flatnonzero(b==b.min())
    def cutoff(x):
        c=np.sort(x)[9]
        return dict(cutoff_tie_count=int((x==c).sum()),cutoff_straddles_top10=bool((x<c).sum()<10<(x<=c).sum()))
    ca,cb=cutoff(a),cutoff(b)
    return dict(spearman=rho,elite_overlap=len(set(ia)&set(ib)),winner_agreement=int(ma[0]==mb[0]),
        winner_mismatch=int(ma[0]!=mb[0]),left_winner=int(ma[0]),right_winner=int(mb[0]),
        left_minimum_ties=len(ma),right_minimum_ties=len(mb),minimum_sets_disjoint=int(not set(ma)&set(mb)),
        left_cutoff_ties=ca['cutoff_tie_count'],right_cutoff_ties=cb['cutoff_tie_count'],
        left_cutoff_ambiguous=ca['cutoff_straddles_top10'],right_cutoff_ambiguous=cb['cutoff_straddles_top10'])


def load_inputs(directory):
    directory=Path(directory);manifest=read(directory/'lcfm_history_ranking_inputs_manifest.json');path=directory/'lcfm_history_ranking_inputs.json'
    if manifest['protocol_sha256']!=PROTOCOL_SHA or sha(path)!=manifest['input_sha256']:raise ValueError('Input export hash or protocol mismatch')
    payload=read(path);receipt=source_summary()
    if payload['protocol_sha256']!=PROTOCOL_SHA or payload['source_summary_sha256']!=protocol()['source_summary_sha256'] or payload['horizon']!=6 or payload['modality']!='official':
        raise ValueError('Input endpoint or protocol mismatch')
    records=payload['records'];expected={(t,e,b) for t in TASKS for e in range(8) for b in BANKS}
    keys=[(r['task'],r['episode'],r['bank']) for r in records]
    if len(keys)!=32 or set(keys)!=expected:raise ValueError('Complete32 unique context/banks required')
    hashes={(r['task'],r['episode']):r['hashes']['scores.json'] for r in receipt['sources']}
    for r in records:
        if r['original_scores_sha256']!=hashes[r['task'],r['episode']] or set(r['arms'])!=set(SELECTED):raise ValueError('Source binding or arm registry mismatch')
        for value in r['arms'].values():validate_score(value)
        if r['arms']['all_blocks_persistent_donor']!=r['arms']['coherent_raw_h3']:raise ValueError('Coherent reference parity failed')
    return records,manifest


def calculate(records):
    pairs=[];contrasts=[]
    for r in records:
        base={k:r[k] for k in ('task','episode','bank')};scores=r['arms'];cf=scores['coherent_raw_h3']
        pairs.append(dict(**base,site='reference',comparison='native_vs_counterfactual',**agreement(scores['native'],cf)))
        for site in SITES:
            a,b=arms(site);h3=agreement(scores[a],cf);persistent=agreement(scores[b],cf)
            for name,value in (('h3_vs_counterfactual',h3),('persistent_vs_counterfactual',persistent),('persistent_vs_h3',agreement(scores[b],scores[a]))):
                pairs.append(dict(**base,site=site,comparison=name,**value))
            contrasts.append(dict(**base,site=site,**{m+'_difference':persistent[m]-h3[m] if persistent[m] is not None and h3[m] is not None else None for m in METRICS}))
    if len(pairs)!=704 or len(contrasts)!=224:raise ValueError('Incomplete pairwise registry')
    return pd.DataFrame(pairs),pd.DataFrame(contrasts)


def describe(frame,keys,metrics):
    rows=[]
    for key,g in frame.groupby(keys,sort=True):
        if sorted(g.episode)!=list(range(8)):raise ValueError('Require all eight paired contexts in every cell')
        for metric in metrics:
            values=g[metric].to_numpy(float);valid=np.isfinite(values)
            rows.append(dict(zip(keys,key),metric=metric,n=8,n_defined=int(valid.sum()),
                mean=float(values.mean()) if valid.all() else None,min=float(values.min()) if valid.all() else None,max=float(values.max()) if valid.all() else None,
                sum=float(values.sum()) if valid.all() else None))
    return pd.DataFrame(rows)


def run(inputs,output):
    records,manifest=load_inputs(inputs);pairs,contrasts=calculate(records)
    summary=describe(pairs,['task','bank','site','comparison'],(*METRICS,'winner_mismatch','minimum_sets_disjoint'))
    differences=describe(contrasts,['task','bank','site'],tuple(m+'_difference' for m in METRICS))
    output=Path(output);output.mkdir(parents=True,exist_ok=True);outputs={}
    for name,frame in (('cases',pairs),('paired_differences',contrasts),('summary',summary),('difference_summary',differences)):
        path=output/f'lcfm_history_ranking_{name}.csv'
        if path.exists():raise ValueError('Refuse to overwrite existing analysis')
        frame.to_csv(path,index=False,float_format='%.17g');outputs[path.name]=dict(sha256=sha(path),rows=len(frame))
    report=dict(status='complete_post_hoc_existing_data_ranking_analysis',protocol_sha256=PROTOCOL_SHA,
        analysis_source_sha256=sha(__file__),input_sha256=manifest['input_sha256'],source_summary_sha256=protocol()['source_summary_sha256'],
        tasks=list(TASKS),n_per_task=8,context_banks=32,horizon=6,sites=list(SITES),paired_rows=len(pairs),paired_difference_rows=len(contrasts),
        new_model_calls=0,new_physical_rollouts=0,protected_confirmation=False,inference='Descriptive only; all context values retained, no new tests or intervals',
        undefined_spearman=int(pairs.spearman.isna().sum()),pairs_with_minimum_ties=int(((pairs.left_minimum_ties>1)|(pairs.right_minimum_ties>1)).sum()),
        pairs_with_cutoff_ambiguity=int((pairs.left_cutoff_ambiguous|pairs.right_cutoff_ambiguous).sum()),outputs=outputs)
    write(output/'lcfm_history_ranking_report.json',report)
    print(json.dumps(report,indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--export-source',type=Path);ap.add_argument('--inputs',type=Path,default=DATA);ap.add_argument('--output',type=Path,default=DATA)
    args=ap.parse_args()
    if args.export_source:export_inputs(args.export_source,args.output)
    else:run(args.inputs,args.output)

if __name__=='__main__':main()
