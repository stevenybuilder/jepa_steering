#!/usr/bin/env python3
"""Freeze144 path-point references to132 distinct development action histories."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    for k in ('new_histories','existing_donors','output'):p.add_argument('--'+k.replace('_','-'),type=Path,required=True)
    a=p.parse_args();new=json.loads((a.new_histories/'DONE.json').read_text());old=json.loads((a.existing_donors/'DONE.json').read_text())
    if not new['complete'] or not old['complete']:raise ValueError('Both receipts must be complete before any tensor loading')
    nr={r['path']:r for r in new['outputs'] if r['path'].endswith('.pt')};er={r['path']:r for r in old['outputs'] if r['path'].endswith('.pt')};rows=[]
    for e in range(4):
        for pair in range(4):
            for c in (-4.,-2.,-1.,-.5,0.,.5,1.,2.,4.):
                stem=f'near-dev-{e:03}';sign='minus' if c<0 else 'plus'
                if c==0:
                    name=stem+'-central-reused.pt';root=a.new_histories;entry=nr[name]
                elif abs(c)==1:
                    name=stem+f'-pair{pair}-radius4-t{sign}025.pt';root=a.existing_donors;entry=er[name]
                else:
                    radius=1 if abs(c)==.5 else 4;t='100' if abs(c)==4 else '050'
                    name=stem+f'-pair{pair}-radius{radius}-t{sign}{t}.pt';root=a.new_histories;entry=nr[name]
                path=root/name
                if not path.is_file() or path.stat().st_size!=entry['bytes']:raise ValueError('Missing or incomplete physical history '+str(path))
                rows.append(dict(episode=e,pair=pair,coefficient=c,path=str(path),sha256=entry['sha256'],bytes=entry['bytes'],
                    split='development_external',source_done_sha256=sha(root/'DONE.json')))
    if len(rows)!=144 or len({r['path'] for r in rows})!=132:raise ValueError('Wrong distinct action-history accounting')
    if a.output.exists():raise FileExistsError('Immutable index exists')
    a.output.write_text(json.dumps(dict(complete=True,rows=rows,rows_count=144,distinct_histories=132,
        initial_states=4,source_new_histories=100,source_existing_pm1=32,coefficient_zero_reused_across_pairs=True,
        actual_history_rule='At predictionH provide sourceinitial and actualframes only throughH−1',
        source_done_hashes=dict(new=sha(a.new_histories/'DONE.json'),existing=sha(a.existing_donors/'DONE.json'))),indent=2)+'\n')
    print(json.dumps(dict(complete=True,rows=144,distinct_histories=132,index_sha256=sha(a.output))))


if __name__=='__main__':main()
