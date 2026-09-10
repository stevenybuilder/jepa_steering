#!/usr/bin/env python3
"""Stream-check full remote curvature tensors or compact local reports."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''): value.update(block)
    return value.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--compact-only',action='store_true');p.add_argument('--results-name',default='results-v2',choices=['results-v2','results-v1']);p.add_argument('--actual-context',action='store_true');args=p.parse_args()
    if args.output.exists():raise FileExistsError('Immutable verification receipt exists')
    started=time.monotonic();done_path=args.root/args.results_name/'DONE.json';done=json.loads(done_path.read_text())
    expected=(done.get('paths')==32) if args.results_name=='results-v2' else (done.get('targets')==64)
    if not done['complete'] or not expected or (not args.actual_context and done['methods']!=6):raise ValueError('Wrong completed curvature design')
    if args.actual_context:
        protocol=json.loads((done_path.parent/'protocol.json').read_text())
        if protocol['modes']!=['native','actual_past_context'] or not protocol['H1_identity_required']:raise ValueError('Wrong actual-context protocol')
    verified=[]
    for row in done['outputs']:
        if args.compact_only and not row['path'].endswith('.json'):continue
        path=done_path.parent/row['path']
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise ValueError('Failed checksum '+str(path))
        verified.append(row)
    receipt=dict(complete=True,scope='compact_reports_only' if args.compact_only else 'all_full_tensors_and_reports',
        root=str(args.root),outputs=verified,verified_bytes=sum(r['bytes'] for r in verified),
        done_sha256=sha(done_path),seconds=time.monotonic()-started)
    code=args.root/'code-v2'/'scripts'/'geometry_map'
    if not code.exists():code=args.root/'code-v1'/'scripts'/'geometry_map'
    if code.exists():receipt['executed_sources']={str(p.relative_to(args.root)):sha(p) for p in sorted(code.glob('*.py'))}
    args.output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('outputs','executed_sources')}))


if __name__=='__main__':main()
