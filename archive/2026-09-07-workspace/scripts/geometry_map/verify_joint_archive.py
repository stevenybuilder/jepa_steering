#!/usr/bin/env python3
"""CPU-only byte manifest for this experiment's immutable source/backup roots."""
import argparse
import hashlib
import json
import time
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--verify',action='store_true');a=p.parse_args();t=time.monotonic()
    manifest=a.root/'ARCHIVE_MANIFEST.json'
    if a.verify:
        d=json.loads(manifest.read_text())
        for r in d['entries']:
            q=a.root/r['path']
            if q.stat().st_size!=r['bytes'] or sha(q)!=r['sha256']:raise ValueError('Backup mismatch '+str(q))
        print(json.dumps(dict(complete=True,root=str(a.root),manifest_sha256=sha(manifest),files=len(d['entries']),bytes=sum(r['bytes'] for r in d['entries']),seconds=time.monotonic()-t)),flush=True)
    else:
        rows=[dict(path=str(q.relative_to(a.root)),bytes=q.stat().st_size,sha256=sha(q)) for q in sorted(a.root.rglob('*')) if q.is_file() and q!=manifest]
        d=dict(complete=True,root=str(a.root),entries=rows,files=len(rows),bytes=sum(r['bytes'] for r in rows),seconds=time.monotonic()-t)
        with manifest.open('x') as f:json.dump(d,f,indent=2);f.write('\n')
        print(json.dumps(dict(complete=True,root=str(a.root),manifest_sha256=sha(manifest),files=len(rows),bytes=d['bytes'],seconds=d['seconds'])),flush=True)


if __name__=='__main__':main()
