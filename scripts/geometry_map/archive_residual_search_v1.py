#!/usr/bin/env python3
"""Preserve completed search outputs on a second leased worker without local PTs."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def ssh(host, port):
    return ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=10", "-p", str(port), "root@"+host]


def remote(args, code):
    return subprocess.check_output(args+["/opt/conda/bin/python -c "+shlex.quote(code)], text=True)


def archive(args):
    source, target = ssh(args.source_host, args.source_port), ssh(args.target_host, args.target_port)
    source_path = Path(args.source_root)
    if not source_path.is_absolute() or not args.target_root.startswith("/root/geometry-map-jepawm-reach-wall-v1/residual-search-v1/remote-backups/"):
        raise ValueError("Explicit owned search source and backup root required")
    marker = "FAILED.json" if args.allow_failed else "DONE.json"
    reader = """from pathlib import Path
import json,hashlib
p=Path(ROOT)
if FAILED:
 assert (p/'FAILED.json').exists()
 rows=[]
 for f in sorted(p.glob('*.DONE.json')):
  d=json.loads(f.read_text()); assert d['complete']; rows.extend(d['outputs'])
 for f in sorted(p.glob('*.json')):
  rows.append({'path':f.name,'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
 r={'complete':False,'outputs':rows}
else:r=json.loads((p/'DONE.json').read_text())
print(json.dumps(r))
"""
    receipt = json.loads(remote(source, reader.replace("ROOT", repr(str(source_path))).replace("FAILED", repr(args.allow_failed), 1)))
    if not receipt["complete"] and not args.allow_failed:
        raise ValueError("Incomplete source cannot be released")
    for row in receipt["outputs"]:
        if Path(row["path"]).name != row["path"]:
            raise ValueError("Only flat artifact names accepted")
    # No local tensor file: the source stream is piped directly into target SSH.
    remote(target, f"from pathlib import Path; Path({args.target_root!r}).mkdir(parents=True, exist_ok=False)")
    producer = subprocess.Popen(source+["tar -C "+shlex.quote(str(source_path))+" -cf - ."], stdout=subprocess.PIPE)
    consumer = subprocess.Popen(target+["tar -C "+shlex.quote(args.target_root)+" -xf -"], stdin=producer.stdout)
    producer.stdout.close()
    if consumer.wait() or producer.wait():
        raise RuntimeError("Backup transport failed; preserve partial target for diagnosis")
    verify = """import json,hashlib
from pathlib import Path
p=Path(ROOT)
r=RECEIPT
for x in r['outputs']:
 assert hashlib.sha256((p/x['path']).read_bytes()).hexdigest()==x['sha256'],x['path']
print(json.dumps({'complete':r['complete'],'file_count':len(r['outputs']),'bytes':sum(x['bytes'] for x in r['outputs']),'marker_sha256':hashlib.sha256((p/MARKER).read_bytes()).hexdigest()}))
"""
    verify = verify.replace("RECEIPT", repr(receipt)).replace("MARKER", repr(marker))
    source_check = json.loads(remote(source, verify.replace("ROOT", repr(str(source_path)))))
    target_check = json.loads(remote(target, verify.replace("ROOT", repr(args.target_root))))
    if source_check != target_check:
        raise RuntimeError("Source and backup receipts differ")
    args.local_root.mkdir(parents=True, exist_ok=False)
    compact = list(dict.fromkeys([marker]+[r["path"] for r in receipt["outputs"] if r["path"].endswith(".json")]))
    producer = subprocess.Popen(source+["tar -C "+shlex.quote(str(source_path))+" -cf - "+" ".join(map(shlex.quote, compact))], stdout=subprocess.PIPE)
    consumer = subprocess.Popen(["tar", "-C", str(args.local_root), "-xf", "-"], stdin=producer.stdout)
    producer.stdout.close()
    if consumer.wait() or producer.wait():
        raise RuntimeError("Compact copy failed")
    for row in receipt["outputs"]:
        if row["path"] in compact:
            assert hashlib.sha256((args.local_root/row["path"]).read_bytes()).hexdigest()==row["sha256"]
    report = {"source_host": args.source_host, "source_root": str(source_path), "backup_host": args.target_host,
              "backup_root": args.target_root, "source_and_backup_sha_verified": True, "local_tensors_copied": False,
              "compact_local_sha_verified": True, **source_check}
    with (args.local_root/"REMOTE_BACKUP_VERIFIED.json").open("x") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-host", "target-host", "source-root", "target-root"):
        parser.add_argument("--"+name, required=True)
    for name in ("source-port", "target-port"):
        parser.add_argument("--"+name, type=int, required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--allow-failed", action="store_true", help="Preserve failed marker and every completed partial shard; never label complete")
    archive(parser.parse_args())
