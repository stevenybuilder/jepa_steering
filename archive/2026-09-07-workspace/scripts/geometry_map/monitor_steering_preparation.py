#!/usr/bin/env python3
"""Collect sealed baseline receipts/tensor bytes; never deserialize held tensors.

Only monitors already launched, project-owned workers. No launch, stop, rental,
model call or evaluation-outcome inspection is possible through this program.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda:f.read(8<<20),b""):h.update(part)
    return h.hexdigest()


def verify_local(job, project):
    root=project/job["local_directory"]
    path=root/"SEALED_DONE.json"
    if not path.exists():return None
    receipt=json.loads(path.read_text())
    if not receipt.get("complete"):raise RuntimeError("Incomplete sealed receipt")
    entries=receipt["outputs"]
    if sorted(r["episode"] for r in entries)!=job["episode_ids"]:raise RuntimeError("Wrong/overlapping episodeIDs")
    for row in entries:
        name=Path(row["path"])
        if name.name!=str(name) or name.suffix!=".pt":raise RuntimeError("Unsafe output filename")
        target=root/name
        if not target.exists():return None
        if sha(target)!=row["sha256"]:raise RuntimeError("Copied tensor checksum mismatch")
    if not (root/"DONE.json").exists():return None
    if sha(root/"DONE.json")!=receipt["native_done_sha256"]:raise RuntimeError("NativeDONE checksum mismatch")
    return {"instance_id":job["instance_id"],"task":job["task"],"episode_ids":job["episode_ids"],
            "local_directory":job["local_directory"],"complete":True,"files":len(entries),
            "bytes":sum((root/r["path"]).stat().st_size for r in entries),
            "sealed_done_sha256":sha(path),"native_done_sha256":receipt["native_done_sha256"],
            "held_tensors_deserialized":False,"outputs":entries}


def execute(command,timeout=180):
    p=subprocess.run(command,text=True,capture_output=True,timeout=timeout)
    if p.returncode:raise RuntimeError(f"Command failed({p.returncode}): {p.stderr[-700:]}")
    return p.stdout


def provider_state():
    rows=json.loads(execute(["vastai","show","instances","--raw"],60))
    return {int(r["id"]):{k:r.get(k) for k in ("id","actual_status","ssh_host","ssh_port")} for r in rows}


def remote_progress(job):
    # Source is restricted to progress lines already emitted by outcome-filtering
    # collectors. Native DONE and all tensor payloads remain unparsed here.
    code="""import json,pathlib,subprocess
root=pathlib.Path(ROOT)
rows=[]
log=pathlib.Path(LOG)
for line in log.read_text().splitlines() if log.exists() else []:
 try:
  value=json.loads(line)
  if isinstance(value,dict) and value.get('event') in ('episode','episode_saved','replan','checkpoint_verified'):
   rows.append({k:v for k,v in value.items() if k in ('event','episode','completed','replan','seconds','elapsed_seconds','sha256','checkpoint_sha256')})
 except ValueError:pass
processes=subprocess.check_output(['ps','-eo','args'],text=True).splitlines()
active=any(line.startswith('python -u ') and TOKEN in line for line in processes)
print(json.dumps({'sealed_done':(root/'SEALED_DONE.json').exists(),'active':active,'completed_progress_count':sum(r['event'] in ('episode','episode_saved') for r in rows),'last_progress':rows[-1] if rows else None}))
""".replace("ROOT",repr(job["remote_directory"])).replace("LOG",repr(job["remote_log"])).replace("TOKEN",repr(job["process_token"]))
    output=execute(["ssh","-o","ConnectTimeout=15","-p",str(job["port"]),"root@"+job["host"],"python -c "+shlex.quote(code)],45)
    return json.loads(output.strip().splitlines()[-1])


def collect(job,project,provider,board):
    done=verify_local(job,project)
    if done:return done
    text=board.read_text()
    table_row=next((line for line in text.splitlines() if line.startswith("| `"+str(job["instance_id"])+"` |")),"")
    if "rep_geometry_transcoder" not in table_row or "DESTROYED" in table_row or "QUARANTINED" in table_row:
        raise RuntimeError("Project lease check failed; monitor will not touch this worker")
    if job["instance_id"] not in provider:raise RuntimeError("Provider no longer lists pending ownedworker")
    status=remote_progress(job)
    if status["sealed_done"]:
        local=project/job["local_directory"]
        local.mkdir(parents=True,exist_ok=True)
        execute(["rsync","-a","--partial","-e",f"ssh -o ConnectTimeout=15 -p {job['port']}",
                 "root@"+job["host"]+":"+job["remote_directory"].rstrip("/")+"/",str(local)+"/"],600)
        done=verify_local(job,project)
        if not done:raise RuntimeError("Incomplete copy after sealedDONE")
        return done
    if not status["active"]:raise RuntimeError("Baselineprocess absent without sealedDONE; no automatic experimental retry")
    return {"instance_id":job["instance_id"],"task":job["task"],"complete":False,"episode_ids":job["episode_ids"],**status}


def save(path,value):
    temp=path.with_suffix(".tmp")
    temp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
    temp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--project",type=Path,required=True)
    parser.add_argument("--board",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    config=json.loads(args.manifest.read_text())
    jobs=config["jobs"]
    args.output.mkdir(parents=True,exist_ok=True)
    errors={}
    while True:
        provider=provider_state()
        rows=[]
        def checked(job):
            key=job["key"]
            try:
                result=collect(job,args.project,provider,args.board)
                errors[key]=0
                return {"key":key,**result}
            except Exception as exc:
                errors[key]=errors.get(key,0)+1
                return {"key":key,"task":job["task"],"complete":False,"error":str(exc),"consecutive_errors":errors[key]}
        with ThreadPoolExecutor(max_workers=5) as pool:rows=list(pool.map(checked,jobs))
        verified={task:sum(r.get("files",0) for r in rows if r.get("complete") and r["task"]==task) for task in ("reach_wall","pusht_scripted_v2")}
        progress={"updated_utc":datetime.now(timezone.utc).isoformat(),"complete":all(r["complete"] for r in rows),
                  "scope":"new38Reach+new50scriptedPush unsteeredbaseline preparation only; no efficacy or steeringclaim",
                  "verified_baseline_files":verified,"held_tensors_deserialized":False,"rows":rows,
                  "manifest_sha256":sha(args.manifest),"monitor_script_sha256":sha(Path(__file__))}
        save(args.output/"PROGRESS.json",progress)
        print(json.dumps({"updated_utc":progress["updated_utc"],"complete":progress["complete"],"verified_baseline_files":verified,
                          "pending":[r["key"] for r in rows if not r["complete"]]}),flush=True)
        if progress["complete"]:
            if verified!={"reach_wall":38,"pusht_scripted_v2":50}:raise RuntimeError("Wrong final baselinecoverage")
            save(args.output/"DONE.json",progress)
            return
        if any(n>=3 for n in errors.values()):
            save(args.output/"NEEDS_ATTENTION.json",progress)
        time.sleep(60)


if __name__=="__main__":main()
