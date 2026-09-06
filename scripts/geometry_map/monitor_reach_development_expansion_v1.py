#!/usr/bin/env python3
"""Read-only monitoring plus authorized two-host preservation of DEV66..73."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

LOCAL = Path(__file__).resolve().parents[2]/'artifacts/geometry_map/reach_development_expansion_v1'
BASE = '/root/geometry-map-jepawm-reach-wall-v1/development-expansion-66-73-v1'
HOSTS = {49902461:('83.195.246.217',50048),49982193:('60.53.150.159',65122)}
KEY = '/Users/stevenyang/.ssh/id_ed25519'


def ssh(worker):
    host,port=HOSTS[worker]
    return ['ssh','-i',KEY,'-o','BatchMode=yes','-o','ConnectTimeout=12','-p',str(port),'root@'+host]


def remote(worker, code, timeout=60):
    p=subprocess.run(ssh(worker)+['/opt/conda/bin/python -'],input=code,text=True,capture_output=True,timeout=timeout)
    if p.returncode:
        raise RuntimeError(p.stderr+'\n'+p.stdout)
    return json.loads(p.stdout.strip().splitlines()[-1])


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def status(worker):
    root=f'{BASE}/worker-{worker}'
    return remote(worker,f"""
import json,subprocess
from pathlib import Path
r=Path({root!r});out=r/'results-v2'
events=[]
for line in (r/'job-v2.log').read_text().splitlines():
 try:
  x=json.loads(line)
  if x.get('event') in ('full99_replan_complete','new_development_episode_complete'):events.append(x)
 except ValueError:pass
done=(out/'DONE.json').exists();failed=(out/'FAILED.json').exists()
pid=json.loads((out/'protocol.json').read_text())['process_pid']
running=subprocess.run(['ps','-p',str(pid)],capture_output=True).returncode==0
print(json.dumps(dict(worker={worker},pid=pid,running=running,done=done,failed=failed,
 last_event=events[-1] if events else None,completed_episodes=sum(x['event']=='new_development_episode_complete' for x in events),
 failure=json.loads((out/'FAILED.json').read_text()) if failed else None)))
""")


def tree(worker, root):
    return remote(worker,f"""
from pathlib import Path
import hashlib,json
r=Path({root!r});d=json.loads((r/'results-v2/DONE.json').read_text());assert d['complete']
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
for e in d['outputs']:
 p=r/'results-v2'/e['path'];assert p.stat().st_size==e['bytes'];assert sha(p)==e['sha256']
files=[dict(path=str(p.relative_to(r)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(r.rglob('*')) if p.is_file()]
print(json.dumps(dict(files=files,bytes=sum(x['bytes'] for x in files),done_sha256=sha(r/'results-v2/DONE.json'))))
""",timeout=180)


def preserve(worker):
    other=next(x for x in HOSTS if x!=worker)
    source=f'{BASE}/worker-{worker}';dest=f'{BASE}/backup-from-{worker}'
    original=tree(worker,source)
    subprocess.run(ssh(other)+['test ! -e '+shlex.quote(dest)+' && mkdir '+shlex.quote(dest)],check=True,capture_output=True)
    producer=subprocess.Popen(ssh(worker)+['tar -czf - -C '+shlex.quote(source)+' .'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    consumer=subprocess.Popen(ssh(other)+['tar -xzf - -C '+shlex.quote(dest)],stdin=producer.stdout,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    producer.stdout.close()
    _,err=consumer.communicate(timeout=1200)
    if consumer.returncode or producer.wait(timeout=30):raise RuntimeError('Remote backup failed: '+err.decode())
    copied=tree(other,dest)
    if copied!=original:raise RuntimeError('Source/backup tree SHA disagreement')
    selected=[x['path'] for x in original['files'] if x['path'].endswith('.json') and '/code-' not in '/'+x['path']]
    data=subprocess.check_output(ssh(worker)+['tar -czf - -C '+shlex.quote(source)+' '+shlex.join(selected)],timeout=180)
    target=LOCAL/f'worker-{worker}'
    with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
        for name in selected:
            raw=archive.extractfile(name).read()
            expected=next(x for x in original['files'] if x['path']==name)
            if hashlib.sha256(raw).hexdigest()!=expected['sha256']:raise RuntimeError('Local JSON SHA mismatch')
            path=target/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    result=dict(complete=True,source_worker=worker,source_root=source,backup_worker=other,backup_root=dest,
                source_and_backup_identical=True,local_full_tensors=False,local_json_files=len(selected),**original)
    write(target/'VERIFIED.json',result)
    return result


def monitor(worker):
    while True:
        s=status(worker);write(LOCAL/f'worker-{worker}'/'STATUS.json',s)
        print(json.dumps(s),flush=True)
        if s['failed']:raise RuntimeError(f'Worker{worker} failed: '+json.dumps(s['failure']))
        if s['done'] and not s['running']:break
        if not s['running']:raise RuntimeError(f'Worker{worker} exited without DONE')
        time.sleep(35)
    print(json.dumps(dict(event='backup_started',worker=worker)),flush=True)
    result=preserve(worker)
    print(json.dumps(dict(event='worker_verified',worker=worker,bytes=result['bytes'],done_sha256=result['done_sha256'])),flush=True)
    return result


def main():
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={k:pool.submit(monitor,k) for k in HOSTS}
        results=[]
        for worker,future in futures.items():
            try:results.append(future.result())
            except Exception as exc:
                write(LOCAL/f'worker-{worker}'/'MONITOR_FAILED.json',dict(complete=False,error=repr(exc)))
    if len(results)==2:
        write(LOCAL/'ALL_VERIFIED.json',dict(complete=True,workers=[{k:r[k] for k in ('source_worker','backup_worker','bytes','done_sha256')} for r in results]))


if __name__=='__main__':main()
