from dispatch import *
import time,io,tarfile

def tree(w,r):
 code="""from pathlib import Path
import json,hashlib
r=Path(ROOT)
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
for sub in ['pilot-v1','pilot-fullgrid-v1','margin-diagnostic-v1']:
 d=json.load((r/sub/'DONE.json').open());assert d['complete']
 for x in d['outputs']:assert sha(r/sub/x['path'])==x['sha256'],x['path']
files=[]
for p in sorted(r.rglob('*')):
 rel=p.relative_to(r)
 if p.is_file()and not any(x.startswith('backup-')or x=='__pycache__'for x in rel.parts):files.append(dict(path=str(rel),bytes=p.stat().st_size,sha256=sha(p)))
print(json.dumps(files))
""".replace('ROOT',repr(r))
 return json.loads(run(w,'/opt/conda/bin/python -c '+shlex.quote(code),180))
def preserve(w):
 other={49155754:49902461,49902461:49982193,49982193:49155754}[w];r=root(w);dest=BASE+f'/autoresearch-steering-20260906/verified-backups/fullgrid-worker-{w}'
 original=tree(w,r);run(other,'mkdir -p '+dest)
 names=[x['path']for x in original]
 prod=subprocess.Popen(ssh(w)+['tar -czf - -C '+r+' '+shlex.join(names)],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
 cons=subprocess.Popen(ssh(other)+['tar -xzf - -C '+dest],stdin=prod.stdout,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE);prod.stdout.close();_,err=cons.communicate(timeout=900);assert not cons.returncode and not prod.wait(timeout=30),err
 copied=tree(other,dest);assert original==copied
 selected=[x for x in original if x['path'].endswith('.json')and not x['path'].startswith('code-')]
 data=subprocess.check_output(ssh(w)+['tar -czf - -C '+r+' '+shlex.join(x['path']for x in selected)],timeout=180)
 with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz')as tar:
  for x in selected:
   raw=tar.extractfile(x['path']).read();assert hashlib.sha256(raw).hexdigest()==x['sha256'];p=LOCAL/f'worker-{w}'/x['path'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
 receipt=dict(complete=True,worker=w,source=r,backup_worker=other,backup=dest,files=original,bytes=sum(x['bytes']for x in original),source_and_backup_identical=True,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));(LOCAL/f'backup-fullgrid-{w}.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items()if k!='files'}),flush=True)
with ThreadPoolExecutor(3)as ex:list(ex.map(preserve,HOSTS))
