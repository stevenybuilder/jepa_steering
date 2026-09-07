from dispatch import *
import time,io,tarfile

def tree(w,r):
 code="""from pathlib import Path
import json,hashlib
r=Path(ROOT);sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest();d=json.load((r/'results-v1/DONE.json').open());assert d['complete']
for x in d['outputs']:assert sha(r/'results-v1'/x['path'])==x['sha256']
s=json.load((r/'results-v1/summary.json').open());checkpoint=Path(s['provenance']['checkpoint']);assert sha(checkpoint)=='c3297772c7af4e84c28bd0c0937e22398a888848305014bb3ab3f1508a310bd8'
files=[dict(path=str(p.relative_to(r)),bytes=p.stat().st_size,sha256=sha(p))for p in sorted(r.rglob('*'))if p.is_file()and '__pycache__' not in p.parts];print(json.dumps(dict(files=files,checkpoint_sha256=sha(checkpoint))))
""".replace('ROOT',repr(r))
 return json.loads(run(w,'/opt/conda/bin/python -c '+shlex.quote(code),180))
def preserve(w):
 r=root(w)+'/horizon-diagnostic-v1'
 # Short local reads preserve compact conclusions first; hashvalidated again withalloutputs.
 for _ in range(20):
  state=run(w,f'test -f {r}/results-v1/DONE.json && echo DONE; test -f {r}/results-v1/FAILED.json && echo FAILED; true').strip()
  if state=='DONE':break
  if 'FAILED'in state:raise RuntimeError(run(w,'cat '+r+'/results-v1/FAILED.json'))
  time.sleep(2)
 else:raise RuntimeError('No completion afterboundedmonitor')
 original=tree(w,r);other={49155754:49902461,49902461:49982193,49982193:49155754}[w];dest=BASE+f'/autoresearch-steering-20260906/verified-backups/horizon-diagnostic-worker-{w}';run(other,'mkdir -p '+dest)
 prod=subprocess.Popen(ssh(w)+['tar -czf - -C '+r+' .'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL);cons=subprocess.Popen(ssh(other)+['tar -xzf - -C '+dest],stdin=prod.stdout,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE);prod.stdout.close();_,err=cons.communicate(timeout=900);assert not cons.returncode and not prod.wait(timeout=30),err
 copied=tree(other,dest);assert original==copied
 selected=[x for x in original['files']if x['path'].endswith('.json')];data=subprocess.check_output(ssh(w)+['tar -czf - -C '+r+' '+shlex.join(x['path']for x in selected)],timeout=180)
 with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz')as tar:
  for x in selected:
   raw=tar.extractfile(x['path']).read();assert hashlib.sha256(raw).hexdigest()==x['sha256'];p=LOCAL/f'horizon-diagnostic-worker-{w}'/x['path'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
 receipt=dict(complete=True,worker=w,source=r,backup_worker=other,backup=dest,**original,bytes=sum(x['bytes']for x in original['files']),source_and_backup_identical=True,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));(LOCAL/f'horizon-diagnostic-backup-{w}.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items()if k!='files'}),flush=True)
with ThreadPoolExecutor(3)as ex:list(ex.map(preserve,HOSTS))
