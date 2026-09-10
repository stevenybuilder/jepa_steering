from dispatch import *
import time

def tree(w,r):
 code="from pathlib import Path;import json,hashlib; r=Path("+repr(r)+"); sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest(); d=json.load((r/'pilot-v1/DONE.json').open()); assert d['complete']; [(None if sha(r/'pilot-v1'/x['path'])==x['sha256'] else (_ for _ in ()).throw(ValueError(x['path']))) for x in d['outputs']]; print(json.dumps([dict(path=str(p.relative_to(r)),bytes=p.stat().st_size,sha256=sha(p))for p in sorted(r.rglob('*'))if p.is_file()]))"
 return json.loads(run(w,'/opt/conda/bin/python -c '+shlex.quote(code),180))
def preserve(w):
 other={49155754:49902461,49902461:49982193,49982193:49155754}[w];r=root(w);dest=root(other)+f'/backup-from-{w}-pilot-v1';original=tree(w,r)
 run(other,'mkdir '+dest)
 prod=subprocess.Popen(ssh(w)+['tar -czf - -C '+r+' .'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
 cons=subprocess.Popen(ssh(other)+['tar -xzf - -C '+dest],stdin=prod.stdout,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE);prod.stdout.close();_,err=cons.communicate(timeout=600);assert not cons.returncode and not prod.wait(timeout=30),err
 copied=tree(other,dest);assert original==copied
 for x in original:
  if x['path'].startswith('pilot-v1/')and x['path'].endswith('.json'):
   raw=subprocess.check_output(ssh(w)+['cat '+r+'/'+x['path']],timeout=60);assert hashlib.sha256(raw).hexdigest()==x['sha256'];p=LOCAL/f'worker-{w}'/x['path'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
 receipt=dict(complete=True,worker=w,source=r,backup_worker=other,backup=dest,files=original,bytes=sum(x['bytes']for x in original),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));(LOCAL/f'backup-{w}.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items()if k!='files'}),flush=True)
if __name__=='__main__':
 # Sequential copies prevent recursively including another in-progress backup in workerroots.
 for w in HOSTS:preserve(w)
