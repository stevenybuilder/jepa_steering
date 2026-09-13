"""Resume only the pinned prepared-runtime object after a stopped writer."""
import argparse, json, shlex, subprocess
from pathlib import Path
from fresh_gcs_download import token, URL, SHA

REMOTE = '''import hashlib,json,sys,urllib.request,time,subprocess,os
from pathlib import Path
r=json.loads(sys.stdin.read()); p=Path('/workspace/prepared-runtime-inputs-v2.tgz')
active=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
 try:
  for fd in (proc/'fd').iterdir():
   try:
    if fd.resolve()==p:active.append(proc.name)
   except FileNotFoundError:pass
 except (FileNotFoundError,PermissionError):pass
if active: raise RuntimeError('Existing archive file descriptors remain: '+repr(active))
n=p.stat().st_size
if not 0<n<=1471380149: raise ValueError('Unexpected partial size')
h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
receipt={'resume_offset':n,'prefix_sha256':h.hexdigest(),'expected_sha256':r['sha256'],'utc_unix':time.time()}
with Path('/workspace/runtime-range-resume.json').open('x') as f:json.dump(receipt,f)
if n<1471380149:
 req=urllib.request.Request(r['url'],headers={'Authorization':'Bearer '+r['token'],'Range':'bytes='+str(n)+'-'})
 with urllib.request.urlopen(req,timeout=60) as response,p.open('ab') as out:
  if response.status!=206 or not response.headers['Content-Range'].startswith('bytes '+str(n)+'-'):raise ValueError('Range response mismatch')
  for b in iter(lambda:response.read(1048576),b''):out.write(b);h.update(b)
  out.flush();os.fsync(out.fileno())
if p.stat().st_size!=1471380149 or h.hexdigest()!=r['sha256']:raise ValueError('Complete archive hash differs')
print(json.dumps({'resume_offset':n,'complete_bytes':p.stat().st_size,'sha256':h.hexdigest(),'download_verified':True}),flush=True)
'''

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--host',required=True);p.add_argument('--port',required=True);p.add_argument('--known-hosts',required=True);a=p.parse_args()
    scoped=token()
    ssh=['ssh','-i','/Users/stevenyang/.ssh/id_ed25519','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+a.known_hosts,'-p',a.port,'root@'+a.host,shlex.join(['python3','-c',REMOTE])]
    subprocess.run(ssh,input=json.dumps({'token':scoped,'url':URL,'sha256':SHA}),text=True,check=True,timeout=300)
