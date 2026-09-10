from pathlib import Path
import subprocess,shlex,json,hashlib
from concurrent.futures import ThreadPoolExecutor
LOCAL=Path(__file__).resolve().parent
HOSTS={49155754:('192.220.55.116',20566),49902461:('83.195.246.217',50048),49982193:('60.53.150.159',65122)}
BASE='/root/geometry-map-jepawm-reach-wall-v1'
def ssh(w):
 h,p=HOSTS[w];return ['ssh','-i','/Users/stevenyang/.ssh/id_ed25519','-o','BatchMode=yes','-o','ConnectTimeout=12','-p',str(p),'root@'+h]
def root(w):return f'{BASE}/autoresearch-steering-20260906/worker-{w}'
def run(w,cmd,timeout=60):
 p=subprocess.run(ssh(w)+[cmd],capture_output=True,text=True,timeout=timeout)
 if p.returncode:raise RuntimeError(p.stderr+p.stdout)
 return p.stdout
def upload(w,p,dest):
 with open(p,'rb')as f:subprocess.run(ssh(w)+['cat > '+shlex.quote(dest)],stdin=f,check=True,capture_output=True)
def extract(w):
 r=root(w);run(w,'mkdir -p '+r+'/code-v1');upload(w,LOCAL/'extract_fit.py',r+'/code-v1/extract_fit.py')
 dirs=[f'{BASE}/residual-search-v1/full99-pop0-v2-worker-{w}',f'{BASE}/residual-search-v1/repeat067-v1/worker-{w}/results-v1']
 cmd=f"CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 /opt/conda/bin/python {r}/code-v1/extract_fit.py extract --sources "+shlex.join(dirs)+f' --output {r}/train-extract-v1'
 output=run(w,cmd,180);print(w,output,flush=True)
 for name in ['rows.npz','DONE.json']:
  p=subprocess.check_output(ssh(w)+['cat '+r+'/train-extract-v1/'+name],timeout=60);(LOCAL/f'train-{w}-{name}').write_bytes(p)
if __name__=='__main__':
 with ThreadPoolExecutor(2)as p:list(p.map(extract,[49155754,49902461]))
