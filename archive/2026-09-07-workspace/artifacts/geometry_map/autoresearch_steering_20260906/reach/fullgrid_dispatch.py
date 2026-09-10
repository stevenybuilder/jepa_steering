from dispatch import *
import time

def transfer(w,src,other,dest):
 prod=subprocess.Popen(ssh(w)+['cat '+src],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
 with subprocess.Popen(ssh(other)+['cat > '+dest],stdin=prod.stdout,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)as cons:
  prod.stdout.close();_,err=cons.communicate(timeout=240);assert not cons.returncode and not prod.wait(timeout=30),err
 s=run(w,'sha256sum '+src).split()[0];t=run(other,'sha256sum '+dest).split()[0];assert s==t;return s

def extract(w):
 r=root(w);run(w,f'mkdir -p {r}/code-v2; cp {r}/code-v1/*.py {r}/code-v2/')
 upload(w,LOCAL/'extract_fit_fullgrid.py',r+'/code-v2/extract_fit_fullgrid.py')
 dirs=[f'{BASE}/residual-search-v1/full99-pop0-v2-worker-{w}',f'{BASE}/residual-search-v1/repeat067-v1/worker-{w}/results-v1']
 print(w,run(w,f"CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 /opt/conda/bin/python {r}/code-v2/extract_fit_fullgrid.py extract --sources "+shlex.join(dirs)+f' --output {r}/train-fullgrid-v1',240),flush=True)

with ThreadPoolExecutor(2)as ex:list(ex.map(extract,[49155754,49902461]))
r=root(49155754);transfer(49902461,root(49902461)+'/train-fullgrid-v1/rows.npz',49155754,r+'/train-fullgrid-from461.npz')
print(run(49155754,f"CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 /opt/conda/bin/python {r}/code-v2/extract_fit_fullgrid.py fit --sources {r}/train-fullgrid-v1/rows.npz {r}/train-fullgrid-from461.npz --output {r}/fit-fullgrid-v1",240),flush=True)
receipt=json.loads(run(49155754,'cat '+r+'/fit-fullgrid-v1/DONE.json'));(LOCAL/'fit-fullgrid-DONE.json').write_text(json.dumps(receipt,indent=2))
p=json.loads((LOCAL/'protocol-v1.json').read_text());p.update(parent='pooled correction pilot-v1: forecast improvement8/8 but0/8 actionchanges vsbank/sham',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),operator='Full spatial256x400 output residual correction, same observed pooled features and fixed ridge760/dose.5. Native model, objective and bank retained. Explicit OUTPUT mechanistic comparison.',model='Dual ridge prediction of full256x400 physical-minus-native error from same76development-calibration examples and813observables',sham='Same signed channel permutation aspooledpilot plus fixedtokenpermutation; exact fullvisual/fullproprio norm eachcandidate',frozen_correction_sha256=receipt['parameters'],runner_sha256=hashlib.sha256((LOCAL/'pilot_fullgrid.py').read_bytes()).hexdigest(),source_code_sha256={f:hashlib.sha256((LOCAL/f).read_bytes()).hexdigest()for f in ['pilot_fullgrid.py','extract_fit_fullgrid.py']})
(LOCAL/'protocol-fullgrid-v1.json').write_text(json.dumps(p,indent=2)+'\n')
def launch(w):
 rr=root(w)
 if w==49982193:run(w,f'mkdir -p {rr}/code-v2; cp {rr}/code-v1/*.py {rr}/code-v2/')
 for f,d in [('pilot_fullgrid.py','code-v2/pilot_fullgrid.py'),('extract_fit_fullgrid.py','code-v2/extract_fit_fullgrid.py'),('protocol-fullgrid-v1.json','protocol-fullgrid-v1.json')]:upload(w,LOCAL/f,rr+'/'+d)
 if w!=49155754:transfer(49155754,r+'/fit-fullgrid-v1/correction.npz',w,rr+'/fullgrid-correction.npz')
 else:run(w,f'cp {r}/fit-fullgrid-v1/correction.npz {r}/fullgrid-correction.npz')
 assert not run(w,'nvidia-smi --query-compute-apps=pid --format=csv,noheader').strip()
 old=json.loads((LOCAL/f'launch-{w}.json').read_text());cmd=old['command'].replace('/code-v1','/code-v2').replace('/pilot.py','/pilot_fullgrid.py').replace('/correction.npz','/fullgrid-correction.npz').replace('/protocol-v1.json','/protocol-fullgrid-v1.json').replace('/pilot-v1','/pilot-fullgrid-v1')
 pid=run(w,'nohup bash -c '+shlex.quote(cmd)+f' > {rr}/pilot-fullgrid-v1.log 2>&1 < /dev/null & echo $!').strip();receipt=dict(worker=w,pid=pid,command=cmd,output=rr+'/pilot-fullgrid-v1',utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));(LOCAL/f'launch-fullgrid-{w}.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
with ThreadPoolExecutor(3)as ex:list(ex.map(launch,HOSTS))
