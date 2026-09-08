"""Verify the new owned worker and launch its four frozen DROID queues."""
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]
CODE = '/workspace/jepa-runtime/droid-coupling-code-20260908-v3'
PANEL = '/workspace/jepa-runtime/droid-coupling-behavior-20260908-v3'
DRIVER = '/usr/lib/x86_64-linux-gnu/libcuda.so.1'


def main():
    proof = ROOT / 'artifacts/offline_study/droid-parallel-worker-20260908-v3-retry-v2'
    receipt = json.loads((proof / 'VERIFIED.json').read_text())
    if receipt['instance'] != 50259194 or receipt['status'] != 'droid_parallel_inputs_source_staged_verified_not_gpu_clearance':
        raise ValueError('Missing complete staging proof')
    rows = json.loads(subprocess.check_output(['vastai','show','instances','--raw'],text=True))
    worker = next(row for row in rows if row['id']==50259194)
    if (worker['actual_status']!='running' or worker['geolocation']!='Texas, US' or
            worker['label']!='jepa-droid-parallel-us-v3' or sum(x['instance']['totalHour'] for x in rows)>7):
        raise ValueError('Explicit owned US worker or budget changed')
    ssh = ['ssh','-i','/tmp/jepa_vast_50123620_ed25519','-o','IdentitiesOnly=yes','-o','BatchMode=yes',
        '-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-p',str(worker['ports']['22/tcp'][0]['HostPort']),
        'root@'+worker['public_ipaddr']]
    env = ['env','CUDA_VISIBLE_DEVICES=',f'PYTHONPATH={CODE}/src','LD_LIBRARY_PATH=/opt/conda/lib',
        f'LD_PRELOAD={DRIVER}','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',
        '/workspace/jepa-droid-python/bin/python']
    result = subprocess.run(ssh+[shlex.join(env+['-m','unittest','discover','-s',CODE+'/tests','-q'])],
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (proof/'receiving-tests.log').write_text(result.stdout); print(result.stdout,flush=True);result.check_returncode()
    check = f'''
import hashlib,json,pathlib,subprocess,torch
from offline_study.author_fit import source_hash
if source_hash()!={receipt['source_sha256']!r}:raise ValueError('Receiving source changed')
freeze=pathlib.Path('{PANEL}/freeze/protocol.json')
if hashlib.sha256(freeze.read_bytes()).hexdigest()!={receipt['freeze_sha256']!r}:raise ValueError('Receiving freeze changed')
if hashlib.sha256(pathlib.Path('{DRIVER}').read_bytes()).hexdigest()!='b7759e577409c86949f455f18fcc044d72b466be5e2982b6ba9e20348e86fa97':raise ValueError('Expected host driver library changed')
uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).split()
if len(uuids)!=4 or len(set(uuids))!=4:raise ValueError('Wrong four-device allocation')
if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip():raise ValueError('GPU already occupied')
if torch.cuda.device_count()!=4:raise ValueError('Driver/runtime does not expose all four GPUs')
for gpu in range(4):
 a=torch.ones((128,128),device='cuda:'+str(gpu)); b=a@a
 torch.cuda.synchronize(gpu)
 if float(b[0,0])!=128:raise ValueError('Receiving CUDA arithmetic failed')
 del a,b
 torch.cuda.empty_cache()
metadata={{'purpose':'four_gpu_paired_droid_v3','instance':50259194,'gpu_uuids':uuids,
 'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'driver_library':'{DRIVER}',
 'driver_library_sha256':'b7759e577409c86949f455f18fcc044d72b466be5e2982b6ba9e20348e86fa97',
 'required_environment':{{'LD_PRELOAD':'{DRIVER}'}},'all_four_basic_cuda_checks_passed':True,
 'source_sha256':source_hash(),'full_planner_engineering_complete':False}}
with pathlib.Path('{CODE}/WORKER.json').open('x') as f:json.dump(metadata,f,indent=2)
print(json.dumps(metadata))
'''
    # Expose all four only for this finite receiving check; no model job shares them.
    checked_env = ['CUDA_VISIBLE_DEVICES=0,1,2,3' if item=='CUDA_VISIBLE_DEVICES=' else item for item in env]
    result = subprocess.check_output(ssh+[shlex.join(checked_env+['-c',check])],text=True)
    (proof/'WORKER.json').write_text(result); print(result,flush=True)
    launch = f'''
import json,pathlib,subprocess
panel=pathlib.Path('{PANEL}')
if list(panel.glob('queue-gpu*')) or (panel/'LAUNCH.json').exists():raise ValueError('Queues already exist')
result={{'instance':50259194,'queues':[],'source_sha256':{receipt['source_sha256']!r}}}
for gpu in range(4):
 with (panel/f'launch-gpu{{gpu}}.log').open('x') as log:
  p=subprocess.Popen({env+['-u',CODE+'/run_droid_parallel_queue.py']!r}+['--gpu',str(gpu)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 result['queues'].append({{'gpu':gpu,'pid':p.pid}})
with (panel/'coordinator.log').open('x') as log:
 p=subprocess.Popen({env+['-u',CODE+'/run_droid_parallel_queue.py','--analyze']!r},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
result['coordinator_pid']=p.pid
with (panel/'LAUNCH.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
'''
    result = subprocess.check_output(ssh+[shlex.join(['python','-c',launch])],text=True)
    (proof/'LAUNCH.json').write_text(result); print(result,flush=True)


if __name__=='__main__':main()
