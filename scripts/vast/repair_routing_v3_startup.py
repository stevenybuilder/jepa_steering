"""Repair only eight pre-import CPU waiter failures; frozen source is unchanged."""
import json
from pathlib import Path
import shlex
import subprocess


def main():
    rows = json.loads(subprocess.check_output(['vastai', 'show', 'instances', '--raw'], text=True))
    worker = next(x for x in rows if x['id'] == 50233992)
    if (worker['actual_status'] != 'running' or worker['label'] != 'jepa-fixed-behavior-us-v1' or
            not worker['geolocation'].endswith(', US') or sum(x['instance']['totalHour'] for x in rows) > 7):
        raise ValueError('Owned worker/budget changed')
    ssh = ['ssh', '-i', '/tmp/jepa_vast_50123620_ed25519', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
        '-p', str(worker['ports']['22/tcp'][0]['HostPort']), 'root@' + worker['public_ipaddr']]
    code = r'''
import json,os,pathlib,signal,subprocess,time
root=pathlib.Path('/workspace/jepa-runtime');panel=root/'hmm-fixed-response-behavior-20260908-v3'
def cmd(pid):
 p=pathlib.Path(f'/proc/{pid}/cmdline')
 return p.read_bytes().rstrip(b'\0').decode().split('\0') if p.exists() else []
for gpu in range(8):
 if cmd(35090+gpu) or (panel/f'queue-gpu{gpu}').exists():raise ValueError('Old waiter not a pre-import failure')
 if "ModuleNotFoundError: No module named 'torch'" not in (panel/f'waiter-gpu{gpu}.log').read_text():raise ValueError('Different failure; do not retry')
 if not any('run_fixed_behavior_queue' in x for x in cmd(5202+gpu)):raise ValueError('Original GPU producer changed')
if cmd(35098)!=['/workspace/jepa-planning-python/bin/python','-u',str(root/'finish_routing_panel_v3.py')]:raise ValueError('Old coordinator changed')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
 LD_LIBRARY_PATH='/opt/conda/lib',PYTHONPATH=str(root/'hmm-fixed-response-code-20260908-v3/src')+':/workspace/jepa-python/lib/python3.10/site-packages')
subprocess.run(['/workspace/jepa-planning-python/bin/python','-c',"from offline_study.routing_behavior import ObservedRouting; from offline_study.author_fit import source_hash; assert source_hash()=='d6cc7a7fcdad2d36d695a43b55fd99c4f1831fb833082c5e974bca229b88d9dc'"],env=env,check=True)
os.kill(35098,signal.SIGTERM)
with (panel/'STARTUP_FAILURE_CORRECTION.json').open('x') as f:json.dump({'failure':'missing Python3.10 torch overlay at parent startup; no HMM GPU calls','unchanged_frozen_source':True,'original_gpu_jobs_interrupted':False,'previous_coordinator_pid':35098},f,indent=2)
launched=[];children=[]
for gpu in range(8):
 with (panel/f'waiter-corrected-gpu{gpu}.log').open('x') as log:
  p=subprocess.Popen(['/workspace/jepa-planning-python/bin/python','-u',str(root/'run_routing_queue_v3.py'),'--gpu',str(gpu)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 launched.append({'gpu':gpu,'pid':p.pid});children.append(p)
deadline=time.monotonic()+30
while not all((panel/f'queue-gpu{gpu}/LAUNCH.json').exists() for gpu in range(8)):
 if any(p.poll() is not None for p in children) or time.monotonic()>deadline:raise ValueError('Corrected waiter did not reach verified waiting state')
 time.sleep(.5)
with (panel/'coordinator-corrected.log').open('x') as log:
 p=subprocess.Popen(['/workspace/jepa-planning-python/bin/python','-u',str(root/'finish_routing_panel_v3.py')],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'status':'all_eight_corrected_hmm_waiters_reached_verified_wait_state','workers':launched,'coordinator_pid':p.pid,'gpu_engineering_complete':False,'scientific_outcomes':0}
with (panel/'ACTIVATED_RUNTIME_CORRECTED.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
'''
    result = subprocess.check_output(ssh + [shlex.join(['python', '-c', code])], text=True)
    out = Path(__file__).resolve().parents[2] / 'artifacts/offline_study/hmm-fixed-response-preparation-20260908-v3/ACTIVATED_RUNTIME_CORRECTED.json'
    with out.open('x') as stream:
        stream.write(result)
    print(result)


if __name__ == '__main__':
    main()
