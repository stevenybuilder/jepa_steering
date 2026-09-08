"""Replace only unstarted CPU HMM waiters after verified same-pass preparation."""
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    proof = ROOT / 'artifacts/offline_study/hmm-fixed-response-preparation-20260908-v3'
    done = json.loads((proof / 'DONE.json').read_text())
    if done['protocol_sha256'] != 'c6acb23c1abc275b61f6b4983732862211439cea713f401b2d13d3cf61a7f0cf':
        raise ValueError('Wrong verified preparation')
    workers = json.loads(subprocess.check_output(['vastai', 'show', 'instances', '--raw'], text=True))
    worker = next(r for r in workers if r['id'] == 50233992)
    if (worker['actual_status'] != 'running' or not worker['geolocation'].endswith(', US') or
            worker['label'] != 'jepa-fixed-behavior-us-v1' or sum(r['instance']['totalHour'] for r in workers) > 7):
        raise ValueError('Owned US worker/budget changed')
    ssh = ['ssh', '-i', '/tmp/jepa_vast_50123620_ed25519', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
        '-o', 'ServerAliveInterval=15', '-p', str(worker['ports']['22/tcp'][0]['HostPort']), 'root@' + worker['public_ipaddr']]
    script = r'''
import hashlib,json,os,pathlib,signal,subprocess,sys,time
root=pathlib.Path('/workspace/jepa-runtime');old=root/'hmm-fixed-response-behavior-20260908-v2';new=root/'hmm-fixed-response-behavior-20260908-v3'
members=json.load(sys.stdin)
for name,want in members.items():
 p=root/name
 if p.is_symlink() or p.stat().st_size!=want['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=want['sha256']:raise ValueError('Receiving member changed')
if hashlib.sha256((new/'freeze/protocol.json').read_bytes()).hexdigest()!='c6acb23c1abc275b61f6b4983732862211439cea713f401b2d13d3cf61a7f0cf':raise ValueError('Freeze changed')
def command(pid):
 p=pathlib.Path(f'/proc/{pid}/cmdline')
 return p.read_bytes().rstrip(b'\0').decode().split('\0') if p.exists() else []
expected={25776+gpu:['/workspace/jepa-planning-python/bin/python','-u',str(root/'run_routing_queue_v2.py'),'--gpu',str(gpu)] for gpu in range(8)}
expected[26098]=['/workspace/jepa-planning-python/bin/python','-u',str(root/'finish_routing_panel_v2.py')]
for pid,want in expected.items():
 if command(pid)!=want:raise ValueError('Only exact old CPU waiter identities may be replaced')
if (old/'engineering').exists() or any(old.glob('*/*/shard-*')) or any(new.glob('queue-gpu*')):raise ValueError('HMM GPU work may have started; do not replace')
for gpu in range(8):
 if not any('run_fixed_behavior_queue' in item for item in command(5202+gpu)):raise ValueError('Original GPU producer changed')
 if (old/f'queue-gpu{gpu}/PREDECESSOR_VERIFIED.json').exists():raise ValueError('Old waiter passed GPU handoff')
with (old/'SUPERSEDED_BEFORE_GPU_EXECUTION.json').open('x') as f:json.dump({'reason':'exact same-pass runtime, unchanged scientific panel','replacement':str(new),'old_cpu_pids':list(expected),'old_gpu_outcomes':0,'timestamp':time.time(),'original_gpu_jobs_interrupted':False},f,indent=2)
# Stop the old coordinator first; old waiter FAILED markers represent this
# deliberate administrative cancellation, not a model/engineering failure.
for pid in [26098,*range(25776,25784)]:os.kill(pid,signal.SIGTERM)
deadline=time.monotonic()+30
while any(command(pid) for pid in expected):
 if time.monotonic()>deadline:raise TimeoutError('Old CPU waiters did not terminate')
 time.sleep(.5)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
launched=[]
for gpu in range(8):
 with (new/f'waiter-gpu{gpu}.log').open('x') as log:
  p=subprocess.Popen(['/workspace/jepa-planning-python/bin/python','-u',str(root/'run_routing_queue_v3.py'),'--gpu',str(gpu)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 launched.append({'gpu':gpu,'pid':p.pid})
with (new/'coordinator.log').open('x') as log:
 p=subprocess.Popen(['/workspace/jepa-planning-python/bin/python','-u',str(root/'finish_routing_panel_v3.py')],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'status':'same_pass_hmm_cpu_waiters_active_original_gpu_jobs_untouched','workers':launched,'coordinator_pid':p.pid,'old_waiters_superseded':list(expected),'gpu_engineering_complete':False,'scientific_hmm_outcomes':0}
with (new/'ACTIVATED.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
'''
    result = subprocess.check_output(ssh + [shlex.join(['python', '-c', script])],
        input=(proof / 'FILES.json').read_text(), text=True)
    with (proof / 'ACTIVATED.json').open('x') as stream:
        stream.write(result)
    print(result)


if __name__ == '__main__':
    main()
