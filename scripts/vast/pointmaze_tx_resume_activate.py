"""Start the CPU preparation/waiter only after complete immutable staging."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from pointmaze_tx_resume_stage import KEY, LABEL, REMOTE, ROOT


def main():
    proof = ROOT / 'artifacts/offline_study' / LABEL
    receipt = json.loads((proof / 'STAGED.json').read_text())
    if receipt['target'] != 50259194 or receipt['status'] != 'pointmaze_cpu_relocation_bytes_verified_not_gpu_clearance':
        raise ValueError('Staging is incomplete')
    rows = json.loads(subprocess.check_output(['vastai', 'show', 'instances', '--raw'], text=True))
    target = next(row for row in rows if row['id'] == 50259194)
    if (target['label'] != 'jepa-droid-parallel-us-v3' or target['geolocation'] != 'Texas, US' or
            target['actual_status'] != 'running' or sum(row['instance']['totalHour'] for row in rows) > 7):
        raise ValueError('Explicit owned US worker or budget changed')
    ssh = ['ssh', '-i', KEY, '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
        '-p', str(target['ports']['22/tcp'][0]['HostPort']), 'root@' + target['public_ipaddr']]
    script = ROOT / 'scripts/vast/pointmaze_tx_resume_worker.py'
    remote_script = REMOTE + '/pointmaze_tx_resume_worker.py'
    sha = hashlib.sha256(script.read_bytes()).hexdigest()
    upload = 'import pathlib,sys; pathlib.Path(' + repr(remote_script) + ').open("xb").write(sys.stdin.buffer.read())'
    subprocess.run(ssh + [shlex.join(['python', '-c', upload])], input=script.read_bytes(), check=True)
    launch = '''import hashlib,json,os,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]); script=root/'pointmaze_tx_resume_worker.py'
if hashlib.sha256(script.read_bytes()).hexdigest()!=sys.argv[2]:raise ValueError('Receiving worker code changed')
if (root/'LAUNCH.json').exists() or (root/'receiving').exists() or (root/'queue').exists():raise ValueError('Existing attempt; do not overwrite/restart')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JEPA_VERIFIED_LOCAL_DINO='1',PYTHONPATH='/workspace/jepa-runtime/pointmaze-history-code-20260908-v1/src:/workspace/jepa-python/lib/python3.10/site-packages',LD_LIBRARY_PATH='/opt/conda/lib',LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libcuda.so.1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (root/'worker.log').open('x') as log:
 p=subprocess.Popen(['/workspace/jepa-planning-python/bin/python','-u',str(script),'prepare-and-wait'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'pid':p.pid,'instance':50259194,'worker_sha256':sys.argv[2],'cpu_preparation_started':True,'gpu_job_started':False,'predecessor_droid_gpu0_queue':1296,'output':str(root)}
with (root/'LAUNCH.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
'''
    result = subprocess.check_output(ssh + [shlex.join(['python', '-c', launch, REMOTE, sha])], text=True)
    (proof / 'LAUNCH.json').write_text(result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
