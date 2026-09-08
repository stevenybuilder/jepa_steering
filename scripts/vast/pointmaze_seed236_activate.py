"""Deploy/test only the additive CPU waiter for the required third Maze seed."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]
LABEL = 'pointmaze-seed236-tx-20260908-v1'
REMOTE = '/workspace/jepa-runtime/' + LABEL
KEY = '/tmp/jepa_vast_50123620_ed25519'
CODE = '/workspace/jepa-runtime/pointmaze-history-code-20260908-v1'


def main():
    rows = json.loads(subprocess.check_output(['vastai', 'show', 'instances', '--raw'], text=True))
    row = next(r for r in rows if r['id'] == 50259194)
    if (row['actual_status'] != 'running' or row['label'] != 'jepa-droid-parallel-us-v3' or
            row['geolocation'] != 'Texas, US' or sum(r['instance']['totalHour'] for r in rows) > 7):
        raise ValueError('Owned US target or authorized aggregate budget changed')
    proof = ROOT / 'artifacts/offline_study' / LABEL
    proof.mkdir(exist_ok=False)
    names = ['scripts/vast/pointmaze_seed236_worker.py', 'tests/test_pointmaze_seed236_queue.py']
    payload = {name: (ROOT / name).read_text() for name in names}
    files = {name: {'bytes': len(value.encode()), 'sha256': hashlib.sha256(value.encode()).hexdigest()}
        for name, value in payload.items()}
    (proof / 'FILES.json').write_text(json.dumps(files, indent=2, sort_keys=True) + '\n')
    ssh = ['ssh', '-i', KEY, '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
        '-p', str(row['ports']['22/tcp'][0]['HostPort']), 'root@' + row['public_ipaddr']]
    stage = '''import hashlib,json,pathlib,shutil,sys
root=pathlib.Path(sys.argv[1]); payload=json.load(sys.stdin)
if root.exists() or shutil.disk_usage(root.parent).free<32*1024**3:raise ValueError('Existing attempt or insufficient reserve')
root.mkdir(); files={}
for name,value in payload.items():
 if name not in ('scripts/vast/pointmaze_seed236_worker.py','tests/test_pointmaze_seed236_queue.py'):raise ValueError('Unexpected staged member')
 p=root/name; p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:f.write(value)
 files[name]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
with (root/'FILES.json').open('x') as f:json.dump(files,f,indent=2,sort_keys=True);f.write(chr(10))
print(json.dumps(files))
'''
    actual = json.loads(subprocess.check_output(ssh + [shlex.join(['python', '-c', stage, REMOTE])],
        input=json.dumps(payload), text=True))
    if actual != files:
        raise ValueError('Receiving source changed')
    env = ['env', 'CUDA_VISIBLE_DEVICES=', 'JEPA_VERIFIED_LOCAL_DINO=1',
        'PYTHONPATH=' + CODE + '/src:/workspace/jepa-python/lib/python3.10/site-packages',
        'LD_LIBRARY_PATH=/opt/conda/lib', 'LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.1',
        'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1',
        '/workspace/jepa-planning-python/bin/python']
    for label, tests in [('handoff', REMOTE + '/tests'), ('native-pointmaze', CODE + '/tests')]:
        result = subprocess.run(ssh + [shlex.join(env + ['-m', 'unittest', 'discover', '-s', tests, '-q'])],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (proof / (label + '-tests.log')).write_text(result.stdout)
        print(result.stdout, flush=True)
        result.check_returncode()
    launch = '''import hashlib,json,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]); command=json.loads(sys.argv[2]); want=json.loads(sys.argv[3])
for name,row in want.items():
 p=root/name
 if p.stat().st_size!=row['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Source changed after tests')
if (root/'LAUNCH.json').exists() or (root/'receiving').exists():raise ValueError('Existing waiter; do not duplicate')
with (root/'worker.log').open('x') as log:
 p=subprocess.Popen(command+['-u',str(root/'scripts/vast/pointmaze_seed236_worker.py'),'prepare-and-wait'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'pid':p.pid,'instance':50259194,'gpu':1,'seed':236,'predecessor_pid':1297,'source_files':want,'gpu_started':False,'cpu_waiter_started':True}
with (root/'LAUNCH.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
'''
    result = subprocess.check_output(ssh + [shlex.join(['python', '-c', launch, REMOTE, json.dumps(env), json.dumps(files)])], text=True)
    (proof / 'LAUNCH.json').write_text(result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
