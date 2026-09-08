"""Stage immutable CPU-validated DROID code and bounded owned-GPU handoff."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
REMOTE = '/workspace/jepa-runtime'
CODE = 'droid-coupling-code-20260908-v2'


def main():
    sys.path.insert(0, str(ROOT / 'src'))
    from offline_study.author_fit import source_hash
    source = source_hash()
    workers = json.loads(subprocess.check_output(['vastai', 'show', 'instances', '--raw'], text=True))
    worker = next(r for r in workers if r['id'] == 50189244)
    if (worker['actual_status'] != 'running' or worker['geolocation'] != 'Virginia, US' or
            worker['label'] != 'jepa-us-droid-history-20260907' or sum(r['instance']['totalHour'] for r in workers) > 7):
        raise ValueError('Owned US worker/budget changed')
    ssh = ['ssh', '-i', '/tmp/jepa_vast_50123620_ed25519', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
        '-o', 'ServerAliveInterval=15', '-p', str(worker['ports']['22/tcp'][0]['HostPort']), 'root@' + worker['public_ipaddr']]
    subprocess.run(ssh + [f'test ! -e {REMOTE}/{CODE} && test ! -e {REMOTE}/droid-coupling-behavior-20260908-v2 && '
        f'test -f {REMOTE}/droid-fit-audit-20260908-v2/DONE.json'], check=True)
    members = {f'{CODE}/src/offline_study/{p.name}': p for p in (ROOT / 'src/offline_study').glob('*.py')}
    for name in ('test_droid_coupling.py', 'test_droid_coupling_behavior.py', 'test_droid_fit_audit.py'):
        members[f'{CODE}/tests/{name}'] = ROOT / 'tests' / name
    members[f'{CODE}/configs/droid_assets.json'] = ROOT / 'configs/droid_assets.json'
    members[f'{CODE}/run_droid_coupling_handoff.py'] = ROOT / 'scripts/vast/run_droid_coupling_handoff.py'
    manifest = {name: {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
                for name, path in members.items()}
    proof = ROOT / 'artifacts/offline_study/droid-coupling-preparation-20260908-v2'
    proof.mkdir(exist_ok=False)
    (proof / 'FILES.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    with tempfile.TemporaryFile() as stream:
        with tarfile.open(fileobj=stream, mode='w:gz', format=tarfile.USTAR_FORMAT) as archive:
            for name, path in members.items():
                archive.add(path, arcname=name, recursive=False)
        stream.seek(0)
        subprocess.run(ssh + [f'tar --keep-old-files -C {REMOTE} -xzf -'], stdin=stream, check=True)
    check = f"""
import hashlib,json,pathlib,sys
root=pathlib.Path('{REMOTE}');manifest=json.load(sys.stdin)
for name,want in manifest.items():
 p=root/name
 if p.is_symlink() or p.stat().st_size!=want['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=want['sha256']:raise ValueError('Changed receiving member: '+name)
sys.path.insert(0,'{REMOTE}/{CODE}/src')
from offline_study.author_fit import source_hash
if source_hash()!={source!r}:raise ValueError('Changed DROID receiving source')
receipt={{'source_sha256':source_hash(),'members':len(manifest),'status':'droid_receiving_source_verified','gpu_calls':0}}
with (root/'{CODE}/RECEIVING.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
"""
    env = ['env', 'CUDA_VISIBLE_DEVICES=', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
        'LD_LIBRARY_PATH=/opt/conda/lib', f'PYTHONPATH={REMOTE}/{CODE}/src',
        '/workspace/jepa-droid-python/bin/python']
    result = subprocess.check_output(ssh + [shlex.join(env + ['-c', check])], input=json.dumps(manifest), text=True)
    (proof / 'RECEIVING.json').write_text(result); print(result, flush=True)
    tests = subprocess.run(ssh + [shlex.join(env + ['-m', 'unittest', 'discover', '-s', f'{REMOTE}/{CODE}/tests', '-q'])],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
    (proof / 'receiving-tests.log').write_text(tests.stdout); print(tests.stdout, flush=True)
    preflight = subprocess.run(ssh + [shlex.join(env + ['-u', '-m', 'offline_study.droid_runtime_check',
        '--vendor', '/workspace/jepa_steering/vendor/jepa-wms', '--encoder-source', '/workspace/jepa_steering/dinov3',
        '--audit', f'{REMOTE}/droid-fit-audit-20260908-v2', '--output', f'{REMOTE}/{CODE}/runtime-check'])],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (proof / 'runtime-check.log').write_text(preflight.stdout); print(preflight.stdout, flush=True)
    preflight.check_returncode()
    launch = f"""
import pathlib,subprocess,json
with pathlib.Path('{REMOTE}/{CODE}/handoff.log').open('x') as log:
 p=subprocess.Popen({env + ['-u', f'{REMOTE}/{CODE}/run_droid_coupling_handoff.py']!r},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({{'status':'droid_checkpoint_handoff_waiter_launched','instance':50189244,'pid':p.pid,'source_sha256':{source!r},'gpu_job_running_yet':False}}))
"""
    result = subprocess.check_output(ssh + [shlex.join(['python', '-c', launch])], text=True)
    (proof / 'LAUNCH.json').write_text(result); print(result, flush=True)


if __name__ == '__main__':
    main()
