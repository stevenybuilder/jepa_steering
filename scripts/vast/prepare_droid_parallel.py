"""Prepare immutable v3 source/freeze on owned NJ CPU while GPUs keep working."""
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
CODE = 'droid-coupling-code-20260908-v3'
RELAY = REMOTE + '/droid-relay-20260908-v2'


def main():
    sys.path.insert(0, str(ROOT / 'src'))
    from offline_study.author_fit import source_hash
    source = source_hash()
    rows = json.loads(subprocess.check_output(['vastai', 'show', 'instances', '--raw'], text=True))
    row = next(x for x in rows if x['id'] == 50239185)
    if (row['actual_status'] != 'running' or row['geolocation'] != 'New Jersey, US' or
            row['label'] != 'jepa-pusht-wall-history-us-v1' or sum(x['instance']['totalHour'] for x in rows) > 7):
        raise ValueError('Owned staging endpoint/budget changed')
    ssh = ['ssh', '-i', '/tmp/jepa_vast_50123620_ed25519', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
        '-o', 'ServerAliveInterval=15', '-p', str(row['ports']['22/tcp'][0]['HostPort']), 'root@' + row['public_ipaddr']]
    subprocess.run(ssh + [f'test -f {RELAY}/VERIFIED.json && test ! -e {REMOTE}/{CODE}'], check=True)
    members = {f'{CODE}/src/offline_study/{p.name}': p for p in (ROOT / 'src/offline_study').glob('*.py')}
    for name in ('test_droid_coupling.py', 'test_droid_coupling_behavior.py', 'test_droid_fit_audit.py'):
        members[f'{CODE}/tests/{name}'] = ROOT / 'tests' / name
    members[f'{CODE}/configs/droid_assets.json'] = ROOT / 'configs/droid_assets.json'
    members[f'{CODE}/run_droid_parallel_queue.py'] = ROOT / 'scripts/vast/run_droid_parallel_queue.py'
    manifest = {name: {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
        for name, path in members.items()}
    proof = ROOT / 'artifacts/offline_study/droid-coupling-preparation-20260908-v3'
    proof.mkdir(exist_ok=False)
    (proof / 'FILES.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    with tempfile.TemporaryFile() as stream:
        with tarfile.open(fileobj=stream, mode='w:gz', format=tarfile.USTAR_FORMAT) as archive:
            for name, path in members.items():
                archive.add(path, arcname=name, recursive=False)
        stream.seek(0)
        subprocess.run(ssh + [f'tar --keep-old-files -C {REMOTE} -xzf -'], stdin=stream, check=True)
    python = RELAY + '/workspace/jepa-droid-python/bin/python'
    env = ['env', 'CUDA_VISIBLE_DEVICES=', f'PYTHONPATH={REMOTE}/{CODE}/src', 'LD_LIBRARY_PATH=/opt/conda/lib',
        'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1', python]
    check = f"""
import hashlib,json,pathlib,sys
root=pathlib.Path('{REMOTE}');manifest=json.load(sys.stdin)
for name,want in manifest.items():
 p=root/name
 if p.is_symlink() or p.stat().st_size!=want['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=want['sha256']:raise ValueError('Receiving source differs')
from offline_study.author_fit import source_hash
if source_hash()!={source!r}:raise ValueError('Source hash differs')
receipt={{'source_sha256':source_hash(),'members':len(manifest),'status':'droid_v3_source_cpu_prepared','gpu_calls':0}}
with (root/'{CODE}/RECEIVING.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
"""
    result = subprocess.check_output(ssh + [shlex.join(env + ['-c', check])], input=json.dumps(manifest), text=True)
    (proof / 'RECEIVING.json').write_text(result); print(result, flush=True)
    result = subprocess.run(ssh + [shlex.join(env + ['-m', 'unittest', 'discover', '-s', f'{REMOTE}/{CODE}/tests', '-q'])],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (proof / 'receiving-tests.log').write_text(result.stdout); print(result.stdout, flush=True); result.check_returncode()
    common = ['--vendor', '/workspace/jepa_steering/vendor/jepa-wms', '--assets', RELAY + '/workspace/jepa-runtime/droid-assets-20260907-v1',
        '--manifest', f'{REMOTE}/{CODE}/configs/droid_assets.json', '--encoder-source', '/workspace/jepa_steering/dinov3',
        '--encoder-root', RELAY + '/workspace/jepa-runtime/dinov3-native-20260907-v2',
        '--fit', RELAY + '/workspace/jepa-runtime/droid-coupling-fit-20260908-v2',
        '--reference', RELAY + '/workspace/jepa-runtime/droid-native-replication-20260907-v1/shard-all',
        '--native-engineering', RELAY + '/workspace/jepa-runtime/droid-native-engineering-20260907-v7',
        '--freeze', f'{REMOTE}/droid-coupling-behavior-20260908-v3/freeze', '--new-native-reference']
    result = subprocess.run(ssh + [shlex.join(env + ['-m', 'offline_study.droid_coupling_behavior', 'freeze'] + common)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (proof / 'freeze.log').write_text(result.stdout); print(result.stdout, flush=True); result.check_returncode()
    for name in ('protocol.json', 'FROZEN.json'):
        data = subprocess.check_output(ssh + [f'cat {REMOTE}/droid-coupling-behavior-20260908-v3/freeze/{name}'])
        (proof / ('freeze-' + name)).write_bytes(data)
    print(json.dumps({'status': 'v3_cpu_preparation_complete_no_gpu_or_new_lease', 'source_sha256': source,
        'freeze_sha256': hashlib.sha256((proof / 'freeze-protocol.json').read_bytes()).hexdigest()}), flush=True)


if __name__ == '__main__':
    main()
