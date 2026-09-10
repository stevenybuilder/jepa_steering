"""CPU-only stage/test first; --activate is a separate, explicitly reviewed operation."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile
import time

import routing_priority_common as c

PROJECT = Path(__file__).resolve().parents[2]
PROOF = PROJECT / 'artifacts/offline_study/routing-priority-preparation-20260908-v3'


def authority():
    board = Path('/Users/stevenyang/Documents/GPU_RESOURCE_BOARD.md').read_text()
    if '50233992' not in board or 'rep_geometry_transcoder/root' not in board or 'routing-priority-20260908-v3' not in board:
        raise ValueError('Explicit shared-board preparation reservation required')
    rows = json.loads(subprocess.check_output(['/Users/stevenyang/.local/bin/vastai', 'show', 'instances', '--raw'], text=True))
    worker = next(row for row in rows if row['id'] == 50233992)
    total = sum(row['instance']['totalHour'] for row in rows)
    if (worker['actual_status'] != 'running' or worker['intended_status'] != 'running' or
            worker['label'] != 'jepa-fixed-behavior-us-v1' or not worker['geolocation'].endswith(', US') or total > 7):
        raise ValueError('Owned US running worker or aggregate budget changed')
    ssh = ['ssh', '-i', '/tmp/jepa_vast_50123620_ed25519', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
        '-p', str(worker['ports']['22/tcp'][0]['HostPort']), 'root@' + worker['public_ipaddr']]
    return ssh, {'instance': 50233992, 'owner': 'rep_geometry_transcoder/root',
        'actual_status': worker['actual_status'], 'intended_status': worker['intended_status'],
        'label': worker['label'], 'geolocation': worker['geolocation'], 'aggregate_usd_hour': total,
        'checked_unix': time.time(), 'board_sha256': hashlib.sha256(board.encode()).hexdigest()}


def receiving_env():
    return ['env', 'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1', 'OMP_NUM_THREADS=1',
        'MKL_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'LD_LIBRARY_PATH=/opt/conda/lib',
        'PYTHONPATH=' + str(c.CODE / 'src') + ':/workspace/jepa-python/lib/python3.10/site-packages', c.PYTHON]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--activate', action='store_true')
    args = parser.parse_args()
    ssh, authorized = authority()
    if args.activate:
        if not (PROOF / 'DONE.json').exists():
            raise ValueError('Complete receiving CPU preparation required')
        script = "import json,pathlib,sys; p=pathlib.Path('/workspace/jepa-runtime/routing-priority-20260908-v3/AUTHORIZATION.json'); p.open('x').write(json.dumps(json.load(sys.stdin),indent=2)+'\\n')"
        subprocess.run(ssh + [shlex.join([c.PYTHON, '-c', script])], input=json.dumps(authorized), text=True, check=True)
        result = subprocess.check_output(ssh + [shlex.join(receiving_env() + ['-u', str(c.CONTROL / 'routing_priority_control.py'), 'activate'])], text=True)
        c.write(PROOF / 'ACTIVATED.json', {'output': result, 'authority': authorized})
        print(result)
        return
    PROOF.mkdir(exist_ok=False)
    members = {p.name: p for p in (PROJECT / 'scripts/vast').glob('routing_priority_*.py')}
    members['test_routing_priority.py'] = PROJECT / 'tests/test_routing_priority.py'
    manifest = {name: {'sha256': c.digest(path), 'bytes': path.stat().st_size} for name, path in members.items()}
    c.write(PROOF / 'FILES.json', manifest)
    subprocess.run(ssh + [f'test ! -e {c.CONTROL} && mkdir {c.CONTROL}'], check=True)
    with tempfile.TemporaryFile() as stream:
        with tarfile.open(fileobj=stream, mode='w:gz') as archive:
            for name, path in members.items():
                archive.add(path, arcname=name, recursive=False)
            archive.add(PROOF / 'FILES.json', arcname='FILES.json', recursive=False)
        stream.seek(0)
        subprocess.run(ssh + [f'tar --keep-old-files -C {c.CONTROL} -xzf -'], stdin=stream, check=True)
    verify = "import hashlib,json,pathlib; r=pathlib.Path('/workspace/jepa-runtime/routing-priority-20260908-v3'); m=json.loads((r/'FILES.json').read_text()); assert all(not (r/n).is_symlink() and (r/n).stat().st_size==v['bytes'] and hashlib.sha256((r/n).read_bytes()).hexdigest()==v['sha256'] for n,v in m.items()); print(json.dumps({'receiving_members_verified':len(m)}))"
    result = subprocess.check_output(ssh + [shlex.join(receiving_env() + ['-c', verify])], text=True)
    c.write(PROOF / 'RECEIVING.json', json.loads(result[result.index('{'):]))
    command = receiving_env() + ['-m', 'unittest', 'discover', '-s', str(c.CONTROL), '-p', 'test_routing_priority.py', '-v']
    tests = subprocess.run(ssh + [shlex.join(command)], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with (PROOF / 'receiving-tests.log').open('x') as stream:
        stream.write(tests.stdout)
    print(tests.stdout, flush=True)
    prepared = subprocess.check_output(ssh + [shlex.join(receiving_env() + ['-u', str(c.CONTROL / 'routing_priority_control.py'), 'prepare'])], text=True)
    value = subprocess.check_output(ssh + [shlex.join(['cat', str(c.CONTROL / 'PLAN.json')])], text=True)
    # SSH banners are stderr on this host; JSON parsing also rejects contamination.
    plan = json.loads(value)
    c.write(PROOF / 'PLAN.json', plan)
    c.write(PROOF / 'DONE.json', {'status': 'cpu_stage_tests_and_plan_complete_no_production_signals',
        'plan_sha256': c.digest(PROOF / 'PLAN.json'), 'source_sha256': c.SOURCE, 'freeze_sha256': c.FREEZE})
    print(prepared)


if __name__ == '__main__':
    main()
