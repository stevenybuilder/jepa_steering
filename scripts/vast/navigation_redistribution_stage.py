"""CPU-only staging from the immutable Nebraska snapshot; never activates workers."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile
import time

import navigation_redistribution_common as c

PROJECT = Path(__file__).resolve().parents[2]
PROOF = PROJECT / 'artifacts/offline_study/navigation-redistribution-20260908-v1'
HOSTS = {50231985: ('jepa-fixed-offline-us-v1', 'Nebraska, US'),
         50205763: ('jepa-navigation-offline-indiana', 'Indiana, US'),
         50259194: ('jepa-droid-parallel-us-v3', 'Texas, US')}


def connections():
    board = Path('/Users/stevenyang/Documents/GPU_RESOURCE_BOARD.md').read_text()
    if 'navigation-redistribution-20260908-v1' not in board or 'rep_geometry_transcoder/root' not in board:
        raise ValueError('Explicit board reservation required')
    rows = json.loads(subprocess.check_output(['/Users/stevenyang/.local/bin/vastai', 'show', 'instances', '--raw'], text=True))
    total = sum(row['instance']['totalHour'] for row in rows)
    if total > 7:
        raise ValueError('Authorized aggregate cap exceeded')
    result = {}
    for number, (label, region) in HOSTS.items():
        row = next(r for r in rows if r['id'] == number)
        if row['label'] != label or row['geolocation'] != region or row['actual_status'] != 'running' or row['intended_status'] != 'running':
            raise ValueError('Explicit owned US worker changed')
        result[number] = ['ssh', '-i', '/tmp/jepa_vast_50123620_ed25519', '-o', 'BatchMode=yes',
            '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=8',
            '-p', str(row['ports']['22/tcp'][0]['HostPort']), 'root@' + row['public_ipaddr']]
    return result, {'instances': list(HOSTS), 'aggregate_usd_hour': total, 'checked_unix': time.time(),
        'board_sha256': hashlib.sha256(board.encode()).hexdigest(), 'owner': 'rep_geometry_transcoder/root', 'gpu_calls': 0}


def command(mode, *extra):
    return ['env', 'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1', '/usr/bin/python3', '-u',
            str(c.CONTROL / 'navigation_redistribution_control.py'), mode, *map(str, extra)]


def get(connection, argv):
    return subprocess.check_output(connection + [shlex.join(argv)], text=True, timeout=600)


def put_json(connection, path, value):
    # Append-only transfer, not arbitrary remote filesystem mutation.
    if Path(path).parent != c.CONTROL or Path(path).name not in ('ALL_READY.json', 'PLAN.json', 'CUTOVER.json'):
        raise ValueError('Unregistered control-document destination')
    program = 'import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.open("xb").write(sys.stdin.buffer.read())'
    data = (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()
    subprocess.run(connection + [shlex.join(['/usr/bin/python3', '-c', program, str(path)])], input=data, check=True)


def stage_operations(connection, manifest, members):
    create = 'import pathlib,sys; pathlib.Path(sys.argv[1]).mkdir(exist_ok=False)'
    subprocess.run(connection + [shlex.join(['/usr/bin/python3', '-c', create, str(c.CONTROL)])], check=True)
    with tempfile.TemporaryFile() as stream:
        with tarfile.open(fileobj=stream, mode='w:gz') as archive:
            for name, path in members.items():
                archive.add(path, arcname=name, recursive=False)
            data = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
            entry = tarfile.TarInfo('FILES.json'); entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))
        stream.seek(0)
        subprocess.run(connection + [shlex.join(['tar', '--keep-old-files', '-C', str(c.CONTROL), '-xzf', '-'])], stdin=stream, check=True)
    verify = 'import pathlib,json,sys; sys.path.insert(0,sys.argv[1]); import navigation_redistribution_common as c; c.verify_members(c.CONTROL,c.read(c.CONTROL/"FILES.json")); print(json.dumps({"operations_members_verified":len(c.read(c.CONTROL/"FILES.json"))}))'
    return json.loads(get(connection, ['/usr/bin/python3', '-c', verify, str(c.CONTROL)]))


def stage():
    ssh, authority = connections()
    PROOF.mkdir(exist_ok=False)
    c.write(PROOF / 'AUTHORITY.json', authority)
    members = {p.name: p for p in (PROJECT / 'scripts/vast').glob('navigation_redistribution_*.py')}
    members['routing_priority_common.py'] = PROJECT / 'scripts/vast/routing_priority_common.py'
    members['test_navigation_redistribution.py'] = PROJECT / 'tests/test_navigation_redistribution.py'
    manifest = {name: {'bytes': path.stat().st_size, 'sha256': c.digest(path)} for name, path in members.items()}
    c.write(PROOF / 'FILES.json', manifest)
    with ThreadPoolExecutor(max_workers=3) as pool:
        receipts = list(pool.map(lambda n: (n, stage_operations(ssh[n], manifest, members)), HOSTS))
    c.write(PROOF / 'OPERATIONS_STAGED.json', dict(receipts))
    source_manifest = json.loads(get(ssh[50231985], command('inventory-inputs')))
    c.write(PROOF / 'INPUTS.json', source_manifest)
    # One source read/download, then the same verified packet for all three hosts.
    # The file is an owned temporary transport artifact, never a scientific output.
    with tempfile.TemporaryFile() as packet:
        subprocess.run(ssh[50231985] + [shlex.join(command('export-inputs'))], stdout=packet, check=True, timeout=1200)
        packet.seek(0)
        with tarfile.open(fileobj=packet, mode='r:gz') as archive:
            embedded = json.load(archive.extractfile('INPUTS.json'))
        if embedded != source_manifest:
            raise ValueError('Source export manifest changed')
        # Dup the file descriptor is not an independent seek offset; reopen the
        # transport via /dev/fd only sequentially to avoid corrupt parallel reads.
        for number in HOSTS:
            packet.seek(0)
            subprocess.run(ssh[number] + [shlex.join(command('receive-inputs'))], stdin=packet, check=True, timeout=1200)
            print(json.dumps({'instance': number, 'payload_received_verified': True, 'gpu_calls': 0}), flush=True)
    if json.loads(get(ssh[50231985], command('inventory-inputs'))) != source_manifest:
        raise ValueError('Original frozen source changed during staging')
    all_ready = {}
    for number in HOSTS:
        test_command = ['env', 'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1',
            '/usr/bin/python3', '-m', 'unittest', 'discover', '-s', str(c.CONTROL), '-p', 'test_navigation_redistribution.py', '-v']
        tests = subprocess.check_output(ssh[number] + [shlex.join(test_command)],
            stderr=subprocess.STDOUT, text=True, timeout=600)
        with (PROOF / f'tests-{number}.log').open('x') as output:
            output.write(tests)
        result = get(ssh[number], command('prepare', '--instance', number))
        ready = json.loads(get(ssh[number], ['cat', str(c.CONTROL / 'READY.json')]))
        all_ready[str(number)] = ready
        c.write(PROOF / f'READY-{number}.json', ready)
        print(result, flush=True)
    c.write(PROOF / 'ALL_READY.json', all_ready)
    for number in HOSTS:
        put_json(ssh[number], c.CONTROL / 'ALL_READY.json', all_ready)
    c.write(PROOF / 'DONE.json', {'status': 'navigation_redistribution_cpu_staged_tested_ready_for_root_review',
        'source_sha256': c.SOURCE, 'freeze_sha256': c.FREEZES, 'files_sha256': c.digest(PROOF / 'FILES.json'),
        'all_ready_sha256': c.digest(PROOF / 'ALL_READY.json'), 'parent_signals': 0, 'gpu_calls': 0})


def publish_plan():
    """After separately authorized boundary, copy its exact partition; no launches."""
    ssh, _ = connections()
    plan = json.loads(get(ssh[50231985], ['cat', str(c.CONTROL / 'PLAN.json')]))
    cutover = json.loads(get(ssh[50231985], ['cat', str(c.CONTROL / 'CUTOVER.json')]))
    c.validate_plan(plan)
    data = (json.dumps(plan, indent=2, sort_keys=True) + '\n').encode()
    if hashlib.sha256(data).hexdigest() != cutover['plan_sha256'] or not cutover['old_parents_terminal']:
        raise ValueError('Verified ownership cutover required')
    for number in (50205763, 50259194):
        put_json(ssh[number], c.CONTROL / 'PLAN.json', plan)
        put_json(ssh[number], c.CONTROL / 'CUTOVER.json', cutover)
    c.write(PROOF / 'PLAN.json', plan); c.write(PROOF / 'CUTOVER.json', cutover)
    print(json.dumps({'plan_distributed': True, 'gpu_calls': 0, 'run_authorization_still_required': True}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('stage', 'publish-plan'))
    args = parser.parse_args()
    stage() if args.command == 'stage' else publish_plan()
