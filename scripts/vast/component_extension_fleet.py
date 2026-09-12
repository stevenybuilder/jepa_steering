"""One explicitly owned MetaWorld component worker, immutable inputs, retained disks.

No training, old confirmation restart, credential copies or automatic purchases.
The local watchdog is registered before activation and verifies Drive before a
routine stop. An emergency stop preserves the worker's disk for later recovery.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

PROJECT = Path(__file__).resolve().parents[2]
BASE = PROJECT / 'artifacts/offline_study/table-completion-20260911-v1/component-extension-v1'
REMOTE = '/workspace/metaworld-components-20260911-v1'
VAST = '/Users/stevenyang/.local/bin/vastai'
PYTHON = str(PROJECT / '.venv/bin/python')
GUARD = 'com.steven.jepa.components.20260911'
INSTANCE = 50544130
LABEL = 'jepa-confirmation-0911-reach-wall-0'
PLACE = 'Florida, US'


def read(p):
    return json.loads(p.read_text())


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(value, f, indent=2)


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def provider(*args):
    result = subprocess.run([VAST, *args, '--raw'], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    raw = result.stdout
    if not raw.strip():
        # CLI emits a structured sold-offer response on stderr with exit code0.
        # Record an explicit rejection, not an ambiguous JSON parse failure.
        try:
            error = json.loads(result.stderr)
        except ValueError:
            raise RuntimeError('Provider returned no parseable response; reconcile contract before retry') from None
        if args[0] == 'create' and error.get('error') and 'no_such_ask' in error.get('msg', ''):
            return {'success': False, 'error': 'offer_no_longer_available',
                'status_code': error.get('status_code')}
        raise RuntimeError('Provider request failed; reconcile before retry')
    result.check_returncode()
    if args[0] in ('start', 'stop'):
        return {'raw_response': raw.strip()}
    return json.loads(raw)


def owned(rows):
    r = next(r for r in rows if r['id'] == INSTANCE)
    if r['label'] != LABEL or r['geolocation'] != PLACE:
        raise ValueError('Worker owner/geography changed')
    return r


def account_hourly(rows):
    """Actual retained storage plus running/pending compute; never quoted stopped compute."""
    total = 0.
    for row in rows:
        quoted = float(row.get('dph_total') or 0)
        storage = float(row.get('storage_total_cost') or 0)
        actual = float((row.get('instance') or {}).get('totalHour') or 0)
        active = row.get('cur_state') != 'stopped' or row.get('intended_status') == 'running'
        total += max(actual, quoted) if active else max(actual, storage)
    return total


def connection(r):
    return ['ssh', '-i', '/Users/stevenyang/.ssh/id_ed25519', '-o', 'BatchMode=yes',
        '-o', 'ConnectTimeout=15', '-o', 'StrictHostKeyChecking=accept-new', '-p',
        str(r['ports']['22/tcp'][0]['HostPort']), 'root@' + r['public_ipaddr']]


def bundle():
    out = BASE / 'inputs-v2.tar.gz'
    if out.exists():
        assert sha(out) == read(BASE / 'INPUTS-v2.json')['sha256']
        return out
    roots = [PROJECT / 'src', PROJECT / 'vendor/jepa-wms', BASE / 'freeze',
        PROJECT / 'artifacts/offline_study/primary-durable-20260907/fits-v1/bfloat16/reach/vision_action_coupling',
        PROJECT / 'artifacts/offline_study/primary-durable-20260907/fits-v1/bfloat16/reach-wall/vision_action_coupling']
    stimuli = PROJECT / 'artifacts/offline_study/restored-behavioral-inputs-20260908-v1'
    roots += [p for p in stimuli.iterdir() if p.is_dir()]
    files = [(p, 'code/' + str(p.relative_to(PROJECT))) for root in roots for p in sorted(root.rglob('*'))
        if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and
        not p.name.startswith('._') and p.name != '.env' and
        not ('.git' in p.parts and (p.name == 'config' or 'hooks' in p.parts or 'logs' in p.parts))]
    files += [(PROJECT / p, 'code/' + p) for p in (
        'scripts/vast/component_extension_queue.py', 'docs/METAWORLD_COMPONENT_COMPLETION.md',
        'scripts/vast/component_extension_fleet.py', 'scripts/vast/component_rental_fleet.py',
        'scripts/vast/component_rental_prepare.py', 'scripts/vast/component_rental_bootstrap.sh')]
    # verify_fit checks the parent-level native parity receipt as well as the bank.
    for task in ('reach', 'reach-wall'):
        parent = PROJECT / 'artifacts/offline_study/primary-durable-20260907/fits-v1/bfloat16' / task
        files += [(p, 'code/' + str(p.relative_to(PROJECT))) for p in sorted(parent.glob('*.json'))]
    manifest = {name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p, name in files}
    assert len(manifest) == len(files)
    with tarfile.open(out, 'x:gz', compresslevel=1) as tar:
        for p, name in files:
            tar.add(p, arcname=name, recursive=False)
    write(BASE / 'INPUTS-v2.json', {'sha256': sha(out), 'bytes': out.stat().st_size, 'files': manifest})
    print(json.dumps({'input_bytes': out.stat().st_size, 'files': len(files)}), flush=True)
    return out


def activate():
    assert '2026-09-11 component-extension reservation' in Path('/Users/stevenyang/Documents/GPU_RESOURCE_BOARD.md').read_text()
    bundle()
    rows = provider('show', 'instances')
    r = owned(rows)
    assert r['cur_state'] == 'stopped' and r['actual_status'] == 'exited'
    live_cost = sum(x['dph_total'] for x in rows if x['cur_state'] != 'stopped')
    assert live_cost + r['dph_total'] < 6.8
    assert float(provider('show', 'user')['credit']) > 5
    write(BASE / 'LEASE.json', {'instance': INSTANCE, 'label': LABEL, 'geolocation': PLACE,
        'owner': 'rep_geometry_transcoder/root', 'hourly_usd': r['dph_total'], 'gpus': r['num_gpus'],
        'deadline': time.time() + 6 * 3600, 'credit_reserve': 1.5, 'hourly_cap': 7,
        'old_confirmation_restart': False, 'remote_root': REMOTE})
    subprocess.run(['launchctl', 'submit', '-l', GUARD, '--', '/usr/bin/env',
        'PATH=/usr/local/bin:/usr/bin:/bin:/Users/stevenyang/.local/bin', PYTHON,
        str(Path(__file__).resolve()), 'guard'], check=True)
    result = provider('start', 'instance', str(INSTANCE))
    write(BASE / 'START.json', {'time': time.time(), 'response': result})
    print(json.dumps({'instance': INSTANCE, 'hourly_usd': r['dph_total'], 'response': result}), flush=True)


def stage():
    lease = read(BASE / 'LEASE.json')
    r = owned(provider('show', 'instances'))
    assert r['actual_status'] == 'running' and r['cur_state'] == 'running'
    ssh = connection(r)
    assert not (BASE / 'LAUNCH.json').exists()
    subprocess.run(ssh + ['test ! -e ' + REMOTE + ' && mkdir ' + REMOTE], check=True, timeout=30)
    with bundle().open('rb') as f:
        subprocess.run(ssh + ['tar --keep-old-files --no-same-owner -xz -C ' + REMOTE], stdin=f, check=True, timeout=180)
    code = """import json,os,subprocess,sys
from pathlib import Path
r=Path(sys.argv[1]);deadline=sys.argv[2];gpus=sys.argv[3]
with (r/'supervisor.log').open('x') as log:
 p=subprocess.Popen(['/workspace/decision-python/bin/python','-u',str(r/'code/scripts/vast/component_extension_queue.py'),'--root',str(r),'--deadline',deadline,'--gpus',gpus],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 receipt={'pid':p.pid,'old_confirmation_restarted':False}
 with (r/'LAUNCH.json').open('x') as f:json.dump(receipt,f)
 print(json.dumps(receipt))
"""
    raw = subprocess.check_output(ssh + [shlex.join(['python3', '-c', code, REMOTE,
        str(lease['deadline']), str(lease['gpus'])])], text=True, timeout=45)
    write(BASE / 'LAUNCH.json', {'instance': INSTANCE, 'ssh': ssh, 'response': json.loads(raw)})
    print(raw, flush=True)


def collect(r):
    from final_preservation_drive import request, FOLDER, upload_direct, verify
    with request(FOLDER, '?fields=id,name,mimeType,trashed') as f:
        folder = json.load(f)
    assert folder['id'] == FOLDER and folder['mimeType'] == 'application/vnd.google-apps.folder' and not folder.get('trashed')
    ssh = connection(r)
    # Called only after the remote supervisor terminal marker: every child has exited.
    command = ('test -f ' + REMOTE + '/TERMINAL.json && tar --exclude=' +
        Path(REMOTE).name + '/checkpoints -czf - -C /workspace ' + Path(REMOTE).name)
    local = BASE / 'complete-worker.tar.gz'
    if not local.exists():
        partial = BASE / 'complete-worker.tar.gz.partial'
        with partial.open('wb') as f:
            subprocess.run(ssh + [command], stdout=f, check=True, timeout=300)
        with tarfile.open(partial, 'r:gz') as tar:
            assert any(m.name.endswith('/TERMINAL.json') for m in tar)
        partial.rename(local)
    proof = {'sha256': sha(local), 'bytes': local.stat().st_size, 'archive': str(local)}
    metadata = upload_direct(local, f'jepa-metaworld-components-20260911-v1-{INSTANCE}.tar.gz', BASE / 'UPLOAD.json')
    verified = verify(metadata['id'], metadata['name'], proof['bytes'], proof['sha256'])
    if not (BASE / 'DRIVE_VERIFIED.json').exists():
        write(BASE / 'DRIVE_VERIFIED.json', {**proof, **verified, 'drive_file_id': metadata['id']})


def guard():
    while True:
        try:
            lease = read(BASE / 'LEASE.json')
            rows = provider('show', 'instances')
            r = owned(rows)
            if (BASE / 'START.json').exists() and r['cur_state'] == 'stopped':
                break
            reason = None
            if time.time() > lease['deadline']:
                reason = 'registered_deadline_backstop'
            elif float(provider('show', 'user')['credit']) < lease['credit_reserve']:
                reason = 'credit_reserve'
            elif account_hourly(rows) > lease['hourly_cap']:
                reason = 'fleet_cap'
            elif (BASE / 'START.json').exists() and r['actual_status'] == 'scheduling' and time.time() - read(BASE / 'START.json')['time'] > 60:
                reason = 'gpu_unavailable_cancel_start'
            elif (BASE / 'START.json').exists() and not (BASE / 'LAUNCH.json').exists() and time.time() - read(BASE / 'START.json')['time'] > 900:
                reason = 'unstaged_worker_backstop'
            elif (BASE / 'LAUNCH.json').exists() and r['actual_status'] == 'running':
                ssh = connection(r)
                done = subprocess.run(ssh + ['test -f ' + REMOTE + '/TERMINAL.json'],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30).returncode == 0
                handoff_pending = (BASE / 'HANDOFF_ACTIVE.json').exists() and not (BASE / 'HANDOFF_DONE.json').exists()
                if handoff_pending:
                    state_script = "from pathlib import Path;import json;p=Path('/workspace/metaworld-components-20260911-v1/ops/handoff-v1');print(json.dumps({name:json.loads((p/name).read_text()) for name in ('HANDOFF_DONE.json','HANDOFF_FAILED.json') if (p/name).exists()}))"
                    remote_state = json.loads(subprocess.check_output(ssh + [shlex.join(['python3', '-c', state_script])], text=True, timeout=30))
                    if remote_state:
                        write(BASE / 'HANDOFF_DONE.json', {'time': time.time(), 'remote': remote_state,
                            'handoff_successful': 'HANDOFF_DONE.json' in remote_state})
                        handoff_pending = False
                # A registered continuation owns previously unassigned streams.
                # Do not release this host at the old queue's partial-panel terminal.
                continuation = BASE / 'CONTINUATION.json'
                if done and continuation.exists():
                    tail = read(continuation)['remote_root']
                    if tail != REMOTE + '/ops/continuation-v1':
                        raise ValueError('Unexpected continuation root')
                    done = subprocess.run(ssh + ['test -f ' + tail + '/TERMINAL.json'],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30).returncode == 0
                if done and not handoff_pending:
                    collect(r)
                    reason = 'terminal_outputs_full_drive_readback_verified'
            if reason:
                response = provider('stop', 'instance', str(INSTANCE))
                write(BASE / 'STOP.json', {'reason': reason, 'response': response,
                    'source_disk_retained': True, 'time': time.time()})
                break
        except Exception as e:
            print(json.dumps({'guard_error': type(e).__name__}), flush=True)
        time.sleep(30)
    if not (BASE / 'GUARD_DONE.json').exists():
        write(BASE / 'GUARD_DONE.json', {'source_disk_retained': True, 'time': time.time()})
    subprocess.run(['launchctl', 'remove', GUARD], check=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('bundle', 'activate', 'stage', 'guard'))
    globals()[parser.parse_args().mode]()
