"""Fresh US worker slices of the frozen component panel. Never auto-buys credit."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import time
import urllib.request

import component_extension_fleet as fleet

ATTEMPT = int(os.environ.get('JEPA_COMPONENT_RENTAL_ATTEMPT', '1'))
ROOT = fleet.BASE / ('rentals' if ATTEMPT == 1 else f'rentals-v{ATTEMPT}')
GPU_PRIORITY = {'RTX 5090': 0, 'RTX 4090': 0, 'RTX PRO 6000 WS': 1, 'RTX 5000Ada': 2}


def choose_offer(offers, requested, remaining_hourly, available_slots=tuple(range(8))):
    """Admit compatible US devices, not an arbitrary one-dollar/model cutoff.

    Keep strict-FP32 science unchanged. These priorities are capacity policy,
    not measured performance rankings. Full receiving timings remain required.
    """
    suitable = [r for r in offers if r.get('gpu_name') in GPU_PRIORITY and
        r.get('num_gpus') in (1, 2, 4) and r['num_gpus'] <= len(available_slots) and
        r.get('geolocation', '').endswith(', US') and
        r.get('reliability2', 0) > .95 and int(r['driver_version'].split('.')[0]) >= 570 and
        r.get('cpu_cores_effective', 0) >= 4 * r['num_gpus'] and
        r.get('cpu_ram', 0) >= 24000 * r['num_gpus'] and
        r['dph_base'] + 80 * r['storage_cost'] / 720 < min(2.25 * r['num_gpus'], remaining_hourly)]
    if requested is not None:
        suitable = [r for r in suitable if r['id'] == requested]
    if not suitable:
        raise RuntimeError('No compatible currently offered US device fits the remaining fleet budget')
    return min(suitable, key=lambda r: (GPU_PRIORITY[r['gpu_name']],
        -r['num_gpus'], (r['dph_base'] + 80 * r['storage_cost'] / 720) / r['num_gpus']))


def claimed_slots(base):
    """Keep launched streams claimed even after stop; partial RNG streams need audited recovery."""
    slots = set()
    continuation_streams = []
    for path in base.glob('rentals*/slot-*/RESERVATION.json'):
        directory = path.parent
        stopped = (directory / 'STOP.json').exists()
        launched = (directory / 'LAUNCH.json').exists()
        create = directory / 'CREATE.json'
        rejected = create.exists() and not fleet.read(create).get('success')
        if (stopped and not launched) or rejected:
            continue
        record = fleet.read(path)
        assigned = record.get('logical_slots', list(range(record['gpu_offset'],
            record['gpu_offset'] + record['gpus'])))
        if slots.intersection(assigned):
            raise ValueError('Conflicting existing slot ownership; reconcile before renting')
        slots.update(assigned)
        continuation = directory / 'CONTINUATION.json'
        if continuation.exists():
            continuation_streams.extend(tuple(item) for item in fleet.read(continuation)['extra'])
    if len(continuation_streams) != len(set(continuation_streams)):
        raise ValueError('Duplicate continuation stream reservations')
    extra_slots = {rank for task, rank in continuation_streams}
    if slots.intersection(extra_slots):
        raise ValueError('Continuation overlaps an original worker assignment')
    slots.update(extra_slots)
    return slots


def reserved_hourly(rows, base):
    """Include pending contracts not yet visible (and unaccounted portions of visible ones)."""
    total = fleet.account_hourly(rows)
    visible = {r['id']: r for r in rows}
    for path in base.glob('rentals*/slot-*/LEASE.json'):
        if (path.parent / 'STOP.json').exists():
            continue
        item = fleet.read(path)
        row = visible.get(item['instance'])
        accounted = fleet.account_hourly([row]) if row else 0.
        total += max(0., item['hourly_usd'] - accounted)
    return total


@contextmanager
def rental_lock():
    # Serialize check+reservation+create, not host staging; prevents concurrent overspend.
    with (fleet.BASE / 'RENTAL.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def guard_label(slot):
    return f'com.steven.jepa.components.rental.{slot}' + ('' if ATTEMPT == 1 else f'.v{ATTEMPT}')


def reservation(slot):
    return ROOT / f'slot-{slot:02d}'


def rent(a):
    with rental_lock():
        return rent_locked(a)


def rent_locked(a):
    assert '2026-09-11 fresh component-worker reservation' in Path('/Users/stevenyang/Documents/GPU_RESOURCE_BOARD.md').read_text()
    rows = fleet.provider('show', 'instances')
    available = sorted(set(range(8)) - claimed_slots(fleet.BASE))
    if a.slot not in available:
        raise ValueError('Requested logical slot already claimed; never duplicate a paired stream')
    offers = []
    for model in GPU_PRIORITY:
        query = ('geolocation=US rentable=True rented=False verified=True num_gpus<=4 '
            'gpu_ram>=23 inet_down>=300 disk_space>=80 gpu_name=' + model.replace(' ', '_'))
        offers.extend(fleet.provider('search', 'offers', query, '--order', 'dph_total', '--limit', '80'))
    rejected = set()
    failed_machines = set()
    for path in fleet.BASE.glob('rentals*/slot-*/CREATE.json'):
        if fleet.read(path).get('error') == 'offer_no_longer_available':
            rejected.add(fleet.read(path.parent / 'RESERVATION.json')['offer'])
    for path in fleet.BASE.glob('rentals*/slot-*/STOP.json'):
        if fleet.read(path).get('reason') in ('startup_timeout', 'unstaged_worker_backstop'):
            failed_machines.add(fleet.read(path.parent / 'RESERVATION.json')['machine_id'])
    offers = [r for r in offers if r['id'] not in rejected and r['machine_id'] not in failed_machines]
    held = reserved_hourly(rows, fleet.BASE)
    offer = choose_offer(offers, a.offer, 6.9 - held, available)
    a.offer = offer['id']
    price = offer['dph_base'] + 80 * offer['storage_cost'] / 720
    assert held + price < 6.9  # Additional ten-cent rate margin; bandwidth is usage-based.
    assert float(fleet.provider('show', 'user')['credit']) > 5
    local = reservation(a.slot)
    local.mkdir(parents=True, exist_ok=False)
    label = f'jepa-mw-components-0911-slot{a.slot}-v{ATTEMPT}'
    slots = [a.slot] + [s for s in available if s != a.slot][:offer['num_gpus'] - 1]
    lease = {'offer': a.offer, 'machine_id': offer['machine_id'], 'label': label,
        'geolocation': offer['geolocation'], 'hourly_usd': price, 'gpus': offer['num_gpus'],
        'logical_slots': slots, 'account_hourly_before_rental': held,
        'gpu_model': offer['gpu_name'], 'receiving_timing_required': True,
        'gpu_offset': a.slot, 'total_gpus': 8, 'deadline': time.time() + 24 * 3600,
        'backstop_rationale': 'Operational_ceiling_not_ETA_538s_episode_requires16h_for108_assigned_evaluations;stop_on_verified_completion',
        'credit_reserve': 1.5, 'hourly_cap': 7, 'owner': 'rep_geometry_transcoder/root',
        'remote_root': fleet.REMOTE, 'old_confirmation_restart': False}
    fleet.write(local / 'RESERVATION.json', lease)
    subprocess.run(['launchctl', 'submit', '-l', guard_label(a.slot), '--',
        '/usr/bin/env', 'PATH=/usr/local/bin:/usr/bin:/bin:/Users/stevenyang/.local/bin',
        f'JEPA_COMPONENT_RENTAL_ATTEMPT={ATTEMPT}', fleet.PYTHON,
        str(Path(__file__).resolve()), 'guard', '--slot', str(a.slot)], check=True)
    result = fleet.provider('create', 'instance', str(a.offer), '--image',
        'pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime', '--disk', '80', '--label', label,
        '--ssh', '--direct', '--cancel-unavail')
    fleet.write(local / 'CREATE.json', result)
    assert result.get('success') and result.get('new_contract')
    lease['instance'] = result['new_contract']
    if not (local / 'LEASE.json').exists():
        fleet.write(local / 'LEASE.json', lease)
    fleet.write(local / 'START.json', {'time': time.time(), 'response': result})
    print(json.dumps({'instance': lease['instance'], 'slot': a.slot, 'hourly_usd': price, 'place': lease['geolocation']}), flush=True)


def guard(a):
    local = reservation(a.slot)
    original = fleet.read(local / 'RESERVATION.json')
    while not (local / 'LEASE.json').exists():
        # Recover a contract if the client vanished immediately after creation.
        matches = [r for r in fleet.provider('show', 'instances') if r.get('label') == original['label']]
        if matches:
            assert len(matches) == 1 and matches[0]['geolocation'] == original['geolocation']
            try:
                fleet.write(local / 'LEASE.json', {**original, 'instance': matches[0]['id']})
            except FileExistsError:
                pass
            break
        if (local / 'CREATE.json').exists() and not fleet.read(local / 'CREATE.json').get('success'):
            subprocess.run(['launchctl', 'remove', guard_label(a.slot)], check=False)
            return
        time.sleep(10)
    lease = fleet.read(local / 'LEASE.json')
    fleet.BASE, fleet.INSTANCE, fleet.LABEL, fleet.PLACE = local, lease['instance'], lease['label'], lease['geolocation']
    fleet.GUARD = guard_label(a.slot)
    fleet.guard()


def checkpoint_url():
    values = {}
    for line in (fleet.PROJECT / '.env').read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            k, v = line.split('=', 1)
            values[k.strip()] = v.strip().strip('\"').strip("'")
    token = next(values[k] for k in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HUGGINGFACE_TOKEN') if values.get(k))
    class Redirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            out = super().redirect_request(req, fp, code, msg, headers, newurl)
            if out is not None:
                out.remove_header('Authorization')
            return out
    url = 'https://huggingface.co/facebook/jepa-wms/resolve/9b9c41ef249466630dbf1a20e78391865d07b3b9/jepa_wm_metaworld.pth.tar?download=1&check=' + str(int(time.time()))
    with urllib.request.build_opener(Redirect).open(urllib.request.Request(url, method='HEAD',
            headers={'Authorization': 'Bearer ' + token}), timeout=45) as response:
        return response.url


def stage(a):
    local = reservation(a.slot)
    lease = fleet.read(local / 'LEASE.json')
    row = next(r for r in fleet.provider('show', 'instances') if r['id'] == lease['instance'])
    assert row['label'] == lease['label'] and row['geolocation'] == lease['geolocation']
    assert row['actual_status'] == 'running' and row['cur_state'] == 'running'
    ssh = fleet.connection(row)
    subprocess.run(ssh + ['test ! -e ' + fleet.REMOTE + ' && mkdir ' + fleet.REMOTE], check=True, timeout=30)
    with fleet.bundle().open('rb') as f:
        subprocess.run(ssh + ['tar --keep-old-files --no-same-owner -xz -C ' + fleet.REMOTE],
            stdin=f, check=True, timeout=240)
    subprocess.run(ssh + ['mkdir ' + fleet.REMOTE + '/ops'], check=True, timeout=30)
    names = ('component_extension_queue.py', 'component_rental_bootstrap.sh', 'component_rental_prepare.py', 'component_resume_check.py')
    with subprocess.Popen(ssh + ['tar --keep-old-files --no-same-owner -x -C ' + fleet.REMOTE + '/ops'], stdin=subprocess.PIPE) as remote:
        with tarfile.open(fileobj=remote.stdin, mode='w|') as tar:
            for name in names:
                tar.add(fleet.PROJECT / 'scripts/vast' / name, arcname=name, recursive=False)
        remote.stdin.close()
        assert remote.wait() == 0
    payload = {**lease, 'checkpoint_url': checkpoint_url()}
    script = """import json,os,sys
from pathlib import Path
r=Path('/workspace/metaworld-components-20260911-v1');payload=json.load(sys.stdin)
pid=os.fork()
if pid:
 with (r/'LAUNCH.json').open('x') as f:json.dump({'pid':pid},f)
 print(json.dumps({'pid':pid,'old_confirmation_restarted':False}),flush=True)
else:
 os.setsid()
 with (r/'supervisor.log').open('x') as log:
  os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
 sys.path.insert(0,str(r/'ops'))
 from component_rental_prepare import main
 main(payload)
"""
    raw = subprocess.check_output(ssh + [shlex.join(['python3', '-u', '-c', script])],
        text=True, input=json.dumps(payload), timeout=60)
    fleet.write(local / 'LAUNCH.json', {'instance': lease['instance'], 'ssh': ssh, 'response': json.loads(raw)})
    print(json.dumps({'instance': lease['instance'], 'slot': a.slot, 'preparation': json.loads(raw)}), flush=True)


def wait_stage(a):
    lease = fleet.read(reservation(a.slot) / 'LEASE.json')
    deadline = time.time() + 600
    while time.time() < deadline:
        row = next(r for r in fleet.provider('show', 'instances') if r['id'] == lease['instance'])
        assert row['label'] == lease['label'] and row['cur_state'] != 'stopped'
        if row['actual_status'] == 'running' and (row.get('ports') or {}).get('22/tcp'):
            ssh = fleet.connection(row)
            if subprocess.run(ssh + ['true'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=25).returncode == 0:
                stage(a)
                return
        time.sleep(10)
    raise RuntimeError('Image/SSH did not become ready within ten minutes; watchdog retains disk')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('rent', 'guard', 'stage', 'wait_stage'))
    p.add_argument('--slot', type=int, required=True, choices=range(8))
    p.add_argument('--offer', type=int)
    args = p.parse_args()
    globals()[args.mode](args)
