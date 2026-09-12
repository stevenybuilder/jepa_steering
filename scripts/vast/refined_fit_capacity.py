"""One bounded two-GPU US fitting worker, separately guarded from behavior."""
import argparse
from contextlib import nullcontext
import fcntl
from pathlib import Path
import subprocess
import time

import component_extension_fleet as fleet
import component_rental_fleet as components

ROOT = fleet.PROJECT / 'artifacts/offline_study/table-completion-20260911-v1/refined-fit-capacity-v1'
REMOTE = '/workspace/refined-fit-capacity-20260911-v1'
LABEL = 'jepa-refined-fit-capacity-0911-v1'
GUARD = 'com.steven.jepa.refined.fit.capacity.20260911'


def rent():
    assert '2026-09-11 dedicated two-GPU fitting capacity' in Path('/Users/stevenyang/Documents/GPU_RESOURCE_BOARD.md').read_text()
    with (components.fleet.BASE / 'RENTAL.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = fleet.provider('show', 'instances')
        offers = fleet.provider('search', 'offers', 'geolocation=US num_gpus<=2 gpu_ram>=15 dph_total<0.8', '--order', 'dph_total', '--limit', '30')
        available = [r for r in offers if r['geolocation'].endswith(', US')
            and r['gpu_ram'] >= 15000 and r['reliability2'] >= .95
            and components.reserved_hourly(rows, components.fleet.BASE)
                + r['dph_base'] + 80 * r['storage_cost'] / 720 <= 6.9]
        if not available:
            raise ValueError('No currently offered US fitting capacity inside budget')
        offer = min(available, key=lambda r: r['dph_base'] + 80 * r['storage_cost'] / 720)
        price = offer['dph_base'] + 80 * offer['storage_cost'] / 720
        if (not offer['geolocation'].endswith(', US')
                or offer['gpu_ram'] < 15000 or offer['num_gpus'] not in (1, 2) or offer['reliability2'] < .95
                or components.reserved_hourly(rows, components.fleet.BASE) + price > 6.9
                or float(fleet.provider('show', 'user')['credit']) < 5):
            raise ValueError('Compatible owned-scope offer/budget unavailable')
        ROOT.mkdir(parents=True, exist_ok=False)
        lease = {'label': LABEL, 'geolocation': offer['geolocation'], 'offer': offer['id'],
            'owner': 'rep_geometry_transcoder/root', 'hourly_usd': price, 'gpus': offer['num_gpus'],
            'scope': 'bounded_navigation_refined_fitting_only_no_behavior_on_16GiB',
            'deadline': time.time() + 3 * 3600, 'credit_reserve': 1.5, 'hourly_cap': 7,
            'remote_root': REMOTE}
        fleet.write(ROOT / 'RESERVATION.json', lease)
        subprocess.run(['launchctl', 'submit', '-l', GUARD, '--', '/usr/bin/env',
            'PATH=/usr/local/bin:/usr/bin:/bin:/Users/stevenyang/.local/bin', fleet.PYTHON,
            str(Path(__file__).resolve()), 'guard'], check=True)
        result = fleet.provider('create', 'instance', str(offer['id']), '--image',
            'pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime', '--disk', '80',
            '--label', LABEL, '--ssh', '--direct', '--cancel-unavail')
        fleet.write(ROOT / 'CREATE.json', result)
        if not result.get('success') or not result.get('new_contract'):
            raise ValueError('Advertised offer was not acquired')
        fleet.write(ROOT / 'LEASE.json', {**lease, 'instance': result['new_contract']})
        fleet.write(ROOT / 'START.json', {'time': time.time()})
        print({'instance': result['new_contract'], 'gpus': offer['num_gpus'], 'hourly_usd': price}, flush=True)


def guard():
    lease = fleet.read(ROOT / 'RESERVATION.json')
    while not (ROOT / 'LEASE.json').exists():
        if (ROOT / 'CREATE.json').exists() and not fleet.read(ROOT / 'CREATE.json').get('success'):
            subprocess.run(['launchctl', 'remove', GUARD], check=False); return
        matches = [r for r in fleet.provider('show', 'instances') if r.get('label') == LABEL]
        if matches:
            assert len(matches) == 1 and matches[0]['geolocation'] == lease['geolocation']
            try: fleet.write(ROOT / 'LEASE.json', {**lease, 'instance': matches[0]['id']})
            except FileExistsError: pass
            break
        time.sleep(10)
    lease = fleet.read(ROOT / 'LEASE.json')
    fleet.BASE, fleet.REMOTE = ROOT, REMOTE
    fleet.INSTANCE, fleet.LABEL, fleet.PLACE = lease['instance'], LABEL, lease['geolocation']
    fleet.GUARD = GUARD
    fleet.guard()


def reuse():
    """Request available GPUs on an owned retained disk; never restart old science."""
    assert '2026-09-11T22:03Z owned Michigan reuse' in Path('/Users/stevenyang/Documents/GPU_RESOURCE_BOARD.md').read_text()
    with (components.fleet.BASE / 'RENTAL.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = fleet.provider('show', 'instances')
        row = next(r for r in rows if r['id'] == 50546171)
        assert row['label'] == 'jepa-confirmation-0911-reach-2' and row['geolocation'] == 'Michigan, US'
        assert row['cur_state'] == 'stopped' and row['actual_status'] == 'exited'
        assert row['num_gpus'] == 2 and row['gpu_name'] == 'RTX 4090'
        rate = components.reserved_hourly(rows, components.fleet.BASE) - fleet.account_hourly([row]) + row['dph_total']
        if rate > 6.9 or float(fleet.provider('show', 'user')['credit']) < 5:
            raise ValueError('Owned reuse exceeds current budget')
        ROOT.mkdir(parents=True, exist_ok=False)
        lease = {'instance': row['id'], 'label': row['label'], 'geolocation': row['geolocation'],
            'owner': 'rep_geometry_transcoder/root', 'hourly_usd': row['dph_total'], 'gpus': 2,
            'scope': 'navigation_fits_and_refined_development_receiving_no_old_confirmation',
            'deadline': time.time() + 3 * 3600, 'credit_reserve': 1.5, 'hourly_cap': 7,
            'remote_root': REMOTE, 'old_scientific_queues_restart': False}
        fleet.write(ROOT / 'RESERVATION.json', lease)
        fleet.write(ROOT / 'LEASE.json', lease)
        subprocess.run(['launchctl', 'submit', '-l', GUARD, '--', '/usr/bin/env',
            'PATH=/usr/local/bin:/usr/bin:/bin:/Users/stevenyang/.local/bin', fleet.PYTHON,
            str(Path(__file__).resolve()), 'guard'], check=True)
        result = fleet.provider('start', 'instance', str(row['id']))
        fleet.write(ROOT / 'START.json', {'time': time.time(), 'response': result})
        print({'requested_instance': row['id'], 'projected_account_hourly': rate, 'not_yet_running': True}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('rent', 'guard', 'reuse'))
    globals()[p.parse_args().mode]()
