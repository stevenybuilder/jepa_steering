"""User-approved, one-worker input-preparation lease with a two-hour backstop."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from offline_study.protocol import write_json

OUT = ROOT / 'artifacts/offline_study/protected-preparation-worker-20260912-v1'
VAST = '/Users/stevenyang/.local/bin/vastai'
LABEL = 'protected-input-prep-20260912-cap2'


def command(*args):
    return json.loads(subprocess.check_output([VAST, *args, '--raw'], text=True, timeout=90))


def rent():
    if OUT.exists():
        raise ValueError('Existing lease receipt; never create a duplicate lease')
    rows = command('search', 'offers',
        'geolocation=US reliability>0.99 num_gpus=1 cpu_cores_effective>=16 cpu_ram>=32 inet_down>500 disk_space>80',
        '--storage', '80', '--order', 'dph_total', '--limit', '10')
    rows = [r for r in rows if r['machine_id'] == 147415 and r['geolocation'] == 'Connecticut, US'
            and r['gpu_name'] == 'RTX 3060' and r['dph_total'] <= 0.103]
    if not rows:
        raise ValueError('Approved machine/price no longer available; no replacement rented')
    offer = rows[0]
    OUT.mkdir(parents=True, exist_ok=False)
    write_json(OUT / 'offer.json', offer)
    start = time.time()
    write_json(OUT / 'authorization.json', {'role': 'input_preparation_only',
        'total_cap_usd': 2, 'maximum_seconds': 7200, 'deadline_unix': start + 7200,
        'hourly_quote_with_80GiB': offer['dph_total'], 'model_outcomes_authorized': False,
        'training_authorized': False, 'label': LABEL, 'machine_id': offer['machine_id']})
    result = command('create', 'instance', str(offer['id']), '--disk', '80',
        '--image', 'pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime',
        '--ssh', '--direct', '--cancel-unavail', '--label', LABEL)
    write_json(OUT / 'creation.json', result)
    if not result.get('success') or not result.get('new_contract'):
        raise ValueError('Creation did not confirm success; inspect receipt before any retry')
    log = open(OUT / 'backstop.log', 'ab', buffering=0)
    proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'guard'],
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write_json(OUT / 'guard.json', {'pid': proc.pid, 'started_unix': time.time()})
    print(json.dumps({'instance': result['new_contract'], 'hourly_quote': offer['dph_total'],
        'deadline_utc': datetime.fromtimestamp(start + 7200, timezone.utc).isoformat(), 'guard_pid': proc.pid}))


def guard():
    p = json.loads((OUT / 'authorization.json').read_text())
    instance = json.loads((OUT / 'creation.json').read_text())['new_contract']
    while time.time() < p['deadline_unix']:
        if (OUT / 'CLOSED.json').exists():
            return
        time.sleep(min(30, max(0.1, p['deadline_unix'] - time.time())))
    for attempt in range(4):
        try:
            rows = command('show', 'instances')
            current = next((r for r in rows if r['id'] == instance), None)
            if current is None:
                return
            if current['label'] != LABEL or current['geolocation'] != 'Connecticut, US':
                raise ValueError('Lease identity mismatch; refusing unrelated stop')
            result = command('stop', 'instance', str(instance))
            write_json(OUT / 'BUDGET_STOP.json', {'instance': instance, 'result': result,
                'utc': datetime.now(timezone.utc).isoformat(), 'source_storage_retained': True,
                'storage_still_billable': True})
            return
        except Exception as exc:
            print(str(exc), flush=True)
            time.sleep(10)
    raise RuntimeError('Backstop could not confirm stop; immediate manual attention required')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('rent', 'guard'))
    args = parser.parse_args()
    {'rent': rent, 'guard': guard}[args.mode]()
