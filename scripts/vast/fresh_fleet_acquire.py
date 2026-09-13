"""Acquire three explicitly selected US hosts; retain smoke passes for fresh-bank staging."""
import concurrent.futures as cf
import json
from pathlib import Path
import subprocess
import time
from smoke_eight_gpu_hosts import api, ssh_endpoints, SMOKE, IMAGE, now

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/offline_study/fresh-fleet-20260912-v1'
TARGETS = {68094: ('RTX 3090', .95), 141210: ('RTX 5090', 3.35), 149629: ('RTX 5090', 4.55)}
GPU_COUNT = 8
SEARCH_WAIT_SECONDS = 0
MAX_NEW_HOSTS = 3

def save(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2) + '\n')

def acquire(offer):
    machine = offer['machine_id']
    label = f'fresh4-20260912-{machine}-{OUT.name.rsplit("-", 1)[-1]}'
    rec = {'offer': offer, 'label': label, 'started_utc': now(), 'status': 'creating',
           'scientific_episodes': 0, 'retained_for': 'fresh-bank receiving validation and execution'}
    instance = None
    start = time.monotonic()
    try:
        response = api('create', 'instance', offer['id'], '--disk', 80, '--image', IMAGE,
                       '--ssh', '--direct', '--cancel-unavail', '--label', label, timeout=30)
        rec['creation'] = {k:v for k,v in response.items() if 'key' not in k.lower()} if isinstance(response, dict) else response
        instance = response.get('new_contract') if isinstance(response, dict) else None
        if not instance:
            rec['status'] = 'creation_unconfirmed'
            return rec
        rec['instance_id'] = instance
        print(json.dumps({'event':'created', 'instance':instance, 'machine':machine}), flush=True)
        while time.monotonic() - start < 900:
            try:
                rows = api('show','instances',timeout=15)
            except subprocess.TimeoutExpired:
                rec.setdefault('poll_timeouts', []).append(now())
                save(f'{machine}.json', rec)
                time.sleep(5)
                continue
            row = next((r for r in rows if r.get('id') == instance), None) if isinstance(rows,list) else None
            if row is None:
                time.sleep(5)
                continue
            assert row.get('label') == label
            assert str(row.get('geolocation','')).endswith('US')
            rec['receiving'] = {k:row.get(k) for k in ['id','machine_id','actual_status','intended_status','status_msg','geolocation','num_gpus']}
            save(f'{machine}.json', rec)
            if 'unresolvable CDI' in str(row.get('status_msg','')) or row.get('intended_status') == 'stopped':
                rec['status'] = 'provisioning_failed'
                break
            if row.get('actual_status') == 'running':
                for host,port in ssh_endpoints(row):
                    ssh = ['ssh','-i','/Users/stevenyang/.ssh/id_ed25519','-o','BatchMode=yes',
                           '-o','ConnectTimeout=3','-o','StrictHostKeyChecking=accept-new',
                           '-o',f'UserKnownHostsFile={OUT / (str(machine) + "-known_hosts")}',
                           '-p',port,f'root@{host}']
                    try:
                        probe = subprocess.run(ssh+['true'],capture_output=True,text=True,timeout=5)
                        if probe.returncode: continue
                        smoke = subprocess.run(ssh+['python -'],input=SMOKE,capture_output=True,text=True,timeout=120)
                    except subprocess.TimeoutExpired:
                        rec['status'] = 'ssh_or_smoke_timeout'
                        continue
                    rec['smoke'] = {'returncode':smoke.returncode,'stdout':smoke.stdout,'stderr':smoke.stderr}
                    rec['ssh'] = {'host':host,'port':port}
                    if smoke.returncode == 255:
                        rec.setdefault('ssh_transport_failures', []).append(now())
                        continue
                    if smoke.returncode == 0 and '"infrastructure_smoke_pass": true' in smoke.stdout:
                        rec['status'] = 'smoke_pass_retained_for_staging'
                        return rec
                    rec['status'] = 'smoke_failed'
                    break
                if rec['status'] == 'smoke_failed': break
            time.sleep(5)
        if rec['status'] == 'creating': rec['status'] = 'provisioning_timeout'
    except Exception as exc:
        rec['status'] = 'acquisition_error'
        rec['error'] = str(exc)
    finally:
        # No scientific data have been staged by this script. Successful hosts stay leased.
        if rec['status'] != 'smoke_pass_retained_for_staging':
            if instance is None:
                rows = api('show','instances',timeout=15)
                matches = [r for r in rows if r.get('label') == label] if isinstance(rows,list) else []
                if len(matches) == 1: instance = matches[0]['id']
            if instance is not None:
                rec['instance_id'] = instance
                rec['release'] = api('destroy','instance',instance,'-y',timeout=20)
        rec['finished_utc'] = now()
        rec['elapsed_s'] = time.monotonic() - start
        save(f'{machine}.json', rec)
        print(json.dumps({'machine':machine,'instance':instance,'status':rec['status']}),flush=True)
    return rec

def main():
    if OUT.exists(): raise SystemExit('Existing acquisition receipt; no duplicate rentals')
    existing = api('show','instances',timeout=15)
    if not isinstance(existing,list): raise SystemExit('Cannot verify current fleet')
    active = [r for r in existing if r.get('intended_status') == 'running']
    if any(not str(r.get('label','')).startswith('fresh4-20260912-') for r in active):
        raise SystemExit('Existing running lease: reconcile budget/ownership first')
    existing_hourly = sum(float(r.get('dph_total',999)) for r in active)
    search_deadline = time.monotonic() + SEARCH_WAIT_SECONDS
    while True:
        offers = api('search','offers',f'num_gpus={GPU_COUNT} rentable=true','-n','--storage',80,'--limit',1000,'-o','dph_total',timeout=30)
        if not isinstance(offers,list): raise SystemExit('Offer search failed')
        eligible = any(r.get('machine_id') in TARGETS and r.get('gpu_name') == TARGETS[r['machine_id']][0]
                       and r.get('dph_total',999) <= TARGETS[r['machine_id']][1]
                       and str(r.get('geolocation','')).endswith('US') and r.get('reliability',0) >= .96
                       for r in offers)
        if eligible or time.monotonic() >= search_deadline: break
        time.sleep(15)
    selected = []
    for machine,(gpu,cap) in TARGETS.items():
        if any(r.get('machine_id') == machine for r in active):
            print(json.dumps({'machine':machine,'status':'already_leased_no_duplicate'}),flush=True)
            continue
        matches = [r for r in offers if r.get('machine_id') == machine and r.get('gpu_name') == gpu
                   and r.get('num_gpus') == GPU_COUNT and str(r.get('geolocation','')).endswith('US')
                   and r.get('dph_total',999) <= cap and r.get('reliability',0) >= .96]
        if matches: selected.append(min(matches,key=lambda r:r['dph_total']))
    selected = sorted(selected, key=lambda r:r['dph_total'])[:MAX_NEW_HOSTS]
    # Refresh after a market wait: only current live leases establish the budget.
    current = api('show','instances',timeout=15)
    if not isinstance(current,list): raise SystemExit('Cannot reverify current fleet before rental')
    current_active = [r for r in current if r.get('intended_status') == 'running']
    selected = [r for r in selected if not any(x.get('machine_id') == r['machine_id'] for x in current_active)]
    existing_hourly = sum(float(r.get('dph_total',999)) for r in current_active)
    assert sum(r['dph_total'] for r in selected) + existing_hourly + .03 <= 12
    OUT.mkdir(parents=True,exist_ok=False)
    save('authorization.json', {'hourly_cap':12,'campaign_total_cap':220,'selected':selected,
        'receiving_checks_required':True,'smoke_seconds_max':120,'provisioning_seconds_max':900})
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(acquire,selected))
    save('summary.json', {'results':results,'finished_utc':now()})

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--replacement', action='store_true')
    parser.add_argument('--california', action='store_true')
    parser.add_argument('--cached-retry', action='store_true')
    parser.add_argument('--four-gpu', action='store_true')
    parser.add_argument('--new-value-fleet', action='store_true')
    parser.add_argument('--value-retry', action='store_true')
    parser.add_argument('--memory-qualified', action='store_true')
    parser.add_argument('--memory-retry', action='store_true')
    parser.add_argument('--broad-memory-eight', action='store_true')
    parser.add_argument('--broad-memory-four', action='store_true')
    parser.add_argument('--third-eight', action='store_true')
    parser.add_argument('--watch-third-eight', action='store_true')
    args = parser.parse_args()
    if args.replacement:
        OUT = ROOT / 'artifacts/offline_study/fresh-fleet-20260912-v1-replacement'
        TARGETS = {147839: ('RTX 5090', 5.42)}
    if args.california:
        OUT = ROOT / 'artifacts/offline_study/fresh-fleet-20260912-v1-california'
        TARGETS = {149537: ('RTX 5090', 4.35)}
    if args.cached_retry:
        OUT = ROOT / 'artifacts/offline_study/fresh-fleet-20260912-v2-cached'
        TARGETS = {141210: ('RTX 5090', 3.35)}
    if args.four_gpu:
        OUT = ROOT / 'artifacts/offline_study/fresh-fleet-20260912-v2-four'
        TARGETS = {51755: ('RTX 5070 Ti', 2.20)}
        GPU_COUNT = 4
        SMOKE = SMOKE.replace('device_count()==8', 'device_count()==4').replace('max_workers=8', 'max_workers=4').replace('range(8)', 'range(4)')
    if args.new_value_fleet:
        OUT = ROOT / 'artifacts/offline_study/fresh-fleet-20260912-v3-value'
        TARGETS = {138092: ('RTX 4080S', 1.45), 147341: ('RTX 4080S', 1.52), 148978: ('RTX 4090', 4.10)}
    if args.value_retry:
        OUT = ROOT / 'artifacts/offline_study/fresh-fleet-20260912-v3-value-retry'
        TARGETS = {138092: ('RTX 4080S', 1.45), 148978: ('RTX 4090', 4.10)}
    if args.memory_qualified:
        OUT = ROOT / 'artifacts/offline_study/fresh-fleet-20260912-v4-memory'
        TARGETS = {142274: ('RTX 3090', 3.27), 141210: ('RTX 5090', 3.35)}
    if args.memory_retry:
        OUT = ROOT / 'artifacts/offline_study/fresh-fleet-20260912-v5-memory-retry'
        TARGETS = {141210: ('RTX 5090', 3.35)}
    if args.broad_memory_eight:
        OUT = ROOT / 'artifacts/offline_study/fresh-fleet-20260912-v6-eight'
        TARGETS = {142429: ('Tesla V100', 1.56)}
    if args.broad_memory_four:
        OUT = ROOT / 'artifacts/offline_study/fresh-fleet-20260912-v6-four'
        TARGETS = {7646: ('RTX A5000', .86), 140835: ('RTX 5090', 1.72)}
        GPU_COUNT = 4
        SMOKE = SMOKE.replace('device_count()==8', 'device_count()==4').replace('max_workers=8', 'max_workers=4').replace('range(8)', 'range(4)')
    if args.third_eight:
        OUT = ROOT / 'artifacts/offline_study/fresh-fleet-20260912-v7-third-eight'
        TARGETS = {141210: ('RTX 5090', 3.35)}
    if args.watch_third_eight:
        OUT = ROOT / 'artifacts/offline_study/fresh-fleet-20260912-v8-third-eight-watch'
        TARGETS = {141210: ('RTX 5090', 3.35), 148978: ('RTX 4090', 4.10)}
        SEARCH_WAIT_SECONDS = 900
        MAX_NEW_HOSTS = 1
    main()
