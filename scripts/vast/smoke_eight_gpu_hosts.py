"""One-shot, user-authorized infrastructure trials; no research data or model.

Request termination within 120 seconds of rental submission, including boot.
Only the named machines or the explicit named replacement may be rented;
never automatically substitute or retry rental.
"""
import concurrent.futures as cf
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/offline_study/eight-gpu-infra-smoke-20260912-v1'
VAST = '/Users/stevenyang/.local/bin/vastai'
TARGETS = {145795: 3.25, 149537: 4.35, 147839: 5.42}
GPU = 'RTX 5090'
TARGET_GPUS = {}
PREFIX = 'infra120-20260912-'
IMAGE = 'pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime'


def now():
    return datetime.now(timezone.utc).isoformat()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n')


def api(*args, timeout=12):
    # This CLI treats retry as total attempts; zero skips the HTTP request.
    p = subprocess.run([VAST, *map(str, args), '--retry', '1', '--raw'],
                       capture_output=True, text=True, timeout=timeout)
    if p.returncode == 0 and not p.stdout.strip():
        return {'returncode': 0, 'response_body_empty': True, 'stderr': p.stderr[:1500]}
    try:
        return json.loads(p.stdout)
    except ValueError:
        return {'command_failed': True, 'returncode': p.returncode,
                'stdout': p.stdout[:1500], 'stderr': p.stderr[:1500]}


def ssh_endpoints(row):
    """Prefer the direct mapping requested at creation, then try Vast's proxy."""
    endpoints = []
    for mapping in (row.get('ports') or {}).get('22/tcp', []):
        if row.get('public_ipaddr') and mapping.get('HostPort'):
            endpoint=(row['public_ipaddr'],str(mapping['HostPort']))
            if endpoint not in endpoints:
                endpoints.append(endpoint)
    if row.get('ssh_host') and row.get('ssh_port'):
        endpoint=(row['ssh_host'],str(row['ssh_port']))
        if endpoint not in endpoints:
            endpoints.append(endpoint)
    return endpoints


SMOKE = r'''
import concurrent.futures as cf, json, subprocess, time
start=time.monotonic()
p=subprocess.run(['nvidia-smi','--query-gpu=index,name,uuid,memory.total','--format=csv,noheader'],capture_output=True,text=True,timeout=5)
print(json.dumps({'nvidia_smi_rc':p.returncode,'devices':p.stdout}),flush=True)
import torch
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32=False
assert torch.cuda.device_count()==8, torch.cuda.device_count()
def check(i):
 with torch.cuda.device(i), torch.inference_mode():
  a=torch.ones((1024,1024),device=f'cuda:{i}',dtype=torch.float32)
  b=a@a
  torch.cuda.synchronize(i)
  ok=bool((b==1024).all().item())
  assert ok, i
  return {'gpu':i,'exact_constant_matmul_pass':ok}
with cf.ThreadPoolExecutor(max_workers=8) as pool:
 results=list(pool.map(check,range(8)))
print(json.dumps({'infrastructure_smoke_pass':True,'devices':results,'elapsed_s':time.monotonic()-start}),flush=True)
'''


def trial(offer):
    machine = offer['machine_id']
    label = PREFIX + str(machine)
    start = time.monotonic()
    record = {'machine_id': machine, 'offer': offer, 'label': label,
              'started_utc': now(), 'maximum_trial_seconds': 120,
              'scientific_evaluation': False, 'status': 'rental_submitted'}
    instance = None
    save(f'{machine}.json', record)
    try:
        created = api('create', 'instance', offer['id'], '--disk', 80,
                      '--image', IMAGE, '--ssh', '--direct', '--cancel-unavail',
                      '--label', label)
        record['creation'] = {k:v for k,v in created.items() if 'key' not in k.lower()} if isinstance(created,dict) else created
        instance = created.get('new_contract') if isinstance(created, dict) else None
        if not instance:
            record['status'] = 'creation_not_confirmed'
            return record
        print(json.dumps({'machine': machine, 'instance': instance, 'event': 'created'}), flush=True)
        while time.monotonic() - start < 105:
            rows = api('show', 'instances', timeout=8)
            if not isinstance(rows, list):
                time.sleep(3)
                continue
            row = next((r for r in rows if r.get('id') == instance), None)
            if row is None:
                record['status'] = 'instance_not_listed'
                break
            assert row.get('label') == label, 'Instance identity mismatch'
            record['last_state'] = {k: row.get(k) for k in
                ['id','machine_id','label','actual_status','intended_status','status_msg',
                 'ssh_host','ssh_port','geolocation','num_gpus']}
            save(f'{machine}.json', record)
            state = row.get('actual_status')
            msg = str(row.get('status_msg', ''))
            if 'unresolvable CDI' in msg or (row.get('intended_status') == 'stopped'):
                record['status'] = 'host_startup_failed'
                break
            if state == 'running' and ssh_endpoints(row):
                assert str(row.get('geolocation','')).endswith('US'), 'Receiving geography mismatch'
                remaining = 112 - (time.monotonic() - start)
                if remaining < 8:
                    break
                for host,port in ssh_endpoints(row):
                    remaining=112-(time.monotonic()-start)
                    if remaining<8:
                        break
                    ssh=['ssh','-i','/Users/stevenyang/.ssh/id_ed25519',
                        '-o','BatchMode=yes','-o','ConnectTimeout=2',
                        '-o','StrictHostKeyChecking=no','-o','UserKnownHostsFile=/dev/null',
                        '-p',port,f'root@{host}']
                    try:
                        probe=subprocess.run(ssh+['true'],capture_output=True,text=True,timeout=4)
                    except subprocess.TimeoutExpired:
                        record.setdefault('ssh_attempts',[]).append({'host':host,'port':port,'timeout':True})
                        continue
                    record.setdefault('ssh_attempts',[]).append({'host':host,'port':port,'returncode':probe.returncode,'stderr':probe.stderr})
                    if probe.returncode!=0:
                        continue
                    remaining=112-(time.monotonic()-start)
                    if remaining<5:
                        break
                    p=subprocess.run(ssh+['python -'],input=SMOKE,capture_output=True,
                                     text=True,timeout=min(25,remaining))
                    record['smoke']={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'host':host,'port':port}
                    if p.returncode==0 and '"infrastructure_smoke_pass": true' in p.stdout:
                        record['status']='tiny_cuda_smoke_pass'
                        break
                if record['status']=='tiny_cuda_smoke_pass':
                    break
            time.sleep(3)
        if record['status'] == 'rental_submitted':
            record['status'] = 'not_ready_within_trial_window'
    except Exception as exc:
        record['status'] = 'trial_exception'
        record['error'] = type(exc).__name__ + ': ' + str(exc)[:1000]
    finally:
        record['termination_requested_utc'] = now()
        record['elapsed_before_cleanup_s'] = time.monotonic() - start
        save(f'{machine}.json', record)
        # If creation timed out, identify only our exact unique label before cleanup.
        if instance is None:
            rows = api('show','instances',timeout=8)
            matches = [r for r in rows if r.get('label') == label] if isinstance(rows,list) else []
            if len(matches) == 1:
                instance = matches[0]['id']
        if instance is not None:
            record['instance_id'] = instance
            record['destroy_result'] = api('destroy','instance',instance,'-y',timeout=12)
        record['finished_utc'] = now()
        record['elapsed_total_s'] = time.monotonic() - start
        save(f'{machine}.json', record)
        print(json.dumps(record), flush=True)
    return record


def main():
    if OUT.exists():
        raise SystemExit('Receipt directory exists: refuse duplicate rentals')
    rows = api('search','offers','num_gpus=8 rentable=true',
               '-n','--storage',80,'--limit',1000,'-o','dph_total',timeout=25)
    if not isinstance(rows,list):
        raise SystemExit('Offer refresh failed; no rentals')
    selected=[]
    for machine, cap in TARGETS.items():
        eligible=[r for r in rows if r.get('machine_id')==machine
            and str(r.get('geolocation','')).endswith('US')
            and r.get('num_gpus')==8 and r.get('gpu_name')==TARGET_GPUS.get(machine,GPU)
            and r.get('dph_total',float('inf'))<=cap and r.get('disk_space',0)>=80]
        if eligible:
            selected.append(min(eligible,key=lambda r:r['dph_total']))
        else:
            print(json.dumps({'machine':machine,'status':'no_matching_current_offer_no_substitution'}),flush=True)
    OUT.mkdir(parents=True,exist_ok=False)
    save('authorization.json',{'scope':'three_named_infrastructure_trials_only',
        'maximum_trial_seconds_per_host':120,'selected_offers':selected,
        'deverified_exception_authorized':True,'scientific_runs_authorized':False,
        'aggregate_quote_per_hour':sum(r['dph_total'] for r in selected),
        'note':'Temporary trial pricing does not authorize a full fleet budget increase.'})
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(trial,selected))
    rows=api('show','instances',timeout=12)
    remaining=[{k:r.get(k) for k in ['id','label','actual_status','intended_status']}
        for r in rows if str(r.get('label','')).startswith(PREFIX)] if isinstance(rows,list) else None
    save('summary.json',{'finished_utc':now(),'results':results,'remaining_trial_instances':remaining})
    print(json.dumps({'remaining_trial_instances':remaining,'receipt_directory':str(OUT)}),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--third-4090-replacement',action='store_true')
    group.add_argument('--other-cheap-hosts',action='store_true')
    group.add_argument('--cheap-retest-v3',action='store_true')
    args=parser.parse_args()
    if args.third_4090_replacement:
        OUT=OUT/'third-4090-replacement'
        TARGETS={147859:5.50}
        GPU='RTX 4090'
    if args.other_cheap_hosts:
        OUT=ROOT/'artifacts/offline_study/eight-gpu-infra-smoke-20260912-v2'
        PREFIX='infra120b-20260912-'
        TARGETS={15096:0.60,68094:0.95,141626:1.05}
        TARGET_GPUS={15096:'RTX 2080 Ti',68094:'RTX 3090',141626:'RTX 5060 Ti'}
    if args.cheap_retest_v3:
        OUT=ROOT/'artifacts/offline_study/eight-gpu-infra-smoke-20260912-v3'
        PREFIX='infra120c-20260912-'
        TARGETS={68094:0.95,141626:1.05,145795:3.25}
        TARGET_GPUS={68094:'RTX 3090',141626:'RTX 5060 Ti',145795:'RTX 5090'}
    main()
