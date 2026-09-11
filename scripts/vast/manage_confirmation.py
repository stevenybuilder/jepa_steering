"""Manage only this $65 confirmation fleet; preserve before deletion, fail closed.

No other instance or experiment is in scope. Separate budget mode is a backstop
even if archiving or the local monitoring process fails. Stored keys stay local.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

from final_preservation_drive import upload_direct, verify as drive_verify, request

PROJECT=Path(__file__).resolve().parents[2]
ROOT=PROJECT/'artifacts/offline_study/confirmation-20260911-v1'
VAST='/Users/stevenyang/.local/bin/vastai'
KEY='/tmp/jepa_vast_50123620_ed25519'


def stamp():return datetime.now(timezone.utc).isoformat()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4<<20),b''):h.update(chunk)
    return h.hexdigest()


def command(argv,timeout=90):
    return subprocess.check_output(argv,text=True,timeout=timeout)


def provider():return {r['id']:r for r in json.loads(command([VAST,'show','instances','--raw']))}


def identity(lease,row):
    if (row['id']!=lease['id'] or row['label']!=lease['label'] or row['geolocation']!=lease['geolocation']
        or not row['geolocation'].endswith(', US') or row['num_gpus']!=lease['num_gpus']
        or row['gpu_name']!='RTX 4090'):
        raise ValueError('Refuse mismatched/shared instance')


def ssh(row):
    return ['ssh','-i',KEY,'-o','BatchMode=yes','-o','StrictHostKeyChecking=accept-new',
        '-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3',
        '-p',str(row['ports']['22/tcp'][0]['HostPort']),'root@'+row['public_ipaddr']]


def reserve(spec):
    start=time.time()
    label='jepa-confirmation-0911-'+spec['task']+'-'+str(spec['logical_ranks'][0])
    result=json.loads(command([VAST,'create','instance',str(spec['offer']),
        '--image','pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime','--disk','40',
        '--label',label,'--ssh','--direct','--cancel-unavail','--raw'],180))
    if not result.get('success') or not result.get('new_contract'):
        raise ValueError('Rental not created')
    lease={**spec,'id':result['new_contract'],'label':label,'created_timestamp':start,'created_utc':stamp()}
    write(ROOT/'leases'/f"{lease['id']}.json",lease)
    print(json.dumps({'created':lease['id'],'task':spec['task'],'gpus':spec['num_gpus']}),flush=True)
    return lease


def stage(lease):
    package=json.loads((ROOT/'PACKAGE.json').read_text())
    archive=Path(package['archive'])
    if digest(archive)!=package['sha256']:raise ValueError('Changed input bundle')
    row=None
    for _ in range(90):
        row=provider().get(lease['id'])
        if row:
            identity(lease,row)
            if row['actual_status']=='running' and row.get('ports',{}).get('22/tcp'):
                try:
                    command(ssh(row)+['true'],25)
                    break
                except (subprocess.SubprocessError,KeyError):pass
        time.sleep(5)
    else:raise RuntimeError('Receiving SSH unavailable; preserve/release bootstrap allocation')
    rate=max(float(row['dph_total']),float(row.get('instance',{}).get('totalHour',0)))
    if rate>lease['dph_total']+.05:raise ValueError('Unexpected receiving rental rate')
    write(ROOT/'leases'/f"{lease['id']}-received.json",{k:row.get(k) for k in
        ('id','label','geolocation','gpu_name','num_gpus','public_ipaddr','ports','dph_total','actual_status')})
    shell=shlex.join(ssh(row)[:-1])
    subprocess.run(['rsync','-a','--timeout=90','-e',shell,str(archive),
        ssh(row)[-1]+':/workspace/confirmation-inputs.tar.gz'],check=True,timeout=600)
    received=command(ssh(row)+['sha256sum /workspace/confirmation-inputs.tar.gz'],60)
    if package['sha256'] not in received.split():raise ValueError('Received bundle SHA mismatch')
    command(ssh(row)+['mkdir /workspace/confirmation && tar -xzf /workspace/confirmation-inputs.tar.gz -C /workspace/confirmation'],180)
    launcher=PROJECT/'scripts/vast/launch_confirmation.py'
    subprocess.run(['rsync','-a','--timeout=90','-e',shell,str(launcher),
        ssh(row)[-1]+':/workspace/confirmation/scripts/vast/launch_confirmation.py'],check=True,timeout=90)
    if digest(launcher) not in command(ssh(row)+['sha256sum /workspace/confirmation/scripts/vast/launch_confirmation.py']).split():
        raise ValueError('Receiving supervisor source differs')
    auth=json.loads((ROOT/'AUTHORIZATION.json').read_text())
    argv=['python3','-u','/workspace/confirmation/scripts/vast/launch_confirmation.py',
        '--task',lease['task'],'--deadline',str(auth['deadline_timestamp']),
        '--logical-ranks',*[str(r) for r in lease['logical_ranks']]]
    start='nohup setsid '+shlex.join(argv)+' > /workspace/confirmation-launch.log 2>&1 < /dev/null & echo $!'
    result=command(ssh(row)+[start],60)
    write(ROOT/'leases'/f"{lease['id']}-launched.json",{'utc':stamp(),'launch_response':result,
        'deadline':auth['deadline_timestamp'],'command':argv,'archive_sha256':package['sha256'],
        'supervisor_sha256':digest(launcher)})
    print(json.dumps({'launched_supervisor':lease['id'],'stage':'bootstrap_then_receiving_checks','utc':stamp()}),flush=True)


def leases():
    return [json.loads(p.read_text()) for p in sorted((ROOT/'leases').glob('*.json')) if p.stem.isdigit()]


def mirror(lease,row):
    local=ROOT/'workers'/str(lease['id'])
    local.mkdir(parents=True,exist_ok=True)
    subprocess.run(['rsync','-a','--timeout=90','-e',shlex.join(ssh(row)[:-1]),
        ssh(row)[-1]+':/workspace/confirmation-output/',str(local)+'/'],check=True,timeout=180,
        stdout=subprocess.DEVNULL)
    return local


def verify_archive(path):
    with tarfile.open(path,'r:gz') as src:
        manifest=json.load(src.extractfile('PRESERVATION_MANIFEST.json'))
        found=set()
        for member in src:
            if member.name=='PRESERVATION_MANIFEST.json':continue
            if not member.isfile() or member.name in found or member.name not in manifest['members']:
                raise ValueError('Unexpected archive member')
            h=hashlib.sha256()
            stream=src.extractfile(member)
            for chunk in iter(lambda:stream.read(4<<20),b''):h.update(chunk)
            if {'bytes':member.size,'sha256':h.hexdigest()}!=manifest['members'][member.name]:
                raise ValueError('Archive member checksum differs')
            found.add(member.name)
        if found!=set(manifest['members']):raise ValueError('Incomplete archive coverage')
    return {'members':len(found),'sha256':digest(path),'bytes':path.stat().st_size}


def preserve(lease,row):
    identity(lease,row)
    out=ROOT/'closeout'/str(lease['id'])
    out.mkdir(parents=True,exist_ok=True)
    if (out/'DRIVE_VERIFIED.json').exists():return
    # Never package a moving research tree. All owned streams must be terminal.
    live=command(ssh(row)+['nvidia-smi --query-compute-apps=pid --format=csv,noheader'],60)
    if any(line.strip().isdigit() for line in live.splitlines()):raise ValueError('GPU workload remains active')
    name=f"JEPA-confirmation-{lease['id']}-20260911.tar.gz"
    archive=out/name
    if not (out/'REMOTE_ARCHIVE.json').exists():
        output=command(ssh(row)+[shlex.join(['/workspace/decision-python/bin/python',
            '/workspace/confirmation/scripts/vast/preserve_confirmation_worker.py','--archive','/workspace/'+name])],600)
        receipt=json.loads(output.strip().splitlines()[-1])
        write(out/'REMOTE_ARCHIVE.json',receipt)
    receipt=json.loads((out/'REMOTE_ARCHIVE.json').read_text())
    subprocess.run(['rsync','-a','--timeout=90','-e',shlex.join(ssh(row)[:-1]),
        ssh(row)[-1]+':/workspace/'+name,str(archive)],check=True,timeout=600)
    if digest(archive)!=receipt['sha256']:raise ValueError('Archive download mismatch')
    checked=verify_archive(archive)
    if not (out/'LOCAL_VERIFIED.json').exists():write(out/'LOCAL_VERIFIED.json',checked)
    # Metadata request refreshes existing local OAuth when necessary.
    with request('14zoPlI5qViiF5DwqVkCyKUu2O-Bj8u48','?fields=id,name,mimeType') as response:json.load(response)
    uploaded=upload_direct(archive,name,out/'DRIVE_UPLOAD.json')
    verified=drive_verify(uploaded['id'],name,checked['bytes'],checked['sha256'])
    write(out/'DRIVE_VERIFIED.json',{**checked,**verified,'file_id':uploaded['id'],'utc':stamp()})


def release(lease):
    out=ROOT/'closeout'/str(lease['id'])
    proof=json.loads((out/'DRIVE_VERIFIED.json').read_text())
    row=provider().get(lease['id'])
    if row is None:return
    identity(lease,row)
    drive_verify(proof['file_id'],proof['metadata']['name'],proof['bytes'],proof['sha256'],stream=False)
    # Explicit --yes; success is confirmed by provider absence, NOT stdout JSON.
    response=command([VAST,'destroy','instance',str(lease['id']),'--yes','--raw'],120)
    for _ in range(12):
        if lease['id'] not in provider():
            write(out/'RELEASED.json',{'id':lease['id'],'utc':stamp(),'provider_absence':True,
                                     'drive_verified_first':True,'response':response})
            print(json.dumps({'released':lease['id'],'archive_verified_in_drive':True}),flush=True)
            return
        time.sleep(5)
    raise ValueError('Provider has not confirmed release')


def monitor():
    auth=json.loads((ROOT/'AUTHORIZATION.json').read_text())
    while time.time()<auth['deadline_timestamp']:
        fleet=provider(); ready=True; counts={}
        for lease in leases():
            row=fleet.get(lease['id'])
            if not row:raise ValueError('Unpreserved worker vanished')
            identity(lease,row)
            state=command(ssh(row)+['if test -f /workspace/confirmation-output/WORKER_DONE.json; then echo COMPLETE; '
                'elif test -f /workspace/confirmation-launch-exit.json; then echo FAILED; else echo RUNNING; fi'],60).strip().splitlines()[-1]
            if state=='FAILED':raise RuntimeError('Worker failed; preserve before any retry')
            if state!='COMPLETE':ready=False
            available=command(ssh(row)+['test -d /workspace/confirmation-output && echo PRESENT || echo BOOTSTRAP'],30).strip().splitlines()[-1]
            if available=='PRESENT':
                local=mirror(lease,row)
                counts[lease['id']]=len(list((local/'results').glob('*/*/rank-*/episode-*.json')))
        print(json.dumps({'utc':stamp(),'mirrored_episode_records':counts,'target':960}),flush=True)
        if ready:break
        time.sleep(30)
    else:raise TimeoutError('Compute budget deadline; no unbounded continuation')
    joined=ROOT/'results'
    joined.mkdir(exist_ok=True)
    import shutil
    for lease in leases():
        base=ROOT/'workers'/str(lease['id'])/'results'/lease['task']
        for arm in base.iterdir():
            (joined/lease['task']/arm.name).mkdir(parents=True,exist_ok=True)
            for shard in arm.iterdir():shutil.copytree(shard,joined/lease['task']/arm.name/shard.name)
    package=json.loads((ROOT/'PACKAGE.json').read_text())
    env=dict(os.environ,PYTHONPATH=str(Path(package['stage'])/'src'))
    subprocess.run([str(PROJECT/'.venv/bin/python'),'-m','offline_study.confirmation','analyze',
        '--root',str(joined),'--freeze',str(ROOT/'freeze-v2'),'--output',str(ROOT/'analysis')],
        cwd=PROJECT,env=env,check=True,timeout=180)
    # Second independent reload must reproduce the exact frozen numerical report.
    subprocess.run([str(PROJECT/'.venv/bin/python'),'-m','offline_study.confirmation','analyze',
        '--root',str(joined),'--freeze',str(ROOT/'freeze-v2'),'--output',str(ROOT/'analysis-recheck')],
        cwd=PROJECT,env=env,check=True,timeout=180)
    if digest(ROOT/'analysis/report.json')!=digest(ROOT/'analysis-recheck/report.json'):
        raise ValueError('Independent analysis reload differs')
    name='JEPA-confirmation-final-analysis-20260911.json'
    upload=upload_direct(ROOT/'analysis/report.json',name,ROOT/'ANALYSIS_DRIVE_UPLOAD.json')
    proof=drive_verify(upload['id'],name,(ROOT/'analysis/report.json').stat().st_size,digest(ROOT/'analysis/report.json'))
    write(ROOT/'ANALYSIS_DRIVE_VERIFIED.json',proof)
    for lease in leases():
        preserve(lease,provider()[lease['id']])
        release(lease)
    write(ROOT/'DONE.json',{'utc':stamp(),'episodes':960,'analysis_verified_twice':True,
        'results_logs_inputs_archived_in_drive':True,'owned_workers_released':True})


def budget():
    auth=json.loads((ROOT/'AUTHORIZATION.json').read_text())
    while time.time()<auth['deadline_timestamp']:
        if (ROOT/'DONE.json').exists():return
        time.sleep(min(30,max(.1,auth['deadline_timestamp']-time.time())))
    for lease in leases():
        row=provider().get(lease['id'])
        if row:
            identity(lease,row)
            output=command([VAST,'stop','instance',str(lease['id']),'--raw'],120)
            write(ROOT/'closeout'/str(lease['id'])/'BUDGET_STOP.json',{'utc':stamp(),'response':output,
                'disk_destroyed':False,'reason':'hard new-compute cap; preserve disk if archive is incomplete'})


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('provision','monitor','budget'));args=p.parse_args()
    if args.mode=='provision':
        plan=json.loads((ROOT/'FLEET_PLAN.json').read_text())
        if sum(r['dph_total'] for r in plan)>7 or sum(r['num_gpus'] for r in plan)!=16:
            raise ValueError('Fleet exceeds approved scope or hourly cap')
        for task in ('reach','reach-wall'):
            if sorted(r for x in plan if x['task']==task for r in x['logical_ranks'])!=list(range(8)):
                raise ValueError('Missing/duplicate assigned logical stream')
        with ThreadPoolExecutor(max_workers=len(plan)) as pool:
            fleet=list(pool.map(reserve,plan))
            list(pool.map(stage,fleet))
    elif args.mode=='budget':budget()
    else:
        try:monitor()
        except Exception as error:
            write(ROOT/'MONITOR_FAILED.json',{'utc':stamp(),'error':str(error),'sources_retained':True})
            # Stop paid compute on an observed terminal failure; never delete an
            # unarchived source disk. The independent backstop covers hangs.
            for lease in leases():
                row=provider().get(lease['id'])
                if row:
                    identity(lease,row)
                    command([VAST,'stop','instance',str(lease['id']),'--raw'],120)
            raise


if __name__=='__main__':main()
