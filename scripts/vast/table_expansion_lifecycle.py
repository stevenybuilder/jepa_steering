"""Owned table workers only: live-price-checked activation and idempotent guard.

Never destroys disks or tops up credit. No scientific process is started here.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

PROJECT = Path(__file__).resolve().parents[2]
BASE = Path(os.environ.get('JEPA_TABLE_FLEET_ROOT',str(PROJECT / 'artifacts/offline_study/table-completion-20260911-v1/expansion')))
VAST = '/Users/stevenyang/.local/bin/vastai'
OWNED = {50561030: ('jepa-table-completion-0911-v1', 'Washington, US', 'pointmaze'),
         50544130: ('jepa-confirmation-0911-reach-wall-0', 'Florida, US', 'pusht')}


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open('x') as f:
            json.dump(obj, f, indent=2)


def provider():
    return json.loads(subprocess.check_output([VAST, 'show', 'instances', '--raw'], text=True, timeout=45))


def validate(row):
    label, place, _ = OWNED[row['id']]
    if row['label'] != label or row['geolocation'] != place:
        raise ValueError('Owner/geography binding changed')


def activate():
    board = Path('/Users/stevenyang/Documents/GPU_RESOURCE_BOARD.md').read_text()
    assert '2026-09-11T12:14Z table-fleet expansion reservation' in board
    rows = provider()
    fleet = sum(r['dph_total'] for r in rows if r['cur_state'] != 'stopped')
    selected = [next(r for r in rows if r['id'] == i) for i in OWNED]
    for row in selected:
        validate(row)
        assert row['cur_state'] == 'stopped' and row['actual_status'] == 'exited'
        fleet += row['dph_total']
    assert fleet < 6.8, 'Reserve storage margin within7USD/hour'
    BASE.mkdir(parents=True, exist_ok=False)
    write(BASE/'LEASE.json', {'owner': 'rep_geometry_transcoder/root',
        'workers': {str(r['id']): {'label': r['label'], 'geolocation': r['geolocation'],
            'task': OWNED[r['id']][2], 'gpus': r['num_gpus'], 'hourly_usd': r['dph_total']}
            for r in selected}, 'deadline_timestamp': time.time()+6*3600,
        'hourly_cap_usd': 7, 'credit_reserve_usd': 1, 'confirmation_restart': False})
    label = 'com.steven.jepa.table-expansion.guard.20260911'
    subprocess.run(['launchctl','submit','-l',label,'--',str(PROJECT/'.venv/bin/python'),
                    str(Path(__file__).resolve()),'guard'],check=True)
    for row in selected:
        result = subprocess.check_output([VAST,'start','instance',str(row['id']),'--raw'],text=True,timeout=90)
        write(BASE/f"START-{row['id']}.json", {'response':result})
        print(json.dumps({'instance':row['id'],'task':OWNED[row['id']][2],
                          'start_response':result,'hourly_usd':row['dph_total']}),flush=True)


def guard():
    while True:
        try:
            lease = json.loads((BASE/'LEASE.json').read_text())
            rows = provider()
            live = [r for r in rows if str(r['id']) in lease['workers'] and r['cur_state'] != 'stopped']
            for r in live:
                binding=lease['workers'][str(r['id'])]
                assert r['label']==binding['label'] and r['geolocation']==binding['geolocation']
            # Activation may not have occurred yet immediately after registration.
            started = all((BASE/f'START-{i}.json').exists() for i in lease['workers'])
            if not live and started: break
            user = json.loads(subprocess.check_output([VAST,'show','user','--raw'],text=True,timeout=45))
            reason = ('deadline' if time.time() >= lease['deadline_timestamp'] else
                      'credit_reserve' if float(user['credit']) < lease['credit_reserve_usd'] else None)
            if reason:
                for r in live:
                    result = subprocess.check_output([VAST,'stop','instance',str(r['id']),'--raw'],text=True,timeout=90)
                    write(BASE/f"GUARD_STOP-{r['id']}.json",{'reason':reason,'response':result,
                        'source_disk_retained':True,'scientific_completion':False,'time':time.time()})
        except Exception as exc:
            print(json.dumps({'guard_error':type(exc).__name__}),flush=True)
        time.sleep(30)
    write(BASE/'GUARD_DONE.json',{'owned_workers_stopped':True,'source_disks_retained':True})
    # launchctl submit may relaunch a finished job; unregister after receipts.
    subprocess.run(['launchctl','remove','com.steven.jepa.table-expansion.guard.20260911'],check=False)


def add_push_workers():
    assert '2026-09-11T12:21Z PointMaze rental acquired; Push-T replacement reservation' in Path('/Users/stevenyang/Documents/GPU_RESOURCE_BOARD.md').read_text()
    lease=json.loads((BASE/'LEASE.json').read_text())
    assert len(lease['workers'])==1 and '50588615' in lease['workers']
    chosen_machines=set()
    for place,count,offset in [('Texas, US',4,0),('California, US',2,4),('California, US',2,6)]:
        rows=provider()
        q=f'geolocation=US rentable=True rented=False verified=True num_gpus={count} gpu_name=RTX_4090 inet_down>=1000 disk_space>=100'
        offers=json.loads(subprocess.check_output([VAST,'search','offers',q,'--order','dph_total','--limit','50','--raw'],text=True,timeout=45))
        candidates=[r for r in offers if r['geolocation']==place and r['machine_id'] not in chosen_machines and
                    int(r['driver_version'].split('.')[0])>=570]
        assert candidates,'No current offer: '+place
        spec=min(candidates,key=lambda r:r['dph_base']+100*r['storage_cost']/720)
        cost=spec['dph_base']+100*spec['storage_cost']/720
        assert sum(r['dph_total'] for r in rows if r['cur_state']!='stopped')+cost<6.8
        label=f'jepa-table-0911-pusht-{offset}-v2'
        write(BASE/f'RESERVATION-pusht-{offset}.json',{'offer_id':spec['id'],'machine_id':spec['machine_id'],
            'geolocation':place,'gpus':count,'hourly_usd':cost,'disk_gb':100,'gpu_offset':offset,
            'label':label,'owner':'rep_geometry_transcoder/root','inet_down':spec['inet_down'],
            'inet_up':spec['inet_up'],'inet_down_cost':spec['inet_down_cost'],'inet_up_cost':spec['inet_up_cost']})
        raw=subprocess.check_output([VAST,'create','instance',str(spec['id']),
            '--image','pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime','--disk','100',
            '--label',label,'--ssh','--direct','--cancel-unavail','--raw'],text=True,timeout=180)
        write(BASE/f'CREATE-pusht-{offset}.json',{'response':raw})
        result=json.loads(raw)
        assert result.get('success') and result.get('new_contract')
        i=result['new_contract'];chosen_machines.add(spec['machine_id'])
        lease['workers'][str(i)]={'label':label,'geolocation':place,'task':'pusht',
             'gpus':count,'hourly_usd':cost,'gpu_offset':offset,'total_gpus':8}
        write(BASE/f'START-{i}.json',result)
        write(BASE/f'LEASE-{i}.json',lease)
        temp=BASE/f'lease-{i}.tmp'
        with temp.open('x') as f:json.dump(lease,f,indent=2)
        temp.replace(BASE/'LEASE.json')
        print(json.dumps({'instance':i,'task':'pusht','gpus':count,'gpu_offset':offset,'hourly_usd':cost}),flush=True)


def rent():
    assert '2026-09-11T12:17Z replacement reservation for table expansion' in Path('/Users/stevenyang/Documents/GPU_RESOURCE_BOARD.md').read_text()
    assert not BASE.exists(), 'Refuse duplicate rental campaign'
    rows=provider()
    assert all(r['cur_state']=='stopped' for r in rows), 'Reconcile live costs first'
    query='geolocation=US rentable=True rented=False verified=True num_gpus>=4 num_gpus<=8 gpu_name=RTX_4090 inet_down>=500 disk_space>=100'
    offers=json.loads(subprocess.check_output([VAST,'search','offers',query,'--order','dph_total','--limit','50','--raw'],text=True,timeout=45))
    selected=[]
    for place,count,task in [('Indiana, US',4,'pointmaze'),('Florida, US',8,'pusht')]:
        candidates=[r for r in offers if r['geolocation']==place and r['num_gpus']==count and
                    int(r['driver_version'].split('.')[0])>=570 and r['reliability2']>.94]
        assert candidates,'No suitable available '+place
        spec=min(candidates,key=lambda r:r['dph_base']+100*r['storage_cost']/720)
        selected.append((spec,task,spec['dph_base']+100*spec['storage_cost']/720))
    assert sum(x[2] for x in selected)<6.8
    BASE.mkdir(parents=True)
    lease={'owner':'rep_geometry_transcoder/root','workers':{},'deadline_timestamp':time.time()+6*3600,
           'hourly_cap_usd':7,'credit_reserve_usd':1,'confirmation_restart':False}
    for spec,task,cost in selected:
        label='jepa-table-0911-'+task+'-v2'
        write(BASE/f'RESERVATION-{task}.json',{'offer_id':spec['id'],'machine_id':spec['machine_id'],
            'geolocation':spec['geolocation'],'gpus':spec['num_gpus'],'hourly_usd':cost,'disk_gb':100,
            'inet_down':spec['inet_down'],'inet_up':spec['inet_up'],
            'inet_down_cost':spec['inet_down_cost'],'inet_up_cost':spec['inet_up_cost'],
            'label':label,'owner':'rep_geometry_transcoder/root'})
        result=json.loads(subprocess.check_output([VAST,'create','instance',str(spec['id']),
            '--image','pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime','--disk','100',
            '--label',label,'--ssh','--direct','--cancel-unavail','--raw'],text=True,timeout=180))
        write(BASE/f'CREATE-{task}.json',result)
        if not result.get('success') or not result.get('new_contract'):
            for i in lease['workers']:subprocess.run([VAST,'stop','instance',i,'--raw'],check=True)
            raise ValueError('Rental unavailable; no duplicate request')
        i=result['new_contract']
        lease['workers'][str(i)]={'label':label,'geolocation':spec['geolocation'],
             'task':task,'gpus':spec['num_gpus'],'hourly_usd':cost}
        write(BASE/f'START-{i}.json',result)
        # Persist each acquired contract immediately, before acquiring another.
        write(BASE/f'LEASE-{i}.json',{**lease,'workers':dict(lease['workers'])})
        print(json.dumps({'instance':i,'task':task,'gpus':spec['num_gpus'],'hourly_usd':cost}),flush=True)
    write(BASE/'LEASE.json',lease)
    subprocess.run(['launchctl','submit','-l','com.steven.jepa.table-expansion.guard.20260911','--',
        '/usr/bin/env','JEPA_TABLE_FLEET_ROOT='+str(BASE),str(PROJECT/'.venv/bin/python'),
        str(Path(__file__).resolve()),'guard'],check=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('activate','guard','rent','add-push'))
    {'activate':activate,'guard':guard,'rent':rent,'add-push':add_push_workers}[p.parse_args().command]()
