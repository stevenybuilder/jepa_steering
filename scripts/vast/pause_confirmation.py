"""User-directed interruption and preservation of the exact new confirmation fleet.

No new experiments, instance deletion, old-instance mutation, or efficacy analysis.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import shlex
import time

import manage_confirmation as m

IDS = {50544130, 50546169, 50546170, 50546171, 50546172}
REMOTE_STOP = r'''
import json, os, signal, time
from pathlib import Path
target='/workspace/confirmation/scripts/vast/run_confirmation_worker.py'
def queues():
    found=[]
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:args=path.read_bytes().split(b'\0')
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
        if target.encode() in args:found.append(int(path.parent.name))
    return found
before=queues()
if len(before)>1:raise ValueError('Multiple supervisors; refuse ambiguous interruption')
for pid in before:os.kill(pid,signal.SIGTERM)
for _ in range(45):
    if not queues():break
    time.sleep(1)
else:raise RuntimeError('Owned queue did not terminate; retain source')
receipt={'reason':'user_requested_pause_not_scientific_failure','queue_pids_signalled':before,
         'queue_pids_remaining':queues(),'full_confirmation_complete':False,
         'automatic_restart_authorized':False,'stopped_timestamp':time.time()}
path=Path('/workspace/confirmation-output/USER_INTERRUPTED.json')
if not path.exists():
    with path.open('x') as f:json.dump(receipt,f,indent=2,sort_keys=True)
print(json.dumps(receipt))
'''


def quiesce(lease):
    row=m.provider()[lease['id']];m.identity(lease,row)
    receipt=m.command(m.ssh(row)+[shlex.join(['python3','-c',REMOTE_STOP])],75)
    output=m.ROOT/'pause'/str(lease['id'])
    output.mkdir(parents=True,exist_ok=True)
    m.write(output/'QUEUE_INTERRUPTED.json',json.loads(receipt.strip().splitlines()[-1]))
    for _ in range(12):
        gpu=m.command(m.ssh(row)+['nvidia-smi --query-compute-apps=pid --format=csv,noheader'],30)
        terminal=m.command(m.ssh(row)+['test -f /workspace/confirmation-launch-exit.json && echo TERMINAL || echo WAITING'],30)
        if not any(s.strip().isdigit() for s in gpu.splitlines()) and 'TERMINAL' in terminal.split():break
        time.sleep(2)
    else:raise RuntimeError('GPU or launcher still active; preserve source disk')
    local=m.mirror(lease,row)
    count=len(list((local/'results').glob('*/*/rank-*/episode-*.json')))
    m.write(output/'QUIESCENT_MIRROR.json',{'id':lease['id'],'utc':m.stamp(),
        'complete_episode_records':count,'gpu_processes_empty':True,'launcher_terminal':True,
        'full_confirmation_complete':False})
    print(json.dumps({'science_stopped':lease['id'],'preserved_episode_records':count}),flush=True)


def preserve_stop(lease):
    row=m.provider()[lease['id']];m.identity(lease,row)
    m.preserve(lease,row)
    proof=json.loads((m.ROOT/'closeout'/str(lease['id'])/'DRIVE_VERIFIED.json').read_text())
    row=m.provider()[lease['id']];m.identity(lease,row)
    response=m.command([m.VAST,'stop','instance',str(lease['id']),'--raw'],120)
    for _ in range(20):
        row=m.provider().get(lease['id'])
        if row is None:raise ValueError('Source disk vanished; no deletion authorized')
        m.identity(lease,row)
        if row['actual_status'] in ('exited','stopped'):
            result={'id':lease['id'],'utc':m.stamp(),'status':row['actual_status'],
                'drive_file_id':proof['file_id'],'drive_verified_before_stop':True,
                'disk_retained':True,'full_confirmation_complete':False,'response':response}
            m.write(m.ROOT/'pause'/str(lease['id'])/'INSTANCE_STOPPED.json',result)
            print(json.dumps(result),flush=True)
            return result
        time.sleep(3)
    raise RuntimeError('Provider stop not confirmed')


def main():
    leases=m.leases()
    if {r['id'] for r in leases}!=IDS:raise ValueError('New fleet differs from explicit stop scope')
    m.write(m.ROOT/'USER_PAUSE.json',{'utc':m.stamp(),'requested':'Stop the runs for now',
        'instance_ids':sorted(IDS),'full_confirmation_complete':False,'automatic_restart_authorized':False,
        'preserve_drive_then_stop':True,'destroy_instances':False,'partial_results_not_primary_confirmation':True})
    with ThreadPoolExecutor(max_workers=5) as pool:list(pool.map(quiesce,leases))
    # Keep at most two archive downloads on this low-disk laptop simultaneously.
    results=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(preserve_stop,lease) for lease in leases]):results.append(future.result())
    m.write(m.ROOT/'PAUSED.json',{'utc':m.stamp(),'instances':results,'science_running':False,
        'full_confirmation_complete':False,'raw_results_and_logs_drive_verified':True,
        'retained_storage_may_still_be_billed':True})


if __name__=='__main__':main()
