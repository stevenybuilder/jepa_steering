"""Independent local collector: preserve terminal Wall queue, verify Drive, stop.

No new scientific launches. On a preservation error, source disk is retained and
the independent provider budget backstop still stops compute by12:00UTC.
"""
import json
from pathlib import Path
import shlex
import subprocess
import time

from final_preservation_drive import upload_direct, verify, digest
from backup_results_to_google import verify_archive
from table_completion_control import provider, VAST, write

PROJECT=Path(__file__).resolve().parents[2]
LOCAL=PROJECT/'artifacts/offline_study/table-completion-20260911-v1/wall-closeout'
REMOTE='/workspace/table-completion-20260911-v1'
SSH=['ssh','-i','/Users/stevenyang/.ssh/id_ed25519','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','41811','root@174.164.26.93']


def remote(code,timeout=60):
    return json.loads(subprocess.check_output(SSH+[shlex.join(['python3','-c',code])],text=True,timeout=timeout))


def main():
    LOCAL.mkdir(exist_ok=False)
    write(LOCAL/'LAUNCH.json',{'pid':__import__('os').getpid(),'instance':50561030,'deadline_timestamp':1789128000.0,'source_sha256':digest(Path(__file__))})
    while time.time()<1789128000.0:
        try:
            state=remote("from pathlib import Path;import json;r=Path('"+REMOTE+"/wall-resume-20260911-v1');print(json.dumps({'done':(r/'QUEUE_DONE.json').exists(),'failed':(r/'QUEUE_FAILED.json').exists(),'episodes':len(list(r.glob('conditions/*/shard-*/episode-*.json')))}))")
            print(json.dumps(state),flush=True)
            if state['done'] or state['failed']:break
        except (subprocess.SubprocessError,ValueError) as exc:
            print(json.dumps({'observation_error':type(exc).__name__}),flush=True)
        time.sleep(30)
    else:
        write(LOCAL/'BUDGET_STOP_HANDOFF.json',{'source_disk_retained':True,'collector_did_not_establish_completion':True})
        return
    # These are explicit new outputs plus receiving checks and source, not raw
    # datasets or secret configuration. Older restored evidence is already in
    # Drive and its verification markers bind the recovery paths in PLAN.json.
    build=r'''
import json,hashlib,tarfile
from pathlib import Path
r=Path('/workspace/table-completion-20260911-v1')
names=['wall-resume-20260911-v1','wall-resume-20260911-v1.log',
 'wall-receiving-engineering-v2','wall-receiving-engineering-v2.log',
 'wall-receiving-engineering-v1.log','WALL_ENGINEERING_LAUNCH.json',
 'WALL_ENGINEERING_LAUNCH_V2.json','WALL_RESUME_LAUNCH.json',
 'code/scripts/vast/resume_wall_table.py','code/scripts/vast/collect_wall_table.py',
 'packages.txt','bootstrap-final-inherited.log']
files=[]
for name in names:
 p=r/name
 if not p.exists():raise ValueError('Missing preservation target: '+name)
 files.extend([q for q in p.rglob('*') if q.is_file()] if p.is_dir() else [p])
files.extend((r/'restored').glob('*/*/RESTORED_VERIFIED.json'))
manifest={}
for p in sorted(files):
 if p.is_symlink() or p.name in ('.env','rclone.conf'):raise ValueError('Unsafe output member')
 manifest[str(p.relative_to(r))]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
out=r/'wall-table-closeout-20260911-v1';out.mkdir(exist_ok=False)
with (out/'FILES.json').open('x') as f:json.dump(manifest,f,indent=2)
archive=out/'wall-table-resume-20260911-v1.tar.gz'
with tarfile.open(archive,'w:gz') as t:
 for name in manifest:t.add(r/name,arcname=name,recursive=False)
print(json.dumps({'path':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'manifest':manifest}))
'''
    archive=remote(build,timeout=300)
    write(LOCAL/'WORKER_ARCHIVE.json',archive)
    target=LOCAL/'wall-table-resume-20260911-v1.tar.gz'
    assert archive['bytes']<500<<20,'Unexpected preservation size'
    subprocess.run(['scp','-i','/Users/stevenyang/.ssh/id_ed25519','-o','BatchMode=yes','-P','41811',
        'root@174.164.26.93:'+archive['path'],str(target)],check=True,timeout=300)
    assert digest(target)==archive['sha256'] and target.stat().st_size==archive['bytes']
    verify_archive(['cat',str(target)],archive['sha256'],archive['bytes'],archive['manifest'])
    subprocess.run(['rclone','about','gdrive:','--json'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=60)
    uploaded=upload_direct(target,target.name,LOCAL/'UPLOADED.json')
    verified=verify(uploaded['id'],target.name,archive['bytes'],archive['sha256'])
    write(LOCAL/'DRIVE_VERIFIED.json',{**verified,'archive_sha256':archive['sha256'],'files':len(archive['manifest']),
        'queue_state':state,'full_six_task_study_complete':False})
    upload_direct(LOCAL/'DRIVE_VERIFIED.json','wall-table-resume-20260911-v1-DRIVE_VERIFIED.json',LOCAL/'RECEIPT_UPLOADED.json')
    row=next(r for r in provider() if r['id']==50561030)
    assert row['label']=='jepa-table-completion-0911-v1' and row['geolocation']=='Washington, US'
    response=subprocess.check_output([VAST,'stop','instance','50561030','--raw'],text=True,timeout=90)
    write(LOCAL/'STOP_RESPONSE.json',{'response':response,'source_disk_retained':True})
    for _ in range(10):
        row=next(r for r in provider() if r['id']==50561030)
        if row['cur_state']=='stopped':
            write(LOCAL.parent/'REPLACEMENT_STOPPED_VERIFIED.json',{'instance':50561030,'state':row['cur_state'],
                'actual_status':row['actual_status'],'source_disk_retained':True,'drive_verified':True})
            print(json.dumps({'status':'new_wall_results_drive_verified_compute_stopped','full_study_complete':False}),flush=True)
            return
        time.sleep(10)
    raise RuntimeError('Provider stop not verified; independent backstop remains')


if __name__=='__main__':main()
