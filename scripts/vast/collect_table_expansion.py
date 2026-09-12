"""Preserve periodic raw snapshots and stop terminal owned workers after Drive verification.

Uses absolute tool paths. All archive members are immutable byte snapshots;
partial episodes remain partial and never count as completed comparisons.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from final_preservation_drive import upload_direct,verify,digest
from backup_results_to_google import verify_archive

PROJECT=Path(__file__).resolve().parents[2]
BASE=PROJECT/'artifacts/offline_study/table-completion-20260911-v1/expansion-v2'
VAST='/Users/stevenyang/.local/bin/vastai'
REMOTE='/workspace/table-completion-20260911-v1'
os.environ['PATH']='/usr/local/bin:/usr/bin:/bin:/Users/stevenyang/.local/bin'


def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():
        with p.open('x') as f:json.dump(x,f,indent=2)


def remote(ssh,code):
    return json.loads(subprocess.check_output(ssh+[shlex.join(['python3','-c',code])],text=True,timeout=120))


BUILD=r'''
import json,hashlib,tarfile,io,time
from pathlib import Path
r=Path('/workspace/table-completion-20260911-v1')
files=[p for p in r.glob('expansion-20260911-v1/**/*') if p.is_file()]
files += [p for p in r.glob('expansion-*.log') if p.is_file()]
files += [p for p in r.glob('EXPANSION_*.json') if p.is_file()]
files += [p for p in [r/'packages.txt',r/'PUSHT_INPUTS_VERIFIED.json',r/'RECEIVED_FILES.json'] if p.is_file()]
files += list((r/'code/scripts/vast').glob('table*.py'))
files += list((r/'code/scripts/vast').glob('bootstrap_table*.sh'))
files += [p for p in [r/'BOOTSTRAP_REPAIR.json',r/'packages-recovery-v2.txt'] if p.is_file()]
files += [p for p in (r/'pointmaze-runtime-repair-v1').glob('*') if p.is_file() and p.suffix in ('.json','.log','.py','.sh','.pth')]
assert files
folder=r/'expansion-snapshots';folder.mkdir(exist_ok=True)
archive=folder/('snapshot-'+str(time.time_ns())+'.tar.gz');manifest={}
with tarfile.open(archive,'w:gz') as t:
 for p in sorted(files):
  assert not p.is_symlink()
  raw=p.read_bytes();name=str(p.relative_to(r));assert name not in manifest
  manifest[name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
  member=tarfile.TarInfo(name);member.size=len(raw);t.addfile(member,io.BytesIO(raw))
print(json.dumps({'path':str(archive),'bytes':archive.stat().st_size(),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'manifest':manifest}))
'''.replace("archive.stat().st_size()","archive.stat().st_size")


def collect(instance,spec):
    local=BASE/instance;last=-1;last_time=0
    while True:
        try:
            rows=json.loads(subprocess.check_output([VAST,'show','instances','--raw'],text=True,timeout=45))
            row=next(r for r in rows if str(r['id'])==instance)
            assert row['label']==spec['label'] and row['geolocation']==spec['geolocation']
            if row['cur_state']=='stopped':
                write(local/'STOPPED_OBSERVED.json',{'source_disk_retained':True,'verified_final_archive':(local/'FINAL_DRIVE_VERIFIED.json').exists()})
                return
            if not (local/'LAUNCH.json').exists():time.sleep(15);continue
            ssh=json.loads((local/'CONNECTION.json').read_text())['ssh']
            state=remote(ssh,"from pathlib import Path;import json;r=Path('"+REMOTE+"');prefix='EXPANSION_RECOVERY_' if (r/'EXPANSION_RECOVERY_LAUNCH.json').exists() else 'EXPANSION_';print(json.dumps({'done':(r/(prefix+'DONE.json')).exists(),'failed':(r/(prefix+'FAILED.json')).exists(),'episodes':len(list(r.glob('expansion-20260911-v1/*/conditions/*/shard-*/episode-*.json'))),'streams':len(list(r.glob('expansion-20260911-v1/*/conditions/*/shard-*/DONE.json')))}))")
            terminal=state['done'] or state['failed']
            if spec.get('additional_queues'):
                queues=remote(ssh,"from pathlib import Path;import json;r=Path('"+REMOTE+"');names="+
                    repr(spec['additional_queues'])+";print(json.dumps({n:{'done':(r/n/'QUEUE_DONE.json').exists(),'failed':(r/n/'QUEUE_FAILED.json').exists()} for n in names}))")
                state['additional_queues']=queues
                terminal=terminal and all(q['done'] or q['failed'] for q in queues.values())
            if terminal or (state['streams']!=last and time.time()-last_time>180):
                snapshot=remote(ssh,BUILD);assert snapshot['bytes']<500<<20
                target=local/Path(snapshot['path']).name
                with target.open('xb') as f:
                    subprocess.run(ssh+['cat '+shlex.quote(snapshot['path'])],stdout=f,check=True,timeout=180)
                verify_archive(['cat',str(target)],snapshot['sha256'],snapshot['bytes'],snapshot['manifest'])
                subprocess.run(['/usr/local/bin/rclone','about','gdrive:','--json'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True,timeout=60)
                name=f'jepa-table-{instance}-{target.name}'
                obj=upload_direct(target,name,target.with_suffix('.uploaded.json'))
                verified=verify(obj['id'],name,snapshot['bytes'],snapshot['sha256'])
                receipt={**verified,'drive_file_id':obj['id'],'archive_sha256':snapshot['sha256'],
                         'manifest':snapshot['manifest'],'state':state,'source_disk_retained':True}
                write(target.with_suffix('.verified.json'),receipt)
                print(json.dumps({'instance':instance,'snapshot_verified':True,**state}),flush=True)
                last=state['streams'];last_time=time.time()
                if terminal:
                    write(local/'FINAL_DRIVE_VERIFIED.json',receipt)
                    result=subprocess.check_output([VAST,'stop','instance',instance,'--raw'],text=True,timeout=90)
                    write(local/'STOP_RESPONSE.json',{'response':result,'source_disk_retained':True})
            time.sleep(30)
        except Exception as e:
            print(json.dumps({'instance':instance,'collector_error':type(e).__name__,'detail':str(e)[:220]}),flush=True)
            time.sleep(30)


def main():
    lease=json.loads((BASE/'LEASE.json').read_text())
    # Preserve code/fits/reference inputs once, shared by all four workers.
    target=BASE/'input-bundle.tar.gz';proof=json.loads((BASE/'INPUT_BUNDLE.json').read_text())
    assert digest(target)==proof['sha256']
    subprocess.run(['/usr/local/bin/rclone','about','gdrive:','--json'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True,timeout=60)
    obj=upload_direct(target,'jepa-table-expansion-20260911-v2-input-bundle.tar.gz',BASE/'INPUT_UPLOADED.json')
    result=verify(obj['id'],'jepa-table-expansion-20260911-v2-input-bundle.tar.gz',proof['bytes'],proof['sha256'])
    write(BASE/'INPUT_DRIVE_VERIFIED.json',{**result,'drive_file_id':obj['id'],'archive_sha256':proof['sha256']})
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda kv:collect(*kv),lease['workers'].items()))
    write(BASE/'COLLECTOR_DONE.json',{'workers_terminal':True,'full_study_complete':False})
    subprocess.run(['launchctl','remove','com.steven.jepa.table-expansion.collect.20260911'],check=False)


if __name__=='__main__':main()
