"""Stage owned workers with a single reusable input bundle; launch bounded queues."""
import argparse
import configparser
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

PROJECT=Path(__file__).resolve().parents[2]
BASE=PROJECT/'artifacts/offline_study/table-completion-20260911-v1'
FLEET=BASE/'expansion-v2'
REMOTE='/workspace/table-completion-20260911-v1'
VAST='/Users/stevenyang/.local/bin/vastai'


def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2)


def bundle():
    out=FLEET/'input-bundle.tar.gz'
    if out.exists():
        assert hashlib.sha256(out.read_bytes()).hexdigest()==json.loads((FLEET/'INPUT_BUNDLE.json').read_text())['sha256']
        return out
    paths=[]
    for prefix,root in [('code/src',PROJECT/'src'),('code/vendor/jepa-wms',PROJECT/'vendor/jepa-wms')]:
        for p in sorted(root.rglob('*')):
            relative=p.relative_to(root)
            if (not p.is_file() or p.is_symlink() or '__pycache__' in relative.parts or
                p.name.startswith('._') or '.env'==p.name or
                ('.git' in relative.parts and (p.name=='config' or 'logs' in relative.parts or 'hooks' in relative.parts))):continue
            paths.append((p,str(Path(prefix)/relative)))
    for p in sorted((BASE/'restored').rglob('*')):
        if not p.is_file() or p.is_symlink() or '__pycache__' in p.parts:continue
        relative=p.relative_to(BASE/'restored')
        if relative.name=='RESTORED_VERIFIED.json' or (len(relative.parts)>3 and relative.parts[3].startswith(('navigation-','pusht-'))):
            paths.append((p,'restored/'+str(relative)))
    for name in ['bootstrap_table_completion.sh','table_resume_streams.py','table_pusht_inputs.py','table_worker_supervisor.py']:
        p=PROJECT/'scripts/vast'/name;paths.append((p,'code/scripts/vast/'+name))
    p=PROJECT/'artifacts/offline_study/pusht-native-recovery-20260908-v1/PUBLIC_INPUTS.json'
    paths.append((p,'code/'+str(p.relative_to(PROJECT))))
    manifest={name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p,name in paths}
    assert len(manifest)==len(paths)
    with tarfile.open(out,'w:gz',compresslevel=1) as t:
        for p,name in paths:t.add(p,arcname=name,recursive=False)
    write(FLEET/'INPUT_BUNDLE.json',{'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'manifest':manifest})
    print(json.dumps({'bundle_bytes':out.stat().st_size,'files':len(paths)}),flush=True)
    return out


def stage(instance,prepare_only=False):
    lease=json.loads((FLEET/'LEASE.json').read_text());spec=lease['workers'][str(instance)]
    local=FLEET/str(instance);local.mkdir(exist_ok=True)
    if (local/'LAUNCH.json').exists():return {'instance':instance,'already_launched':True}
    deadline=time.time()+600
    while True:
        rows=json.loads(subprocess.check_output([VAST,'show','instances','--raw'],text=True,timeout=45))
        row=next(r for r in rows if r['id']==instance)
        assert row['label']==spec['label'] and row['geolocation']==spec['geolocation'] and row['cur_state']=='running'
        if row['actual_status']=='running' and row.get('ports',{}).get('22/tcp'):break
        if time.time()>deadline:raise RuntimeError('Receiving image startup exceeded10minutes')
        time.sleep(10)
    ssh=['ssh','-i','/Users/stevenyang/.ssh/id_ed25519','-o','BatchMode=yes',
        '-o','ConnectTimeout=15','-o','StrictHostKeyChecking=accept-new',
        '-p',str(row['ports']['22/tcp'][0]['HostPort']),'root@'+row['public_ipaddr']]
    if not (local/'CONNECTION.json').exists():
        write(local/'CONNECTION.json',{'instance':instance,'ssh':ssh,'label':spec['label'],'geolocation':spec['geolocation']})
    for attempt in range(12):
        if subprocess.run(ssh+['true'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=25).returncode==0:break
        time.sleep(5)
    else:raise RuntimeError('Advertised SSH endpoint did not become reachable')
    # New root only; neither overwrite nor execute any earlier confirmation tree.
    subprocess.run(ssh+['mkdir -p '+REMOTE],check=True,timeout=30)
    proof=json.loads((FLEET/'INPUT_BUNDLE.json').read_text())
    if (FLEET/'INPUT_DRIVE_VERIFIED.json').exists():
        cloud=json.loads((FLEET/'INPUT_DRIVE_VERIFIED.json').read_text())
        cfg=configparser.ConfigParser(interpolation=None);cfg.read('/Users/stevenyang/.config/rclone/rclone.conf')
        access=json.loads(cfg['gdrive']['token'])['access_token']
        download="""import hashlib,json,sys,urllib.request,subprocess
from pathlib import Path
d=json.load(sys.stdin);r=Path('/workspace/table-completion-20260911-v1')
headers={'Authorization':'Bearer '+d['access'],'X-Goog-User-Project':'project-flash-490419'}
url='https://www.googleapis.com/drive/v3/files/'+d['file_id']
with urllib.request.urlopen(urllib.request.Request(url+'?fields=id,name,size,sha256Checksum,mimeType,parents,trashed',headers=headers),timeout=60) as s:m=json.load(s)
assert m['id']==d['file_id'] and not m.get('trashed') and int(m['size'])==d['bytes'] and m['sha256Checksum']==d['sha256'] and m['parents']==['14zoPlI5qViiF5DwqVkCyKUu2O-Bj8u48']
p=r/'receiving-input-bundle.tar.gz'
if not p.exists():
 with urllib.request.urlopen(urllib.request.Request(url+'?alt=media',headers=headers),timeout=90) as s,p.open('xb') as f:
  while b:=s.read(4<<20):f.write(b)
assert p.stat().st_size==d['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==d['sha256']
subprocess.run(['tar','--no-same-owner','--skip-old-files','-xzf',str(p),'-C',str(r)],check=True)
print(json.dumps({'direct_drive_receiving_bytes':p.stat().st_size}))
"""
        subprocess.run(ssh+[shlex.join(['python3','-c',download])],input=json.dumps({'access':access,
            'file_id':cloud['drive_file_id'],'bytes':proof['bytes'],'sha256':proof['sha256']}),text=True,check=True,timeout=600)
    else:
        with (FLEET/'input-bundle.tar.gz').open('rb') as f:
            subprocess.run(ssh+['tar --no-same-owner --skip-old-files -xzf - -C '+REMOTE],stdin=f,check=True,timeout=600)
    code="""import hashlib,json,sys,subprocess
from pathlib import Path
r=Path('/workspace/table-completion-20260911-v1');m=json.load(sys.stdin)
for n,v in m.items():
 p=r/n
 assert p.is_file() and not p.is_symlink() and p.stat().st_size==v['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256'],n
if (r/'RECEIVED_FILES.json').exists():assert json.loads((r/'RECEIVED_FILES.json').read_text())==m
else:
 with (r/'RECEIVED_FILES.json').open('x') as f:json.dump(m,f)
assert subprocess.check_output(['git','-C',str(r/'code/vendor/jepa-wms'),'rev-parse','HEAD'],text=True).strip()=='13cf1d9c7e476f53c17714d2e0f1dc239a883ce0'
assert not subprocess.check_output(['git','-C',str(r/'code/vendor/jepa-wms'),'status','--porcelain','--untracked-files=no']).strip()
print(json.dumps({'received_members_verified':len(m)}))
"""
    result=subprocess.check_output(ssh+[shlex.join(['python3','-c',code])],input=json.dumps(proof['manifest']),text=True,timeout=180)
    if not (local/'RECEIVED.json').exists():write(local/'RECEIVED.json',json.loads(result))
    if prepare_only:
        if not (local/'READY_FOR_RECOVERY.json').exists():write(local/'READY_FOR_RECOVERY.json',{'received':True,'scientific_episodes':0})
        print(json.dumps({'instance':instance,'ready_for_corrected_bootstrap':True}),flush=True)
        return
    args=['python3','-u',REMOTE+'/code/scripts/vast/table_worker_supervisor.py','--task',spec['task'],
          '--instance',str(instance),'--gpus',str(spec['gpus']),'--deadline',str(lease['deadline_timestamp']),
          '--gpu-offset',str(spec.get('gpu_offset',0)),'--total-gpus',str(spec.get('total_gpus',spec['gpus']))]
    launch="""import json,subprocess,sys
from pathlib import Path
r=Path('/workspace/table-completion-20260911-v1');cmd=json.load(sys.stdin)
with (r/'expansion-supervisor.log').open('x') as f:
 p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
record={'pid':p.pid,'command':cmd,'stage':'bootstrap_then_per_gpu_receiving_then_scientific_queue'}
with (r/'EXPANSION_LAUNCH.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record))
"""
    result=json.loads(subprocess.check_output(ssh+[shlex.join(['python3','-c',launch])],input=json.dumps(args),text=True,timeout=45))
    write(local/'LAUNCH.json',result)
    print(json.dumps({'instance':instance,'task':spec['task'],'supervisor_pid':result['pid']}),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle-only',action='store_true');p.add_argument('--instances',nargs='+',type=int)
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();bundle()
    if not a.bundle_only:
        with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda i:stage(i,a.prepare_only),a.instances))
