"""CPU-only delivery of the completed native Push-T panel to its paired worker."""
import hashlib
import json
import shlex
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE='/workspace/jepa-runtime/pusht-planning-native-20260908-v1'


def main():
    output=ROOT/'artifacts/offline_study/pusht-reference-delivery-20260908-v1';output.mkdir(exist_ok=False)
    workers=json.loads(subprocess.check_output(['vastai','show','instances','--raw'],text=True))
    connections={}
    for number in (50239185,50245262):
        row=next(r for r in workers if r['id']==number)
        if row['actual_status']!='running' or not row['geolocation'].endswith(', US'):raise ValueError('Expected owned US workers')
        connections[number]=['ssh','-i','/tmp/jepa_vast_50123620_ed25519','-o','BatchMode=yes','-o','ConnectTimeout=15',
            '-o','ServerAliveInterval=15','-p',str(row['ports']['22/tcp'][0]['HostPort']),'root@'+row['public_ipaddr']]
    source,destination=connections[50239185],connections[50245262]
    deadline=time.monotonic()+8*3600
    while subprocess.run(source+['test -f '+BASE+'/DONE.json'],stdout=subprocess.DEVNULL).returncode:
        if time.monotonic()>deadline:raise TimeoutError('Native producer did not complete')
        if subprocess.run(source+['test ! -f '+BASE+'/QUEUE_FAILED.json'],stdout=subprocess.DEVNULL).returncode:
            raise ValueError('Native producer failed; no candidate retry')
        time.sleep(30)
    inventory="""
import pathlib,json,hashlib
r=pathlib.Path('/workspace/jepa-runtime/pusht-planning-native-20260908-v1')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report=json.loads((r/'report.json').read_text())
if json.loads((r/'DONE.json').read_text())['report_sha256']!=sha(r/'report.json') or report['episodes']!=96 or report['status']!='all96_native_pusht_replication_episodes_verified':raise ValueError('Incomplete native panel')
files=[p for p in (r/'native').rglob('*') if p.is_file() and p.name!='progress.json']+[r/'report.json',r/'DONE.json']
if any(p.is_symlink() or p.suffix!='.json' for p in files):raise ValueError('Unexpected native member')
print(json.dumps({str(p.relative_to(r)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(files)}))
"""
    manifest=json.loads(subprocess.check_output(source+[shlex.join(['python','-c',inventory])],text=True))
    (output/'FILES.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    receiver="""
import pathlib,json,hashlib,sys
r=pathlib.Path('/workspace/jepa-runtime/pusht-planning-native-20260908-v1');files=json.load(sys.stdin);missing=[]
for name,want in files.items():
 p=r/name
 if p.is_symlink() or '..' in p.parts:raise ValueError('Unsafe receiving path')
 if not p.exists():missing.append(name);continue
 if p.stat().st_size!=want['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=want['sha256']:raise ValueError('Existing immutable reference changed: '+name)
print(json.dumps(missing))
"""
    missing=json.loads(subprocess.check_output(destination+[shlex.join(['python','-c',receiver])],input=json.dumps(manifest),text=True))
    if missing:
        with tempfile.TemporaryFile() as archive:
            subprocess.run(source+[shlex.join(['tar','-C',BASE,'-czf','-']+missing)],stdout=archive,check=True)
            archive.seek(0)
            with tarfile.open(fileobj=archive,mode='r:gz') as tar:
                names=[]
                for member in tar:
                    if not member.isfile() or member.name not in missing:raise ValueError('Unexpected reference archive member')
                    data=tar.extractfile(member).read();expected=manifest[member.name]
                    if len(data)!=expected['bytes'] or hashlib.sha256(data).hexdigest()!=expected['sha256']:raise ValueError('Reference transport changed')
                    names.append(member.name)
                if sorted(names)!=sorted(missing):raise ValueError('Missing transport member')
            archive.seek(0);subprocess.run(destination+['tar --keep-old-files -C '+BASE+' -xzf -'],stdin=archive,check=True)
    remaining=json.loads(subprocess.check_output(destination+[shlex.join(['python','-c',receiver])],input=json.dumps(manifest),text=True))
    if remaining:raise ValueError('Reference delivery incomplete')
    (output/'DONE.json').write_text(json.dumps({'status':'all96_native_reference_bytes_verified_on_paired_worker',
        'source_instance':50239185,'destination_instance':50245262,'files':len(manifest),
        'manifest_sha256':hashlib.sha256((output/'FILES.json').read_bytes()).hexdigest(),
        'candidate_results_complete':False,'fresh_confirmation':False},indent=2)+'\n')
    print(json.dumps({'status':'native_reference_delivery_complete','files':len(manifest),'new_files':len(missing)}),flush=True)


if __name__=='__main__':main()
