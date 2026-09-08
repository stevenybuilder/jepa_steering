"""Bounded CPU-only raw-DROID staging on the owned Virginia worker."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile

ROOT=Path(__file__).resolve().parents[2]
REMOTE='/workspace/jepa-runtime'
CODE='droid-fit-input-code-20260908-v1'
EVIDENCE='droid-fit-input-evidence-20260908-v1'
OUTPUT='droid-fit-download-20260908-v1'


def main():
    workers=json.loads(subprocess.check_output(['vastai','show','instances','--raw'],text=True))
    worker=next(r for r in workers if r['id']==50189244)
    if (worker['actual_status']!='running' or worker['geolocation']!='Virginia, US' or
            sum(r['instance']['totalHour'] for r in workers)>7):
        raise ValueError('Owned US worker/budget changed')
    ssh=['ssh','-i','/tmp/jepa_vast_50123620_ed25519','-o','BatchMode=yes','-o','ConnectTimeout=15',
        '-o','ServerAliveInterval=15','-p',str(worker['ports']['22/tcp'][0]['HostPort']),'root@'+worker['public_ipaddr']]
    subprocess.run(ssh+[f'test ! -e {REMOTE}/{CODE} && test ! -e {REMOTE}/{EVIDENCE} && test ! -e {REMOTE}/{OUTPUT}'],check=True)
    members={}
    for name in ('__init__.py','protocol.py','droid_catalogue.py','droid_fit_inputs.py',
                 'droid_fit_availability.py','droid_fit_download.py'):
        members[f'{CODE}/src/offline_study/{name}']=ROOT/'src/offline_study'/name
    for name in ('test_droid_fit_inputs.py','test_droid_fit_availability.py','test_droid_fit_download.py'):
        members[f'{CODE}/tests/{name}']=ROOT/'tests'/name
    evidence=ROOT/'artifacts/offline_study/droid-fit-inputs-20260908-v4'
    for path in sorted(evidence.rglob('*.json')):
        members[f'{EVIDENCE}/{path.relative_to(evidence)}']=path
    manifest={name:{'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size} for name,path in members.items()}
    proof=ROOT/'artifacts/offline_study/droid-fit-input-stage-20260908-v1';proof.mkdir(exist_ok=False)
    (proof/'FILES.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    with tempfile.TemporaryFile() as stream:
        with tarfile.open(fileobj=stream,mode='w:gz',format=tarfile.USTAR_FORMAT) as archive:
            for name,path in members.items():archive.add(path,arcname=name,recursive=False)
        stream.seek(0);subprocess.run(ssh+[f'tar --keep-old-files -C {REMOTE} -xzf -'],stdin=stream,check=True)
    check="""
import hashlib,json,pathlib,sys
r=pathlib.Path('/workspace/jepa-runtime');members=json.load(sys.stdin)
for name,want in members.items():
 p=r/name
 if p.is_symlink() or '..' in p.parts or p.stat().st_size!=want['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=want['sha256']:raise ValueError('Changed receiving member: '+name)
print(json.dumps({'status':'all_droid_input_stage_members_verified','members':len(members),'gpu_calls':0}))
"""
    result=subprocess.check_output(ssh+[shlex.join(['python','-c',check])],input=json.dumps(manifest),text=True)
    (proof/'RECEIVING.json').write_text(result);print(result,flush=True)
    env=['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',
        f'PYTHONPATH={REMOTE}/{CODE}/src:/workspace/jepa-python/lib/python3.10/site-packages',
        '/workspace/jepa-planning-python/bin/python']
    tests=subprocess.run(ssh+[shlex.join(env+['-m','unittest','discover','-s',f'{REMOTE}/{CODE}/tests','-q'])],
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,check=True)
    (proof/'receiving-tests.log').write_text(tests.stdout);print(tests.stdout,flush=True)
    launch=f"""
import json,pathlib,subprocess
base=pathlib.Path('{REMOTE}')
with (base/'{OUTPUT}.log').open('x') as log:
 p=subprocess.Popen({env!r}+['-u','-m','offline_study.droid_fit_download','--metadata','{REMOTE}/{EVIDENCE}/metadata','--output','{REMOTE}/{OUTPUT}'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({{'status':'cpu_download_launched','instance':50189244,'pid':p.pid,'gpu_calls':0}}))
"""
    result=subprocess.check_output(ssh+[shlex.join(['python','-c',launch])],text=True)
    (proof/'LAUNCH.json').write_text(result);print(result,flush=True)


if __name__=='__main__':main()
