"""CPU-only preparation on the owned eight-GPU US MetaWorld worker."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile

ROOT=Path(__file__).resolve().parents[2]
REMOTE='/workspace/jepa-runtime'
CODE='hmm-fixed-response-code-20260908-v3'
EVIDENCE='hmm-fixed-response-evidence-20260908-v3'
PANEL='hmm-fixed-response-behavior-20260908-v3'


def main():
    workers=json.loads(subprocess.check_output(['vastai','show','instances','--raw'],text=True))
    worker=next(r for r in workers if r['id']==50233992)
    if (worker['actual_status']!='running' or not worker['geolocation'].endswith(', US') or
            worker['label']!='jepa-fixed-behavior-us-v1' or sum(r['instance']['totalHour'] for r in workers)>7):
        raise ValueError('US owned lease/budget changed')
    ssh=['ssh','-i','/tmp/jepa_vast_50123620_ed25519','-o','BatchMode=yes','-o','ConnectTimeout=15',
        '-o','ServerAliveInterval=15','-p',str(worker['ports']['22/tcp'][0]['HostPort']),'root@'+worker['public_ipaddr']]
    subprocess.run(ssh+[f'test ! -e {REMOTE}/{CODE} && test ! -e {REMOTE}/{EVIDENCE} && test ! -e {REMOTE}/{PANEL}'],check=True)
    members={}
    original=ROOT/'artifacts/offline_study/hmm-reference-source-20260908-v1'
    digest=hashlib.sha256()
    for path in sorted(original.glob('*.py')):
        digest.update(path.name.encode()+b'\0'+path.read_bytes())
        members[f'{CODE}/src/offline_study/{path.name}']=path
    if digest.hexdigest()!='fc7578672eabd8e08469afcc1f7f4d9b1c2ed18c447f4f003539b61da7950f58':
        raise ValueError('Original MetaWorld source snapshot changed')
    for name in ('routing_hmm.py','routing_fit.py','routing_intervention.py','routing_behavior.py','routing_analysis.py','routing_one_pass.py'):
        if (original/name).exists():raise ValueError('Not an additive HMM module')
        members[f'{CODE}/src/offline_study/{name}']=ROOT/'src/offline_study'/name
    for name in ('test_routing_hmm.py','test_routing_intervention.py','test_routing_behavior.py','test_routing_one_pass.py','test_fixed_response.py'):
        members[f'{CODE}/tests/{name}']=ROOT/'tests'/name
    archived='archive/2026-09-07-workspace/scripts/geometry_map/run_reach_hmm_routing.py'
    members[f'{CODE}/{archived}']=ROOT/archived
    members[f'{CODE}/configs/hmm_fixed_response.json']=ROOT/'configs/hmm_fixed_response.json'
    members[f'{CODE}/docs/HMM_FIXED_RESPONSE_BEHAVIOR.md']=ROOT/'docs/HMM_FIXED_RESPONSE_BEHAVIOR.md'
    members['run_routing_queue_v3.py']=ROOT/'scripts/vast/run_routing_queue.py'
    members['finish_routing_panel_v3.py']=ROOT/'scripts/vast/finish_routing_panel.py'
    fits=ROOT/'artifacts/offline_study/hmm-fixed-response-20260908-v1/fits'
    for task in ('reach','reach-wall'):
        for name in ('model.pt','protocol.json','FROZEN.json','diagnostics.json','report.json','DONE.json'):
            members[f'{EVIDENCE}/fits/{task}/{name}']=fits/task/name
    manifest={name:{'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size} for name,path in members.items()}
    output=ROOT/'artifacts/offline_study/hmm-fixed-response-preparation-20260908-v3';output.mkdir(exist_ok=False)
    (output/'FILES.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    with tempfile.TemporaryFile() as stream:
        with tarfile.open(fileobj=stream,mode='w:gz',format=tarfile.USTAR_FORMAT) as archive:
            for name,path in members.items():archive.add(path,arcname=name,recursive=False)
        stream.seek(0);subprocess.run(ssh+[f'tar --keep-old-files -C {REMOTE} -xzf -'],stdin=stream,check=True)
    check="""
import hashlib,json,pathlib,sys
r=pathlib.Path('/workspace/jepa-runtime'); members=json.load(sys.stdin)
for name,want in members.items():
 p=r/name
 if p.is_symlink() or '..' in p.parts or p.stat().st_size!=want['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=want['sha256']:raise ValueError('Changed receiving member: '+name)
print(json.dumps({'status':'all_routing_preparation_members_verified','members':len(members),'gpu_job_launched':False}))
"""
    result=subprocess.check_output(ssh+[shlex.join(['python','-c',check])],input=json.dumps(manifest),text=True)
    (output/'RECEIVING.json').write_text(result);print(result,flush=True)
    env=['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',
        f'PYTHONPATH={REMOTE}/{CODE}/src:/workspace/jepa-python/lib/python3.10/site-packages',
        'LD_LIBRARY_PATH=/opt/conda/lib','/workspace/jepa-planning-python/bin/python']
    proof=subprocess.run(ssh+[shlex.join(env+['-m','unittest','discover','-s',f'{REMOTE}/{CODE}/tests','-p','test_routing*.py','-v'])],
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,check=True)
    (output/'receiving-tests.log').write_text(proof.stdout);print(proof.stdout,flush=True)
    command="""
import importlib.util,subprocess,sys
spec=importlib.util.spec_from_file_location('queue','/workspace/jepa-runtime/run_routing_queue_v3.py');q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
args=q.arguments();i=args.index('--freeze');del args[i:i+2]
subprocess.run([sys.executable,'-m','offline_study.routing_behavior','freeze']+args+['--output','/workspace/jepa-runtime/hmm-fixed-response-behavior-20260908-v3/freeze'],check=True)
"""
    subprocess.run(ssh+[shlex.join(env+['-c',command])],check=True,timeout=300)
    (output/'freeze').mkdir()
    for name in ('protocol.json','FROZEN.json'):
        value=subprocess.check_output(ssh+[shlex.join(['cat',f'{REMOTE}/{PANEL}/freeze/{name}'])])
        (output/'freeze'/name).write_bytes(value)
    digest=hashlib.sha256((output/'freeze/protocol.json').read_bytes()).hexdigest()
    if digest!=json.loads((output/'freeze/FROZEN.json').read_text())['protocol_sha256']:raise ValueError('Receiving freeze changed')
    (output/'DONE.json').write_text(json.dumps({'status':'receiving_cpu_source_fit_tests_and_pre_outcome_freeze_complete',
        'protocol_sha256':digest,'instance':50233992,'gpu_engineering_complete':False,'scientific_jobs_launched':False},indent=2)+'\n')
    print(json.dumps({'status':'prepared_not_gpu_engineered','protocol_sha256':digest}),flush=True)


if __name__=='__main__':main()
