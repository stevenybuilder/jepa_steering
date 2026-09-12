"""Retry the failed pre-fit handoff, retaining original source and downloaded bytes."""
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile
import time
import component_extension_fleet as f


def main():
    assert 'September11 Wall pre-fit handoff recovery' in Path('/Users/stevenyang/Documents/GPU_RESOURCE_BOARD.md').read_text()
    row=next(r for r in f.provider('show','instances') if r['id']==50632757)
    assert row['label']=='jepa-mw-components-0911-slot2-v3' and row['geolocation']=='Missouri, US' and row['actual_status']=='running'
    root='/workspace/refined-nav-wall-prerequisite-20260911-v2'
    local=f.PROJECT/'artifacts/offline_study/table-completion-20260911-v1/refined-wall-worker-v2'
    local.mkdir(parents=True,exist_ok=False)
    ssh=f.connection(row)
    subprocess.run(ssh+['test ! -e '+root+' && mkdir -p '+root+'/ops'],check=True,timeout=30)
    with tempfile.TemporaryFile() as stream:
        with tarfile.open(fileobj=stream,mode='w') as archive:
            for name in ('component_boundary_job.py','component_queue_handoff.py'):archive.add(Path(__file__).with_name(name),arcname=name,recursive=False)
        stream.seek(0);subprocess.run(ssh+['tar --keep-old-files --no-same-owner -x -C '+root+'/ops'],stdin=stream,check=True,timeout=30)
    subprocess.run(['launchctl','submit','-l','com.steven.jepa.wall.prerequisite.collect.v2.20260911','--','/usr/bin/env',
        'PATH=/usr/local/bin:/usr/bin:/bin:/Users/stevenyang/.local/bin',f.PYTHON,
        str(Path(__file__).with_name('collect_refined_prerequisite.py')),'--task','wall-v2'],check=True)
    script='''import json,os,sys,shutil,hashlib,subprocess,time
from pathlib import Path
r=Path(sys.argv[1]);old=r.with_name('refined-nav-wall-prerequisite-20260911-v1')
assert json.loads((old/'TERMINAL.json').read_text())['status']=='incomplete_preserve_all'
assert not (old/'FIT_STARTED.json').exists() and not (old/'fit-v1').exists()
for name in ('code','original','cohort'):shutil.copytree(old/name,r/name,ignore=shutil.ignore_patterns('__pycache__'))
shutil.copytree(old/'data',r/'data',copy_function=os.link)
os.link(old/'checkpoint.pth.tar',r/'checkpoint.pth.tar')
job=json.loads((old/'JOB.json').read_text());job['command'][-1]=str(r);job['cwd']=str(r/'code')
for name in ('component_boundary_job.py','component_queue_handoff.py'):
 p=r/'ops'/name;job['files']['ops/'+name]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
with (r/'JOB.json').open('x') as out:json.dump(job,out,indent=2)
sys.path.insert(0,str(r/'ops'))
from component_boundary_job import validate_job,validate_queue
from component_queue_handoff import process
validate_job(job,r);queues=[];component=Path('/workspace/metaworld-components-20260911-v1')
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:validate_queue(process(int(p.name)),component)
 except (ValueError,IndexError):continue
 queues.append(int(p.name))
assert len(queues)==1
with (r/'boundary.log').open('x') as log:
 p=subprocess.Popen(['/workspace/component-python/bin/python','-u',str(r/'ops/component_boundary_job.py'),'--job',str(r/'JOB.json'),'--component-root',str(component),'--queue-pid',str(queues[0])],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
receipt={'pid':p.pid,'time':time.time(),'unchanged_scientific_source_and_data':True}
with (r/'BOUNDARY_LAUNCH.json').open('x') as out:json.dump(receipt,out)
print(json.dumps(receipt))'''
    result=json.loads(subprocess.check_output(ssh+[shlex.join(['python3','-c',script,root])],text=True,timeout=90))
    f.write(local/'LAUNCH.json',result);print(json.dumps(result))


if __name__=='__main__':main()
