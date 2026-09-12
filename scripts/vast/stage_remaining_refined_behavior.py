"""Wire the three remaining refined task panels on existing owned worker CPUs."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import hashlib
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile
import time

import component_extension_fleet as fleet

ASSIGNMENTS = {
    'pointmaze': (50626847, 'jepa-mw-components-0911-slot1-v1', 'New Jersey, US', 'GPU-2637202e-56e9-f947-e534-7aaaa0f453e3'),
    'wall': (50632757, 'jepa-mw-components-0911-slot2-v3', 'Missouri, US', 'GPU-86601922-233d-a4a8-08e5-971651bd2cde'),
    'droid': (50640703, 'jepa-mw-components-0911-slot3-v11', 'Nevada, US', 'GPU-4ae37887-efa6-38ae-14be-123371ce74f5')}


def navigation_reference_source(reference):
    source = reference / 'baseline-source/src/offline_study'
    digest = hashlib.sha256()
    paths = sorted(source.glob('*.py'))
    if not paths:raise ValueError('Historical source directory is empty')
    for path in paths:digest.update(path.name.encode()+b'\0'+path.read_bytes())
    expected = json.loads((reference/'freeze/protocol.json').read_text())['source_sha256']
    if digest.hexdigest()!=expected:raise ValueError('Historical native source digest differs before staging')
    for name in ('backends.py','model_loader.py','planning_native_smoke.py','planning_contract.py'):
        if fleet.sha(source/name)!=fleet.sha(fleet.PROJECT/'src/offline_study'/name):
            raise ValueError('Shared native execution source changed')
    return source


def stage(task, attempt=1):
    if attempt != 1 and not ((task == 'pointmaze' and attempt in (2,3)) or (task=='wall' and attempt==2)):
        raise ValueError('Only the diagnosed PointMaze CPU setup retry is registered')
    assert 'September11 remaining refined behavioral setup' in Path('/Users/stevenyang/Documents/GPU_RESOURCE_BOARD.md').read_text()
    instance, label, place, uuid = ASSIGNMENTS[task]
    rows = fleet.provider('show', 'instances')
    row = next(r for r in rows if r['id'] == instance)
    assert row['label'] == label and row['geolocation'] == place and row['actual_status'] == 'running'
    assert fleet.account_hourly(rows) <= 7
    root = fleet.PROJECT
    remote = f'/workspace/refined-{task}-behavior-20260911-v{attempt}'
    local = root / f'artifacts/offline_study/table-completion-20260911-v1/refined-{task}-behavior-worker-v{attempt}'
    local.mkdir(parents=True, exist_ok=False)
    files = {'code/src/offline_study/' + p.name: p for p in (root / 'src/offline_study').glob('*.py')}
    def add(name, path):
        files.update({name + '/' + str(p.relative_to(path)):p for p in path.rglob('*')
                      if p.is_file() and '__pycache__' not in p.parts})
    spec = {'task':task,'logical_ranks':list(range(8)),'vendor':fleet.REMOTE+'/code/vendor/jepa-wms',
            'physical_gpu_uuid':uuid,'fit':remote+'/fit','reference':remote+'/reference'}
    base = root / 'artifacts/offline_study'
    if task == 'droid':
        old='/workspace/refined-droid-prerequisite-20260911-v1'
        spec.update(prerequisite='/workspace/refined-droid-prerequisite-20260911-v2',
            assets='/workspace/jepa-runtime/droid-assets-20260907-v1',
            manifest=remote+'/code/configs/droid_assets.json', **{'native-engineering':remote+'/native-engineering',
            'encoder-source':old+'/dinov3','encoder-root':old+'/encoder'})
        add('reference',base/'primary-durable-20260907/droid-native-replication-20260907-v1')
        add('native-engineering',base/'primary-durable-20260907/droid-native-engineering-20260907-v7')
        for name in ('report.json','DONE.json','protocol.json'):
            files['asset-receipt/'+name]=base/'table-completion-20260911-v1/refined-droid-assets-v1'/name
        files['code/configs/droid_assets.json']=root/'configs/droid_assets.json'
    else:
        prerequisite=f'/workspace/refined-nav-{task}-prerequisite-20260911-v{2 if task=="wall" and attempt==2 else 1}'
        spec.update(prerequisite=prerequisite,checkpoint=prerequisite+'/checkpoint.pth.tar',
            cohort=prerequisite+'/cohort/cohort.json', **{'data-root':prerequisite+'/data','reference-source':remote+'/reference-source'})
        reference=base/'navigation-coupling-reference-20260908-v1'/task
        add('reference',reference)
        add('reference-source',navigation_reference_source(reference))
        if task=='pointmaze':spec['python']=remote+'/runtime/python/bin/python'
    for name in ('refined_panel_queue.py','component_boundary_job.py','component_queue_handoff.py','prepare_remaining_refined_behavior.py'):
        files['ops/'+name]=Path(__file__).with_name(name)
    files['code/docs/REFINED_SIX_TASK_COMPLETION.md']=root/'docs/REFINED_SIX_TASK_COMPLETION.md'
    fleet.write(local/'PANEL.json',spec)
    files['PANEL.json']=local/'PANEL.json'
    manifest={name:{'bytes':p.stat().st_size,'sha256':fleet.sha(p)} for name,p in files.items()}
    fleet.write(local/'SOURCE.json',manifest)
    files['SOURCE.json']=local/'SOURCE.json'
    ssh=fleet.connection(row)
    subprocess.run(ssh+['test ! -e '+remote+' && mkdir '+remote],check=True,timeout=35)
    with tempfile.TemporaryFile() as stream:
        with tarfile.open(fileobj=stream,mode='w:gz',compresslevel=1) as archive:
            for name,p in files.items():archive.add(p,arcname=name,recursive=False)
        stream.seek(0)
        subprocess.run(ssh+['tar --keep-old-files --no-same-owner -xz -C '+remote],stdin=stream,check=True,timeout=90)
    guard=f'com.steven.jepa.{task}.behavior.collect.20260911'+('' if attempt==1 else f'.v{attempt}')
    subprocess.run(['launchctl','submit','-l',guard,'--','/usr/bin/env',
        'PATH=/usr/local/bin:/usr/bin:/bin:/Users/stevenyang/.local/bin',fleet.PYTHON,
        str(Path(__file__).with_name('collect_refined_prerequisite.py')),'--task',task+'-behavior','--attempt',str(attempt)],check=True)
    script='''import json,hashlib,sys,os,subprocess
from pathlib import Path
r=Path(sys.argv[1])
for name,want in json.loads((r/'SOURCE.json').read_text()).items():
 p=r/name
 assert p.stat().st_size==want['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==want['sha256']
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
with (r/'preparation.log').open('x') as log:
 p=subprocess.Popen(['/workspace/component-python/bin/python','-u',str(r/'ops/prepare_remaining_refined_behavior.py'),str(r)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
with (r/'PREPARATION_LAUNCH.json').open('x') as f:json.dump({'pid':p.pid},f)
print(json.dumps({'pid':p.pid,'cpu_preparation_started':True}))'''
    receipt=json.loads(subprocess.check_output(ssh+[shlex.join(['python3','-c',script,remote])],text=True,timeout=35))
    fleet.write(local/'LAUNCH.json',{**receipt,'instance':instance,'time':time.time()})
    print(json.dumps({'task':task,**receipt}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',choices=tuple(ASSIGNMENTS));p.add_argument('--attempt',type=int,default=1)
    a=p.parse_args()
    if a.task:stage(a.task,a.attempt)
    else:
        if a.attempt!=1:raise ValueError('Retry must name its task')
        with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(stage,ASSIGNMENTS))
