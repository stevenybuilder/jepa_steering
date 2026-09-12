"""Stage the verified Push-T prerequisite on one explicitly owned US worker."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time

import component_extension_fleet as fleet

PROJECT = fleet.PROJECT
LOCAL = PROJECT / 'artifacts/offline_study/table-completion-20260911-v1/refined-pusht-worker-v1'
REMOTE = '/workspace/refined-pusht-prerequisite-20260911-v1'
INSTANCE = 50638073
LABEL = 'jepa-mw-components-0911-slot0-v5'
UUID = 'GPU-252aa065-5b0e-4054-6a87-e8ef908cef5f'
CHECKPOINT = '9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb'
ORIGINAL = 'artifacts/offline_study/primary-durable-20260907/pusht-author-replication-20260907'


def main():
    if '2026-09-11 bounded Push-T prerequisite' not in Path('/Users/stevenyang/Documents/GPU_RESOURCE_BOARD.md').read_text():
        raise ValueError('Missing explicit resource-board assignment')
    LOCAL.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(PROJECT / 'src'))
    from offline_study.author_fit import source_hash
    source = source_hash()
    rows = fleet.provider('show', 'instances')
    row = next(r for r in rows if r['id'] == INSTANCE)
    if (row['label'] != LABEL or row['geolocation'] != 'Nevada, US'
            or row['actual_status'] != 'running' or fleet.account_hourly(rows) > 7):
        raise ValueError('Owner, geography, active runtime or budget changed')
    ssh = fleet.connection(row)
    files = {'code/src/offline_study/' + p.name: p
             for p in (PROJECT / 'src/offline_study').glob('*.py')}
    fit_parent = PROJECT / ORIGINAL / 'fits-v1/bfloat16/pusht'
    cohort = PROJECT / ORIGINAL / 'cohorts/pusht/cohort.json'
    for p in [*fit_parent.glob('*.json'), *(fit_parent / 'operator_rank').glob('*'), cohort]:
        if p.is_file():
            files['code/' + str(p.relative_to(PROJECT))] = p
    data = PROJECT / 'artifacts/offline_study/table-completion-20260911-v1/refined-pusht-inputs-v1/fit-data'
    files.update({'data/' + str(p.relative_to(data)): p for p in data.rglob('*') if p.is_file()})
    for name in ('component_boundary_job.py', 'component_queue_handoff.py'):
        files['ops/' + name] = Path(__file__).with_name(name)
    files['code/docs/REFINED_SIX_TASK_COMPLETION.md'] = PROJECT / 'docs/REFINED_SIX_TASK_COMPLETION.md'
    from huggingface_hub import hf_hub_download
    from dotenv import dotenv_values
    values = dotenv_values(PROJECT / '.env')
    token = next(values[k] for k in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HUGGINGFACE_TOKEN') if values.get(k))
    try:
        checkpoint = Path(hf_hub_download('facebook/jepa-wms', 'jepa_wm_pusht.pth.tar',
            revision='9b9c41ef249466630dbf1a20e78391865d07b3b9', token=token,
            local_dir=LOCAL / 'official_hf'))
    except Exception as error:
        raise RuntimeError('Official checkpoint download failed: ' + type(error).__name__) from None
    if fleet.sha(checkpoint) != CHECKPOINT:
        raise ValueError('Wrong Push-T checkpoint')
    files['checkpoints/jepa_wm_pusht.pth.tar'] = checkpoint
    manifest = {name: {'bytes': p.stat().st_size, 'sha256': fleet.sha(p)} for name, p in files.items()}
    command = ['/workspace/component-python/bin/python', '-u', '-m', 'offline_study.refined_task_fit',
        '--cohort', f'{REMOTE}/code/{ORIGINAL}/cohorts/pusht/cohort.json',
        '--fit', f'{REMOTE}/code/{ORIGINAL}/fits-v1/bfloat16/pusht/operator_rank',
        '--vendor', fleet.REMOTE + '/code/vendor/jepa-wms',
        '--checkpoint', REMOTE + '/checkpoints/jepa_wm_pusht.pth.tar',
        '--data-root', REMOTE + '/data', '--output', REMOTE + '/fit-v1', '--device', 'cuda:0']
    job = {'task': 'pusht', 'timeout_seconds': 3900, 'command': command,
        'cwd': REMOTE + '/code', 'gpu_uuid': UUID, 'files': manifest,
        'source_sha256': source, 'instance': INSTANCE, 'no_new_rental': True}
    fleet.write(LOCAL / 'JOB.json', job)
    subprocess.run(ssh + ['test ! -e ' + REMOTE + ' && mkdir ' + REMOTE], check=True, timeout=30)
    with tempfile.TemporaryFile() as stream:
        with tarfile.open(fileobj=stream, mode='w:gz', compresslevel=1) as archive:
            for name, path in files.items():
                archive.add(path, arcname=name, recursive=False)
            archive.add(LOCAL / 'JOB.json', arcname='JOB.json', recursive=False)
        stream.seek(0)
        subprocess.run(ssh + ['tar --keep-old-files --no-same-owner -xz -C ' + REMOTE],
            stdin=stream, check=True, timeout=600)
    check = '''import json,sys
from pathlib import Path
r=Path(sys.argv[1]);sys.path.insert(0,str(r/'ops'));sys.path.insert(0,str(r/'code/src'))
from component_boundary_job import validate_job
from offline_study.author_fit import source_hash
from offline_study.refined_task_fit import source_inputs
from offline_study.refined_fit_inputs import verify
from types import SimpleNamespace
j=json.loads((r/'JOB.json').read_text());validate_job(j,r)
assert source_hash()==j['source_sha256']
c=j['command'];a=SimpleNamespace(cohort=Path(c[c.index('--cohort')+1]),fit=Path(c[c.index('--fit')+1]))
source_inputs(a);verify(a.cohort,r/'data')
from offline_study.model_loader import verified_local_dino_cache
with verified_local_dino_cache() as proof:pass
print(json.dumps({'status':'exact_inputs_and_source_verified_cpu_only','source_sha256':source_hash()}))
'''
    env = ['env', 'CUDA_VISIBLE_DEVICES=', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1',
        'MKL_NUM_THREADS=1', 'PYTHONDONTWRITEBYTECODE=1', '/workspace/component-python/bin/python']
    receipt = json.loads(subprocess.check_output(ssh + [shlex.join(env + ['-c', check, REMOTE])],
        text=True, timeout=180))
    fleet.write(LOCAL / 'RECEIVING.json', receipt)
    launch = '''import json,subprocess,sys
from pathlib import Path
r=Path(sys.argv[1]);c=Path(sys.argv[2]);sys.path.insert(0,str(r/'ops'))
from component_queue_handoff import process
from component_boundary_job import validate_queue
queues=[]
for p in Path('/proc').iterdir():
 if p.name.isdigit():
  item=process(int(p.name))
  try:validate_queue(item,c)
  except (ValueError,IndexError):continue
  queues.append(int(p.name))
assert len(queues)==1,queues
with (r/'boundary.log').open('x') as log:
 child=subprocess.Popen(['/workspace/component-python/bin/python','-u',str(r/'ops/component_boundary_job.py'),'--job',str(r/'JOB.json'),'--component-root',str(c),'--queue-pid',str(queues[0])],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'boundary_pid':child.pid,'component_queue_pid':queues[0]}))
'''
    receipt = json.loads(subprocess.check_output(ssh + [shlex.join(['python3', '-c', launch, REMOTE, fleet.REMOTE])], text=True, timeout=30))
    fleet.write(LOCAL / 'LAUNCH.json', {**receipt, 'time': time.time(), 'instance': INSTANCE})
    print(json.dumps({'instance': INSTANCE, 'status': 'boundary_waiter_launched', **receipt}), flush=True)


if __name__ == '__main__':
    main()
