"""Stage both navigation prerequisites concurrently on two owned worker CPUs."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time

import component_extension_fleet as fleet

ASSIGNMENTS = {
    'pointmaze': (50626847, 'jepa-mw-components-0911-slot1-v1', 'New Jersey, US', 'GPU-2637202e-56e9-f947-e534-7aaaa0f453e3'),
    'wall': (50632757, 'jepa-mw-components-0911-slot2-v3', 'Missouri, US', 'GPU-86601922-233d-a4a8-08e5-971651bd2cde')}


def stage(task):
    if '2026-09-11 bounded navigation prerequisites' not in Path('/Users/stevenyang/Documents/GPU_RESOURCE_BOARD.md').read_text():
        raise ValueError('Explicit resource-board assignment required')
    instance, label, place, uuid = ASSIGNMENTS[task]
    rows = fleet.provider('show', 'instances')
    row = next(r for r in rows if r['id'] == instance)
    if row['label'] != label or row['geolocation'] != place or row['actual_status'] != 'running' or fleet.account_hourly(rows) > 7:
        raise ValueError('Owned active US worker or budget changed')
    root = fleet.PROJECT
    local = root / f'artifacts/offline_study/table-completion-20260911-v1/refined-{task}-worker-v1'
    remote = f'/workspace/refined-nav-{task}-prerequisite-20260911-v1'
    local.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root / 'src'))
    from offline_study.author_fit import source_hash
    source = source_hash()
    files = {'code/src/offline_study/' + p.name: p for p in (root / 'src/offline_study').glob('*.py')}
    restored = root / 'artifacts/offline_study/table-completion-20260911-v1/refined-source-recovery-v1/restored'
    for key, directory in [('original', restored / f'navigation-fits-20260907-v1/bfloat16/{task}/operator_rank'),
            ('cohort', restored / 'navigation-offline-cohorts-20260907-v1' / task)]:
        files.update({key + '/' + p.name: p for p in directory.iterdir() if p.is_file()})
    for name in ('component_boundary_job.py', 'component_queue_handoff.py', 'prepare_navigation_refined_worker.py'):
        files['ops/' + name] = Path(__file__).with_name(name)
    files['code/configs/navigation_assets.json'] = root / 'configs/navigation_assets.json'
    files['code/docs/REFINED_SIX_TASK_COMPLETION.md'] = root / 'docs/REFINED_SIX_TASK_COMPLETION.md'
    proof = {'source_sha256': source, 'files': {name: {'bytes': p.stat().st_size, 'sha256': fleet.sha(p)} for name, p in files.items()}}
    fleet.write(local / 'SOURCE.json', proof)
    ssh = fleet.connection(row)
    subprocess.run(ssh + ['test ! -e ' + remote + ' && mkdir ' + remote], check=True, timeout=30)
    with tempfile.TemporaryFile() as stream:
        with tarfile.open(fileobj=stream, mode='w:gz', compresslevel=1) as archive:
            for name, path in files.items(): archive.add(path, arcname=name, recursive=False)
            archive.add(local / 'SOURCE.json', arcname='SOURCE.json', recursive=False)
        stream.seek(0)
        subprocess.run(ssh + ['tar --keep-old-files --no-same-owner -xz -C ' + remote], stdin=stream, check=True, timeout=180)
    from dotenv import dotenv_values
    values = dotenv_values(root / '.env')
    token = next(values[k] for k in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HUGGINGFACE_TOKEN') if values.get(k))
    script = '''import os,sys,json,subprocess,hashlib
from pathlib import Path
r=Path(sys.argv[1]);data=sys.stdin.read();env=dict(os.environ,PYTHONPATH=str(r/'code/src'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_DISABLE_PROGRESS_BARS='1',CUDA_VISIBLE_DEVICES='')
proof=json.loads((r/'SOURCE.json').read_text())
for name,want in proof['files'].items():
 p=r/name
 if p.is_symlink() or p.stat().st_size!=want['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=want['sha256']:raise ValueError('Receiving file differs')
if (r/'PREPARATION_LAUNCH.json').exists():
 print((r/'PREPARATION_LAUNCH.json').read_text());sys.exit(0)
with (r/'preparation.log').open('x') as log:
 p=subprocess.Popen(['/workspace/component-python/bin/python','-u',str(r/'ops/prepare_navigation_refined_worker.py')],env=env,stdin=subprocess.PIPE,stdout=log,stderr=log,start_new_session=True)
 p.stdin.write(data.encode());p.stdin.close()
 receipt={'preparation_pid':p.pid,'gpu_calls_at_launch':0}
 with (r/'PREPARATION_LAUNCH.json').open('x') as f:json.dump(receipt,f)
print(json.dumps(receipt))
'''
    for attempt in range(3):
        try:
            result = json.loads(subprocess.check_output(ssh + [shlex.join(['python3', '-c', script, remote])],
                text=True, input=json.dumps({'hf_token': token, 'root': remote, 'task': task, 'gpu_uuid': uuid}), timeout=40))
            break
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            if attempt == 2: raise
            time.sleep(3)
    fleet.write(local / 'LAUNCH.json', {**result, 'instance': instance, 'time': time.time()})
    print(json.dumps({'task': task, 'preparation_launched': True, 'instance': instance, **result}), flush=True)


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(stage, ASSIGNMENTS))
