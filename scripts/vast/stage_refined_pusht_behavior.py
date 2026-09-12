"""Freeze and queue the completed Push-T fit for its actual behavioral panel."""
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile
import time

import component_extension_fleet as fleet

LOCAL = fleet.PROJECT / 'artifacts/offline_study/table-completion-20260911-v1/refined-pusht-behavior-worker-v1'
REMOTE = '/workspace/refined-pusht-behavior-20260911-v1'
INSTANCE = 50638073
LABEL = 'jepa-mw-components-0911-slot0-v5'
UUID = 'GPU-252aa065-5b0e-4054-6a87-e8ef908cef5f'


def main():
    assert 'September11 Push-T refined behavioral handoff' in Path('/Users/stevenyang/Documents/GPU_RESOURCE_BOARD.md').read_text()
    rows = fleet.provider('show', 'instances')
    row = next(r for r in rows if r['id'] == INSTANCE)
    if row['label'] != LABEL or row['geolocation'] != 'Nevada, US' or row['actual_status'] != 'running' or fleet.account_hourly(rows) > 7:
        raise ValueError('Owned active worker or budget changed')
    base = fleet.PROJECT / 'artifacts/offline_study/table-completion-20260911-v1'
    prerequisite = base / 'refined-pusht-worker-v1'
    if not fleet.read(prerequisite / 'COLLECTED.json')['drive_verified']:
        raise ValueError('Completed fit not yet durably preserved')
    fit = prerequisite / 'restored/refined-pusht-prerequisite-20260911-v1/fit-v1'
    original = base / 'restored/50239185/batch-000/jepa-runtime'
    cohort = fleet.PROJECT / 'artifacts/offline_study/primary-durable-20260907/pusht-author-replication-20260907/cohorts/pusht'
    files = {'code/src/offline_study/' + p.name: p for p in (fleet.PROJECT / 'src/offline_study').glob('*.py')}
    for key, directory in [('fit', fit), ('cohort', cohort),
            ('reference/freeze', original / 'pusht-planning-native-20260908-v1/freeze'),
            ('reference-source', original / 'pusht-planning-code-20260908-v1/src/offline_study')]:
        files.update({key + '/' + p.name: p for p in directory.iterdir() if p.is_file()})
    for name in ('component_boundary_job.py', 'component_queue_handoff.py', 'refined_panel_queue.py', 'prepare_refined_pusht_behavior.py'):
        files['ops/' + name] = Path(__file__).with_name(name)
    files['code/docs/REFINED_SIX_TASK_COMPLETION.md'] = fleet.PROJECT / 'docs/REFINED_SIX_TASK_COMPLETION.md'
    LOCAL.mkdir(parents=True, exist_ok=False)
    remote_fit = '/workspace/refined-pusht-prerequisite-20260911-v1'
    spec = {'task': 'pusht', 'logical_ranks': list(range(8)), 'vendor': fleet.REMOTE + '/code/vendor/jepa-wms',
        'checkpoint': remote_fit + '/checkpoints/jepa_wm_pusht.pth.tar', 'fit': REMOTE + '/fit',
        'cohort': REMOTE + '/cohort/cohort.json', 'reference': REMOTE + '/reference',
        'reference-source': REMOTE + '/reference-source', 'data-root': remote_fit + '/data',
        'old_solver': False, 'physical_gpu_uuid': UUID}
    fleet.write(LOCAL / 'PANEL.json', spec)
    files['PANEL.json'] = LOCAL / 'PANEL.json'
    manifest = {name: {'bytes': p.stat().st_size, 'sha256': fleet.sha(p)} for name, p in files.items()}
    fleet.write(LOCAL / 'SOURCE.json', manifest)
    files['SOURCE.json'] = LOCAL / 'SOURCE.json'
    ssh = fleet.connection(row)
    subprocess.run(ssh + ['test ! -e ' + REMOTE + ' && mkdir ' + REMOTE], check=True, timeout=30)
    with tempfile.TemporaryFile() as stream:
        with tarfile.open(fileobj=stream, mode='w:gz', compresslevel=1) as archive:
            for name, path in files.items(): archive.add(path, arcname=name, recursive=False)
        stream.seek(0)
        subprocess.run(ssh + ['tar --keep-old-files --no-same-owner -xz -C ' + REMOTE], stdin=stream, check=True, timeout=180)
    script = '''import os,json,sys,subprocess,hashlib
from pathlib import Path
r=Path(sys.argv[1]);data=sys.stdin.read()
for name,want in json.loads((r/'SOURCE.json').read_text()).items():
 p=r/name
 if p.is_symlink() or p.stat().st_size!=want['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=want['sha256']:raise ValueError('Stage checksum mismatch')
env=dict(os.environ,PYTHONPATH=str(r/'code/src'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_DISABLE_PROGRESS_BARS='1',CUDA_VISIBLE_DEVICES='')
with (r/'preparation.log').open('x') as log:
 p=subprocess.Popen(['/workspace/component-python/bin/python','-u',str(r/'ops/prepare_refined_pusht_behavior.py')],stdin=subprocess.PIPE,stdout=log,stderr=log,env=env,start_new_session=True)
 p.stdin.write(data.encode());p.stdin.close()
print(json.dumps({'preparation_pid':p.pid,'gpu_calls_at_launch':0}))
'''
    from dotenv import dotenv_values
    values = dotenv_values(fleet.PROJECT / '.env')
    token = next(values[k] for k in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HUGGINGFACE_TOKEN') if values.get(k))
    receipt = json.loads(subprocess.check_output(ssh + [shlex.join(['python3', '-c', script, REMOTE])],
        input=json.dumps({'root': REMOTE, 'hf_token': token, 'gpu_uuid': UUID}), text=True, timeout=60))
    fleet.write(LOCAL / 'LAUNCH.json', {**receipt, 'instance': INSTANCE, 'time': time.time()})
    print(json.dumps({'task': 'pusht', **receipt}), flush=True)


if __name__ == '__main__': main()
