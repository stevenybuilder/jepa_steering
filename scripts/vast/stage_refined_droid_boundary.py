"""Stage small immutable metadata; download large public inputs on the worker."""
import configparser
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time

import component_extension_fleet as fleet

ROOT = fleet.PROJECT
LOCAL = ROOT / 'artifacts/offline_study/table-completion-20260911-v1/refined-droid-worker-v1'
REMOTE = '/workspace/refined-droid-prerequisite-20260911-v1'
INSTANCE = 50640703
LABEL = 'jepa-mw-components-0911-slot3-v11'


def main():
    if '2026-09-11 bounded DROID prerequisite' not in Path('/Users/stevenyang/Documents/GPU_RESOURCE_BOARD.md').read_text():
        raise ValueError('Explicit DROID resource-board assignment required')
    rows = fleet.provider('show', 'instances')
    row = next(r for r in rows if r['id'] == INSTANCE)
    if row['label'] != LABEL or row['geolocation'] != 'Nevada, US' or row['actual_status'] != 'running' or fleet.account_hourly(rows) > 7:
        raise ValueError('Owned US worker/budget changed')
    resuming = '--resume-launch' in sys.argv
    LOCAL.mkdir(parents=True, exist_ok=resuming)
    if (LOCAL / 'LAUNCH.json').exists():
        raise ValueError('Already launched; do not duplicate preparation')
    sys.path.insert(0, str(ROOT / 'src'))
    from offline_study.author_fit import source_hash
    source = source_hash()
    files = {'code/src/offline_study/' + p.name: p for p in (ROOT / 'src/offline_study').glob('*.py')}
    base = ROOT / 'artifacts/offline_study/droid-fit-completed-inputs-20260908-v1'
    for key, directory in [('inputs', base / 'droid-fit-native-eligible-20260908-v1'),
                           ('audit', base / 'droid-fit-audit-20260908-v2')]:
        files.update({key + '/' + p.name: p for p in directory.iterdir() if p.is_file()})
    encoder = ROOT / 'artifacts/offline_study/table-completion-20260911-v1/refined-droid-encoder-v1'
    for name in ('protocol.json', 'report.json', 'DONE.json'):
        files['encoder/' + name] = encoder / name
    for name in ('prepare_droid_refined_worker.py', 'restore_droid_refined_inputs.py',
                 'component_boundary_job.py', 'component_queue_handoff.py'):
        files['ops/' + name] = Path(__file__).with_name(name)
    files['code/configs/droid_assets.json'] = ROOT / 'configs/droid_assets.json'
    files['code/docs/REFINED_SIX_TASK_COMPLETION.md'] = ROOT / 'docs/REFINED_SIX_TASK_COMPLETION.md'
    proof = {'source_sha256': source, 'files': {name: {'bytes': p.stat().st_size, 'sha256': fleet.sha(p)} for name, p in files.items()}}
    if resuming:
        if fleet.read(LOCAL / 'SOURCE.json') != proof:
            raise ValueError('Resumed preparation source differs')
    else:
        fleet.write(LOCAL / 'SOURCE.json', proof)
    ssh = fleet.connection(row)
    if not resuming:
        subprocess.run(ssh + ['test ! -e ' + REMOTE + ' && mkdir ' + REMOTE], check=True, timeout=30)
        with tempfile.TemporaryFile() as stream:
            with tarfile.open(fileobj=stream, mode='w:gz', compresslevel=1) as archive:
                for name, path in files.items(): archive.add(path, arcname=name, recursive=False)
                archive.add(LOCAL / 'SOURCE.json', arcname='SOURCE.json', recursive=False)
            stream.seek(0)
            subprocess.run(ssh + ['tar --keep-old-files --no-same-owner -xz -C ' + REMOTE], stdin=stream, check=True, timeout=180)
    from dotenv import dotenv_values
    from final_preservation_drive import request
    # Refresh existing local auth if necessary, ground the exact source again.
    with request('1PEj3wA0J2Z2SheKa9VEqoszqaUJ9TAgU', '?fields=id,sha256Checksum,size,trashed') as response:
        meta = json.load(response)
    if meta['sha256Checksum'] != 'fc933429ede48304e832fa11bbcc5d838eb539ca83b1746b93c8625526407b9a' or meta.get('trashed'):
        raise ValueError('Archived encoder changed')
    config = configparser.ConfigParser(interpolation=None)
    config.read('/Users/stevenyang/.config/rclone/rclone.conf')
    access = json.loads(config['gdrive']['token'])['access_token']
    values = dotenv_values(ROOT / '.env')
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
 p=subprocess.Popen(['/workspace/component-python/bin/python','-u',str(r/'ops/prepare_droid_refined_worker.py')],env=env,stdin=subprocess.PIPE,stdout=log,stderr=log,start_new_session=True)
 p.stdin.write(data.encode());p.stdin.close()
 receipt={'preparation_pid':p.pid,'gpu_calls_at_launch':0}
 with (r/'PREPARATION_LAUNCH.json').open('x') as f:json.dump(receipt,f)
print(json.dumps(receipt))
'''
    for attempt in range(3):
        try:
            result = json.loads(subprocess.check_output(ssh + [shlex.join(['python3', '-c', script, REMOTE])],
                text=True, input=json.dumps({'drive_access': access, 'hf_token': token}), timeout=40))
            break
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            if attempt == 2: raise
            time.sleep(3)
    fleet.write(LOCAL / 'LAUNCH.json', {**result, 'instance': INSTANCE, 'time': time.time()})
    print(json.dumps({'droid_worker_preparation_launched': True, 'instance': INSTANCE, **result}), flush=True)


if __name__ == '__main__':
    main()
