"""CPU-only pinned validation-data restoration, freeze and boundary submission."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile


def main(payload):
    root = Path(payload['root'])
    sys.path.insert(0, str(root / 'code/src'))
    from offline_study.protocol import sha256, write_json
    from offline_study.pusht_planning_replication import checked_cohort, input_hashes
    try:
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download('facebook/jepa-wms', 'pusht/pusht_noise.zip',
            repo_type='dataset', revision='6116f042ae7ae4c8e3f1fd2f194f432615664182',
            token=payload['hf_token'], local_dir=root / 'downloads'))
        if path.stat().st_size != 2785304515 or sha256(path) != '442f5dee246edf670964ed7bdecd248683cd6d00580fa0e4d458abb53f92da08':
            raise ValueError('Pinned archive differs')
        spec = json.loads((root / 'PANEL.json').read_text())
        target_root = Path(spec['data-root'])
        prior = json.loads((root / 'reference/freeze/protocol.json').read_text())
        with zipfile.ZipFile(path) as archive:
            for name, digest in prior['input_files_sha256'].items():
                if Path(name).is_absolute() or '..' in Path(name).parts:
                    raise ValueError('Unsafe input member')
                target = target_root / 'val' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    with archive.open('pusht_noise/val/' + name) as source, target.open('xb') as output:
                        shutil.copyfileobj(source, output, 4 << 20)
                if sha256(target) != digest:
                    raise ValueError('Historical validation input differs')
        payload.pop('hf_token', None)
        checked_cohort(root / 'cohort/cohort.json')
        if input_hashes(target_root / 'val') != prior['input_files_sha256']:
            raise ValueError('Validation set changed')
        command = ['/workspace/component-python/bin/python', '-u', '-m', 'offline_study.refined_task_behavior', 'freeze']
        for key in ('task', 'vendor', 'checkpoint', 'fit', 'cohort', 'reference', 'reference-source', 'data-root'):
            command += ['--' + key, spec[key]]
        command += ['--output', str(root / 'freeze')]
        subprocess.run(command, check=True)
        write_json(root / 'INPUTS_READY.json', {'time': time.time(), 'task': 'pusht',
            'historical_validation_bytes_verified': True, 'gpu_calls': 0})
        files = {str(p.relative_to(root)): {'bytes': p.stat().st_size, 'sha256': sha256(p)}
            for base in ('code', 'ops', 'cohort', 'reference', 'reference-source', 'fit', 'freeze')
            for p in (root / base).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
        files['PANEL.json'] = {'bytes': (root / 'PANEL.json').stat().st_size, 'sha256': sha256(root / 'PANEL.json')}
        write_json(root / 'JOB.json', {'kind': 'refined_panel', 'task': 'pusht',
            'timeout_seconds': 21600, 'command': ['/workspace/component-python/bin/python', '-u',
                str(root / 'ops/refined_panel_queue.py'), '--root', str(root)],
            'cwd': str(root / 'code'), 'files': files, 'gpu_uuid': payload['gpu_uuid']})
        sys.path.insert(0, str(root / 'ops'))
        from component_boundary_job import validate_queue
        from component_queue_handoff import process
        component = Path('/workspace/metaworld-components-20260911-v1')
        queues = []
        for p in Path('/proc').iterdir():
            if not p.name.isdigit(): continue
            try: validate_queue(process(int(p.name)), component)
            except (ValueError, IndexError): continue
            queues.append(int(p.name))
        if len(queues) != 1: raise ValueError('Require one active owned coordinator')
        with (root / 'boundary.log').open('x') as log:
            child = subprocess.Popen(['/workspace/component-python/bin/python', '-u',
                str(root / 'ops/component_boundary_job.py'), '--job', str(root / 'JOB.json'),
                '--component-root', str(component), '--queue-pid', str(queues[0])],
                stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        write_json(root / 'BOUNDARY_LAUNCH.json', {'time': time.time(), 'pid': child.pid})
        print(json.dumps({'task': 'pusht', 'frozen_and_boundary_submitted': True}), flush=True)
    except Exception as error:
        write_json(root / 'PREPARATION_FAILED.json', {'error_type': type(error).__name__, 'time': time.time()})
        raise RuntimeError('Behavior preparation failed: ' + type(error).__name__) from None


if __name__ == '__main__': main(json.load(sys.stdin))
