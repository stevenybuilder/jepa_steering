"""Direct verified navigation inputs; no evaluation outcomes during preparation."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile


def main(payload):
    root, task = Path(payload['root']), payload['task']
    sys.path.insert(0, str(root / 'code/src'))
    from offline_study.protocol import sha256, write_json
    from offline_study.navigation_assets import safe_members
    from offline_study.navigation_cohort import verify_inputs
    from offline_study.navigation_rank_reconstruction import original_contract
    from offline_study.author_fit import source_hash
    try:
        spec = json.loads((root / 'code/configs/navigation_assets.json').read_text())
        from huggingface_hub import hf_hub_download
        completed = []
        for item in [a for a in spec['assets'] if a['task'] == task]:
            kind = item['repo_type']
            path = Path(hf_hub_download(item['repo_id'], item['filename'],
                repo_type=None if kind == 'model' else kind, revision=item['revision'],
                token=payload['hf_token'], local_dir=root / 'downloads' / kind))
            if path.stat().st_size != item['size'] or sha256(path) != item['sha256']:
                raise ValueError('Pinned official input checksum differs')
            if item['kind'] == 'checkpoint':
                path.rename(root / 'checkpoint.pth.tar')
            else:
                files = json.loads((root / 'cohort/input_files.json').read_text())
                prefix = 'wall_single/' if task == 'wall' else 'point_maze/'
                with zipfile.ZipFile(path) as archive:
                    safe_members(archive)
                    members = {m.filename: m for m in archive.infolist()}
                    size = sum(members[prefix + name].file_size for name in files)
                    if shutil.disk_usage(root).free < size + 8 * 1024**3:
                        raise ValueError('Insufficient space for exact input subset plus reserve')
                    for name in files:
                        target = root / 'data' / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(prefix + name) as source, target.open('xb') as output:
                            shutil.copyfileobj(source, output, 4 << 20)
                        if sha256(target) != files[name]:
                            raise ValueError('Original cohort input bytes differ')
            completed.append(item['filename'])
            print(json.dumps({'task': task, 'verified_asset': item['filename']}), flush=True)
        payload.pop('hf_token', None)
        verify_inputs(root / 'cohort/cohort.json', root / 'data')
        original_contract(root / 'original', root / 'cohort/cohort.json')
        source = json.loads((root / 'SOURCE.json').read_text())
        if source_hash() != source['source_sha256']:
            raise ValueError('Fitting source changed')
        write_json(root / 'INPUTS_READY.json', {'task': task, 'time': time.time(), 'gpu_calls': 0,
            'original_cohort_input_hashes_verified': True, 'downloaded_assets': completed})
        files = {str(p.relative_to(root)): {'bytes': p.stat().st_size, 'sha256': sha256(p)}
            for base in ('code', 'original', 'cohort') for p in (root / base).rglob('*')
            if p.is_file() and '__pycache__' not in p.parts}
        write_json(root / 'JOB.json', {'task': task, 'timeout_seconds': 7500,
            'command': ['/workspace/component-python/bin/python', '-u', '-m',
                'offline_study.navigation_refined_pipeline', '--root', str(root)],
            'cwd': str(root / 'code'), 'files': files, 'gpu_uuid': payload['gpu_uuid']})
        sys.path.insert(0, str(root / 'ops'))
        from component_boundary_job import validate_queue
        from component_queue_handoff import process
        component = Path('/workspace/metaworld-components-20260911-v1')
        queues = []
        for path in Path('/proc').iterdir():
            if not path.name.isdigit(): continue
            try: validate_queue(process(int(path.name)), component)
            except (ValueError, IndexError): continue
            queues.append(int(path.name))
        if len(queues) != 1:
            raise ValueError('Require one active owned component coordinator')
        with (root / 'boundary.log').open('x') as log:
            child = subprocess.Popen(['/workspace/component-python/bin/python', '-u',
                str(root / 'ops/component_boundary_job.py'), '--job', str(root / 'JOB.json'),
                '--component-root', str(component), '--queue-pid', str(queues[0])],
                stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
        write_json(root / 'BOUNDARY_LAUNCH.json', {'pid': child.pid, 'time': time.time()})
        print(json.dumps({'task': task, 'boundary_waiter_launched': True}), flush=True)
    except Exception as error:
        write_json(root / 'PREPARATION_FAILED.json', {'task': task,
            'error_type': type(error).__name__, 'time': time.time()})
        print(json.dumps({'task': task, 'preparation_failed': type(error).__name__, 'inputs_retained': True}), flush=True)


if __name__ == '__main__':
    main(json.load(sys.stdin))
