"""CPU-only direct input recovery, then one complete-stream-boundary fit."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path('/workspace/refined-droid-prerequisite-20260911-v1')
COMPONENT = Path('/workspace/metaworld-components-20260911-v1')
PYTHON = '/workspace/component-python/bin/python'
SOURCE = '6876159a11b4df116f30f667f8c9888617df0751'


def main(payload):
    sys.path.insert(0, str(ROOT / 'code/src'))
    from offline_study.protocol import sha256, write_json
    from offline_study.droid_fixed_response_fit import input_contract
    from offline_study.droid_native import verified_report
    from offline_study.droid_assets import verify_download
    try:
        def raw():
            from types import SimpleNamespace
            sys.path.insert(0, str(ROOT / 'ops'))
            import restore_droid_refined_inputs
            restore_droid_refined_inputs.main(SimpleNamespace(inputs=ROOT / 'inputs', output=ROOT / 'recovered'))
        def encoder():
            path = ROOT / 'encoder/native_state_from_official_hf.pth'
            h = hashlib.sha256(); size = 0
            req = urllib.request.Request('https://www.googleapis.com/drive/v3/files/1PEj3wA0J2Z2SheKa9VEqoszqaUJ9TAgU?alt=media',
                headers={'Authorization': 'Bearer ' + payload['drive_access'],
                    'X-Goog-User-Project': 'project-flash-490419'})
            with urllib.request.urlopen(req, timeout=90) as response, path.open('xb') as output:
                while chunk := response.read(4 << 20):
                    output.write(chunk); h.update(chunk); size += len(chunk)
            if size != 1213030147 or h.hexdigest() != 'fc933429ede48304e832fa11bbcc5d838eb539ca83b1746b93c8625526407b9a':
                raise ValueError('DROID archived encoder differs')
            source = ROOT / 'dinov3'
            with (ROOT / 'encoder-source.log').open('x') as log:
                subprocess.run(['git', 'clone', '--no-checkout', '--filter=blob:none',
                    'https://github.com/facebookresearch/dinov3.git', str(source)], stdout=log, stderr=log, check=True, timeout=180)
                subprocess.run(['git', '-C', str(source), 'checkout', '--detach', SOURCE],
                    stdout=log, stderr=log, check=True, timeout=180)
            if subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() != SOURCE:
                raise ValueError('Encoder source revision differs')
        def assets():
            from huggingface_hub import hf_hub_download
            spec = json.loads((ROOT / 'code/configs/droid_assets.json').read_text())
            records = []
            # Model first, then the exact same33 released files for later behavior.
            for item in sorted(spec['assets'], key=lambda x: x['repo_type'] != 'model'):
                kind = item['repo_type']
                target = ROOT / 'assets/downloads' / kind
                path = Path(hf_hub_download('facebook/jepa-wms', item['filename'],
                    repo_type=None if kind == 'model' else 'dataset',
                    revision=spec['model_revision' if kind == 'model' else 'dataset_revision'],
                    token=payload['hf_token'], local_dir=target))
                records.append(verify_download(path, item))
            write_json(ROOT / 'assets/VERIFIED.json', {'assets': records, 'model_calls': 0})
        with ThreadPoolExecutor(max_workers=3) as pool:
            jobs = [pool.submit(fn) for fn in (raw, encoder, assets)]
            for job in jobs: job.result()
        payload.clear()
        from types import SimpleNamespace
        input_contract(SimpleNamespace(inputs=ROOT / 'inputs', audit=ROOT / 'audit', raw=ROOT / 'recovered/raw'))
        verified_report(ROOT / 'encoder')
        from offline_study.author_fit import source_hash
        if source_hash() != json.loads((ROOT / 'SOURCE.json').read_text())['source_sha256']:
            raise ValueError('Fitting source changed')
        # Native imports and loader availability, no model construction/GPU call.
        from offline_study.vendor import use_vendor
        use_vendor(COMPONENT / 'code/vendor/jepa-wms')
        from app.plan_common.datasets.droid_dset import DROIDVideoDataset
        write_json(ROOT / 'INPUTS_READY.json', {'time': time.time(), 'gpu_calls': 0,
            'recordings': 128, 'source_sha256': source_hash(), 'encoder_reconverted': False})
        command = [PYTHON, '-u', '-m', 'offline_study.droid_fixed_response_fit',
            '--vendor', str(COMPONENT / 'code/vendor/jepa-wms'), '--assets', str(ROOT / 'assets'),
            '--manifest', str(ROOT / 'code/configs/droid_assets.json'), '--encoder-source', str(ROOT / 'dinov3'),
            '--encoder-root', str(ROOT / 'encoder'), '--inputs', str(ROOT / 'inputs'),
            '--audit', str(ROOT / 'audit'), '--raw', str(ROOT / 'recovered/raw'), '--output', str(ROOT / 'fit-v1')]
        files = {str(p.relative_to(ROOT)): {'bytes': p.stat().st_size, 'sha256': sha256(p)}
            for base in ('code', 'inputs', 'audit', 'encoder') for p in (ROOT / base).rglob('*')
            if p.is_file() and '__pycache__' not in p.parts}
        device = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
        if device != 'GPU-4ae37887-efa6-38ae-14be-123371ce74f5':
            raise ValueError('Assigned DROID prerequisite GPU changed')
        write_json(ROOT / 'JOB.json', {'command': command, 'cwd': str(ROOT / 'code'),
            'task': 'droid', 'timeout_seconds': 3900, 'files': files, 'gpu_uuid': device})
        sys.path.insert(0, str(ROOT / 'ops'))
        from component_boundary_job import validate_queue
        from component_queue_handoff import process
        queues = []
        for path in Path('/proc').iterdir():
            if not path.name.isdigit(): continue
            try: validate_queue(process(int(path.name)), COMPONENT)
            except (ValueError, IndexError): continue
            queues.append(int(path.name))
        if len(queues) != 1:
            raise ValueError('No single active owned component coordinator')
        with (ROOT / 'boundary.log').open('x') as log:
            child = subprocess.Popen([PYTHON, '-u', str(ROOT / 'ops/component_boundary_job.py'),
                '--job', str(ROOT / 'JOB.json'), '--component-root', str(COMPONENT),
                '--queue-pid', str(queues[0])], stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
        write_json(ROOT / 'BOUNDARY_LAUNCH.json', {'pid': child.pid, 'time': time.time()})
        print(json.dumps({'droid_prerequisite_ready_boundary_waiter_launched': True}), flush=True)
    except Exception as error:
        # Never serialize exception text/tracebacks containing bearer URLs.
        write_json(ROOT / 'PREPARATION_FAILED.json', {'error_type': type(error).__name__, 'time': time.time()})
        print(json.dumps({'preparation_failed': type(error).__name__, 'inputs_retained': True}), flush=True)


if __name__ == '__main__':
    main(json.load(sys.stdin))
