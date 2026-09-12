"""Isolate declared DINOv3 dependencies, test CPU imports, retry unchanged fit.

No pip changes to the running MetaWorld environment. Preserve the failed attempt
and all original input/source hashes; only the output root and import overlay
differ. GPU fitting waits for a complete existing scientific child stream.
"""
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/workspace/refined-droid-prerequisite-20260911-v2')
OLD = Path('/workspace/refined-droid-prerequisite-20260911-v1')
COMPONENT = Path('/workspace/metaworld-components-20260911-v1')
PYTHON = '/workspace/component-python/bin/python'
PACKAGES = ('torchmetrics==1.8.2', 'lightning-utilities==0.15.2', 'ftfy==6.3.1',
    'wcwidth==0.2.13', 'scikit-learn==1.7.2', 'joblib==1.5.2', 'threadpoolctl==3.6.0')


def main():
    sys.path.insert(0, str(ROOT / 'ops'))
    from component_boundary_job import validate_job, validate_queue
    from component_queue_handoff import process, write
    try:
        old_job = json.loads((OLD / 'JOB.json').read_text())
        if (json.loads((OLD / 'TERMINAL.json').read_text())['status'] != 'incomplete_preserve_all'
                or "No module named 'torchmetrics'" not in (OLD / 'fit-v1/FAILED.json').read_text()):
            raise ValueError('This recovery is only for the observed missing dependency')
        validate_job(old_job, OLD)
        shutil.copytree(OLD / 'code', ROOT / 'code', ignore=shutil.ignore_patterns('__pycache__'))
        before = {name: importlib.metadata.version(name) for name in ('torch','numpy','mujoco','metaworld','gym','gymnasium')}
        with (ROOT / 'dependency-install.log').open('x') as log:
            subprocess.run([PYTHON, '-m', 'pip', 'install', '--no-deps', '--no-cache-dir',
                '--index-url', 'https://pypi.org/simple', '--timeout', '45', '--retries', '1',
                '--target', str(ROOT / 'runtime-packages'), *PACKAGES],
                stdout=log, stderr=log, check=True, timeout=300)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
            PYTHONPATH=os.pathsep.join([str(ROOT / 'code/src'), str(ROOT / 'runtime-packages')]),
            OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        check = '''import importlib.metadata,json,runpy,sys,torch
from pathlib import Path
from types import SimpleNamespace
r=Path(sys.argv[1]);old=Path(sys.argv[2]);source=old/'dinov3'
assert not torch.cuda.is_initialized()
sys.path.insert(0,str(source));hub=runpy.run_path(str(source/'hubconf.py'))
assert callable(hub['dinov3_vitl16'])
from offline_study.vendor import use_vendor
use_vendor(Path('/workspace/metaworld-components-20260911-v1/code/vendor/jepa-wms'))
from offline_study.droid_fixed_response_fit import input_contract
input_contract(SimpleNamespace(inputs=old/'inputs',audit=old/'audit',raw=old/'recovered/raw'))
assert not torch.cuda.is_initialized()
from offline_study.protocol import write_json
write_json(r/'IMPORTS_VERIFIED.json',{'time':__import__('time').time(),'full_pinned_dinov3_hub_import':True,'original128_recording512_prefix_contract_verified':True,'cuda_initialized':False,'versions':{n:importlib.metadata.version(n) for n in ('torch','numpy','torchmetrics','lightning-utilities','ftfy','scikit-learn')}})
'''
        with (ROOT / 'dependency-check.log').open('x') as log:
            subprocess.run([PYTHON, '-c', check, str(ROOT), str(OLD)], env=env,
                stdout=log, stderr=log, check=True, timeout=180)
        after = {name: importlib.metadata.version(name) for name in before}
        if before != after: raise ValueError('Original shared runtime changed')
        sys.path.insert(0, str(ROOT / 'code/src'))
        from offline_study.protocol import sha256
        command = list(old_job['command'])
        command[command.index('--output') + 1] = str(ROOT / 'fit-v1')
        write(ROOT / 'RUNTIME_REPAIR.json', {'time': time.time(), 'packages': PACKAGES,
            'base_runtime_before': before, 'base_runtime_after': after,
            'method_population_dose_and_fitting_code_unchanged': True,
            'old_attempt_preserved': str(OLD), 'gpu_fit_complete': False})
        files = {str(p.relative_to(ROOT)): {'bytes': p.stat().st_size, 'sha256': sha256(p)}
            for name in ('code','ops','runtime-packages') for p in (ROOT/name).rglob('*')
            if p.is_file() and '__pycache__' not in p.parts}
        job = {**old_job, 'command': command, 'cwd': str(ROOT / 'code'), 'files': files,
            'pythonpath_additions': [str(ROOT / 'runtime-packages')],
            'inherited_job': {'path': str(OLD / 'JOB.json'), 'sha256': sha256(OLD / 'JOB.json')}}
        write(ROOT / 'JOB.json', job)
        validate_job(job, ROOT)
        queues = []
        for path in Path('/proc').iterdir():
            if not path.name.isdigit(): continue
            try: validate_queue(process(int(path.name)), COMPONENT)
            except (ValueError, IndexError): continue
            queues.append(int(path.name))
        if len(queues) != 1: raise ValueError('Require one active owned coordinator')
        with (ROOT / 'boundary.log').open('x') as log:
            child = subprocess.Popen([PYTHON, '-u', str(ROOT / 'ops/component_boundary_job.py'),
                '--job', str(ROOT / 'JOB.json'), '--component-root', str(COMPONENT),
                '--queue-pid', str(queues[0])], stdin=subprocess.DEVNULL, stdout=log,
                stderr=log, start_new_session=True)
        write(ROOT / 'BOUNDARY_LAUNCH.json', {'time': time.time(), 'pid': child.pid})
        print(json.dumps({'droid_dependency_repaired': True, 'full_cpu_import_preflight_passed': True,
            'unchanged_gpu_fit_queued_not_complete': True}), flush=True)
    except Exception as error:
        write(ROOT / 'PREPARATION_FAILED.json', {'error_type': type(error).__name__, 'time': time.time()})
        raise


if __name__ == '__main__': main()
