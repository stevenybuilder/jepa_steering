"""Finish existing PointMaze streams as owned Push-T GPU slots become free.

Orchestration only. Executes the same immutable science with the restored legacy
interpreter. No changes to fits, scenarios, doses, planner budget, or precision.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = Path('/workspace/table-completion-20260911-v1')
REPAIR = ROOT/'pointmaze-runtime-repair-v1'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2)


def main(payload):
    outputs = [ROOT/'expansion-20260911-v1'/name for name in ('pointmaze-a', 'pointmaze-b')]
    try:
        report = REPAIR/'report.json'
        ready = json.loads((REPAIR/'READY.json').read_text())
        assert hashlib.sha256(report.read_bytes()).hexdigest() == ready['report_sha256']
        assert json.loads(report.read_text())['status'] == 'all_97_original_pointmaze_inputs_exact_cpu_only'
        asset = payload['asset']; target = Path(asset['target'])
        if not target.exists():
            partial = target.with_suffix('.pointmaze-repaired.partial')
            with urllib.request.urlopen(asset['url'], timeout=90) as r, partial.open('xb') as f:
                shutil.copyfileobj(r, f, 4 << 20)
            assert hashlib.sha256(partial.read_bytes()).hexdigest() == asset['sha256']
            partial.rename(target)
        assert hashlib.sha256(target.read_bytes()).hexdigest() == asset['sha256']
        write(REPAIR/'CHECKPOINT_VERIFIED.json', {'sha256': asset['sha256'], 'bytes': target.stat().st_size})
        # Reconcile shared immutable inputs before two queue processes read them.
        sys.path.insert(0, str(REPAIR))
        from table_resume_streams_v3 import inventory
        existing, _ = inventory(ROOT, 'pointmaze')
        assert len(existing) == 53

        def run(spec):
            offset, devices, output = spec
            try:
                while True:
                    if time.time()+2700 > payload['deadline']:
                        raise RuntimeError('Insufficient deadline for receiving checks')
                    busy = subprocess.check_output(['nvidia-smi', '--id='+','.join(map(str, devices)),
                        '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
                    if not busy: break
                    time.sleep(5)
                env = dict(os.environ, MUJOCO_PY_FORCE_CPU='1', D4RL_SUPPRESS_IMPORT_ERROR='1',
                    MUJOCO_PY_MUJOCO_PATH=str(REPAIR/'mujoco210'),
                    LD_LIBRARY_PATH=str(REPAIR/'mujoco210/bin')+':/usr/lib/x86_64-linux-gnu:/opt/conda/lib',
                    OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
                cmd = [str(REPAIR/'python/bin/python'), '-u', str(REPAIR/'table_resume_streams_v3.py'),
                    '--task', 'pointmaze', '--instance', '50588893', '--gpus', '2',
                    '--total-gpus', '4', '--gpu-offset', str(offset), '--physical-gpus', *map(str, devices),
                    '--python', str(REPAIR/'python/bin/python'), '--output', str(output),
                    '--deadline', str(payload['deadline'])]
                with (REPAIR/(output.name+'.log')).open('x') as f:
                    subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
            except Exception as e:
                if not (output/'QUEUE_FAILED.json').exists():
                    write(output/'QUEUE_FAILED.json', {'error_type': type(e).__name__, 'detail': str(e)[:200],
                        'original_results_unchanged': True})
                raise

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(run, [(0, [2, 3], outputs[0]), (2, [0, 1], outputs[1])]))
        write(REPAIR/'CONTINUATION_DONE.json', {'new_scientific_episodes': 132, 'full_study_complete': False})
    except Exception as e:
        # Terminal markers permit preservation/stop, even if input staging fails.
        for output in outputs:
            if not (output/'QUEUE_FAILED.json').exists() and not (output/'QUEUE_DONE.json').exists():
                write(output/'QUEUE_FAILED.json', {'error_type': type(e).__name__, 'original_results_unchanged': True})
        raise


if __name__ == '__main__':
    main(json.load(sys.stdin))
