"""Run explicitly reserved, previously unassigned whole streams after a worker queue.

Operational-only: original scientific source, frozen scenarios and per-device
receiving checks are reused. No process is interrupted or existing output edited.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


EXTRA = {1: [('reach', 5), ('reach-wall', 5), ('reach', 7)],
         0: [('reach', 4), ('reach-wall', 4), ('reach-wall', 7)],
         3: [('reach', 6), ('reach-wall', 6)], 2: []}
ARMS = ('native', 'visual_only', 'action_condition_only', 'joint')


def coverage(prior):
    """Require exactly the observed four original assignments before reserving extras."""
    expected = {(task, rank) for task in ('reach', 'reach-wall') for rank in range(8)}
    old = [tuple(item) for slot in range(4) for item in prior[slot]]
    if set(old) != {(task, rank) for task in ('reach', 'reach-wall') for rank in range(4)} or len(old) != 8:
        raise ValueError('Original deployed allocation changed')
    combined = old + [item for items in EXTRA.values() for item in items]
    if len(combined) != len(expected) or set(combined) != expected:
        raise ValueError('Incomplete or duplicated panel allocation')
    return {slot: list(EXTRA[slot]) for slot in range(4)}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open('x') as output:
        json.dump(value, output, indent=2)


def run(root):
    spec = read(root / 'PLAN.json')
    amendment = root / 'ORDER_AMENDMENT.json'
    if amendment.exists():
        change = read(amendment)
        if (change['prior_plan_sha256'] != hashlib.sha256((root / 'PLAN.json').read_bytes()).hexdigest()
                or change['extra'] != sorted(spec['extra'], key=lambda item: item[0] != 'reach')):
            raise ValueError('Ordering amendment may not change any reserved scenario stream')
        spec['extra'] = change['extra']
    component = Path(spec['component_root'])
    code = component / 'code'
    freeze = code / 'artifacts/offline_study/table-completion-20260911-v1/component-extension-v1/freeze'
    completed = []
    try:
        while not (component / 'TERMINAL.json').exists():
            if time.time() >= spec['deadline'] - 180:
                raise TimeoutError('Original queue did not finish before retained-disk backstop')
            time.sleep(5)
        if read(component / 'TERMINAL.json')['status'] != 'assigned_component_streams_complete':
            raise ValueError('Original queue incomplete; no continuation of partial work')
        actual = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines()
        if actual != [spec['uuid']]:
            raise ValueError('Physical GPU identity changed')
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']).strip():
            raise ValueError('Another GPU workload is active')
        env = dict(os.environ, PYTHONPATH=str(code / 'src'), PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   JEPA_VERIFIED_LOCAL_DINO='1', SDL_VIDEODRIVER='dummy', MUJOCO_GL='egl',
                   PYOPENGL_PLATFORM='egl', CUDA_VISIBLE_DEVICES='0',
                   JEPAWM_DSET=str(component / 'data'), JEPAWM_LOGS=str(component / 'logs'),
                   JEPAWM_HOME=str(code / 'vendor'), JEPAWM_CKPT=str(Path(spec['checkpoint']).parent))
        python = '/workspace/component-python/bin/python'
        inputs = ['--vendor', str(code / 'vendor/jepa-wms'), '--original', str(code / 'artifacts/offline_study/primary-durable-20260907'),
                  '--stimuli', str(code / 'artifacts/offline_study/restored-behavioral-inputs-20260908-v1'),
                  '--freeze', str(freeze), '--checkpoint', spec['checkpoint']]
        for task, rank in spec['extra']:
            engineering = component / 'engineering' / spec['uuid'] / task
            for arm in ARMS:
                if time.time() >= spec['deadline'] - 180:
                    raise TimeoutError('Deadline before new intact stream')
                target = component / 'results' / task / arm / f'shard-{rank:02d}'
                if target.exists():
                    raise ValueError('Reserved stream already exists; do not overwrite or replay')
                command = [python, '-u', '-m', 'offline_study.metaworld_component_behavior', 'run',
                           *inputs, '--task', task, '--engineering', str(engineering),
                           '--arm', arm, '--logical-ranks', str(rank), '--output', str(target)]
                with (root / f'{task}-{arm}-{rank}.log').open('x') as log:
                    subprocess.run(command, env=env, stdout=log, stderr=log, check=True,
                                   timeout=max(1, spec['deadline'] - time.time() - 120))
                verify = [python, str(root / 'component_resume_check.py'), '--root', str(target),
                          '--freeze', str(freeze), '--engineering', str(engineering), '--task', task,
                          '--uuid', spec['uuid'], '--arm', arm, '--rank', str(rank)]
                digest = subprocess.check_output(verify, env=env, text=True, timeout=180).strip()
                completed.append({'task': task, 'rank': rank, 'arm': arm, 'report_sha256': digest})
                write(root / f'COMPLETE-{task}-{arm}-{rank}.json', completed[-1])
        write(root / 'TERMINAL.json', {'status': 'reserved_streams_complete', 'completed': completed,
                                     'full_panel_analysis_complete': False, 'time': time.time()})
    except Exception as error:
        write(root / 'TERMINAL.json', {'status': 'incomplete_preserve_all', 'error': str(error),
                                     'completed': completed, 'time': time.time()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)
