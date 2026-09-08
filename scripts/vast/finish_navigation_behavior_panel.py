"""CPU-only watcher for the fixed, complete two-task navigation factorial.

Never starts/retries GPU jobs or opens confirmation. Native reference and every
candidate shard are verified by the unchanged scientific analyzer before results.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path('/workspace/jepa-runtime')
    code = root / 'navigation-coupling-code-20260908-v2'
    evidence = root / 'navigation-coupling-evidence-20260908-v2/artifacts/offline_study'
    output = root / 'navigation-coupling-behavior-20260908-v1'
    launch = json.loads((root / 'navigation-coupling-launch-20260908-v1.json').read_text())
    if launch['instance'] != 50231985 or set(launch['queues']) != {'wall', 'pointmaze'}:
        raise ValueError('Wrong owned queue launch receipt')
    arms = ('visual_only', 'action_condition_only', 'joint', 'joint_equal_standardized_energy',
            'permuted_visual', 'permuted_joint', 'matched_random', 'matched_random_equal_standardized_energy')
    paths = {}
    for task in ('wall', 'pointmaze'):
        source = evidence / 'navigation-coupling-behavior-20260908-v1' / task / 'freeze'
        target = output / task / 'freeze'
        if not target.exists():
            shutil.copytree(source, target)
        for name in ('protocol.json', 'FROZEN.json'):
            if digest(source / name) != digest(target / name):
                raise ValueError('Never replace an existing differing freeze')
        if digest(target / 'protocol.json') != json.loads((target / 'FROZEN.json').read_text())['protocol_sha256']:
            raise ValueError('Invalid copied freeze')
        paths[task] = [output / task / 'conditions' / arm / f'shard-{i}' for arm in arms for i in range(8)]
    previous, started = None, time.monotonic()
    while True:
        counts = {}
        for task, candidates in paths.items():
            if any((p / 'FAILED.json').exists() for p in candidates + [output / task / 'engineering']):
                raise RuntimeError('Preserve failed required job; no automatic retry')
            counts[task] = sum((p / 'DONE.json').exists() for p in candidates)
            if counts[task] < 64:
                process = Path(f"/proc/{launch['queues'][task]['pid']}")
                if not process.exists() or b'run_navigation_behavior_queue_v2.py' not in (process / 'cmdline').read_bytes():
                    raise RuntimeError('Queue terminated/identity changed with incomplete receipts')
        if counts != previous:
            print(json.dumps({'complete_shards': counts, 'required_per_task': 64,
                              'partial_selection_forbidden': True}), flush=True)
            previous = counts.copy()
        if sum(counts.values()) == 128:
            break
        if time.monotonic() - started > 172800:
            raise TimeoutError('Monitor deadline only; existing jobs are not stopped')
        time.sleep(20)
    env = dict(os.environ, PYTHONPATH=str(code / 'src') + ':/workspace/jepa-python/lib/python3.10/site-packages',
               CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    subprocess.run(['/workspace/jepa-planning-python/bin/python', '-u', '-m',
        'offline_study.navigation_coupling_analysis', '--root', str(output),
        '--reference', str(evidence / 'navigation-coupling-reference-20260908-v1'),
        '--output', str(root / 'navigation-coupling-behavior-analysis-20260908-v1')],
        env=env, check=True, timeout=3600)
    print(json.dumps({'status': 'complete_two_task_navigation_panel_analyzed',
                      'confirmation_launched': False, 'full_study_complete': False}), flush=True)


if __name__ == '__main__':
    main()
