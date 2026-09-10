"""CPU-only completion coordinator for the already frozen 40-shard panel.

No outcome-based early selection, retries, new GPU work, or confirmation access.
Only complete receipts let the unchanged scientific analyzer run.
"""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/workspace/jepa-runtime')
ARMS = ('native', 'fixed_rank4', 'matched_random_fixed_rank4',
        'coupling_only', 'matched_random_coupling')


def main():
    code = ROOT / 'fixed-response-code-20260908-v4'
    evidence = ROOT / 'fixed-response-behavior-evidence-20260908-v1/artifacts/offline_study'
    result = ROOT / 'fixed-response-behavior-20260908-v1'
    launch = json.loads((ROOT / 'fixed-response-behavior-launch-20260908-v1/receipt.json').read_text())
    if launch['instance'] != 50233992 or [w['gpu'] for w in launch['workers']] != list(range(8)):
        raise ValueError('Require the complete existing eight-worker assignment')
    expected = {i: [result / ('reach' if i < 4 else 'reach-wall') / arm / f'shard-gpu{i}'
                    for arm in ARMS] for i in range(8)}
    started, previous = time.monotonic(), None
    while True:
        completed = 0
        for worker in launch['workers']:
            gpu = worker['gpu']
            paths = expected[gpu]
            check = ROOT / f'fixed-response-worker-checks-20260908-v1/gpu-{gpu}'
            if any((p / 'FAILED.json').exists() for p in [check] + paths):
                raise RuntimeError(f'GPU{gpu} has a failed required job; preserve it, no auto-retry')
            count = sum((p / 'DONE.json').exists() for p in paths)
            completed += count
            if count < 5:
                process = Path(f"/proc/{worker['pid']}")
                if not process.exists():
                    raise RuntimeError(f'GPU{gpu} queue is terminal with incomplete receipts')
                command = (process / 'cmdline').read_bytes()
                if b'run_fixed_behavior_queue.py' not in command:
                    raise RuntimeError(f'GPU{gpu} queue identity no longer matches')
        if completed != previous:
            print(json.dumps({'complete_shards': completed, 'required_shards': 40,
                              'partial_selection_forbidden': True}), flush=True)
            previous = completed
        if completed == 40:
            break
        if time.monotonic() - started > 86400:
            raise TimeoutError('Monitoring deadline only; existing experiment jobs are not stopped')
        time.sleep(20)
    env = dict(os.environ, PYTHONPATH=str(code / 'src') + ':/workspace/jepa-python/lib/python3.10/site-packages',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    subprocess.run(['/workspace/jepa-planning-python/bin/python', '-u', '-m',
        'offline_study.fixed_response_behavior_analysis', '--root', str(result),
        '--freeze', str(evidence / 'fixed-response-20260908-v1/behavioral-freeze-v1'),
        '--output', str(ROOT / 'fixed-response-behavior-analysis-20260908-v1')],
        env=env, check=True, timeout=3600)
    print(json.dumps({'status': 'complete_panel_analyzed', 'confirmation_launched': False,
                      'full_six_task_study_complete': False}), flush=True)


if __name__ == '__main__':
    main()
