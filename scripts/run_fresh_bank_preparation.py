"""Bounded parallel input preparation; never invokes a learned planner."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / 'artifacts/offline_study/fresh-simulator-banks-20260912-v1'
LOGS = BANK / 'parallel-preparation'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(task, rank):
    target = BANK / task / f'rank-{rank:02d}'
    report = json.loads((target / 'report.json').read_text())
    assert report['task'] == task and report['rank'] == rank
    assert report['scientific_candidates'] == 12 and report['excluded_engineering'] == 1
    assert report['learned_policy_calls'] == 0
    assert report['protocol_sha256'] == digest(BANK / 'protocol.json')
    assert report['records_sha256'] == digest(target / 'records.json')
    records = json.loads((target / 'records.json').read_text())
    assert len(records) == 13
    for row in records:
        assert digest(target / row['tensor_file']) == row['tensor_sha256']
    return report


def run(job):
    task, rank = job
    target = BANK / task / f'rank-{rank:02d}'
    if target.exists():
        return verify(task, rank)  # Never overwrite or retry a failed/partial shard.
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               SDL_VIDEODRIVER='dummy', SDL_AUDIODRIVER='dummy', MPLBACKEND='Agg')
    if task == 'pointmaze':
        env.update(CUDA_VISIBLE_DEVICES='', MUJOCO_PY_FORCE_CPU='1', D4RL_SUPPRESS_IMPORT_ERROR='1',
            MUJOCO_PY_MUJOCO_PATH='/workspace/preparation-runtime/mujoco210',
            LD_LIBRARY_PATH='/workspace/preparation-runtime/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/opt/conda/lib')
    else:
        env['MUJOCO_GL'] = 'egl'
    with (LOGS / f'{task}-{rank:02d}.log').open('x') as log:
        subprocess.run([sys.executable, str(ROOT / 'scripts/prepare_fresh_simulator_banks.py'),
            'run', '--task', task, '--rank', str(rank)], cwd=ROOT, env=env,
            stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
    return verify(task, rank)


if __name__ == '__main__':
    LOGS.mkdir(exist_ok=False)
    jobs = [(task, rank) for task in ('reach', 'reach-wall', 'pointmaze', 'wall') for rank in range(8)]
    results, failures = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(run, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            try:
                results.append(future.result())
                print(json.dumps({'verified_shards': len(results), 'job': job}), flush=True)
            except Exception as exc:
                failures.append({'job': job, 'error': str(exc)})
                print(json.dumps(failures[-1]), flush=True)
    report = {'verified_shards': len(results), 'failures': failures, 'shards': results,
              'model_calls': 0, 'scientific_launch_ready': False}
    (LOGS / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    if failures:
        raise SystemExit(1)
