"""Start fixed task stripes after asset staging; each worker gates science on engineering."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--worker-id', type=int, required=True)
p.add_argument('--gpu-offset', type=int, default=0)
p.add_argument('--tasks', nargs='+', choices=['reach','reach-wall','pointmaze','wall'], required=True)
p.add_argument('--engineering-only', action='store_true')
p.add_argument('--attempt', default='primary')
a = p.parse_args()
root = a.root.resolve()
assert 0 <= a.worker_id < 6
tasks = ['reach','reach-wall','pointmaze','wall']
for task in a.tasks:
    asset = 'metaworld' if task.startswith('reach') else task
    assert (root / f'ASSET_READY_{asset}.json').is_file(), f'Missing asset gate: {asset}'
    gpu = a.gpu_offset + tasks.index(task)
    logs = root / ('worker-logs' if a.attempt == 'primary' else 'worker-logs-' + a.attempt)
    logs.mkdir(exist_ok=True)
    receipt = logs / f'{task}-worker{a.worker_id}-LAUNCHED.json'
    if receipt.exists():
        raise RuntimeError(f'Existing launch receipt; inspect before retry: {receipt}')
    command = [sys.executable, '-u', str(root / 'scripts/run_fresh_confirmation.py'),
        'worker', '--project', str(root), '--freeze', str(root / 'fresh-freeze'),
        '--task', task, '--worker-id', str(a.worker_id), '--workers', '6',
        '--output', str(root / 'results')]
    if a.engineering_only:
        command = [sys.executable, '-u', str(root / 'scripts/run_fresh_confirmation.py'),
            'engineering', '--project', str(root), '--freeze', str(root / 'fresh-freeze'),
            '--task', task, '--episode', '0', '--output', str(root / ('engineering-' + a.attempt) / task)]
    with (logs / f'{task}-worker{a.worker_id}.log').open('xb') as log:
        child = subprocess.Popen(command, cwd=root, env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu)},
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    data = {'pid':child.pid,'task':task,'gpu':gpu,'worker_id':a.worker_id,'workers_per_task':6,
            'command':command,'stage':'receiving_engineering_then_science_if_pass',
            'scientific_completion_claimed':False}
    with receipt.open('x') as f: json.dump(data,f,indent=2)
    print(json.dumps(data),flush=True)
