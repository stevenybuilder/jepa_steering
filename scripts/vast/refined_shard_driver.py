"""Run an explicit subset of a refined panel's whole RNG streams on one GPU.

Mirrors refined_panel_queue.jobs() command construction exactly (same source,
flags, per-logical-rank seeds). Differences: (1) accepts a shard subset so
distinct hosts can divide the eight streams, (2) skips a shard only when its
DONE.json report hash verifies, (3) a partial or unverified shard directory is
removed and the whole stream rerun from its seed -- never resumed mid-stream.
All three arms of a shard run on this same physical GPU, native first.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ARMS = ('native', 'fixed_rank4', 'matched_random_fixed_rank4')


def read(path):
    return json.loads(path.read_text())


def verified(target):
    done, report = target / 'DONE.json', target / 'report.json'
    if not done.exists() or not report.exists() or (target / 'FAILED.json').exists():
        return False
    return read(done)['report_sha256'] == hashlib.sha256(report.read_bytes()).hexdigest()


def build(root, spec):
    python = spec.get('python', '/workspace/component-python/bin/python')
    droid = spec['task'] == 'droid'
    base = [python, '-u', '-m', 'offline_study.droid_fixed_response_behavior' if droid else 'offline_study.refined_task_behavior']
    keys = (('vendor', 'fit', 'assets', 'manifest', 'reference', 'native-engineering', 'encoder-source', 'encoder-root')
            if droid else ('task', 'vendor', 'checkpoint', 'fit', 'cohort', 'reference', 'reference-source', 'data-root'))
    flags = []
    for key in keys:
        flags.extend(['--' + key, spec[key]])
    flags += ['--freeze', str(root / 'freeze')]
    return base, flags


def main(root, shards):
    spec = read(root / 'PANEL.json')
    base, flags = build(root, spec)
    env = dict(os.environ)
    if spec['task'] == 'pointmaze':
        env.update(MUJOCO_PY_FORCE_CPU='1', D4RL_SUPPRESS_IMPORT_ERROR='1',
            MUJOCO_PY_MUJOCO_PATH=str(root / 'runtime/mujoco210'),
            LD_LIBRARY_PATH=str(root / 'runtime/mujoco210/bin') + ':/usr/lib/x86_64-linux-gnu:/opt/conda/lib')
    engineering = root / 'engineering'
    log_dir = root / 'driver-logs'
    log_dir.mkdir(exist_ok=True)

    def run(label, command):
        with (log_dir / f'{label}.{int(time.time())}.log').open('x') as log:
            subprocess.run(command, cwd=root / 'code', env=env, stdout=log, stderr=subprocess.STDOUT,
                check=True, timeout=7200)
        print(json.dumps({'task': spec['task'], 'completed': label, 'time': time.time()}), flush=True)

    if not (engineering / 'report.json').exists():
        run('engineering', base + ['engineer'] + flags + ['--output', str(engineering)])
    for rank in shards:
        native = root / 'conditions/native' / f'shard-{rank}'
        for arm in ARMS:
            target = root / 'conditions' / arm / f'shard-{rank}'
            if verified(target):
                print(json.dumps({'task': spec['task'], 'skip_verified': f'{arm}-{rank}'}), flush=True)
                continue
            if target.exists():
                shutil.rmtree(target)  # whole-stream replay; no partial resume
                print(json.dumps({'task': spec['task'], 'replaced_partial': f'{arm}-{rank}'}), flush=True)
            command = base + ['run'] + flags + ['--engineering', str(engineering),
                '--arm', arm, '--logical-ranks', str(rank), '--output', str(target)]
            if arm != 'native':
                command += ['--native', str(native)]
            run(f'{arm}-{rank}', command)
            if not verified(target):
                raise ValueError(f'Unverified child {arm}-{rank}')
    (root / f'DRIVER_DONE.{"-".join(map(str, shards))}.json').write_text(json.dumps(
        {'task': spec['task'], 'shards': list(shards), 'time': time.time()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--shards', type=int, nargs='+', required=True)
    args = parser.parse_args()
    main(args.root, args.shards)
