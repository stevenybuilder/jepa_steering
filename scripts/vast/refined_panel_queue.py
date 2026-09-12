"""One unchanged full-budget task panel on one physical GPU, then resume MW.

Native and both edited arms use the same original whole RNG streams. No
partial replay, outcome-based selection, candidate-count reduction or solver.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)


def jobs(root, spec):
    if spec['task'] not in ('pusht', 'pointmaze', 'wall', 'droid') or spec['logical_ranks'] != list(range(8)):
        raise ValueError('Expected all eight original single-device scenario streams')
    python = spec.get('python', '/workspace/component-python/bin/python')
    if python != '/workspace/component-python/bin/python' and not (
            spec['task'] == 'pointmaze' and python == str(root / 'runtime/python/bin/python')):
        raise ValueError('Only the isolated PointMaze interpreter may replace the shared interpreter')
    droid = spec['task'] == 'droid'
    base = [python, '-u', '-m', 'offline_study.droid_fixed_response_behavior' if droid else 'offline_study.refined_task_behavior']
    flags = []
    keys = (('vendor', 'fit', 'assets', 'manifest', 'reference', 'native-engineering', 'encoder-source', 'encoder-root')
            if droid else ('task', 'vendor', 'checkpoint', 'fit', 'cohort', 'reference', 'reference-source', 'data-root'))
    for key in keys:
        flags.extend(['--' + key, spec[key]])
    flags += ['--freeze', str(root / 'freeze')]
    engineering = root / 'engineering'
    result = [('engineering', engineering, base + ['engineer'] + flags + ['--output', str(engineering)])]
    for rank in spec['logical_ranks']:
        native = root / 'conditions/native' / f'shard-{rank}'
        for arm in ('native', 'fixed_rank4', 'matched_random_fixed_rank4'):
            target = root / 'conditions' / arm / f'shard-{rank}'
            command = base + ['run'] + flags + ['--engineering', str(engineering),
                '--arm', arm, '--logical-ranks', str(rank), '--output', str(target)]
            if arm != 'native': command += ['--native', str(native)]
            result.append((f'{arm}-{rank}', target, command))
    return result


def main(root):
    spec = read(root / 'PANEL.json')
    env = dict(os.environ)
    if spec['task'] == 'pointmaze':
        env.update(MUJOCO_PY_FORCE_CPU='1', D4RL_SUPPRESS_IMPORT_ERROR='1',
            MUJOCO_PY_MUJOCO_PATH=str(root / 'runtime/mujoco210'),
            LD_LIBRARY_PATH=str(root / 'runtime/mujoco210/bin') + ':/usr/lib/x86_64-linux-gnu:/opt/conda/lib')
    completed = []
    try:
        for label, target, command in jobs(root, spec):
            if target.exists():
                raise ValueError('No partial-stream restart or overwrite')
            with (root / (label + '.log')).open('x') as log:
                subprocess.run(command, cwd=root / 'code', env=env, stdout=log, stderr=subprocess.STDOUT,
                    check=True, timeout=3600)
            report = target / 'report.json'
            if (target / 'FAILED.json').exists() or read(target / 'DONE.json')['report_sha256'] != hashlib.sha256(report.read_bytes()).hexdigest():
                raise ValueError('Missing verified complete child')
            completed.append(label)
            print(json.dumps({'task': spec['task'], 'completed_child': label,
                'complete_scientific_streams': len(completed) - 1}), flush=True)
        write(root / 'PANEL_DONE.json', {'task': spec['task'], 'completed': completed,
            'scientific_evaluations': 192 if spec['task'] == 'droid' else 288,
            'episodes_per_arm': 64 if spec['task'] == 'droid' else 96,
            'analysis_complete': False, 'time': time.time()})
    except Exception as error:
        write(root / 'PANEL_FAILED.json', {'error_type': type(error).__name__,
            'completed': completed, 'partial_preserved_not_counted': True, 'time': time.time()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
