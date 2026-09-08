"""One owned GPU: frozen navigation engineering, then all eight candidate arms.

Each scientific job is one intact 12-episode logical stream. No offline gate,
partial selection, automatic retry, support/rank solver, or confirmation exposure.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=('wall', 'pointmaze'), required=True)
    parser.add_argument('--gpu', choices=(0, 1), type=int, required=True)
    args = parser.parse_args()
    if (args.task, args.gpu) not in (('wall', 1), ('pointmaze', 0)):
        raise ValueError('Wrong reserved Nebraska task/device mapping')
    root = Path('/workspace/jepa-runtime')
    code = root / 'navigation-coupling-code-20260908-v2'
    evidence = root / 'navigation-coupling-evidence-20260908-v2/artifacts/offline_study'
    output = root / 'navigation-coupling-behavior-20260908-v1' / args.task
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), JEPA_VERIFIED_LOCAL_DINO='1',
        MUJOCO_PY_FORCE_CPU='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        PYTHONPATH=str(code / 'src') + ':/workspace/jepa-maze-python/lib/python3.10/site-packages:'
            '/workspace/jepa-planning-python/lib/python3.10/site-packages:/workspace/jepa-python/lib/python3.10/site-packages',
        LD_LIBRARY_PATH='/root/.mujoco/mujoco210/bin:/opt/conda/lib')
    base = ['/workspace/jepa-planning-python/bin/python', '-u', '-m', 'offline_study.navigation_coupling_behavior']
    fit = evidence / 'primary-durable-20260907/navigation-fits-20260907-v1/bfloat16' / args.task
    common = ['--vendor', '/workspace/jepa_steering/vendor/jepa-wms', '--task', args.task,
        '--reference', str(evidence / 'navigation-coupling-reference-20260908-v1'),
        '--fit', str(fit / 'vision_action_coupling'), '--cohort', str(fit / 'cohort.json'),
        '--freeze', str(evidence / 'navigation-coupling-behavior-20260908-v1' / args.task / 'freeze'),
        '--checkpoint', str(root / f'navigation-assets-20260907-v1/downloads/model/jepa_wm_{args.task}.pth.tar')]
    child = None
    def interrupted(signum, frame):
        raise RuntimeError('Owned navigation queue interrupted; preserve all partial results')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    def execute(command, name, cap):
        nonlocal child
        with (root / f'nav-coupling-{args.task}-{name}-20260908-v1.log').open('x') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            if child.wait(timeout=cap) != 0:
                raise RuntimeError('Required navigation job failed; do not advance')
        child = None
    try:
        execute(base + ['engineer'] + common + ['--output', str(output / 'engineering')], 'engineering', 3600)
        for arm in ('visual_only', 'action_condition_only', 'joint', 'joint_equal_standardized_energy',
                    'permuted_visual', 'permuted_joint', 'matched_random', 'matched_random_equal_standardized_energy'):
            for rank in range(8):
                print(json.dumps({'stage': 'scientific_development', 'task': args.task, 'arm': arm,
                    'logical_rank': rank, 'episodes_in_shard': 12, 'total_per_task_condition': 96}), flush=True)
                execute(base + ['run'] + common + ['--engineering', str(output / 'engineering'),
                    '--arm', arm, '--logical-ranks', str(rank), '--output',
                    str(output / 'conditions' / arm / f'shard-{rank}')], f'{arm}-shard-{rank}', 5400)
        print(json.dumps({'status': 'task_candidate_shards_complete', 'two_task_analysis_pending': True}), flush=True)
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()


if __name__ == '__main__':
    main()
