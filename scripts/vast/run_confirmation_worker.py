"""One newly owned host, one task, disjoint exclusive logical streams."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import time

PROJECT = Path('/workspace/confirmation')
PYTHON = '/workspace/decision-python/bin/python'
RUNTIME = Path('/workspace/confirmation-output')
CHECKPOINT = '/workspace/decision-runtime/checkpoints/jepa_wm_metaworld.pth.tar'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=('reach','reach-wall'), required=True)
    p.add_argument('--deadline', type=float, required=True)
    p.add_argument('--logical-ranks', type=int, nargs='+', required=True)
    args = p.parse_args()
    if len(set(args.logical_ranks)) != len(args.logical_ranks) or any(r not in range(8) for r in args.logical_ranks):
        raise ValueError('Invalid logical stream allocation')
    RUNTIME.mkdir(exist_ok=False)
    children = {}
    def stop(signum, frame):
        for child in list(children.values()):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        raise SystemExit('Bounded queue interrupted; preserve outputs')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    def run(gpu):
        rank = args.logical_ranks[gpu]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu),
            MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', JEPA_VERIFIED_LOCAL_DINO='1',
            OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
            PYTHONPATH=str(PROJECT/'src'), JEPAWM_DSET='/workspace/decision-runtime/data',
            JEPAWM_LOGS='/workspace/decision-runtime/logs', JEPAWM_HOME=str(PROJECT/'vendor'),
            JEPAWM_CKPT='/workspace/decision-runtime/checkpoints')
        fixed = PROJECT/'artifacts/offline_study/fixed-response-20260908-v1'
        evidence = PROJECT/'artifacts/offline_study'
        engineering = RUNTIME/'engineering'/f'gpu-{gpu}'
        def execute(argv, name, cap):
            remaining = args.deadline-time.time()
            if remaining < 60:
                raise RuntimeError('Compute deadline: preserve partial run, do not extend')
            with (RUNTIME/(name+'.log')).open('x') as log:
                child = subprocess.Popen(argv, cwd=PROJECT, env=env, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
                children[gpu] = child
                try:
                    code = child.wait(timeout=min(cap, remaining))
                    if code:
                        raise RuntimeError(f'{name} exited {code}; preserve source')
                finally:
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGTERM)
                        try: child.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                    children.pop(gpu, None)
        try:
            execute([PYTHON, '-u', '-m', 'offline_study.fixed_response_behavior', 'worker-engineering',
                '--vendor', str(PROJECT/'vendor/jepa-wms'), '--fits', str(fixed/'fits'),
                '--original', str(evidence/'primary-durable-20260907'),
                '--stimuli', str(evidence/'restored-behavioral-inputs-20260908-v1'),
                '--checkpoint', CHECKPOINT, '--task', args.task,
                '--numerical-check', str(fixed/'numerical-checks'/args.task),
                '--checked-source', str(fixed/'numerical-source-v1/src/offline_study'),
                '--output', str(engineering)], f'engineering-{gpu}', 2700)
            for arm in ('native','fixed_rank4','matched_random_fixed_rank4','coupling_only','matched_random_coupling'):
                execute([PYTHON, '-u', '-m', 'offline_study.confirmation', 'run',
                    '--project', str(PROJECT), '--vendor', str(PROJECT/'vendor/jepa-wms'),
                    '--freeze', str(evidence/'confirmation-20260911-v1/freeze-v2'), '--checkpoint', CHECKPOINT,
                    '--engineering', str(engineering), '--task', args.task, '--arm', arm,
                    '--logical-rank', str(rank), '--output', str(RUNTIME/'results'/args.task/arm/f'rank-{rank}')],
                    f'{args.task}-{arm}-{gpu}', 7200)
            (RUNTIME/f'GPU-{gpu}-DONE.json').write_text(json.dumps({'task': args.task, 'gpu': gpu, 'episodes': 60}))
        except Exception as error:
            (RUNTIME/f'GPU-{gpu}-FAILED.json').write_text(json.dumps({'error': str(error), 'partial_not_complete': True}))
            raise
    with ThreadPoolExecutor(max_workers=len(args.logical_ranks)) as pool:
        list(pool.map(run, range(len(args.logical_ranks))))
    (RUNTIME/'WORKER_DONE.json').write_text(json.dumps({'task': args.task, 'episodes': 60*len(args.logical_ranks),
        'final_pair_analysis_pending': True}))


if __name__ == '__main__':
    main()
