"""One owned GPU, receiving-worker check then two intact streams in all five arms."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess

ROOT = Path('/workspace/jepa-runtime')
PYTHON = '/workspace/jepa-planning-python/bin/python'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, choices=range(8), required=True)
    args = parser.parse_args()
    task = 'reach' if args.gpu < 4 else 'reach-wall'
    rank = args.gpu % 4
    code = ROOT / 'fixed-response-code-20260908-v4'
    evidence = ROOT / 'fixed-response-behavior-evidence-20260908-v1/artifacts/offline_study'
    fixed = evidence / 'fixed-response-20260908-v1'
    output = ROOT / f'fixed-response-worker-checks-20260908-v1/gpu-{args.gpu}'
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
        JEPA_VERIFIED_LOCAL_DINO='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        PYTHONPATH=str(code / 'src') + ':/workspace/jepa-python/lib/python3.10/site-packages',
        LD_LIBRARY_PATH='/opt/conda/lib')
    common = ['--vendor', '/workspace/jepa_steering/vendor/jepa-wms', '--fits', str(fixed / 'fits'),
        '--original', str(evidence / 'primary-durable-20260907'),
        '--stimuli', str(evidence / 'restored-behavioral-inputs-20260908-v1'),
        '--checkpoint', str(ROOT / 'fixed-response-assets-20260908-v1/jepa_wm_metaworld.pth.tar'),
        '--task', task]
    base = [PYTHON, '-u', '-m', 'offline_study.fixed_response_behavior']
    running = None
    def interrupted(signum, frame):
        raise RuntimeError('Owned behavioral queue interrupted; preserve partial outputs')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    def execute(command, log, cap):
        nonlocal running
        with log.open('x') as stream:
            running = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            if running.wait(timeout=cap) != 0:
                raise RuntimeError('Required job failed; do not advance or promote a partial panel')
        running = None
    try:
        print(json.dumps({'stage': 'receiving_worker_full_cem', 'gpu': args.gpu, 'task': task}), flush=True)
        execute(base + ['worker-engineering'] + common + ['--checked-source',
            str(fixed / 'numerical-source-v1/src/offline_study'), '--numerical-check',
            str(fixed / 'numerical-checks' / task), '--output', str(output)],
            ROOT / f'fixed-behavior-worker-check-gpu{args.gpu}-20260908-v1.log', 3600)
        for arm in ('native', 'fixed_rank4', 'matched_random_fixed_rank4', 'coupling_only', 'matched_random_coupling'):
            target = ROOT / f'fixed-response-behavior-20260908-v1/{task}/{arm}/shard-gpu{args.gpu}'
            print(json.dumps({'stage': 'scientific_development', 'gpu': args.gpu, 'task': task,
                'arm': arm, 'logical_ranks': [rank, rank+4], 'episodes_in_this_shard': 24,
                'episodes_per_task_condition_across_all_gpus': 96}), flush=True)
            execute(base + ['run'] + common + ['--freeze', str(fixed / 'behavioral-freeze-v1'),
                '--engineering', str(output), '--checked-source', str(code / 'src/offline_study'),
                '--arm', arm, '--logical-ranks', str(rank), str(rank+4), '--output', str(target)],
                ROOT / f'fixed-behavior-{task}-{arm}-gpu{args.gpu}-20260908-v1.log', 10800)
        print(json.dumps({'status': 'assigned_five_arm_streams_complete', 'gpu': args.gpu,
                          'full_panel_analysis_still_required': True}), flush=True)
    finally:
        if running is not None and running.poll() is None:
            os.killpg(running.pid, signal.SIGTERM)
            try:
                running.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(running.pid, signal.SIGKILL)
                running.wait()


if __name__ == '__main__':
    main()
