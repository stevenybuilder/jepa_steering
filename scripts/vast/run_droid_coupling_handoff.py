"""Owned Virginia: preserve a complete Maze epoch, run DROID, resume exact history.

Scheduling only: no changes to model, sampler, optimizer, evaluation count or
method. Original partial directory remains immutable after its deliberate stop.
Every exit after stopping training attempts the predeclared exact-checkpoint resume.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path('/workspace/jepa-runtime')
CODE = ROOT / 'droid-coupling-code-20260908-v2'
PANEL = ROOT / 'droid-coupling-behavior-20260908-v2'
FIT = ROOT / 'droid-coupling-fit-20260908-v2'
PYTHON = '/workspace/jepa-droid-python/bin/python'
DEVICE = 'GPU-5bbfd503-dbb3-c9ad-7b41-2a4f985b18fc'
OLD_QUEUE, OLD_CHILD = None, 11185
TRAINING = ROOT / 'pointmaze-training-history-20260908-v1/seed-234'
CHECKPOINT_EPOCH = 3


def command(pid):
    p = Path(f'/proc/{pid}/cmdline')
    return p.read_bytes().rstrip(b'\0').decode().split('\0') if p.exists() else []


def main():
    sys.path.insert(0, str(CODE / 'src'))
    from offline_study.author_fit import source_hash
    from offline_study.protocol import sha256, write_json
    from offline_study.droid_native import verified_report
    expected = json.loads((CODE / 'RECEIVING.json').read_text())
    if source_hash() != expected['source_sha256']:
        raise ValueError('Receiving DROID source changed')
    runtime, runtime_hash = verified_report(CODE / 'runtime-check')
    if runtime['status'] != 'droid_runtime_and_all512_native_prefixes_verified' or runtime['gpu_initialized']:
        raise ValueError('Missing complete correct-interpreter CPU preflight')
    if subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip() != DEVICE:
        raise ValueError('Wrong owned device')
    PANEL.mkdir(exist_ok=False)
    write_json(PANEL / 'QUEUE_LAUNCH.json', {'pid': os.getpid(), 'instance': 50189244, 'device_uuid': DEVICE,
        'source_sha256': source_hash(), 'predecessor_queue': OLD_QUEUE, 'predecessor_child': OLD_CHILD,
        'pause_after_complete_epoch': CHECKPOINT_EPOCH, 'outcome_based_selection': False,
        'training_resume_required': True, 'runtime_preflight_report_sha256': runtime_hash, 'fresh_confirmation': False})
    old_args = command(OLD_CHILD)
    if ('offline_study.pointmaze_training_history' not in old_args or
            str(TRAINING / 'resumed-after-droid-v1') not in old_args):
        raise ValueError('Original training process identity changed')
    training_env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', JEPA_VERIFIED_LOCAL_DINO='1',
        PYTHONPATH=str(ROOT / 'pointmaze-history-code-20260908-v1/src') + ':/workspace/jepa-python/lib/python3.10/site-packages',
        LD_LIBRARY_PATH='/opt/conda/lib', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    # DROID's verified Python3.11 runtime must not import the Python3.10 overlay.
    env = dict(training_env, PYTHONPATH=str(CODE / 'src'),
               MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl')
    child, paused, checkpoint = None, False, None
    def interrupt(signum, frame):
        raise RuntimeError('Owned DROID queue interrupted; preserve all outputs and resume training')
    signal.signal(signal.SIGTERM, interrupt); signal.signal(signal.SIGINT, interrupt)
    def execute(module, args, name, cap):
        nonlocal child
        logs = PANEL / 'queue-logs'; logs.mkdir(exist_ok=True)
        with (logs / (name + '.log')).open('x') as log:
            child = subprocess.Popen([PYTHON, '-u', '-m', module] + args, env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            if child.wait(timeout=cap):
                raise ValueError('Required DROID stage failed: ' + name)
        child = None
    try:
        marker = TRAINING / 'remaining-epochs' / f'epoch-{CHECKPOINT_EPOCH:03d}.json'
        deadline = time.monotonic() + 3600
        while not marker.exists():
            if command(OLD_CHILD) != old_args or time.monotonic() > deadline:
                raise ValueError('No expected completed training checkpoint; no pause performed')
            time.sleep(2)
        row = json.loads(marker.read_text())
        checkpoint = Path(row['checkpoint']['path'])
        if (row['epoch'] != CHECKPOINT_EPOCH or row['updates'] != 1139 or
                checkpoint != TRAINING / 'remaining-epochs' / f'jepa-e{CHECKPOINT_EPOCH-1}.pth.tar' or
                sha256(checkpoint) != row['checkpoint']['sha256']):
            raise ValueError('Training checkpoint incomplete or changed')
        if command(OLD_CHILD) != old_args:
            raise ValueError('Original training child changed before handoff')
        write_json(PANEL / 'TRAINING_PAUSE.json', {'reason': 'prioritize requested current-checkpoint DROID comparisons',
            'checkpoint': row['checkpoint'], 'epoch_receipt_sha256': sha256(marker),
            'original_command': old_args, 'original_output_preserved': True,
            'resume_output': str(TRAINING / 'resumed-after-droid-v2')})
        paused = True
        os.kill(OLD_CHILD, signal.SIGTERM)
        deadline = time.monotonic() + 90
        while subprocess.check_output(['nvidia-smi', '-i', '0', '--query-compute-apps=pid', '--format=csv,noheader']).strip():
            if time.monotonic() > deadline:
                raise TimeoutError('GPU not released; do not overlap independent workloads')
            time.sleep(2)
        common = ['--vendor', '/workspace/jepa_steering/vendor/jepa-wms',
            '--assets', str(ROOT / 'droid-assets-20260907-v1'), '--manifest', str(CODE / 'configs/droid_assets.json'),
            '--encoder-source', '/workspace/jepa_steering/dinov3', '--encoder-root', str(ROOT / 'dinov3-native-20260907-v2')]
        execute('offline_study.droid_coupling_fit', common + [
            '--inputs', str(ROOT / 'droid-fit-native-eligible-20260908-v1'),
            '--audit', str(ROOT / 'droid-fit-audit-20260908-v2'), '--output', str(FIT)], 'fit', 3900)
        behavior = common + ['--fit', str(FIT), '--reference', str(ROOT / 'droid-native-replication-20260907-v1/shard-all'),
            '--native-engineering', str(ROOT / 'droid-native-engineering-20260907-v7'), '--freeze', str(PANEL / 'freeze')]
        execute('offline_study.droid_coupling_behavior', ['freeze'] + behavior, 'freeze', 300)
        engineering = PANEL / 'engineering'
        execute('offline_study.droid_coupling_behavior', ['engineer'] + behavior + ['--output', str(engineering)], 'engineering', 7200)
        from offline_study.droid_coupling import ARMS
        for rank in range(8):
            for arm in ARMS[1:]:
                execute('offline_study.droid_coupling_behavior', ['run'] + behavior + ['--engineering', str(engineering),
                    '--logical-ranks', str(rank), '--arm', arm, '--output', str(PANEL / 'conditions' / arm / f'shard-{rank}')],
                    arm + f'-shard-{rank}', 3600)
        execute('offline_study.droid_coupling_behavior', ['analyze'] + behavior + ['--panel', str(PANEL),
            '--output', str(PANEL / 'analysis')], 'analysis', 1800)
        write_json(PANEL / 'QUEUE_DONE.json', {'analysis_report_sha256': sha256(PANEL / 'analysis/report.json'),
            'full_study_complete': False, 'fresh_confirmation': False})
    except Exception as exc:
        write_json(PANEL / 'QUEUE_FAILED.json', {'error': str(exc), 'partial_not_complete': True})
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
        if paused:
            if subprocess.check_output(['nvidia-smi', '-i', '0', '--query-compute-apps=pid', '--format=csv,noheader']).strip():
                write_json(PANEL / 'RESUME_BLOCKED.json', {'reason': 'GPU remains occupied; no overlap permitted'})
            else:
                resume_args = list(old_args)
                resume_args[resume_args.index('--resume-from') + 1] = str(checkpoint)
                resume_args[resume_args.index('--output') + 1] = str(TRAINING / 'resumed-after-droid-v2')
                with (PANEL / 'pointmaze-resume.log').open('x') as log:
                    resumed = subprocess.Popen(resume_args, env=training_env, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                write_json(PANEL / 'TRAINING_RESUMED.json', {'pid': resumed.pid, 'command': resume_args,
                    'checkpoint_sha256': sha256(checkpoint), 'completion_not_yet_verified': True})


if __name__ == '__main__':
    main()
