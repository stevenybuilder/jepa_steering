"""Four disjoint owned devices; eight unchanged persistent DROID RNG streams.

New-device native streams precede their paired candidates. No 64-per-GPU inflation,
cross-hardware reference reuse, outcome-based selection or shared GPU workloads.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path('/workspace/jepa-runtime')
CODE = ROOT / 'droid-coupling-code-20260908-v3'
PANEL = ROOT / 'droid-coupling-behavior-20260908-v3'
PYTHON = '/workspace/jepa-droid-python/bin/python'


def arguments():
    return ['--vendor', '/workspace/jepa_steering/vendor/jepa-wms',
        '--assets', str(ROOT / 'droid-assets-20260907-v1'), '--manifest', str(CODE / 'configs/droid_assets.json'),
        '--encoder-source', '/workspace/jepa_steering/dinov3', '--encoder-root', str(ROOT / 'dinov3-native-20260907-v2'),
        '--fit', str(ROOT / 'droid-coupling-fit-20260908-v2'),
        '--reference', str(ROOT / 'droid-native-replication-20260907-v1/shard-all'),
        '--native-engineering', str(ROOT / 'droid-native-engineering-20260907-v7'), '--freeze', str(PANEL / 'freeze')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, choices=range(4))
    parser.add_argument('--analyze', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(CODE / 'src'))
    from offline_study.author_fit import source_hash
    from offline_study.droid_coupling import ARMS
    from offline_study.protocol import sha256, write_json
    receiving = json.loads((CODE / 'RECEIVING.json').read_text())
    worker = json.loads((CODE / 'WORKER.json').read_text())
    if source_hash() != receiving['source_sha256'] or worker['purpose'] != 'four_gpu_paired_droid_v3':
        raise ValueError('Frozen receiving source or owned worker changed')
    env = dict(os.environ, PYTHONPATH=str(CODE / 'src'), CUDA_VISIBLE_DEVICES='' if args.analyze else str(args.gpu),
        LD_LIBRARY_PATH='/opt/conda/lib', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl')
    if args.analyze:
        deadline = time.monotonic() + 36*3600
        while not all((PANEL / f'queue-gpu{gpu}/DONE.json').is_file() for gpu in range(4)):
            if list(PANEL.glob('queue-gpu*/FAILED.json')) or time.monotonic() > deadline:
                raise ValueError('DROID panel incomplete; no partial analysis')
            time.sleep(30)
        subprocess.run([PYTHON, '-u', '-m', 'offline_study.droid_coupling_behavior', 'analyze'] + arguments() +
            ['--panel', str(PANEL), '--output', str(PANEL / 'analysis')], env=env, check=True, timeout=1800)
        write_json(PANEL / 'PANEL_DONE.json', {'analysis_sha256': sha256(PANEL / 'analysis/report.json'),
            'full_study_complete': False, 'fresh_confirmation': False})
        return
    if args.gpu is None:
        raise ValueError('Require an explicitly assigned GPU')
    gpu = args.gpu
    uuid = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    if uuid != worker['gpu_uuids'][gpu]:
        raise ValueError('Assigned receiving GPU changed')
    if subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid', '--format=csv,noheader']).strip():
        raise ValueError('Assigned GPU already occupied; no overlap')
    output = PANEL / f'queue-gpu{gpu}'
    output.mkdir(exist_ok=False)
    ranks = list(range(gpu, 8, 4))
    write_json(output / 'LAUNCH.json', {'pid': os.getpid(), 'instance': worker['instance'], 'device_uuid': uuid,
        'logical_ranks': ranks, 'episodes_per_arm_on_device': 16, 'episodes_per_arm_global': 64,
        'source_sha256': source_hash(), 'selection_from_outcomes': False, 'native_policy': 'new_same_device_paired'})
    child = None
    def interrupt(signum, frame):
        raise RuntimeError('Owned DROID queue interrupted; preserve all outputs')
    signal.signal(signal.SIGTERM, interrupt); signal.signal(signal.SIGINT, interrupt)
    def execute(command, extra, name):
        nonlocal child
        with (output / (name + '.log')).open('x') as log:
            child = subprocess.Popen([PYTHON, '-u', '-m', 'offline_study.droid_coupling_behavior', command] +
                arguments() + extra, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
            if child.wait(timeout=7200):
                raise ValueError('Required DROID stage failed: ' + name)
        child = None
    try:
        engineering = PANEL / f'engineering-gpu{gpu}'
        execute('engineer', ['--output', str(engineering)], 'engineering')
        for rank in ranks:
            for arm in ARMS:
                execute('run', ['--engineering', str(engineering), '--logical-ranks', str(rank), '--arm', arm,
                    '--output', str(PANEL / 'conditions' / arm / f'shard-{rank}')], f'{arm}-shard-{rank}')
        write_json(output / 'DONE.json', {'completed_paired_streams': ranks, 'arms': list(ARMS),
            'episodes': 144, 'not_full_study_completion': True, 'gpu_available_after_process_exit': True})
    except Exception as exc:
        write_json(output / 'FAILED.json', {'error': str(exc), 'partial_not_complete': True}); raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()


if __name__ == '__main__':
    main()
