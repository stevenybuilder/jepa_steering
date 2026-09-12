"""Receiving checks then paired whole-stream work on one process per GPU."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2)


def assignments(gpus, total_gpus=None, offset=0, logical_slots=None):
    total = total_gpus or gpus
    slots = list(range(offset, offset + gpus)) if logical_slots is None else list(logical_slots)
    if (gpus not in (1, 2, 4, 8) or total not in (1, 2, 4, 8, 16) or len(slots) != gpus
            or len(set(slots)) != gpus or any(not 0 <= s < total for s in slots)):
        raise ValueError('Invalid physical-to-logical allocation')
    work = [(task, rank) for task in ('reach', 'reach-wall') for rank in range(8)]
    return {g: work[slots[g]::total] for g in range(gpus)}


def main(a):
    root, code = a.root, a.root / 'code'
    artifact = code / 'artifacts/offline_study'
    freeze = artifact / 'table-completion-20260911-v1/component-extension-v1/freeze'
    checkpoint = a.checkpoint or Path('/workspace/decision-runtime/checkpoints/jepa_wm_metaworld.pth.tar')
    env = dict(os.environ, PYTHONPATH=str(code / 'src'), JEPA_VERIFIED_LOCAL_DINO='1',
        PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        SDL_VIDEODRIVER='dummy', MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
        JEPAWM_DSET=str(root / 'data'), JEPAWM_LOGS=str(root / 'logs'),
        JEPAWM_HOME=str(code / 'vendor'), JEPAWM_CKPT=str(checkpoint.parent))
    tasks = ['reach', 'reach-wall']
    arms = ['native', 'visual_only', 'action_condition_only', 'joint']
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    errors = []
    try:
        audit = root / ('resume-' + str(time.time_ns())) if a.resume_verified else root
        audit.mkdir(parents=True, exist_ok=True)
        assert a.gpus in (1, 2, 4, 8)
        assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']).strip()
        devices = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines()
        assert len(devices) == a.gpus
        for path in (root / 'data', root / 'logs'):
            path.mkdir(exist_ok=True)
        subprocess.run([sys.executable, str(code / 'vendor/jepa-wms/setup_macros.py')], env=env, check=True, timeout=60)
        # Read-only exact-package/cache/freeze check. No pip changes to a reused runtime.
        check = """import importlib.metadata as m,json,torch
from pathlib import Path
from offline_study.metaworld_component_behavior import validate_protocol
from offline_study.author_fit import source_hash
from offline_study.model_loader import verified_local_dino_cache
from offline_study.protocol import sha256
import sys
p=validate_protocol(Path(sys.argv[1]));assert p['source_sha256']==source_hash()
assert sha256(Path(sys.argv[2]))==p['checkpoint_sha256']
assert torch.__version__=='2.7.1+cu128' and torch.cuda.is_available()
versions={x:m.version(x) for x in ('numpy','metaworld','mujoco','gym','gymnasium')}
assert versions=={'numpy':'2.2.6','metaworld':'3.1.1','mujoco':'3.3.0','gym':'0.23.1','gymnasium':'1.3.0'}
with verified_local_dino_cache() as proof:print(json.dumps({'packages':versions,'dino':proof}))
"""
        with (audit / 'RUNTIME_CHECK.log').open('x') as log:
            subprocess.run([sys.executable, '-c', check, str(freeze), str(checkpoint)],
                env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
        total = a.total_gpus or a.gpus
        queues = assignments(a.gpus, total, a.gpu_offset, a.logical_slots)
        plan = {'queues': queues, 'arms': arms, 'devices': devices,
            'evaluations': 768, 'missing_cell_evaluations': 576, 'paired_reference_evaluations': 192,
            'deadline': a.deadline, 'one_process_per_gpu': True, 'confirmation': False,
            'total_gpus': total, 'gpu_offset': a.gpu_offset,
            'logical_slots': a.logical_slots,
            'assigned_evaluations': 48 * sum(len(q) for q in queues.values())}
        if a.resume_verified:
            prior = json.loads((root / 'QUEUE_PLAN.json').read_text())
            for key in ('arms', 'devices', 'assigned_evaluations', 'total_gpus', 'gpu_offset'):
                if prior[key] != plan[key]:
                    raise ValueError('Recovery must preserve device and complete stream allocation')
            if prior['queues'] != json.loads(json.dumps(plan['queues'])):
                raise ValueError('Recovery cannot add or reassign original streams')
        write(audit / 'QUEUE_PLAN.json', plan)
        base = [sys.executable, '-u', '-m', 'offline_study.metaworld_component_behavior']
        inputs = ['--vendor', str(code / 'vendor/jepa-wms'), '--original', str(artifact / 'primary-durable-20260907'),
            '--stimuli', str(artifact / 'restored-behavioral-inputs-20260908-v1'), '--freeze', str(freeze),
            '--checkpoint', str(checkpoint)]

        def worker(gpu):
            child_env = dict(env, CUDA_VISIBLE_DEVICES=str(gpu))
            validated = set()
            def run(command, label):
                if stop.is_set() or time.time() >= a.deadline - 180:
                    raise RuntimeError('Queue stopped before starting a new intact stream')
                with (audit / (label + '.log')).open('x') as log:
                    process = subprocess.Popen(command, env=child_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    print(json.dumps({'gpu': gpu, 'job': label, 'pid': process.pid}), flush=True)
                    while process.poll() is None:
                        if stop.is_set() or time.time() >= a.deadline - 120:
                            os.killpg(process.pid, signal.SIGTERM)
                            try:
                                process.wait(timeout=20)
                            except subprocess.TimeoutExpired:
                                os.killpg(process.pid, signal.SIGKILL)
                                process.wait()
                            raise RuntimeError('Partial stream retained after bounded interruption')
                        time.sleep(3)
                    if process.returncode:
                        raise RuntimeError('Job failed: ' + label)
            def already_complete(output, engineering, task, arm=None, rank=None):
                if not a.resume_verified or not output.exists():
                    return False
                command = [sys.executable, str(Path(__file__).with_name('component_resume_check.py')),
                    '--root', str(output), '--freeze', str(freeze), '--engineering', str(engineering),
                    '--task', task, '--uuid', devices[gpu]]
                if arm is not None:
                    command += ['--arm', arm, '--rank', str(rank)]
                digest = subprocess.check_output(command, env=child_env, text=True, timeout=180).strip()
                write(audit / f'reused-{gpu}-{task}-{arm}-{rank}.json', {'output': str(output),
                    'completed_receipt_sha256': digest, 'partial_stream_reused': False})
                return True
            try:
                for task, rank in queues[gpu]:
                    engineering = root / 'engineering' / devices[gpu] / task
                    if task not in validated:
                        if not already_complete(engineering, engineering, task):
                            run(base + ['engineering', *inputs, '--task', task, '--output', str(engineering)],
                                f'gpu-{gpu}-{task}-engineering')
                        validated.add(task)
                    for arm in arms:
                        out = root / 'results' / task / arm / f'shard-{rank:02d}'
                        if already_complete(out, engineering, task, arm, rank):
                            continue
                        run(base + ['run', *inputs, '--task', task, '--engineering', str(engineering),
                            '--arm', arm, '--logical-ranks', str(rank), '--output', str(out)],
                            f'gpu-{gpu}-{task}-{arm}-{rank}')
                write(root / f'GPU-{gpu}-DONE.json', {'queue_complete': True})
            except Exception as e:
                errors.append({'gpu': gpu, 'error': str(e)})
                stop.set()
                write(root / f'GPU-{gpu}-FAILED.json', errors[-1])
        with ThreadPoolExecutor(max_workers=a.gpus) as pool:
            list(pool.map(worker, range(a.gpus)))
        if errors:
            raise RuntimeError('At least one receiving/paired stream failed; no completed-panel claim')
        if total != a.gpus:
            write(root / 'TERMINAL.json', {'status': 'assigned_component_streams_complete',
                'full_panel_analysis_complete': False, 'time': time.time()})
            return
        analysis_env = dict(env, CUDA_VISIBLE_DEVICES='')
        with (root / 'analysis.log').open('x') as log:
            subprocess.run([sys.executable, '-u', '-m', 'offline_study.metaworld_component_analysis',
                '--root', str(root / 'results'), '--freeze', str(freeze), '--engineering-root', str(root / 'engineering'),
                '--output', str(root / 'analysis')], env=analysis_env, stdout=log, stderr=subprocess.STDOUT,
                check=True, timeout=300)
        write(root / 'TERMINAL.json', {'status': 'component_panel_analyzed', 'historical_native_equivalence_checked': False,
            'all_six_task_blanks_complete': False, 'time': time.time()})
    except Exception as e:
        write(root / 'TERMINAL.json', {'status': 'incomplete_preserve_all', 'error': str(e), 'children': errors, 'time': time.time()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--gpus', type=int, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--total-gpus', type=int)
    parser.add_argument('--gpu-offset', type=int, default=0)
    parser.add_argument('--logical-slots', type=int, nargs='+')
    parser.add_argument('--resume-verified', action='store_true',
        help='Reuse only complete hash-verified streams on the same GPU; reject partial outputs')
    main(parser.parse_args())
