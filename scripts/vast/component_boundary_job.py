"""Run one bounded prerequisite between complete component streams on one GPU.

Suspend only the coordinator, never its scientific child. Its unchanged queue
continues in finally. No partial replay, second GPU process, or new rental.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from component_queue_handoff import direct_children, process, write


def validate_queue(item, root):
    if (not item or item['state'] in ('T', 't', 'Z') or
            len(item['args']) < 4 or not item['args'][2].endswith('/component_extension_queue.py')):
        raise ValueError('Require an active owned component coordinator')
    args = item['args']
    if args[args.index('--root') + 1] != str(root):
        raise ValueError('Wrong component root')
    if args[args.index('--gpus') + 1] != '1':
        raise ValueError('This insertion supports only a single-device coordinator')


def wait_gpu_release(timeout=30):
    """Process exit and driver-context release need not be simultaneous."""
    deadline=time.monotonic()+timeout
    while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip():
        if time.monotonic()>=deadline:raise ValueError('GPU remains occupied after child exit; no multiplexing')
        time.sleep(1)


def validate_job(job, root):
    command = job['command']
    panel = job.get('kind') == 'refined_panel'
    additions = job.get('pythonpath_additions', [])
    if additions and (job['task'] != 'droid' or additions != [str(root / 'runtime-packages')]):
        raise ValueError('Only the isolated owned DROID dependency overlay is allowed')
    inherited = job.get('inherited_job')
    if inherited:
        prior_root = root.with_name('refined-droid-prerequisite-20260911-v1')
        prior = prior_root / 'JOB.json'
        if (job['task'] != 'droid' or inherited['path'] != str(prior)
                or hashlib.sha256(prior.read_bytes()).hexdigest() != inherited['sha256']):
            raise ValueError('Inherited DROID input manifest changed')
        old = json.loads(prior.read_text())
        if old.get('inherited_job'):
            raise ValueError('Do not chain inherited jobs')
        validate_job(old, prior_root)
    modules = {'pusht': 'offline_study.refined_task_fit',
               'droid': 'offline_study.droid_fixed_response_fit',
               'pointmaze': 'offline_study.navigation_refined_pipeline',
               'wall': 'offline_study.navigation_refined_pipeline'}
    navigation = job['task'] in ('pointmaze', 'wall')
    if panel:
        if (job['task'] not in modules or not 0 < job['timeout_seconds'] <= 21600
                or command != ['/workspace/component-python/bin/python', '-u',
                    str(root / 'ops/refined_panel_queue.py'), '--root', str(root)]):
            raise ValueError('Unexpected registered refined panel command')
    elif (command[:4] != ['/workspace/component-python/bin/python', '-u', '-m',
                        modules.get(job['task'])] or
            not 0 < job['timeout_seconds'] <= (7500 if navigation else 3900)):
        raise ValueError('Only a registered bounded task-specific fit is allowed')
    if panel:
        pass
    elif navigation:
        if command[4:] != ['--root', str(root)]:
            raise ValueError('Unexpected navigation pipeline root')
    elif Path(command[command.index('--output') + 1]) != root / 'fit-v1':
        raise ValueError('Unexpected fit output')
    if Path(job['cwd']) != root / 'code':
        raise ValueError('Unexpected source root')
    for name, want in job['files'].items():
        path = root / name
        if (not path.resolve().is_relative_to(root.resolve()) or path.is_symlink()
                or not path.is_file() or path.stat().st_size != want['bytes']):
            raise ValueError('Missing or invalid frozen input: ' + name)
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(4 << 20), b''):
                h.update(chunk)
        if h.hexdigest() != want['sha256']:
            raise ValueError('Changed frozen input: ' + name)


def main(args):
    job = json.loads(args.job.read_text())
    root = args.job.parent
    validate_job(job, root)
    original = process(args.queue_pid)
    validate_queue(original, args.component_root)
    if (args.component_root / 'TERMINAL.json').exists():
        raise ValueError('Component coordinator already terminal')
    device = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid',
        '--format=csv,noheader'], text=True).strip()
    if device != job['gpu_uuid']:
        raise ValueError('Physical device changed')
    suspended, child_fit = False, None
    def interrupted(*_):
        raise InterruptedError('Operational insertion interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        os.kill(args.queue_pid, signal.SIGSTOP)
        suspended = True
        children = direct_children(args.queue_pid)
        if len(children) != 1 or 'offline_study.metaworld_component_behavior' not in children[0][1]['args']:
            raise ValueError('Require exactly one continuing component child')
        pid, child = children[0]
        child_root = Path(child['args'][child['args'].index('--output') + 1])
        if not child_root.is_relative_to(args.component_root):
            raise ValueError('Component child output outside owned root')
        write(root / 'WAITING_BOUNDARY.json', {'time': time.time(), 'queue_pid': args.queue_pid,
            'continuing_scientific_pid': pid, 'continuing_output': str(child_root),
            'scientific_child_interrupted': False})
        deadline = time.monotonic() + 7200
        while time.monotonic() < deadline:
            current = process(pid)
            if current is None or current['state'] in ('Z','X'):
                break
            if not current['args']:
                # /proc can lose argv during normal exit before becoming Z/X.
                time.sleep(.1)
                continue
            if current['args'] != child['args']:
                raise ValueError('Scientific PID identity changed')
            time.sleep(3)
        else:
            raise TimeoutError('Complete-stream boundary not reached in two hours')
        if not (child_root / 'DONE.json').exists() or (child_root / 'FAILED.json').exists():
            raise ValueError('Component stream did not complete; do not insert')
        wait_gpu_release()
        panel = job.get('kind') == 'refined_panel'
        write(root / ('PANEL_STARTED.json' if panel else 'FIT_STARTED.json'), {'time': time.time(), 'complete_component_stream': str(child_root),
            'job_sha256': hashlib.sha256(args.job.read_bytes()).hexdigest()})
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root / 'code/src'),
            *job.get('pythonpath_additions', [])]), CUDA_VISIBLE_DEVICES='0',
            JEPA_VERIFIED_LOCAL_DINO='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', SDL_VIDEODRIVER='dummy',
            MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl')
        with (root / ('panel.log' if panel else 'fit.log')).open('x') as log:
            child_fit = subprocess.Popen(job['command'], cwd=job['cwd'], env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            code = child_fit.wait(timeout=job['timeout_seconds'])
        complete = root / ('PANEL_DONE.json' if panel else 'fit-v1/DONE.json')
        failed = root / ('PANEL_FAILED.json' if panel else 'fit-v1/FAILED.json')
        if code or not complete.exists() or failed.exists():
            raise RuntimeError('Fit incomplete; preserve output and resume components')
        write(root / 'TERMINAL.json', {'status': ('behavioral_panel_complete_analysis_still_required' if panel
            else 'fit_complete_behavioral_validation_still_required'),
            'time': time.time(), 'no_component_episode_interrupted': True})
    except Exception as error:
        write(root / 'TERMINAL.json', {'status': 'incomplete_preserve_all',
            'error_type': type(error).__name__, 'error': str(error), 'time': time.time()})
    finally:
        if child_fit and child_fit.poll() is None:
            os.killpg(child_fit.pid, signal.SIGTERM)
            try:
                child_fit.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child_fit.pid, signal.SIGKILL)
                child_fit.wait()
        current = process(args.queue_pid)
        if suspended and current and current['args'] == original['args']:
            os.kill(args.queue_pid, signal.SIGCONT)
            write(root / 'COMPONENTS_RESUMED.json', {'time': time.time(), 'queue_pid': args.queue_pid})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job', type=Path, required=True)
    p.add_argument('--component-root', type=Path, required=True)
    p.add_argument('--queue-pid', type=int, required=True)
    main(p.parse_args())
