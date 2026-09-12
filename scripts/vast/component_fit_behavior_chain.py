"""Adopt an already-held coordinator and run fit -> freeze -> behavior once.

Operational only. Original jobs/source/data remain immutable. Never signal a
scientific child; the predecessor finishes intact before GPU work is admitted.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time

from component_boundary_job import validate_job, wait_gpu_release
from component_queue_handoff import direct_children, process, write
from refined_panel_queue import jobs


def read(path):
    return json.loads(path.read_text())


def validate_adoption(spec):
    component = Path(spec['component_root'])
    queue = process(spec['queue_pid'])
    if (not queue or queue['state'] not in ('T', 't')
            or queue['args'] != spec['queue_args']
            or queue['args'][queue['args'].index('--root') + 1] != str(component)):
        raise ValueError('Require the exact previously suspended owned coordinator')
    receipt = read(Path(spec['fit_root']) / 'WAITING_BOUNDARY.json')
    if (receipt['queue_pid'] != spec['queue_pid']
            or receipt['continuing_scientific_pid'] != spec['scientific_pid']
            or receipt['continuing_output'] != spec['scientific_output']
            or not Path(spec['scientific_output']).is_relative_to(component / 'results')):
        raise ValueError('Predecessor identity changed')
    child = process(spec['scientific_pid'])
    if child and child['state'] not in ('Z', 'X') and child['args'] != spec['scientific_args']:
        raise ValueError('Scientific PID identity changed')
    if (Path(spec['fit_root']) / 'FIT_STARTED.json').exists():
        raise ValueError('Cannot adopt a fit that already started')
    return queue


def run_command(command, env, cwd, log, deadline):
    remaining = deadline - time.time()
    if remaining <= 120:
        raise TimeoutError('Insufficient operational lease for another job')
    with log.open('x') as output:
        child = subprocess.Popen(command, cwd=cwd, env=env, stdout=output,
                                 stderr=output, start_new_session=True)
        try:
            code = child.wait(timeout=remaining - 60)
            if code:
                raise subprocess.CalledProcessError(code, command)
        except BaseException:
            # A panel owns subprocesses too. Release the entire launched group
            # before a finally block can resume a different GPU workload.
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise


def finalize_behavior(root, fit_root, env, deadline):
    """Complete the CPU-only part already prepared before fitting."""
    if (not (root / 'RUNTIME_READY.json').exists()
            or read(root / 'RESET_PARITY.json').get('all97_original_inputs_exact') is not True):
        raise ValueError('Navigation receiving preparation is not complete')
    spec = read(root / 'PANEL.json')
    if spec['task'] != 'wall' or Path(spec['prerequisite']) != fit_root:
        raise ValueError('Only the prepared Wall fit/behavior pairing is registered')
    if not (fit_root / 'fit-v1/DONE.json').exists() or (fit_root / 'fit-v1/FAILED.json').exists():
        raise ValueError('Fit did not finish')
    shutil.copytree(fit_root / 'fit-v1', root / 'fit')
    command = jobs(root, spec)[0][2][:]
    command[4] = 'freeze'
    command[command.index('--output') + 1] = str(root / 'freeze')
    cpu_env = dict(env, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(root / 'code/src'))
    run_command(command, cpu_env, root / 'code', root / 'freeze.log', min(deadline, time.time() + 420))
    write(root / 'INPUTS_READY.json', {'time': time.time(), 'task': 'wall', 'gpu_calls': 0,
                                     'direct_fit_to_behavior_handoff': True})
    files = {}
    for name in ('code', 'ops', 'fit', 'freeze', 'reference', 'reference-source'):
        for p in (root / name).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:
                files[str(p.relative_to(root))] = {'bytes': p.stat().st_size,
                                                   'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    p = root / 'PANEL.json'
    files['PANEL.json'] = {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    job = {'kind': 'refined_panel', 'task': 'wall', 'timeout_seconds': 21600,
           'cwd': str(root / 'code'), 'gpu_uuid': spec['physical_gpu_uuid'], 'files': files,
           'command': ['/workspace/component-python/bin/python', '-u',
                       str(root / 'ops/refined_panel_queue.py'), '--root', str(root)]}
    write(root / 'JOB.json', job)
    validate_job(job, root)
    return job


def main(root):
    spec = read(root / 'PLAN.json')
    original = validate_adoption(spec)
    fit_root, panel_root = Path(spec['fit_root']), Path(spec['panel_root'])
    fit_job = read(fit_root / 'JOB.json')
    validate_job(fit_job, fit_root)
    if fit_job['task'] != 'wall':
        raise ValueError('Only the registered waiting Wall fit may be adopted')
    if subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip() != fit_job['gpu_uuid']:
        raise ValueError('Device changed')
    write(root / 'ADOPTED.json', {'time': time.time(), 'queue_pid': spec['queue_pid'],
                                'scientific_child_interrupted': False})
    def interrupt(*_):
        raise InterruptedError('Operational chain interrupted')
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    owns_gpu = False
    try:
        while True:
            child = process(spec['scientific_pid'])
            if child is None or child['state'] in ('Z', 'X'):
                break
            if child['args'] and child['args'] != spec['scientific_args']:
                raise ValueError('Scientific child identity changed')
            if time.time() >= spec['deadline'] - 180:
                raise TimeoutError('Predecessor exceeded lease')
            time.sleep(1)
        predecessor = Path(spec['scientific_output'])
        if (not (predecessor / 'DONE.json').exists() or (predecessor / 'FAILED.json').exists()
                or read(predecessor / 'DONE.json')['report_sha256'] != hashlib.sha256((predecessor / 'report.json').read_bytes()).hexdigest()):
            raise ValueError('Predecessor did not complete intact')
        wait_gpu_release()
        owns_gpu = True
        env = dict(os.environ, PYTHONPATH=str(fit_root / 'code/src'), CUDA_VISIBLE_DEVICES='0',
                   JEPA_VERIFIED_LOCAL_DINO='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', SDL_VIDEODRIVER='dummy',
                   MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl')
        write(fit_root / 'FIT_STARTED.json', {'time': time.time(), 'direct_chain': str(root)})
        run_command(fit_job['command'], env, fit_root / 'code', fit_root / 'fit.log',
                    min(spec['deadline'], time.time() + fit_job['timeout_seconds'] + 60))
        if not (fit_root / 'fit-v1/DONE.json').exists() or (fit_root / 'fit-v1/FAILED.json').exists():
            raise ValueError('Unverified fit')
        # Hold the coordinator throughout CPU freezing. There is no intervening
        # SIGCONT and therefore no second full-batch wait or GPU race.
        job = finalize_behavior(panel_root, fit_root, env, spec['deadline'])
        write(fit_root / 'TERMINAL.json', {'status': 'fit_complete_behavioral_validation_still_required', 'time': time.time()})
        wait_gpu_release()
        write(panel_root / 'PANEL_STARTED.json', {'time': time.time(), 'direct_chain': str(root)})
        env['PYTHONPATH'] = str(panel_root / 'code/src')
        run_command(job['command'], env, panel_root / 'code', panel_root / 'panel.log',
                    min(spec['deadline'], time.time() + job['timeout_seconds'] + 60))
        if not (panel_root / 'PANEL_DONE.json').exists() or (panel_root / 'PANEL_FAILED.json').exists():
            raise ValueError('Behavioral panel incomplete')
        write(panel_root / 'TERMINAL.json', {'status': 'behavioral_panel_complete_analysis_still_required', 'time': time.time()})
        write(root / 'TERMINAL.json', {'status': 'fit_and_behavior_complete', 'time': time.time()})
    except Exception as error:
        write(root / 'TERMINAL.json', {'status': 'incomplete_preserve_all', 'error': str(error), 'time': time.time()})
        for target in (fit_root, panel_root):
            if not (target / 'TERMINAL.json').exists():
                write(target / 'TERMINAL.json', {'status': 'incomplete_preserve_all', 'chain': str(root), 'error': str(error), 'time': time.time()})
        raise
    finally:
        current = process(spec['queue_pid'])
        if current and current['args'] == original['args']:
            if owns_gpu:
                wait_gpu_release()
            os.kill(spec['queue_pid'], signal.SIGCONT)
            for target in (root, fit_root, panel_root):
                if not (target / 'COMPONENTS_RESUMED.json').exists():
                    write(target / 'COMPONENTS_RESUMED.json', {'time': time.time(), 'queue_pid': spec['queue_pid']})


def adopt_panel_timer(root):
    """Replace only a short operational timer, without restarting the panel."""
    spec = read(root / 'PLAN.json')
    component, panel = Path(spec['component_root']), Path(spec['panel_root'])
    queue, boundary = process(spec['queue_pid']), process(spec['boundary_pid'])
    active = process(spec['panel_pid'])
    if (not queue or queue['state'] not in ('T', 't') or queue['args'] != spec['queue_args']
            or direct_children(spec['queue_pid']) or not active or active['args'] != spec['panel_args']
            or not boundary or boundary['args'] != spec['boundary_args']):
        raise ValueError('Only idle operational parents of the active panel may be replaced')
    if not (panel / 'PANEL_STARTED.json').exists() or (panel / 'TERMINAL.json').exists():
        raise ValueError('Panel lifecycle changed')
    validate_job(read(panel / 'JOB.json'), panel)
    if not time.time() < spec['deadline'] <= float(queue['args'][queue['args'].index('--deadline') + 1]):
        raise ValueError('Timer cannot extend beyond the existing lease')
    retired = False
    try:
        os.kill(spec['boundary_pid'], signal.SIGSTOP)
        children = direct_children(spec['boundary_pid'])
        if (len(children) != 1 or children[0][0] != spec['panel_pid']
                or process(spec['panel_pid'])['args'] != active['args'] or direct_children(spec['queue_pid'])):
            raise ValueError('Active work changed during adoption')
        write(root / 'ADOPTED.json', {'time': time.time(), 'active_panel_pid_unchanged': spec['panel_pid'],
            'prior_timeout_seconds': read(panel / 'JOB.json')['timeout_seconds'],
            'replacement_deadline': spec['deadline'], 'scientific_process_signaled': False})
        # The panel has its own session and remains alive. Never killpg the
        # adopted experiment except at its registered lease deadline.
        os.kill(spec['boundary_pid'], signal.SIGKILL)
        retired = True
        while True:
            current = process(spec['panel_pid'])
            if not current or current['state'] in ('Z', 'X'): break
            if current['args'] and current['args'] != active['args']:
                raise ValueError('Panel PID identity changed')
            if time.time() >= spec['deadline'] - 120:
                os.killpg(spec['panel_pid'], signal.SIGTERM)
                for _ in range(20):
                    current = process(spec['panel_pid'])
                    if not current or current['state'] in ('Z', 'X'): break
                    time.sleep(1)
                else: os.killpg(spec['panel_pid'], signal.SIGKILL)
                raise TimeoutError('Registered lease reached; retain panel output')
            time.sleep(3)
        wait_gpu_release()
        complete = (panel / 'PANEL_DONE.json').exists() and not (panel / 'PANEL_FAILED.json').exists()
        if not (panel / 'TERMINAL.json').exists():
            write(panel / 'TERMINAL.json', {'status': 'behavioral_panel_complete_analysis_still_required' if complete
                else 'incomplete_preserve_all', 'operational_timer_repaired': True, 'time': time.time()})
        write(root / 'TERMINAL.json', {'status': 'panel_terminated',
            'panel_complete': complete, 'full_study_complete': False, 'time': time.time()})
    except BaseException as error:
        if not (root / 'TERMINAL.json').exists():
            write(root / 'TERMINAL.json', {'status': 'incomplete_preserve_all', 'error': str(error), 'time': time.time()})
        if not retired:
            if process(spec['boundary_pid']): os.kill(spec['boundary_pid'], signal.SIGCONT)
        # If retirement succeeded, do not start a competing GPU process after
        # an error. The registered spending guard retains all data on stop.
        raise
    finally:
        active_now = process(spec['panel_pid'])
        if retired and (not active_now or active_now['state'] in ('Z', 'X')):
            wait_gpu_release()
            current = process(spec['queue_pid'])
            if current and current['args'] == queue['args']:
                os.kill(spec['queue_pid'], signal.SIGCONT)
                write(panel / 'COMPONENTS_RESUMED.json', {'time': time.time(), 'queue_pid': spec['queue_pid']})
            if not (panel / 'TERMINAL.json').exists():
                write(panel / 'TERMINAL.json', {'status': 'incomplete_preserve_all', 'time': time.time()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--adopt-panel-timer', action='store_true')
    options = parser.parse_args()
    (adopt_panel_timer if options.adopt_panel_timer else main)(options.root)
