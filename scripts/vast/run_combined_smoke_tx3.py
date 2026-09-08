"""One bounded full-planner check after the successful numerical child exits."""
import argparse
import fcntl
import importlib
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import run_combined_check_tx3 as prior

ROOT = Path('/workspace/jepa-runtime/combined-full-planner-tx3-20260908-v2')
# Receiving snapshot retains the numerical archive's AppleDouble metadata files.
# Imported .py bytes match local c15f5ed1; do not rewrite the checked snapshot.
SCIENCE = 'e64debad97734af166b4bf8b21db0d0256747895c596ee38ee7bceef175729a3'
REPORT = 'c6faf9cd35eb981457269df3876e2226853e3b3fe2d7e8fecc52fe955ee52897'
PRIOR_SCRIPT = '5d3bc3451590792fa2e09396ec2a5eb0ad42c9dff694e3eb58898f2157323e39'
LIMIT = 3610
GL_LIBRARIES = {
    'libEGL.so.1': '69816a7062d7d624144472f5ef71f5b23732b9003df7472662920602e62805db',
    'libGLdispatch.so.0': 'a4f2642498fba54d2f39e90ca3daa94340cdd52198473185a5285608ae318cb6',
    'libOpenGL.so.0': '35ea98e979c5bcadf06a79b8ca5a767a197c77bc77428a9a8a16b95bf78e99e9',
}


def command():
    inputs = prior.ROOT / 'inputs'
    return ['/workspace/jepa-planning-python/bin/python', '-u', '-m', 'offline_study.fixed_combined_smoke',
        '--vendor', '/workspace/jepa_steering/vendor/jepa-wms',
        '--checkpoint', str(inputs/'fixed-response-assets-20260908-v1/jepa_wm_metaworld.pth.tar'),
        '--fit', str(inputs/'fixed-response-fit-20260908-v1/reach'),
        '--coupling-fit', str(inputs/'combined-fit-stimulus-preparation-20260908-v1/coupling-evidence/reach/vision_action_coupling'),
        '--numerical-check', str(prior.ROOT/'check'), '--numerical-report-sha256', REPORT,
        '--checked-source', str(prior.ROOT/'src/offline_study'), '--output', str(ROOT/'check')]


def validate_authority(value, script_sha, tests_sha, *, now=None):
    expected = {'owner': 'rep_geometry_transcoder/root', 'instance': 50259194, 'gpu': 3,
        'gpu_uuid': prior.UUID, 'source_sha256': SCIENCE, 'script_sha256': script_sha,
        'receiving_tests_sha256': tests_sha, 'hourly_cap_usd': 7, 'outer_seconds': LIMIT,
        'operation': 'four_combined_full_planner_engineering_episodes', 'automatic_retry': False,
        'behavioral_launch_authorized': False, 'numerical_report_sha256': REPORT}
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError('Exact full-planner engineering authority required')
    now = time.time() if now is None else now
    issued, expires, rate = (value.get(k) for k in ('issued_unix', 'expires_unix', 'observed_account_hourly_usd'))
    if (any(type(v) not in (int, float) or not math.isfinite(v) for v in (issued, expires, rate)) or
            not issued <= now < now+LIMIT+5 <= expires or not 0 < expires-issued <= 43200 or not 0 <= rate <= 7):
        raise ValueError('Authority must cover whole check/cleanup within budget')


def execute(c, environment, status):
    child = None
    try:
        with (status/'child.log').open('x') as log:
            start = time.monotonic()
            child = subprocess.Popen(command(), env=environment, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            c.write(status/'SPAWNED.json', {'pid': child.pid, 'command': command()})
            binding = c.started(child.pid, command())
            c.write(status/'CHILD.json', binding)
            code = child.wait(timeout=max(0, start+LIMIT-time.monotonic()))
        c.write(status/'EXIT.json', {'identity': binding, 'returncode': code})
        if code: raise ValueError('Full-planner check failed; no automatic retry')
        return binding
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=5)
            c.write(status/'CLEANUP.json', {'pid': child.pid, 'returncode': child.returncode,
                'only_owned_unreaped_child_signalled': True, 'predecessor_signals': 0})


def child_environment(c):
    if any(prior.sha(ROOT/'lib'/name) != digest for name, digest in GL_LIBRARIES.items()):
        raise ValueError('Private same-Ubuntu EGL dispatch libraries changed')
    environment = c.environment(50259194, 3)
    environment.update(MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', MUJOCO_EGL_DEVICE_ID='3',
        LD_LIBRARY_PATH=str(ROOT/'lib')+':/opt/conda/lib',
        PYTHONPATH=str(ROOT/'src')+':/workspace/jepa-planning-python/lib/python3.10/site-packages:/workspace/jepa-python/lib/python3.10/site-packages')
    environment.pop('MUJOCO_PY_FORCE_CPU', None)
    return environment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authorization-sha256', required=True)
    args = parser.parse_args(argv)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('PYTHONOPTIMIZE') not in (None, '', '0'):
        raise ValueError('CUDA-hidden nonoptimized supervisor required')
    if prior.sha(Path(prior.__file__)) != PRIOR_SCRIPT or prior.sha(ROOT/'AUTHORIZATION.json') != args.authorization_sha256:
        raise ValueError('Reviewed launcher helper/authority bytes changed')
    authority = prior.read(ROOT/'AUTHORIZATION.json')
    validate_authority(authority, prior.sha(Path(__file__)), prior.sha(ROOT/'receiving-tests-v2.log'))
    if prior.source_digest(ROOT/'src/offline_study') != SCIENCE or prior.sha(prior.NAV/'FILES.json') != prior.NAV_FILES:
        raise ValueError('Reviewed source changed')
    for name, expected in prior.read(prior.NAV/'FILES.json').items():
        path = prior.NAV/name
        if Path(name).name != name or path.is_symlink() or prior.sha(path) != expected['sha256']:
            raise ValueError('Frozen operations helper changed')
    sys.path.insert(0, str(prior.NAV))
    c = importlib.import_module('navigation_redistribution_common')
    if any(Path(m.__file__).resolve().parent != prior.NAV for m in (c, c.lifecycle)):
        raise ValueError('Wrong lifecycle helper')
    status = ROOT/'run'; status.mkdir(exist_ok=False)
    try:
        c.write(status/'LAUNCH.json', {'identity': c.process(os.getpid()),
            'authorization_sha256': args.authorization_sha256, 'source_sha256': SCIENCE,
            'instance': 50259194, 'gpu': 3, 'gpu_uuid': prior.UUID})
        previous = prior.read(prior.ROOT/'run/DONE.json')
        child = previous['child_identity']
        parent = prior.read(prior.ROOT/'run/LAUNCH.json')['identity']
        if (previous['report_sha256'] != REPORT or prior.sha(prior.ROOT/'check/report.json') != REPORT or
                child['pid'] != 34447 or child['starttime'] != 43159340 or
                parent['pid'] != 34439 or parent['starttime'] != 43158993 or
                (prior.ROOT/'run/FAILED.json').exists() or (prior.ROOT/'check/FAILED.json').exists()):
            raise ValueError('Wrong/incomplete predecessor check')
        if c.alive(parent) or c.alive(child): raise ValueError('Predecessor still owns GPU')
        lock = ROOT.parent/('gpu-ownership-'+prior.UUID+'.lock')
        if lock.is_symlink(): raise ValueError('Unsafe ownership lock')
        with lock.open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                release = c.wait_gpu_release(3, prior.UUID, known_children=[parent, child])
                c.write(status/'DEVICE_RELEASE.json', release)
                validate_authority(authority, prior.sha(Path(__file__)), prior.sha(ROOT/'receiving-tests-v2.log'))
                if (ROOT/'check').exists(): raise FileExistsError('No repeated check')
                environment = child_environment(c)
                binding = execute(c, environment, status)
                c.wait_gpu_release(3, prior.UUID, known_children=[binding])
                report, done = prior.read(ROOT/'check/report.json'), prior.read(ROOT/'check/DONE.json')
                if (prior.sha(ROOT/'check/report.json') != done['report_sha256'] or
                        report['status'] != 'fixed_combined_full_cem_engineering_complete' or
                        report['full_simulator_episodes'] != 4 or not report['native_repeat_exact'] or
                        not report['full_cem_schedule_verified'] or (ROOT/'check/FAILED.json').exists()):
                    raise ValueError('Incomplete full-planner check')
                c.write(status/'DONE.json', {'report_sha256': done['report_sha256'],
                    'child_identity': binding, 'behavioral_launch_ready': False})
            finally: fcntl.flock(stream, fcntl.LOCK_UN)
    except BaseException as error:
        c.write(status/'FAILED.json', {'error': repr(error), 'automatic_retry': False,
            'predecessor_signals': 0, 'behavioral_launch_ready': False})
        raise


if __name__ == '__main__': main()
