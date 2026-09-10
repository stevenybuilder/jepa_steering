"""One bounded numerical check after verified tx3 navigation; no queue/retry."""
import argparse
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/workspace/jepa-runtime/combined-engineering-tx3-20260908-v1')
NAV = Path('/workspace/jepa-runtime/navigation-redistribution-20260908-v2')
UUID = 'GPU-23f2a085-92f0-239e-c7ec-c67c11f2f571'
SCIENCE = '014cf59678fea3105db0d8d16292228030f593555e55a4b60e2b603ecee84351'
NAV_FILES = '9497a41e8d8121be87fa94337b18384da9c35558e58e795975adb0cf5f1f9f27'
NAV_PLAN = '378cec55b21b84cf3d91b35fe0cfbc4f4b793a0f33b4cc6a726d17f1a4a46a63'
LIMIT = 1210


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''): h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def source_digest(path):
    h = hashlib.sha256()
    for p in sorted(Path(path).glob('*.py')):
        if p.is_symlink(): raise ValueError('Scientific source symlink')
        h.update(p.name.encode() + b'\0' + p.read_bytes())
    return h.hexdigest()


def validate_authority(value, script_sha, tests_sha, *, now=None):
    expected = {'owner': 'rep_geometry_transcoder/root', 'instance': 50259194, 'gpu': 3,
        'gpu_uuid': UUID, 'source_sha256': SCIENCE, 'script_sha256': script_sha,
        'receiving_tests_sha256': tests_sha, 'hourly_cap_usd': 7, 'outer_seconds': LIMIT,
        'operation': 'one_combined_numerical_check_after_tx3', 'automatic_retry': False,
        'behavioral_launch_authorized': False}
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError('Exact bounded root authorization required')
    now = time.time() if now is None else now
    issued, expires, rate = (value.get(k) for k in ('issued_unix', 'expires_unix', 'observed_account_hourly_usd'))
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in (issued, expires, rate)):
        raise ValueError('Invalid authority time/rate')
    if not issued <= now < now + LIMIT + 5 <= expires or not 0 < expires-issued <= 43200 or not 0 <= rate <= 7:
        raise ValueError('Authority must cover complete check/cleanup within budget')


def command():
    inputs = ROOT / 'inputs'
    prepared = inputs / 'combined-fit-stimulus-preparation-20260908-v1'
    assets = inputs / 'fixed-response-assets-20260908-v1'
    return ['/workspace/jepa-planning-python/bin/python', '-u', '-m', 'offline_study.fixed_combined_check',
        '--task', 'mw-reach', '--vendor', '/workspace/jepa_steering/vendor/jepa-wms',
        '--fit', str(inputs / 'fixed-response-fit-20260908-v1/reach'),
        '--coupling-fit', str(prepared / 'coupling-evidence/reach/vision_action_coupling'),
        '--checkpoint', str(assets / 'jepa_wm_metaworld.pth.tar'),
        '--checkpoint-sha256', 'c3297772c7af4e84c28bd0c0937e22398a888848305014bb3ab3f1508a310bd8',
        '--data-root', str(assets / 'metaworld/data'), '--stimulus', str(prepared / 'stimulus'),
        '--stimulus-sha256', 'b44823daf3e47ec688e185fb9f6b67b8c56a5ebcd31235284fecdb203a50028c',
        '--output', str(ROOT / 'check')]


def execute(c, environment, status):
    """Clean up only the unreaped child, while the caller still holds its lock."""
    child = None
    try:
        with (status/'child.log').open('x') as log:
            started = time.monotonic()
            child = subprocess.Popen(command(), env=environment, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            c.write(status/'SPAWNED.json', {'pid': child.pid, 'command': command()})
            binding = c.started(child.pid, command())
            c.write(status/'CHILD.json', binding)
            code = child.wait(timeout=max(0, started + LIMIT - time.monotonic()))
        c.write(status/'EXIT.json', {'identity': binding, 'returncode': code})
        if code: raise ValueError('Numerical check failed; no automatic retry')
        return binding
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=5)
            c.write(status/'CLEANUP.json', {'pid': child.pid, 'returncode': child.returncode,
                'only_owned_unreaped_child_signalled': True, 'predecessor_signals': 0})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authorization-sha256', required=True)
    args = parser.parse_args(argv)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('PYTHONOPTIMIZE') not in (None, '', '0'):
        raise ValueError('Require CUDA-hidden nonoptimized supervisor')
    if sha(ROOT/'AUTHORIZATION.json') != args.authorization_sha256:
        raise ValueError('Root authorization bytes changed')
    authority = read(ROOT/'AUTHORIZATION.json')
    validate_authority(authority, sha(Path(__file__)), sha(ROOT/'receiving-tests.log'))
    if source_digest(ROOT/'src/offline_study') != SCIENCE or sha(NAV/'FILES.json') != NAV_FILES or sha(NAV/'PLAN.json') != NAV_PLAN:
        raise ValueError('Reviewed scientific/operations source changed')
    for name, expected in read(NAV/'FILES.json').items():
        p = NAV/name
        if Path(name).name != name or p.is_symlink() or p.stat().st_size != expected['bytes'] or sha(p) != expected['sha256']:
            raise ValueError('Frozen navigation helper changed')
    sys.path.insert(0, str(NAV))
    c = importlib.import_module('navigation_redistribution_common')
    control = importlib.import_module('navigation_redistribution_control')
    if any(Path(m.__file__).resolve().parent != NAV for m in (c, control, c.lifecycle)):
        raise ValueError('Wrong operations import')
    status = ROOT/'run'; status.mkdir(exist_ok=False)
    try:
        identity = c.process(os.getpid())
        c.write(status/'LAUNCH.json', {'identity': identity, 'authorization_sha256': args.authorization_sha256,
            'source_sha256': SCIENCE, 'instance': 50259194, 'gpu': 3, 'gpu_uuid': UUID})
        proof = control.verify_completed('tx3')
        predecessor = proof['identity']
        if proof.get('verified_endpoints') != 120 or predecessor['pid'] != 3494 or predecessor['starttime'] != 41526025:
            raise ValueError('Require whole original tx3 assignment')
        if c.alive(predecessor): raise ValueError('Navigation still owns device; do not overlap')
        lock = ROOT.parent/('gpu-ownership-'+UUID+'.lock')
        if lock.is_symlink(): raise ValueError('Unsafe ownership lock')
        with lock.open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                release = c.wait_gpu_release(3, UUID, known_children=[predecessor])
                c.write(status/'PREDECESSOR_VERIFIED.json', proof)
                c.write(status/'DEVICE_RELEASE.json', release)
                validate_authority(authority, sha(Path(__file__)), sha(ROOT/'receiving-tests.log'))
                if (ROOT/'check').exists(): raise FileExistsError('Never rerun an existing check')
                environment = c.environment(50259194, 3)
                environment['PYTHONPATH'] = str(ROOT/'src') + ':/workspace/jepa-planning-python/lib/python3.10/site-packages:/workspace/jepa-python/lib/python3.10/site-packages'
                child_identity = execute(c, environment, status)
                c.wait_gpu_release(3, UUID, known_children=[child_identity])
                report, done = read(ROOT/'check/report.json'), read(ROOT/'check/DONE.json')
                if (sha(ROOT/'check/report.json') != done['report_sha256'] or
                        report['status'] != 'fit_only_native_prefix_combined_engineering_complete' or
                        report['source_and_bound_inputs_unchanged'] is not True or (ROOT/'check/FAILED.json').exists()):
                    raise ValueError('Incomplete numerical result')
                c.write(status/'DONE.json', {'report_sha256': done['report_sha256'],
                    'child_identity': child_identity, 'behavioral_launch_ready': False})
            finally: fcntl.flock(stream, fcntl.LOCK_UN)
    except BaseException as error:
        c.write(status/'FAILED.json', {'error': repr(error),
            'automatic_retry': False, 'predecessor_signals': 0, 'behavioral_launch_ready': False})
        raise


if __name__ == '__main__': main()
