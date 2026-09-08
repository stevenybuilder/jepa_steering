"""One CPU waiter for Indiana's disposable native32 Push-T GPU check.

No leases, staging, predecessor signals, retries, training histories or scientific
imports. Root stages/freezes this file and authorizes its exact PLAN separately.
PLAN schema=1: root, source_sha256, source_manifest={path,sha256}, fixed_files=
{absolute_path:{bytes,sha256, optional symlink}}, runtime={python,overlay,
ld_library_path}, batch_root, vendor, instance, gpu, gpu_uuid. Outputs are root/
queue (operations) and root/check (the one child). --prepare-only writes only an
immutable root/PREPARED.json; it never creates a queue or claims a GPU.
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path('/workspace/jepa-runtime/robotics-training-gpu-check-20260908-v1')
NAV = Path('/workspace/jepa-runtime/navigation-redistribution-20260908-v2')
INSTANCE = 50205763
GPU_UUID = 'GPU-1541ee72-c8fc-ca75-7fea-470ae793c81e'
NAV_PLAN_SHA = '378cec55b21b84cf3d91b35fe0cfbc4f4b793a0f33b4cc6a726d17f1a4a46a63'
NAV_LAUNCH_SHA = 'ae9ac3397e0a0611e375d4e2657fe429041d2367a8fab3b64f0ebc544c3a2597'
NAV_FILES_SHA = '9497a41e8d8121be87fa94337b18384da9c35558e58e795975adb0cf5f1f9f27'
NAV_SOURCE_SHA = 'fc7578672eabd8e08469afcc1f7f4d9b1c2ed18c447f4f003539b61da7950f58'
NAV_VERIFIER_SHA = '8557f9a8f276a1b4321d8ee299b494d2b10e0ef3c0310ef89d0950fa4969a43c'
BATCH = Path('/workspace/jepa-runtime/robotics-training-batch-20260908-v1/evidence')
INPUTS = {'report.json': '4b14268d293bb708ad8b96c074624a4d52c4b70ca57988c9858273a8eaa1b7aa',
          'protocol.json': 'a5975c987b5cd614de7b56e1943f3927f0365ccdd0175dd5e62c4a36fa01e0ef',
          'batch.pt': '2ada0bc241424e26899300b77da849f38c9c056ecab14295c6c99be27526bcc4',
          'DONE.json': 'ee084195f8973004c41ad733f3d086a2f0ec5d6212307e35f55db0301c6eaea1'}
VENDOR = '/workspace/jepa_steering/vendor/jepa-wms'
PYTHON = '/workspace/jepa-planning-python/bin/python'
CHILD_SECONDS = 1210
KILL_GRACE = 5
SOURCE_MEMBERS = ('robotics_training_gpu_check.py', 'robotics_training_model.py',
    'robotics_training_pilot.py', 'robotics_training_batch.py',
    'robotics_training_inputs.py', 'model_loader.py', 'vendor.py')


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    """Atomic, append-only receipt; failed publication is retained too."""
    path = Path(path)
    pending = path.with_name(path.name + '.publishing')
    with pending.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.link(pending, path)
    pending.unlink()


def require(actual, expected, label):
    if any(actual.get(k) != v for k, v in expected.items()):
        raise ValueError(label + ' binding mismatch')


def bounded_path(value):
    path = Path(value)
    if not path.is_absolute() or '..' in path.parts or path == Path('/'):
        raise ValueError('Require an explicit bounded absolute path')
    return path


def sha_string(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Require a SHA256 digest')


def validate_plan(plan):
    require(plan, {'schema': 1, 'root': str(ROOT), 'instance': INSTANCE, 'gpu': 0,
        'gpu_uuid': GPU_UUID, 'batch_root': str(BATCH), 'vendor': VENDOR}, 'Indiana plan')
    sha_string(plan['source_sha256'])
    manifest = plan['source_manifest']
    if bounded_path(manifest['path']) != ROOT / 'SOURCE_FILES.json':
        raise ValueError('Source manifest must be in the isolated root')
    sha_string(manifest['sha256'])
    runtime = plan['runtime']
    if runtime['python'] != PYTHON:
        raise ValueError('Require the verified Python3.10 runtime')
    for key in ('overlay', 'ld_library_path'):
        values = runtime[key].split(':')
        if not values:
            raise ValueError('Missing pinned runtime search paths')
        for value in values:
            bounded_path(value)
    fixed = plan['fixed_files']
    if not fixed or PYTHON not in fixed or any(str(BATCH / name) not in fixed for name in INPUTS):
        raise ValueError('Runtime binary and all original batch proof files must be pinned')
    for name, row in fixed.items():
        bounded_path(name)
        sha_string(row['sha256'])
        if type(row['bytes']) is not int or row['bytes'] < 0:
            raise ValueError('Invalid fixed file size')
    for name, pin in INPUTS.items():
        if fixed[str(BATCH / name)]['sha256'] != pin:
            raise ValueError('Original training-only batch cannot be substituted')
    return plan


def environment(plan, gpu=False):
    env = dict(os.environ)
    for key in ('LD_PRELOAD', 'PYTHONHOME', 'TORCH_HOME', 'XDG_CACHE_HOME'):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES='0' if gpu else '', PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=str(ROOT / 'src') + ':' + plan['runtime']['overlay'],
        LD_LIBRARY_PATH=plan['runtime']['ld_library_path'], JEPA_VERIFIED_LOCAL_DINO='1',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        PYTHONNOUSERSITE='1')
    return env


def verify_file(path, row):
    path = Path(path)
    if path.is_symlink():
        if row.get('symlink') != os.readlink(path):
            raise ValueError('Unbound file symlink: ' + str(path))
    elif 'symlink' in row:
        raise ValueError('Pinned file symlink was replaced')
    if not path.is_file() or path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
        raise ValueError('Pinned file changed: ' + str(path))


def source_hash(directory):
    value = hashlib.sha256()
    files = sorted(Path(directory).glob('*.py'))
    if not files:
        raise ValueError('Missing scientific source')
    for path in files:
        value.update(path.name.encode() + b'\0' + path.read_bytes())
    return value.hexdigest()


def verify_static(plan):
    """Bytes only; do not import Torch, models, loaders or mutable helpers."""
    validate_plan(plan)
    if ROOT.is_symlink() or BATCH.is_symlink():
        raise ValueError('Isolated source and original batch roots must not be symlinks')
    manifest = plan['source_manifest']
    if digest(manifest['path']) != manifest['sha256']:
        raise ValueError('Frozen source manifest changed')
    files = read(manifest['path'])
    required = {'scripts/vast/robotics_gpu_check_queue.py',
        'tests/test_robotics_gpu_check_queue.py'} | {'src/offline_study/' + x for x in SOURCE_MEMBERS}
    if not required <= files.keys():
        raise ValueError('Queue, tests and all child sources must be frozen')
    for name, row in files.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or not relative.parts:
            raise ValueError('Unsafe source manifest path')
        path = ROOT / relative
        if any(p.is_symlink() for p in path.parents if p != ROOT and ROOT in p.parents):
            raise ValueError('Source parent symlink is not allowed')
        verify_file(path, row)
    if (ROOT / 'scripts/vast/robotics_gpu_check_queue.py').resolve() != Path(__file__).resolve():
        raise ValueError('Run only the staged frozen waiter')
    if source_hash(ROOT / 'src/offline_study') != plan['source_sha256']:
        raise ValueError('Scientific source tree changed')
    for path, row in plan['fixed_files'].items():
        verify_file(path, row)
    if (BATCH / 'FAILED.json').exists():
        raise ValueError('Failed input proof is not usable')
    return {'source_sha256': plan['source_sha256'], 'source_manifest_sha256': manifest['sha256'],
            'fixed_files_verified': len(plan['fixed_files']), 'model_imports': 0, 'gpu_calls': 0}


def authorization(path, expected_sha, plan_sha, *, fresh=False, now=None):
    now = time.time() if now is None else now
    if digest(path) != expected_sha:
        raise ValueError('Root authorization changed')
    value = read(path)
    require(value, {'status': 'root_authorized_native32_gpu_check', 'owner': 'rep_geometry_transcoder/root',
        'plan_sha256': plan_sha, 'instance': INSTANCE, 'gpu': 0, 'gpu_uuid': GPU_UUID,
        'child_timeout_seconds': CHILD_SECONDS, 'kill_grace_seconds': KILL_GRACE,
        'full_history_authorized': False}, 'Root authorization')
    issued, expires = value['issued_unix'], value['expires_unix']
    if (type(issued) not in (float, int) or type(expires) not in (float, int) or
            not math.isfinite(issued) or not math.isfinite(expires) or
            not issued <= now < expires or not 0 < expires - issued <= 12 * 3600 or
            (fresh and now - issued > 300)):
        raise ValueError('Authorization is stale, not yet valid or beyond the twelve-hour limit')
    return value


def predecessor():
    return {'pid': 10073, 'starttime': 17416272, 'command': ['/usr/bin/python3', '-u',
        str(NAV / 'navigation_redistribution_control.py'), 'run', '--worker', 'in0']}


def process(pid, proc=Path('/proc')):
    root = proc / str(pid)
    try:
        first = (root / 'stat').read_text().rsplit(') ', 1)[1].split()
        command = [x.decode() for x in (root / 'cmdline').read_bytes().split(b'\0') if x]
        last = (root / 'stat').read_text().rsplit(') ', 1)[1].split()
    except FileNotFoundError:
        return None
    if first[19] != last[19]:
        raise ValueError('PID reused during process read')
    return {'pid': pid, 'starttime': int(last[19]), 'state': last[0], 'command': command}


def same_live(binding, current):
    if current is None:
        return False
    if current['starttime'] != binding['starttime']:
        raise ValueError('Process PID reused; do not use its device')
    if current['state'] in ('Z', 'X'):
        return False
    if current['command'] != binding['command']:
        raise ValueError('Process command changed')
    return True


def predecessor_pending(reader=None):
    reader = process if reader is None else reader
    failed, done = NAV / 'workers/in0/FAILED.json', NAV / 'workers/in0/DONE.json'
    if failed.exists():
        raise ValueError('Navigation failed; never bypass it')
    if done.exists():
        return False
    current = reader(predecessor()['pid'])
    # Publication may precede normal exit between either of these reads.
    if failed.exists():
        raise ValueError('Navigation failed during handoff')
    if done.exists():
        return False
    if not same_live(predecessor(), current):
        raise ValueError('Navigation exited without full-assignment completion')
    return True


def verify_navigation_static():
    """Rehash every operations dependency before invoking its frozen CLI."""
    for name, expected in [('PLAN.json', NAV_PLAN_SHA), ('FILES.json', NAV_FILES_SHA),
            ('workers/in0/LAUNCH.json', NAV_LAUNCH_SHA)]:
        if digest(NAV / name) != expected:
            raise ValueError('Original navigation binding changed: ' + name)
    for name, row in read(NAV / 'FILES.json').items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe navigation source manifest')
        verify_file(NAV / relative, row)
    if digest(NAV / 'navigation_redistribution_control.py') != NAV_VERIFIER_SHA:
        raise ValueError('Navigation completion verifier changed')
    if source_hash(NAV / 'payload/code/src/offline_study') != NAV_SOURCE_SHA:
        raise ValueError('Frozen navigation scientific source changed')
    launch = read(NAV / 'workers/in0/LAUNCH.json')
    require(launch['identity'], predecessor(), 'Original navigation process')
    require(launch, {'worker': 'in0', 'plan_sha256': NAV_PLAN_SHA,
        'instance': INSTANCE, 'gpu': 0, 'gpu_uuid': GPU_UUID}, 'Original navigation launch')
    assignment = read(NAV / 'PLAN.json')['workers']['in0']
    if len(assignment['jobs']) != 15 or launch['jobs'] != assignment['jobs']:
        raise ValueError('Require all fifteen original intact streams')
    return launch


def verify_predecessor(plan):
    launch = verify_navigation_static()
    status = NAV / 'workers/in0'
    if (status / 'FAILED.json').exists():
        raise ValueError('Failed navigation assignment')
    done = read(status / 'DONE.json')
    require(done, {'status': 'navigation_redistribution_assignment_complete', 'worker': 'in0',
        'plan_sha256': NAV_PLAN_SHA, 'launch_sha256': NAV_LAUNCH_SHA,
        'identity': launch['identity'], 'endpoints': 180}, 'Whole navigation completion')
    result = json.loads(subprocess.check_output(['/usr/bin/python3',
        str(NAV / 'navigation_redistribution_control.py'), 'verify-completed', '--worker', 'in0'],
        env=environment(plan), text=True, timeout=900))
    require(result, {'worker': 'in0', 'instance': INSTANCE, 'gpu': 0, 'gpu_uuid': GPU_UUID,
        'plan_sha256': NAV_PLAN_SHA, 'identity': launch['identity'], 'verified_endpoints': 180,
        'scientific_validators_unchanged': True, 'done_sha256': digest(status / 'DONE.json')},
        'Frozen whole-assignment scientific verification')
    if (status / 'FAILED.json').exists():
        raise ValueError('Navigation failed while verifying completion')
    return result


def gpu_snapshot():
    uuid = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=uuid',
        '--format=csv,noheader'], text=True, timeout=15).strip()
    raw = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-compute-apps=pid,gpu_uuid',
        '--format=csv,noheader'], text=True, timeout=15)
    rows = []
    for line in raw.splitlines():
        values = [x.strip() for x in line.split(',')]
        if len(values) != 2 or not values[0].isdigit() or values[1] != GPU_UUID:
            raise ValueError('Unrecognized NVML ownership row')
        rows.append(int(values[0]))
    return uuid, rows


def wait_release(*, snapshot=None, reader=None, clock=None, sleeper=None, timeout=120):
    snapshot, reader = snapshot or gpu_snapshot, reader or process
    clock, sleeper = clock or time.monotonic, sleeper or time.sleep
    started, stable, observations = clock(), 0, []
    while True:
        if (NAV / 'workers/in0/FAILED.json').exists():
            raise ValueError('Navigation failed during device release')
        parent = reader(predecessor()['pid'])
        terminal = not same_live(predecessor(), parent)
        uuid, pids = snapshot()
        if uuid != GPU_UUID:
            raise ValueError('Physical GPU changed')
        for pid in pids:
            current = reader(pid)
            if current is not None and current['state'] not in ('Z', 'X'):
                raise ValueError('GPU still has live work; never overlap or signal it')
        stable = stable + 1 if terminal and not pids else 0
        observations.append({'elapsed_seconds': clock() - started,
            'predecessor_terminal': terminal, 'nvml_pids': pids, 'stable_empty_samples': stable})
        if stable >= 2:
            return {'gpu_uuid': uuid, 'stable_empty_samples': stable, 'observations': observations}
        if clock() - started >= timeout:
            raise TimeoutError('Predecessor/context teardown did not complete within its bound')
        sleeper(0.5)


@contextlib.contextmanager
def device_lock(path=None):
    path = Path(path) if path is not None else Path('/workspace/jepa-runtime') / ('gpu-ownership-' + GPU_UUID + '.lock')
    if path.is_symlink():
        raise ValueError('Shared device lock must not be a symlink')
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def child_command(plan):
    return [plan['runtime']['python'], '-u', '-m', 'offline_study.robotics_training_gpu_check',
        '--vendor', VENDOR, '--input-proof', str(BATCH), '--output', str(ROOT / 'check'),
        '--expected-device-uuid', GPU_UUID]


def stop_owned_child(child):
    """Only this unreaped Popen child; never a predecessor PID or process group."""
    if child.poll() is not None:
        return []
    actions = ['terminate_owned_popen']
    child.terminate()
    try:
        child.wait(timeout=KILL_GRACE)
    except subprocess.TimeoutExpired:
        child.kill()
        actions.append('kill_owned_popen_after_five_seconds')
        child.wait(timeout=KILL_GRACE)
    return actions


def execute_child(plan, queue):
    command, child = child_command(plan), None
    try:
        with (queue / 'child.log').open('x') as log:
            started = time.monotonic()
            child = subprocess.Popen(command, env=environment(plan, gpu=True), stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write(queue / 'CHILD_SPAWNED.json', {'pid': child.pid, 'command': command,
                'identity_pending': True, 'timeout_seconds': CHILD_SECONDS})
            deadline = time.monotonic() + 5
            binding = None
            while time.monotonic() < deadline:
                current = process(child.pid)
                if current is None or current['state'] in ('Z', 'X'):
                    raise ValueError('Owned GPU child exited before verified identity')
                if binding is not None and current['starttime'] != binding['starttime']:
                    raise ValueError('Owned GPU child identity changed during launch')
                binding = current
                if current['command'] == command:
                    break
                time.sleep(0.01)
            else:
                raise TimeoutError('Owned child never reached its exact frozen command')
            write(queue / 'CHILD_LAUNCH.json', {**binding, 'gpu_uuid': GPU_UUID})
            code = child.wait(timeout=max(0, started + CHILD_SECONDS - time.monotonic()))
            write(queue / 'CHILD_EXIT.json', {'pid': child.pid, 'returncode': code,
                'launch_sha256': digest(queue / 'CHILD_LAUNCH.json')})
            if code:
                raise ValueError('Disposable GPU check failed; no automatic retry')
    except BaseException:
        if child is not None:
            previous = {s: signal.signal(s, signal.SIG_IGN) for s in (signal.SIGTERM, signal.SIGINT)}
            try:
                actions = stop_owned_child(child)
                write(queue / 'CHILD_CLEANUP.json', {'pid': child.pid, 'actions': actions,
                    'returncode': child.returncode, 'predecessor_signals': 0})
            finally:
                for signum, handler in previous.items():
                    signal.signal(signum, handler)
        raise


def verify_child_result(plan):
    root = ROOT / 'check'
    if (root / 'FAILED.json').exists():
        raise ValueError('Failed disposable check cannot pass')
    report, protocol = read(root / 'report.json'), read(root / 'protocol.json')
    require(read(root / 'DONE.json'), {'report_sha256': digest(root / 'report.json')}, 'Child DONE')
    require(report, {'status': 'native32_pusht_disposable_gpu_arithmetic_check_complete',
        'protocol_sha256': digest(root / 'protocol.json'), 'engineering_only': True,
        'training_history_updates': 0, 'history_launch_authorized': False,
        'validation_or_confirmation_access': False,
        'live_model_rng_optimizer_scheduler_unchanged_by_verification': True,
        'native_loader_lpips_rng_or_resume_verified': False,
        'physical_ddp_allreduce_order_reproduced': False}, 'Disposable check report')
    require(protocol, {'task': 'pusht', 'model_seed': 234, 'timed_updates': 4,
        'input_proof': str(BATCH), 'vendor': VENDOR, 'expected_device_uuid': GPU_UUID,
        'logical_ranks': 32, 'microbatch': 8, 'global_batch': 256,
        'numerical_tolerance': 'bitwise_no_relaxation', 'engineering_only': True,
        'validation_or_confirmation_access': False, 'history_launch_authorized': False,
        'source_sha256': {x: digest(ROOT / 'src/offline_study' / x) for x in SOURCE_MEMBERS}}, 'Child protocol')
    if report['device']['device_uuid'].removeprefix('GPU-') != GPU_UUID.removeprefix('GPU-'):
        raise ValueError('Child used another physical device')
    require(report['inputs'], {'root': str(BATCH),
        'files_sha256': {k: v for k, v in INPUTS.items() if k != 'DONE.json'}}, 'Child inputs')
    if [x['update'] for x in report['timed_engineering_updates']] != [1, 2, 3, 4]:
        raise ValueError('Missing or repeated disposable update')
    calls = report['actual_call_timings']
    if len(calls) != 6 or any(x.get('completed') is not True for x in calls):
        raise ValueError('Require both parity calls and all four timed updates')
    for name in ('constructor', 'numerical_proof', 'clone_state'):
        if digest(root / (name + '.json')) != report[name + '_sha256']:
            raise ValueError('Child raw proof changed')
    if any(root.rglob('*.pth*')) or any(root.rglob('*.pt')):
        raise ValueError('Disposable check unexpectedly wrote model checkpoints')
    return {'status': report['status'], 'report_sha256': digest(root / 'report.json'),
        'protocol_sha256': digest(root / 'protocol.json'), 'timed_engineering_updates': 4,
        'training_history_updates': 0, 'history_launch_authorized': False}


def run(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU waiter must start with CUDA explicitly hidden')
    if digest(args.plan) != args.plan_sha256:
        raise ValueError('Root-reviewed plan changed')
    plan = validate_plan(read(args.plan))
    auth = lambda fresh=False: authorization(args.authorization, args.authorization_sha256,
        args.plan_sha256, fresh=fresh)
    auth(True)
    if args.prepare_only:
        static = verify_static(plan)
        verify_navigation_static()
        prepared = {'status': 'cpu_preparation_verified_no_queue_or_gpu_launch',
            **static, 'plan_sha256': args.plan_sha256,
            'authorization_sha256': args.authorization_sha256,
            'navigation_launch_sha256': NAV_LAUNCH_SHA}
        write(ROOT / 'PREPARED.json', prepared)
        return prepared
    require(read(ROOT / 'PREPARED.json'), {
        'status': 'cpu_preparation_verified_no_queue_or_gpu_launch',
        'plan_sha256': args.plan_sha256, 'authorization_sha256': args.authorization_sha256,
        'source_sha256': plan['source_sha256'],
        'source_manifest_sha256': plan['source_manifest']['sha256'],
        'navigation_launch_sha256': NAV_LAUNCH_SHA, 'model_imports': 0, 'gpu_calls': 0},
        'Receiving preparation')
    queue = ROOT / 'queue'
    if (ROOT / 'check').exists():
        raise FileExistsError('Preserve prior disposable GPU check; no retry')
    queue.mkdir(exist_ok=False)
    try:
        write(queue / 'LAUNCH.json', {'identity': process(os.getpid()), 'plan_sha256': args.plan_sha256,
            'authorization_sha256': args.authorization_sha256, 'source_manifest_sha256': plan['source_manifest']['sha256'],
            'instance': INSTANCE, 'gpu_uuid': GPU_UUID, 'gpu_work_started': False})
        write(queue / 'PREPARED.json', verify_static(plan))
        verify_navigation_static()
        while predecessor_pending():
            auth()
            time.sleep(30)
        auth()
        verified = verify_predecessor(plan)
        write(queue / 'PREDECESSOR_VERIFIED.json', verified)
        with device_lock():
            write(queue / 'GPU_RELEASE.json', wait_release())
            verify_static(plan)
            auth()
            # Recheck after the input/source rehash, immediately before Popen.
            write(queue / 'FINAL_GPU_RELEASE.json', wait_release())
            if digest(NAV / 'workers/in0/DONE.json') != verified['done_sha256']:
                raise ValueError('Navigation completion changed before GPU child launch')
            if auth()['expires_unix'] - time.time() < CHILD_SECONDS + 2 * KILL_GRACE:
                raise ValueError('Insufficient authorization lifetime for the bounded child and cleanup')
            execute_child(plan, queue)
            verify_static(plan)
            result = verify_child_result(plan)
            write(queue / 'DONE.json', {**result, 'plan_sha256': args.plan_sha256,
                'child_exit_sha256': digest(queue / 'CHILD_EXIT.json'),
                'predecessor_verified_sha256': digest(queue / 'PREDECESSOR_VERIFIED.json'),
                'predecessor_signals': 0, 'automatic_retry': False,
                'gpu_available_only_after_process_exit': True})
            return result
    except BaseException as error:
        write(queue / 'FAILED.json', {'error_type': type(error).__name__, 'error': str(error),
            'plan_sha256': args.plan_sha256, 'automatic_retry': False,
            'predecessor_signals': 0, 'history_launch_authorized': False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'authorization'):
        parser.add_argument('--' + name, type=Path, required=True)
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    def cancelled(signum, frame):
        raise RuntimeError('CPU queue cancelled; preserve attempt and never signal navigation')
    signal.signal(signal.SIGTERM, cancelled)
    signal.signal(signal.SIGINT, cancelled)
    print(json.dumps(run(args)))


if __name__ == '__main__':
    main()
