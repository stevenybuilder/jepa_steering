"""Explicit, one-shot Indiana outage recovery; never resumes inside an RNG stream.

This operations file imports only the hash-verified original navigation helpers.
The old PLAN, LAUNCH, eight completed streams and interrupted tree stay untouched.
Root separately authorizes run after CPU preparation. No provider/staging actions.
"""
import argparse
import ast
import contextlib
import copy
import fcntl
import hashlib
import importlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time

NAV = Path('/workspace/jepa-runtime/navigation-redistribution-20260908-v2')
ROOT = Path('/workspace/jepa-runtime/navigation-recovery-in0-20260908-v1')
GPU_CHECK = Path('/workspace/jepa-runtime/robotics-training-gpu-check-20260908-v1')
INSTANCE, GPU = 50205763, 0
UUID = 'GPU-1541ee72-c8fc-ca75-7fea-470ae793c81e'
PLAN_SHA = '378cec55b21b84cf3d91b35fe0cfbc4f4b793a0f33b4cc6a726d17f1a4a46a63'
LAUNCH_SHA = 'ae9ac3397e0a0611e375d4e2657fe429041d2367a8fab3b64f0ebc544c3a2597'
FILES_SHA = '9497a41e8d8121be87fa94337b18384da9c35558e58e795975adb0cf5f1f9f27'
SECONDS = 7200
EXPECTED = ([{'task': 'wall', 'arm': 'permuted_visual', 'rank': r} for r in range(4, 8)] +
    [{'task': 'wall', 'arm': 'permuted_joint', 'rank': r} for r in range(3)] +
    [{'task': 'pointmaze', 'arm': 'permuted_visual', 'rank': r} for r in range(3, 8)] +
    [{'task': 'pointmaze', 'arm': 'permuted_joint', 'rank': r} for r in range(3)])


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    """Publish atomically without replacing any prior receipt, even on failure."""
    path = Path(path)
    pending = path.with_name(path.name + '.publishing')
    with pending.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.link(pending, path)
    pending.unlink()


def require(value, expected, label):
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError(label + ' binding differs')


def descriptor(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Require regular evidence file: ' + str(path))
    return {'bytes': path.stat().st_size, 'sha256': digest(path)}


def tree(path):
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError('Require unchanged regular evidence tree')
    result = {}
    for file in sorted(path.rglob('*')):
        if file.is_symlink():
            raise ValueError('Evidence symlink refused')
        if file.is_file():
            result[str(file.relative_to(path))] = descriptor(file)
    return result


def frozen_ops():
    if digest(NAV / 'FILES.json') != FILES_SHA:
        raise ValueError('Original operations manifest differs')
    for name, row in read(NAV / 'FILES.json').items():
        if Path(name).name != name or descriptor(NAV / name) != row:
            raise ValueError('Original operations source differs')
    sys.path.insert(0, str(NAV))
    c = importlib.import_module('navigation_redistribution_common')
    control = importlib.import_module('navigation_redistribution_control')
    for module in (c, control, c.lifecycle):
        if Path(module.__file__).resolve().parent != NAV.resolve():
            raise ValueError('Do not import mutable/local operations helpers')
    return c, control


def namespace():
    fields = Path('/proc/1/stat').read_text().rsplit(') ', 1)[1].split()
    return {'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'pid1_starttime': int(fields[19])}


def verify_runtime(path, expected_sha, c, *, live=True):
    if digest(path) != expected_sha:
        raise ValueError('Root runtime evidence changed')
    value = read(path)
    require(value, {'status': 'same_device_post_outage_runtime_verified',
        'owner': 'rep_geometry_transcoder/root', 'instance': INSTANCE, 'gpu': GPU,
        'gpu_uuid': UUID, 'original_launch_sha256': LAUNCH_SHA,
        'source_sha256': c.SOURCE, 'freeze_sha256': c.FREEZES,
        'pre_outage_payload_hashes_reverified': True, 'historical_host_libraries_bound': False}, 'Runtime')
    if (not isinstance(value.get('unbound_historical_runtime_paths'), list) or
            not value['unbound_historical_runtime_paths'] or
            value.get('input_manifest_sha256') != digest(NAV / 'INPUTS.json')):
        raise ValueError('Disclose unbound historical host libraries and bind verified payload')
    if not isinstance(value.get('files'), dict) or c.PYTHON not in value['files']:
        raise ValueError('Runtime files and original Python must be bound')
    if any(not isinstance(name, str) or name not in value['files']
           for name in value['unbound_historical_runtime_paths']):
        raise ValueError('Historically unbound host libraries must be hash-bound now')
    for name, row in value['files'].items():
        path = Path(name)
        if not path.is_absolute() or '..' in path.parts or not path.is_file():
            raise ValueError('Invalid bound runtime path')
        if path.is_symlink() != ('symlink' in row) or (path.is_symlink() and os.readlink(path) != row['symlink']):
            raise ValueError('Runtime link changed')
        if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise ValueError('Runtime bytes changed')
    if live and value['namespace'] != namespace():
        raise ValueError('Container/process namespace changed after runtime verification')
    return value


def original(c):
    if digest(NAV / 'PLAN.json') != PLAN_SHA or digest(NAV / 'workers/in0/LAUNCH.json') != LAUNCH_SHA:
        raise ValueError('Original assignment/identity changed')
    plan, launch = read(NAV / 'PLAN.json'), read(NAV / 'workers/in0/LAUNCH.json')
    c.validate_plan(plan)
    worker = plan['workers']['in0']
    require(worker, {'instance': INSTANCE, 'gpu': GPU, 'gpu_uuid': UUID, 'jobs': EXPECTED}, 'Original worker')
    require(launch, {**worker, 'worker': 'in0', 'plan_sha256': PLAN_SHA}, 'Original launch')
    if (NAV / 'workers/in0/DONE.json').exists():
        raise ValueError('Original worker completed; outage recovery is unnecessary')
    return launch


def verify_sources(c):
    c.verify_members(NAV, read(NAV / 'FILES.json'))
    c.verify_members(NAV, read(NAV / 'INPUTS.json'))
    c.verify_source()


def stopped(c, launch):
    """No signals; a reused identity is an error, not permission to take its GPU."""
    bindings = [launch['identity']]
    waiter = read(GPU_CHECK / 'ACTIVATED.json')
    bindings.append({'pid': waiter['pid'], 'starttime': waiter['start_ticks'], 'command': waiter['argv']})
    for binding in bindings:
        if c.alive(binding):
            raise ValueError('Original navigation/check waiter still alive')
    return bindings


def scientific_verify(c, control, jobs, proofs):
    """Execute the byte-pinned original CPU verifier body on explicit result paths."""
    body = ast.parse(inspect.getsource(control.verify_completed))
    program = next(ast.literal_eval(n.value) for n in ast.walk(body)
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'program' for t in n.targets))
    result = control.python_check(INSTANCE, program, json.dumps(jobs), json.dumps(proofs),
        json.dumps([c.arguments(t) for t in c.TASKS]), c.FREEZES['wall'], c.FREEZES['pointmaze'])
    require(result, {'verified_endpoints': len(jobs) * 12, 'scientific_validators_unchanged': True}, 'Scientific union')
    return result


def inventory(c, control):
    completed, proofs = [], {}
    for task in c.TASKS:
        proof = read(NAV / 'workers/in0' / ('ENGINEERING-' + task + '.json'))
        if proof['gpu_uuid'] != UUID or control.engineering_proof(INSTANCE, task, proof['path']) != proof['report_sha256']:
            raise ValueError('Original same-device coupling engineering changed')
        proofs[task] = proof
    for job in EXPECTED[:8]:
        path = NAV / 'results' / c.relative(job)
        completed.append({**job, 'output': str(path),
            'report_sha256': c.verify_shard(path, job, proofs[job['task']]['report_sha256']),
            'members': tree(path)})
    interrupted = NAV / 'results' / c.relative(EXPECTED[8])
    if (interrupted / 'DONE.json').exists() or (interrupted / 'report.json').exists():
        raise ValueError('Interrupted stream is actually complete/ambiguous')
    partial = tree(interrupted)
    expected_name = 'episode-048.json'
    if {n for n in partial if n.startswith('episode-')} != {expected_name}:
        raise ValueError('Outage scope changed: require the one preserved first episode')
    launch = read(interrupted / 'protocol.json')
    require(launch, {'task': 'pointmaze', 'arm': 'permuted_visual', 'logical_ranks': [4],
        'source_sha256': c.SOURCE, 'freeze_sha256': c.FREEZES['pointmaze'],
        'engineering_report_sha256': proofs['pointmaze']['report_sha256'],
        'engineering_only': False, 'fresh_confirmation': False,
        'expected_episodes': c.expected_rows(4)}, 'Interrupted protocol')
    require(read(interrupted / expected_name), {**c.expected_rows(4)[0], 'arm': 'permuted_visual'}, 'Interrupted episode')
    for job in EXPECTED[9:]:
        if (NAV / 'results' / c.relative(job)).exists():
            raise ValueError('An expected unstarted stream now exists')
    scientific_verify(c, control, completed, proofs)
    return {'completed': completed, 'engineering': proofs,
        'interrupted': {'job': EXPECTED[8], 'output': str(interrupted), 'members': partial,
                        'completed_prefix_files': [expected_name]}, 'unstarted': EXPECTED[9:]}


def prepare(runtime, runtime_sha):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or ROOT.is_symlink():
        raise ValueError('CPU-hidden preparation in isolated root required')
    if (ROOT / 'PLAN.json').exists() or (ROOT / 'run').exists():
        raise FileExistsError('Preserve existing attempt; no preparation retry')
    c, control = frozen_ops()
    launch = original(c)
    verify_sources(c)
    runtime = Path(runtime)
    verify_runtime(runtime, runtime_sha, c)
    dead = stopped(c, launch)
    state = inventory(c, control)
    plan = {'schema': 1, 'status': 'outage_recovery_prepared_not_activated',
        'root': str(ROOT), 'instance': INSTANCE, 'gpu': GPU, 'gpu_uuid': UUID,
        'original_plan_sha256': PLAN_SHA, 'original_launch_sha256': LAUNCH_SHA,
        'source_sha256': c.SOURCE, 'freeze_sha256': c.FREEZES,
        'operations_sha256': digest(Path(__file__)), 'runtime': {'path': str(runtime), 'sha256': runtime_sha},
        'namespace': namespace(), 'original_processes': dead,
        'original_waiter_launch': descriptor(GPU_CHECK / 'ACTIVATED.json'),
        'original_worker_tree': tree(NAV / 'workers/in0'), **state,
        'recovered': [{**j, 'output': str(ROOT / 'results' / c.relative(j))} for j in EXPECTED[8:]],
        'expected_endpoints': 180, 'reused_streams': 8, 'recovered_streams': 7,
        'partial_prefix_excluded_from_analysis': True, 'whole_stream_restart_only': True,
        'automatic_retry': False, 'new_episode_identities': 0}
    write(ROOT / 'PLAN.json', plan)
    return {'plan_sha256': digest(ROOT / 'PLAN.json'), 'gpu_calls': 0,
            'reused_streams': 8, 'recovered_streams': 7, 'verified_original_endpoints': 96}


def verify_plan(plan, c, *, live=True):
    require(plan, {'schema': 1, 'root': str(ROOT), 'instance': INSTANCE, 'gpu': GPU, 'gpu_uuid': UUID,
        'original_plan_sha256': PLAN_SHA, 'original_launch_sha256': LAUNCH_SHA,
        'source_sha256': c.SOURCE, 'freeze_sha256': c.FREEZES,
        'operations_sha256': digest(Path(__file__)), 'expected_endpoints': 180,
        'reused_streams': 8, 'recovered_streams': 7, 'partial_prefix_excluded_from_analysis': True,
        'whole_stream_restart_only': True, 'automatic_retry': False, 'new_episode_identities': 0}, 'Recovery plan')
    original(c)
    keys = ('task', 'arm', 'rank')
    all_jobs = plan['completed'] + plan['recovered']
    if [{k: row[k] for k in keys} for row in all_jobs] != EXPECTED:
        raise ValueError('Recovery assignment grew, overlapped or changed order')
    for index, row in enumerate(all_jobs):
        base = NAV if index < 8 else ROOT
        if row['output'] != str(base / 'results' / c.relative(EXPECTED[index])):
            raise ValueError('Recovery output path changed')
    if plan['unstarted'] != EXPECTED[9:] or plan['interrupted']['job'] != EXPECTED[8]:
        raise ValueError('Interrupted/unstarted identities changed')
    if plan['interrupted']['completed_prefix_files'] != ['episode-048.json']:
        raise ValueError('Interrupted prefix verification cannot be omitted')
    if plan['interrupted']['output'] != str(NAV / 'results' / c.relative(EXPECTED[8])):
        raise ValueError('Partial preservation path changed')
    if tree(plan['interrupted']['output']) != plan['interrupted']['members']:
        raise ValueError('Original interrupted evidence changed')
    if tree(NAV / 'workers/in0') != plan['original_worker_tree']:
        raise ValueError('Original worker evidence changed')
    if descriptor(GPU_CHECK / 'ACTIVATED.json') != plan['original_waiter_launch']:
        raise ValueError('Original CPU waiter launch changed')
    for row in plan['completed']:
        if tree(row['output']) != row['members']:
            raise ValueError('Original completed evidence changed')
    for job in EXPECTED[9:]:
        if (NAV / 'results' / c.relative(job)).exists():
            raise ValueError('Original parent restarted an unstarted stream')
    verify_runtime(plan['runtime']['path'], plan['runtime']['sha256'], c, live=live)
    if live and namespace() != plan['namespace']:
        raise ValueError('Prepared recovery container restarted')


def scientific_bytes(record):
    """All persisted scientific fields exact; only two explicit timing paths omitted."""
    value = copy.deepcopy(record)
    value.pop('seconds', None)
    for call in value['result']['planning_calls']:
        call.pop('seconds', None)
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def compare_prefix(plan):
    old = Path(plan['interrupted']['output'])
    new = Path(plan['recovered'][0]['output'])
    records = []
    for name in plan['interrupted']['completed_prefix_files']:
        a, b = old / name, new / name
        if scientific_bytes(read(a)) != scientific_bytes(read(b)):
            raise ValueError('Interrupted prefix differs scientifically; stop, never select or retry')
        records.append({'file': name, 'original_sha256': digest(a), 'recovered_sha256': digest(b),
            'raw_bytes_equal': a.read_bytes() == b.read_bytes(), 'scientific_bytes_equal': True})
    return {'records': records, 'excluded_timing_paths': ['seconds', 'result.planning_calls[*].seconds'],
            'partial_records_in_analysis': 0}


def native_command(c):
    return [c.PYTHON, '-u', '-m', 'offline_study.navigation_smoke', '--vendor',
        '/workspace/jepa_steering/vendor/jepa-wms', '--task', 'pointmaze', '--checkpoint',
        str(c.PAYLOAD / 'models/jepa_wm_pointmaze.pth.tar'), '--output', str(ROOT / 'native-repeat')]


def verify_native_repeat(c, control):
    # Existing native CLI does two complete excluded-seed episodes. Reuse its
    # original comparison/validation code, then require equality to the old proof.
    program = '''import json,pathlib,sys
from offline_study.navigation_smoke import comparison_record,validate_complete
from offline_study.behavioral_development import verified_report
from offline_study.protocol import sha256
new,old=map(pathlib.Path,sys.argv[1:3]); report,h=verified_report(new); prior,ph=verified_report(old)
protocol=json.loads((new/'protocol.json').read_text())
assert report['status']=='full_native_navigation_engineering_passed' and report['task']=='pointmaze'
assert report['parameters_unchanged'] and report['same_seed_actions_and_outcomes_exact'] and report['full_native_cem_schedule']
assert report['scientific_efficacy_measurement'] is False and report['protected_outcomes_accessed'] is False
assert protocol['source_sha256']==sys.argv[3] and protocol['repetitions']==2
old_protocol=json.loads((old/'protocol.json').read_text())
assert protocol['task']=='pointmaze' and protocol['planning_contract']==old_protocol['planning_contract']
assert protocol['checkpoint_sha256']==old_protocol['checkpoint_sha256']
assert len(report['repetition_sha256'])==len(prior['repetition_sha256'])==2
old_record=json.loads((old/'repetition-0.json').read_text())
for i in range(2):
 p=new/f'repetition-{i}.json'; op=old/f'repetition-{i}.json'
 assert sha256(p)==report['repetition_sha256'][i] and sha256(op)==prior['repetition_sha256'][i]
 row=json.loads(p.read_text()); validate_complete(row['result'],[tuple(v) for v in row['unroll_calls']])
 a=comparison_record(row['result'],row['action_sha256']); b=comparison_record(old_record['result'],old_record['action_sha256'])
 assert json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False)
print(json.dumps({'status':'post_outage_native_repeat_exact','report_sha256':h,'original_report_sha256':ph,'full_native_episodes':2,'same_actions_and_scientific_values':True}))
'''
    old = c.EVIDENCE / 'navigation-coupling-reference-20260908-v1/pointmaze/smoke'
    if (ROOT / 'native-repeat/FAILED.json').exists():
        raise ValueError('Post-outage native repeat failed')
    return control.python_check(INSTANCE, program, ROOT / 'native-repeat', old, c.SOURCE)


def command(c, job, proof):
    return [c.PYTHON, '-u', '-m', 'offline_study.navigation_coupling_behavior', 'run'] + c.arguments(job['task']) + [
        '--engineering', proof['path'], '--arm', job['arm'], '--logical-ranks', str(job['rank']), '--output', job['output']]


def execute(c, argv, label, status):
    """Finite bound; signals can affect only our unreaped Popen child, never originals."""
    child = None
    try:
        with (status / (label + '.log')).open('x') as log:
            started = time.monotonic()
            child = subprocess.Popen(argv, env=c.environment(INSTANCE, GPU), stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write(status / (label + '-SPAWNED.json'), {'pid': child.pid, 'command': argv, 'identity_pending': True})
            binding = c.started(child.pid, argv)
            write(status / (label + '-CHILD.json'), binding)
            code = child.wait(timeout=max(0, started + SECONDS - time.monotonic()))
            write(status / (label + '-EXIT.json'), {'identity': binding, 'returncode': code,
                'child_sha256': digest(status / (label + '-CHILD.json'))})
            if code:
                raise ValueError('Frozen child failed; no automatic retry')
            c.wait_gpu_release(GPU, UUID, [binding])
            return binding
    except BaseException:
        if child is not None and child.poll() is None:
            previous = {s: signal.signal(s, signal.SIG_IGN) for s in (signal.SIGTERM, signal.SIGINT)}
            actions = []
            try:
                child.terminate(); actions.append('terminate_owned_child')
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill(); actions.append('kill_owned_child_after_five_seconds'); child.wait(timeout=5)
            finally:
                write(status / (label + '-CLEANUP.json'), {'pid': child.pid, 'actions': actions,
                    'returncode': child.returncode, 'original_process_signals': 0})
                for s, handler in previous.items(): signal.signal(s, handler)
        raise


@contextlib.contextmanager
def device_lock():
    path = ROOT.parent / ('gpu-ownership-' + UUID + '.lock')
    if path.is_symlink(): raise ValueError('Device lock symlink refused')
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try: yield
        finally: fcntl.flock(stream, fcntl.LOCK_UN)


def authority(args, plan, *, required_seconds=0):
    if digest(args.authorization) != args.authorization_sha256:
        raise ValueError('Root authorization changed')
    value = read(args.authorization)
    require(value, {'owner': 'rep_geometry_transcoder/root', 'operation': 'explicit_navigation_outage_recovery',
        'plan_sha256': args.plan_sha256, 'operations_sha256': plan['operations_sha256'],
        'runtime_sha256': plan['runtime']['sha256'], 'instance': INSTANCE, 'gpu': GPU, 'gpu_uuid': UUID,
        'original_plan_sha256': PLAN_SHA, 'original_launch_sha256': LAUNCH_SHA,
        'whole_stream_restarts': EXPECTED[8:9], 'unstarted_streams': EXPECTED[9:],
        'maximum_child_seconds': SECONDS, 'native_repeat_episodes': 2,
        'automatic_retry': False, 'new_episode_identities': 0}, 'Explicit recovery authorization')
    issued, expires = value['issued_unix'], value['expires_unix']
    if (type(issued) not in (float, int) or type(expires) not in (float, int) or
            not math.isfinite(issued) or not math.isfinite(expires) or
            not issued <= time.time() < expires or not 0 < expires - issued <= 12 * 3600):
        raise ValueError('Recovery authority expired or unbounded')
    if required_seconds < 0 or time.time() + required_seconds > expires:
        raise ValueError('Recovery authority cannot cover the complete child and cleanup')


def run(args):
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or digest(args.plan) != args.plan_sha256 or
            Path(args.plan) != ROOT / 'PLAN.json' or Path(args.authorization) != ROOT / 'AUTHORIZATION.json' or
            os.environ.get('PYTHONOPTIMIZE') not in (None, '', '0')):
        raise ValueError('CPU-hidden supervisor and exact plan required')
    c, control = frozen_ops(); plan = read(args.plan)
    verify_plan(plan, c); verify_sources(c); authority(args, plan)
    stopped(c, original(c))
    status = ROOT / 'run'; status.mkdir(exist_ok=False)
    try:
        identity = c.process(os.getpid())
        write(status / 'LAUNCH.json', {'status': 'explicit_navigation_outage_recovery',
            'identity': identity, 'plan_sha256': args.plan_sha256,
            'authorization_sha256': args.authorization_sha256, 'namespace': namespace(),
            'original_launch_sha256': LAUNCH_SHA, 'instance': INSTANCE, 'gpu': GPU, 'gpu_uuid': UUID})
        with device_lock():
            write(status / 'GPU_RELEASE.json', c.wait_gpu_release(GPU, UUID))
            authority(args, plan, required_seconds=SECONDS + 5)
            execute(c, native_command(c), 'native-repeat', status)
            write(status / 'NATIVE_REPEAT_VERIFIED.json', verify_native_repeat(c, control))
            for index, job in enumerate(plan['recovered']):
                verify_plan(plan, c); verify_sources(c); authority(args, plan)
                stopped(c, original(c)); c.wait_gpu_release(GPU, UUID)
                authority(args, plan, required_seconds=SECONDS + 5)
                if Path(job['output']).exists():
                    raise FileExistsError('Recovery output exists; no scientific retry')
                label = 'stream-' + str(index)
                execute(c, command(c, job, plan['engineering'][job['task']]), label, status)
                c.verify_shard(job['output'], {k: job[k] for k in ('task', 'arm', 'rank')}, plan['engineering'][job['task']]['report_sha256'])
                if index == 0:
                    write(status / 'INTERRUPTED_PREFIX_PARITY.json', compare_prefix(plan))
            completed = []
            for row in plan['completed'] + plan['recovered']:
                job = {k: row[k] for k in ('task', 'arm', 'rank')}
                completed.append({**job, 'output': row['output'], 'report_sha256':
                    c.verify_shard(row['output'], job, plan['engineering'][job['task']]['report_sha256'])})
            science = scientific_verify(c, control, completed, plan['engineering'])
            verify_plan(plan, c); verify_sources(c)
            write(status / 'DONE.json', {'status': 'navigation_outage_recovery_union_complete',
                'worker': 'in0', 'plan_sha256': args.plan_sha256, 'original_plan_sha256': PLAN_SHA,
                'original_launch_sha256': LAUNCH_SHA, 'launch_sha256': digest(status / 'LAUNCH.json'),
                'identity': identity, 'instance': INSTANCE, 'gpu': GPU, 'gpu_uuid': UUID,
                'source_sha256': c.SOURCE, 'freeze_sha256': c.FREEZES, 'completed_jobs': completed,
                'engineering': plan['engineering'], **science, 'reused_streams': 8, 'recovered_streams': 7,
                'native_repeat_sha256': digest(status / 'NATIVE_REPEAT_VERIFIED.json'),
                'prefix_parity_sha256': digest(status / 'INTERRUPTED_PREFIX_PARITY.json'),
                'gpu_available_only_after_process_exit': True, 'original_outputs_modified': False,
                'partial_records_in_analysis': 0, 'new_episode_identities': 0})
        return verify_completed(args.plan_sha256)
    except BaseException as error:
        write(status / 'FAILED.json', {'error': str(error), 'error_type': type(error).__name__,
            'automatic_retry': False, 'original_process_signals': 0, 'original_outputs_modified': False})
        raise


def verify_completed(plan_sha):
    c, control = frozen_ops(); plan = read(ROOT / 'PLAN.json'); status = ROOT / 'run'
    if digest(ROOT / 'PLAN.json') != plan_sha or (status / 'FAILED.json').exists():
        raise ValueError('Wrong plan or failed recovery')
    verify_plan(plan, c, live=False); verify_sources(c)
    done, launch = read(status / 'DONE.json'), read(status / 'LAUNCH.json')
    require(launch, {'status': 'explicit_navigation_outage_recovery', 'plan_sha256': plan_sha,
        'original_launch_sha256': LAUNCH_SHA, 'instance': INSTANCE, 'gpu': GPU, 'gpu_uuid': UUID,
        'authorization_sha256': digest(ROOT / 'AUTHORIZATION.json'), 'namespace': plan['namespace']}, 'Recovery launch')
    require(done, {'status': 'navigation_outage_recovery_union_complete', 'worker': 'in0',
        'plan_sha256': plan_sha, 'original_plan_sha256': PLAN_SHA, 'original_launch_sha256': LAUNCH_SHA,
        'launch_sha256': digest(status / 'LAUNCH.json'), 'identity': launch['identity'],
        'instance': INSTANCE, 'gpu': GPU, 'gpu_uuid': UUID, 'source_sha256': c.SOURCE,
        'freeze_sha256': c.FREEZES, 'verified_endpoints': 180, 'reused_streams': 8, 'recovered_streams': 7,
        'engineering': plan['engineering'], 'original_outputs_modified': False,
        'partial_records_in_analysis': 0, 'new_episode_identities': 0}, 'Recovery DONE')
    expected = plan['completed'] + plan['recovered']
    if len(done['completed_jobs']) != 15:
        raise ValueError('Whole original assignment required')
    for index, (row, spec) in enumerate(zip(done['completed_jobs'], expected)):
        require(row, {k: spec[k] for k in ('task', 'arm', 'rank', 'output')}, 'Union result')
        if c.verify_shard(row['output'], EXPECTED[index], plan['engineering'][row['task']]['report_sha256']) != row['report_sha256']:
            raise ValueError('Union shard changed')
        if index >= 8:
            label = 'stream-' + str(index - 8)
            child, exited = read(status / (label + '-CHILD.json')), read(status / (label + '-EXIT.json'))
            require(child, {'command': command(c, spec, plan['engineering'][spec['task']]), 'ppid': launch['identity']['pid']}, 'Recovered child')
            require(exited, {'identity': child, 'returncode': 0, 'child_sha256': digest(status / (label + '-CHILD.json'))}, 'Recovered exit')
    for name, field in [('NATIVE_REPEAT_VERIFIED.json', 'native_repeat_sha256'), ('INTERRUPTED_PREFIX_PARITY.json', 'prefix_parity_sha256')]:
        if digest(status / name) != done[field]: raise ValueError('Recovery gate receipt changed')
    if read(status / 'INTERRUPTED_PREFIX_PARITY.json') != compare_prefix(plan):
        raise ValueError('Preserved interrupted prefix comparison changed')
    if read(status / 'NATIVE_REPEAT_VERIFIED.json') != verify_native_repeat(c, control):
        raise ValueError('Native repeat proof changed')
    child, exited = read(status / 'native-repeat-CHILD.json'), read(status / 'native-repeat-EXIT.json')
    require(child, {'command': native_command(c), 'ppid': launch['identity']['pid']}, 'Native repeat child')
    require(exited, {'identity': child, 'returncode': 0,
        'child_sha256': digest(status / 'native-repeat-CHILD.json')}, 'Native repeat exit')
    science = scientific_verify(c, control, done['completed_jobs'], plan['engineering'])
    return {'worker': 'in0', 'done_sha256': digest(status / 'DONE.json'), 'plan_sha256': PLAN_SHA,
        'recovery_plan_sha256': plan_sha, 'identity': done['identity'], 'instance': INSTANCE,
        'gpu': GPU, 'gpu_uuid': UUID, **science, 'completed_jobs': done['completed_jobs'],
        'original_launch_sha256': LAUNCH_SHA, 'recovery_root': str(ROOT), 'partial_records_in_analysis': 0}


def export_result(job, plan_sha):
    proof = verify_completed(plan_sha)
    rows = [row for row in proof['completed_jobs'] if all(row[k] == job[k] for k in ('task', 'arm', 'rank'))]
    if len(rows) != 1 or set(job) != {'task', 'arm', 'rank'}:
        raise ValueError('Only exact union stream exports are allowed')
    path = Path(rows[0]['output']); manifest = tree(path)
    with tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz') as archive:
        data = json.dumps(manifest, sort_keys=True).encode()
        entry = tarfile.TarInfo('MEMBERS.json'); entry.size = len(data)
        archive.addfile(entry, io.BytesIO(data))
        for name in manifest:
            if Path(name).name != name: raise ValueError('Scientific shard must be flat')
            archive.add(path / name, arcname=name, recursive=False)
    if tree(path) != manifest: raise ValueError('Union output changed during export')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run', 'verify-completed', 'export-result'))
    parser.add_argument('--plan', type=Path, default=ROOT / 'PLAN.json')
    parser.add_argument('--plan-sha256')
    parser.add_argument('--runtime', type=Path); parser.add_argument('--runtime-sha256')
    parser.add_argument('--authorization', type=Path); parser.add_argument('--authorization-sha256')
    parser.add_argument('--job', type=json.loads)
    args = parser.parse_args()
    def cancelled(signum, frame): raise RuntimeError('Recovery cancelled; no automatic retry')
    signal.signal(signal.SIGTERM, cancelled); signal.signal(signal.SIGINT, cancelled)
    if args.mode == 'prepare': result = prepare(args.runtime, args.runtime_sha256)
    elif args.mode == 'run': result = run(args)
    elif args.mode == 'verify-completed': result = verify_completed(args.plan_sha256)
    else:
        export_result(args.job, args.plan_sha256); return
    print(json.dumps(result))


if __name__ == '__main__': main()
