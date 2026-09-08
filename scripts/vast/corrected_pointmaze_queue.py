"""Fail-closed fresh PointMaze histories after an owned navigation assignment.

Local preparation only until the resource-board owner stages a frozen PLAN.json.
This script does not rent instances, stage files, signal predecessors, or resume
historical extra-shuffled runs. Each invocation owns exactly one of seeds234–236.

PLAN fields: schema=1, source_root/source_sha256, vendor, python, overlay,
driver/driver_sha256, output_root, assets, input_check, input_receipt, data_root,
fixed_files={absolute_path:sha256}, performance_approval={path,sha256}, and
assignments=[{seed,instance,gpu,gpu_uuid,predecessor:{pid,start_ticks,argv,
done,failed,launch,launch_sha256,expected_done,verifier:{python,script,
script_sha256,args,expected_result}}}]. All three seeds must occur exactly once.

The parent freezes the predecessor adapter only after navigation redistribution
is finalized. Its hash-bound CPU verifier must independently validate all assigned
scientific shards and emit one JSON object. No tentative assignment is runnable.
The performance approval binds this exact corrected source, sampler, seed triplet,
and a completed bounded profiling receipt. It is not inferred from elapsed time.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


SEEDS = [234, 235, 236]
VENDOR_COMMIT = '13cf1d9c7e476f53c17714d2e0f1dc239a883ce0'
POLICY = {'version': 'dataset_specific_native_sampler_v1', 'task': 'pointmaze',
    'logical_ranks': 16, 'train_shuffle': False, 'validation_shuffle': False,
    'validation_sampler_epoch': 0, 'slice_permutation_seed': 234,
    'training_drop_last': True, 'validation_drop_last': False}
APPROVAL_STATUS = 'corrected_pointmaze_execution_approved_after_performance_audit'


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    # Publish immutable receipts atomically; never replace another attempt.
    path = Path(path)
    temporary = path.with_name(path.name + '.publishing')
    with temporary.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def source_hash(root):
    result = hashlib.sha256()
    files = sorted((Path(root) / 'src/offline_study').glob('*.py'))
    if not files:
        raise ValueError('Missing frozen scientific source')
    for path in files:
        result.update(path.name.encode() + b'\0' + path.read_bytes())
    return result.hexdigest()


def require_fields(actual, expected, label):
    if not expected or any(actual.get(key) != value for key, value in expected.items()):
        raise ValueError(label + ' binding mismatch')


def checked_report(root):
    root = Path(root)
    if (root / 'FAILED.json').exists():
        raise ValueError('Failed stage is not complete: ' + str(root))
    report = read(root / 'report.json')
    if read(root / 'DONE.json').get('report_sha256') != digest(root / 'report.json'):
        raise ValueError('Unverified report: ' + str(root))
    if report.get('protocol_sha256') != digest(root / 'protocol.json'):
        raise ValueError('Unverified protocol: ' + str(root))
    return report, read(root / 'protocol.json')


def validate_plan(plan, seed):
    if plan.get('schema') != 1 or sorted(a['seed'] for a in plan['assignments']) != SEEDS:
        raise ValueError('Exactly the predefined three seeds are required')
    assignments = plan['assignments']
    if len({(a['instance'], a['gpu']) for a in assignments}) != 3:
        raise ValueError('Concurrent seeds must have three distinct owned devices')
    for item in assignments:
        if not item['gpu_uuid'].startswith('GPU-') or item['gpu'] < 0 or item['instance'] <= 0:
            raise ValueError('Missing exact owned NVIDIA device identity')
        before = item['predecessor']
        if before['pid'] <= 0 or before['start_ticks'] <= 0 or not before['argv']:
            raise ValueError('Navigation process identity is not frozen')
        # A verifier must prove complete assigned scope, not merely no live PID.
        if not before['expected_done'] or not before['verifier']['expected_result']:
            raise ValueError('Missing full-assignment completion contract')
    for key in ('source_root', 'vendor', 'python', 'overlay', 'driver', 'output_root',
                'assets', 'input_check', 'input_receipt', 'data_root'):
        path = Path(plan[key])
        if not path.is_absolute() or '..' in path.parts or str(path) == '/':
            raise ValueError('Expected explicit bounded path: ' + key)
    if 'corrected' not in Path(plan['output_root']).name:
        raise ValueError('Fresh corrected histories need a distinctly named root')
    if not plan['fixed_files']:
        raise ValueError('Runtime, input and native-test file bindings are required')
    test = str(Path(plan['source_root']) / 'tests/test_native_training_sampler.py')
    if test not in plan['fixed_files']:
        raise ValueError('Actual upstream sampler test must be frozen')
    matches = [a for a in assignments if a['seed'] == seed]
    if len(matches) != 1:
        raise ValueError('Unknown or duplicated seed')
    return matches[0]


def environment(plan, gpu=None):
    return dict(os.environ, CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
        PYTHONPATH=str(Path(plan['source_root']) / 'src') + ':' + plan['overlay'],
        LD_LIBRARY_PATH='/opt/conda/lib', LD_PRELOAD=plan['driver'],
        JEPA_VERIFIED_LOCAL_DINO='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1')


def verify_static(plan):
    if source_hash(plan['source_root']) != plan['source_sha256'] or digest(plan['driver']) != plan['driver_sha256']:
        raise ValueError('Corrected source or pinned driver changed')
    for path, expected in plan['fixed_files'].items():
        if not Path(path).is_absolute() or digest(path) != expected:
            raise ValueError('Pinned runtime/input/test file changed: ' + path)
    approval = plan['performance_approval']
    if digest(approval['path']) != approval['sha256']:
        raise ValueError('Performance decision is not frozen')
    decision = read(approval['path'])
    require_fields(decision, {'status': APPROVAL_STATUS, 'source_sha256': plan['source_sha256'],
        'sampler_policy': POLICY, 'training_seeds': SEEDS,
        'execution': 'native_accumulation_uncached'}, 'Performance approval')
    if digest(decision['profiling_receipt']) != decision['profiling_receipt_sha256']:
        raise ValueError('Bounded performance audit receipt changed')
    vendor = plan['vendor']
    if (subprocess.check_output(['git', '-C', vendor, 'rev-parse', 'HEAD'], text=True).strip() != VENDOR_COMMIT or
            subprocess.check_output(['git', '-C', vendor, 'status', '--porcelain'], text=True).strip()):
        raise ValueError('Pinned native source changed')


def process_identity(pid, proc=Path('/proc')):
    path = proc / str(pid)
    try:
        stat = (path / 'stat').read_text().rsplit(') ', 1)[1].split()
        argv = (path / 'cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:
        return None
    return {'state': stat[0], 'start_ticks': int(stat[19]),
        'argv': [value.decode() for value in argv if value]}


def predecessor_pending(binding, proc=Path('/proc')):
    failed, done = Path(binding['failed']), Path(binding['done'])
    if failed.exists():
        raise ValueError('Navigation predecessor failed; preserve and do not bypass')
    if done.exists():
        return False
    identity = process_identity(binding['pid'], proc)
    # DONE may have been published between the first check and process exit.
    if failed.exists():
        raise ValueError('Navigation predecessor failed during handoff')
    if done.exists():
        return False
    if identity is None or identity['state'] == 'Z':
        raise ValueError('Navigation predecessor exited without completion')
    if identity['start_ticks'] != binding['start_ticks'] or identity['argv'] != binding['argv']:
        raise ValueError('Navigation predecessor identity changed')
    return True


def verify_predecessor(binding, plan):
    if Path(binding['failed']).exists() or digest(binding['launch']) != binding['launch_sha256']:
        raise ValueError('Failed or changed navigation launch')
    require_fields(read(binding['done']), binding['expected_done'], 'Navigation completion')
    verifier = binding['verifier']
    if digest(verifier['script']) != verifier['script_sha256']:
        raise ValueError('Navigation scientific verifier changed')
    output = subprocess.check_output([verifier['python'], verifier['script']] + verifier['args'],
        env=environment(plan), text=True, timeout=900)
    result = json.loads(output)
    require_fields(result, verifier['expected_result'], 'Entire navigation assignment verification')
    return result


def gpu_empty(assignment):
    gpu = str(assignment['gpu'])
    uuid = subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    if uuid != assignment['gpu_uuid']:
        raise ValueError('Owned GPU identity changed')
    return not subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-compute-apps=pid', '--format=csv,noheader'], text=True).split()


def training_arguments(plan, seed, output):
    output = Path(output)
    common = ['--vendor', plan['vendor'], '--data-root', plan['data_root'],
        '--input-check', plan['input_check'], '--input-receipt', plan['input_receipt'],
        '--pilot', str(output / 'receiving-pilot'), '--seed', str(seed)]
    engineering = output / 'epoch-one-engineering'
    return (common + ['--engineering-only', '--output', str(engineering)],
        common + ['--resume-from', str(engineering / 'jepa-e0.pth.tar'),
            '--engineering-proof', str(engineering), '--output', str(output / 'remaining-epochs')])


def verify_pilot(root):
    report, protocol = checked_report(root)
    require_fields(report, {'status': 'native_training_accumulation_pilot_passed',
        'task': 'pointmaze', 'updates_complete': 5, 'updates_per_epoch': 1139,
        'training_clips': 145800, 'sampler_policy': POLICY}, 'Receiving numerical pilot')
    if protocol.get('sampler_policy') != POLICY or report['parity_sha256'] != digest(Path(root) / 'PARITY.json'):
        raise ValueError('Numerical pilot does not use the corrected sampler')


def verify_engineering(root, seed, pilot):
    root = Path(root)
    report, protocol = checked_report(root)
    require_fields(report, {'status': 'one_epoch_pointmaze_training_engineering_complete',
        'seed': seed, 'checkpoints': 1, 'validation_events': 5}, 'Seed-specific first epoch')
    require_fields(protocol, {'seed': seed, 'sampler_policy': POLICY, 'resume_checkpoint': None,
        'pilot_report_sha256': digest(Path(pilot) / 'report.json')}, 'Own fresh initialization')
    if report['resume_parity_sha256'] != digest(root / 'RESUME_PARITY.json'):
        raise ValueError('Exact-resume proof changed')
    proof = read(root / 'RESUME_PARITY.json')
    require_fields(proof, {'status': 'actual_native_checkpoint_loader_and_resumed_update_exact',
        'checkpoint_sha256': digest(root / 'jepa-e0.pth.tar'),
        'next_update_parameters_optimizer_scaler_cpu_rng_exact': True,
        'validation_cursor_cuda_rng_loader_state_restored': True}, 'Actual native resume proof')


def verify_history(root, seed, engineering):
    root = Path(root)
    report, protocol = checked_report(root)
    require_fields(report, {'status': 'all50_pointmaze_training_epochs_complete', 'seed': seed,
        'checkpoints': 50, 'validation_events': 250, 'behavioral_evaluations_complete': False}, 'Complete history')
    require_fields(protocol, {'seed': seed, 'sampler_policy': POLICY,
        'resume_checkpoint': str(Path(engineering) / 'jepa-e0.pth.tar')}, 'History resume origin')
    if report['checkpoint_manifest_sha256'] != digest(root / 'CHECKPOINTS.json'):
        raise ValueError('History manifest changed')
    history = read(root / 'CHECKPOINTS.json')['history']
    if [row['epoch'] for row in history] != list(range(1, 51)):
        raise ValueError('Incomplete frozen epoch coverage')
    for row in history:
        if digest(row['path']) != row['sha256']:
            raise ValueError('Checkpoint history member changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if digest(args.plan) != args.plan_sha256:
        raise ValueError('Queue plan changed')
    plan = read(args.plan)
    assignment = validate_plan(plan, args.seed)
    verify_static(plan)
    output = Path(plan['output_root']) / ('seed-' + str(args.seed))
    output.mkdir(parents=True, exist_ok=False)
    queue = output / 'queue'
    queue.mkdir()
    child = None

    def interrupted(signum, frame):
        raise RuntimeError('Owned corrected queue interrupted; preserve results')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def execute(module, arguments, label, timeout):
        nonlocal child
        verify_static(plan)
        if not gpu_empty(assignment):
            raise ValueError('GPU is occupied before ' + label)
        with (queue / (label + '.log')).open('x') as log:
            child = subprocess.Popen([plan['python'], '-u', '-m', module] + arguments,
                env=environment(plan, assignment['gpu']), stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            if child.wait(timeout=timeout):
                raise ValueError('Corrected training stage failed: ' + label)
        child = None

    try:
        with (queue / 'actual-upstream-sampler-tests.log').open('x') as log:
            subprocess.run([plan['python'], '-m', 'unittest', 'discover', '-s',
                str(Path(plan['source_root']) / 'tests'), '-p', 'test_native_training_sampler.py', '-v'],
                env=environment(plan), stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
        write(queue / 'PREPARED.json', {'plan_sha256': args.plan_sha256, 'seed': args.seed,
            'source_sha256': plan['source_sha256'], 'sampler_policy': POLICY, 'gpu_calls': 0,
            'actual_upstream_test_log_sha256': digest(queue / 'actual-upstream-sampler-tests.log')})
        if args.prepare_only:
            return  # This attempt is CPU-only; activation uses a fresh output_root.
        write(queue / 'LAUNCH.json', {'pid': os.getpid(), 'seed': args.seed,
            'instance': assignment['instance'], 'gpu': assignment['gpu'], 'device_uuid': assignment['gpu_uuid'],
            'plan_sha256': args.plan_sha256, 'worker_sha256': digest(__file__),
            'source_sha256': plan['source_sha256'], 'gpu_job_started': False})
        before = assignment['predecessor']
        deadline = time.monotonic() + 96 * 3600
        print(json.dumps({'status': 'waiting_for_whole_navigation_assignment', 'seed': args.seed}), flush=True)
        while predecessor_pending(before):
            if time.monotonic() > deadline:
                raise TimeoutError('Navigation assignment did not complete; no bypass')
            time.sleep(30)
        verified = verify_predecessor(before, plan)
        write(queue / 'PREDECESSOR_VERIFIED.json', verified)
        # DONE is necessary but the original supervisor must also be terminal.
        for _ in range(24):
            identity = process_identity(before['pid'])
            if identity is not None and identity['start_ticks'] != before['start_ticks']:
                raise ValueError('Predecessor PID was reused during handoff')
            if (identity is None or identity['state'] == 'Z') and gpu_empty(assignment):
                break
            time.sleep(5)
        else:
            raise ValueError('Completed navigation supervisor/GPU did not become free')
        if shutil.disk_usage(output).free < 36 * 1024**3:
            raise ValueError('Insufficient remaining space for three corrected histories')
        # Cooperating new queues cannot claim the same device simultaneously.
        lock = Path(plan['output_root']) / (assignment['gpu_uuid'] + '.lock')
        with lock.open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            pilot = output / 'receiving-pilot'
            execute('offline_study.training_pilot', ['--vendor', plan['vendor'], '--assets', plan['assets'],
                '--input-check', plan['input_check'], '--task', 'pointmaze', '--output', str(pilot)],
                'receiving-pilot', 3600)
            verify_pilot(pilot)
            engineering, remaining = training_arguments(plan, args.seed, output)
            execute('offline_study.pointmaze_training_history', engineering, 'epoch-one-engineering', 6 * 3600)
            verify_engineering(output / 'epoch-one-engineering', args.seed, pilot)
            execute('offline_study.pointmaze_training_history', remaining, 'remaining-epochs', 120 * 3600)
            verify_history(output / 'remaining-epochs', args.seed, output / 'epoch-one-engineering')
        write(queue / 'DONE.json', {'status': 'corrected_pointmaze_seed_history_complete', 'seed': args.seed,
            'history_report_sha256': digest(output / 'remaining-epochs/report.json'),
            'plan_sha256': args.plan_sha256, 'behavioral_history_evaluation_complete': False,
            'full_study_complete': False})
    except Exception as exc:
        write(queue / 'FAILED.json', {'error': str(exc), 'partial_files_preserved': True})
        raise
    finally:
        # Cleanup only this queue's own newly spawned process group.
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()


if __name__ == '__main__':
    main()
