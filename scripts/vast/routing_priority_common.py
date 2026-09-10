"""Operations-only, same-device HMM priority handoff helpers.

No scientific source, fit, freeze, episode schedule, or original queue is edited.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path('/workspace/jepa-runtime')
CONTROL = ROOT / 'routing-priority-20260908-v3'
PANEL = ROOT / 'hmm-fixed-response-behavior-20260908-v3'
CODE = ROOT / 'hmm-fixed-response-code-20260908-v3'
PYTHON = '/workspace/jepa-planning-python/bin/python'
FREEZE = 'c6acb23c1abc275b61f6b4983732862211439cea713f401b2d13d3cf61a7f0cf'
SOURCE = 'd6cc7a7fcdad2d36d695a43b55fd99c4f1831fb833082c5e974bca229b88d9dc'
ORIGINAL = ROOT / 'fixed-response-code-20260908-v4/scripts/vast/run_fixed_behavior_queue.py'
ORIGINAL_SHA = '28ade0b758704f20fd16118ead1fd9993a5648d7769aaf48009cb782613c466d'
REQUIRED = ('native', 'fixed_rank4', 'matched_random_fixed_rank4')
ORIGINAL_ARMS = REQUIRED + ('coupling_only', 'matched_random_coupling')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    """Append-only control receipts. Never overwrite scientific or old status files."""
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def stat_fields(raw):
    # comm may contain spaces or parentheses; all following fields are scalar.
    suffix = raw[raw.rfind(')') + 2:].split()
    return {'state': suffix[0], 'ppid': int(suffix[1]), 'starttime': int(suffix[19])}


def process(pid, proc=Path('/proc')):
    """Consistent identity snapshot; normal exit/reap is not PID reuse."""
    folder = proc / str(pid)
    try:
        first = stat_fields((folder / 'stat').read_text())
        command = (folder / 'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
        second = stat_fields((folder / 'stat').read_text())
    except FileNotFoundError:
        return None
    if first['starttime'] != second['starttime']:
        raise ValueError('PID changed during identity snapshot')
    return {'pid': pid, **second, 'command': [] if command == [''] else command}


def alive(binding, reader=process):
    current = reader(binding['pid'])
    if current is None:
        return False
    if current['starttime'] != binding['starttime']:
        raise ValueError('PID reused; never signal replacement process')
    if current['state'] in ('Z', 'X'):
        return False
    if current['command'] != binding['command']:
        raise ValueError('Bound process command changed')
    return True


def started(pid, expected, reader=process, timeout=5):
    """Popen may return while /proc argv is temporarily empty during exec."""
    deadline = time.monotonic() + timeout
    identity = None
    while True:
        current = reader(pid)
        if current is None or current['state'] in ('Z', 'X'):
            raise ValueError('New child exited before stable identity capture')
        if identity is not None and current['starttime'] != identity:
            raise ValueError('New child PID reused during startup')
        identity = current['starttime']
        if current['command'] == expected:
            return current
        if time.monotonic() > deadline:
            raise TimeoutError('New child never reached its exact expected argv')
        time.sleep(.01)


def send(binding, signum):
    """pidfd closes the check/signal PID-reuse race on the receiving Linux host."""
    if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        # Packaged research Python omits these Linux APIs. The installed system
        # Python supplies them; it imports only this standard-library helper.
        result = subprocess.check_output(['/usr/bin/python3', str(Path(__file__).resolve()),
            'signal-helper', json.dumps(binding), str(int(signum))], text=True)
        return json.loads(result)['sent']
    try:
        descriptor = os.pidfd_open(binding['pid'])
    except ProcessLookupError:
        return False
    try:
        if not alive(binding):
            return False
        try:
            signal.pidfd_send_signal(descriptor, signum)
        except ProcessLookupError:
            return False
        return True
    finally:
        os.close(descriptor)


def pidfd_available():
    check = 'import os,signal; fd=os.pidfd_open(os.getpid()); assert hasattr(signal,"pidfd_send_signal"); os.close(fd)'
    subprocess.run(['/usr/bin/python3', '-c', check], check=True)


def gpu_processes(gpu):
    output = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    return [] if not output else [int(line.strip()) for line in output.splitlines()]


def gpu_uuid(gpu):
    value = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    return normalize_nvml_uuid(value)


def normalize_nvml_uuid(value):
    """Only remove NVML's literal GPU- prefix; Torch receipts use bare UUIDs."""
    if not re.fullmatch(r'GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', value):
        raise ValueError('Unexpected receiving NVIDIA UUID format')
    return value.removeprefix('GPU-')


def scientific_arguments():
    spec = importlib.util.spec_from_file_location('original_routing_queue', ROOT / 'run_routing_queue_v3.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = module.arguments()
    args = argparse.Namespace(**{values[i][2:].replace('-', '_'): Path(values[i + 1])
        for i in range(0, len(values), 2)})
    return values, args


def environment(gpu=''):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_GL='egl',
        PYOPENGL_PLATFORM='egl', JEPA_VERIFIED_LOCAL_DINO='1', PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        LD_LIBRARY_PATH='/opt/conda/lib',
        PYTHONPATH=str(CODE / 'src') + ':/workspace/jepa-python/lib/python3.10/site-packages')


def frozen_dependencies():
    sys.path.insert(0, str(CODE / 'src'))
    from offline_study.routing_behavior import read_protocol
    _, args = scientific_arguments()
    protocol = read_protocol(args)
    if protocol['source_sha256'] != SOURCE or digest(PANEL / 'freeze/protocol.json') != FREEZE:
        raise ValueError('Frozen scientific source or protocol changed')
    if digest(ORIGINAL) != ORIGINAL_SHA:
        raise ValueError('Original producer source changed')
    return args, protocol


def reference_gate(gpu, args, protocol):
    """Verify all required records/traces, exact streams and same receiving device."""
    from offline_study.routing_behavior import reference_records
    task = 'reach' if gpu < 4 else 'reach-wall'
    rank = gpu % 4
    bindings = {}
    for arm in REQUIRED:
        root = args.reference / task / arm / f'shard-gpu{gpu}'
        if (root / 'FAILED.json').exists():
            raise ValueError('Required original reference failed')
        if not (root / 'DONE.json').exists():
            return None
    device = gpu_uuid(gpu)
    for arm, mapped in zip(REQUIRED, ('native', 'constant_gate', 'matched_random_constant_gate')):
        rows, hashes = reference_records(args, protocol, task, mapped, [rank, rank + 4])
        if len(rows) != 24 or any(row['device_uuid'] != device for row in rows):
            raise ValueError('Wrong reference stream or device')
        bindings.update(hashes)
    return bindings


def validate_original_child(child, gpu):
    command = child['command']
    prefix = [PYTHON, '-u', '-m', 'offline_study.fixed_response_behavior', 'run']
    if command[:len(prefix)] != prefix or child['ppid'] != 5202 + gpu:
        raise ValueError('Unknown child; never pause or interrupt it')
    task = 'reach' if gpu < 4 else 'reach-wall'
    arm = command[command.index('--arm') + 1]
    target = ROOT / f'fixed-response-behavior-20260908-v1/{task}/{arm}/shard-gpu{gpu}'
    rank = gpu % 4
    if (arm not in ORIGINAL_ARMS or command[command.index('--task') + 1] != task or
            command[command.index('--output') + 1] != str(target) or
            command[command.index('--logical-ranks') + 1:command.index('--logical-ranks') + 3] != [str(rank), str(rank + 4)]):
        raise ValueError('Child arm/task/output/stream identity differs')
    evidence = ROOT / 'fixed-response-behavior-evidence-20260908-v1/artifacts/offline_study'
    fixed = evidence / 'fixed-response-20260908-v1'
    code = ROOT / 'fixed-response-code-20260908-v4'
    expected = prefix + ['--vendor', '/workspace/jepa_steering/vendor/jepa-wms', '--fits', str(fixed / 'fits'),
        '--original', str(evidence / 'primary-durable-20260907'),
        '--stimuli', str(evidence / 'restored-behavioral-inputs-20260908-v1'),
        '--checkpoint', str(ROOT / 'fixed-response-assets-20260908-v1/jepa_wm_metaworld.pth.tar'),
        '--task', task, '--freeze', str(fixed / 'behavioral-freeze-v1'),
        '--engineering', str(ROOT / f'fixed-response-worker-checks-20260908-v1/gpu-{gpu}'),
        '--checked-source', str(code / 'src/offline_study'), '--arm', arm,
        '--logical-ranks', str(rank), str(rank + 4), '--output', str(target)]
    if command != expected:
        raise ValueError('Original scientific child full argv changed')
    return target


def verify_boundary(target, gpu, args, protocol):
    """Completion evidence, not outcomes, authorizes release of the original GPU."""
    from offline_study.behavioral_development import assigned_rows, schedule, validate_coverage, verified_report
    from offline_study.fixed_response_smoke import verify_episode
    report, report_hash = verified_report(target)
    launch = json.loads((target / 'protocol.json').read_text())
    task = 'reach' if gpu < 4 else 'reach-wall'
    arm = target.parent.name
    rank = gpu % 4
    expected = assigned_rows(schedule(), [rank, rank + 4])
    if ((target / 'FAILED.json').exists() or report['status'] != 'fixed_response_behavioral_shard_complete' or
            report['episodes'] != 24 or report['task'] != task or report['arm'] != arm or
            report['parameters_unchanged'] is not True or report['fresh_confirmation'] is not False or
            report['protocol_sha256'] != digest(target / 'protocol.json') or
            launch['expected_episodes'] != expected or launch['device_uuid'] != gpu_uuid(gpu) or
            launch['source_sha256'] != protocol['reference_source_sha256'] or
            launch['freeze_sha256'] != protocol['reference_freeze_sha256']):
        raise ValueError('Invalid completed original boundary shard')
    rows = []
    wanted = {f"episode-{row['episode']:03d}.json" for row in expected}
    if set(report['episode_files_sha256']) != wanted:
        raise ValueError('Missing or extra original records')
    for name, wanted_hash in report['episode_files_sha256'].items():
        if digest(target / name) != wanted_hash:
            raise ValueError('Original episode hash changed')
        row = json.loads((target / name).read_text())
        if row['arm'] != arm:
            raise ValueError('Original record arm changed')
        calls = target / f"calls-{row['episode']:03d}"
        for filename, key in (('unroll_calls.json', 'unroll_calls_sha256'), ('action_trace.json', 'action_trace_sha256')):
            if digest(calls / filename) != row[key]:
                raise ValueError('Original raw trace changed')
        verify_episode(row['result'], json.loads((calls / 'unroll_calls.json').read_text()))
        rows.append(row)
    validate_coverage(rows, expected)
    return report_hash


def safe_to_resume(parent, active_child, gpu, reader=process, occupied=gpu_processes):
    """Resume never races an active HMM or original scientific child."""
    if active_child is not None and alive(active_child, reader):
        return False
    if occupied(gpu):
        return False
    if not alive(parent, reader):
        return False
    current = reader(parent['pid'])
    return current is not None and current['starttime'] == parent['starttime'] and current['state'] in ('T', 't')


def resume_parent(parent, child, gpu, status, timeout=24 * 3600):
    started = time.monotonic()
    while True:
        if not alive(parent):
            write(status / 'RESUME_NOT_NEEDED.json', {'parent_terminal': True})
            return
        if safe_to_resume(parent, child, gpu):
            send(parent, signal.SIGCONT)
            write(status / 'PARENT_RESUMED.json', {'parent': parent, 'active_child_terminal': True,
                'gpu_empty_before_resume': True, 'original_rng_and_outputs_unchanged': True})
            return
        current = process(parent['pid'])
        if current and current['state'] not in ('T', 't'):
            raise ValueError('Parent resumed externally; do not launch or signal anything')
        if time.monotonic() - started > timeout:
            raise TimeoutError('Cannot resume while GPU or child is active; preserve all work')
        time.sleep(2)


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] != 'signal-helper':
        raise ValueError('Only the narrow pidfd signal helper has a CLI')
    if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        raise ValueError('System signal helper requires native pidfd APIs')
    requested = int(sys.argv[3])
    if requested not in (signal.SIGSTOP, signal.SIGCONT, signal.SIGTERM):
        raise ValueError('Unregistered administrative signal')
    print(json.dumps({'sent': send(json.loads(sys.argv[2]), requested)}))
