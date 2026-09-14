"""Stop one identified replication process after its current whole case.

This external operational controller never edits the frozen experiment runner.
A briefly started next case is retained as an incomplete technical attempt and
must be reassigned only after the controller confirms the old process exited.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time


def read(path):
    return json.loads(Path(path).read_text())


def process_identity(pid):
    proc = Path('/proc')/str(pid)
    try:
        command = (proc/'cmdline').read_bytes().rstrip(b'\0')
        args = command.split(b'\0') if command else []
        stat = (proc/'stat').read_text()
    except FileNotFoundError:
        return None
    tail = stat[stat.rfind(')')+2:].split()
    if tail[0] == 'Z':
        return None
    return {'start_ticks': tail[19], 'args': [value.decode() for value in args]}


def assert_same_process(current, original):
    # Linux can clear cmdline before the dying process reaches zombie state.
    # Its unchanged start time distinguishes that transition from PID reuse.
    if current['start_ticks'] != original['start_ticks']:
        raise ValueError('PID identity changed')
    if current['args'] and current['args'] != original['args']:
        raise ValueError('Executor command changed')


def validate_process(identity, task, manifest_sha, output):
    args = identity['args']
    if '-m' not in args or args[args.index('-m')+1:args.index('-m')+3] != ['offline_study.experiments.lcfm_replication', 'execute']:
        raise ValueError('PID is not the registered replication executor')
    for flag, value in (('--task', task), ('--manifest-sha256', manifest_sha), ('--output', str(output))):
        if args.count(flag) != 1 or args[args.index(flag)+1] != value:
            raise ValueError('Executor ownership differs: '+flag)
    if '--engineering' in args:
        raise ValueError('This controller only drains assigned scientific work')


def snapshot(output, task, manifest_sha):
    complete = []; incomplete = []
    for directory in sorted((output/task).glob('episode-*')):
        started = directory/'STARTED.json'
        if not started.exists():
            continue
        try:
            binding = read(started)
            receipt = read(directory/'REPLICATION_CASE.json') if (directory/'REPLICATION_CASE.json').exists() else None
        except json.JSONDecodeError:
            continue
        if (binding['execution_manifest_sha256'] != manifest_sha or binding['input_binding']['task'] != task
                or binding['input_binding']['role'] != 'scientific_replication'):
            raise ValueError('Case ownership differs')
        episode = binding['input_binding']['episode']
        if receipt is not None:
            if (receipt.get('complete') is not True or receipt.get('manifest_sha256') != manifest_sha
                    or receipt.get('forward_count') != 70 or receipt.get('kernel_done_sha256') !=
                    hashlib.sha256((directory/'DONE.json').read_bytes()).hexdigest()):
                raise ValueError('Completed case receipt invalid')
            complete.append(episode)
        else:
            incomplete.append(episode)
    return {'complete': complete, 'incomplete': incomplete}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--task', choices=['reach', 'reach-wall'], required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--deadline-epoch', type=float, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise ValueError('Preserve prior controller receipts')
    identity = process_identity(args.pid)
    if identity is None:
        raise ValueError('Executor already exited; reconcile directly')
    validate_process(identity, args.task, args.manifest_sha256, args.output)
    initial = snapshot(args.output, args.task, args.manifest_sha256)
    target = None
    while time.time() < args.deadline_epoch:
        current = process_identity(args.pid)
        if current is None:
            reason = 'Executor finished before a stop was needed'
            break
        assert_same_process(current, identity)
        if not current['args']:
            time.sleep(.1)
            continue
        state = snapshot(args.output, args.task, args.manifest_sha256)
        if target is None and state['incomplete']:
            if len(state['incomplete']) != 1:
                raise ValueError('Multiple incomplete cases require reconciliation')
            target = state['incomplete'][0]
        if target is not None and target in state['complete']:
            # SIGTERM may arrive just after the next case starts. Its retained
            # partial directory never counts as completed work.
            os.kill(args.pid, signal.SIGTERM)
            for _ in range(300):
                current = process_identity(args.pid)
                if current is None:
                    break
                assert_same_process(current, identity)
                time.sleep(.1)
            else:
                raise RuntimeError('Executor has not exited; do not reassign its work')
            reason = 'Current whole case completed; executor stopped before queue reassignment'
            break
        time.sleep(.2)
    else:
        raise TimeoutError('Drain deadline reached; executor left running, no work may be reassigned')
    final = snapshot(args.output, args.task, args.manifest_sha256)
    result = {'pid': args.pid, 'process_identity': identity, 'task': args.task,
        'manifest_sha256': args.manifest_sha256, 'initial': initial, 'boundary_case': target,
        'final': final, 'executor_exited': True, 'reason': reason, 'at_epoch': time.time(),
        'controller_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'partial_attempt_policy': 'Retain every started directory; reassign incomplete cases in a new attempt directory'}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps({'task': args.task, 'boundary_case': target, 'executor_exited': True,
                      'completed_cases': len(final['complete']), 'partial_cases': final['incomplete']}))


if __name__ == '__main__':
    main()
