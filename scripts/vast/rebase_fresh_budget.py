"""Approved operational guard replacement; never restart/signal scientific workers.

Arm a fresh guard for the same receipt-bound worker assignment before retiring
only explicitly selected old watchdog process identities. Dry-run is default.
"""
import argparse
from datetime import datetime
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import fresh_budget_watchdog as guard
from start_fresh_v2_navigation_handoff import exclusive, read, sha, reap_unregistered, wait_exact_identity
from resume_fresh_v2_host import FREEZE, SOURCE, validate_files

BUDGET_KEYS = ('as_of_utc', 'accrued_usd', 'fleet_cap_usd_hour', 'campaign_cap_usd', 'closeout_reserve_usd')


def load_budget(path, digest):
    path = Path(path).resolve()
    if not re.fullmatch(r'[0-9a-f]{64}', digest or '') or sha(path) != digest:
        raise ValueError('Approved budget receipt SHA256 differs')
    record = read(path)
    budget = {k: record['budget'][k] for k in BUDGET_KEYS}
    if (record.get('approved') is not True
            or not re.fullmatch(r'[0-9a-f]{64}', record.get('source_snapshot_sha256', ''))
            or budget['fleet_cap_usd_hour'] != 39 or budget['campaign_cap_usd'] != 320):
        raise ValueError('Explicit approved 39/hour and320-total reconciliation required')
    stamp = datetime.fromisoformat(budget['as_of_utc'].replace('Z', '+00:00'))
    if stamp.tzinfo is None:
        raise ValueError('Budget snapshot needs explicit timezone')
    until = guard.deadline(stamp.timestamp(), budget['accrued_usd'], 39, 320, budget['closeout_reserve_usd'])
    return budget, until, {'path': str(path), 'sha256': digest}


def guard_command_matches(command, receipt, root):
    def option(flag, default=None):
        if flag not in command:
            return default
        index = command.index(flag)
        return command[index + 1] if index + 1 < len(command) else None
    return (any(Path(a).name == 'fresh_budget_watchdog.py' for a in command)
            and option('--root') == str(root)
            and option('--instance') == str(receipt['instance'])
            and option('--freeze-sha256') == FREEZE
            and option('--assignment-name', 'ASSIGNMENT.json') == receipt['budget'].get('assignment_name', 'ASSIGNMENT.json')
            and option('--generation') == receipt['budget'].get('generation'))


def plan(root, request, budget_path, budget_sha, boot_id, *, inspect=guard.identity, validate=validate_files):
    logs = root / 'worker-logs-v2'
    generation = request['generation']
    guard.guard_names(generation)
    if not generation.startswith('rebase-') or not request['old_guards']:
        raise ValueError('Unique rebase generation and explicit old guards required')
    if any((logs / name).exists() for name in [f'REBASE_STARTED.{generation}.json', *guard.guard_names(generation)]):
        raise ValueError('Rebase generation already attempted; reconcile instead of retrying')
    assignment_name = guard.local_name(request['assignment_name'])
    assignment_sha = sha(logs / assignment_name)
    protocol = read(root / 'fresh-freeze-v2/protocol.json')
    if (assignment_sha != request['assignment_sha256'] or sha(root / 'fresh-freeze-v2/protocol.json') != FREEZE
            or protocol['source_sha256'] != SOURCE):
        raise ValueError('Assignment/source/protocol changed')
    budget, until, binding = load_budget(budget_path, budget_sha)
    workers, verified_assignment_sha = guard.load_workers(root, request['instance'], FREEZE, assignment_name)
    if assignment_sha != verified_assignment_sha:
        raise ValueError('Worker assignment changed during validation')
    old = []
    for item in request['old_guards']:
        path = logs / guard.local_name(item['receipt_name'])
        if sha(path) != item['receipt_sha256']:
            raise ValueError('Old guard receipt SHA differs')
        receipt = read(path)
        current = inspect(receipt['watchdog_pid'])
        if (receipt['instance'] != request['instance'] or receipt['assignment_sha256'] != assignment_sha
                or current is None or not guard_command_matches(current['command'], receipt, root)
                or 'run_fresh_confirmation.py' in ' '.join(current['command'])
                or until < receipt['deadline_unix'] or any(receipt['watchdog_pid'] == w['pid'] for w in workers)):
            raise ValueError('Old watchdog identity/assignment/deadline differs')
        # The old receipt's live scientific identities must be included unchanged.
        for previous in receipt['workers']:
            live = inspect(previous['pid'])
            if live is not None and live == {'command': previous['command'], 'start_tick': previous['start_tick']}:
                if not any(all(w[k] == previous[k] for k in ('pid', 'command', 'start_tick')) for w in workers):
                    raise ValueError('Old live worker not covered by replacement assignment')
        old.append({'receipt_name': item['receipt_name'], 'receipt_sha256': item['receipt_sha256'],
                    'pid': receipt['watchdog_pid'], **current, 'boot_id': boot_id})
    if len({g['pid'] for g in old}) != len(old):
        raise ValueError('Duplicate old watchdog identities')
    validate(root)
    return {'request': request, 'instance': request['instance'], 'generation': generation,
            'assignment_name': assignment_name, 'assignment_sha256': assignment_sha,
            'freeze_sha256': FREEZE, 'source_sha256': SOURCE, 'boot_id': boot_id,
            'workers': workers, 'old_guards': old, 'budget': budget, 'deadline_unix': until,
            'budget_binding': binding}


def approve(path, proposal):
    record = read(path)
    if (record.get('approved') is not True or record.get('proposal') != proposal
            or record.get('only_named_watchdogs_may_be_retired') is not True):
        raise ValueError('Exact reviewed guard-only rebase approval required')
    return sha(path)


def verify_new_guard(root, proposal, watcher, receipt, *, inspect=guard.identity, expected_watchdog_identity=None):
    current_watchdog = inspect(watcher.pid)
    if (receipt['watchdog_pid'] != watcher.pid or current_watchdog is None
            or (expected_watchdog_identity is not None and current_watchdog != expected_watchdog_identity)
            or receipt['assignment_sha256'] != proposal['assignment_sha256']
            or receipt['deadline_unix'] != proposal['deadline_unix']
            or receipt['instance'] != proposal['instance']
            or any(receipt['budget'][k] != proposal['budget'][k] for k in BUDGET_KEYS)):
        raise ValueError('New guard identity/assignment/deadline differs')
    expected = {(w['launch_receipt'], w['launch_sha256']) for w in proposal['workers']}
    actual = {(w['launch_receipt'], w['launch_sha256']) for w in receipt['workers']}
    if expected != actual or len(receipt['workers']) != len(proposal['workers']):
        raise ValueError('Replacement guard receipt coverage differs')
    for worker in proposal['workers']:
        current = inspect(worker['pid'])
        if current is None:
            continue
        if current != {'command': worker['command'], 'start_tick': worker['start_tick']}:
            raise ValueError('Scientific PID changed; refuse ambiguous replacement')
        if not any(all(w[k] == worker[k] for k in ('pid', 'command', 'start_tick')) for w in receipt['workers']):
            raise ValueError('Replacement guard lacks exact live scientific identity')


def retire_old_watchdogs(proposal, *, inspect=guard.identity, quiesce=guard.quiesce):
    science = {w['pid'] for w in proposal['workers']}
    for old in proposal['old_guards']:
        if (old['pid'] in science or not any(Path(a).name == 'fresh_budget_watchdog.py' for a in old['command'])
                or inspect(old['pid']) != {'command': old['command'], 'start_tick': old['start_tick']}):
            raise ValueError('Old watchdog PID changed; nothing retired')
    # quiesce signals ONLY these explicit watchdog processes, never their covered workers.
    return quiesce([{k: old[k] for k in ('pid', 'command', 'start_tick')} for old in proposal['old_guards']], grace=1)


def launch(root, proposal, approval_path):
    logs, generation = root / 'worker-logs-v2', proposal['generation']
    approval_sha = approve(approval_path, proposal)
    def check():
        if (sha(approval_path) != approval_sha or sha(Path(proposal['budget_binding']['path'])) != proposal['budget_binding']['sha256']
                or Path('/proc/sys/kernel/random/boot_id').read_text().strip() != proposal['boot_id']
                or (logs / 'REBASE_STOP.json').exists() or time.time() + 120 >= proposal['deadline_unix']):
            raise ValueError('Rebase approval/boot/budget/STOP changed')
    check()
    current = plan(root, proposal['request'], proposal['budget_binding']['path'], proposal['budget_binding']['sha256'], proposal['boot_id'])
    if current != proposal:
        raise ValueError('Rebase preflight changed')
    lock = (logs / 'BUDGET_REBASE.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    exclusive(logs / f'REBASE_STARTED.{generation}.json', {'proposal': proposal, 'approval_sha256': approval_sha})
    watcher, armed_verified = None, False
    try:
        argv = [sys.executable, '-u', str(Path(guard.__file__).resolve()), '--root', str(root),
                '--instance', str(proposal['instance']), '--freeze-sha256', FREEZE,
                '--assignment-name', proposal['assignment_name'], '--generation', generation]
        for key, value in proposal['budget'].items():
            argv += ['--' + key.replace('_', '-'), str(value)]
        with (logs / f'GUARD.{generation}.log').open('xb') as output:
            watcher = subprocess.Popen(argv, cwd=root, stdin=subprocess.DEVNULL, stdout=output,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        new_guard_identity = wait_exact_identity(watcher, argv, [])
        path = logs / guard.guard_names(generation)[1]
        until = time.monotonic() + 30
        while not path.exists():
            check()
            if watcher.poll() is not None or time.monotonic() >= until:
                raise ValueError('New guard failed to arm; old guards retained')
            time.sleep(.1)
        verify_new_guard(root, proposal, watcher, read(path), expected_watchdog_identity=new_guard_identity)
        armed_verified = True
        check()
        retired = retire_old_watchdogs(proposal)
        if not retired['quiesced']:
            raise ValueError('Old watchdog retirement incomplete; new guard retained, reconcile')
        result = {'new_guard_sha256': sha(path), 'new_watchdog_pid': watcher.pid,
                  'retired_watchdogs': retired, 'scientific_workers_signaled': False,
                  'old_receipts_preserved': True}
        exclusive(logs / f'REBASE_DONE.{generation}.json', result)
        return result
    except BaseException as error:
        stopped_new = reap_unregistered(watcher) if watcher is not None and not armed_verified else None
        exclusive(logs / f'REBASE_FAILED.{generation}.json', {'error': str(error),
             'new_guard_verified_and_retained': armed_verified, 'unverified_new_guard_cleanup': stopped_new,
             'scientific_workers_signaled': False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--budget-receipt', type=Path, required=True)
    parser.add_argument('--budget-sha', required=True)
    parser.add_argument('--approval', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--launch', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    proposal = plan(root, read(args.request), args.budget_receipt, args.budget_sha,
                    Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    if not args.launch:
        print(json.dumps({'dry_run': True, 'proposal': proposal}))
        return
    if args.approval is None:
        parser.error('--launch requires exact reviewed --approval')
    print(json.dumps(launch(root, proposal, args.approval)))


if __name__ == '__main__':
    main()
