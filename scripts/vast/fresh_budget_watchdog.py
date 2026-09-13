"""Host-local budget deadline: quiesce owned workers, preserve files, never destroy.

This has no provider/cloud credentials and DOES NOT end instance billing. A
separate preservation/STOP controller must consume BUDGET_QUIESCED.json.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import re
import time


def deadline(as_of, accrued, fleet_cap, total_cap=220., reserve=5.):
    values = (as_of, accrued, fleet_cap, total_cap, reserve)
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Nonfinite budget')
    if accrued < 0 or fleet_cap <= 0 or reserve < 5 or total_cap <= reserve:
        raise ValueError('Invalid budget or insufficient closeout reserve')
    return as_of + max(0., total_cap - reserve - accrued) / fleet_cap * 3600


def identity(pid, proc_root=Path('/proc')):
    """Bind both exact argv and Linux process start tick; never PID alone."""
    try:
        base = proc_root / str(pid)
        argv = (base / 'cmdline').read_bytes().rstrip(b'\0').split(b'\0')
        argv = [v.decode() for v in argv if v]
        stat = (base / 'stat').read_text().rsplit(')', 1)[1].split()
        if not argv or stat[0] == 'Z':
            return None
        return {'command': argv, 'start_tick': int(stat[19])}
    except (FileNotFoundError, ProcessLookupError):
        return None


def local_name(name):
    if not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('Receipt must be an immediate filename')
    return name


def guard_names(generation=None):
    if generation is not None and not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}', generation):
        raise ValueError('Invalid watchdog generation')
    suffix = '' if generation is None else '.' + generation
    return ('BUDGET_WATCHDOG' + suffix + '.lock',
            'BUDGET_WATCHDOG_ARMED' + suffix + '.json',
            'BUDGET_QUIESCED' + suffix + '.json')


def load_workers(root, instance, expected_sha, assignment_name='ASSIGNMENT.json'):
    root = root.resolve()
    logs = root / 'worker-logs-v2'
    assignment_path = logs / local_name(assignment_name)
    assignment_bytes = assignment_path.read_bytes()
    assignment = json.loads(assignment_bytes)
    if assignment['instance'] != instance or assignment['freeze_sha256'] != expected_sha:
        raise ValueError('Assignment instance/freeze mismatch')
    protocol = root / 'fresh-freeze-v2/protocol.json'
    if hashlib.sha256(protocol.read_bytes()).hexdigest() != expected_sha:
        raise ValueError('Frozen protocol differs')
    mapping = {tuple(row) for row in assignment['mapping']}
    expected_count = assignment.get('expected_gpu_count', 8)
    if (type(expected_count) is not int or not 1 <= expected_count <= 8
            or len(mapping) != expected_count
            or len({row[0] for row in mapping}) != expected_count
            or any(type(row[0]) is not int or not 0 <= row[0] < 8 for row in mapping)):
        raise ValueError('Expected distinct explicitly assigned GPU lanes')
    workers, seen = [], set()
    receipt_names = assignment.get('launch_receipts')
    if receipt_names is None:
        if assignment_name != 'ASSIGNMENT.json':
            raise ValueError('Active assignment requires explicit launch receipts')
        paths = sorted(logs.glob('*-LAUNCHED.json'))
    else:
        if len(receipt_names) != expected_count or len(set(receipt_names)) != expected_count:
            raise ValueError('Missing or duplicate receipt selection')
        paths = [logs / local_name(name) for name in receipt_names]
    for path in paths:
        record = json.loads(path.read_bytes())
        row = (record['gpu'], record['task'], record['worker_id'], record['workers'])
        if row not in mapping or row in seen or not isinstance(record['pid'], int) or record['pid'] <= 1:
            raise ValueError('Unexpected launch identity')
        seen.add(row)
        expected = ['-u', str(root / 'scripts/run_fresh_confirmation.py'), 'worker',
                    '--project', str(root), '--freeze', str(root / 'fresh-freeze-v2'),
                    '--task', record['task'], '--worker-id', str(record['worker_id']),
                    '--workers', str(record['workers']), '--output', str(root / 'results-v2')]
        if record['command'][1:] != expected:
            raise ValueError('Worker command is not the frozen fresh-bank launcher')
        current = identity(record['pid'])
        if current is not None and current['command'] != record['command']:
            raise ValueError('Launch PID reused or command differs')
        if current is not None and record.get('start_tick', current['start_tick']) != current['start_tick']:
            raise ValueError('Launch process start tick differs')
        workers.append({'pid': record['pid'], 'command': record['command'],
                        'start_tick': None if current is None else current['start_tick'],
                        'launch_receipt': str(path),
                        'launch_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    if len(workers) != len(mapping) or len({w['pid'] for w in workers}) != len(workers):
        raise ValueError('Missing or duplicate launched workers')
    return workers, hashlib.sha256(assignment_bytes).hexdigest()


def quiesce(workers, *, inspect=identity, send=os.kill, sleep=time.sleep, grace=30):
    """Only exact, still-matching process identities may receive a signal."""
    sent, mismatches = [], []
    def matching(worker):
        current = inspect(worker['pid'])
        if current is None:
            return False
        if worker['start_tick'] is None or current != {
                'command': worker['command'], 'start_tick': worker['start_tick']}:
            mismatches.append(worker['pid'])
            return False
        return True
    for worker in workers:
        if matching(worker):
            try:
                send(worker['pid'], signal.SIGTERM)
                sent.append({'pid': worker['pid'], 'signal': 'TERM'})
            except ProcessLookupError:
                pass
    if sent:
        sleep(grace)
    for worker in workers:
        if matching(worker):
            try:
                send(worker['pid'], signal.SIGKILL)
                sent.append({'pid': worker['pid'], 'signal': 'KILL'})
            except ProcessLookupError:
                pass
    if sent:
        sleep(1)
    alive = [w['pid'] for w in workers if matching(w)]
    return {'signals': sent, 'identity_mismatches': sorted(set(mismatches)),
            'owned_workers_still_alive': alive,
            'quiesced': not alive and not mismatches,
            'files_deleted': False, 'cloud_verified': False,
            'provider_stopped': False, 'billing_continues': True}


def publish(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('x') as handle:
        json.dump(data, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--instance', type=int, required=True)
    parser.add_argument('--freeze-sha256', required=True)
    parser.add_argument('--as-of-utc', required=True)
    parser.add_argument('--accrued-usd', type=float, required=True)
    parser.add_argument('--fleet-cap-usd-hour', type=float, default=20.)
    parser.add_argument('--campaign-cap-usd', type=float, default=220.)
    parser.add_argument('--closeout-reserve-usd', type=float, default=5.)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--assignment-name', default='ASSIGNMENT.json')
    parser.add_argument('--generation')
    args = parser.parse_args()
    lock_name, armed_name, quiesced_name = guard_names(args.generation)
    as_of = datetime.fromisoformat(args.as_of_utc.replace('Z', '+00:00'))
    if as_of.tzinfo is None:
        parser.error('Budget snapshot must carry an explicit UTC offset')
    stop_at = deadline(as_of.timestamp(), args.accrued_usd,
                       args.fleet_cap_usd_hour, args.campaign_cap_usd,
                       args.closeout_reserve_usd)
    workers, assignment_sha = load_workers(args.root, args.instance, args.freeze_sha256,
                                         args.assignment_name)
    plan = {'instance': args.instance, 'assignment_sha256': assignment_sha,
            'workers': workers, 'deadline_unix': stop_at,
            'deadline_utc': datetime.fromtimestamp(stop_at, timezone.utc).isoformat(),
            'budget': vars(args).copy(), 'billing_stopped_by_this_watchdog': False}
    plan['budget']['root'] = str(args.root)
    if args.dry_run:
        print(json.dumps({**plan, 'dry_run': True, 'signals_sent': False}))
        return
    logs = args.root / 'worker-logs-v2'
    # Host-local exclusivity; the descriptor stays open for the entire guard.
    import fcntl
    lock = (logs / lock_name).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    armed = logs / armed_name
    if armed.exists():
        raise ValueError('Existing deadline guard; inspect before replacing')
    publish(armed, {**plan, 'watchdog_pid': os.getpid(), 'armed_unix': time.time()})
    print(json.dumps({'armed': True, 'deadline_utc': plan['deadline_utc']}), flush=True)
    while time.time() < stop_at:
        time.sleep(max(0., min(15., stop_at - time.time())))
    result = quiesce(workers)
    publish(logs / quiesced_name, {**plan, **result, 'completed_unix': time.time(),
            'next_action': 'Preserve complete and partial records; provider STOP retains disk; never destroy without independently verified cloud backup.'})
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
