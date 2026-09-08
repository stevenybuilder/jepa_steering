"""Identity-bound user-directed pause of explicitly owned extension workers."""
import argparse
import json
import os
from pathlib import Path
import signal
import time

ROOT = Path('/workspace/jepa-runtime')
NAV = str(ROOT / 'navigation-redistribution-20260908-v2/navigation_redistribution_control.py')
SELECTORS = {
    'tx': [NAV, str(ROOT / 'pointmaze-corrected-queue-20260908-v1/code/scripts/vast/corrected_pointmaze_queue.py'),
           'offline_study.fixed_combined_smoke'],
    'ne': [NAV],
    'in': [str(ROOT / 'navigation-recovery-in0-20260908-v1/navigation_recovery.py')],
    'nj': [str(ROOT / 'run_pusht_native_queue_v1.py'), 'offline_study.training_history'],
}


def process(pid):
    try:
        folder = Path('/proc') / str(pid)
        raw = (folder / 'stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        command = (folder / 'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
        after = (folder / 'stat').read_text().split(')')[-1].split()
        if int(fields[19]) != int(after[19]):
            raise ValueError('Process identity changed during capture')
        return {'pid': pid, 'ppid': int(after[1]), 'state': after[0], 'starttime': int(after[19]), 'command': command}
    except FileNotFoundError:
        return None


def live(item):
    current = process(item['pid'])
    if current is None or current['state'] in ('Z', 'X'):
        return False
    if current['starttime'] != item['starttime'] or current['command'] != item['command']:
        raise ValueError('Process identity changed; do not signal replacement')
    return True


def send(item, sig):
    try:
        fd = os.pidfd_open(item['pid'])
    except ProcessLookupError:
        return
    try:
        if live(item):
            signal.pidfd_send_signal(fd, sig)
    except ProcessLookupError:
        pass
    finally:
        os.close(fd)


def table():
    result = [process(int(p.name)) for p in Path('/proc').iterdir() if p.name.isdigit()]
    return {p['pid']: p for p in result if p and p['state'] not in ('Z', 'X')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', choices=SELECTORS, required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    output = ROOT / 'core-priority-pause-20260908-v1'
    if args.verify_only:
        owned = json.loads((output / 'BOUND_PROCESSES.json').read_text())
        if any(live(item) for item in owned):
            raise ValueError('Some exact owned processes still live')
        with (output / 'POST_PAUSE_VERIFIED.json').open('x') as f:
            json.dump({'worker': args.worker, 'owned_processes_terminal': len(owned),
                       'time': time.time(), 'files_deleted': 0,
                       'initial_verifier_exit_race_not_scientific_failure': True}, f, indent=2)
        print(json.dumps({'worker': args.worker, 'verified_terminal': len(owned)})); return
    items = table()
    matched = {pid: item for pid, item in items.items()
               if any(token in item['command'] for token in SELECTORS[args.worker])}
    roots = [item for item in matched.values() if item['ppid'] not in matched]
    if not roots:
        raise ValueError('No expected live extension roots; inspect rather than assume success')
    if not args.apply:
        print(json.dumps({'worker': args.worker, 'roots': roots})); return
    output.mkdir(exist_ok=False)
    with (output / 'INTENT.json').open('x') as f:
        json.dump({'user_directed_pause': True, 'roots': roots, 'time': time.time(),
                   'no_output_deleted': True, 'partial_not_complete': True,
                   'training_resume_requires_last_complete_checkpoint': True}, f, indent=2)
    # Freeze only these supervisors briefly so no new child can be scheduled
    # between the ownership snapshot and cancellation.
    for item in roots:
        send(item, signal.SIGSTOP)
    items = table()
    owned = {item['pid']: item for item in roots}
    while True:
        additions = {pid: item for pid, item in items.items() if item['ppid'] in owned and pid not in owned}
        if not additions:
            break
        owned.update(additions)
    with (output / 'BOUND_PROCESSES.json').open('x') as f:
        json.dump(list(owned.values()), f, indent=2)
    # TERM first, CONT second delivers cancellation to stopped supervisors.
    # Existing handlers retain outputs and do not schedule further work.
    for item in reversed(list(owned.values())):
        send(item, signal.SIGTERM)
    for item in roots:
        send(item, signal.SIGCONT)
    deadline = time.monotonic() + 55
    while time.monotonic() < deadline:
        remaining = [item for item in owned.values() if live(item)]
        if not remaining:
            with (output / 'PAUSED.json').open('x') as f:
                json.dump({'worker': args.worker, 'owned_processes_terminal': len(owned),
                           'files_deleted': 0, 'time': time.time(), 'not_a_scientific_failure': True}, f, indent=2)
            print(json.dumps({'worker': args.worker, 'paused_processes': len(owned)})); return
        time.sleep(1)
    raise TimeoutError('Some bound processes remain; inspect, do not kill unrelated work')


if __name__ == '__main__':
    main()
