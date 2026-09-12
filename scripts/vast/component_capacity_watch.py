"""Bounded capacity recovery for the already frozen MetaWorld panel only.

No new science, credit purchases, result-based selection or partial-stream replay.
Lease/price/US checks and immutable-source staging remain in the rental helper.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import component_rental_fleet as rental


def next_attempt(base):
    values = [1]
    for path in base.glob('rentals-v*'):
        if path.name.removeprefix('rentals-v').isdigit():
            values.append(int(path.name.removeprefix('rentals-v')))
    return max(values) + 1


def main(args):
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    children = []
    deadline = time.time() + args.hours * 3600
    rental.fleet.write(root / 'INTENT.json', {'deadline': deadline,
        'scope': 'only_unassigned_frozen_metaworld_component_streams',
        'hourly_cap': 7, 'credit_purchase': False, 'replay_interrupted_streams': False})
    try:
        while not stopped and time.time() < deadline:
            free = sorted(set(range(8)) - rental.claimed_slots(rental.fleet.BASE))
            if not free:
                break
            attempt, slot = next_attempt(rental.fleet.BASE), free[0]
            env = dict(os.environ, JEPA_COMPONENT_RENTAL_ATTEMPT=str(attempt))
            command = [rental.fleet.PYTHON, str(Path(rental.__file__).resolve())]
            stamp = str(time.time_ns())
            with (root / f'acquire-{stamp}.log').open('x') as log:
                process = subprocess.run(command + ['rent', '--slot', str(slot)],
                    env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300)
            acquired = process.returncode == 0
            event = {'time': time.time(), 'attempt': attempt, 'slot': slot,
                'acquired': acquired, 'free_before_attempt': free,
                'stage_processes': [{'pid': p.pid, 'returncode': p.poll()} for p in children]}
            rental.fleet.write(root / f'event-{stamp}.json', event)
            print(json.dumps(event), flush=True)
            if acquired:
                with (root / f'stage-{stamp}.log').open('x') as log:
                    children.append(subprocess.Popen(command + ['wait_stage', '--slot', str(slot)],
                        env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
            # A failed market search is not a reason to spin, duplicate or loosen safety.
            for _ in range(5 if acquired else 60):
                if stopped or time.time() >= deadline:
                    break
                time.sleep(1)
    finally:
        rental.fleet.write(root / 'WATCH_ENDED.json', {'time': time.time(),
            'unassigned': sorted(set(range(8)) - rental.claimed_slots(rental.fleet.BASE)),
            'does_not_imply_experiments_complete': True,
            'acquired_instances_retain_independent_guards': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hours', type=float, default=8)
    args = parser.parse_args()
    if not 0 < args.hours <= 12:
        parser.error('Bounded capacity watch must be at most twelve hours')
    main(args)
