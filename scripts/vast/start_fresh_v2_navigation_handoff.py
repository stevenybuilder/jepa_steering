"""One reviewed A100 navigation handoff; dry-run by default, never rents.

Preserves original assignment/launch/guard history. The new generation must arm
before this launcher reports management handoff; failure quiesces only its new
workers. It never stops the original watchdog or the four unchanged lanes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import fresh_budget_watchdog as guard

INSTANCE = 50819364
FREEZE = '7d5def0122f0dcf80e07e43ddc4f28ef6532fb04e6eb9acc4bc428427165ba90'
GENERATION = 'navigation-handoff-v2'
NEW = [(1, 'pointmaze', 2, 4), (3, 'wall', 2, 4),
       (4, 'pointmaze', 3, 4), (5, 'wall', 3, 4)]
OLD = {1: ('reach', 4, 8), 3: ('reach-wall', 3, 8),
       4: ('reach-wall', 4, 8), 5: ('reach-wall', 5, 8)}
ACTIVE = 'ACTIVE_ASSIGNMENT.' + GENERATION + '.json'
HANDOFF = 'HANDOFF.' + GENERATION + '.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def exclusive(path, data):
    with path.open('x') as handle:
        json.dump(data, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())


def command(root, task, worker_id, workers):
    return [sys.executable, '-u', str(root / 'scripts/run_fresh_confirmation.py'),
            'worker', '--project', str(root), '--freeze', str(root / 'fresh-freeze-v2'),
            '--task', task, '--worker-id', str(worker_id), '--workers', str(workers),
            '--output', str(root / 'results-v2')]


def plan(root, uuids, inspect=guard.identity):
    root = root.resolve()
    logs = root / 'worker-logs-v2'
    old = read(logs / 'ASSIGNMENT.json')
    if old['instance'] != INSTANCE or old['freeze_sha256'] != FREEZE or sha(root / 'fresh-freeze-v2/protocol.json') != FREEZE:
        raise ValueError('Wrong receiving instance or frozen protocol')
    if set(uuids) != set(range(8)) or len(set(uuids.values())) != 8:
        raise ValueError('Expected eight distinct physical GPUs')
    if (logs / HANDOFF).exists() or (logs / ACTIVE).exists():
        raise ValueError('Existing handoff: reconcile instead of retrying')
    mapping = {row[0]: tuple(row[1:]) for row in old['mapping']}
    if len(mapping) != 8 or any(mapping.get(gpu) != stripe for gpu, stripe in OLD.items()):
        raise ValueError('Original lane assignment differs')
    selected, replaced, jobs = [], [], []
    for gpu in range(8):
        task, w, n = mapping[gpu]
        receipt_name = f'{task}-worker{w}-LAUNCHED.json'
        receipt = read(logs / receipt_name)
        if (receipt['gpu'], receipt['task'], receipt['worker_id'], receipt['workers']) != (gpu, task, w, n):
            raise ValueError('Original launch identity differs')
        if gpu in OLD:
            if inspect(receipt['pid']) is not None:
                raise ValueError('Original lane process is still alive; no handoff')
            replaced.append({'gpu': gpu, 'launch_receipt': receipt_name,
                             'launch_sha256': sha(logs / receipt_name), 'pid': receipt['pid']})
        else:
            selected.append(receipt_name)
    for gpu, task, w, n in NEW:
        episodes = list(range(w, 96, n))
        if any((root / 'results-v2' / task / f'scenario-{e:03d}').exists() for e in episodes):
            raise ValueError('Incoming stripe already has local output; inspect ownership')
        name = f'{task}-worker{w}-{GENERATION}-LAUNCHED.json'
        if (logs / name).exists() or (logs / (name[:-5] + '.log')).exists():
            raise ValueError('Existing replacement receipt/log')
        jobs.append({'gpu': gpu, 'device_uuid': uuids[gpu], 'task': task,
                     'worker_id': w, 'workers': n, 'episodes': episodes,
                     'command': command(root, task, w, n), 'launch_receipt': name})
        selected.append(name)
        mapping[gpu] = (task, w, n)
    return {'instance': INSTANCE, 'freeze_sha256': FREEZE, 'generation': GENERATION,
            'original_assignment_sha256': sha(logs / 'ASSIGNMENT.json'),
            'mapping': [[gpu, *mapping[gpu]] for gpu in range(8)],
            'device_uuids': uuids, 'launch_receipts': selected, 'replaced': replaced,
            'jobs': jobs, 'new_scenarios': sum(len(j['episodes']) for j in jobs),
            'global_ownership_verified_by_this_script': False}


def ownership(path, proposal):
    record = read(path)
    if (record.get('approved') is not True or record.get('instance') != INSTANCE
            or record.get('freeze_sha256') != FREEZE
            or record.get('stock_slot2_launcher_disabled') is not True
            or record.get('mapping') != [list(row) for row in NEW]
            or record.get('globally_unstarted_and_unclaimed') is not True):
        raise ValueError('Explicit reviewed global ownership handoff required')
    return {'path': str(path.resolve()), 'sha256': sha(path)}


def reap_unregistered(child):
    """Own the Popen child before /proc identity is available; never signal by PID.

    Popen checks/reaps its own child before signaling, so an exited/reaped child
    cannot be confused with an unrelated process that later reuses its PID.
    """
    if child.poll() is None:
        child.terminate()
    try:
        code = child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        child.kill()
        code = child.wait(timeout=30)
    return {'pid': child.pid, 'returncode': code, 'owned_popen_child_reaped': True}


def wait_exact_identity(child, expected, observations, *, timeout=2., inspect=None,
                        clock=time.monotonic, sleep=time.sleep):
    """Bounded observation, not relaxed command matching or a claimed race fix."""
    inspect = guard.identity if inspect is None else inspect
    started = clock()
    first_tick = None
    while True:
        current = inspect(child.pid)
        elapsed = clock() - started
        argv = [] if current is None else current['command']
        tick = None if current is None else current['start_tick']
        match = (current is not None and argv == expected and isinstance(tick, int) and tick > 0)
        # Hash arbitrary argv rather than recording potentially sensitive values.
        sample = {'elapsed_seconds': round(elapsed, 6), 'identity_present': current is not None,
                  'argv_sha256': hashlib.sha256(json.dumps(argv).encode()).hexdigest(),
                  'executable_basename': Path(argv[0]).name if argv else None,
                  'known_script_basenames': [Path(arg).name for arg in argv if Path(arg).name in
                                           ('run_fresh_confirmation.py', 'start_fresh_v2_navigation_handoff.py')],
                  'argc': len(argv), 'start_tick': tick, 'exact_command_match': match,
                  'child_returncode': None}
        observations.append(sample)
        if tick is not None:
            if first_tick is not None and tick != first_tick:
                raise ValueError('New worker process start tick changed during verification')
            first_tick = tick
        if match:
            return current
        sample['child_returncode'] = child.poll()
        if sample['child_returncode'] is not None or elapsed >= timeout:
            raise ValueError('New worker did not retain exact frozen command within startup bound')
        sleep(min(.05, timeout - elapsed))


def launch(root, proposal, approval):
    logs = root / 'worker-logs-v2'
    prior = read(logs / 'BUDGET_WATCHDOG_ARMED.json')
    prior_identity = guard.identity(prior['watchdog_pid'])
    if (prior_identity is None or prior['instance'] != INSTANCE
            or prior['assignment_sha256'] != proposal['original_assignment_sha256']
            or str(root) not in prior_identity['command']
            or not any(Path(arg).name == 'fresh_budget_watchdog.py' for arg in prior_identity['command'])):
        raise ValueError('Original budget watchdog is not live')
    if any((logs / name).exists() for name in guard.guard_names(GENERATION)):
        raise ValueError('Existing watchdog generation; inspect before retry')
    if prior['deadline_unix'] - time.time() < 300:
        raise ValueError('Insufficient budget deadline headroom for handoff')
    exclusive(logs / HANDOFF, {**proposal, 'ownership': approval, 'created_unix': time.time(),
                             'prior_guard_identity': prior_identity,
                             'prior_guard_receipt_sha256': sha(logs / 'BUDGET_WATCHDOG_ARMED.json')})
    owned, unregistered, identity_observations = [], [], []
    try:
        for job in proposal['jobs']:
            with (logs / (job['launch_receipt'][:-5] + '.log')).open('xb') as output:
                child = subprocess.Popen(job['command'], cwd=root,
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(job['gpu']), 'JEPA_VERIFIED_LOCAL_DINO': '1'},
                    stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                unregistered.append(child)
            observations = []
            identity_observations.append({'pid': child.pid, 'gpu': job['gpu'], 'samples': observations})
            current = wait_exact_identity(child, job['command'], observations)
            worker = {'pid': child.pid, **current}
            owned.append(worker)
            unregistered.remove(child)
            exclusive(logs / job['launch_receipt'], {**job, **worker, 'launched_at': time.time(),
                                                   'identity_observations': observations})
        exclusive(logs / ACTIVE, {**proposal, 'ownership': approval, 'created_unix': time.time()})
        args = [sys.executable, '-u', str(Path(__file__).with_name('fresh_budget_watchdog.py')),
                '--root', str(root), '--instance', str(INSTANCE), '--freeze-sha256', FREEZE,
                '--assignment-name', ACTIVE, '--generation', GENERATION]
        for flag, key in (('as-of-utc', 'as_of_utc'), ('accrued-usd', 'accrued_usd'),
                          ('fleet-cap-usd-hour', 'fleet_cap_usd_hour'),
                          ('campaign-cap-usd', 'campaign_cap_usd'),
                          ('closeout-reserve-usd', 'closeout_reserve_usd')):
            args += ['--' + flag, str(prior['budget'][key])]
        with (logs / ('BUDGET_WATCHDOG.' + GENERATION + '.log')).open('xb') as output:
            watcher = subprocess.Popen(args, cwd=root, stdin=subprocess.DEVNULL,
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        armed = logs / guard.guard_names(GENERATION)[1]
        until = time.monotonic() + 30
        while not armed.exists() and watcher.poll() is None and time.monotonic() < until:
            time.sleep(.1)
        receipt = read(armed)
        if (receipt['watchdog_pid'] != watcher.pid or guard.identity(watcher.pid) is None
                or receipt['assignment_sha256'] != sha(logs / ACTIVE)
                or receipt['deadline_unix'] > prior['deadline_unix']
                or any(not any(all(record.get(k) == worker[k] for k in ('pid', 'command', 'start_tick'))
                                   for record in receipt['workers']) for worker in owned)):
            raise ValueError('New watchdog did not arm for the complete active assignment')
        result = {'new_guard_armed': True, 'watchdog_pid': watcher.pid,
                  'armed_receipt': str(armed), 'original_guard_untouched': True,
                  'active_assignment': str(logs / ACTIVE), 'new_worker_pids': [w['pid'] for w in owned]}
        exclusive(logs / ('HANDOFF_MANAGED.' + GENERATION + '.json'), result)
        return result
    except BaseException:
        reaped = []
        for child in unregistered:
            try:
                reaped.append(reap_unregistered(child))
            except Exception as error:
                reaped.append({'pid': child.pid, 'owned_popen_child_reaped': False,
                               'cleanup_error': str(error), 'requires_operator_attention': True})
        result = guard.quiesce(owned)
        exclusive(logs / ('HANDOFF_FAILED.' + GENERATION + '.json'),
                  {**result, 'unregistered_children': reaped,
                   'identity_observations': identity_observations})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--launch', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    parser.add_argument('--ownership', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    uuids = {int(row.split(',')[0]): row.split(',')[1].strip().removeprefix('GPU-') for row in rows.splitlines()}
    proposal = plan(root, uuids)
    if not args.launch:
        print(json.dumps({**proposal, 'dry_run': True, 'launched': False}))
        return
    if args.ownership is None:
        parser.error('--launch requires a reviewed --ownership receipt')
    approval = ownership(args.ownership, proposal)
    print(json.dumps(launch(root, proposal, approval)), flush=True)


if __name__ == '__main__':
    main()
