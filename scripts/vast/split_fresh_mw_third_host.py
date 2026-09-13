"""Reviewed same-GPU 8-to-24 stripe split; dry-run default, no old-worker signals.

An operator must first quiesce exact old process identities after receiving DONE.
The frozen worker itself revalidates engineering and resumes complete atomic arms.
This helper never changes scientific code, moves partial scenarios, or rents.
"""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import fresh_budget_watchdog as guard
from start_fresh_v2_navigation_handoff import exclusive, read, sha, command
from start_fresh_v2_mw_waves import cleanup, verify_bundle

INSTANCE = 50827072
FREEZE = '7d5def0122f0dcf80e07e43ddc4f28ef6532fb04e6eb9acc4bc428427165ba90'
GENERATION = 'mw-third-split24-v1'
OLD = [[gpu, task, worker, 8] for task, offset in [('reach', 0), ('reach-wall', 4)]
       for gpu, worker in zip(range(offset, offset + 4), range(3, 7))]
NEW = [[gpu, task, worker, 24] for gpu, task, worker, _ in OLD]


def plan(root, uuids, budget_name, budget_sha, inspect=guard.identity):
    logs = root / 'worker-logs-v2'
    assignment = read(logs / 'ASSIGNMENT.json')
    if (assignment['instance'] != INSTANCE or assignment['freeze_sha256'] != FREEZE
            or assignment['mapping'] != OLD or sha(root / 'fresh-freeze-v2/protocol.json') != FREEZE):
        raise ValueError('Original host/assignment/freeze differs')
    if set(uuids) != set(range(8)) or len(set(uuids.values())) != 8:
        raise ValueError('Eight distinct physical GPU UUIDs required')
    path = logs / guard.local_name(budget_name)
    if sha(path) != budget_sha:
        raise ValueError('Approved budget receipt SHA differs')
    prior = read(path)
    budget = prior['budget']
    expected_deadline = guard.deadline(datetime.fromisoformat(budget['as_of_utc'].replace('Z', '+00:00')).timestamp(),
        budget['accrued_usd'], budget['fleet_cap_usd_hour'], budget['campaign_cap_usd'], budget['closeout_reserve_usd'])
    if (prior['instance'] != INSTANCE or prior['assignment_sha256'] != sha(logs / 'ASSIGNMENT.json')
            or budget['campaign_cap_usd'] != 220 or budget['fleet_cap_usd_hour'] != 35
            or abs(prior['deadline_unix'] - expected_deadline) > .001):
        raise ValueError('Unapproved budget/assignment/deadline')
    if any((logs / name).exists() for name in [f'HANDOFF.{GENERATION}.json', f'ACTIVE_ASSIGNMENT.{GENERATION}.json',
                                               *guard.guard_names(GENERATION)]):
        raise ValueError('Existing split generation; no automatic retry')
    prior_workers = {Path(w['launch_receipt']).name: w for w in prior['workers']}
    if set(prior_workers) != {f'{t}-worker{w}-LAUNCHED.json' for _, t, w, _ in OLD}:
        raise ValueError('Prior guard does not cover original eight receipts')
    jobs, proofs, originals = [], [], []
    protocol = read(root / 'fresh-freeze-v2/protocol.json')
    for gpu, task, worker, _ in OLD:
        name = f'{task}-worker{worker}-LAUNCHED.json'
        receipt = read(logs / name)
        if ([receipt[k] for k in ('gpu', 'task', 'worker_id', 'workers')] != [gpu, task, worker, 8]
                or receipt['command'] != command(root, task, worker, 8)
                or sha(logs / name) != prior_workers[name]['launch_sha256']):
            raise ValueError('Original exact command/receipt differs')
        if inspect(receipt['pid']) is not None:
            raise ValueError('Original worker must already be dead; helper never stops it')
        uuid = uuids[gpu]
        engineering_sha = verify_bundle(root, task, 0, uuid, True)
        initial = root / 'results-v2' / task / f'scenario-{worker:03d}'
        initial_hashes = {}
        if initial.exists():
            row = next(r for r in protocol['tasks'][task]['records'] if r['role'] == 'scientific_candidates' and r['episode'] == worker)
            if read(initial / 'STARTED.json') != {'task': task, 'scenario': row, 'freeze_sha256': FREEZE,
                                                   'device_uuid': uuid, 'engineering': False}:
                raise ValueError('Initial partial scenario cannot move GPU or change input')
            initial_hashes = {str(p.relative_to(initial)): sha(p) for p in initial.rglob('*') if p.is_file()}
        # Reject ALL later old-stripe output, including the cases assigned away.
        for episode in range(worker + 8, 96, 8):
            if (root / 'results-v2' / task / f'scenario-{episode:03d}').exists():
                raise ValueError('Later coarse-stripe scenario started; root must reconcile ownership')
        if list(logs.glob(f'MW_CLAIM.{task}.*.json')):
            raise ValueError('Existing task claims require ownership reconciliation')
        launch = f'{task}-worker{worker}-{GENERATION}-LAUNCHED.json'
        if (logs / launch).exists() or (logs / (launch[:-5] + '.log')).exists():
            raise ValueError('Existing split launch receipt/log')
        jobs.append({'gpu': gpu, 'task': task, 'worker_id': worker, 'episode': worker,
                     'workers': 24, 'episodes': list(range(worker, 96, 24)),
                     'device_uuid': uuid, 'launch_receipt': launch})
        proofs.append({'gpu': gpu, 'task': task, 'engineering_report_sha256': engineering_sha,
                       'initial_episode': worker, 'initial_files_sha256': initial_hashes})
        originals.append({'launch_receipt': name, 'sha256': sha(logs / name), 'pid': receipt['pid']})
    proposal = {'instance': INSTANCE, 'freeze_sha256': FREEZE, 'generation': GENERATION,
        'mapping': NEW, 'device_uuids': {str(k): v for k, v in uuids.items()}, 'jobs': jobs,
        'engineering_and_resume_proofs': proofs, 'original_receipts': originals,
        'original_assignment_sha256': sha(logs / 'ASSIGNMENT.json'),
        'budget_receipt_name': budget_name, 'budget_receipt_sha256': budget_sha,
        'deadline_unix': prior['deadline_unix'], 'external_worker_ids_mod24': [[11, 12, 13, 14], [19, 20, 21, 22]],
        'launch_receipts': [j['launch_receipt'] for j in jobs]}
    return proposal, prior


def approval(path, proposal):
    record = read(path)
    if (record.get('approved') is not True or record.get('proposal') != proposal
            or record.get('old_coarse_launchers_disabled') is not True
            or record.get('global_split_ownership_verified') is not True):
        raise ValueError('Exact global split approval required')
    return sha(path)


def launch(root, proposal, prior, manifest):
    logs = root / 'worker-logs-v2'
    digest = approval(manifest, proposal)
    identity = guard.identity(prior['watchdog_pid'])
    if (identity is None or str(root) not in identity['command']
            or not any(Path(a).name == 'fresh_budget_watchdog.py' for a in identity['command'])):
        raise ValueError('Approved original guard must remain live')
    def check():
        if (sha(manifest) != digest or guard.identity(prior['watchdog_pid']) != identity
                or (logs / 'MW_SPLIT_STOP.json').exists() or list(logs.glob('BUDGET_QUIESCED*.json'))
                or prior['deadline_unix'] - time.time() < 4 * 2100 + 300):
            raise ValueError('Approval/guard/STOP/deadline changed; no further launch')
    check()
    # Recheck all old PIDs, input bytes, engineering and exact budget immediately before claiming.
    current, _ = plan(root, {int(k): v for k, v in proposal['device_uuids'].items()},
                      proposal['budget_receipt_name'], proposal['budget_receipt_sha256'])
    if current != proposal:
        raise ValueError('Prelaunch proposal changed')
    lock = (logs / (GENERATION + '.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    exclusive(logs / f'HANDOFF.{GENERATION}.json', {'proposal': proposal, 'approval_sha256': digest})
    owned, pending, observations = [], [], []
    try:
        # Same tested lifecycle as handoff helper, with this task's frozen workers=24 CLI.
        for job in proposal['jobs']:
            check()
            from start_fresh_v2_navigation_handoff import wait_exact_identity
            with (logs / (job['launch_receipt'][:-5] + '.log')).open('xb') as output:
                child = subprocess.Popen(command(root, job['task'], job['worker_id'], 24), cwd=root,
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(job['gpu']), 'JEPA_VERIFIED_LOCAL_DINO': '1'},
                    stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                pending.append(child)
            samples = []
            observations.append({'pid': child.pid, 'samples': samples})
            exact = wait_exact_identity(child, command(root, job['task'], job['worker_id'], 24), samples)
            worker = {'pid': child.pid, **exact}
            owned.append(worker)
            pending.remove(child)
            exclusive(logs / job['launch_receipt'], {**job, **worker})
        active = f'ACTIVE_ASSIGNMENT.{GENERATION}.json'
        exclusive(logs / active, proposal)
        argv = [sys.executable, '-u', str(Path(guard.__file__).resolve()), '--root', str(root),
            '--instance', str(INSTANCE), '--freeze-sha256', FREEZE, '--assignment-name', active, '--generation', GENERATION]
        for key in ('as_of_utc', 'accrued_usd', 'fleet_cap_usd_hour', 'campaign_cap_usd', 'closeout_reserve_usd'):
            argv += ['--' + key.replace('_', '-'), str(prior['budget'][key])]
        with (logs / f'GUARD.{GENERATION}.log').open('xb') as output:
            watcher = subprocess.Popen(argv, cwd=root, stdin=subprocess.DEVNULL, stdout=output,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        armed_path = logs / guard.guard_names(GENERATION)[1]
        until = time.monotonic() + 30
        while not armed_path.exists():
            check()
            if watcher.poll() is not None or time.monotonic() >= until:
                raise ValueError('Replacement guard failed to arm')
            time.sleep(.1)
        armed = read(armed_path)
        if (armed['watchdog_pid'] != watcher.pid or guard.identity(watcher.pid) is None
                or armed['assignment_sha256'] != sha(logs / active) or armed['deadline_unix'] != prior['deadline_unix']
                or len(armed['workers']) != 8 or any(not any(all(r.get(k) == w[k] for k in ('pid', 'command', 'start_tick'))
                    for r in armed['workers']) for w in owned)):
            raise ValueError('Replacement guard identity/coverage/deadline differs')
        result = {'guard_armed': True, 'guard_sha256': sha(armed_path), 'guard_pid': watcher.pid,
                  'new_pids': [w['pid'] for w in owned], 'old_workers_signaled': False}
        exclusive(logs / f'HANDOFF_MANAGED.{GENERATION}.json', result)
        return result
    except BaseException as error:
        exclusive(logs / f'HANDOFF_FAILED.{GENERATION}.json', {**cleanup(owned, pending),
            'error': str(error), 'identity_observations': observations, 'old_workers_signaled': False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--budget-receipt-name', required=True)
    parser.add_argument('--budget-receipt-sha256', required=True)
    parser.add_argument('--ownership', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--launch', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    uuids = {int(r.split(',')[0]): r.split(',')[1].strip().removeprefix('GPU-') for r in rows.splitlines()}
    proposal, prior = plan(root, uuids, args.budget_receipt_name, args.budget_receipt_sha256)
    if not args.launch:
        print(json.dumps({'dry_run': True, 'proposal': proposal}))
        return
    if args.ownership is None:
        parser.error('--launch requires exact reviewed --ownership')
    print(json.dumps(launch(root, proposal, prior, args.ownership)))


if __name__ == '__main__':
    main()
