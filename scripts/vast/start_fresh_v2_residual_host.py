"""One reviewed four-GPU residual MetaWorld host; frozen worker, dry-run default."""
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
from start_fresh_v2_navigation_handoff import exclusive, read, sha, command, wait_exact_identity
from start_fresh_v2_mw_waves import cleanup
from resume_fresh_v2_host import validate_files, FREEZE, SOURCE, duplicate_workers

GENERATION = 'residual-four-v1'
BUDGET = {'as_of_utc': '2026-09-13T00:54:48Z', 'accrued_usd': 52.274813,
          'fleet_cap_usd_hour': 35, 'campaign_cap_usd': 220, 'closeout_reserve_usd': 5}
DEADLINE = guard.deadline(datetime.fromisoformat(BUDGET['as_of_utc'].replace('Z', '+00:00')).timestamp(),
                          BUDGET['accrued_usd'], 35, 220, 5)


def plan(root, instance, task, uuids, *, validate=validate_files, scan=duplicate_workers):
    if task not in ('reach', 'reach-wall') or instance <= 0:
        raise ValueError('Explicit valid instance and MetaWorld task required')
    if set(uuids) != set(range(4)) or len(set(uuids.values())) != 4:
        raise ValueError('Four distinct full physical GPU UUIDs required')
    if any(not (root / marker).is_file() for marker in ('RUNTIME_READY.json', 'ASSETS_READY.json')):
        raise ValueError('Receiving runtime/assets not ready')
    protocol = read(root / 'fresh-freeze-v2/protocol.json')
    if sha(root / 'fresh-freeze-v2/protocol.json') != FREEZE or protocol['source_sha256'] != SOURCE:
        raise ValueError('Frozen protocol/source changed')
    logs = root / 'worker-logs-v2'
    if (root / 'results-v2').exists() or (logs / 'ASSIGNMENT.json').exists() or list(logs.glob('*LAUNCHED.json')):
        raise ValueError('Existing results/assignment/launch: reconcile, never retry implicitly')
    if list(logs.glob('RESIDUAL*')) or list(logs.glob('BUDGET_QUIESCED*')) or any((logs / n).exists() for n in guard.guard_names(GENERATION)):
        raise ValueError('Existing residual or guard generation')
    if scan(root):
        raise ValueError('Existing scoped scientific worker')
    mapping = [[gpu, task, worker, 32] for gpu, worker in enumerate((7, 15, 23, 31))]
    jobs = [{'gpu': gpu, 'task': task, 'worker_id': worker, 'workers': count,
             'device_uuid': uuids[gpu], 'episodes': list(range(worker, 96, count)),
             'command': command(root, task, worker, count),
             'launch_receipt': f'{task}-worker{worker}-{GENERATION}-LAUNCHED.json'}
            for gpu, task, worker, count in mapping]
    validate(root)
    return {'instance': instance, 'task': task, 'freeze_sha256': FREEZE, 'source_sha256': SOURCE,
            'expected_gpu_count': 4, 'generation': GENERATION, 'mapping': mapping, 'jobs': jobs,
            'device_uuids': {str(k): v for k, v in uuids.items()},
            'launch_receipts': [j['launch_receipt'] for j in jobs], 'budget': BUDGET,
            'deadline_unix': DEADLINE, 'scientific_scenarios': 12}


def approve(path, proposal):
    record = read(path)
    if (record.get('approved') is not True or record.get('proposal') != proposal
            or record.get('globally_unstarted_and_unclaimed') is not True
            or record.get('other_residual_launchers_disabled') is not True
            or record.get('provider_and_budget_authorized') is not True):
        raise ValueError('Exact instance/task/mapping/budget/ownership approval required')
    return sha(path)


def launch(root, proposal, approval_path):
    logs = root / 'worker-logs-v2'
    approval_sha = approve(approval_path, proposal)
    def check():
        if (sha(approval_path) != approval_sha or time.time() + 600 >= DEADLINE
                or (logs / 'RESIDUAL_STOP.json').exists() or list(logs.glob('BUDGET_QUIESCED*'))):
            raise ValueError('STOP/approval/deadline prevents launch')
    check()
    current = plan(root, proposal['instance'], proposal['task'],
                   {int(k): v for k, v in proposal['device_uuids'].items()})
    if current != proposal:
        raise ValueError('Reviewed preflight changed')
    logs.mkdir(exist_ok=True)
    lock = (logs / 'residual-four.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    exclusive(logs / 'ASSIGNMENT.json', {**proposal, 'approval_sha256': approval_sha})
    owned, pending, observations = [], [], []
    try:
        for job in proposal['jobs']:
            check()
            with (logs / (job['launch_receipt'][:-5] + '.log')).open('xb') as output:
                child = subprocess.Popen(job['command'], cwd=root,
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(job['gpu']), 'JEPA_VERIFIED_LOCAL_DINO': '1'},
                    stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                pending.append(child)
            samples = []
            observations.append({'pid': child.pid, 'samples': samples})
            identity = wait_exact_identity(child, job['command'], samples)
            worker = {'pid': child.pid, **identity}
            owned.append(worker)
            pending.remove(child)
            exclusive(logs / job['launch_receipt'], {**job, **worker})
        argv = [sys.executable, '-u', str(Path(guard.__file__).resolve()), '--root', str(root),
                '--instance', str(proposal['instance']), '--freeze-sha256', FREEZE, '--generation', GENERATION]
        for key, value in BUDGET.items():
            argv += ['--' + key.replace('_', '-'), str(value)]
        with (logs / f'GUARD.{GENERATION}.log').open('xb') as output:
            watcher = subprocess.Popen(argv, cwd=root, stdin=subprocess.DEVNULL, stdout=output,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        path = logs / guard.guard_names(GENERATION)[1]
        until = time.monotonic() + 30
        while not path.exists():
            check()
            if watcher.poll() is not None or time.monotonic() >= until:
                raise ValueError('Four-lane guard failed to arm')
            time.sleep(.1)
        armed = read(path)
        if (armed['watchdog_pid'] != watcher.pid or guard.identity(watcher.pid) is None
                or armed['assignment_sha256'] != sha(logs / 'ASSIGNMENT.json')
                or armed['deadline_unix'] != DEADLINE or len(armed['workers']) != 4
                or any(not any(all(r.get(k) == w[k] for k in ('pid', 'command', 'start_tick'))
                                  for r in armed['workers']) for w in owned)):
            raise ValueError('Four-lane guard exact coverage/deadline differs')
        result = {'guard_armed': True, 'guard_pid': watcher.pid, 'guard_sha256': sha(path),
                  'worker_pids': [w['pid'] for w in owned], 'scientific_gate_pass_not_implied': True}
        exclusive(logs / f'RESIDUAL_MANAGED.{GENERATION}.json', result)
        return result
    except BaseException as error:
        exclusive(logs / f'RESIDUAL_FAILED.{GENERATION}.json', {**cleanup(owned, pending),
                  'error': str(error), 'identity_observations': observations})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--instance', type=int, required=True)
    parser.add_argument('--task', choices=('reach', 'reach-wall'), required=True)
    parser.add_argument('--approval', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--launch', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    uuids = {int(r.split(',')[0]): r.split(',')[1].strip().removeprefix('GPU-') for r in rows.splitlines()}
    proposal = plan(root, args.instance, args.task, uuids)
    if not args.launch:
        print(json.dumps({'dry_run': True, 'proposal': proposal}))
        return
    if args.approval is None:
        parser.error('--launch requires exact reviewed --approval')
    print(json.dumps(launch(root, proposal, args.approval)))


if __name__ == '__main__':
    main()
