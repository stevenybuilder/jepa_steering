"""Fixed, approved PRO MetaWorld waves; dry-run default, no rentals or analysis.

Four two-job waves use released navigation GPUs; two eight-job waves
follow. Every job invokes the unchanged frozen worker for one whole scenario.
Existing workers/guards are never signaled by this coordinator. Each new wave
gets its own exact eight-receipt guard; prior generations remain alive.
"""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import fresh_budget_watchdog as guard
from start_fresh_v2_navigation_handoff import (exclusive, read, sha, command,
                                               wait_exact_identity, reap_unregistered)

INSTANCE = 50806821
FREEZE = '7d5def0122f0dcf80e07e43ddc4f28ef6532fb04e6eb9acc4bc428427165ba90'
GENERATION = 'mw-residual24-v2'
DEADLINE_UTC = '2026-09-13T08:18:42.193091+00:00'
ARMS = ('native', 'fixed_rank4', 'matched_random_fixed_rank4', 'coupling_only',
        'matched_random_coupling', 'joint', 'visual_only', 'action_condition_only')
ORIGINAL = [[i, 'reach', i, 8] for i in range(3)] + [
    [i + 3, 'reach-wall', i, 8] for i in range(3)] + [[6, 'pointmaze', 0, 4], [7, 'wall', 0, 4]]
# Observed ~2,024 seconds/scenario; round upward and add startup/receipt margin.
SCENARIO_ALLOWANCE = 2100
FIRST_ENGINEERING_ALLOWANCE = 660
CLOSEOUT_ALLOWANCE = 300


class Paused(RuntimeError):
    pass


def wave_specs():
    remaining = [e for e in range(96) if e % 8 == 7]
    waves = []
    for index in range(4):
        waves.append({'number': index + 1, 'jobs': [
            {'gpu': 6, 'task': 'reach', 'episode': remaining[index]},
            {'gpu': 7, 'task': 'reach-wall', 'episode': remaining[index]}]})
    for index in range(2):
        jobs = []
        for task, gpus in (('reach', (0, 1, 2, 6)), ('reach-wall', (3, 4, 5, 7))):
            jobs += [{'gpu': gpu, 'task': task, 'episode': remaining[4 + index * 4 + j]}
                     for j, gpu in enumerate(gpus)]
        waves.append({'number': index + 5, 'jobs': jobs})
    return waves


def inspect_root(root, uuids, budget_receipt_name, budget_receipt_sha256):
    root = root.resolve()
    logs = root / 'worker-logs-v2'
    old = read(logs / 'ASSIGNMENT.json')
    budget_path = logs / guard.local_name(budget_receipt_name)
    if sha(budget_path) != budget_receipt_sha256:
        raise ValueError('Approved budget receipt SHA256 differs')
    armed = read(budget_path)
    if (old['instance'] != INSTANCE or old['freeze_sha256'] != FREEZE
            or old['mapping'] != ORIGINAL or sha(root / 'fresh-freeze-v2/protocol.json') != FREEZE
            or armed['instance'] != INSTANCE or armed['assignment_sha256'] != sha(logs / 'ASSIGNMENT.json')):
        raise ValueError('Original instance/assignment/freeze/guard differs')
    if (armed['deadline_utc'] != DEADLINE_UTC or armed['budget']['campaign_cap_usd'] != 220
            or armed['budget']['fleet_cap_usd_hour'] != 22
            or armed['budget']['closeout_reserve_usd'] < 5):
        raise ValueError('Unapproved budget/deadline override')
    calculated = guard.deadline(
        datetime.fromisoformat(armed['budget']['as_of_utc'].replace('Z', '+00:00')).timestamp(),
        armed['budget']['accrued_usd'], 22, 220, armed['budget']['closeout_reserve_usd'])
    if abs(calculated - armed['deadline_unix']) > .001:
        raise ValueError('Guard budget arithmetic differs')
    if set(uuids) != set(range(8)) or len(set(uuids.values())) != 8:
        raise ValueError('Eight distinct physical GPUs required')
    receipt_names = [f'{t}-worker{w}-LAUNCHED.json' for _, t, w, _ in ORIGINAL]
    original_workers = {Path(w['launch_receipt']).name: w for w in armed['workers']}
    if set(original_workers) != set(receipt_names):
        raise ValueError('Original guard receipt coverage differs')
    for name in receipt_names:
        if sha(logs / name) != original_workers[name]['launch_sha256']:
            raise ValueError('Original launch receipt changed')
    proposal = {'instance': INSTANCE, 'freeze_sha256': FREEZE, 'generation': GENERATION,
                'original_assignment_sha256': sha(logs / 'ASSIGNMENT.json'),
                'original_guard_sha256': budget_receipt_sha256,
                'budget_guard_receipt_name': budget_receipt_name,
                'device_uuids': {str(k): v for k, v in uuids.items()}, 'waves': wave_specs(),
                'deadline_utc': DEADLINE_UTC, 'campaign_cap_usd': 220, 'fleet_cap_usd_hour': 22,
                'external_reservation': {'instance': 50827072, 'tasks': ['reach', 'reach-wall'],
                                         'episode_modulus': 8, 'remainders': [3, 4, 5, 6]}}
    return proposal, armed, original_workers


def validate_approval(manifest, proposal):
    if (manifest.get('approved') is not True
            or manifest.get('globally_unstarted_and_unclaimed') is not True
            or manifest.get('other_residual24_launchers_disabled') is not True
            or manifest.get('proposal') != proposal):
        raise ValueError('Exact reviewed global ownership manifest required; no implicit remapping')


def check_unstarted(root, jobs):
    logs = root / 'worker-logs-v2'
    for job in jobs:
        if (root / 'results-v2' / job['task'] / f'scenario-{job["episode"]:03d}').exists():
            raise ValueError('Incoming scenario already has output; no automatic resume/retry')
        if list(logs.glob(f'MW_CLAIM.{job["task"]}.{job["episode"]:03d}.*.json')):
            raise ValueError('Incoming scenario already claimed')


def budget_gate(root, stop_at, allowance, *, stopped=False, now=None):
    now = time.time() if now is None else now
    logs = root / 'worker-logs-v2'
    if stopped or (logs / 'MW_WAVES_STOP.json').exists() or list(logs.glob('BUDGET_QUIESCED*.json')):
        raise Paused('STOP/cancel/budget quiescence: no further launches')
    if stop_at - now < allowance + CLOSEOUT_ALLOWANCE:
        raise Paused('Insufficient measured scenario plus closeout headroom; bound unchanged')


def verify_bundle(root, task, episode, uuid, engineering=False):
    protocol = read(root / 'fresh-freeze-v2/protocol.json')
    role = 'excluded_engineering' if engineering else 'scientific_candidates'
    expected = next(r for r in protocol['tasks'][task]['records'] if r['role'] == role and r['episode'] == episode)
    folder = (root / 'results-v2/engineering' / task / uuid if engineering else
              root / 'results-v2' / task / f'scenario-{episode:03d}')
    report = read(folder / 'report.json')
    if (sha(folder / 'report.json') != read(folder / 'DONE.json')['report_sha256']
            or report['task'] != task or report['scenario'] != expected
            or report['freeze_sha256'] != FREEZE or report['device_uuid'] != uuid
            or report['engineering'] != engineering or report['parameters_unchanged'] is not True):
        raise ValueError('Complete scenario metadata/hash identity differs')
    names = ('native', 'native_repeat') if engineering else ARMS
    if set(report['records_sha256']) != set(names):
        raise ValueError('Incomplete arm coverage')
    for arm in names:
        path = folder / (arm + '.json')
        if sha(path) != report['records_sha256'][arm]:
            raise ValueError('Changed arm bytes')
        record = read(path)
        if (record['arm'] != arm or record['device_uuid'] != uuid
                or record['freeze_sha256'] != FREEZE or record['scenario'] != expected
                or record['scientific_efficacy_measurement'] != (not engineering)):
            raise ValueError('Arm/input/device/role identity differs')
    if not engineering and report['engineering_report_sha256'] != verify_bundle(root, task, 0, uuid, True):
        raise ValueError('Engineering provenance differs')
    return sha(folder / 'report.json')


def wave_assignment(proposal, wave):
    number = wave['number']
    generation = f'{GENERATION}-w{number:02d}'
    jobs = [{**job, 'worker_id': job['episode'], 'workers': 96,
             'launch_receipt': f'{generation}-gpu{job["gpu"]}-LAUNCHED.json',
             'device_uuid': proposal['device_uuids'][str(job['gpu'])]} for job in wave['jobs']]
    retained = ORIGINAL[:6] if number <= 4 else []
    assignment = {'instance': INSTANCE, 'freeze_sha256': FREEZE,
        'mapping': [*retained, *[[j['gpu'], j['task'], j['episode'], 96] for j in jobs]],
        'launch_receipts': [*[f'{t}-worker{w}-LAUNCHED.json' for _, t, w, _ in retained],
                            *[j['launch_receipt'] for j in jobs]], 'wave': wave,
        'device_uuids': proposal['device_uuids']}
    return generation, jobs, assignment


def cleanup(owned, pending):
    unregistered = []
    for child in pending:
        try:
            unregistered.append(reap_unregistered(child))
        except Exception as error:
            unregistered.append({'pid': child.pid, 'cleanup_error': str(error), 'requires_operator_attention': True})
    return {**guard.quiesce(owned), 'unregistered_children': unregistered}


def spawn_jobs(root, jobs, owned, pending, check, observations):
    logs = root / 'worker-logs-v2'
    children = []
    for job in jobs:
        check()
        argv = command(root, job['task'], job['episode'], 96)
        with (logs / (job['launch_receipt'][:-5] + '.log')).open('xb') as output:
            child = subprocess.Popen(argv, cwd=root,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(job['gpu']), 'JEPA_VERIFIED_LOCAL_DINO': '1'},
                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            pending.append(child)
        samples = []
        observations.append({'pid': child.pid, 'samples': samples})
        current = wait_exact_identity(child, argv, samples)
        record = {**job, 'pid': child.pid, **current}
        owned.append({k: record[k] for k in ('pid', 'command', 'start_tick')})
        pending.remove(child)
        children.append((child, job))
        exclusive(logs / job['launch_receipt'], {**record, 'identity_observations': samples})
    return children


def arm_wave(root, generation, assignment, prior, owned, check):
    logs = root / 'worker-logs-v2'
    name = 'ACTIVE_ASSIGNMENT.' + generation + '.json'
    exclusive(logs / name, assignment)
    argv = [sys.executable, '-u', str(Path(guard.__file__).resolve()), '--root', str(root),
            '--instance', str(INSTANCE), '--freeze-sha256', FREEZE,
            '--assignment-name', name, '--generation', generation]
    for key in ('as_of_utc', 'accrued_usd', 'fleet_cap_usd_hour', 'campaign_cap_usd', 'closeout_reserve_usd'):
        argv += ['--' + key.replace('_', '-'), str(prior['budget'][key])]
    with (logs / ('GUARD.' + generation + '.log')).open('xb') as output:
        watcher = subprocess.Popen(argv, cwd=root, stdin=subprocess.DEVNULL,
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    path = logs / guard.guard_names(generation)[1]
    end = time.monotonic() + 30
    while not path.exists():
        check()
        if watcher.poll() is not None or time.monotonic() >= end:
            raise RuntimeError('Wave guard failed to arm')
        time.sleep(.1)
    receipt = read(path)
    if (receipt['watchdog_pid'] != watcher.pid or guard.identity(watcher.pid) is None
            or receipt['assignment_sha256'] != sha(logs / name)
            or receipt['deadline_unix'] != prior['deadline_unix']
            or any(not any(all(r.get(k) == w[k] for k in ('pid', 'command', 'start_tick'))
                           for r in receipt['workers']) for w in owned)):
        raise ValueError('Wave guard coverage/deadline differs')
    return {'path': str(path), 'sha256': sha(path), 'pid': watcher.pid}


def run(root, proposal, prior, originals, manifest_path):
    logs = root / 'worker-logs-v2'
    manifest_bytes = manifest_path.read_bytes()
    validate_approval(json.loads(manifest_bytes), proposal)
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    prior_identity = guard.identity(prior['watchdog_pid'])
    if (prior_identity is None or str(root) not in prior_identity['command']
            or not any(Path(arg).name == 'fresh_budget_watchdog.py' for arg in prior_identity['command'])):
        raise ValueError('Original watchdog is not live with the expected command')
    stopped = False
    def cancel(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    def check(allowance=0):
        budget_gate(root, prior['deadline_unix'], allowance, stopped=stopped)
        if sha(manifest_path) != manifest_digest:
            raise Paused('Approved manifest changed; no further launches')
        if guard.identity(prior['watchdog_pid']) != prior_identity:
            raise Paused('Original guard identity changed; reconcile before continuing')
    lock = (logs / (GENERATION + '.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    exclusive(logs / ('CONTROLLER.' + GENERATION + '.json'),
              {'proposal': proposal, 'manifest_sha256': manifest_digest,
               'pid': os.getpid(), 'identity': guard.identity(os.getpid()),
               'original_guard_identity': prior_identity})
    check_unstarted(root, [job for wave in proposal['waves'] for job in wave['jobs']])
    owned, pending, observations = [], [], []
    completed = []
    def wait_original(gpus):
        for gpu in gpus:
            _, task, worker, count = ORIGINAL[gpu]
            old = originals[f'{task}-worker{worker}-LAUNCHED.json']
            while True:
                check()
                current = guard.identity(old['pid'])
                if current is None:
                    break
                if current != {'command': old['command'], 'start_tick': old['start_tick']}:
                    raise ValueError('Original worker identity changed')
                time.sleep(1)
            for episode in range(worker, 96, count):
                verify_bundle(root, task, episode, proposal['device_uuids'][str(gpu)])
    try:
        wait_original((6, 7))
        for wave in proposal['waves']:
            if wave['number'] == 5:
                wait_original(range(6))
            allowance = SCENARIO_ALLOWANCE + (FIRST_ENGINEERING_ALLOWANCE if wave['number'] == 1 else 0)
            check(allowance)
            generation, jobs, assignment = wave_assignment(proposal, wave)
            check_unstarted(root, jobs)
            for job in jobs:
                exclusive(logs / f'MW_CLAIM.{job["task"]}.{job["episode"]:03d}.{generation}.json',
                          {**job, 'manifest_sha256': manifest_digest, 'freeze_sha256': FREEZE})
            owned, pending, observations = [], [], []
            children = spawn_jobs(root, jobs, owned, pending, lambda: check(allowance), observations)
            armed = arm_wave(root, generation, assignment, prior, owned, check)
            proofs = {}
            while children:
                check()
                for child, job in list(children):
                    code = child.poll()
                    if code is None:
                        continue
                    if code != 0:
                        raise RuntimeError(f'Frozen worker exited {code}; no retry or gate change')
                    proofs[f'{job["task"]}/{job["episode"]}'] = verify_bundle(
                        root, job['task'], job['episode'], job['device_uuid'])
                    children.remove((child, job))
                if children:
                    time.sleep(1)
            exclusive(logs / ('WAVE_DONE.' + generation + '.json'),
                      {'wave': wave, 'source_report_sha256': proofs, 'guard': armed,
                       'efficacy_values_not_reported': True})
            completed.append(wave['number'])
            owned = []
        exclusive(logs / ('WAVES_DONE.' + GENERATION + '.json'),
                  {'completed_waves': completed, 'new_scenarios': 24,
                   'cloud_verified': False, 'whole_campaign_completion_not_implied': True})
    except BaseException as error:
        result = cleanup(owned, pending)
        exclusive(logs / ('WAVES_PAUSED.' + GENERATION + '.json'),
                  {**result, 'error_type': type(error).__name__, 'error': str(error),
                   'completed_waves': completed, 'identity_observations': observations,
                   'original_workers_signaled_by_controller': False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--budget-receipt-name', required=True)
    parser.add_argument('--budget-receipt-sha256', required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.run:
        parser.error('Execution remains disabled pending root review and residual24 global approval. '
                     'The retired 120-scenario allocation must never run; 96 scenarios remain '
                     'reserved for instance 50827072.')
    root = args.root.resolve()
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    uuids = {int(row.split(',')[0]): row.split(',')[1].strip().removeprefix('GPU-') for row in output.splitlines()}
    proposal, prior, originals = inspect_root(root, uuids, args.budget_receipt_name,
                                             args.budget_receipt_sha256)
    check_unstarted(root, [job for wave in proposal['waves'] for job in wave['jobs']])
    if args.manifest is not None:
        validate_approval(read(args.manifest), proposal)
    if not args.run:
        print(json.dumps({'dry_run': True, 'proposal': proposal, 'original_completion_gates_not_yet_implied': True}))
        return
    if args.manifest is None:
        parser.error('--run requires exact approved --manifest')
    run(root, proposal, prior, originals, args.manifest)


if __name__ == '__main__':
    main()
