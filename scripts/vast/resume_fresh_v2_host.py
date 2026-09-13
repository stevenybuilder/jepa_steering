"""Same-host frozen-work resume after provider stop; dry-run default, no rentals.

Only the reviewed prior GPU/task/stripe assignment may resume. Atomic completed
arms and receiving bundles are revalidated by the unchanged frozen worker.
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
from start_fresh_v2_navigation_handoff import exclusive, read, sha, command, wait_exact_identity
from start_fresh_v2_mw_waves import cleanup

FREEZE = '7d5def0122f0dcf80e07e43ddc4f28ef6532fb04e6eb9acc4bc428427165ba90'
SOURCE = 'cde8274dad2bbc91efea3c5baba2e2266217949f9993aa77ce695d3bbabd4d82'
LATEST = datetime.fromisoformat('2026-09-13T05:33:45.447806+00:00').timestamp()
HOSTS = {50806821: ('ASSIGNMENT.json', set(range(8))),
         50819364: ('ACTIVE_ASSIGNMENT.navigation-handoff-v2.json', {1, 3, 4, 5, 6, 7}),
         50827072: ('ACTIVE_ASSIGNMENT.mw-third-split24-v1.json', set(range(8))),
         50828584: ('ASSIGNMENT.json', set(range(8))),
         50828583: ('ASSIGNMENT.json', set(range(8))),
         50836730: ('ASSIGNMENT.json', set(range(4)))}


def duplicate_workers(root, proc=Path('/proc')):
    matches = []
    for path in proc.iterdir():
        if not path.name.isdigit():
            continue
        current = guard.identity(int(path.name), proc)
        if current and str(root / 'scripts/run_fresh_confirmation.py') in current['command'] and 'worker' in current['command']:
            matches.append({'pid': int(path.name), **current})
    return matches


def validate_files(root):
    subprocess.run([sys.executable, str(root / 'scripts/run_fresh_confirmation.py'), 'validate',
                    '--project', str(root), '--freeze', str(root / 'fresh-freeze-v2')],
                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)


def plan(root, instance, assignment_name, assignment_sha, budget_name, budget_sha,
         uuids, generation, boot_id, *, inspect=guard.identity, scan=duplicate_workers, validate=validate_files):
    logs = root / 'worker-logs-v2'
    if instance not in HOSTS or assignment_name != HOSTS[instance][0]:
        raise ValueError('Unapproved instance/active assignment')
    guard.guard_names(generation)
    if not generation.startswith('resume-'):
        raise ValueError('New resume generation required')
    apath, bpath = logs / guard.local_name(assignment_name), logs / guard.local_name(budget_name)
    if sha(apath) != assignment_sha or sha(bpath) != budget_sha:
        raise ValueError('Exact approved assignment/budget receipt SHA required')
    assignment, prior = read(apath), read(bpath)
    protocol = read(root / 'fresh-freeze-v2/protocol.json')
    if (sha(root / 'fresh-freeze-v2/protocol.json') != FREEZE or protocol['source_sha256'] != SOURCE
            or assignment['instance'] != instance or assignment['freeze_sha256'] != FREEZE
            or prior['instance'] != instance or prior['assignment_sha256'] != assignment_sha):
        raise ValueError('Instance/freeze/source/guard binding differs')
    budget = prior['budget']
    until = guard.deadline(datetime.fromisoformat(budget['as_of_utc'].replace('Z', '+00:00')).timestamp(),
        budget['accrued_usd'], budget['fleet_cap_usd_hour'], budget['campaign_cap_usd'], budget['closeout_reserve_usd'])
    if (budget['campaign_cap_usd'] != 220 or budget['fleet_cap_usd_hour'] != 35
            or abs(until - prior['deadline_unix']) > .001 or until > LATEST + .001):
        raise ValueError('Unapproved budget/deadline extension')
    mapping = assignment['mapping']
    expected_count = 4 if instance == 50836730 else 8
    if (assignment.get('expected_gpu_count', 8) != expected_count
            or len(mapping) != expected_count or {r[0] for r in mapping} != set(range(expected_count))
            or set(uuids) != set(range(expected_count)) or len(set(uuids.values())) != expected_count):
        raise ValueError('Expected unique lanes/physical GPU UUIDs required')
    if instance == 50836730 and mapping != [[g, 'reach-wall', w, 32] for g, w in enumerate((7, 15, 23, 31))]:
        raise ValueError('Four-GPU replacement must retain exact residual Reach-Wall assignment')
    if scan(root):
        raise ValueError('Existing scoped worker process: no duplicate resume')
    old_by_name = {Path(r['launch_receipt']).name: r for r in prior['workers']}
    names = assignment.get('launch_receipts', [f'{t}-worker{w}-LAUNCHED.json' for _, t, w, _ in mapping])
    if len(names) != expected_count or set(names) != set(old_by_name):
        raise ValueError('Prior guard must bind exactly the selected lane receipts')
    for name in [f'RESUME.{generation}.json', f'ACTIVE_ASSIGNMENT.{generation}.json', *guard.guard_names(generation)]:
        if (logs / name).exists():
            raise ValueError('Existing resume generation; no automatic retry')
    jobs, selected, proofs = [], [], []
    for gpu, task, worker, count in mapping:
        candidates = [n for n in names if [read(logs / n)[k] for k in ('gpu', 'task', 'worker_id', 'workers')]
                      == [gpu, task, worker, count]]
        if len(candidates) != 1:
            raise ValueError('Missing/duplicate exact lane receipt')
        name = candidates[0]
        old = read(logs / name)
        if (sha(logs / name) != old_by_name[name]['launch_sha256']
                or old['command'][1:] != command(root, task, worker, count)[1:]
                or Path(old['command'][0]).resolve() != Path(sys.executable).resolve()):
            raise ValueError('Old receipt/interpreter/exact frozen command differs')
        # A reused PID is not evidence the old process survived; fail closed and reconcile it.
        if inspect(old['pid']) is not None:
            raise ValueError('Old PID live or reused; exact identity requires reconciliation')
        uuid = uuids[gpu]
        if str(gpu) in assignment.get('device_uuids', {}) and assignment['device_uuids'][str(gpu)] != uuid:
            raise ValueError('Physical hardware UUID differs')
        if old.get('device_uuid', uuid) != uuid:
            raise ValueError('Historical launch hardware UUID differs')
        folders = [root / 'results-v2/engineering' / task / uuid]
        folders += [root / 'results-v2' / task / f'scenario-{e:03d}' for e in range(worker, 96, count)]
        starts = {}
        for folder in folders:
            if not folder.exists():
                continue
            intent = read(folder / 'STARTED.json')
            engineering = folder == folders[0]
            expected_rows = [r for r in protocol['tasks'][task]['records']
                             if r['role'] == ('excluded_engineering' if engineering else 'scientific_candidates')]
            if (intent['task'] != task or intent['device_uuid'] != uuid or intent['freeze_sha256'] != FREEZE
                    or intent['engineering'] != engineering or intent['scenario'] not in expected_rows
                    or (engineering and intent['scenario']['episode'] != 0)
                    or (not engineering and folder.name != f'scenario-{intent["scenario"]["episode"]:03d}')):
                raise ValueError('Existing input/partial/physical GPU identity differs')
            starts[str(folder.relative_to(root))] = sha(folder / 'STARTED.json')
        # Every original launched lane entered receiving; absent UUID-bound history is not accepted.
        if not (folders[0] / 'STARTED.json').exists():
            raise ValueError('No historical receiving UUID binding')
        resume = gpu in HOSTS[instance][1]
        if instance == 50819364 and resume and task not in ('pointmaze', 'wall'):
            raise ValueError('Failed A100 MetaWorld lanes must remain stopped')
        new_name = f'{generation}-gpu{gpu}-LAUNCHED.json' if resume else name
        if resume and any((logs / n).exists() for n in (new_name, new_name[:-5] + '.log')):
            raise ValueError('Existing resume launch log/receipt')
        selected.append(new_name)
        if resume:
            jobs.append({'gpu': gpu, 'task': task, 'worker_id': worker, 'workers': count,
                         'command': old['command'], 'device_uuid': uuid, 'launch_receipt': new_name})
        proofs.append({'gpu': gpu, 'old_launch_receipt': name, 'sha256': sha(logs / name),
                       'old_pid': old['pid'], 'old_start_tick': old.get('start_tick'),
                       'resume': resume, 'started_sha256': starts})
    validate(root)
    proposal = {'instance': instance, 'freeze_sha256': FREEZE, 'source_sha256': SOURCE,
        'generation': generation, 'boot_id': boot_id, 'mapping': mapping,
        'device_uuids': {str(k): v for k, v in uuids.items()}, 'jobs': jobs, 'launch_receipts': selected,
        'original_assignment_name': assignment_name, 'original_assignment_sha256': assignment_sha,
        'budget_receipt_name': budget_name, 'budget_receipt_sha256': budget_sha,
        'deadline_unix': until, 'history': proofs}
    if expected_count == 4:
        proposal['expected_gpu_count'] = 4
    return proposal, prior


def approve(path, proposal):
    record = read(path)
    if (record.get('approved') is not True or record.get('proposal') != proposal
            or record.get('global_same_assignment_resume_verified') is not True
            or record.get('provider_resume_authorized') is not True):
        raise ValueError('Exact reviewed same-assignment resume approval required')
    return sha(path)


def launch(root, proposal, prior, approval_path):
    logs, generation = root / 'worker-logs-v2', proposal['generation']
    approved_sha = approve(approval_path, proposal)
    def check():
        if (sha(approval_path) != approved_sha or time.time() + 600 >= prior['deadline_unix']
                or (logs / 'RESUME_STOP.json').exists() or list(logs.glob('BUDGET_QUIESCED*.json'))):
            raise ValueError('STOP/approval/deadline precludes resume')
    check()
    current, _ = plan(root, proposal['instance'], proposal['original_assignment_name'],
        proposal['original_assignment_sha256'], proposal['budget_receipt_name'], proposal['budget_receipt_sha256'],
        {int(k): v for k, v in proposal['device_uuids'].items()}, generation,
        Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    if current != proposal:
        raise ValueError('Resume preflight changed')
    lock = (logs / f'{generation}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    exclusive(logs / f'RESUME.{generation}.json', {'proposal': proposal, 'approval_sha256': approved_sha})
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
            exact = wait_exact_identity(child, job['command'], samples)
            worker = {'pid': child.pid, **exact}
            owned.append(worker)
            pending.remove(child)
            exclusive(logs / job['launch_receipt'], {**job, **worker, 'boot_id': proposal['boot_id']})
        # Historical inactive PID rows must still be absent when the guard snapshots them.
        if any(guard.identity(r['old_pid']) is not None for r in proposal['history'] if not r['resume']):
            raise ValueError('Inactive historical PID reused; refuse ambiguous guard identity')
        active = f'ACTIVE_ASSIGNMENT.{generation}.json'
        exclusive(logs / active, proposal)
        argv = [sys.executable, '-u', str(Path(guard.__file__).resolve()), '--root', str(root),
            '--instance', str(proposal['instance']), '--freeze-sha256', FREEZE,
            '--assignment-name', active, '--generation', generation]
        for key in ('as_of_utc', 'accrued_usd', 'fleet_cap_usd_hour', 'campaign_cap_usd', 'closeout_reserve_usd'):
            argv += ['--' + key.replace('_', '-'), str(prior['budget'][key])]
        with (logs / f'GUARD.{generation}.log').open('xb') as output:
            watcher = subprocess.Popen(argv, cwd=root, stdin=subprocess.DEVNULL, stdout=output,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        path = logs / guard.guard_names(generation)[1]
        until = time.monotonic() + 30
        while not path.exists():
            check()
            if watcher.poll() is not None or time.monotonic() >= until:
                raise ValueError('Resume guard failed to arm')
            time.sleep(.1)
        armed = read(path)
        if (armed['watchdog_pid'] != watcher.pid or guard.identity(watcher.pid) is None
                or armed['assignment_sha256'] != sha(logs / active)
                or len(armed['workers']) != proposal.get('expected_gpu_count', 8)
                or armed['deadline_unix'] != prior['deadline_unix']
                or any(not any(all(r.get(k) == w[k] for k in ('pid', 'command', 'start_tick'))
                              for r in armed['workers']) for w in owned)):
            raise ValueError('Resume guard exact coverage/deadline differs')
        result = {'guard_armed': True, 'guard_sha256': sha(path), 'guard_pid': watcher.pid,
                  'worker_pids': [w['pid'] for w in owned], 'old_receipts_preserved': True}
        exclusive(logs / f'RESUME_MANAGED.{generation}.json', result)
        return result
    except BaseException as error:
        exclusive(logs / f'RESUME_FAILED.{generation}.json', {**cleanup(owned, pending),
            'error': str(error), 'observations': observations})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--instance', type=int, required=True)
    for name in ('assignment-name', 'assignment-sha256', 'budget-receipt-name', 'budget-receipt-sha256', 'generation'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--approval', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--launch', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    uuids = {int(r.split(',')[0]): r.split(',')[1].strip().removeprefix('GPU-') for r in rows.splitlines()}
    proposal, prior = plan(root, args.instance, args.assignment_name, args.assignment_sha256,
        args.budget_receipt_name, args.budget_receipt_sha256, uuids, args.generation,
        Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    if not args.launch:
        print(json.dumps({'dry_run': True, 'proposal': proposal}))
        return
    if args.approval is None:
        parser.error('--launch requires --approval')
    print(json.dumps(launch(root, proposal, prior, args.approval)))


if __name__ == '__main__':
    main()
