"""Root-approved static whole-scenario queues; no donor actions or scheduling search.

Independent GPU lanes invoke the unchanged frozen worker once per scenario.
Every child gets its own one-lane identity-bound budget guard. Default is dry-run.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import fresh_budget_watchdog as guard
from start_fresh_v2_navigation_handoff import exclusive, read, sha, command, wait_exact_identity
from start_fresh_v2_mw_waves import cleanup, verify_bundle
from resume_fresh_v2_host import validate_files, FREEZE, SOURCE
from start_fresh_v2_residual_host import BUDGET, DEADLINE
from rebase_fresh_budget import load_budget


def selected_busy(root, selected, uuids, proc=Path('/proc')):
    """Reject selected-GPU activity, including workers between CUDA calls."""
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                      '--format=csv,noheader'], text=True)
    for row in output.splitlines():
        _, uuid = row.split(',', 1)
        if uuid.strip().removeprefix('GPU-') in {uuids[g] for g in selected}:
            return True
    for path in proc.iterdir():
        if not path.name.isdigit():
            continue
        identity = guard.identity(int(path.name), proc)
        if not identity or str(root / 'scripts/run_fresh_confirmation.py') not in identity['command'] or 'worker' not in identity['command']:
            continue
        try:
            env = dict(v.split('=', 1) for v in (path / 'environ').read_text().split('\0') if '=' in v)
        except FileNotFoundError:
            continue
        visible = env.get('CUDA_VISIBLE_DEVICES', '')
        if visible.isdigit():
            if int(visible) in selected:
                return True
        elif visible.removeprefix('GPU-') in set(uuids.values()):
            if visible.removeprefix('GPU-') in {uuids[g] for g in selected}:
                return True
        else:
            raise ValueError('Scoped worker has ambiguous CUDA device mapping')
    return False


def input_identity(root, task, episode, uuid, allowed_partial):
    folder = root / 'results-v2' / task / f'scenario-{episode:03d}'
    if not folder.exists():
        if allowed_partial is not None:
            raise ValueError('Approved partial disappeared')
        return None
    if allowed_partial is None:
        raise ValueError('Incoming scenario already exists without explicit same-GPU approval')
    path = folder / 'STARTED.json'
    row = next(r for r in read(root / 'fresh-freeze-v2/protocol.json')['tasks'][task]['records']
               if r['role'] == 'scientific_candidates' and r['episode'] == episode)
    if sha(path) != allowed_partial or read(path) != {'task': task, 'scenario': row,
            'device_uuid': uuid, 'freeze_sha256': FREEZE, 'engineering': False}:
        raise ValueError('Partial scenario input/UUID/hash differs')
    return allowed_partial


def job_id(generation, gpu, episode):
    value = f'{generation}-g{gpu}-e{episode:02d}'
    guard.guard_names(value)
    return value


def plan(root, request, uuids, boot_id, *, validate=validate_files, busy=selected_busy,
         budget_receipt=None, budget_sha=None):
    if (budget_receipt is None) != (budget_sha is None):
        raise ValueError('Budget receipt and SHA must be provided together')
    budget, deadline, binding = BUDGET, DEADLINE, None
    if budget_receipt is not None:
        budget, deadline, binding = load_budget(budget_receipt, budget_sha)
    if type(request.get('instance')) is not int or request['instance'] <= 0:
        raise ValueError('Explicit receiver instance required')
    generation = request['generation']
    if not generation.startswith('queue-'):
        raise ValueError('Queue generation required')
    guard.guard_names(generation)
    queues = request['queues']
    selected = [q['gpu'] for q in queues]
    if (not queues or len(selected) != len(set(selected)) or any(type(g) is not int or g not in uuids for g in selected)
            or len(set(uuids.values())) != len(uuids)):
        raise ValueError('Distinct existing physical GPU lanes required')
    protocol = read(root / 'fresh-freeze-v2/protocol.json')
    if sha(root / 'fresh-freeze-v2/protocol.json') != FREEZE or protocol['source_sha256'] != SOURCE:
        raise ValueError('Frozen source/protocol changed')
    logs = root / 'worker-logs-v2'
    if (logs / f'QUEUE_STARTED.{generation}.json').exists():
        raise ValueError('Queue generation already attempted; no automatic retry')
    if busy(root, selected, uuids):
        raise ValueError('Selected GPU still busy; donor quiescence is external')
    cases, planned = set(), []
    for queue in queues:
        gpu, task, episodes = queue['gpu'], queue['task'], queue['episodes']
        partials = queue.get('same_uuid_partials', {})
        if (task not in protocol['tasks'] or queue['device_uuid'] != uuids[gpu] or not episodes
                or set(partials) - {str(e) for e in episodes}):
            raise ValueError('Task/UUID/partial approval differs')
        jobs = []
        for episode in episodes:
            if type(episode) is not int or not 0 <= episode < 96 or (task, episode) in cases:
                raise ValueError('Invalid or duplicate scientific scenario')
            cases.add((task, episode))
            input_identity(root, task, episode, uuids[gpu], partials.get(str(episode)))
            name = job_id(generation, gpu, episode)
            if (list(logs.glob(f'STATIC_CLAIM.{task}.{episode:03d}.*.json'))
                    or list(logs.glob(f'MW_CLAIM.{task}.{episode:03d}.*.json'))
                    or any((logs / path).exists() for path in [f'{name}-LAUNCHED.json', f'{name}.log',
                           f'ASSIGNMENT.{name}.json', *guard.guard_names(name)])):
                raise ValueError('Incoming scenario already claimed or job generation exists')
            jobs.append({'gpu': gpu, 'task': task, 'episode': episode, 'worker_id': episode, 'workers': 96,
                'device_uuid': uuids[gpu], 'generation': name, 'launch_receipt': f'{name}-LAUNCHED.json',
                'same_uuid_partial_sha256': partials.get(str(episode))})
        planned.append({'gpu': gpu, 'task': task, 'jobs': jobs})
    validate(root)
    claims = [{'task': j['task'], 'episode': j['episode'], 'instance': request['instance'],
               'gpu': j['gpu'], 'device_uuid': j['device_uuid']} for q in planned for j in q['jobs']]
    proposal = {'instance': request['instance'], 'generation': generation, 'boot_id': boot_id,
            'freeze_sha256': FREEZE, 'source_sha256': SOURCE, 'queues': planned,
            'request': request, 'device_uuids': {str(k): v for k, v in uuids.items()},
            'global_claims': claims, 'budget': budget, 'deadline_unix': deadline}
    if binding is not None:
        proposal['budget_binding'] = binding
    return proposal


def approve(path, proposal):
    record = read(path)
    if (record.get('approved') is not True or record.get('proposal') != proposal
            or record.get('donors_quiesced_and_started_reconciled') is not True
            or record.get('global_claims') != proposal['global_claims']
            or record.get('competing_launchers_disabled_for_claims') is not True):
        raise ValueError('Exact root approval, reconciled donors and global claims required')
    return sha(path)


def assignment(proposal, job):
    return {'instance': proposal['instance'], 'freeze_sha256': FREEZE, 'expected_gpu_count': 1,
            'mapping': [[job['gpu'], job['task'], job['episode'], 96]],
            'launch_receipts': [job['launch_receipt']], 'boot_id': proposal['boot_id'],
            'device_uuids': proposal['device_uuids']}


def start_job(root, proposal, job, owned, pending, observations, check):
    logs, generation = root / 'worker-logs-v2', job['generation']
    receiving = root / 'results-v2/engineering' / job['task'] / job['device_uuid'] / 'DONE.json'
    check(allowance=(2100 if job['task'].startswith('reach') else 1200)
                   + (0 if receiving.exists() else 660))
    input_identity(root, job['task'], job['episode'], job['device_uuid'], job['same_uuid_partial_sha256'])
    exclusive(logs / f'STATIC_CLAIM.{job["task"]}.{job["episode"]:03d}.{proposal["generation"]}.json',
              {**job, 'approval_global_claim': True})
    argv = command(root, job['task'], job['episode'], 96)
    with (logs / f'{generation}.log').open('xb') as output:
        child = subprocess.Popen(argv, cwd=root,
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(job['gpu']), 'JEPA_VERIFIED_LOCAL_DINO': '1'},
            stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        pending.append(child)
    samples = []
    observations.append({'pid': child.pid, 'samples': samples})
    exact = wait_exact_identity(child, argv, samples)
    worker = {'pid': child.pid, **exact}
    owned.append(worker)
    pending.remove(child)
    exclusive(logs / job['launch_receipt'], {**job, **worker, 'boot_id': proposal['boot_id']})
    aname = f'ASSIGNMENT.{generation}.json'
    exclusive(logs / aname, assignment(proposal, job))
    gargv = [sys.executable, '-u', str(Path(guard.__file__).resolve()), '--root', str(root),
             '--instance', str(proposal['instance']), '--freeze-sha256', FREEZE,
             '--assignment-name', aname, '--generation', generation]
    for key, value in proposal['budget'].items():
        gargv += ['--' + key.replace('_', '-'), str(value)]
    with (logs / f'GUARD.{generation}.log').open('xb') as output:
        watcher = subprocess.Popen(gargv, cwd=root, stdin=subprocess.DEVNULL, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    path = logs / guard.guard_names(generation)[1]
    until = time.monotonic() + 30
    while not path.exists():
        check()
        if watcher.poll() is not None or time.monotonic() >= until:
            raise ValueError('Per-job guard failed to arm')
        time.sleep(.1)
    armed = read(path)
    if (armed['watchdog_pid'] != watcher.pid or guard.identity(watcher.pid) is None
            or armed['assignment_sha256'] != sha(logs / aname) or armed['deadline_unix'] != proposal['deadline_unix']
            or len(armed['workers']) != 1 or any(armed['workers'][0].get(k) != worker[k]
                                               for k in ('pid', 'command', 'start_tick'))):
        raise ValueError('Per-job guard exact identity/coverage/deadline differs')
    return {'child': child, 'job': job, 'worker': worker, 'guard_sha256': sha(path)}


def run_lanes(queues, start, finish, check, sleep=time.sleep):
    """At most one live scenario per GPU; a completed lane advances independently."""
    lanes = {q['gpu']: {'todo': list(q['jobs']), 'active': None} for q in queues}
    while any(lane['todo'] or lane['active'] is not None for lane in lanes.values()):
        check()
        for gpu, lane in lanes.items():
            if lane['active'] is not None:
                code = lane['active']['child'].poll()
                if code is not None:
                    if code != 0:
                        raise ValueError(f'Frozen worker exited {code}; no automatic retry')
                    finish(lane['active'])
                    lane['active'] = None
            if lane['active'] is None and lane['todo']:
                lane['active'] = start(lane['todo'].pop(0))
        if any(lane['active'] is not None for lane in lanes.values()):
            sleep(.5)


def launch(root, proposal, approval_path):
    logs = root / 'worker-logs-v2'
    approved_sha = approve(approval_path, proposal)
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    old_handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    def check(allowance=0):
        if (stopped or sha(approval_path) != approved_sha or time.time() + allowance + 300 >= proposal['deadline_unix']
                or ('budget_binding' in proposal and sha(Path(proposal['budget_binding']['path'])) != proposal['budget_binding']['sha256'])
                or (logs / 'STATIC_QUEUE_STOP.json').exists() or list(logs.glob('BUDGET_QUIESCED*.json'))):
            raise ValueError('STOP/approval/budget: no further queue launches')
    owned, pending, observations, complete = [], [], [], []
    locks, claimed = [], False
    try:
        check()
        budget_options = ({'budget_receipt': proposal['budget_binding']['path'],
                           'budget_sha': proposal['budget_binding']['sha256']} if 'budget_binding' in proposal else {})
        current = plan(root, proposal['request'], {int(k): v for k, v in proposal['device_uuids'].items()},
                       Path('/proc/sys/kernel/random/boot_id').read_text().strip(), **budget_options)
        if current != proposal:
            raise ValueError('Prelaunch queue proposal changed')
        for q in sorted(proposal['queues'], key=lambda q: q['gpu']):
            lock = (logs / f'STATIC_QUEUE_GPU{q["gpu"]}.lock').open('a')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(lock)
        for claim in sorted(proposal['global_claims'], key=lambda c: (c['task'], c['episode'])):
            lock = (logs / f'STATIC_CASE.{claim["task"]}.{claim["episode"]:03d}.lock').open('a')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(lock)
        exclusive(logs / f'QUEUE_STARTED.{proposal["generation"]}.json',
                  {'proposal': proposal, 'approval_sha256': approved_sha, 'controller_pid': os.getpid(),
                   'controller_identity': guard.identity(os.getpid())})
        claimed = True
        def start(job):
            return start_job(root, proposal, job, owned, pending, observations, check)
        def finish(active):
            j = active['job']
            report_sha = verify_bundle(root, j['task'], j['episode'], j['device_uuid'])
            exclusive(logs / f'QUEUE_JOB_DONE.{j["generation"]}.json',
                      {'job': j, 'report_sha256': report_sha, 'guard_sha256': active['guard_sha256']})
            owned.remove(active['worker'])
            complete.append({'task': j['task'], 'episode': j['episode']})
        run_lanes(proposal['queues'], start, finish, check)
        result = {'completed': complete, 'cloud_verified': False, 'whole_campaign_completion_not_implied': True}
        exclusive(logs / f'QUEUE_DONE.{proposal["generation"]}.json', result)
        return result
    except BaseException as error:
        # No scientific or operational write before QUEUE_STARTED is needed for failed dry preflight.
        if claimed:
            exclusive(logs / f'QUEUE_FAILED.{proposal["generation"]}.json',
                      {**cleanup(owned, pending), 'error': str(error), 'completed': complete,
                       'identity_observations': observations, 'donors_signaled': False})
        raise
    finally:
        for sig, previous in old_handlers.items():
            signal.signal(sig, previous)
        for lock in locks:
            lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--approval', type=Path)
    parser.add_argument('--budget-receipt', type=Path)
    parser.add_argument('--budget-sha')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--launch', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    uuids = {int(r.split(',')[0]): r.split(',')[1].strip().removeprefix('GPU-') for r in rows.splitlines()}
    proposal = plan(root, read(args.request), uuids, Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                    budget_receipt=args.budget_receipt, budget_sha=args.budget_sha)
    if not args.launch:
        print(json.dumps({'dry_run': True, 'proposal': proposal}))
        return
    if args.approval is None:
        parser.error('--launch requires exact reviewed --approval')
    print(json.dumps(launch(root, proposal, args.approval)))


if __name__ == '__main__':
    main()
