"""Prepare/activate an operations-only HMM priority scheduler; no provider mutations."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import routing_priority_common as c


def original_parent(gpu):
    binding = c.process(5202 + gpu)
    expected = [c.PYTHON, '-u', str(c.ORIGINAL), '--gpu', str(gpu)]
    if binding is None or binding['state'] in ('Z', 'X', 'T', 't') or binding['command'] != expected:
        raise ValueError('Original live parent identity or state changed')
    return binding


def waiting_only():
    if (c.PANEL / 'engineering').exists() or any(c.PANEL.glob('*/*/shard-*')):
        raise ValueError('HMM GPU work may already exist; no supervisor replacement')
    bindings = []
    for gpu in range(8):
        folder = c.PANEL / f'queue-gpu{gpu}'
        if set(p.name for p in folder.iterdir()) != {'LAUNCH.json'}:
            raise ValueError('Old HMM supervisor is not exclusively waiting')
        launch = json.loads((folder / 'LAUNCH.json').read_text())
        binding = c.process(36120 + gpu)
        expected = [c.PYTHON, '-u', str(c.ROOT / 'run_routing_queue_v3.py'), '--gpu', str(gpu)]
        if (binding is None or binding['state'] in ('Z', 'X', 'T', 't') or binding['command'] != expected or
                launch['pid'] != binding['pid'] or launch['predecessor_pid'] != 5202 + gpu or
                launch['source_sha256'] != c.SOURCE or launch['freeze_sha256'] != c.FREEZE):
            raise ValueError('Old waiting supervisor binding changed')
        bindings.append(binding)
    coordinator = c.process(36135)
    if coordinator is None or coordinator['command'] != [c.PYTHON, '-u', str(c.ROOT / 'finish_routing_panel_v3.py')]:
        raise ValueError('Old CPU coordinator identity changed')
    return bindings, coordinator


def prepared():
    os.environ.update(c.environment())
    args, protocol = c.frozen_dependencies()
    old, coordinator = waiting_only()
    c.pidfd_available()
    workers = []
    for gpu in range(8):
        binding = original_parent(gpu)
        rank = gpu % 4
        references = c.reference_gate(gpu, args, protocol)
        workers.append({'gpu': gpu, 'gpu_uuid': c.gpu_uuid(gpu), 'parent': binding,
            'task': 'reach' if gpu < 4 else 'reach-wall', 'logical_ranks': [rank, rank + 4],
            'three_references_complete_at_preparation': references is not None,
            'reference_hashes_at_preparation': references})
    return {'status': 'priority_handoff_prepared_not_activated', 'instance': 50233992,
        'owner': 'rep_geometry_transcoder/root', 'required_reference_arms': list(c.REQUIRED),
        'preserved_original_arms': list(c.ORIGINAL_ARMS), 'workers': workers,
        'old_cpu_waiters': old, 'old_cpu_coordinator': coordinator,
        'source_sha256': c.SOURCE, 'freeze_sha256': c.FREEZE,
        'method_fit_seed_sample_or_source_changed': False,
        'current_child_policy': 'never interrupt; complete and verify its entire24-episode stream',
        'post_hmm_policy': 'resume exact original parent with preserved Python wait/RNG state',
        'failure_policy': 'drain active child and verify empty GPU before original parent resume',
        'prepared_unix': time.time()}


def activate():
    authorization = json.loads((c.CONTROL / 'AUTHORIZATION.json').read_text())
    if (authorization['instance'] != 50233992 or authorization['owner'] != 'rep_geometry_transcoder/root' or
            authorization['actual_status'] != 'running' or authorization['intended_status'] != 'running' or
            authorization['label'] != 'jepa-fixed-behavior-us-v1' or
            not authorization['geolocation'].endswith(', US') or authorization['aggregate_usd_hour'] > 7 or
            not 0 <= time.time() - authorization['checked_unix'] <= 120):
        raise ValueError('Fresh owned US running-state/budget authorization required')
    plan = json.loads((c.CONTROL / 'PLAN.json').read_text())
    members = json.loads((c.CONTROL / 'FILES.json').read_text())
    for name, wanted in members.items():
        if c.digest(c.CONTROL / name) != wanted['sha256']:
            raise ValueError('Priority scheduling source/tests changed')
    c.frozen_dependencies()
    old, coordinator = waiting_only()
    for expected, current in zip(plan['old_cpu_waiters'] + [plan['old_cpu_coordinator']], old + [coordinator]):
        if not c.alive(expected) or current['starttime'] != expected['starttime']:
            raise ValueError('Prepared supervisor identity changed')
    for worker in plan['workers']:
        if not c.alive(worker['parent']) or original_parent(worker['gpu'])['starttime'] != worker['parent']['starttime']:
            raise ValueError('Original producer changed since preparation')
    # Administrative cancellation is explicitly separate from a scientific failure.
    c.write(c.CONTROL / 'CANCELLED_OLD_CPU_SUPERVISORS.json', {
        'reason': 'priority scheduling after three scientifically required references',
        'old_waiters': old, 'old_coordinator': coordinator,
        'old_status_files_preserved': True, 'old_waiter_failed_markers_are_administrative': True,
        'no_hmm_gpu_work_at_cancellation': True})
    for binding in [coordinator, *old]:
        c.send(binding, signal.SIGTERM)
    deadline = time.monotonic() + 60
    while any(c.alive(binding) for binding in [coordinator, *old]):
        if time.monotonic() > deadline:
            raise TimeoutError('Old CPU supervisors did not exit; no GPU launch')
        time.sleep(.2)
    launched = []
    for gpu in range(8):
        command = [c.PYTHON, '-u', str(c.CONTROL / 'routing_priority_queue.py'), '--gpu', str(gpu)]
        with (c.CONTROL / f'worker-gpu{gpu}.log').open('x') as log:
            child = subprocess.Popen(command,
                env=c.environment(), stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        binding = c.started(child.pid, command)
        launched.append({'gpu': gpu, 'process': binding})
    c.write(c.CONTROL / 'ACTIVATED_WORKERS.json', {'workers': launched})
    command = [c.PYTHON, '-u', str(c.CONTROL / 'routing_priority_control.py'), 'coordinate']
    with (c.CONTROL / 'coordinator.log').open('x') as log:
        child = subprocess.Popen(command,
            env=c.environment(), stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
    c.write(c.CONTROL / 'ACTIVATED.json', {'workers': launched, 'coordinator': c.started(child.pid, command),
        'source_sha256': c.SOURCE, 'freeze_sha256': c.FREEZE, 'outcome_selection': False})
    print(json.dumps({'status': 'priority_supervisors_activated', 'workers': launched, 'coordinator_pid': child.pid}))


def rescue(gpu, parent, status):
    """If a supervisor dies, preserve running model children and resume safely."""
    if not (status / 'SUSPEND_INTENT.json').exists() or (status / 'PARENT_RESUMED.json').exists():
        return
    candidates = []
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit():
            continue
        item = c.process(int(folder.name))
        if item is None or item['state'] in ('Z', 'X'):
            continue
        command = item['command']
        if (item['ppid'] == parent['pid'] or
                ('offline_study.routing_behavior' in command and
                 str(c.ROOT / f'fixed-response-worker-checks-20260908-v1/gpu-{gpu}') in command)):
            candidates.append(item)
    if len(candidates) > 1:
        raise ValueError('Unknown multiple children during rescue; no parent resume')
    c.resume_parent(parent, candidates[0] if candidates else None, gpu, status)


def coordinate():
    launched = json.loads((c.CONTROL / 'ACTIVATED_WORKERS.json').read_text())['workers']
    plan = json.loads((c.CONTROL / 'PLAN.json').read_text())
    failed = False
    rescued = set()
    deadline = time.monotonic() + 3 * 86400
    while True:
        complete = 0
        for worker in launched:
            gpu = worker['gpu']; status = c.CONTROL / f'worker-gpu{gpu}'
            if (status / 'DONE.json').exists():
                complete += 1
            elif not c.alive(worker['process']) and gpu not in rescued:
                failed = True
                try:
                    rescue(gpu, plan['workers'][gpu]['parent'], status)
                except BaseException as error:
                    c.write(c.CONTROL / f'rescue-gpu{gpu}-FAILED.json', {'error': str(error)})
                rescued.add(gpu)
        if complete + len(rescued) == 8:
            break
        if time.monotonic() > deadline:
            raise TimeoutError('Coordinator monitoring timeout; no child or GPU stopped')
        time.sleep(5)
    if failed:
        c.write(c.CONTROL / 'FAILED.json', {'no_partial_analysis': True, 'failed_gpu_assignments': sorted(rescued)})
        raise ValueError('A priority assignment failed; original queues safely released where possible')
    common, _ = c.scientific_arguments()
    subprocess.run([c.PYTHON, '-u', '-m', 'offline_study.routing_analysis'] + common +
        ['--panel', str(c.PANEL), '--output', str(c.PANEL / 'analysis')],
        env=c.environment(), check=True, timeout=3600)
    c.write(c.CONTROL / 'DONE.json', {'full_hmm_development_panel_analyzed': True,
        'new_shards': 32, 'new_episodes': 768, 'full_study_complete': False,
        'original_five_arm_analysis_coordinator_unchanged': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'activate', 'coordinate'))
    mode = parser.parse_args().mode
    if mode == 'prepare':
        plan = prepared(); c.write(c.CONTROL / 'PLAN.json', plan)
        print(json.dumps({'status': plan['status'], 'gpu_work_launched': False,
            'three_reference_complete_gpus': [w['gpu'] for w in plan['workers'] if w['three_references_complete_at_preparation']]}))
    elif mode == 'activate':
        activate()
    else:
        coordinate()
