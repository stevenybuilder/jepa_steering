"""One-time user-directed administrative cancellation; unchanged core queues resume."""
import argparse
import json
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0, '/workspace/jepa-runtime/routing-priority-20260908-v3')
import routing_priority_common as c

OUTPUT = Path('/workspace/jepa-runtime/core-priority-pause-20260908-v1')


def inventory():
    launch = json.loads((c.CONTROL / 'ACTIVATED.json').read_text())
    plan = json.loads((c.CONTROL / 'PLAN.json').read_text())
    if not c.alive(launch['coordinator']):
        raise ValueError('Expected priority coordinator is not live')
    children = []
    for worker in launch['workers']:
        gpu, supervisor = worker['gpu'], worker['process']
        parent = plan['workers'][gpu]['parent']
        if not c.alive(supervisor) or not c.alive(parent):
            raise ValueError('Expected supervisor or core parent is not live')
        if c.process(parent['pid'])['state'] not in ('T', 't'):
            raise ValueError('Original core parent not suspended')
        ids = (Path('/proc') / str(supervisor['pid']) / 'task' / str(supervisor['pid']) / 'children').read_text().split()
        live = [c.process(int(pid)) for pid in ids]
        live = [x for x in live if x and x['state'] not in ('Z', 'X')]
        if len(live) != 1:
            raise ValueError('Expected exactly one owned HMM child')
        child = live[0]
        command = child['command']
        if ('offline_study.routing_behavior' not in command or '--output' not in command or
                command[command.index('--output') + 1] != str(c.PANEL / plan['workers'][gpu]['task'] / 'hmm_filtered_gate' / f'shard-gpu{gpu}')):
            raise ValueError('Unexpected HMM child; refuse broad cancellation')
        target = Path(command[command.index('--output') + 1])
        children.append({'gpu': gpu, 'supervisor': supervisor, 'child': child, 'core_parent': parent,
                         'partial_output': str(target), 'published_episodes': len(list(target.glob('episode-*.json')))})
    return {'coordinator': launch['coordinator'], 'workers': children}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    bindings = inventory()
    if not args.apply:
        print(json.dumps(bindings)); return
    OUTPUT.mkdir(exist_ok=False)
    c.write(OUTPUT / 'AUTHORIZATION.json', {
        'reason': 'User agreed to finish core controls first and pause extensions',
        'administrative_not_scientific_failure': True, 'bindings': bindings,
        'all_existing_files_preserved': True, 'partial_streams_not_complete': True,
        'future_resume_requires_intact_rng_or_whole_stream_replay': True,
        'issued_unix': time.time(), 'script_sha256': c.digest(__file__)})
    # Stop only the HMM coordinator. Original paired-analysis coordinator stays live.
    c.send(bindings['coordinator'], signal.SIGTERM)
    for worker in bindings['workers']:
        # The original tested supervisor handles nonzero exit and verifies an empty
        # GPU before releasing its original parent. Never manually overlap jobs.
        c.send(worker['child'], signal.SIGTERM)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if all((c.CONTROL / f"worker-gpu{w['gpu']}" / 'PARENT_RESUMED.json').exists()
               and not c.alive(w['child']) and not c.alive(w['supervisor']) for w in bindings['workers']):
            c.write(OUTPUT / 'CORE_RESUMED.json', {'all_eight_original_queues_resumed': True,
                'all_hmm_children_and_priority_supervisors_terminal': True,
                'episode_seed_method_or_planner_changed': False, 'verified_unix': time.time()})
            print(json.dumps({'status': 'core_resumed_hmm_paused', 'gpus': 8})); return
        time.sleep(1)
    raise TimeoutError('Inspect preserved handoff receipts; do not force GPU overlap')


if __name__ == '__main__':
    main()
