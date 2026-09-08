"""Pause only original queue parent, drain its child, run HMM, resume original."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import routing_priority_common as c


def run(gpu):
    plan = json.loads((c.CONTROL / 'PLAN.json').read_text())
    binding = plan['workers'][gpu]
    parent = binding['parent']
    status = c.CONTROL / f'worker-gpu{gpu}'
    status.mkdir(exist_ok=False)
    c.write(status / 'LAUNCH.json', {'pid': os.getpid(), 'parent': parent, 'gpu': gpu,
        'plan_sha256': c.digest(c.CONTROL / 'PLAN.json'), 'freeze_sha256': c.FREEZE})
    os.environ.update(c.environment(gpu))
    args, protocol = c.frozen_dependencies()
    if c.gpu_uuid(gpu) != binding['gpu_uuid']:
        raise ValueError('Receiving GPU identity changed')
    active = None
    running = None
    suspended = False
    interrupted = False

    def cancel(signum, frame):
        nonlocal interrupted
        # First signal exits scientific scheduling; cleanup drains, never kills a child.
        if not interrupted:
            interrupted = True
            raise RuntimeError('Priority supervisor interrupted; preserve child and resume original safely')
    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)

    def execute(extra, label, cap):
        nonlocal active, running
        c.write(status / (label + '-START.json'), {'stage': label, 'gpu': gpu})
        command = [c.PYTHON, '-u', '-m', 'offline_study.routing_behavior'] + extra
        with (status / (label + '.log')).open('x') as log:
            running = subprocess.Popen(command,
                env=c.environment(gpu), stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
            active = c.started(running.pid, command)
            c.write(status / (label + '-CHILD.json'), active)
            result = running.wait(timeout=cap)
            if result:
                raise ValueError('Required HMM stage failed; no retry: ' + label)
        active = None
        running = None

    try:
        # Freeze/validation is CPU-only. Existing producer continues until all three
        # required references finish; it may start coupling during this check.
        deadline = time.monotonic() + 24 * 3600
        while c.reference_gate(gpu, args, protocol) is None:
            if not c.alive(parent) or time.monotonic() > deadline:
                raise ValueError('Required references incomplete after original producer ended')
            time.sleep(5)
        if c.alive(parent):
            c.write(status / 'SUSPEND_INTENT.json', {'parent': parent,
                'scientific_children_must_finish_untouched': True})
            suspended = c.send(parent, signal.SIGSTOP)
            while suspended and c.alive(parent) and c.process(parent['pid'])['state'] not in ('T', 't'):
                time.sleep(.1)
            if suspended:
                ids = (Path('/proc') / str(parent['pid']) / 'task' / str(parent['pid']) / 'children').read_text().split()
                children = [c.process(int(pid)) for pid in ids]
                children = [child for child in children if child is not None]
                if len(children) > 1:
                    raise ValueError('Unknown independent child overlap')
                if children:
                    active = children[0]
                    # An already-zombie child has no cmdline; recover the exact
                    # command from its last original queue log stage and require
                    # all created shards complete, never guess an active child.
                    if active['state'] in ('Z', 'X'):
                        active = None
                        targets = [args.reference / binding['task'] / arm / f'shard-gpu{gpu}'
                            for arm in c.ORIGINAL_ARMS]
                        for target in targets:
                            if target.exists():
                                c.verify_boundary(target, gpu, args, protocol)
                    else:
                        target = c.validate_original_child(active, gpu)
                        c.write(status / 'ORIGINAL_CHILD_DRAIN.json', {'child': active, 'target': str(target)})
                        while c.alive(active):
                            if time.monotonic() > deadline:
                                raise TimeoutError('Original child did not finish; no HMM launch')
                            time.sleep(2)
                        c.verify_boundary(target, gpu, args, protocol)
                        active = None
        # If parent naturally completed before suspension, all five reports are
        # required. If stopped, the 3 references + current full shard suffice.
        if not suspended:
            for arm in c.ORIGINAL_ARMS:
                c.verify_boundary(args.reference / binding['task'] / arm / f'shard-gpu{gpu}', gpu, args, protocol)
        while c.gpu_processes(gpu):
            if time.monotonic() > deadline:
                raise TimeoutError('GPU not empty after original child completion')
            time.sleep(2)
        references = c.reference_gate(gpu, args, protocol)
        if references is None:
            raise ValueError('Required references disappeared')
        c.write(status / 'HANDOFF_VERIFIED.json', {'reference_hashes': references,
            'gpu_empty': True, 'parent_suspended': suspended, 'gpu_uuid': binding['gpu_uuid']})
        common, _ = c.scientific_arguments()
        common += ['--checkpoint', str(c.ROOT / 'fixed-response-assets-20260908-v1/jepa_wm_metaworld.pth.tar'),
            '--task', binding['task'], '--worker-reference', str(args.reference_workers / f'gpu-{gpu}')]
        engineering = c.PANEL / 'engineering' / binding['task'] / binding['gpu_uuid']
        execute(['engineer'] + common + ['--output', str(engineering)], 'engineering', 4 * 3600)
        from offline_study.behavioral_development import verified_report
        from offline_study.routing_behavior import NEW_ARMS, verify_engineering
        verify_engineering(engineering, protocol, binding['task'], c.FREEZE)
        for arm in NEW_ARMS:
            target = c.PANEL / binding['task'] / arm / f'shard-gpu{gpu}'
            execute(['run'] + common + ['--engineering', str(engineering), '--arm', arm,
                '--logical-ranks', *map(str, binding['logical_ranks']), '--output', str(target)], arm, 8 * 3600)
            verified_report(target)
        c.write(status / 'ROUTED_COMPLETE.json', {'new_episodes': 96, 'new_shards': 4,
            'full_panel_analysis_complete': False, 'fresh_confirmation': False})
    except BaseException as error:
        c.write(status / 'FAILED.json', {'error': str(error), 'partial_not_complete': True,
            'active_child_not_killed': active, 'original_resume_required': suspended})
        raise
    finally:
        if suspended:
            # Do not allow a second cancellation to interrupt cleanup. The active
            # child can finish its intact stream before the original parent resumes.
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                # Even if exact argv capture failed during exec, retain the Popen
                # handle and drain that known child before checking an empty GPU.
                if running is not None and running.poll() is None:
                    running.wait(timeout=24 * 3600)
                c.resume_parent(parent, active, gpu, status)
            except BaseException as error:
                c.write(status / 'RESUME_BLOCKED.json', {'error': str(error), 'do_not_overlap_gpu': True})
    if (status / 'ROUTED_COMPLETE.json').exists():
        if (status / 'RESUME_BLOCKED.json').exists():
            raise RuntimeError('Routed work finished but safe original resume remains blocked')
        c.write(status / 'DONE.json', {'status': 'priority_routed_shards_complete_original_queue_released',
            'new_episodes': 96, 'original_resume_receipt': (status / 'PARENT_RESUMED.json').exists(),
            'full_study_complete': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, choices=range(8), required=True)
    run(parser.parse_args().gpu)
