"""Operational-only handoff after the active scientific child finishes intact.

The coordinator alone is suspended; its detached model process runs normally.
No scientific code, output, RNG state, action budget or physical device changes.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def process(pid):
    try:
        root = Path('/proc') / str(pid)
        args = [x.decode() for x in (root / 'cmdline').read_bytes().split(b'\0') if x]
        stat = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'args': args, 'state': stat[0], 'parent': int(stat[1])}
    except FileNotFoundError:
        return None


def write(path, value):
    with path.open('x') as out:
        json.dump(value, out, indent=2)


def direct_children(pid):
    children = []
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            item = process(int(path.name))
            if item and item['parent'] == pid and item['state'] != 'Z':
                children.append((int(path.name), item))
    return children


def main(a):
    root = a.root
    output = root / 'ops/handoff-v1'
    queue = process(a.queue_pid)
    if (not queue or len(queue['args']) < 4 or
            not queue['args'][2].endswith('/component_extension_queue.py') or
            queue['args'][queue['args'].index('--root') + 1] != str(root)
            or (root / 'TERMINAL.json').exists()):
        raise ValueError('Original owned queue is not actively running')
    if not time.time() + 3600 < a.deadline < time.time() + 25 * 3600:
        raise ValueError('Require explicit bounded replacement deadline')
    write(output / 'HANDOFF_STARTED.json', {'queue_pid': a.queue_pid, 'deadline': a.deadline,
        'time': time.time(), 'scientific_source_changed': False})
    suspended = False
    retired = False
    try:
        os.kill(a.queue_pid, signal.SIGSTOP)
        suspended = True
        children = direct_children(a.queue_pid)
        if len(children) != 1 or 'offline_study.metaworld_component_behavior' not in children[0][1]['args']:
            raise ValueError('Handoff requires exactly one intact active component child')
        pid, child = children[0]
        child_output = Path(child['args'][child['args'].index('--output') + 1])
        task = child['args'][child['args'].index('--task') + 1]
        if not child_output.is_relative_to(root):
            raise ValueError('Child output outside owned instance root')
        write(output / 'CHILD_CONTINUING.json', {'pid': pid, 'output': str(child_output),
            'task': task, 'supervisor_only_suspended': True})
        wait_until = min(time.time() + 2 * 3600, a.deadline - 900)
        while time.time() < wait_until:
            current = process(pid)
            if current is None or current['state'] == 'Z':
                break
            if current['args'] != child['args']:
                raise ValueError('Child PID identity changed')
            time.sleep(3)
        else:
            raise TimeoutError('Intact child did not finish within bounded handoff window')
        if not (child_output / 'DONE.json').exists() or (child_output / 'FAILED.json').exists():
            raise ValueError('Active child failed; partial output must not be skipped')
        # Completion receipt is fully checked again by the new queue before reuse.
        # Retire only the old operational coordinator, never the scientific child.
        os.kill(a.queue_pid, signal.SIGKILL)
        retired = True
        for _ in range(60):
            if (root / 'TERMINAL.json').exists():
                break
            time.sleep(1)
        if not (root / 'TERMINAL.json').exists():
            raise ValueError('Original preparation process did not record termination')
        (root / 'TERMINAL.json').rename(output / 'PRIOR_TERMINAL.json')
        command = queue['args'][:]
        command[2] = str(output / 'component_extension_queue.py')
        command[command.index('--deadline') + 1] = str(a.deadline)
        command += ['--resume-verified']
        env = dict(os.environ, PYTHONPATH=str(root / 'code/src'), PYTHONDONTWRITEBYTECODE='1')
        with (output / 'resumed-queue.log').open('x') as log:
            new = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(output / 'HANDOFF_DONE.json', {'new_queue_pid': new.pid, 'old_queue_pid': a.queue_pid,
            'completed_child_output': str(child_output), 'partial_stream_reused': False,
            'scientific_source_changed': False, 'time': time.time()})
    except Exception as error:
        write(output / 'HANDOFF_FAILED.json', {'error_type': type(error).__name__,
            'old_supervisor_retired': retired, 'source_outputs_retained': True, 'time': time.time()})
        if suspended and not retired and process(a.queue_pid):
            os.kill(a.queue_pid, signal.SIGCONT)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--queue-pid', type=int, required=True)
    p.add_argument('--deadline', type=float, required=True)
    main(p.parse_args())
