"""CPU-only replacement watcher: exact assignment receipts, immutable collection."""
import argparse
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
import time

import navigation_redistribution_common as c
import navigation_redistribution_stage as stage


def present(connection, path):
    result = subprocess.run(connection + [shlex.join(['test', '-f', str(path)])], stdout=subprocess.DEVNULL)
    if result.returncode not in (0, 1):
        raise ValueError('Read-only receiving state check failed')
    return result.returncode == 0


def copy_job(source, destination, job, original=False):
    argv = stage.command('export-result', '--job', json.dumps(job)) + (['--original'] if original else [])
    with tempfile.TemporaryFile() as packet:
        subprocess.run(source + [shlex.join(argv)], stdout=packet, check=True, timeout=300)
        packet.seek(0)
        result = subprocess.check_output(destination + [shlex.join(stage.command('import-result', '--job', json.dumps(job)))], stdin=packet, text=True, timeout=300)
    return json.loads(result)


def watch(once=False):
    ssh, authority = stage.connections()
    plan = json.loads(stage.get(ssh[50231985], ['cat', str(c.CONTROL / 'PLAN.json')]))
    c.validate_plan(plan)
    plan_hash = json.loads(stage.get(ssh[50231985], ['cat', str(c.CONTROL / 'CUTOVER.json')]))['plan_sha256']
    folder = stage.PROOF / 'collection'; folder.mkdir(exist_ok=True)
    lock = (folder / 'watcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Only completed pre-boundary work is read; no outcome summaries are emitted.
    for value in plan['completed']:
        job = {k: value[k] for k in ('task', 'arm', 'rank')}
        receipt = folder / (job['task'] + '-' + job['arm'] + '-' + str(job['rank']) + '.json')
        if receipt.exists():
            if c.read(receipt)['report_sha256'] != value['report_sha256']:
                raise ValueError('Existing collection receipt differs')
        else:
            copied = copy_job(ssh[50231985], ssh[50231985], job, True)
            if copied['report_sha256'] != value['report_sha256']:
                raise ValueError('Pre-boundary scientific receipt changed')
            c.write(receipt, copied)
    deadline = time.monotonic() + 48 * 3600
    while True:
        verified = {}
        for name, worker in plan['workers'].items():
            connection = ssh[worker['instance']]
            status = c.CONTROL / 'workers' / name
            if present(connection, status / 'FAILED.json'):
                raise ValueError('Required worker failed; preserve all outputs, no retry')
            if not present(connection, status / 'DONE.json'):
                continue
            proof = json.loads(stage.get(connection, stage.command('verify-completed', '--worker', name)))
            if proof['plan_sha256'] != plan_hash:
                raise ValueError('Receiving verifier binds another partition')
            for job in worker['jobs']:
                receipt = folder / (job['task'] + '-' + job['arm'] + '-' + str(job['rank']) + '.json')
                if not receipt.exists():
                    c.write(receipt, copy_job(connection, ssh[50231985], job))
            verified[name] = proof
        print(json.dumps({'verified_workers': sorted(verified), 'required_workers': 7,
                          'outcome_based_selection': False, 'gpu_calls': 0}), flush=True)
        if len(verified) == 7:
            value = {'plan_sha256': plan_hash, 'workers': verified, 'candidate_streams': 128,
                     'episodes_per_task_condition': 96, 'source_sha256': c.SOURCE, 'freeze_sha256': c.FREEZES}
            data = (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()
            program = 'import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.open("xb").write(sys.stdin.buffer.read())'
            subprocess.run(ssh[50231985] + [shlex.join(['/usr/bin/python3', '-c', program,
                str(c.CONTROL / 'COLLECTION_COMPLETE.json')])], input=data, check=True)
            c.write(folder / 'COLLECTION_COMPLETE.json', value)
            print(stage.get(ssh[50231985], stage.command('analyze')), flush=True)
            return
        if once:
            return
        if time.monotonic() > deadline:
            raise TimeoutError('Watcher deadline; no scientific process is stopped')
        time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    watch(args.once)
