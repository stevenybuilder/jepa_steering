"""Reserve uncovered streams on existing owned GPUs, without additional rentals."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile
import time

import component_extension_fleet as fleet
from component_completion_tail import coverage

WORKERS = {0: (5, 50638073), 1: (1, 50626847), 2: (3, 50632757), 3: (11, 50640703)}


def main():
    assert 'September11 completion-coverage repair' in Path('/Users/stevenyang/Documents/GPU_RESOURCE_BOARD.md').read_text()
    rows = fleet.provider('show', 'instances')
    assert fleet.account_hourly(rows) <= 7
    workers, prior = {}, {}
    for slot, (generation, instance) in WORKERS.items():
        local = fleet.BASE / ('rentals' if generation == 1 else f'rentals-v{generation}') / f'slot-{slot:02d}'
        lease = fleet.read(local / 'LEASE.json')
        row = next(r for r in rows if r['id'] == instance)
        assert row['id'] == lease['instance'] and row['label'] == lease['label']
        assert row['geolocation'] == lease['geolocation'] and row['geolocation'].endswith(', US')
        assert row['actual_status'] == 'running'
        ssh = fleet.connection(row)
        command = "from pathlib import Path;print((Path('/workspace/metaworld-components-20260911-v1')/'QUEUE_PLAN.json').read_text())"
        plan = None
        for attempt in range(3):
            try:
                plan = json.loads(subprocess.check_output(ssh + [shlex.join(['python3', '-c', command])], text=True, timeout=35))
                break
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                if attempt == 2: raise
        prior[slot] = plan['queues']['0']
        workers[slot] = (generation, local, lease, row, plan)
    extra = coverage(prior)
    proof = fleet.BASE / 'coverage-repair-v1'
    proof.mkdir(exist_ok=True)
    if not (proof / 'COVERAGE.json').exists():
        fleet.write(proof / 'COVERAGE.json', {'time': time.time(), 'original': prior, 'additional': extra,
            'total_evaluations': 768, 'new_rentals': 0, 'scientific_source_changed': False})

    def deploy(slot):
        if not extra[slot]: return
        generation, local, lease, row, plan = workers[slot]
        remote = fleet.REMOTE + '/ops/continuation-v1'
        ssh = fleet.connection(row)
        read_runtime = "from pathlib import Path;import json;out=[]\nfor p in Path('/proc').iterdir():\n if p.name.isdigit():\n  try:a=[v.decode() for v in (p/'cmdline').read_bytes().split(b'\\0') if v]\n  except FileNotFoundError:continue\n  if len(a)>3 and a[2].endswith('/component_extension_queue.py') and '--root' in a and a[a.index('--root')+1]=='/workspace/metaworld-components-20260911-v1':out.append(a)\nprint(json.dumps(out))"
        commands = json.loads(subprocess.check_output(ssh + [shlex.join(['python3', '-c', read_runtime])], text=True, timeout=35))
        assert len(commands) == 1
        command = commands[0]
        checkpoint = command[command.index('--checkpoint') + 1] if '--checkpoint' in command else '/workspace/decision-runtime/checkpoints/jepa_wm_metaworld.pth.tar'
        spec = {'component_root': fleet.REMOTE, 'uuid': plan['devices'][0], 'extra': extra[slot],
                'checkpoint': checkpoint, 'deadline': lease['deadline'], 'original_queue': plan}
        # Register ownership and stop-guard dependency BEFORE remote submission.
        if not (local / 'CONTINUATION.json').exists():
            fleet.write(local / 'CONTINUATION.json', {'remote_root': remote, 'instance': row['id'],
                'extra': extra[slot], 'deadline': lease['deadline'], 'time': time.time()})
        label = f'com.steven.jepa.components.rental.{slot}' + ('' if generation == 1 else f'.v{generation}')
        subprocess.run(['launchctl', 'remove', label], check=True)
        subprocess.run(['launchctl', 'submit', '-l', label, '--', '/usr/bin/env',
            'PATH=/usr/local/bin:/usr/bin:/bin:/Users/stevenyang/.local/bin',
            f'JEPA_COMPONENT_RENTAL_ATTEMPT={generation}', fleet.PYTHON,
            str(Path(__file__).with_name('component_rental_fleet.py')), 'guard', '--slot', str(slot)], check=True)
        subprocess.run(ssh + ['mkdir -p ' + remote], check=True, timeout=35)
        with tempfile.TemporaryFile() as stream:
            with tarfile.open(fileobj=stream, mode='w') as archive:
                for name in ('component_completion_tail.py', 'component_resume_check.py'):
                    archive.add(Path(__file__).with_name(name), arcname=name, recursive=False)
            stream.seek(0)
            subprocess.run(ssh + ['tar --keep-old-files --no-same-owner -x -C ' + remote], stdin=stream, check=True, timeout=35)
        script = '''import json,sys,subprocess,hashlib
from pathlib import Path
r=Path(sys.argv[1]);spec=json.load(sys.stdin)
with (r/'PLAN.json').open('x') as out:json.dump(spec,out,indent=2)
with (r/'queue.log').open('x') as log:
 p=subprocess.Popen(['/workspace/component-python/bin/python','-u',str(r/'component_completion_tail.py'),'--root',str(r)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
receipt={'pid':p.pid,'files':{q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in r.glob('*.py')}}
with (r/'LAUNCH.json').open('x') as out:json.dump(receipt,out)
print(json.dumps(receipt))'''
        receipt = json.loads(subprocess.check_output(ssh + [shlex.join(['python3', '-c', script, remote])],
            input=json.dumps(spec), text=True, timeout=35))
        for name, digest in receipt['files'].items(): assert fleet.sha(Path(__file__).with_name(name)) == digest
        fleet.write(local / 'CONTINUATION_LAUNCH.json', {**receipt, 'time': time.time()})
        print(json.dumps({'instance': row['id'], 'additional_task_streams': extra[slot], 'pid': receipt['pid'], 'gpu_work_interrupted': False}), flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(deploy, (0, 1, 3)))


if __name__ == '__main__': main()
