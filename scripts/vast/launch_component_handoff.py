"""Deploy the tested operational handoff to three explicitly owned US workers."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

import component_extension_fleet as fleet

ASSIGNMENTS = ((1, 1, 2839), (3, 2, 2629), (5, 0, None))


def run(item):
    generation, slot, queue_pid = item
    local = fleet.BASE / ('rentals' if generation == 1 else f'rentals-v{generation}') / f'slot-{slot:02d}'
    lease = fleet.read(local / 'LEASE.json')
    assert lease['deadline'] == fleet.read(local / 'DEADLINE_AMENDMENT.json')['deadline']
    row = next(r for r in fleet.provider('show', 'instances') if r['id'] == lease['instance'])
    assert row['label'] == lease['label'] and row['geolocation'] == lease['geolocation'] and row['geolocation'].endswith(', US')
    assert row['cur_state'] == 'running' and row['actual_status'] == 'running'
    ssh = fleet.connection(row)
    remote = fleet.REMOTE + '/ops/handoff-v1'
    # Resolve exactly one owned coordinator read-only; supplied old PIDs must agree.
    lookup = "from pathlib import Path;import json;out=[]\nfor p in Path('/proc').iterdir():\n if p.name.isdigit():\n  try:a=[v.decode() for v in (p/'cmdline').read_bytes().split(b'\\0') if v]\n  except FileNotFoundError:continue\n  if len(a)>2 and a[2]=='/workspace/metaworld-components-20260911-v1/ops/component_extension_queue.py':out.append(int(p.name))\nprint(json.dumps(out))"
    found = json.loads(subprocess.check_output(ssh + [shlex.join(['python3', '-c', lookup])], text=True, timeout=30))
    assert len(found) == 1 and (queue_pid is None or found[0] == queue_pid)
    queue_pid = found[0]
    fleet.write(local / 'HANDOFF_ACTIVE.json', {'queue_pid': queue_pid, 'deadline': lease['deadline'],
        'instance': lease['instance'], 'time': time.time(), 'only_coordinator_changes': True})
    label = f'com.steven.jepa.components.rental.{slot}' + ('' if generation == 1 else f'.v{generation}')
    subprocess.run(['launchctl', 'remove', label], check=True)
    subprocess.run(['launchctl', 'submit', '-l', label, '--', '/usr/bin/env',
        'PATH=/usr/local/bin:/usr/bin:/bin:/Users/stevenyang/.local/bin',
        f'JEPA_COMPONENT_RENTAL_ATTEMPT={generation}', fleet.PYTHON,
        str(Path(__file__).with_name('component_rental_fleet.py')), 'guard', '--slot', str(slot)], check=True)
    subprocess.run(ssh + ['mkdir -p ' + remote], check=True, timeout=30)
    with subprocess.Popen(ssh + ['tar --no-same-owner -x -C ' + remote], stdin=subprocess.PIPE) as transfer:
        with tarfile.open(fileobj=transfer.stdin, mode='w|') as tar:
            for name in ('component_queue_handoff.py', 'component_extension_queue.py', 'component_resume_check.py'):
                tar.add(Path(__file__).with_name(name), arcname=name, recursive=False)
        transfer.stdin.close()
        assert transfer.wait() == 0
    launch = "import json,subprocess,sys;from pathlib import Path;p=json.load(sys.stdin);r=Path(p['root']);out=r/'ops/handoff-v1'\nwith (out/'handoff.log').open('x') as log:\n process=subprocess.Popen(['/workspace/component-python/bin/python','-u',str(out/'component_queue_handoff.py'),'--root',str(r),'--queue-pid',str(p['pid']),'--deadline',str(p['deadline'])],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps({'handoff_pid':process.pid}))"
    result = json.loads(subprocess.check_output(ssh + [shlex.join(['python3', '-c', launch])],
        text=True, input=json.dumps({'root': fleet.REMOTE, 'pid': queue_pid, 'deadline': lease['deadline']}), timeout=30))
    fleet.write(local / 'HANDOFF_LAUNCH.json', {**result, 'time': time.time()})
    print(json.dumps({'instance': lease['instance'], 'queue_pid': queue_pid, **result}), flush=True)


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(run, ASSIGNMENTS))
