"""One authorized per-GPU boundary stop; no scientific edits or output deletion."""
import hashlib, json, os, sys, time
from pathlib import Path
import fresh_budget_watchdog as guard
from offline_study.fresh_confirmation import verify_bundle

root = Path('/workspace/fresh-four-20260912-v1')
logs = root / 'worker-logs-v2'
protocol = json.loads((root / 'fresh-freeze-v2/protocol.json').read_bytes())
freeze = hashlib.sha256((root / 'fresh-freeze-v2/protocol.json').read_bytes()).hexdigest()
assert freeze == '7d5def0122f0dcf80e07e43ddc4f28ef6532fb04e6eb9acc4bc428427165ba90'
assignment = json.loads((logs / 'ASSIGNMENT.json').read_bytes())
assert assignment['instance'] == 50827072
workers, _ = guard.load_workers(root, 50827072, freeze)
pending = {w['pid']: w for w in workers}
end = time.monotonic() + 600
while pending and time.monotonic() < end:
    for pid, worker in list(pending.items()):
        receipt = json.loads(Path(worker['launch_receipt']).read_bytes())
        task, episode = receipt['task'], receipt['worker_id']
        expected_row = next(r for r in protocol['tasks'][task]['records'] if r['role'] == 'excluded_engineering' and r['episode'] == 0)
        candidates = list((root / 'results-v2/engineering' / task).glob('*/DONE.json'))
        import subprocess
        uuid = subprocess.check_output(['nvidia-smi', '--id=' + str(receipt['gpu']), '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip().removeprefix('GPU-')
        folder = root / 'results-v2/engineering' / task / uuid
        if not (folder / 'DONE.json').exists():
            continue
        verify_bundle(folder, protocol, task, expected_row, freeze, True, uuid)
        assert guard.identity(pid) == {'command': worker['command'], 'start_tick': worker['start_tick']}
        initial = root / 'results-v2' / task / f'scenario-{episode:03d}'
        before = {'files': sorted(str(p.relative_to(initial)) for p in initial.rglob('*') if p.is_file()),
                  'started': json.loads((initial / 'STARTED.json').read_bytes()) if (initial / 'STARTED.json').exists() else None}
        record = {'pid': pid, 'gpu': receipt['gpu'], 'task': task, 'episode': episode, 'device_uuid': uuid,
                  'engineering_verified': True, 'before_quiesce': before, 'command': worker['command'],
                  'start_tick': worker['start_tick'], 'authorized_by': 'root boundary handoff 2026-09-13', 'utc_unix': time.time()}
        guard.publish(logs / f'COARSE_BOUNDARY_INTENT.gpu{receipt["gpu"]}.json', record)
        result = guard.quiesce([worker], grace=1)
        guard.publish(logs / f'COARSE_BOUNDARY_QUIESCED.gpu{receipt["gpu"]}.json', {**record, **result})
        assert result['quiesced']
        print(json.dumps({'gpu': receipt['gpu'], 'pid': pid, 'engineering_verified': True, 'quiesced': True, 'initial_files': before['files']}), flush=True)
        del pending[pid]
    if pending: time.sleep(2)
if pending: raise RuntimeError('Boundary wait expired; remaining PIDs=' + repr(sorted(pending)))
