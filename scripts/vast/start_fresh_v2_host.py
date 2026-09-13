"""Operational first-host assignment; scientific gate remains in frozen worker."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from start_fresh_v2_navigation_handoff import wait_exact_identity, reap_unregistered

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('root', type=Path)
parser.add_argument('--instance', type=int, required=True)
parser.add_argument('--host-slot', type=int, choices=range(6), required=True)
parser.add_argument('--ownership', type=Path)
parser.add_argument('--dry-run', action='store_true')
args = parser.parse_args()
root = args.root.resolve()
stripes = (
    [('reach', i, 8) for i in range(3)] + [('reach-wall', i, 8) for i in range(3)]
    + [('pointmaze', 0, 4), ('wall', 0, 4)],
    [('reach', i, 8) for i in range(3, 6)] + [('reach-wall', i, 8) for i in range(3, 6)]
    + [('pointmaze', 1, 4), ('wall', 1, 4)],
    [('reach', i, 8) for i in range(6, 8)] + [('reach-wall', i, 8) for i in range(6, 8)]
    + [('pointmaze', i, 4) for i in range(2, 4)] + [('wall', i, 4) for i in range(2, 4)],
    [('reach', i, 8) for i in range(3, 7)] + [('reach-wall', i, 8) for i in range(3, 7)],
    [('reach', i, 24) for i in range(11, 15)] + [('reach-wall', i, 24) for i in range(11, 15)],
    [('reach', i, 24) for i in range(19, 23)] + [('reach-wall', i, 24) for i in range(19, 23)],
)
mapping = [(gpu, *stripe) for gpu, stripe in enumerate(stripes[args.host_slot])]
if args.dry_run:
    print(json.dumps({'instance': args.instance, 'mapping': mapping, 'launched': False}))
    sys.exit(0)
allowed = {4: 50828584, 5: 50828583}
if allowed.get(args.host_slot) != args.instance:
    parser.error('Historical slots are retired. Only reviewed new MetaWorld slots '
                 '4/5 may launch; third-host splits use their separate handoff.')
freeze_sha = '7d5def0122f0dcf80e07e43ddc4f28ef6532fb04e6eb9acc4bc428427165ba90'
if args.ownership is None:
    parser.error('Reviewed global ownership receipt required')
approval = json.loads(args.ownership.read_bytes())
assert approval.get('approved') is True and approval.get('globally_unstarted_and_unclaimed') is True
assert approval.get('instance') == args.instance and approval.get('freeze_sha256') == freeze_sha
assert approval.get('mapping') == [list(row) for row in mapping], 'Unapproved mapping'
assert approval.get('old_mw_wave_plan_disabled') is True, 'Conflicting old wave plan'
assert approval.get('third_host_coarse_workers_quiesced') is True, 'Old coarse workers may duplicate work'
assert hashlib.sha256((root / 'fresh-freeze-v2/protocol.json').read_bytes()).hexdigest() == freeze_sha, 'Not approved unseen-input freeze'
logs = root / 'worker-logs-v2'
logs.mkdir(exist_ok=True)
for marker in ('RUNTIME_READY.json', 'ASSETS_READY.json'):
    assert (root / marker).is_file(), marker
assert not list(logs.glob('*LAUNCHED.json')), 'Existing launches: inspect before retry'
assert not (root / 'results-v2').exists(), 'Existing results: inspect before any restart'
with (logs / 'ASSIGNMENT.json').open('x') as handle:
    json.dump({'created_at': time.time(), 'instance': args.instance, 'host_slot': args.host_slot,
               'mapping': mapping, 'freeze': 'fresh-freeze-v2', 'freeze_sha256': freeze_sha,
               'ownership_sha256': hashlib.sha256(args.ownership.read_bytes()).hexdigest(),
               'output': 'results-v2', 'unassigned_host_slots_remain_pending': True,
               'technical_restart': None if args.host_slot != 0 else {
                   'prior_instance': 50795722,
                   'reason': 'Thermal throttling, inaccessible GPU and CUDA errors; no complete scientific scenario',
                   'prior_attempt': 'gs://rgt-jepa-archive-2026/fresh-campaign-20260912-v2/50795722-quiescent-technical-stop.tgz',
                   'policy': 'Preserve old attempt; rerun complete eight-arm scenarios on receiving physical GPU; do not merge prior partial arm',
               }}, handle, indent=2)
children = []
try:
    for gpu, task, worker_id, workers in mapping:
        command = [sys.executable, '-u', str(root / 'scripts/run_fresh_confirmation.py'),
                   'worker', '--project', str(root), '--freeze', str(root / 'fresh-freeze-v2'),
                   '--task', task, '--worker-id', str(worker_id), '--workers', str(workers),
                   '--output', str(root / 'results-v2')]
        with (logs / f'{task}-worker{worker_id}.log').open('xb') as log:
            child = subprocess.Popen(command, cwd=root,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'JEPA_VERIFIED_LOCAL_DINO': '1'},
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(child)
        observations = []
        identity = wait_exact_identity(child, command, observations)
        receipt = {'pid': child.pid, 'gpu': gpu, 'task': task, 'worker_id': worker_id,
                   'workers': workers, **identity, 'launched_at': time.time(),
                   'identity_observations': observations,
                   'stage': 'per-device-engineering-then-science-only-if-pass'}
        with (logs / f'{task}-worker{worker_id}-LAUNCHED.json').open('x') as handle:
            json.dump(receipt, handle, indent=2)
        print(json.dumps(receipt), flush=True)
except BaseException:
    cleanup = [reap_unregistered(child) for child in children]
    with (logs / 'LAUNCH_FAILED.json').open('x') as handle:
        json.dump({'owned_children_cleanup': cleanup}, handle, indent=2)
    raise
