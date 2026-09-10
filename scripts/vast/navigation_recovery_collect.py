"""Explicit in0 recovery override for immutable CPU-only navigation collection.

Never launches/restarts science, changes old worker receipts, or runs analysis.
The original importer is reused; ambiguous/partial transfers stop for review.
Only read-only SSH transport failures are automatically observed again.
"""
import argparse
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time

PROJECT = Path(__file__).resolve().parents[2]
NAV = Path('/workspace/jepa-runtime/navigation-redistribution-20260908-v2')
RECOVERY = Path('/workspace/jepa-runtime/navigation-recovery-in0-20260908-v1')
LOCAL = PROJECT / 'artifacts/offline_study/navigation-recovery-in0-20260908-v1/collection'
PLAN_SHA = '378cec55b21b84cf3d91b35fe0cfbc4f4b793a0f33b4cc6a726d17f1a4a46a63'
LAUNCH_SHA = 'ae9ac3397e0a0611e375d4e2657fe429041d2367a8fab3b64f0ebc544c3a2597'
FILES_SHA = '9497a41e8d8121be87fa94337b18384da9c35558e58e795975adb0cf5f1f9f27'
INSTANCE, UUID = 50205763, 'GPU-1541ee72-c8fc-ca75-7fea-470ae793c81e'
KEYS = ('task', 'arm', 'rank')
EXPECTED = ([{'task': 'wall', 'arm': 'permuted_visual', 'rank': r} for r in range(4, 8)] +
    [{'task': 'wall', 'arm': 'permuted_joint', 'rank': r} for r in range(3)] +
    [{'task': 'pointmaze', 'arm': 'permuted_visual', 'rank': r} for r in range(3, 8)] +
    [{'task': 'pointmaze', 'arm': 'permuted_joint', 'rank': r} for r in range(3)])


def sha(data):
    return hashlib.sha256(data).hexdigest()


def valid_sha(value):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('Explicit SHA256 required')
    return value


def write(path, value):
    """Atomic append-only local receipt, preserving even interrupted publication."""
    path = Path(path)
    pending = path.with_name(path.name + '.publishing')
    with pending.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.link(pending, path); pending.unlink()


def event(folder, kind, **fields):
    write(folder / (str(time.time_ns()) + '-' + kind + '.json'),
          {'kind': kind, 'observed_unix': time.time(), 'gpu_calls': 0,
           'scientific_restarts': 0, **fields})


def frozen_helpers():
    path = PROJECT / 'artifacts/offline_study/navigation-redistribution-20260908-v2/FILES.json'
    if path.is_symlink() or sha(path.read_bytes()) != FILES_SHA:
        raise ValueError('Frozen local operations manifest changed')
    for name, wanted in json.loads(path.read_bytes()).items():
        if Path(name).name != name:
            raise ValueError('Unsafe frozen source name')
        file = PROJECT / ('tests' if name.startswith('test_') else 'scripts/vast') / name
        if file.is_symlink() or {'bytes': file.stat().st_size, 'sha256': sha(file.read_bytes())} != wanted:
            raise ValueError('Frozen local helper changed: ' + name)
    directory = PROJECT / 'scripts/vast'
    sys.path.insert(0, str(directory))
    modules = [importlib.import_module(name) for name in
               ('navigation_redistribution_common', 'navigation_redistribution_stage', 'navigation_redistribution_collect')]
    for module in [*modules, modules[0].lifecycle]:
        if Path(module.__file__).resolve().parent != directory.resolve():
            raise ValueError('Wrong local frozen helper imported')
    return modules


def fresh_connections(stage):
    connections, authority = stage.connections()  # Original live owner/US/running checks.
    age = time.time() - authority['checked_unix']
    cost = authority['aggregate_usd_hour']
    if (set(connections) != {50231985, 50205763, 50259194} or
            set(authority['instances']) != set(connections) or
            authority['owner'] != 'rep_geometry_transcoder/root' or authority['gpu_calls'] != 0 or
            not math.isfinite(cost) or not 0 <= cost <= 7 or not math.isfinite(age) or not 0 <= age <= 60):
        raise ValueError('Fresh owned US provider evidence within $7/hour required')
    valid_sha(authority['board_sha256'])
    return connections, authority


# Remote code is stdlib-only; commands are restricted to read/verify/export/import.
# Verified script bytes are executed directly, not reopened after their hash gate.
# The original manifest and original PLAN are checked before every remote action.
BOUND = '''import hashlib,json,pathlib,sys
nav=pathlib.Path(sys.argv[1]); recovery=pathlib.Path(sys.argv[2]); pins=json.loads(sys.argv[3]); request=json.loads(sys.argv[4])
def data(path,wanted=None):
 if path.is_symlink() or not path.is_file():raise ValueError('Regular bound evidence required')
 value=path.read_bytes()
 if wanted is not None and hashlib.sha256(value).hexdigest()!=wanted:raise ValueError('Bound evidence changed')
 return value
manifest=json.loads(data(nav/'FILES.json',pins['files']))
for name,row in manifest.items():
 if pathlib.Path(name).name!=name:raise ValueError('Unsafe manifest member')
 value=data(nav/name,row['sha256'])
 if len(value)!=row['bytes']:raise ValueError('Bound member size changed')
original=json.loads(data(nav/'PLAN.json',pins['original']))
cutover=json.loads(data(nav/'CUTOVER.json'))
if cutover['plan_sha256']!=pins['original']:raise ValueError('Wrong original cutover')
kind=request['kind']
if kind in ('recovery','recovery-plan'):
 code=data(recovery/'navigation_recovery.py',pins['script'])
 plan=json.loads(data(recovery/'PLAN.json',pins['recovery']))
 if plan['operations_sha256']!=pins['script'] or plan['original_plan_sha256']!=pins['original']:raise ValueError('Wrong recovery binding')
if kind=='original-plan':print(json.dumps(original))
elif kind=='recovery-plan':print(json.dumps(plan))
elif kind=='status':
 worker=request['worker']
 if worker not in original['workers']:raise ValueError('Unknown worker')
 root=recovery/'run' if worker=='in0' else nav/'workers'/worker
 print(json.dumps({name:(root/name).exists() for name in ('FAILED.json','DONE.json')}))
elif kind=='done':
 worker=request['worker']
 if worker not in original['workers'] or worker=='in0':raise ValueError('Use recovery proof for in0')
 print(data(nav/'workers'/worker/'DONE.json',request['sha256']).decode())
elif kind in ('control','recovery'):
 mode=request['mode']
 if mode not in ('verify-completed','export-result','import-result'):raise ValueError('No scientific launches or analysis')
 if kind=='recovery' and mode=='import-result':raise ValueError('Only original importer allowed')
 path=recovery/'navigation_recovery.py' if kind=='recovery' else nav/'navigation_redistribution_control.py'
 if kind=='control':code=data(path,manifest[path.name]['sha256'])
 args=[str(path),mode]
 if kind=='recovery':args+=['--plan-sha256',pins['recovery']]
 elif mode=='verify-completed':args+=['--worker',request['worker']]
 if mode in ('export-result','import-result'):
  job=request['job']; jobs=original['completed']+[j for w in original['workers'].values() for j in w['jobs']]
  if set(job)!={'task','arm','rank'} or not any(all(j[k]==job[k] for k in job) for j in jobs):raise ValueError('Unassigned stream')
  if kind=='recovery' and job not in original['workers']['in0']['jobs']:raise ValueError('Wrong recovery stream')
  args+=['--job',json.dumps(job)]
  if request.get('original'):
   if kind!='control' or mode!='export-result' or not any(all(j[k]==job[k] for k in job) for j in original['completed']):raise ValueError('Wrong preboundary export')
   args+=['--original']
 sys.path.insert(0,str(nav)); sys.argv=args
 exec(compile(code,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
elif kind=='publish':
 value=request['value']
 if value['plan_sha256']!=pins['original'] or set(value['workers'])!=set(original['workers']):raise ValueError('Incomplete collection')
 path=nav/'COLLECTION_COMPLETE.json'; payload=(json.dumps(value,sort_keys=True,indent=2)+'\\n').encode()
 if path.exists():
  if data(path)!=payload:raise FileExistsError('Different existing collection completion')
 else:
  import os
  pending=path.with_name(path.name+'.publishing')
  with pending.open('xb') as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())
  os.link(pending,path);pending.unlink()
 print(json.dumps({'sha256':hashlib.sha256(data(path)).hexdigest(),'bytes':len(payload)}))
else:raise ValueError('Unknown bounded operation')
'''


class ObservationUnavailable(Exception):
    """Read-only SSH transport did not yield evidence; not a scientific failure."""


def command(pins, request):
    return ['env', 'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1',
            'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1',
            '/usr/bin/python3', '-c', BOUND, str(NAV), str(RECOVERY), json.dumps(pins), json.dumps(request)]


def observe(stage, connection, pins, request):
    try:
        return json.loads(stage.get(connection, command(pins, request)))
    except subprocess.TimeoutExpired as error:
        raise ObservationUnavailable('Read-only SSH timed out') from error
    except subprocess.CalledProcessError as error:
        if error.returncode == 255:
            raise ObservationUnavailable('Read-only SSH transport unavailable') from error
        raise  # Verifier/protocol failures are never recast as transient observations.


def require(value, wanted, label):
    if any(value.get(k) != v for k, v in wanted.items()):
        raise ValueError(label + ' binding changed')


def validate_recovery(c, original, plan, script_sha):
    c.validate_plan(original)
    require(original['workers']['in0'], {'instance': INSTANCE, 'gpu': 0, 'gpu_uuid': UUID,
        'jobs': EXPECTED}, 'Exact original in0 worker')
    require(plan, {'schema': 1, 'root': str(RECOVERY), 'instance': INSTANCE, 'gpu': 0, 'gpu_uuid': UUID,
        'original_plan_sha256': PLAN_SHA, 'original_launch_sha256': LAUNCH_SHA,
        'operations_sha256': script_sha, 'source_sha256': c.SOURCE, 'freeze_sha256': c.FREEZES,
        'expected_endpoints': 180, 'reused_streams': 8, 'recovered_streams': 7,
        'whole_stream_restart_only': True, 'partial_prefix_excluded_from_analysis': True,
        'automatic_retry': False, 'new_episode_identities': 0}, 'Recovery plan')
    rows = plan['completed'] + plan['recovered']
    if len(plan['completed']) != 8 or len(plan['recovered']) != 7 or [{k: r[k] for k in KEYS} for r in rows] != EXPECTED:
        raise ValueError('Recovery cannot add/omit/reorder/duplicate streams')
    for index, row in enumerate(rows):
        if row['output'] != str((NAV if index < 8 else RECOVERY) / 'results' / c.relative(EXPECTED[index])):
            raise ValueError('Mixed union output paths changed')
    if plan['unstarted'] != EXPECTED[9:] or plan['interrupted']['job'] != EXPECTED[8]:
        raise ValueError('Interrupted/unstarted assignment changed')


def validate_proof(c, name, worker, proof, pins, recovery_plan):
    require(proof, {'worker': name, 'plan_sha256': PLAN_SHA, 'instance': worker['instance'],
        'gpu': worker['gpu'], 'gpu_uuid': worker['gpu_uuid'],
        'verified_endpoints': len(worker['jobs']) * 12, 'scientific_validators_unchanged': True}, 'Worker proof')
    valid_sha(proof['done_sha256'])
    identity = proof['identity']
    if (not isinstance(identity, dict) or type(identity.get('pid')) is not int or identity['pid'] <= 0 or
            type(identity.get('starttime')) is not int or identity['starttime'] <= 0 or
            not isinstance(identity.get('command'), list) or not identity['command']):
        raise ValueError('Exact completed worker process identity required')
    if name == 'in0':
        require(proof, {'recovery_plan_sha256': pins['recovery'], 'original_launch_sha256': LAUNCH_SHA,
            'recovery_root': str(RECOVERY), 'partial_records_in_analysis': 0}, 'Recovery union proof')
        expected = recovery_plan['completed'] + recovery_plan['recovered']
        rows = proof['completed_jobs']
        if len(rows) != 15:
            raise ValueError('Recovery proof must contain all fifteen streams')
        for row, spec in zip(rows, expected):
            require(row, {k: spec[k] for k in (*KEYS, 'output')}, 'Recovery proof stream')
            valid_sha(row['report_sha256'])
            if 'report_sha256' in spec and row['report_sha256'] != spec['report_sha256']:
                raise ValueError('Reused original stream changed')


def copy_job(stage, connections, pins, worker, job, *, original=False):
    """Both ends hash-gated; original frozen importer remains the only receiver."""
    source = connections[50231985 if original else worker['instance']]
    kind = 'recovery' if not original and worker['instance'] == INSTANCE else 'control'
    request = {'kind': kind, 'mode': 'export-result', 'job': job, 'original': original}
    with tempfile.TemporaryFile() as packet:
        subprocess.run(source + [shlex.join(command(pins, request))], stdout=packet, check=True, timeout=600)
        packet.seek(0)
        result = subprocess.check_output(connections[50231985] + [shlex.join(command(pins,
            {'kind': 'control', 'mode': 'import-result', 'job': job}))], stdin=packet, text=True, timeout=600)
    copied = json.loads(result)
    require(copied, {'job': job}, 'Imported stream')
    valid_sha(copied['report_sha256'])
    return copied


def collect_rows(stage, folder, pins, worker, rows, *, original=False):
    for row in rows:
        job = {k: row[k] for k in KEYS}
        name = '-'.join(str(job[k]) for k in KEYS) + '.json'
        receipt = folder / name
        # Re-export/re-import complete existing streams verification-only after a
        # manual collector restart. This also checks the remote destination, not
        # merely a possibly stale local receipt. Partial destinations fail closed.
        connections, authority = fresh_connections(stage)
        event(folder, 'TRANSFER_AUTHORITY', job=job, authority=authority)
        copied = copy_job(stage, connections, pins, worker, job, original=original)
        require(copied, {'report_sha256': row['report_sha256']}, 'Exact source report')
        if receipt.exists():
            existing = json.loads(receipt.read_bytes())
            require(existing, {'job': job, 'report_sha256': copied['report_sha256']}, 'Existing local receipt')
        else:
            write(receipt, copied)


def poll(c, stage, folder, pins):
    connections, authority = fresh_connections(stage)
    event(folder, 'OBSERVATION_AUTHORITY', authority=authority)
    plan = observe(stage, connections[50231985], pins, {'kind': 'original-plan'})
    recovery = observe(stage, connections[INSTANCE], pins, {'kind': 'recovery-plan'})
    validate_recovery(c, plan, recovery, pins['script'])
    verified, jobs = {}, {}
    for name, worker in plan['workers'].items():
        connection = connections[worker['instance']]
        try:
            status = observe(stage, connection, pins, {'kind': 'status', 'worker': name})
            if status['FAILED.json']:
                raise ValueError('Required worker failed: ' + name + '; no scientific retry')
            if not status['DONE.json']:
                continue
            request = {'kind': 'recovery' if name == 'in0' else 'control',
                       'mode': 'verify-completed', 'worker': name}
            proof = observe(stage, connection, pins, request)
            validate_proof(c, name, worker, proof, pins, recovery)
            if name == 'in0':
                rows = proof['completed_jobs']
            else:
                done = observe(stage, connection, pins, {'kind': 'done', 'worker': name, 'sha256': proof['done_sha256']})
                rows = done['completed_jobs']
                if [{k: r[k] for k in KEYS} for r in rows] != worker['jobs']:
                    raise ValueError('Original worker export assignment changed')
                for row in rows:
                    if row['output'] != str(NAV / 'results' / c.relative({k: row[k] for k in KEYS})):
                        raise ValueError('Original worker result path changed')
                    valid_sha(row['report_sha256'])
            verified[name], jobs[name] = proof, rows
        except ObservationUnavailable as error:
            event(folder, 'OBSERVATION_RETRYABLE', worker=name, error=str(error))
    # Each terminal worker is collected once during this watcher lifetime; a new
    # process revalidates its existing destination using the immutable importer.
    return plan, verified, jobs


def watch(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Collector requires CUDA_VISIBLE_DEVICES empty')
    pins = {'files': FILES_SHA, 'original': PLAN_SHA,
            'script': valid_sha(args.recovery_script_sha256), 'recovery': valid_sha(args.recovery_plan_sha256)}
    if sha(Path(__file__).read_bytes()) != valid_sha(args.collector_sha256):
        raise ValueError('Root-reviewed collector bytes changed')
    if args.worker != 'in0' or args.recovery_root != RECOVERY or args.original_plan_sha256 != PLAN_SHA:
        raise ValueError('Only the explicit fixed in0 recovery override is allowed')
    c, stage, _original_collector = frozen_helpers()
    folder = LOCAL; folder.mkdir(parents=True, exist_ok=True)
    if folder.is_symlink(): raise ValueError('Regular local collection root required')
    with (stage.PROOF / 'collection/watcher.lock').open('a') as old_lock, (folder / 'watcher.lock').open('a') as lock:
        # Coordinate with both the original watcher and another recovery watcher.
        fcntl.flock(old_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding = {'collector_sha256': args.collector_sha256, 'pins': pins,
                   'override_worker': 'in0', 'recovery_root': str(RECOVERY), 'analysis_launched': False}
        binding_path = folder / 'BINDING.json'
        if binding_path.exists():
            if json.loads(binding_path.read_bytes()) != binding: raise ValueError('Existing collector binding changed')
        else: write(binding_path, binding)
        event(folder, 'START', pid=os.getpid(), argv=sys.argv, binding=binding)
        deadline, copied = time.monotonic() + 48 * 3600, set()
        try:
            while True:
                try:
                    plan, verified, jobs = poll(c, stage, folder, pins)
                    if 'preboundary' not in copied:
                        collect_rows(stage, folder, pins, None, plan['completed'], original=True)
                        copied.add('preboundary')
                    for name in verified:
                        if name not in copied:
                            collect_rows(stage, folder, pins, plan['workers'][name], jobs[name])
                            copied.add(name)
                    print(json.dumps({'verified_workers': sorted(verified), 'required_workers': 7,
                        'override_worker': 'in0', 'scientific_restarts': 0, 'gpu_calls': 0}), flush=True)
                    if set(verified) == set(c.WORKERS):
                        value = {'plan_sha256': PLAN_SHA, 'workers': verified, 'candidate_streams': 128,
                            'episodes_per_task_condition': 96, 'source_sha256': c.SOURCE, 'freeze_sha256': c.FREEZES,
                            'recovery_override': binding, 'partial_records_in_analysis': 0}
                        connections, authority = fresh_connections(stage)
                        event(folder, 'PUBLICATION_AUTHORITY', authority=authority)
                        # Publication is not an observation: ambiguity stops for review.
                        remote = json.loads(stage.get(connections[50231985], command(pins, {'kind': 'publish', 'value': value})))
                        data = (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()
                        if remote != {'sha256': sha(data), 'bytes': len(data)}: raise ValueError('Completion publication readback differs')
                        target = folder / 'COLLECTION_COMPLETE.json'
                        if target.exists():
                            if target.read_bytes() != data: raise FileExistsError('Different local completion')
                        else: write(target, value)
                        event(folder, 'COMPLETE', publication=remote, analysis_launched=False)
                        return
                except ObservationUnavailable as error:
                    event(folder, 'OBSERVATION_RETRYABLE', error=str(error))
                if args.once: return
                if time.monotonic() >= deadline: raise TimeoutError('48-hour collector observation limit; jobs untouched')
                time.sleep(30)
        except BaseException as error:
            event(folder, 'STOPPED_FOR_REVIEW', error=str(error), error_type=type(error).__name__,
                  automatic_retry=False, partial_evidence_preserved=True)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', required=True, choices=('in0',))
    parser.add_argument('--recovery-root', required=True, type=Path, choices=(RECOVERY,))
    parser.add_argument('--original-plan-sha256', required=True, choices=(PLAN_SHA,))
    parser.add_argument('--recovery-plan-sha256', required=True)
    parser.add_argument('--recovery-script-sha256', required=True)
    parser.add_argument('--collector-sha256', required=True)
    parser.add_argument('--once', action='store_true')
    watch(parser.parse_args())


if __name__ == '__main__': main()
