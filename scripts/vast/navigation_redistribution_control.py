"""CPU preparation; separately authorized pause/drain and exclusive stream workers."""
import argparse
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time

import navigation_redistribution_common as c


def roots():
    return {'payload/code': c.ROOT / 'navigation-coupling-code-20260908-v2',
            'payload/evidence': c.ROOT / 'navigation-coupling-evidence-20260908-v2',
            'payload/maze-python': Path('/workspace/jepa-maze-python'),
            'payload/mujoco210': Path('/root/.mujoco/mujoco210'),
            **{f'payload/models/jepa_wm_{t}.pth.tar': c.ROOT / f'navigation-assets-20260907-v1/downloads/model/jepa_wm_{t}.pth.tar' for t in c.TASKS}}


def input_manifest():
    if c.source_hash(roots()['payload/code'] / 'src/offline_study') != c.SOURCE:
        raise ValueError('Only original frozen Nebraska source may be exported')
    result = {}
    for prefix, path in roots().items():
        members = ({'': {'bytes': path.stat().st_size, 'sha256': c.digest(path)}} if path.is_file() else c.file_manifest(path))
        for name, row in members.items():
            target = str(Path(prefix) / name)
            source = path / name if name else path
            if source.name in ('.env', 'rclone.conf', 'credentials.json', 'id_rsa', 'id_ed25519') or source.suffix == '.key' or (source.suffix == '.pem' and not (source.name == 'cacert.pem' and 'certifi' in source.parts)):
                raise ValueError('Credential-like runtime member refused: ' + target)
            result[target] = row
    return result


def export_inputs():
    manifest = input_manifest()
    with tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz') as archive:
        content = json.dumps(manifest, indent=2, sort_keys=True).encode()
        info = tarfile.TarInfo('INPUTS.json'); info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
        for name in manifest:
            prefix = next(k for k in roots() if name == k or name.startswith(k + '/'))
            remainder = name[len(prefix):].lstrip('/')
            source = roots()[prefix] / remainder if remainder else roots()[prefix]
            archive.add(source, arcname=name, recursive=False)
    if input_manifest() != manifest:
        raise ValueError('Frozen source bytes changed during export')


def receive_inputs():
    """Never tar-extract through links or overwrite an existing destination."""
    if (c.CONTROL / 'INPUTS.json').exists() or c.PAYLOAD.exists():
        raise FileExistsError('Payload already exists; preserve previous preparation')
    with tarfile.open(fileobj=sys.stdin.buffer, mode='r|gz') as archive:
        members = iter(archive)
        first = next(members)
        if first.name != 'INPUTS.json' or not first.isfile():
            raise ValueError('Input manifest must precede payload')
        manifest = json.load(archive.extractfile(first))
        c.write(c.CONTROL / 'INPUTS.json', manifest)
        seen = set()
        for member in members:
            name = member.name
            if name in seen or name not in manifest or not name.startswith('payload/'):
                raise ValueError('Unregistered/duplicate payload member')
            path = c.CONTROL / c.safe_member(name)
            for parent in path.parents:
                if parent == c.CONTROL:
                    break
                if parent.is_symlink():
                    raise ValueError('Extraction through symlink refused')
            path.parent.mkdir(parents=True, exist_ok=True)
            wanted = manifest[name]
            if member.issym() and set(wanted) == {'symlink'} and wanted['symlink'] == member.linkname:
                path.symlink_to(member.linkname)
            elif member.isfile() and set(wanted) == {'bytes', 'sha256'} and wanted['bytes'] == member.size:
                with path.open('xb') as destination:
                    shutil.copyfileobj(archive.extractfile(member), destination, 4 << 20)
                path.chmod(member.mode)
                os.utime(path, (member.mtime, member.mtime))
                if c.digest(path) != wanted['sha256']:
                    raise ValueError('Transferred payload changed')
            else:
                raise ValueError('Unsupported payload member type')
            seen.add(name)
    if seen != set(manifest):
        raise ValueError('Incomplete payload')
    c.verify_members(c.CONTROL, manifest)
    c.write(c.CONTROL / 'INPUTS_VERIFIED.json', {'members': len(manifest), 'inputs_sha256': c.digest(c.CONTROL / 'INPUTS.json'), 'gpu_calls': 0})


def python_check(instance, program, *args):
    return json.loads(subprocess.check_output([c.PYTHON, '-c', program, *map(str, args)],
        env=c.environment(instance), text=True, timeout=300))


def prepare(instance):
    c.verify_members(c.CONTROL, c.read(c.CONTROL / 'FILES.json'))
    c.verify_members(c.CONTROL, c.read(c.CONTROL / 'INPUTS.json'))
    c.verify_source()
    # These two exact runtime paths are absent on IN/TX. Existing Nebraska paths
    # are immutable originals, not replaced by links or a different installation.
    for original, payload in ((Path('/workspace/jepa-maze-python'), c.PAYLOAD / 'maze-python'),
                              (Path('/root/.mujoco/mujoco210'), c.PAYLOAD / 'mujoco210')):
        if not original.exists() and not original.is_symlink():
            original.parent.mkdir(parents=True, exist_ok=True)
            original.symlink_to(payload, target_is_directory=True)
        if c.file_manifest(original) != c.file_manifest(payload):
            raise ValueError('Existing runtime differs from allowlisted source')
    program = '''import json,pathlib,sys,subprocess,argparse
from offline_study.navigation_coupling_behavior import read_contract
from offline_study.model_loader import verified_local_dino_cache
from offline_study.vendor import use_vendor
import torch,mujoco_py
vendor=pathlib.Path('/workspace/jepa_steering/vendor/jepa-wms')
assert subprocess.check_output(['git','-C',str(vendor),'rev-parse','HEAD'],text=True).strip()=='13cf1d9c7e476f53c17714d2e0f1dc239a883ce0'
assert not subprocess.check_output(['git','-C',str(vendor),'status','--porcelain'],text=True).strip()
use_vendor(vendor)
for values in json.loads(sys.argv[1]):
 args=argparse.Namespace(**{values[i][2:].replace('-','_'):(values[i+1] if values[i]=='--task' else pathlib.Path(values[i+1])) for i in range(0,len(values),2)})
 read_contract(args)
with verified_local_dino_cache() as proof: encoder=dict(proof)
print(json.dumps({'frozen_contracts_verified':2,'encoder':encoder,'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'gpu_calls':0}))
'''
    proof = python_check(instance, program, json.dumps([c.arguments(t) for t in c.TASKS]))
    workers = {}
    for name, (number, gpu, task) in c.WORKERS.items():
        if number != instance:
            continue
        item = {'instance': number, 'gpu': gpu, 'gpu_uuid': c.gpu_uuid(gpu)}
        if name.startswith('ne'):
            pid = 4253 if task == 'pointmaze' else 4104
            expected = [c.PYTHON, '-u', str(c.ROOT / 'run_navigation_behavior_queue_v2.py'), '--task', task, '--gpu', str(gpu)]
            parent = c.process(pid)
            if parent is None or parent['command'] != expected or not c.alive(parent):
                raise ValueError('Original navigation parent changed before preparation')
            item['parent'] = parent
            item['engineering_report_sha256'] = c.digest(c.ORIGINAL / task / 'engineering/report.json')
        elif name.startswith('tx'):
            pid = 1296 + gpu
            expected = ['/workspace/jepa-droid-python/bin/python', '-u', str(c.ROOT / 'droid-coupling-code-20260908-v3/run_droid_parallel_queue.py'), '--gpu', str(gpu)]
            parent = c.process(pid)
            if parent is None or parent['command'] != expected or not c.alive(parent):
                raise ValueError('Original DROID predecessor changed before preparation')
            item['predecessor'] = {**parent, 'gpu_uuid': item['gpu_uuid']}
        workers[name] = item
    if instance == 50231985:
        watcher = c.process(4384)
        if watcher is None or watcher['command'] != [c.PYTHON, '-u', str(c.ROOT / 'finish_navigation_behavior_panel_v1.py')]:
            raise ValueError('Original CPU watcher changed')
    else:
        watcher = None
    c.write(c.CONTROL / 'READY.json', {'status': 'navigation_cpu_prepared_not_activation', 'instance': instance,
        'source_sha256': c.SOURCE, 'freeze_sha256': c.FREEZES, 'workers': workers, 'watcher': watcher,
        'files_sha256': c.digest(c.CONTROL / 'FILES.json'), 'inputs_sha256': c.digest(c.CONTROL / 'INPUTS.json'),
        'cpu_verification': proof, 'parent_signals': 0, 'gpu_calls': 0})
    print(json.dumps({'ready_sha256': c.digest(c.CONTROL / 'READY.json'), 'instance': instance}))


def children(parent, include_terminal=False):
    ids = (Path('/proc') / str(parent['pid']) / 'task' / str(parent['pid']) / 'children').read_text().split()
    return [p for p in (c.process(int(pid)) for pid in ids) if p and (include_terminal or p['state'] not in ('Z', 'X'))]


def wait_terminal(binding, timeout=7200):
    deadline = time.monotonic() + timeout
    while c.alive(binding):
        if time.monotonic() > deadline:
            raise TimeoutError('Bound process still active; never kill its scientific child')
        time.sleep(2)


def validate_child(child, worker):
    command = child['command']
    prefix = [c.PYTHON, '-u', '-m', 'offline_study.navigation_coupling_behavior', 'run']
    if child['ppid'] != worker['parent']['pid'] or command[:5] != prefix:
        raise ValueError('Unrecognized current child')
    task = c.WORKERS['ne' + str(worker['gpu'])][2]
    values = dict(zip(command[5::2], command[6::2]))
    job = {'task': values['--task'], 'arm': values['--arm'], 'rank': int(values['--logical-ranks'])}
    c.key(job)
    original_evidence = c.ROOT / 'navigation-coupling-evidence-20260908-v2/artifacts/offline_study'
    expected = c.arguments(task)
    expected = [v.replace(str(c.EVIDENCE), str(original_evidence)).replace(str(c.PAYLOAD / 'models'), str(c.ROOT / 'navigation-assets-20260907-v1/downloads/model')) for v in expected]
    target = c.ORIGINAL / c.relative(job)
    expected = prefix + expected + ['--engineering', str(c.ORIGINAL / task / 'engineering'),
        '--arm', job['arm'], '--logical-ranks', str(job['rank']), '--output', str(target)]
    if job['task'] != task or command != expected:
        raise ValueError('Original full child argv differs')
    return job, target


def boundary():
    """Only a root-reviewed authorization permits parent signals. Children drain."""
    ready = c.read(c.CONTROL / 'READY.json')
    c.require_authority('boundary', ready_hash=c.digest(c.CONTROL / 'READY.json'))
    if ready['instance'] != 50231985:
        raise ValueError('Boundary belongs only to Nebraska')
    all_ready = c.read(c.CONTROL / 'ALL_READY.json')
    if all_ready['50231985'] != ready:
        raise ValueError('Nebraska preparation differs')
    status = c.CONTROL / 'boundary'; status.mkdir(exist_ok=False)
    paused = []
    committed = False
    def cancelled(signum, frame):
        raise RuntimeError('Boundary cancelled; keep scientific children intact')
    signal.signal(signal.SIGTERM, cancelled); signal.signal(signal.SIGINT, cancelled)
    try:
        c.verify_source()
        for name in ('ne0', 'ne1'):
            worker = ready['workers'][name]
            parent = worker['parent']
            if c.gpu_uuid(worker['gpu']) != worker['gpu_uuid'] or not c.alive(parent):
                raise ValueError('Original parent/device changed')
            c.write(status / (name + '-PAUSE_INTENT.json'), {'parent': parent})
            if not c.send(parent, signal.SIGSTOP):
                raise ValueError('Original parent exited before pause')
            paused.append(worker)
            deadline = time.monotonic() + 5
            while c.process(parent['pid'])['state'] not in ('T', 't'):
                if time.monotonic() > deadline:
                    raise TimeoutError('Original parent did not stop')
                time.sleep(.05)
        # Pause both producers before any long drain. Reinspect their child after
        # suspension, including any final exec race, rather than using a stale PID.
        for worker in paused:
            bound = children(worker['parent'], include_terminal=True)
            live = [child for child in bound if child['state'] not in ('Z', 'X')]
            if len(bound) > 1:
                raise ValueError('Unexpected original independent child overlap')
            if live:
                child = live[0]
                if not child['command']:
                    deadline = time.monotonic() + 5
                    while not child['command'] and time.monotonic() < deadline:
                        time.sleep(.05); child = c.process(child['pid'])
                        if child is None or child['state'] in ('Z', 'X'):
                            break
                if child and child['state'] not in ('Z', 'X'):
                    job, target = validate_child(child, worker)
                    # Bind the verified post-exec argv, not an earlier empty
                    # fork snapshot, for the subsequent identity/release guard.
                    bound = [child]
                    c.write(status / ('ne' + str(worker['gpu']) + '-DRAIN.json'), {'child': child, 'job': job})
                    wait_terminal(child)
                    c.verify_shard(target, job)
            observations = []
            try:
                release = c.wait_gpu_release(worker['gpu'], worker['gpu_uuid'], bound,
                    paused_parent=worker['parent'], observations=observations)
                c.write(status / ('ne' + str(worker['gpu']) + '-GPU_RELEASE.json'), release)
            finally:
                c.write(status / ('ne' + str(worker['gpu']) + '-GPU_RELEASE_OBSERVATIONS.json'),
                    {'observations': observations, 'production_signals_in_release_guard': 0})
        completed, pristine = c.inventory()
        plan = {'status': 'intact_navigation_redistribution_frozen', 'source_sha256': c.SOURCE,
            'freeze_sha256': c.FREEZES, 'completed': completed, 'workers': c.allocation(pristine, all_ready),
            'ready_sha256': {k: __import__('hashlib').sha256((json.dumps(v, indent=2, sort_keys=True) + '\n').encode()).hexdigest() for k, v in all_ready.items()},
            'total_candidate_streams': 128, 'episodes_per_task_condition': 96,
            'outcome_based_selection': False, 'old_children_completed_untouched': True}
        c.validate_plan(plan)
        c.write(c.CONTROL / 'PLAN.json', plan)
        committed = True  # No old producer may resume after this ownership commit.
        watcher = ready['watcher']
        if c.alive(watcher):
            if children(watcher):
                raise ValueError('Old watcher has a child; do not interrupt analysis')
            c.send(watcher, signal.SIGTERM)
            wait_terminal(watcher, 30)
        for worker in paused:
            parent = worker['parent']
            if children(parent) or c.gpu_processes(worker['gpu']):
                raise ValueError('Refuse parent termination while any child/GPU remains active')
            c.send(parent, signal.SIGTERM)
            c.send(parent, signal.SIGCONT)
            wait_terminal(parent, 30)
        c.write(c.CONTROL / 'CUTOVER.json', {'plan_sha256': c.digest(c.CONTROL / 'PLAN.json'),
            'old_parents_terminal': True, 'old_watcher_terminal': True, 'old_children_not_interrupted': True,
            'completed_streams': len(completed), 'assigned_streams': len(pristine)})
    except BaseException as error:
        c.write(status / 'FAILED.json', {'error': str(error), 'ownership_committed': committed,
            'automatic_retry': False, 'scientific_children_not_killed': True})
        raise
    finally:
        if not committed:
            signal.signal(signal.SIGTERM, signal.SIG_IGN); signal.signal(signal.SIGINT, signal.SIG_IGN)
            for worker in paused:
                parent = worker['parent']
                # An uncommitted handoff launches nothing, so safely resume the
                # original supervisor even if its own untouched child is still live.
                if c.alive(parent) and c.process(parent['pid'])['state'] in ('T', 't'):
                    c.send(parent, signal.SIGCONT)
                    c.write(status / ('ne' + str(worker['gpu']) + '-RESUMED.json'), {'no_assignment_committed': True})


def engineering_proof(instance, task, path):
    program = '''import json,pathlib,sys
from offline_study.navigation_coupling_behavior import validate_engineering
p=pathlib.Path(sys.argv[1]); protocol=json.loads((p/'protocol.json').read_text())
print(json.dumps({'report_sha256':validate_engineering(pathlib.Path(sys.argv[2]),protocol,sys.argv[3])}))
'''
    return python_check(instance, program, c.freeze_path(task), path, c.FREEZES[task])['report_sha256']


def early_root():
    return c.CONTROL / 'early' / 'in0-wall'


def early_command():
    return [c.PYTHON, '-u', '-m', 'offline_study.navigation_coupling_behavior', 'engineer'] + c.arguments('wall') + ['--output', str(early_root() / 'engineering')]


def profile_release(worker):
    release = c.read(c.CONTROL / 'PROFILE_RELEASE.json')
    if (release.get('owner') != 'rep_geometry_transcoder/root' or
            release.get('gpu_uuid') != worker['gpu_uuid'] or release.get('profiling_finished') is not True):
        raise ValueError('Root profiling/cache pilot must explicitly release Indiana first')


def validate_early_binding(launch, worker, ready_hash):
    if (launch.get('status') != 'early_original_wall_engineering' or launch.get('worker') != 'in0' or
            any(launch.get(k) != worker[k] for k in ('instance', 'gpu', 'gpu_uuid')) or
            launch.get('instance') != 50205763 or launch.get('gpu') != 0 or
            launch.get('source_sha256') != c.SOURCE or launch.get('freeze_sha256') != c.FREEZES['wall'] or
            launch.get('ready_sha256') != ready_hash or launch.get('original_engineering_only') is not True):
        raise ValueError('Early proof belongs to another source/task/device/preparation')


def run_early():
    """Exactly the original 11-case Wall suite once; no PLAN or science needed."""
    ready_hash = c.digest(c.CONTROL / 'READY.json')
    c.require_authority('early', ready_hash=ready_hash)
    ready = c.read(c.CONTROL / 'READY.json')
    if ready['instance'] != 50205763:
        raise ValueError('Only Indiana Wall early engineering is authorized')
    worker = ready['workers']['in0']
    profile_release(worker)
    c.verify_source()
    if c.gpu_uuid(0) != worker['gpu_uuid'] or c.gpu_processes(0):
        raise ValueError('Indiana device changed or is still occupied')
    root = early_root(); root.mkdir(parents=True, exist_ok=False)
    identity, running = c.process(os.getpid()), None
    c.write(root / 'LAUNCH.json', {'status': 'early_original_wall_engineering', 'worker': 'in0',
        **{k: worker[k] for k in ('instance', 'gpu', 'gpu_uuid')}, 'identity': identity,
        'source_sha256': c.SOURCE, 'freeze_sha256': c.FREEZES['wall'], 'ready_sha256': ready_hash,
        'original_engineering_only': True, 'scientific_episodes_started': False})
    def cancelled(signum, frame):
        raise RuntimeError('Early engineering cancelled; preserve and drain its original suite')
    signal.signal(signal.SIGTERM, cancelled); signal.signal(signal.SIGINT, cancelled)
    try:
        with (root / 'engineering.log').open('x') as log:
            running = subprocess.Popen(early_command(), env=c.environment(50205763, 0),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            child = c.started(running.pid, early_command())
            c.write(root / 'CHILD.json', child)
            if running.wait(timeout=7200):
                raise ValueError('Original receiving engineering failed; no automatic retry')
        release = c.wait_gpu_release(0, worker['gpu_uuid'], [child])
        c.write(root / 'GPU_RELEASE.json', release)
        proof_hash = engineering_proof(50205763, 'wall', root / 'engineering')
        c.write(root / 'DONE.json', {'status': 'early_navigation_engineering_complete',
            'launch_sha256': c.digest(root / 'LAUNCH.json'), 'child_sha256': c.digest(root / 'CHILD.json'),
            'gpu_release_sha256': c.digest(root / 'GPU_RELEASE.json'), 'engineering_report_sha256': proof_hash,
            'source_sha256': c.SOURCE, 'freeze_sha256': c.FREEZES['wall'], 'gpu_uuid': worker['gpu_uuid']})
    except BaseException as error:
        c.write(root / 'FAILED.json', {'error': str(error), 'automatic_retry': False,
            'child_not_killed': True, 'child_pid': None if running is None else running.pid})
        raise
    finally:
        if running is not None and running.poll() is None:
            signal.signal(signal.SIGTERM, signal.SIG_IGN); signal.signal(signal.SIGINT, signal.SIG_IGN)
            running.wait()


def wait_early_proof(worker, ready_hash):
    """The final worker waits for its owned early suite and reuses it, never repeats."""
    root = early_root()
    launch = c.read(root / 'LAUNCH.json')
    validate_early_binding(launch, worker, ready_hash)
    deadline = time.monotonic() + 7200
    while not (root / 'DONE.json').exists():
        if (root / 'FAILED.json').exists():
            raise ValueError('Early receiving suite failed; do not duplicate/retry it')
        if not c.alive(launch['identity']) or time.monotonic() > deadline:
            if not (root / 'DONE.json').exists():
                raise ValueError('Early supervisor ended without verified completion')
        time.sleep(2)
    if (root / 'FAILED.json').exists():
        raise ValueError('Failed early suite cannot authorize science')
    wait_terminal(launch['identity'], 60)
    done, child = c.read(root / 'DONE.json'), c.read(root / 'CHILD.json')
    if (done['status'] != 'early_navigation_engineering_complete' or
            done['launch_sha256'] != c.digest(root / 'LAUNCH.json') or done['child_sha256'] != c.digest(root / 'CHILD.json') or
            done['gpu_release_sha256'] != c.digest(root / 'GPU_RELEASE.json') or
            done['source_sha256'] != c.SOURCE or done['freeze_sha256'] != c.FREEZES['wall'] or
            done['gpu_uuid'] != worker['gpu_uuid'] or child['ppid'] != launch['identity']['pid'] or
            child['command'] != early_command()):
        raise ValueError('Early completion/child identity changed')
    c.wait_gpu_release(worker['gpu'], worker['gpu_uuid'], [child])
    proof_hash = engineering_proof(50205763, 'wall', root / 'engineering')
    if proof_hash != done['engineering_report_sha256']:
        raise ValueError('Early scientific engineering proof changed')
    return {'path': str(root / 'engineering'), 'report_sha256': proof_hash,
            'gpu_uuid': worker['gpu_uuid'], 'early_done_sha256': c.digest(root / 'DONE.json')}


def droid_gate(worker):
    gpu, predecessor = worker['gpu'], worker['predecessor']
    panel = c.ROOT / 'droid-coupling-behavior-20260908-v3'
    queue = panel / f'queue-gpu{gpu}'
    deadline = time.monotonic() + 36 * 3600
    while not (queue / 'DONE.json').exists():
        if (queue / 'FAILED.json').exists() or not c.alive(predecessor) or time.monotonic() > deadline:
            # Publication and exit can race the /proc snapshot; recheck DONE.
            if not (queue / 'DONE.json').exists():
                raise ValueError('DROID predecessor ended without its whole assignment')
        time.sleep(5)
    ranks = c.verify_droid_completion(panel, gpu, predecessor)
    wait_terminal(predecessor, 60)
    program = '''import json,pathlib,sys
from offline_study.droid_coupling_behavior import scientific_shard
from offline_study.author_fit import source_hash
p=pathlib.Path(sys.argv[1]); protocol=json.loads((p/'freeze/protocol.json').read_text())
assert source_hash()==sys.argv[2]
hashes={}
for rank in json.loads(sys.argv[3]):
 for arm in protocol['arms']:
  _,_,h=scientific_shard(p/'conditions'/arm/f'shard-{rank}',protocol,sys.argv[4],rank,arm,sys.argv[5])
  hashes[arm+':'+str(rank)]=h
print(json.dumps({'verified_shards':len(hashes),'verified_endpoints':144,'report_hashes':hashes}))
'''
    env = c.environment(50259194)
    env['PYTHONPATH'] = str(c.ROOT / 'droid-coupling-code-20260908-v3/src')
    result = json.loads(subprocess.check_output(['/workspace/jepa-droid-python/bin/python', '-c', program,
        str(panel), c.DROID_SOURCE, json.dumps(ranks), c.DROID_FREEZE, worker['gpu_uuid'].removeprefix('GPU-')],
        env=env, text=True, timeout=300))
    if result['verified_shards'] != 18 or c.gpu_processes(gpu):
        raise ValueError('DROID proof incomplete or GPU not released')
    return result


def run_worker(name):
    plan = c.read(c.CONTROL / 'PLAN.json'); c.validate_plan(plan)
    plan_hash = c.digest(c.CONTROL / 'PLAN.json')
    c.require_authority('run', plan_hash=plan_hash)
    if c.read(c.CONTROL / 'CUTOVER.json')['plan_sha256'] != plan_hash:
        raise ValueError('Verified Nebraska ownership cutover required')
    worker = plan['workers'][name]
    ready = c.read(c.CONTROL / 'READY.json')
    if ready['instance'] != worker['instance'] or c.digest(c.CONTROL / 'READY.json') != plan['ready_sha256'][str(worker['instance'])]:
        raise ValueError('Receiving preparation changed')
    if c.gpu_uuid(worker['gpu']) != worker['gpu_uuid']:
        raise ValueError('Assigned GPU identity changed')
    c.verify_source()
    status = c.CONTROL / 'workers' / name; status.mkdir(parents=True, exist_ok=False)
    identity = c.process(os.getpid())
    c.write(status / 'LAUNCH.json', {'worker': name, 'identity': identity, 'plan_sha256': plan_hash,
        'instance': worker['instance'], 'gpu': worker['gpu'], 'gpu_uuid': worker['gpu_uuid'], 'jobs': worker['jobs']})
    running = None
    def cancelled(signum, frame):
        raise RuntimeError('Worker cancelled; drain current intact stream, do not schedule another')
    signal.signal(signal.SIGTERM, cancelled); signal.signal(signal.SIGINT, cancelled)
    def execute(command, label):
        nonlocal running
        if c.gpu_processes(worker['gpu']):
            raise ValueError('Exclusive assigned GPU is occupied')
        with (status / (label + '.log')).open('x') as log:
            running = subprocess.Popen(command, env=c.environment(worker['instance'], worker['gpu']),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            binding = c.started(running.pid, command)
            c.write(status / (label + '-CHILD.json'), binding)
            if running.wait(timeout=7200):
                raise ValueError('Required unchanged scientific stage failed: ' + label)
        running = None
    try:
        if name.startswith('tx'):
            c.write(status / 'PREDECESSOR_VERIFIED.json', droid_gate(worker))
        if name == 'in0':
            profile_release(worker)
        early_proofs = {}
        if name == 'in0' and early_root().exists():
            early_proofs['wall'] = wait_early_proof(worker, c.digest(c.CONTROL / 'READY.json'))
            c.write(status / 'EARLY_ENGINEERING_REUSED.json', early_proofs['wall'])
        if c.gpu_processes(worker['gpu']):
            raise ValueError('Assigned device not released')
        proofs = {}
        for task in c.TASKS:
            selected = [j for j in worker['jobs'] if j['task'] == task]
            if not selected:
                continue
            path = (Path(early_proofs[task]['path']) if task in early_proofs else
                    c.ORIGINAL / task / 'engineering' if name.startswith('ne') else status / ('engineering-' + task))
            if not name.startswith('ne') and task not in early_proofs:
                execute([c.PYTHON, '-u', '-m', 'offline_study.navigation_coupling_behavior', 'engineer'] + c.arguments(task) + ['--output', str(path)], 'engineering-' + task)
            proof_hash = engineering_proof(worker['instance'], task, path)
            if name.startswith('ne') and proof_hash != worker['engineering_report_sha256']:
                raise ValueError('Original same-device engineering changed')
            proofs[task] = {'path': str(path), 'report_sha256': proof_hash, 'gpu_uuid': worker['gpu_uuid']}
            c.write(status / ('ENGINEERING-' + task + '.json'), proofs[task])
            for job in selected:
                target = c.CONTROL / 'results' / c.relative(job)
                if target.exists():
                    raise FileExistsError('Assigned result already exists; never retry/overwrite')
                execute([c.PYTHON, '-u', '-m', 'offline_study.navigation_coupling_behavior', 'run'] + c.arguments(task) +
                    ['--engineering', str(path), '--arm', job['arm'], '--logical-ranks', str(job['rank']), '--output', str(target)],
                    task + '-' + job['arm'] + '-' + str(job['rank']))
                c.verify_shard(target, job, proof_hash)
        completed = [{**j, 'output': str(c.CONTROL / 'results' / c.relative(j)),
            'report_sha256': c.verify_shard(c.CONTROL / 'results' / c.relative(j), j, proofs[j['task']]['report_sha256'])} for j in worker['jobs']]
        c.write(status / 'DONE.json', {'status': 'navigation_redistribution_assignment_complete', 'worker': name,
            'plan_sha256': plan_hash, 'source_sha256': c.SOURCE, 'freeze_sha256': c.FREEZES,
            'identity': identity, 'instance': worker['instance'], 'gpu': worker['gpu'], 'gpu_uuid': worker['gpu_uuid'],
            'engineering': proofs, 'completed_jobs': completed, 'endpoints': 12 * len(completed),
            'launch_sha256': c.digest(status / 'LAUNCH.json'), 'gpu_available_only_after_process_exit': True})
    except BaseException as error:
        c.write(status / 'FAILED.json', {'error': str(error), 'scientific_child_not_killed': True,
            'child_pid': None if running is None else running.pid, 'automatic_retry': False})
        raise
    finally:
        if running is not None and running.poll() is None:
            signal.signal(signal.SIGTERM, signal.SIG_IGN); signal.signal(signal.SIGINT, signal.SIG_IGN)
            # Cancellation/failure never truncates an in-flight original 12-stream.
            running.wait()


def verify_completed(name):
    plan = c.read(c.CONTROL / 'PLAN.json'); c.validate_plan(plan)
    worker = plan['workers'][name]
    status = c.CONTROL / 'workers' / name
    done, launch = c.read(status / 'DONE.json'), c.read(status / 'LAUNCH.json')
    if (status / 'FAILED.json').exists():
        raise ValueError('Failed administrative worker is not complete')
    if (done['status'] != 'navigation_redistribution_assignment_complete' or done['worker'] != name or
            done['plan_sha256'] != c.digest(c.CONTROL / 'PLAN.json') or done['source_sha256'] != c.SOURCE or
            done['freeze_sha256'] != c.FREEZES or done['launch_sha256'] != c.digest(status / 'LAUNCH.json') or
            any(done[k] != worker[k] or launch[k] != worker[k] for k in ('instance', 'gpu', 'gpu_uuid')) or
            done['identity'] != launch['identity'] or launch['jobs'] != worker['jobs'] or
            done['endpoints'] != 12 * len(worker['jobs'])):
        raise ValueError('Worker completion identity/coverage changed')
    keys = ('task', 'arm', 'rank')
    if [{k: j[k] for k in keys} for j in done['completed_jobs']] != worker['jobs']:
        raise ValueError('Missing/duplicate/reordered completed assignment')
    for job in worker['jobs']:
        row = next(x for x in done['completed_jobs'] if all(x[k] == job[k] for k in keys))
        path = c.CONTROL / 'results' / c.relative(job)
        proof = done['engineering'][job['task']]
        if row['output'] != str(path) or proof['gpu_uuid'] != worker['gpu_uuid'] or c.verify_shard(path, job, proof['report_sha256']) != row['report_sha256']:
            raise ValueError('Completed shard/proof changed')
    # Invoke the unchanged scientific validators in a fresh CPU-only process;
    # expose only count/hash metadata, not outcomes or treatment estimates.
    program = '''import argparse,json,pathlib,sys
from offline_study.navigation_coupling_behavior import read_contract,paired_inputs,validate_engineering
from offline_study.navigation_replication import validate_records
from offline_study.behavioral_development import assigned_rows,schedule
from offline_study.vendor import use_vendor
use_vendor(pathlib.Path('/workspace/jepa_steering/vendor/jepa-wms'))
jobs=json.loads(sys.argv[1]); proofs=json.loads(sys.argv[2]); registry=json.loads(sys.argv[3]); completed=0
for values in registry:
 args=argparse.Namespace(**{values[i][2:].replace('-','_'):(values[i+1] if values[i]=='--task' else pathlib.Path(values[i+1])) for i in range(0,len(values),2)})
 if args.task not in proofs:continue
 protocol,_,native=read_contract(args)
 assert validate_engineering(pathlib.Path(proofs[args.task]['path']),protocol,sys.argv[4] if args.task=='wall' else sys.argv[5])==proofs[args.task]['report_sha256']
 for job in jobs:
  if job['task']!=args.task:continue
  path=pathlib.Path(job['output']); report=json.loads((path/'report.json').read_text())
  rows=[json.loads((path/n).read_text()) for n in sorted(report['episode_files_sha256'])]
  validate_records(rows,assigned_rows(schedule(),[job['rank']]))
  for row in rows:paired_inputs(native[row['episode']],row)
  completed+=len(rows)
print(json.dumps({'verified_endpoints':completed,'scientific_validators_unchanged':True}))
'''
    science = python_check(worker['instance'], program, json.dumps(done['completed_jobs']), json.dumps(done['engineering']),
        json.dumps([c.arguments(t) for t in c.TASKS]), c.FREEZES['wall'], c.FREEZES['pointmaze'])
    if science['verified_endpoints'] != done['endpoints']:
        raise ValueError('Scientific coverage mismatch')
    return {'worker': name, 'done_sha256': c.digest(status / 'DONE.json'), 'plan_sha256': done['plan_sha256'],
        'identity': done['identity'], 'instance': worker['instance'], 'gpu': worker['gpu'], 'gpu_uuid': worker['gpu_uuid'], **science}


def export_result(job, original=False):
    path = (c.ORIGINAL if original else c.CONTROL / 'results') / c.relative(job)
    c.verify_shard(path, job)
    manifest = c.file_manifest(path)
    if any(set(value) != {'bytes', 'sha256'} for value in manifest.values()):
        raise ValueError('Scientific shard must contain only regular files')
    with tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz') as archive:
        content = json.dumps(manifest, sort_keys=True).encode()
        info = tarfile.TarInfo('MEMBERS.json'); info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
        for name in manifest:
            archive.add(path / name, arcname=name, recursive=False)
    if c.file_manifest(path) != manifest:
        raise ValueError('Completed scientific shard changed during collection')


def import_result(job):
    c.validate_plan(c.read(c.CONTROL / 'PLAN.json'))
    path = c.CONTROL / 'aggregate' / c.relative(job)
    with tarfile.open(fileobj=sys.stdin.buffer, mode='r|gz') as archive:
        members = iter(archive)
        first = next(members)
        if first.name != 'MEMBERS.json' or not first.isfile():
            raise ValueError('Scientific member manifest required first')
        manifest = json.load(archive.extractfile(first))
        existing = path.exists()
        if existing:
            # Repeated collection is verification-only; partial/different outputs
            # are not overwritten or silently completed by this recovery path.
            if c.file_manifest(path) != manifest:
                raise FileExistsError('Existing collection differs or is incomplete')
            c.verify_shard(path, job)
        else:
            path.mkdir(parents=True, exist_ok=False)
        seen = set()
        for member in members:
            name = member.name
            if name in seen or name not in manifest or not member.isfile() or len(c.safe_member(name).parts) != 1:
                raise ValueError('Unsafe or duplicate scientific member')
            content = archive.extractfile(member).read()
            wanted = manifest[name]
            if len(content) != wanted['bytes'] or __import__('hashlib').sha256(content).hexdigest() != wanted['sha256']:
                raise ValueError('Scientific transfer hash changed')
            if not existing:
                with (path / name).open('xb') as stream:
                    stream.write(content)
            seen.add(name)
        if seen != set(manifest):
            raise ValueError('Incomplete scientific transfer; preserve partial')
    report_hash = c.verify_shard(path, job)
    print(json.dumps({'job': job, 'report_sha256': report_hash, 'existing_verified_not_overwritten': existing}))


def analyze():
    plan = c.read(c.CONTROL / 'PLAN.json'); c.validate_plan(plan)
    complete, missing = c.inventory(c.CONTROL / 'aggregate')
    if missing or len(complete) != 128:
        raise ValueError('Whole two-task 128-stream panel required before analysis')
    verified = c.read(c.CONTROL / 'COLLECTION_COMPLETE.json')
    if verified['plan_sha256'] != c.digest(c.CONTROL / 'PLAN.json') or set(verified['workers']) != set(c.WORKERS):
        raise ValueError('All receiving assignment verifications required')
    for task in c.TASKS:
        target = c.CONTROL / 'aggregate' / task / 'freeze'
        if not target.exists():
            shutil.copytree(c.freeze_path(task), target)
        if c.digest(target / 'protocol.json') != c.FREEZES[task]:
            raise ValueError('Collected freeze differs')
    c.verify_source()
    subprocess.run([c.PYTHON, '-u', '-m', 'offline_study.navigation_coupling_analysis',
        '--root', str(c.CONTROL / 'aggregate'), '--reference', str(c.EVIDENCE / 'navigation-coupling-reference-20260908-v1'),
        '--output', str(c.CONTROL / 'analysis')], env=c.environment(50231985), check=True, timeout=3600)
    print(json.dumps({'analysis_complete': True, 'full_study_complete': False, 'fresh_confirmation': False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('inventory-inputs', 'export-inputs', 'receive-inputs', 'prepare', 'boundary', 'early', 'run', 'verify-completed', 'export-result', 'import-result', 'analyze'))
    parser.add_argument('--instance', type=int, choices=(50231985, 50205763, 50259194))
    parser.add_argument('--worker', choices=tuple(c.WORKERS))
    parser.add_argument('--job', type=json.loads)
    parser.add_argument('--original', action='store_true')
    args = parser.parse_args()
    if args.command == 'inventory-inputs': print(json.dumps(input_manifest(), sort_keys=True))
    elif args.command == 'export-inputs': export_inputs()
    elif args.command == 'receive-inputs': receive_inputs()
    elif args.command == 'prepare': prepare(args.instance)
    elif args.command == 'boundary': boundary()
    elif args.command == 'early': run_early()
    elif args.command == 'run': run_worker(args.worker)
    elif args.command == 'verify-completed': print(json.dumps(verify_completed(args.worker)))
    elif args.command == 'export-result': export_result(args.job, args.original)
    elif args.command == 'import-result': import_result(args.job)
    else: analyze()


if __name__ == '__main__':
    main()
