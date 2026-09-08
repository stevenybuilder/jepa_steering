"""Operations-only navigation relocation; never changes scientific source or seeds."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import routing_priority_common as lifecycle

ROOT = Path('/workspace/jepa-runtime')
CONTROL = ROOT / 'navigation-redistribution-20260908-v1'
PAYLOAD = CONTROL / 'payload'
CODE = PAYLOAD / 'code'
EVIDENCE = PAYLOAD / 'evidence/artifacts/offline_study'
ORIGINAL = ROOT / 'navigation-coupling-behavior-20260908-v1'
PYTHON = '/workspace/jepa-planning-python/bin/python'
SOURCE = 'fc7578672eabd8e08469afcc1f7f4d9b1c2ed18c447f4f003539b61da7950f58'
FREEZES = {'wall': 'd41381c52313e536e53ca80ccbf8324756bf665d248f436282622f9085fc7af3',
           'pointmaze': '2d94a21b4554670dfb289baf7018291be7764b01a00be093a98ac19a3afe1fbf'}
CHECKPOINTS = {'wall': '8efb0623cfba1cb3ca210de26f7579c83dd24936635f11989c515afcb23bea1e',
               'pointmaze': 'a01d99c4592fbedf44af076cf4c339de230c56f9f377c7559f584b97569b59bc'}
TASKS = ('wall', 'pointmaze')
ARMS = ('visual_only', 'action_condition_only', 'joint', 'joint_equal_standardized_energy',
        'permuted_visual', 'permuted_joint', 'matched_random', 'matched_random_equal_standardized_energy')
DEV_SEED = 2026090721
DROID_SOURCE = '249a8cdb9326cd79f3b9180a830e409cdd0ffa4f2f3be2ac393c32e30ab81a54'
DROID_FREEZE = '369d632792b6b482f5c63bbcd599dd7576cf949001ac005baa1cb5f600681862'
DRIVER = '/usr/lib/x86_64-linux-gnu/libcuda.so.1'
DRIVER_SHA = 'b7759e577409c86949f455f18fcc044d72b466be5e2982b6ba9e20348e86fa97'
WORKERS = {'ne0': (50231985, 0, 'pointmaze'), 'ne1': (50231985, 1, 'wall'),
           'in0': (50205763, 0, None), 'tx0': (50259194, 0, 'wall'),
           'tx1': (50259194, 1, 'wall'), 'tx2': (50259194, 2, 'pointmaze'),
           'tx3': (50259194, 3, 'pointmaze')}
read = lambda path: json.loads(Path(path).read_text())
write = lifecycle.write
process = lifecycle.process
alive = lifecycle.alive
send = lifecycle.send
started = lifecycle.started
gpu_processes = lifecycle.gpu_processes


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            value.update(block)
    return value.hexdigest()


def gpu_uuid(gpu):
    return 'GPU-' + lifecycle.gpu_uuid(gpu)


def key(job):
    if set(job) != {'task', 'arm', 'rank'} or job['task'] not in TASKS or job['arm'] not in ARMS or type(job['rank']) is not int or job['rank'] not in range(8):
        raise ValueError('Invalid intact stream identity')
    return job['task'], job['arm'], job['rank']


def jobs():
    return [{'task': t, 'arm': a, 'rank': r} for t in TASKS for a in ARMS for r in range(8)]


def relative(job):
    task, arm, rank = key(job)
    return Path(task) / 'conditions' / arm / f'shard-{rank}'


def expected_rows(rank):
    local = DEV_SEED + rank * 6000
    return [{'episode': e, 'logical_rank': rank, 'local_seed': local,
             'environment_seed': (local * local + e * local) % (2**32 - 2)}
            for e in range(rank * 12, (rank + 1) * 12)]


def source_hash(root):
    value = hashlib.sha256()
    for path in sorted(Path(root).glob('*.py')):
        value.update(path.name.encode() + b'\0' + path.read_bytes())
    return value.hexdigest()


def safe_member(name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or not path.parts or any(p.startswith('._') or p == '__pycache__' for p in path.parts):
        raise ValueError('Unsafe manifest path')
    return path


def file_manifest(root):
    result = {}
    for path in sorted(Path(root).rglob('*')):
        name = str(path.relative_to(root))
        if '__pycache__' in path.parts or path.name.startswith('._'):
            continue
        if path.is_symlink():
            result[name] = {'symlink': os.readlink(path)}
        elif path.is_file():
            result[name] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
    return result


def verify_members(root, manifest):
    for name, wanted in manifest.items():
        path = Path(root) / safe_member(name)
        if set(wanted) == {'symlink'}:
            if not path.is_symlink() or os.readlink(path) != wanted['symlink']:
                raise ValueError('Symlink changed: ' + name)
        elif set(wanted) != {'bytes', 'sha256'} or path.is_symlink() or path.stat().st_size != wanted['bytes'] or digest(path) != wanted['sha256']:
            raise ValueError('Member changed: ' + name)


def freeze_path(task):
    return EVIDENCE / 'navigation-coupling-behavior-20260908-v1' / task / 'freeze'


def arguments(task):
    fit = EVIDENCE / 'primary-durable-20260907/navigation-fits-20260907-v1/bfloat16' / task
    return ['--vendor', '/workspace/jepa_steering/vendor/jepa-wms', '--task', task,
            '--reference', str(EVIDENCE / 'navigation-coupling-reference-20260908-v1'),
            '--fit', str(fit / 'vision_action_coupling'), '--cohort', str(fit / 'cohort.json'),
            '--freeze', str(freeze_path(task)), '--checkpoint', str(PAYLOAD / f'models/jepa_wm_{task}.pth.tar')]


def environment(instance, gpu=''):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), JEPA_VERIFIED_LOCAL_DINO='1',
        PYTHONDONTWRITEBYTECODE='1', MUJOCO_PY_FORCE_CPU='1', OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        PYTHONPATH=str(CODE / 'src') + ':/workspace/jepa-maze-python/lib/python3.10/site-packages:'
            '/workspace/jepa-planning-python/lib/python3.10/site-packages:/workspace/jepa-python/lib/python3.10/site-packages',
        LD_LIBRARY_PATH='/root/.mujoco/mujoco210/bin:/opt/conda/lib')
    env.pop('LD_PRELOAD', None)
    if instance == 50259194:
        if digest(DRIVER) != DRIVER_SHA:
            raise ValueError('Texas verified host driver changed')
        env['LD_PRELOAD'] = DRIVER
    return env


def verify_source():
    if source_hash(CODE / 'src/offline_study') != SOURCE:
        raise ValueError('Frozen scientific source changed')
    for task in TASKS:
        freeze = freeze_path(task)
        if digest(freeze / 'protocol.json') != FREEZES[task] or read(freeze / 'FROZEN.json')['protocol_sha256'] != FREEZES[task]:
            raise ValueError('Original freeze changed')
        if digest(PAYLOAD / f'models/jepa_wm_{task}.pth.tar') != CHECKPOINTS[task]:
            raise ValueError('Original released checkpoint changed')


def verify_shard(root, job, engineering_hash=None):
    """Receipt/identity verification only. Never exposes outcome metrics for selection."""
    task, arm, rank = key(job)
    root = Path(root)
    if (root / 'FAILED.json').exists():
        raise ValueError('Required shard failed; no retry or replacement')
    done, report, launch = read(root / 'DONE.json'), read(root / 'report.json'), read(root / 'protocol.json')
    if (done['report_sha256'] != digest(root / 'report.json') or
            report['status'] != 'complete_coupling_navigation_shard' or report['episodes'] != 12 or
            report['task'] != task or report['arm'] != arm or report['source_sha256'] != SOURCE or
            report['freeze_sha256'] != FREEZES[task] or report['parameters_unchanged'] is not True or
            report['fresh_confirmation'] is not False or report['scientific_efficacy_measurement'] is not True or
            report['protocol_sha256'] != digest(root / 'protocol.json') or
            launch['source_sha256'] != SOURCE or launch['freeze_sha256'] != FREEZES[task] or
            launch['task'] != task or launch['arm'] != arm or launch['logical_ranks'] != [rank] or
            launch['expected_episodes'] != expected_rows(rank) or launch['engineering_only'] is not False or
            launch['fresh_confirmation'] is not False):
        raise ValueError('Unbound navigation shard')
    if engineering_hash is not None and launch['engineering_report_sha256'] != engineering_hash:
        raise ValueError('Shard is bound to another receiving engineering proof')
    expected = {f"episode-{row['episode']:03d}.json": row for row in expected_rows(rank)}
    if set(report['episode_files_sha256']) != set(expected):
        raise ValueError('Missing or extra episodes')
    for name, row in expected.items():
        if digest(root / name) != report['episode_files_sha256'][name]:
            raise ValueError('Episode hash changed')
        record = read(root / name)
        if record['arm'] != arm or any(record[k] != v for k, v in row.items()) or record['result']['elementary_steps'] != 30:
            raise ValueError('Stream identity/order or complete episode changed')
    return done['report_sha256']


def inventory(root=ORIGINAL):
    completed, pristine = [], []
    for job in jobs():
        path = Path(root) / relative(job)
        if path.exists():
            completed.append({**job, 'report_sha256': verify_shard(path, job)})
        else:
            pristine.append(job)
    return completed, pristine


def allocation(pristine, ready):
    """Fixed runtime-only load balance, no efficacy inputs; preserves whole streams."""
    if len({key(j) for j in pristine}) != len(pristine):
        raise ValueError('Duplicate pristine stream')
    workers = {}
    for name, (instance, gpu, _) in WORKERS.items():
        item = ready[str(instance)]['workers'][name]
        if item['instance'] != instance or item['gpu'] != gpu:
            raise ValueError('Receiving ownership differs')
        workers[name] = {**item, 'jobs': []}
    for task in TASKS:
        candidates = [j for j in pristine if j['task'] == task]
        ne = 'ne1' if task == 'wall' else 'ne0'
        tx = ('tx0', 'tx1') if task == 'wall' else ('tx2', 'tx3')
        # The registered 47-stream snapshot gives 17/8/11/11 exactly. Round the
        # same proportions for later boundaries; no original completed work repeats.
        count = len(candidates)
        ni = min(count, round(count * 8 / 47))
        nn = min(count - ni, round(count * 17 / 47))
        remainder = count - ni - nn
        sizes = (nn, ni, (remainder + 1) // 2, remainder // 2)
        offset = 0
        for worker, size in zip((ne, 'in0', *tx), sizes):
            workers[worker]['jobs'].extend(candidates[offset:offset + size])
            offset += size
    assigned = [key(j) for w in workers.values() for j in w['jobs']]
    if len(assigned) != len(set(assigned)) or set(assigned) != {key(j) for j in pristine}:
        raise ValueError('Assignment overlap or incomplete partition')
    return workers


def validate_plan(plan):
    if plan['source_sha256'] != SOURCE or plan['freeze_sha256'] != FREEZES or set(plan['workers']) != set(WORKERS):
        raise ValueError('Changed plan/source/worker registry')
    done = [key({k: j[k] for k in ('task', 'arm', 'rank')}) for j in plan['completed']]
    allocated = []
    for name, row in plan['workers'].items():
        if (row['instance'], row['gpu']) != WORKERS[name][:2]:
            raise ValueError('Worker ownership changed')
        lifecycle.normalize_nvml_uuid(row['gpu_uuid'])
        allocated.extend(key(j) for j in row['jobs'])
    full = done + allocated
    if len(full) != 128 or len(set(full)) != 128 or set(full) != {key(j) for j in jobs()}:
        raise ValueError('Full 128-stream partition is not exact')


def verify_droid_completion(panel, gpu, binding):
    """Whole predecessor queue plus all 18 raw-trace-bound shards, never GPU idleness alone."""
    panel = Path(panel)
    queue = panel / f'queue-gpu{gpu}'
    done, launch = read(queue / 'DONE.json'), read(queue / 'LAUNCH.json')
    ranks = [gpu, gpu + 4]
    if (done['completed_paired_streams'] != ranks or done['arms'] != ['native', *ARMS] or done['episodes'] != 144 or
            launch['pid'] != binding['pid'] or launch['instance'] != 50259194 or
            launch['logical_ranks'] != ranks or launch['device_uuid'] != binding['gpu_uuid'] or
            launch['source_sha256'] != DROID_SOURCE or launch['episodes_per_arm_global'] != 64 or
            launch['episodes_per_arm_on_device'] != 16 or digest(panel / 'freeze/protocol.json') != DROID_FREEZE):
        raise ValueError('Entire DROID queue/identity must match')
    return ranks


def require_authority(operation, ready_hash=None, plan_hash=None):
    value = read(CONTROL / (operation.upper() + '_AUTHORIZATION.json'))
    if value.get('owner') != 'rep_geometry_transcoder/root' or value.get('operation') != operation:
        raise ValueError('Explicit root activation review required')
    if ready_hash is not None and value.get('ready_sha256') != ready_hash:
        raise ValueError('Authorization binds another preparation')
    if plan_hash is not None and value.get('plan_sha256') != plan_hash:
        raise ValueError('Authorization binds another partition')
    return value
