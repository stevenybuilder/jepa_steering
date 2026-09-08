"""Local-only source packaging and final navigation→fresh-history plan binding.

prepare creates a non-runnable proposal from commit caa41aa and existing verified
input/runtime receipts. finalize requires the actual navigation PLAN, CUTOVER and
tx0/tx1/tx2 LAUNCH files; it never invents pending process identities. Neither mode
uses SSH, provider APIs, GPUs, or modifies an existing output directory.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

import corrected_pointmaze_queue as q
from navigation_redistribution_common import validate_plan as validate_navigation_plan


COMMIT = 'caa41aabb307ca57aa9956a4793a54af0bf23460'
CONTROL = Path('/workspace/jepa-runtime/pointmaze-corrected-queue-20260908-v1')
NAVIGATION = Path('/workspace/jepa-runtime/navigation-redistribution-20260908-v1')
REVIEWED_NAVIGATION_ROOTS = (NAVIGATION, NAVIGATION.with_name('navigation-redistribution-20260908-v2'))
CODE = CONTROL / 'code'
VENDOR = Path('/workspace/jepa_steering/vendor/jepa-wms')
RUNTIME = Path('/workspace/jepa-runtime')
NAV_SOURCE = 'fc7578672eabd8e08469afcc1f7f4d9b1c2ed18c447f4f003539b61da7950f58'
FREEZES = {'wall': 'd41381c52313e536e53ca80ccbf8324756bf665d248f436282622f9085fc7af3',
    'pointmaze': '2d94a21b4554670dfb289baf7018291be7764b01a00be093a98ac19a3afe1fbf'}
GPU_UUIDS = ['GPU-99bc03dd-d8ee-d7c3-fb46-5771e42b95a6',
    'GPU-732bb4c8-5057-668e-1126-c3cd87e6cbf9', 'GPU-544a0c82-405a-1e1d-d6af-492d8016e3e2']
SHARED_READY_SHA = 'a0f41d927fe2f129698644222fb127c55c6832d06afc0e4877ed7231a846ccc7'
RUNTIME_MANIFEST_SHA = '8d7e7a8e741dbbbde47df2d6b0f25e21a2e52b1e678462ace886c1ec66ce227e'
HISTORY_MANIFEST_SHA = 'e7112c30a4e3b2f4e9d5d9819d90a8c7ec25a7b8bcaf90698a438a1abd8b71a9'
TESTS = ['test_native_training_sampler.py', 'test_pointmaze_training.py']


def bytes_hash(value):
    return hashlib.sha256(value).hexdigest()


def frozen_source(repo):
    paths = subprocess.check_output(['git', '-C', str(repo), 'ls-tree', '-r', '--name-only', COMMIT,
        'src/offline_study'], text=True).splitlines()
    paths = [p for p in paths if p.endswith('.py') and p.count('/') == 2]
    if not paths:
        raise ValueError('Pinned scientific commit is unavailable')
    selected = paths + ['tests/' + name for name in TESTS]
    members = {name: subprocess.check_output(['git', '-C', str(repo), 'show', COMMIT + ':' + name]) for name in selected}
    value = hashlib.sha256()
    for name in sorted(paths):
        value.update(Path(name).name.encode() + b'\0' + members[name])
    return members, value.hexdigest()


def pinned_input_files(shared):
    checks = [('receiving/READY.json', SHARED_READY_SHA),
        ('RUNTIME_FILES.json', RUNTIME_MANIFEST_SHA), ('HISTORY_FILES.json', HISTORY_MANIFEST_SHA)]
    for name, expected in checks:
        if q.digest(shared / name) != expected:
            raise ValueError('Prior complete runtime/input verification changed: ' + name)
    ready = q.read(shared / 'receiving/READY.json')
    if (ready['dataset_files'], ready['dataset_bytes']) != (2003, 30117624491):
        raise ValueError('Incomplete PointMaze population')
    history = q.read(shared / 'HISTORY_FILES.json')
    runtime = q.read(shared / 'RUNTIME_FILES.json')
    names = ['workspace/jepa-runtime/navigation-assets-20260907-v1/' + n for n in ('DONE.json', 'protocol.json', 'report.json')]
    names += ['workspace/jepa-runtime/navigation-input-check-20260907-v1/' + n for n in ('DONE.json', 'protocol.json', 'report.json', 'pointmaze.json', 'wall.json')]
    names += ['workspace/jepa-runtime/pointmaze-training-inputs-20260908-v1/' + n for n in ('DONE.json', 'protocol.json', 'report.json', 'files.json')]
    files = {'/' + name: history[name]['sha256'] for name in names}
    # Reuse the previously complete 46,520-member runtime audit, additionally pin
    # the actual interpreter/torch entrypoints/weights; native loader rechecks the
    # complete DINO source/weights, and training rehashes all2003 raw input files.
    keys = ['root/.local/share/uv/python/cpython-3.10.18-linux-x86_64-gnu/bin/python3.10',
        'workspace/jepa-python/lib/python3.10/site-packages/torch/__init__.py',
        'workspace/jepa-python/lib/python3.10/site-packages/torch/_C.cpython-310-x86_64-linux-gnu.so',
        'root/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth']
    files.update({'/' + name: runtime[name]['sha256'] for name in keys})
    files[str(RUNTIME / 'pointmaze-tx-resume-20260908-v1/receiving/READY.json')] = SHARED_READY_SHA
    files[str(RUNTIME / 'pointmaze-tx-resume-20260908-v1/RUNTIME_FILES.json')] = RUNTIME_MANIFEST_SHA
    if any('pilot' in path or 'training-history' in path or 'history-code' in path for path in files):
        raise ValueError('Historical pilots/checkpoints/source cannot become new training inputs')
    return files


def prepare(repo, shared, output):
    if output.exists():
        raise FileExistsError('Preserve existing proposal')
    members, source = frozen_source(repo)
    pinned = pinned_input_files(shared)
    script = Path(__file__).with_name('corrected_pointmaze_queue.py')
    members['scripts/vast/corrected_pointmaze_queue.py'] = script.read_bytes()
    members['tests/test_corrected_pointmaze_queue.py'] = (repo / 'tests/test_corrected_pointmaze_queue.py').read_bytes()
    output.mkdir(parents=True)
    manifest = {str(Path('code') / name): {'bytes': len(value), 'sha256': bytes_hash(value)} for name, value in members.items()}
    manifest['code/vendor/jepa-wms'] = {'symlink': str(VENDOR)}
    with tarfile.open(output / 'source.tar.gz', 'x:gz') as archive:
        for name, value in members.items():
            info = tarfile.TarInfo(str(Path('code') / name))
            info.size, info.mode = len(value), 0o644
            archive.addfile(info, io.BytesIO(value))
        info = tarfile.TarInfo('code/vendor/jepa-wms')
        info.type, info.linkname = tarfile.SYMTYPE, str(VENDOR)
        archive.addfile(info)
    q.write(output / 'SOURCE_FILES.json', manifest)
    approval = {'status': q.APPROVAL_STATUS, 'owner': 'rep_geometry_transcoder/root',
        'source_commit': COMMIT, 'source_sha256': source, 'training_seeds': q.SEEDS,
        'sampler_policy': q.POLICY, 'execution': 'native_accumulation_uncached',
        'cache_pilot_required': False, 'scientific_method_changed_by_operations': False,
        'approval_scope': 'reviewed local queue preparation; remote activation still requires resource-board reservation'}
    q.write(output / 'EXECUTION_APPROVAL.json', approval)
    pinned.update({str(CODE / name): bytes_hash(value) for name, value in members.items() if name.startswith('tests/')})
    proposal = {'status': 'local_preparation_not_runnable_navigation_launch_bindings_pending',
        'schema': 1, 'source_root': str(CODE), 'source_commit': COMMIT, 'source_sha256': source,
        'vendor': str(VENDOR), 'python': '/workspace/jepa-planning-python/bin/python',
        'overlay': '/workspace/jepa-python/lib/python3.10/site-packages',
        'driver': '/usr/lib/x86_64-linux-gnu/libcuda.so.1',
        'driver_sha256': 'b7759e577409c86949f455f18fcc044d72b466be5e2982b6ba9e20348e86fa97',
        'output_root': str(CONTROL / 'corrected-histories'), 'assets': str(RUNTIME / 'navigation-assets-20260907-v1'),
        'input_check': str(RUNTIME / 'navigation-input-check-20260907-v1'),
        'input_receipt': str(RUNTIME / 'pointmaze-training-inputs-20260908-v1'),
        'data_root': str(RUNTIME / 'navigation-assets-20260907-v1/extracted/pointmaze/point_maze'),
        'fixed_files': pinned, 'execution_approval': {'path': str(CONTROL / 'EXECUTION_APPROVAL.json'),
            'sha256': q.digest(output / 'EXECUTION_APPROVAL.json')},
        'required_assignment': [{'seed': seed, 'navigation_worker': 'tx' + str(gpu),
            'instance': 50259194, 'gpu': gpu, 'gpu_uuid': GPU_UUIDS[gpu]} for gpu, seed in enumerate(q.SEEDS)],
        'resume_historical_checkpoints': False, 'gpu_calls': 0}
    q.write(output / 'PROPOSAL.json', proposal)
    q.write(output / 'PREPARED.json', {'status': 'corrected_pointmaze_local_source_prepared_not_activated',
        'source_commit': COMMIT, 'source_sha256': source, 'files': len(manifest),
        'source_archive_sha256': q.digest(output / 'source.tar.gz'),
        'source_files_sha256': q.digest(output / 'SOURCE_FILES.json'),
        'proposal_sha256': q.digest(output / 'PROPOSAL.json'),
        'execution_approval_sha256': q.digest(output / 'EXECUTION_APPROVAL.json'), 'gpu_calls': 0})
    return proposal


def navigation_root_from_launch(launch):
    command = list(launch['identity']['command'])
    if len(command) > 1 and command[1] == '-u':
        command.pop(1)
    if (len(command) != 5 or command[0] != '/usr/bin/python3' or command[2:4] != ['run', '--worker'] or
            command[4] != launch['worker'] or Path(command[1]).name != 'navigation_redistribution_control.py'):
        raise ValueError('Unrecognized actual navigation supervisor command')
    root = Path(command[1]).parent
    if root not in REVIEWED_NAVIGATION_ROOTS:
        raise ValueError('Navigation control root has not been reviewed')
    return root


def assignment_binding(name, worker, launch, launch_sha, plan_sha, verifier_sha, navigation_root=NAVIGATION):
    gpu = int(name[-1])
    if name not in ('tx0', 'tx1', 'tx2') or (worker['instance'], worker['gpu'], worker['gpu_uuid']) != (50259194, gpu, GPU_UUIDS[gpu]):
        raise ValueError('Only the three reviewed Texas devices may receive histories')
    if (launch['worker'] != name or launch['plan_sha256'] != plan_sha or launch['jobs'] != worker['jobs'] or
            any(launch[k] != worker[k] for k in ('instance', 'gpu', 'gpu_uuid')) or not worker['jobs']):
        raise ValueError('Actual navigation launch differs from frozen assignment')
    identity = launch['identity']
    if navigation_root_from_launch(launch) != navigation_root:
        raise ValueError('Mixed navigation control versions cannot share a handoff')
    expected_command = ['/usr/bin/python3', str(navigation_root / 'navigation_redistribution_control.py'), 'run', '--worker', name]
    if identity['pid'] <= 0 or identity['starttime'] <= 0 or identity['command'] not in (
            expected_command, expected_command[:1] + ['-u'] + expected_command[1:]):
        raise ValueError('Actual navigation process identity/argv is not bound')
    status = navigation_root / 'workers' / name
    common = {'worker': name, 'plan_sha256': plan_sha, 'identity': identity,
        'instance': worker['instance'], 'gpu': gpu, 'gpu_uuid': worker['gpu_uuid']}
    return {'seed': q.SEEDS[gpu], 'instance': 50259194, 'gpu': gpu, 'gpu_uuid': GPU_UUIDS[gpu],
        'predecessor': {'pid': identity['pid'], 'start_ticks': identity['starttime'], 'argv': identity['command'],
            'done': str(status / 'DONE.json'), 'failed': str(status / 'FAILED.json'),
            'launch': str(status / 'LAUNCH.json'), 'launch_sha256': launch_sha,
            'expected_done': {**common, 'status': 'navigation_redistribution_assignment_complete',
                'source_sha256': NAV_SOURCE, 'freeze_sha256': FREEZES, 'launch_sha256': launch_sha,
                'endpoints': 12 * len(worker['jobs']), 'gpu_available_only_after_process_exit': True},
            'verifier': {'python': '/usr/bin/python3', 'script': str(navigation_root / 'navigation_redistribution_control.py'),
                'script_sha256': verifier_sha, 'args': ['verify-completed', '--worker', name],
                'expected_result': {**common, 'verified_endpoints': 12 * len(worker['jobs']),
                    'scientific_validators_unchanged': True}}}}


def finalize(prepared, navigation, output):
    if output.exists():
        raise FileExistsError('Never overwrite an already frozen plan')
    proof = q.read(prepared / 'PREPARED.json')
    for name, key in [('PROPOSAL.json', 'proposal_sha256'), ('SOURCE_FILES.json', 'source_files_sha256'),
                      ('EXECUTION_APPROVAL.json', 'execution_approval_sha256'), ('source.tar.gz', 'source_archive_sha256')]:
        if q.digest(prepared / name) != proof[key]:
            raise ValueError('Local prepared source/decision changed')
    plan = q.read(prepared / 'PROPOSAL.json')
    nav = q.read(navigation / 'PLAN.json')  # Missing real receipts fail here, before writes.
    validate_navigation_plan(nav)
    nav_sha = q.digest(navigation / 'PLAN.json')
    cutover = q.read(navigation / 'CUTOVER.json')
    q.require_fields(nav, {'status': 'intact_navigation_redistribution_frozen', 'source_sha256': NAV_SOURCE,
        'freeze_sha256': FREEZES, 'total_candidate_streams': 128, 'episodes_per_task_condition': 96,
        'outcome_based_selection': False, 'old_children_completed_untouched': True}, 'Frozen navigation plan')
    q.require_fields(cutover, {'plan_sha256': nav_sha, 'old_parents_terminal': True,
        'old_watcher_terminal': True, 'old_children_not_interrupted': True}, 'Completed ownership cutover')
    ready = q.read(navigation / 'READY-50259194.json')
    if q.digest(navigation / 'FILES.json') != ready['files_sha256'] or q.digest(navigation / 'READY-50259194.json') != nav['ready_sha256']['50259194']:
        raise ValueError('Actual Texas preparation is not the navigation plan binding')
    files = q.read(navigation / 'FILES.json')
    launches = {name: q.read(navigation / 'workers' / name / 'LAUNCH.json') for name in ('tx0', 'tx1', 'tx2')}
    navigation_root = navigation_root_from_launch(launches['tx0'])
    plan['fixed_files'].update({str(navigation_root / name): item['sha256'] for name, item in files.items()})
    plan['fixed_files'][str(navigation_root / 'FILES.json')] = ready['files_sha256']
    plan['fixed_files'][str(navigation_root / 'PLAN.json')] = nav_sha
    plan['fixed_files'][str(navigation_root / 'CUTOVER.json')] = q.digest(navigation / 'CUTOVER.json')
    plan['assignments'] = []
    for gpu in range(3):
        name = 'tx' + str(gpu)
        path = navigation / 'workers' / name / 'LAUNCH.json'
        assignment = assignment_binding(name, nav['workers'][name], launches[name], q.digest(path), nav_sha,
            files['navigation_redistribution_control.py']['sha256'], navigation_root)
        plan['assignments'].append(assignment)
        plan['fixed_files'][assignment['predecessor']['launch']] = q.digest(path)
    plan['status'] = 'corrected_pointmaze_three_seed_queue_frozen_after_real_navigation_launches'
    for seed in q.SEEDS:
        q.validate_plan(plan, seed)
    output.mkdir(parents=True)
    q.write(output / 'PLAN.json', plan)
    q.write(output / 'FINALIZED.json', {'plan_sha256': q.digest(output / 'PLAN.json'),
        'navigation_plan_sha256': nav_sha, 'source_commit': COMMIT, 'source_sha256': plan['source_sha256'],
        'seed_assignment': {'234': 'tx0', '235': 'tx1', '236': 'tx2'}, 'gpu_calls': 0,
        'remote_staging_or_activation_performed': False})
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'finalize'))
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--navigation', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.repo, args.repo / 'artifacts/offline_study/pointmaze-tx-resume-20260908-v1', args.output)
    else:
        if args.prepared is None or args.navigation is None:
            parser.error('finalize requires --prepared and --navigation actual receipt directories')
        finalize(args.prepared, args.navigation, args.output)
    print(json.dumps({'mode': args.mode, 'output': str(args.output), 'gpu_calls': 0}))


if __name__ == '__main__':
    main()
