"""CPU receiving preparation and a fail-closed GPU0 DROID→PointMaze handoff."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import zipfile

ROOT = Path('/workspace/jepa-runtime')
PROOF = ROOT / 'pointmaze-tx-resume-20260908-v1'
CODE = ROOT / 'pointmaze-history-code-20260908-v1'
HISTORY = ROOT / 'pointmaze-training-history-20260908-v1/seed-234'
CHECKPOINT = HISTORY / 'remaining-epochs/jepa-e2.pth.tar'
PANEL = ROOT / 'droid-coupling-behavior-20260908-v3'
PYTHON = '/workspace/jepa-planning-python/bin/python'
DRIVER = '/usr/lib/x86_64-linux-gnu/libcuda.so.1'
UUID = 'GPU-99bc03dd-d8ee-d7c3-fb46-5771e42b95a6'
DROID_RECORD_UUID = UUID.removeprefix('GPU-')
SOURCE_SHA = 'd4bd561e1d1074059dd3ead28fc5509ec7c3a5f6a854dea3f78856b762c0ccab'
CHECKPOINT_SHA = '35f1f4ba625d0dccf867d43303ee2c23e43daa5b72bb724e857c7e687d60c150'
DROID_FREEZE = '369d632792b6b482f5c63bbcd599dd7576cf949001ac005baa1cb5f600681862'
DRIVER_SHA = 'b7759e577409c86949f455f18fcc044d72b466be5e2982b6ba9e20348e86fa97'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    # These immutable lifecycle receipts never overwrite another attempt.
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def source_hash(code=CODE):
    h = hashlib.sha256()
    for path in sorted((code / 'src/offline_study').glob('*.py')):
        h.update(path.name.encode() + b'\0' + path.read_bytes())
    return h.hexdigest()


def verify_members(base, manifest):
    for name, wanted in manifest.items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe receiving member')
        path = base / path
        if 'symlink' in wanted:
            if not path.is_symlink() or os.readlink(path) != wanted['symlink']:
                raise ValueError('Receiving symlink changed: ' + name)
        elif path.is_symlink() or path.stat().st_size != wanted['bytes'] or digest(path) != wanted['sha256']:
            raise ValueError('Receiving member changed: ' + name)


def environment(cuda=''):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=cuda, JEPA_VERIFIED_LOCAL_DINO='1',
        PYTHONPATH=str(CODE / 'src') + ':/workspace/jepa-python/lib/python3.10/site-packages',
        LD_LIBRARY_PATH='/opt/conda/lib', LD_PRELOAD=DRIVER,
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')


def verify_state_metadata(data, original_binding):
    state = data['study_resume']
    if (data['epoch'] != 3 or state['epoch'] != 3 or state['binding'] != original_binding or
            not state['epoch_boundary_only'] or state['scheduler_step'] != 3417 or state['wd_step'] != 3417 or
            state['validation_events'] != 15 or len(state['cpu_rngs']) != 16 or len(state['cuda_rngs']) != 16 or
            [row['epoch'] for row in state['checkpoint_history']] != [1, 2]):
        raise ValueError('Epoch3 optimizer/seed/validation/RNG/source binding changed')


def prepare():
    out = PROOF / 'receiving'
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    try:
        staged = read(PROOF / 'STAGED.json')
        if staged['status'] != 'pointmaze_cpu_relocation_bytes_verified_not_gpu_clearance' or staged['target'] != 50259194:
            raise ValueError('Missing completed CPU staging')
        for name, key in [('HISTORY_FILES.json', 'history_manifest_sha256'), ('RUNTIME_FILES.json', 'runtime_manifest_sha256')]:
            if digest(PROOF / name) != staged[key]:
                raise ValueError('Receiving manifest changed')
            verify_members(Path('/'), read(PROOF / name))
        if source_hash() != SOURCE_SHA or digest(CHECKPOINT) != CHECKPOINT_SHA or digest(DRIVER) != DRIVER_SHA:
            raise ValueError('Receiving source/checkpoint/driver identity changed')
        if subprocess.check_output(['git', '-C', '/workspace/jepa_steering/vendor/jepa-wms', 'rev-parse', 'HEAD'], text=True).strip() != '13cf1d9c7e476f53c17714d2e0f1dc239a883ce0':
            raise ValueError('Pinned vendor changed')
        if subprocess.check_output(['git', '-C', '/workspace/jepa_steering/vendor/jepa-wms', 'status', '--porcelain'], text=True).strip():
            raise ValueError('Pinned vendor is dirty')
        receipt = ROOT / 'pointmaze-training-inputs-20260908-v1'
        report = read(receipt / 'report.json')
        if (digest(receipt / 'report.json') != read(receipt / 'DONE.json')['report_sha256'] or
                report['status'] != 'complete_pointmaze_inputs_match_pinned_archive' or
                report['files_sha256'] != digest(receipt / 'files.json') or report['protocol_sha256'] != digest(receipt / 'protocol.json')):
            raise ValueError('Wrong full source input receipt')
        files = read(receipt / 'files.json')
        if set(files) != {'states.pth', 'actions.pth', 'seq_lengths.pth'} | {f'obses/episode_{i:03d}.pth' for i in range(2000)}:
            raise ValueError('Require all2003 native dataset files')
        archive = ROOT / 'navigation-assets-20260907-v1/downloads/dataset/point_maze/point_maze.zip'
        if digest(archive) != '6c48ccf22c90b9af8dcf0e2cd70849aec8dd8e214ac5f1f09552bf8bc9494acc':
            raise ValueError('Official dataset archive changed')
        target = ROOT / 'navigation-assets-20260907-v1/extracted/pointmaze/point_maze'
        target.mkdir(parents=True, exist_ok=False)
        if shutil.disk_usage(target).free < sum(r['bytes'] for r in files.values()) + 24 * 1024**3:
            raise ValueError('Insufficient data/checkpoint reserve')
        with zipfile.ZipFile(archive) as zipped:
            names = zipped.namelist()
            for index, (name, wanted) in enumerate(sorted(files.items()), 1):
                member = 'point_maze/' + name
                if names.count(member) != 1 or zipped.getinfo(member).file_size != wanted['bytes']:
                    raise ValueError('Missing or duplicate archive member')
                path = target / name
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_name(path.name + '.receiving_tmp')
                with zipped.open(member) as src, temp.open('xb') as dst:
                    shutil.copyfileobj(src, dst, 8 << 20)
                if temp.stat().st_size != wanted['bytes'] or digest(temp) != wanted['sha256'] or path.exists():
                    raise ValueError('Extracted member differs; preserve partial')
                temp.rename(path)
                if index % 200 == 0:
                    print(json.dumps({'extracted_verified': index, 'required': 2003}), flush=True)
        with (out / 'cpu-tests.log').open('x') as log:
            subprocess.run([PYTHON, '-m', 'unittest', 'discover', '-s', str(CODE / 'tests'), '-q'],
                env=environment(), stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
        # Inspect metadata on CPU; no model/validation execution and no CUDA allocation.
        import torch
        sys.path.insert(0, str(CODE / 'src'))
        from offline_study.pointmaze_training_history import source_bindings
        from offline_study.model_loader import verified_local_dino_cache
        vendor = Path('/workspace/jepa_steering/vendor/jepa-wms')
        original_binding = {'task': 'pointmaze', 'seed': 234, 'input_report_sha256': digest(receipt / 'report.json'),
            'native_config_sha256': 'c666b4251f72f56c08c69ab223cfe645625f39705f1c84285320edc98c205a66',
            'source_sha256': source_bindings(vendor),
            'pilot_report_sha256': digest(ROOT / 'pointmaze-training-accumulation-pilot-20260908-v1/report.json'),
            'input_check_report_sha256': digest(ROOT / 'navigation-input-check-20260907-v1/report.json')}
        data = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
        verify_state_metadata(data, original_binding)
        for row in data['study_resume']['checkpoint_history']:
            if digest(row['path']) != row['sha256']:
                raise ValueError('Inherited epoch1/2 checkpoint changed')
        with verified_local_dino_cache() as dino:
            encoder = dict(dino)
        write(out / 'READY.json', {'status': 'cpu_preparation_complete_gpu_numerical_gate_pending',
            'source_sha256': SOURCE_SHA, 'checkpoint_sha256': CHECKPOINT_SHA, 'epoch': 3, 'seed': 234,
            'original_binding': original_binding, 'dataset_files': len(files), 'dataset_bytes': sum(r['bytes'] for r in files.values()),
            'data_files_sha256': digest(receipt / 'files.json'), 'encoder': encoder, 'gpu_calls': 0,
            'seconds': time.monotonic() - started, 'cpu_tests_sha256': digest(out / 'cpu-tests.log')})
        print(json.dumps({'status': 'pointmaze_cpu_preparation_complete', 'gpu_calls': 0}), flush=True)
    except Exception as exc:
        write(out / 'FAILED.json', {'error': str(exc), 'gpu_calls': 0, 'partial_files_preserved': True})
        raise


def verify_queue_done(done, launch):
    arms = ['native', 'visual_only', 'action_condition_only', 'joint', 'joint_equal_standardized_energy',
            'permuted_visual', 'permuted_joint', 'matched_random', 'matched_random_equal_standardized_energy']
    if (done.get('completed_paired_streams') != [0, 4] or done.get('arms') != arms or done.get('episodes') != 144 or
            launch.get('pid') != 1296 or launch.get('instance') != 50259194 or launch.get('device_uuid') != UUID or
            launch.get('logical_ranks') != [0, 4] or launch.get('episodes_per_arm_global') != 64 or
            launch.get('episodes_per_arm_on_device') != 16 or
            launch.get('source_sha256') != '249a8cdb9326cd79f3b9180a830e409cdd0ffa4f2f3be2ac393c32e30ab81a54'):
        raise ValueError('Entire assigned DROID panel did not complete with original identity')


def occupied():
    return subprocess.check_output(['nvidia-smi', '-i', '0', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).split()


def predecessor_pending(previous, command):
    """A DONE publication immediately before process exit is a valid handoff."""
    if (previous / 'DONE.json').exists():
        return False
    if (previous / 'FAILED.json').exists():
        raise ValueError('DROID predecessor failed; do not bypass')
    try:
        fields = command.read_bytes().split(b'\0')
    except FileNotFoundError:
        if (previous / 'DONE.json').exists():
            return False
        raise ValueError('Original predecessor disappeared without completion')
    # Recheck after /proc access: the queue can publish DONE and exit during it.
    if (previous / 'DONE.json').exists():
        return False
    expected = b'/workspace/jepa-runtime/droid-coupling-code-20260908-v3/run_droid_parallel_queue.py'
    if expected not in fields or fields[-3:] != [b'--gpu', b'0', b'']:
        raise ValueError('Original DROID predecessor no longer live without completion')
    return True


def wait_and_resume():
    queue = PROOF / 'queue'
    queue.mkdir(exist_ok=False)
    child = None
    def interrupted(signum, frame):
        raise RuntimeError('Owned waiter interrupted; preserve all outputs')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        ready = read(PROOF / 'receiving/READY.json')
        if ready['status'] != 'cpu_preparation_complete_gpu_numerical_gate_pending' or ready['source_sha256'] != SOURCE_SHA:
            raise ValueError('CPU receiving gate incomplete')
        if source_hash() != SOURCE_SHA or digest(CHECKPOINT) != CHECKPOINT_SHA or digest(DRIVER) != DRIVER_SHA:
            raise ValueError('Bound source/checkpoint/driver changed')
        uuid = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
        if uuid != UUID:
            raise ValueError('Reserved receiving GPU changed')
        write(queue / 'LAUNCH.json', {'pid': os.getpid(), 'instance': 50259194, 'gpu': 0, 'device_uuid': uuid,
            'predecessor_pid': 1296, 'predecessor_output': str(PANEL / 'queue-gpu0'), 'seed': 234,
            'resume_checkpoint_sha256': CHECKPOINT_SHA, 'source_sha256': SOURCE_SHA,
            'receiving_ready_sha256': digest(PROOF / 'receiving/READY.json'), 'outcome_based_selection': False})
        deadline = time.monotonic() + 36 * 3600
        previous = PANEL / 'queue-gpu0'
        print(json.dumps({'status': 'waiting_without_gpu_for_complete_droid_gpu0', 'predecessor_pid': 1296}), flush=True)
        while predecessor_pending(previous, Path('/proc/1296/cmdline')):
            if time.monotonic() > deadline:
                raise ValueError('DROID predecessor failed or did not finish; do not bypass')
            time.sleep(30)
        verify_queue_done(read(previous / 'DONE.json'), read(previous / 'LAUNCH.json'))
        if digest(PANEL / 'freeze/protocol.json') != DROID_FREEZE:
            raise ValueError('Original DROID freeze changed')
        # The preserved DROID verifier is CPU-only and verifies all18 shard receipts/records.
        droid_code = ROOT / 'droid-coupling-code-20260908-v3'
        validation = '''import json,pathlib,sys
from offline_study.droid_coupling_behavior import scientific_shard
from offline_study.author_fit import source_hash
p=pathlib.Path('/workspace/jepa-runtime/droid-coupling-behavior-20260908-v3'); protocol=json.loads((p/'freeze/protocol.json').read_text())
assert source_hash()==protocol['source_sha256']
receipts={}
for rank in (0,4):
 for arm in protocol['arms']:
  records,report,h=scientific_shard(p/'conditions'/arm/f'shard-{rank}',protocol,'369d632792b6b482f5c63bbcd599dd7576cf949001ac005baa1cb5f600681862',rank,arm,sys.argv[1])
  receipts[arm+':'+str(rank)]=h
print(json.dumps({'verified_shards':len(receipts),'verified_endpoints':144,'receipts':receipts}))
'''
        droid_env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(droid_code / 'src'),
            LD_LIBRARY_PATH='/opt/conda/lib', LD_PRELOAD=DRIVER, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        result = subprocess.check_output(['/workspace/jepa-droid-python/bin/python', '-c', validation, DROID_RECORD_UUID], env=droid_env, text=True, timeout=300)
        write(queue / 'PREDECESSOR_VERIFIED.json', json.loads(result))
        for _ in range(24):
            if not occupied():
                break
            time.sleep(5)
        else:
            raise ValueError('DROID completed but device not released; do not stop or multiplex')
        def execute(module, arguments, label):
            nonlocal child
            if occupied():
                raise ValueError('Reserved GPU occupied before ' + label)
            with (queue / (label + '.log')).open('x') as log:
                child = subprocess.Popen([PYTHON, '-u', '-m', module] + arguments,
                    env=environment('0'), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                if child.wait(timeout=72 * 3600 if label == 'resume' else 1800):
                    raise ValueError('Required ' + label + ' failed; preserve outputs, no automatic retry')
            child = None
        vendor = ['--vendor', '/workspace/jepa_steering/vendor/jepa-wms']
        receiving_pilot = PROOF / 'receiving-gpu-pilot'
        execute('offline_study.training_pilot', vendor + ['--assets', str(ROOT / 'navigation-assets-20260907-v1'),
            '--input-check', str(ROOT / 'navigation-input-check-20260907-v1'), '--task', 'pointmaze',
            '--output', str(receiving_pilot)], 'receiving-pilot')
        pilot = read(receiving_pilot / 'report.json')
        if (digest(receiving_pilot / 'report.json') != read(receiving_pilot / 'DONE.json')['report_sha256'] or
                pilot['status'] != 'native_training_accumulation_pilot_passed' or pilot['task'] != 'pointmaze' or
                pilot['updates_complete'] != 5 or pilot['updates_per_epoch'] != 1139 or
                pilot['parity_sha256'] != digest(receiving_pilot / 'PARITY.json')):
            raise ValueError('Receiving numerical parity did not pass')
        # Keep ORIGINAL pilot binding in the original checkpoint. The TX pilot is an extra gate.
        original_pilot = ROOT / 'pointmaze-training-accumulation-pilot-20260908-v1'
        if digest(original_pilot / 'report.json') != ready['original_binding']['pilot_report_sha256']:
            raise ValueError('Original checkpoint pilot binding changed')
        write(queue / 'RECEIVING_GPU_VERIFIED.json', {'receiving_pilot_report_sha256': digest(receiving_pilot / 'report.json'),
            'original_checkpoint_pilot_report_sha256': digest(original_pilot / 'report.json'),
            'original_binding_unchanged': True, 'cross_hardware_bitwise_training_history_claimed': False})
        execute('offline_study.pointmaze_training_history', vendor + [
            '--data-root', str(ROOT / 'navigation-assets-20260907-v1/extracted/pointmaze/point_maze'),
            '--input-check', str(ROOT / 'navigation-input-check-20260907-v1'),
            '--input-receipt', str(ROOT / 'pointmaze-training-inputs-20260908-v1'), '--pilot', str(original_pilot),
            '--seed', '234', '--resume-from', str(CHECKPOINT), '--engineering-proof', str(HISTORY / 'epoch-one-engineering'),
            '--output', str(HISTORY / 'resumed-after-droid-tx-v1')], 'resume')
        result_root = HISTORY / 'resumed-after-droid-tx-v1'
        result = read(result_root / 'report.json')
        if (digest(result_root / 'report.json') != read(result_root / 'DONE.json')['report_sha256'] or
                result['status'] != 'all50_pointmaze_training_epochs_complete' or result['checkpoints'] != 50 or
                result['seed'] != 234 or result['validation_events'] != 250):
            raise ValueError('Full original PointMaze234 history not complete')
        write(queue / 'DONE.json', {'pointmaze_seed234_report_sha256': digest(result_root / 'report.json'),
            'full_study_complete': False, 'planning_history_evaluations_complete': False})
    except Exception as exc:
        write(queue / 'FAILED.json', {'error': str(exc), 'partial_files_preserved': True, 'full_study_complete': False})
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'wait', 'prepare-and-wait'))
    args = parser.parse_args()
    if args.mode in ('prepare', 'prepare-and-wait'):
        prepare()
    if args.mode in ('wait', 'prepare-and-wait'):
        wait_and_resume()
