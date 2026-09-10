"""Required third training seed, after the entire owned DROID GPU1 panel."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

ROOT = Path('/workspace/jepa-runtime')
PROOF = ROOT / 'pointmaze-seed236-tx-20260908-v1'
HISTORY = ROOT / 'pointmaze-training-history-20260908-v1/seed-236'
CODE = ROOT / 'pointmaze-history-code-20260908-v1'
SHARED = ROOT / 'pointmaze-tx-resume-20260908-v1'
PANEL = ROOT / 'droid-coupling-behavior-20260908-v3'
SOURCE_SHA = 'd4bd561e1d1074059dd3ead28fc5509ec7c3a5f6a854dea3f78856b762c0ccab'
SHARED_WORKER_SHA = '7f3075639603f5c4338b1a6e4c4284947381276c3b557a612b866e50e2d00ab4'
SHARED_READY_SHA = 'a0f41d927fe2f129698644222fb127c55c6832d06afc0e4877ed7231a846ccc7'
DROID_SOURCE_SHA = '249a8cdb9326cd79f3b9180a830e409cdd0ffa4f2f3be2ac393c32e30ab81a54'
DROID_FREEZE_SHA = '369d632792b6b482f5c63bbcd599dd7576cf949001ac005baa1cb5f600681862'
UUID = 'GPU-732bb4c8-5057-668e-1126-c3cd87e6cbf9'
RECORD_UUID = UUID.removeprefix('GPU-')
PID = 1297
RANKS = [1, 5]
ARMS = ['native', 'visual_only', 'action_condition_only', 'joint', 'joint_equal_standardized_energy',
        'permuted_visual', 'permuted_joint', 'matched_random', 'matched_random_equal_standardized_energy']


def load_helpers():
    path = SHARED / 'pointmaze_tx_resume_worker.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHARED_WORKER_SHA:
        raise ValueError('Immutable CPU helper source changed')
    spec = importlib.util.spec_from_file_location('pointmaze_tx_immutable_helpers', path)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    return helpers


def predecessor_pending(previous, command):
    if (previous / 'FAILED.json').exists():
        raise ValueError('DROID predecessor failed; do not bypass')
    if (previous / 'DONE.json').exists():
        return False
    try:
        fields = command.read_bytes().split(b'\0')
    except FileNotFoundError:
        if (previous / 'DONE.json').exists():
            return False
        raise ValueError('Original DROID predecessor exited without completion')
    if (previous / 'DONE.json').exists():
        return False
    expected = b'/workspace/jepa-runtime/droid-coupling-code-20260908-v3/run_droid_parallel_queue.py'
    if expected not in fields or fields[-3:] != [b'--gpu', b'1', b'']:
        raise ValueError('Original GPU1 queue identity changed or became terminal')
    return True


def verify_predecessor(done, launch):
    if (done.get('completed_paired_streams') != RANKS or done.get('arms') != ARMS or done.get('episodes') != 144 or
            launch.get('pid') != PID or launch.get('instance') != 50259194 or launch.get('device_uuid') != UUID or
            launch.get('logical_ranks') != RANKS or launch.get('episodes_per_arm_global') != 64 or
            launch.get('episodes_per_arm_on_device') != 16 or launch.get('source_sha256') != DROID_SOURCE_SHA):
        raise ValueError('Incomplete or changed entire assigned GPU1 DROID panel')


def training_arguments(vendor, original_pilot, history=HISTORY):
    """Seed236 initializes its own engineering run, never from another seed."""
    common = ['--vendor', str(vendor), '--data-root', str(ROOT / 'navigation-assets-20260907-v1/extracted/pointmaze/point_maze'),
        '--input-check', str(ROOT / 'navigation-input-check-20260907-v1'),
        '--input-receipt', str(ROOT / 'pointmaze-training-inputs-20260908-v1'),
        '--pilot', str(original_pilot), '--seed', '236']
    engineering = history / 'epoch-one-engineering'
    return (common + ['--engineering-only', '--output', str(engineering)],
        common + ['--resume-from', str(engineering / 'jepa-e0.pth.tar'), '--engineering-proof', str(engineering),
                  '--output', str(history / 'remaining-epochs')])


def prepare(h):
    output = PROOF / 'receiving'
    output.mkdir(exist_ok=False)
    try:
        if h.digest(SHARED / 'receiving/READY.json') != SHARED_READY_SHA or h.source_hash() != SOURCE_SHA:
            raise ValueError('Previously verified complete inputs or frozen source changed')
        shared = h.read(SHARED / 'receiving/READY.json')
        if (shared['dataset_files'] != 2003 or shared['dataset_bytes'] != 30117624491 or
                shared['status'] != 'cpu_preparation_complete_gpu_numerical_gate_pending' or
                shared['original_binding']['seed'] != 234 or h.digest(h.DRIVER) != h.DRIVER_SHA):
            raise ValueError('Original shared source/input/runtime proof changed')
        if HISTORY.exists() or shutil.disk_usage(ROOT).free < 32 * 1024**3:
            raise ValueError('Seed236 already exists or checkpoint reserve insufficient')
        vendor = Path('/workspace/jepa_steering/vendor/jepa-wms')
        if (subprocess.check_output(['git', '-C', str(vendor), 'rev-parse', 'HEAD'], text=True).strip() != '13cf1d9c7e476f53c17714d2e0f1dc239a883ce0' or
                subprocess.check_output(['git', '-C', str(vendor), 'status', '--porcelain'], text=True).strip()):
            raise ValueError('Pinned vendor changed')
        original_pilot = ROOT / 'pointmaze-training-accumulation-pilot-20260908-v1'
        if h.digest(original_pilot / 'report.json') != shared['original_binding']['pilot_report_sha256']:
            raise ValueError('Original pilot binding changed')
        for name, expected in shared['original_binding']['source_sha256'].items():
            if name.startswith('native_'):
                continue  # Native files are additionally checked by the fixed runner/vendor.
            if h.digest(CODE / 'src/offline_study' / name) != expected:
                raise ValueError('Fixed training module changed')
        if h.digest(ROOT / 'pointmaze-training-inputs-20260908-v1/files.json') != shared['data_files_sha256']:
            raise ValueError('Full raw-file binding changed')
        uuid = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
        if uuid != UUID:
            raise ValueError('Owned receiving GPU1 changed')
        h.write(output / 'READY.json', {'status': 'required_pointmaze_seed236_cpu_queue_prepared', 'seed': 236,
            'source_sha256': SOURCE_SHA, 'shared_inputs_receiving_sha256': SHARED_READY_SHA,
            'expected_training_binding': {**shared['original_binding'], 'seed': 236},
            'gpu_uuid': uuid, 'gpu_calls': 0, 'inherits_other_seed_checkpoint': False,
            'epochs': 50, 'updates_per_epoch': 1139, 'validation_events_per_epoch': 5,
            'full_seed_triplet': [234, 235, 236], 'receiving_numerical_gate_complete': False})
    except Exception as exc:
        h.write(output / 'FAILED.json', {'error': str(exc), 'gpu_calls': 0})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'wait', 'prepare-and-wait'))
    args = parser.parse_args()
    h = load_helpers()
    if args.mode in ('prepare', 'prepare-and-wait'):
        prepare(h)
    if args.mode == 'prepare':
        return
    queue = PROOF / 'queue'
    queue.mkdir(exist_ok=False)
    child = None
    def interrupted(signum, frame):
        raise RuntimeError('Owned seed236 queue interrupted; preserve all outputs')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    def occupied():
        return subprocess.check_output(['nvidia-smi', '-i', '1', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).split()
    def execute(module, arguments, label, timeout):
        nonlocal child
        if occupied():
            raise ValueError('GPU1 occupied; do not multiplex before ' + label)
        with (queue / (label + '.log')).open('x') as log:
            child = subprocess.Popen([h.PYTHON, '-u', '-m', module] + arguments,
                env=h.environment('1'), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            if child.wait(timeout=timeout):
                raise ValueError('Required seed236 stage failed: ' + label)
        child = None
    try:
        ready = h.read(PROOF / 'receiving/READY.json')
        if ready['status'] != 'required_pointmaze_seed236_cpu_queue_prepared' or ready['seed'] != 236 or h.source_hash() != SOURCE_SHA:
            raise ValueError('Seed236 receiving preparation changed')
        h.write(queue / 'LAUNCH.json', {'pid': os.getpid(), 'instance': 50259194, 'gpu': 1, 'device_uuid': UUID,
            'predecessor_pid': PID, 'predecessor_logical_ranks': RANKS, 'source_sha256': SOURCE_SHA,
            'seed': 236, 'receiving_sha256': h.digest(PROOF / 'receiving/READY.json'),
            'worker_sha256': h.digest(Path(__file__)), 'initializes_from_seed236': True,
            'inherits_other_seed_checkpoint': False, 'gpu_job_started': False})
        previous = PANEL / 'queue-gpu1'
        deadline = time.monotonic() + 36 * 3600
        print(json.dumps({'status': 'waiting_without_gpu_for_entire_droid_gpu1_panel', 'predecessor_pid': PID}), flush=True)
        while predecessor_pending(previous, Path(f'/proc/{PID}/cmdline')):
            if time.monotonic() > deadline:
                raise ValueError('DROID queue did not finish; do not bypass')
            time.sleep(30)
        verify_predecessor(h.read(previous / 'DONE.json'), h.read(previous / 'LAUNCH.json'))
        if h.digest(PANEL / 'freeze/protocol.json') != DROID_FREEZE_SHA:
            raise ValueError('DROID freeze changed')
        validation = '''import json,pathlib,sys
from offline_study.droid_coupling_behavior import scientific_shard
from offline_study.author_fit import source_hash
p=pathlib.Path('/workspace/jepa-runtime/droid-coupling-behavior-20260908-v3'); protocol=json.loads((p/'freeze/protocol.json').read_text())
assert source_hash()==protocol['source_sha256']
receipts={}
for rank in (1,5):
 for arm in protocol['arms']:
  records,report,sha=scientific_shard(p/'conditions'/arm/f'shard-{rank}',protocol,'369d632792b6b482f5c63bbcd599dd7576cf949001ac005baa1cb5f600681862',rank,arm,sys.argv[1])
  receipts[arm+':'+str(rank)]=sha
print(json.dumps({'verified_shards':len(receipts),'verified_endpoints':144,'receipts':receipts}))
'''
        env = h.environment()
        env['PYTHONPATH'] = str(ROOT / 'droid-coupling-code-20260908-v3/src')
        result = subprocess.check_output(['/workspace/jepa-droid-python/bin/python', '-c', validation, RECORD_UUID],
            env=env, text=True, timeout=300)
        h.write(queue / 'PREDECESSOR_VERIFIED.json', json.loads(result))
        for _ in range(24):
            if not occupied():
                break
            time.sleep(5)
        else:
            raise ValueError('Complete predecessor did not release GPU1; never stop it')
        if subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip() != UUID:
            raise ValueError('GPU1 identity changed before numerical gate')
        if h.digest(h.DRIVER) != h.DRIVER_SHA or h.source_hash() != SOURCE_SHA:
            raise ValueError('Frozen driver/source changed before numerical gate')
        vendor = Path('/workspace/jepa_steering/vendor/jepa-wms')
        receiving_pilot = PROOF / 'receiving-gpu-pilot'
        execute('offline_study.training_pilot', ['--vendor', str(vendor), '--assets', str(ROOT / 'navigation-assets-20260907-v1'),
            '--input-check', str(ROOT / 'navigation-input-check-20260907-v1'), '--task', 'pointmaze',
            '--output', str(receiving_pilot)], 'receiving-pilot', 1800)
        pilot = h.read(receiving_pilot / 'report.json')
        if (h.digest(receiving_pilot / 'report.json') != h.read(receiving_pilot / 'DONE.json')['report_sha256'] or
                pilot['status'] != 'native_training_accumulation_pilot_passed' or pilot['task'] != 'pointmaze' or
                pilot['updates_complete'] != 5 or pilot['updates_per_epoch'] != 1139 or
                pilot['parity_sha256'] != h.digest(receiving_pilot / 'PARITY.json')):
            raise ValueError('Required receiving GPU numerical parity failed')
        original_pilot = ROOT / 'pointmaze-training-accumulation-pilot-20260908-v1'
        if h.digest(original_pilot / 'report.json') != ready['expected_training_binding']['pilot_report_sha256']:
            raise ValueError('Original pilot binding changed')
        h.write(queue / 'RECEIVING_GPU_VERIFIED.json', {'new_pilot_sha256': h.digest(receiving_pilot / 'report.json'),
            'original_pilot_binding_sha256': h.digest(original_pilot / 'report.json'),
            'seed236_engineering_still_required': True})
        HISTORY.mkdir(exist_ok=False)
        engineering_args, remaining_args = training_arguments(vendor, original_pilot)
        execute('offline_study.pointmaze_training_history', engineering_args, 'epoch-one-engineering', 7200)
        engineering = HISTORY / 'epoch-one-engineering'
        proof = h.read(engineering / 'report.json')
        protocol = h.read(engineering / 'protocol.json')
        if (h.digest(engineering / 'report.json') != h.read(engineering / 'DONE.json')['report_sha256'] or
                proof['status'] != 'one_epoch_pointmaze_training_engineering_complete' or proof['seed'] != 236 or
                proof['validation_events'] != 5 or proof['checkpoints'] != 1 or
                proof['protocol_sha256'] != h.digest(engineering / 'protocol.json') or
                proof['resume_parity_sha256'] != h.digest(engineering / 'RESUME_PARITY.json') or
                any(protocol[k] != v for k, v in ready['expected_training_binding'].items())):
            raise ValueError('Missing own complete seed236 first-epoch engineering')
        execute('offline_study.pointmaze_training_history', remaining_args, 'remaining-epochs', 72 * 3600)
        result_root = HISTORY / 'remaining-epochs'
        report = h.read(result_root / 'report.json')
        if (h.digest(result_root / 'report.json') != h.read(result_root / 'DONE.json')['report_sha256'] or
                report['status'] != 'all50_pointmaze_training_epochs_complete' or report['seed'] != 236 or
                report['checkpoints'] != 50 or report['validation_events'] != 250 or
                report['checkpoint_manifest_sha256'] != h.digest(result_root / 'CHECKPOINTS.json')):
            raise ValueError('Full seed236 training history incomplete')
        history = h.read(result_root / 'CHECKPOINTS.json')['history']
        if [row['epoch'] for row in history] != list(range(1, 51)):
            raise ValueError('Missing immutable seed236 history epoch')
        for row in history:
            if h.digest(row['path']) != row['sha256']:
                raise ValueError('Completed seed236 checkpoint changed')
        h.write(queue / 'DONE.json', {'seed236_history_report_sha256': h.digest(result_root / 'report.json'),
            'full_study_complete': False, 'planning_history_evaluations_complete': False})
    except Exception as exc:
        h.write(queue / 'FAILED.json', {'error': str(exc), 'partial_files_preserved': True})
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
    main()
