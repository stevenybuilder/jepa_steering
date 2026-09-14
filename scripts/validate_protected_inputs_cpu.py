"""Input-only parallel checks. FFmpeg checks are NOT native Decord parity.

Never loads model weights, computes model outcomes, or rents infrastructure.
The separately invocable native-droid mode requires the upstream Linux reader.
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('MPLBACKEND', 'Agg')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from offline_study.core.protocol import sha256, write_json

INPUT = ROOT / 'artifacts/offline_study/protected-preparation-20260912-v1'
OUT = ROOT / 'artifacts/offline_study/protected-input-cpu-validation-20260912-v1'
VENDOR = ROOT / 'vendor/jepa-wms'
sys.path.insert(0, str(VENDOR))
SEED = 2026091211


def array_hash(a):
    import numpy as np
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()


def freeze():
    OUT.mkdir(parents=True, exist_ok=False)
    sources = [INPUT / 'droid/report.json', INPUT / 'droid/FILES.json',
               INPUT / 'droid/eligible_candidates.json', INPUT / 'pusht/engineering_candidates.json',
               VENDOR / 'app/plan_common/datasets/droid_dset.py',
               VENDOR / 'app/plan_common/datasets/transforms.py',
               VENDOR / 'evals/simu_env_planning/envs/pusht_env/pusht_env.py']
    write_json(OUT / 'protocol.json', {
        'role': 'CPU_candidate_input_validation_no_model_outcomes', 'seed': SEED,
        'script_sha256': sha256(Path(__file__)),
        'inputs': {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        'droid': {'recordings': 64, 'frames_per_clip': 5, 'fps': 4,
                  'endpoint_rng': 'RandomState(SEED+priority); native one-camera choice then native window selection',
                  'checks': 'full FFmpeg decode; deterministic five-frame extraction; upstream pose-to-action arithmetic',
                  'native_decord_parity': 'separate native-droid mode, must not infer from FFmpeg success'},
        'pusht': {'engineering_pairs': 8, 'steps': 30, 'repeats': 2,
                  'actions': 'fixed bounded sinusoid; engineering only, not scientific trajectory collection'},
        'workers': 4, 'failure_policy': 'preserve partial outputs, no dropping/replacing candidates',
        'scientific_launch_ready': False, 'model_calls': 0, 'gpu_calls': 0})
    write_json(OUT / 'FROZEN.json', {'protocol_sha256': sha256(OUT / 'protocol.json')})


def check():
    p = json.loads((OUT / 'protocol.json').read_text())
    if sha256(OUT / 'protocol.json') != json.loads((OUT / 'FROZEN.json').read_text())['protocol_sha256']:
        raise ValueError('Validation protocol changed')
    if p['script_sha256'] != sha256(Path(__file__)):
        raise ValueError('Validation source changed')
    for name, want in p['inputs'].items():
        if sha256(ROOT / name) != want:
            raise ValueError('Bound input/source changed: ' + name)
    return p


def upstream_pose_converter():
    import numpy as np
    from scipy.spatial.transform import Rotation
    source = VENDOR / 'app/plan_common/datasets/droid_dset.py'
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'poses_to_diffs')
    scope = {'np': np, 'Rotation': Rotation}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
    return scope['poses_to_diffs']


def droid_one(row):
    import h5py
    import numpy as np
    from scipy.spatial.transform import Rotation
    raw = INPUT / 'droid/raw'
    camera = raw / row['left_video']
    # A complete decoder pass catches corrupt frames outside the selected window.
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-err_detect', 'explode',
                    '-threads', '1', '-i', str(camera), '-map', '0:v:0', '-an',
                    '-f', 'null', '-'], check=True, capture_output=True, timeout=300)
    stride = math.ceil(row['fps'] / 4)
    rng = np.random.RandomState(SEED + row['priority'])
    rng.randint(0, 1)
    end = int(rng.randint(5 * stride, row['video_frames']))
    indices = np.arange(end - 5 * stride, end, stride)
    selected = '+'.join(f'eq(n\\,{int(i)})' for i in indices)
    proc = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-threads', '1',
        '-i', str(camera), '-vf', 'select=' + selected, '-vsync', '0',
        '-frames:v', '5', '-pix_fmt', 'rgb24', '-threads', '1', '-f', 'rawvideo', '-'],
        check=True, capture_output=True, timeout=300)
    pixels = np.frombuffer(proc.stdout, dtype='uint8')
    shape = (5, row['height'], row['width'], 3)
    if pixels.size != math.prod(shape):
        raise ValueError('Incomplete frame extraction')
    pixels = pixels.reshape(shape)
    with h5py.File(raw / row['directory'] / 'trajectory.h5', 'r') as f:
        obs = f['observation']
        full = np.c_[np.asarray(obs['robot_state/cartesian_position']),
                     np.asarray(obs['robot_state/gripper_position'])]
        states = full[indices]
        extrinsics = np.asarray(obs['camera_extrinsics'][camera.stem + '_left'])[indices]
    if not np.isfinite(states).all() or extrinsics.shape != (5, 6) or not np.isfinite(extrinsics).all():
        raise ValueError('Invalid camera/state mapping')
    actions = upstream_pose_converter()(states)
    if actions.shape != (4, 7) or not np.isfinite(actions).all():
        raise ValueError('Invalid native action conversion')
    if not np.array_equal(actions[:, :3], np.diff(states[:, :3], axis=0)):
        raise ValueError('XYZ action mismatch')
    angles = Rotation.from_euler('xyz', states[:, 3:6]).as_matrix()
    reconstructed = Rotation.from_euler('xyz', actions[:, 3:6]).as_matrix() @ angles[:-1]
    if not np.allclose(reconstructed, angles[1:], atol=1e-10, rtol=1e-10):
        raise ValueError('Action rotation round-trip failure')
    target = OUT / 'droid' / f'priority-{row["priority"]:03d}.npz'
    np.savez_compressed(target, pixels=pixels, states=states, actions=actions,
                        extrinsics=extrinsics, indices=indices)
    return {**row, 'indices': indices.tolist(), 'pixels_sha256': array_hash(pixels),
            'states_sha256': array_hash(states), 'actions_sha256': array_hash(actions),
            'prepared_file': target.name, 'prepared_sha256': sha256(target),
            'full_ffmpeg_decode_passed': True, 'pose_action_roundtrip_passed': True}


def droid():
    p = check(); target = OUT / 'droid'; target.mkdir(exist_ok=False)
    start = time.monotonic()
    rows = json.loads((INPUT / 'droid/eligible_candidates.json').read_text())['selected']
    files = json.loads((INPUT / 'droid/FILES.json').read_text())
    for name, entry in files.items():
        if sha256(INPUT / 'droid/raw' / name) != entry['sha256']:
            raise ValueError('Downloaded bytes changed')
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=p['workers']) as pool:
        for result in pool.map(droid_one, rows):
            results.append(result)
            write_json(target / 'progress.json', {'checked': len(results), 'target': 64})
            print(json.dumps({'droid_input_checks': len(results), 'target': 64}), flush=True)
    write_json(target / 'clips.json', results)
    write_json(target / 'report.json', {'status': 'ffmpeg_and_state_action_checks_complete',
        'recordings': len(results), 'seconds': time.monotonic() - start,
        'clips_sha256': sha256(target / 'clips.json'), 'model_calls': 0,
        'native_decord_parity': False, 'population_compatibility_approved': False,
        'scientific_launch_ready': False, 'protocol_sha256': sha256(OUT / 'protocol.json')})


def pusht_one(row):
    import numpy as np
    from evals.simu_env_planning.envs.pusht_env.pusht_env import PushTEnv
    traces = []
    for repeat in range(2):
        env = PushTEnv(with_velocity=True, with_target=False)
        try:
            env.seed(row['seed']); env.reset_to_state = np.asarray(row['initial_state'])
            obs, state = env.reset()
            states, hashes = [np.asarray(state).copy()], [array_hash(obs['visual'])]
            for step in range(30):
                action = np.asarray([0.2 * math.sin(step / 4), 0.2 * math.cos(step / 4)])
                obs, _, _, info = env.step(action)
                states.append(np.asarray(info['state']).copy()); hashes.append(array_hash(obs['visual']))
            values = np.asarray(states)
            if values.shape != (31, 7) or not np.isfinite(values).all():
                raise ValueError('Invalid simulator state trace')
            traces.append({'states_sha256': array_hash(values), 'frame_hashes': hashes})
        finally:
            env.close()
    if traces[0] != traces[1]:
        raise ValueError('Repeated simulator state/pixel trace differs')
    return {'candidate': row['candidate'], 'steps': 30, 'repeats_exact': True, **traces[0]}


def pusht():
    p = check(); target = OUT / 'pusht'; target.mkdir(exist_ok=False)
    start = time.monotonic()
    rows = json.loads((INPUT / 'pusht/engineering_candidates.json').read_text())
    with concurrent.futures.ProcessPoolExecutor(max_workers=p['workers']) as pool:
        results = list(pool.map(pusht_one, rows))
    write_json(target / 'report.json', {'status': 'excluded_engineering_physics_render_repeat_complete',
        'cases': results, 'seconds': time.monotonic() - start,
        'packages': {name: importlib.metadata.version(name) for name in ('numpy','gym','pymunk','pygame')},
        'receiving_linux_parity': False, 'fresh_scientific_source_trajectories': 0,
        'model_calls': 0, 'scientific_launch_ready': False})
    print(json.dumps({'pusht_engineering_cases': len(results), 'repeat_exact': True}), flush=True)


def native_droid():
    import numpy as np
    from app.plan_common.datasets.droid_dset import DROIDVideoDataset
    check(); target = OUT / 'native-droid'; target.mkdir(exist_ok=False)
    report = json.loads((OUT / 'droid/report.json').read_text())
    if sha256(OUT / 'droid/clips.json') != report['clips_sha256']:
        raise ValueError('Prepared clip manifest changed')
    results = []
    for row in json.loads((OUT / 'droid/clips.json').read_text()):
        if sha256(OUT / 'droid' / row['prepared_file']) != row['prepared_sha256']:
            raise ValueError('Prepared clip bytes changed')
        expected = np.load(OUT / 'droid' / row['prepared_file'])
        native = DROIDVideoDataset.__new__(DROIDVideoDataset)
        native.rng = np.random.RandomState(SEED + row['priority'])
        native.h5_name = 'trajectory.h5'; native.camera_views = ['left_mp4_path']
        native.frames_per_clip = 5; native.fps = 4; native.frameskip = 1
        native.action_skip = 1; native.camera_frame = False; native.transform = None
        pixels, actions, states, extrinsics, indices = native.loadvideo_decord(
            str(INPUT / 'droid/raw' / row['directory']))
        expected_pixels = (expected['pixels'] / 255.0).astype('float32').transpose(0, 3, 1, 2)
        for actual, want in ((pixels.numpy(), expected_pixels), (actions, expected['actions']),
                             (states, expected['states']), (extrinsics, expected['extrinsics']),
                             (indices, expected['indices'])):
            if not np.array_equal(actual, want):
                raise ValueError('Native decoder/action parity differs at priority ' + str(row['priority']))
        results.append(row['priority'])
    write_json(target / 'report.json', {'status': 'native_raw_reader_selected_clips_exact',
        'recordings': len(results), 'priorities': results, 'model_calls': 0,
        'normalization_preprocessor_parity': False, 'scientific_launch_ready': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'droid', 'pusht', 'native-droid'))
    args = parser.parse_args()
    try:
        {'freeze': freeze, 'droid': droid, 'pusht': pusht, 'native-droid': native_droid}[args.mode]()
    except Exception as exc:
        if OUT.exists():
            write_json(OUT / ('FAILED-' + args.mode + '.json'), {'error': str(exc),
                'partial_inputs_retained': True, 'scientific_launch_ready': False})
        raise
