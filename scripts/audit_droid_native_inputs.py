"""Audit all pinned DROID inputs without amending the original exact-parity gate.

Reports numeric differences, preserves native-reader outputs, and makes no
scientific readiness claim. No filtering, replacement, fitting or model calls.
"""
import importlib.util
import json
from pathlib import Path
import numpy as np

path = Path(__file__).with_name('validate_protected_inputs_cpu.py')
spec = importlib.util.spec_from_file_location('input_validation', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
from app.plan_common.datasets.droid_dset import DROIDVideoDataset


def run():
    mod.check()
    target = mod.OUT / 'native-droid-diagnostic-v1'
    target.mkdir(exist_ok=False)
    manifest = mod.OUT / 'droid/clips.json'
    original = json.loads((mod.OUT / 'droid/report.json').read_text())
    assert mod.sha256(manifest) == original['clips_sha256']
    mod.write_json(target / 'protocol.json', {
        'source_sha256': mod.sha256(Path(__file__)), 'clips_sha256': mod.sha256(manifest),
        'purpose': 'all-recording exact and numeric diagnostic; native input materialization',
        'original_exact_gate_unchanged': True, 'outcome_filtering': False, 'model_calls': 0})
    results = []
    for row in json.loads(manifest.read_text()):
        prepared = mod.OUT / 'droid' / row['prepared_file']
        assert mod.sha256(prepared) == row['prepared_sha256']
        with np.load(prepared) as expected:
            native = DROIDVideoDataset.__new__(DROIDVideoDataset)
            native.rng = np.random.RandomState(mod.SEED + row['priority'])
            native.h5_name = 'trajectory.h5'; native.camera_views = ['left_mp4_path']
            native.frames_per_clip = 5; native.fps = 4; native.frameskip = 1
            native.action_skip = 1; native.camera_frame = False; native.transform = None
            pixels, actions, states, extrinsics, indices = native.loadvideo_decord(
                str(mod.INPUT / 'droid/raw' / row['directory']))
            pairs = {'pixels': (pixels.numpy(), (expected['pixels'] / 255.).astype('float32').transpose(0,3,1,2)),
                'actions': (actions, expected['actions']), 'states': (states, expected['states']),
                'extrinsics': (extrinsics, expected['extrinsics']), 'indices': (indices, expected['indices'])}
            checks = {}
            for name, (actual, want) in pairs.items():
                if actual.shape != want.shape or not np.isfinite(actual).all():
                    raise ValueError(f'Invalid native {name}, priority {row["priority"]}')
                checks[name] = {'exact': bool(np.array_equal(actual, want)),
                    'max_abs_difference': float(np.max(np.abs(actual.astype('float64') - want.astype('float64')))),
                    'different_values': int(np.count_nonzero(actual != want))}
            native_path = target / row['prepared_file']
            np.savez_compressed(native_path, pixels=pixels.numpy(), actions=actions,
                                states=states, extrinsics=extrinsics, indices=indices)
            results.append({'priority': row['priority'], 'directory': row['directory'],
                'checks': checks, 'native_file': native_path.name, 'native_sha256': mod.sha256(native_path)})
            print(json.dumps({'native_inputs_audited': len(results)}), flush=True)
    mod.write_json(target / 'report.json', {'recordings': len(results), 'records': results,
        'original_exact_gate_unchanged': True, 'original_exact_gate_passed': False,
        'normalization_preprocessor_parity': False, 'scientific_launch_ready': False, 'model_calls': 0})


if __name__ == '__main__':
    run()
