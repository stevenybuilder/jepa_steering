"""Read-only decoder/arithmetic comparison of the first selected input, no model."""
import importlib.util
import json
from pathlib import Path
import numpy as np

path = Path(__file__).with_name('validate_protected_inputs_cpu.py')
spec = importlib.util.spec_from_file_location('input_validation', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
from app.plan_common.datasets.droid_dset import DROIDVideoDataset

mod.check()
row = json.loads((mod.OUT / 'droid/clips.json').read_text())[0]
expected = np.load(mod.OUT / 'droid' / row['prepared_file'])
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
print(json.dumps({name: {'same_shape': a.shape == b.shape, 'exact': bool(np.array_equal(a,b)),
    'max_abs_difference': float(np.max(np.abs(a.astype('float64') - b.astype('float64')))),
    'different_values': int(np.count_nonzero(a != b)), 'actual_dtype': str(a.dtype), 'expected_dtype': str(b.dtype)}
    for name, (a,b) in pairs.items()}, indent=2))
