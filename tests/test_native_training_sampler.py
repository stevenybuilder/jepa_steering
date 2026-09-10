"""Compare to actual unchanged upstream caller + init_data sampler construction.

Only dataset IO is replaced by range fixtures: flags, native branches, kwargs,
DistributedSampler, BatchSampler and DataLoader are all the pinned implementation.
This runs locally without the optional simulator/video packages; a separate
read-only receiving check exercised the same call on all2000 real PointMaze rows.
"""
import ast
import logging
from pathlib import Path
from typing import Callable
import unittest
import warnings

import torch
import yaml

from offline_study.navigation_input_check import CONFIGS
from offline_study.pointmaze_training_history import SAMPLER_POLICY, validation_batches
from offline_study.training_pilot import VirtualRankBatchSampler, native_sampler_policy


VENDOR = Path(__file__).resolve().parents[1] / 'vendor/jepa-wms'


def actual_upstream_loaders(task, train_length, validation_length, rank=0):
    """Execute actual source AST; do not prescribe expected shuffle flags."""
    cfg = yaml.safe_load((VENDOR / CONFIGS[task]).read_text())
    data = cfg['data']
    dataset_io = lambda *args, **kwargs: ({'train': range(train_length), 'valid': range(validation_length)},
                                        {'train': range(train_length), 'valid': range(validation_length)})
    source = VENDOR / 'app/plan_common/datasets/utils.py'
    function = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'init_data')
    logger = logging.getLogger('native-sampler-test')
    logger.setLevel(logging.CRITICAL + 1)
    globals_ = {'torch': torch, 'Callable': Callable, 'logger': logger,
        'load_point_maze_slice_train_val': dataset_io, 'load_wall_slice_train_val': dataset_io}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), globals_)
    caller_source = VENDOR / 'app/vjepa_wm/train.py'
    main = next(n for n in ast.parse(caller_source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    first = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign) and
                 any(isinstance(t, ast.Name) and t.id == 'excluded_keys' for t in n.targets))
    last = next(i for i, n in enumerate(main.body) if i > first and isinstance(n, ast.Assign) and
                isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == 'init_data')
    namespace = {'cfgs_data': data, 'cfgs_validation': data['validation'], 'cfgs_loader': data['loader'],
        'cfgs_custom': data['custom'], 'cfgs_droid': data['droid'],
        'dataset_paths': ['/fixture/point_maze' if task == 'pointmaze' else '/fixture/wall_single'],
        'val_dataset_paths': [], 'transform': None, 'world_size': cfg['nodes'] * cfg['tasks_per_node'],
        'rank': rank, 'filter_first_episodes': data['custom']['filter_first_episodes'],
        'num_workers': data['loader']['num_workers'], 'init_data': globals_['init_data']}
    with warnings.catch_warnings():
        # Native16-worker kwargs are preserved, but only samplers are iterated;
        # no DataLoader worker processes are created by this CPU fixture.
        warnings.filterwarnings('ignore', message='This DataLoader will create .*worker processes.*', category=UserWarning)
        exec(compile(ast.Module(body=main.body[first:last+1], type_ignores=[]), str(caller_source), 'exec'), namespace)
    return namespace['unsupervised_loader'], namespace['val_unsupervised_loader'], namespace['data_kwargs']


class ActualNativeTrainingSamplerTests(unittest.TestCase):
    def test_pointmaze_complete_sampler_streams_match_native_construction(self):
        train_length, val_length = 145800, 12200
        actual_val = validation_batches(val_length)
        for epoch in (0, 1):
            actual_train = list(VirtualRankBatchSampler(train_length, epoch=epoch, shuffle=SAMPLER_POLICY['train_shuffle']))
            for rank in range(16):
                native_train, native_val, kwargs = actual_upstream_loaders('pointmaze', train_length, val_length, rank)
                self.assertNotIn('shuffle', kwargs)
                self.assertEqual(native_train.sampler.shuffle, SAMPLER_POLICY['train_shuffle'])
                self.assertEqual(native_val.sampler.shuffle, SAMPLER_POLICY['validation_shuffle'])
                native_train.sampler.set_epoch(epoch)
                self.assertEqual(list(native_train.batch_sampler), [batch[rank*8:(rank+1)*8] for batch in actual_train])
                self.assertEqual(list(native_val.batch_sampler), [batch[rank] for batch in actual_val])
        self.assertEqual(len(actual_train), 1139)
        self.assertEqual(len(actual_val), 191)
        self.assertEqual(sum(map(len, actual_val[-1])), 48)

    def test_historical_extra_shuffle_is_observably_different(self):
        train, val, _ = actual_upstream_loaders('pointmaze', 145800, 12200)
        old_train = next(iter(VirtualRankBatchSampler(145800)))[:8]
        old_val = list(torch.utils.data.DistributedSampler(range(12200), num_replicas=16, rank=0, shuffle=True))[:4]
        self.assertNotEqual(old_train, list(train.sampler)[:8])
        self.assertNotEqual(old_val, list(val.sampler)[:4])
        train.sampler.set_epoch(1)
        self.assertEqual(list(train.sampler)[:8], [0, 16, 32, 48, 64, 80, 96, 112])

    def test_wall_default_behavior_still_matches_native(self):
        policy = native_sampler_policy('wall')
        native_train, native_val, _ = actual_upstream_loaders('wall', 1025, 193)
        self.assertTrue(native_train.sampler.shuffle)
        self.assertTrue(native_val.sampler.shuffle)
        self.assertEqual(policy['train_shuffle'], native_train.sampler.shuffle)
        for epoch in (0, 1):
            native_train.sampler.set_epoch(epoch)
            actual = VirtualRankBatchSampler(1025, epoch=epoch)
            self.assertEqual([row[:8] for row in actual], list(native_train.batch_sampler))
        self.assertFalse(native_sampler_policy('pointmaze')['train_shuffle'])


if __name__ == '__main__':
    unittest.main()
