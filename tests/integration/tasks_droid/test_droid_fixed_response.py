"""DROID tensor/math checks only; not GPU integration or task-success evidence."""
import copy
from types import SimpleNamespace
from pathlib import Path
import unittest

import torch

from offline_study.tasks.droid.droid_fixed_response import METHOD, FieldHook, Intervention, transform_field, validate_bank
from offline_study.tasks.droid.droid_fixed_response_fit import selection, validate_prefix


def bank():
    width = 256 * 1024
    projection = torch.zeros(3, width)
    projection[torch.arange(3), torch.arange(3)] = 1
    operators = {}
    for name, offset in [('fixed_rank4', 0), ('matched_random_fixed_rank4', 4)]:
        basis = torch.zeros(4, width)
        basis[torch.arange(4), torch.arange(4) + offset] = 1
        operators[name] = {'basis': basis.reshape(4, 256, 1024), 'map': torch.eye(4)}
    return {'method': METHOD, 'schema_version': 1, 'dose': .5, 'mean': torch.zeros(width),
        'projection': projection, 'scale': torch.ones(3), 'operators': operators}


class Predictor(torch.nn.Module):
    def __init__(self, blocks=12):
        super().__init__()
        self.predictor_blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(blocks)])
    def forward(self, value, action, proprio):
        for block in self.predictor_blocks:
            value = block(value)
        return value


class Model:
    def __init__(self):
        self.device = 'cpu'
        self.model = SimpleNamespace(predictor=Predictor())
        self.calls = 0
    def unroll(self, context, act_suffix):
        self.calls += 1
        out = [context]
        value = context
        for action in act_suffix:
            value = self.model.predictor(value, action, None)
            out.append(value)
        return torch.stack(out)


class DroidFixedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_native_and_zero_exact_and_one_forecast(self):
        value = torch.ones(2, 258, 1024)
        actions = torch.zeros(3, 2, 7)
        model = Model()
        expected = model.unroll(value, actions)
        for arm in ('native', 'zero_dose'):
            adapter = Intervention(model, bank(), arm)
            before = model.calls
            self.assertTrue(torch.equal(adapter(value, act_suffix=actions), expected))
            self.assertEqual(model.calls - before, 1)

    def test_exact_site_horizon_dose_and_separate_control(self):
        value = torch.ones(2, 258, 1024)
        model = Model()
        actions = torch.zeros(3, 2, 7)
        outputs = []
        for arm in ('fixed_rank4', 'matched_random_fixed_rank4'):
            adapter = Intervention(model, bank(), arm)
            before = model.calls
            result = adapter(value, act_suffix=actions)
            self.assertEqual(model.calls - before, 1)
            self.assertTrue(torch.equal(result[:3], value.expand(3, -1, -1, -1)))
            self.assertTrue(torch.equal(result[3, :, :2], value[:, :2]))
            self.assertTrue(torch.allclose((result[3] - value).flatten(1).norm(dim=1), torch.full((2,), .5), atol=2e-6))
            self.assertEqual(adapter.last_record['response_probe_rollouts'], 0)
            self.assertEqual(adapter.last_record['native_shadow_rollouts'], 0)
            self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks for m in model.model.predictor.modules()))
            outputs.append(result)
        self.assertFalse(torch.equal(*outputs))

    def test_zero_coefficients_do_not_fabricate_direction(self):
        fitted = bank()
        fitted['operators']['fixed_rank4']['map'].zero_()
        value = torch.ones(2, 256, 1024)
        result, record = transform_field(value, fitted, 'fixed_rank4')
        self.assertTrue(torch.equal(value, result))
        self.assertFalse(record['active'].any())

    def test_hooks_removed_on_native_failure_and_other_hooks_rejected(self):
        predictor = Predictor()
        with self.assertRaises(RuntimeError):
            with FieldHook(predictor):
                raise RuntimeError('model failed')
        self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks for m in predictor.modules()))
        hook = predictor.predictor_blocks[0].register_forward_hook(lambda *a: None)
        with self.assertRaises(ValueError):
            with FieldHook(predictor): pass
        hook.remove()
        with self.assertRaises(ValueError):
            with FieldHook(Predictor(6)): pass

    def test_wrong_bank_dimension_nonorthogonal_or_nonfinite_rejected(self):
        for issue in ('method', 'dimension', 'orthogonality', 'finite'):
            fitted = bank()
            if issue == 'method': fitted['method'] = 'fixed_response_rank4_v1'
            elif issue == 'dimension': fitted['mean'] = torch.zeros(256 * 400)
            elif issue == 'orthogonality': fitted['operators']['fixed_rank4']['basis'] *= 2
            else: fitted['operators']['fixed_rank4']['map'][0, 0] = float('nan')
            with self.subTest(issue=issue), self.assertRaises(ValueError): validate_bank(fitted)

    def test_original_recording_order_and_disjoint_calibration_partition(self):
        rows = [{'directory': f'source/{i:03d}'} for i in range(128)]
        first = selection(rows)
        self.assertEqual(first, selection(copy.deepcopy(rows)))
        self.assertEqual([len(v) for v in first.values()], [24, 8])
        self.assertFalse(set(first['calibration']) & set(first['response_audit']))
        with self.assertRaises(ValueError): selection(rows[:-1] + [rows[0]])

    def test_relocation_preserves_directory_pixels_actions_and_state(self):
        original = {'path': '/original/raw/robotics/droid/trajectory', 'indices': [1, 2, 3, 4],
            'state_sha256': 's', 'visual_sha256': 'v', 'actions_sha256': 'a'}
        current = {**original, 'path': '/restored/raw/robotics/droid/trajectory'}
        validate_prefix(current, original, 'robotics/droid/trajectory', Path('/restored/raw'))
        for key, value in [('path', current['path'] + '/camera.mp4'), ('indices', [2, 3, 4, 5]),
                ('state_sha256', 'changed'), ('visual_sha256', 'changed'), ('actions_sha256', 'changed')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_prefix({**current, key: value}, original, 'robotics/droid/trajectory', Path('/restored/raw'))
