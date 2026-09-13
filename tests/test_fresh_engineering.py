"""Contract tests; actual GPU/native/reference parity remains a receiving gate."""
import unittest
from unittest.mock import patch

import torch
from torch import nn

from offline_study import fresh_engineering as checks


class Model(nn.Module):
    def unroll(self, context, act_suffix):
        return {key: value.clone() for key, value in context.items()}


class Backend:
    def __init__(self):
        self.model, self.predictor, self.device = Model(), nn.Identity(), 'cpu'
    def predict(self, context, actions):
        return self.model.unroll(context, act_suffix=actions)


class Adapter:
    def __init__(self, backend, *args):
        self.backend, self.energy = backend, []
        self.last_record = {'response_probe_rollouts': 0, 'native_shadow_rollouts': 0}
    def __call__(self, context, actions):
        self.energy.append({'requested_squared_l2_mean': 0., 'realized_squared_l2_mean': 0.})
        return self.backend.predict(context, actions)


class FreshEngineeringTests(unittest.TestCase):
    def setUp(self):
        self.context = {'visual': torch.ones(1, 1, 4), 'proprio': torch.ones(1, 1, 2)}
        self.actions = torch.zeros(2, 1, 3)
        self.backend = Backend()

    def test_keys_distinguish_observed_context_lengths(self):
        key = checks.context_shape_key(self.context, self.actions)
        other = {k: v.repeat(1, 2, 1) for k, v in self.context.items()}
        self.assertNotEqual(key, checks.context_shape_key(other, self.actions))

    def test_population_must_match_real_planner(self):
        for actions in (torch.zeros(6, 8, 3), torch.zeros(7, 300, 3)):
            with self.assertRaises(ValueError):
                checks.context_shape_key(self.context, actions)

    def test_all_nine_shortened_cases_have_bound_receipts(self):
        with patch.object(checks, 'FixedResponseIntervention', Adapter), patch.object(checks, 'H6StaticPlanningIntervention', Adapter):
            result = checks.validate_all_arms(self.context, self.actions, self.backend, {'dose': 1.}, {}, {})
        self.assertEqual({r['arm'] for r in result}, set(checks.ARMS))
        self.assertEqual(len(result), 9)
        for row in result:
            self.assertEqual(row['context_key'], [2, 1, [1, 1, 4], [1, 1, 2]])
            self.assertTrue(row['source_parity_bitwise'] and row['energy_valid'] and row['hooks_restored'])
            self.assertEqual(row['backend_calls'], 1)

    def test_extra_model_call_rejected_and_wrapper_restored(self):
        before = self.backend.model.unroll
        def twice(context, actions):
            self.backend.predict(context, actions)
            return self.backend.predict(context, actions)
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            checks._one_call(twice, self.backend, self.context, self.actions)
        self.assertEqual(self.backend.model.unroll, before)

    def test_nonfinite_or_changed_output_rejected(self):
        for number in (float('nan'), 2.):
            wrong = {k: v.clone() for k, v in self.context.items()}
            wrong['visual'][0, 0, 0] = number
            with self.assertRaises(ValueError):
                checks._same_output(wrong, self.context)

    def test_refined_energy_active_and_zero_rules(self):
        energy = {'response_probe_rollouts': 0, 'native_shadow_rollouts': 0,
                  'requested_l2': [2.], 'realized_l2': [2.], 'active': [True],
                  'coefficients': [[1., 0., 0., 0.]]}
        checks._refined_energy(energy, 'fixed_rank4', 6, 1, 2.)
        for arm, horizon, dose in (('native', 6, 2.), ('fixed_rank4', 5, 2.), ('fixed_rank4', 6, 3.)):
            with self.assertRaises(ValueError):
                checks._refined_energy(energy, arm, horizon, 1, dose)

    def test_capture_hooks_restored_on_incomplete_rollout(self):
        predictor = nn.Module()
        predictor.predictor_blocks = nn.ModuleList([nn.Identity() for _ in range(6)])
        before = checks._hooks(predictor)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            with checks._capture_b3(predictor):
                pass
        self.assertEqual(checks._hooks(predictor), before)


if __name__ == '__main__':
    unittest.main()
