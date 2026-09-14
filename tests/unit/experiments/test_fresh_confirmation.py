"""CPU contract tests only; these do not establish receiving-GPU readiness."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from offline_study.experiments import fresh_confirmation as fresh
from offline_study.core.protocol import write_json, sha256


class FreshConfirmationTests(unittest.TestCase):
    def test_initial_delivery_keeps_exact_pixels_and_rejects_physics_or_large_render_change(self):
        class TensorMap(dict):
            def clone(self):
                return TensorMap({k: v.clone() for k, v in self.items()})
        visual = torch.zeros(1, 3, 224, 224, dtype=torch.uint8)
        visual[0, 0, 0, 0] = 100
        original = TensorMap({'visual': visual, 'proprio': torch.ones(1, 4)})
        saved = {'initial': {k: v.clone() for k, v in original.items()}}
        metadata = {'initial_sha256': fresh.observation_digest(original)}
        changed = original.clone()
        changed['visual'][0, 0, 0, 1:5] += 1
        recovered = fresh.restore_initial(changed, saved, metadata)
        self.assertEqual(fresh.observation_digest(recovered), metadata['initial_sha256'])
        for mode in ('physics', 'more_pixels', 'bigger_difference'):
            wrong = original.clone()
            if mode == 'physics':
                wrong['proprio'][0, 0] += .01
            elif mode == 'more_pixels':
                wrong['visual'][0, 0, 0, 1:66] += 1
            else:
                wrong['visual'][0, 0, 0, 1] += 2
            with self.assertRaises(ValueError):
                fresh.restore_initial(wrong, saved, metadata)

    def test_only_pointmaze_camera_is_primed(self):
        from unittest.mock import Mock
        env = Mock()
        for task in ('reach', 'reach-wall', 'wall'):
            fresh.initialize_renderer(task, env)
        env.proprio_env.unwrapped.prepare_for_render.assert_not_called()
        fresh.initialize_renderer('pointmaze', env)
        env.proprio_env.unwrapped.prepare_for_render.assert_called_once_with()

    def test_all_24_worker_assignments_are_disjoint_and_complete(self):
        assigned = [(task, episode) for task in fresh.TASKS for worker in range(6)
                    for episode in fresh.assigned_episodes(worker, 6)]
        self.assertEqual(len(assigned), 384)
        self.assertEqual(len(set(assigned)), 384)
        self.assertEqual(set(assigned), {(t, i) for t in fresh.TASKS for i in range(96)})

    def test_invalid_worker_rejected(self):
        for index, total in ((6, 6), (-1, 6), (0, 0), (0, 97)):
            with self.assertRaises(ValueError):
                fresh.assigned_episodes(index, total)

    def test_arm_and_analysis_registry(self):
        self.assertEqual(len(fresh.ARMS), 8)
        self.assertEqual(len(fresh.CONTRASTS) * len(fresh.TASKS), fresh.ANALYSIS['family_size'])
        for weights in fresh.CONTRASTS.values():
            self.assertEqual(sum(weights.values()), 0)
            self.assertTrue(set(weights).issubset(fresh.ARMS))

    def test_numerical_gate_requires_all_arms_and_all_context_shapes(self):
        keys = [[6, 300, [1, 1, 256, 400], [1, 1, 4]], [6, 300, [1, 2, 256, 400], [1, 2, 4]]]
        checks = []
        for key in keys:
            for arm in fresh.ENGINEERING_POLICY['numerical_arms']:
                checks.append({'context_key': key, 'arm': arm, 'backend_calls': 1,
                    **{name: True for name in ('source_parity_bitwise', 'energy_valid', 'inputs_unchanged',
                                              'parameters_unchanged', 'hooks_restored')}})
        native = {'calls': [{'context_key': key} for key in keys], 'numerical_checks': checks}
        fresh.verify_numerical_coverage(native)
        for bad_checks in (checks[:-1], checks + checks[:1], [{**checks[0], 'energy_valid': False}, *checks[1:]]):
            with self.assertRaises(ValueError):
                fresh.verify_numerical_coverage({**native, 'numerical_checks': bad_checks})

    def test_asset_traversal_rejected(self):
        with self.assertRaises(ValueError):
            fresh.checked_path(Path('/tmp/campaign'), '../other')

    def test_refined_dose_checks(self):
        energy = {'response_probe_rollouts': 0, 'native_shadow_rollouts': 0,
                  'requested_l2': [2.], 'realized_l2': [2.], 'active': [True],
                  'coefficients': [[1., 0., 0., 0.]]}
        calls = [{'horizon': 6, 'candidates': 1, 'energy': energy}]
        fresh.verify_refined_energy(calls, 'fixed_rank4', 2.)
        with self.assertRaises(ValueError):
            fresh.verify_refined_energy(calls, 'fixed_rank4', 3.)
        with self.assertRaises(ValueError):
            fresh.verify_refined_energy(calls, 'native', 2.)

    def test_receipt_refuses_changed_arm_device_input_or_role(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scenario = {'episode': 0}
            report = {'task': 'wall', 'scenario': scenario, 'freeze_sha256': 'freeze',
                      'engineering': False, 'device_uuid': 'GPU-A', 'records_sha256': {}}
            records = {}
            for name in fresh.ARMS:
                record = {'arm': name, 'device_uuid': 'GPU-A', 'freeze_sha256': 'freeze',
                          'scientific_efficacy_measurement': True, 'scenario': scenario}
                records[name] = record
                write_json(root / (name + '.json'), record)
                report['records_sha256'][name] = sha256(root / (name + '.json'))
            def receipt():
                write_json(root / 'report.json', report)
                write_json(root / 'DONE.json', {'report_sha256': sha256(root / 'report.json')})
            receipt()
            with patch.object(fresh, 'verify_record'):
                self.assertEqual(len(fresh.verify_bundle(root, {}, 'wall', scenario, 'freeze', False)), 8)
                for field, value in (('device_uuid', 'GPU-B'), ('arm', 'joint'),
                                     ('freeze_sha256', 'old'), ('scientific_efficacy_measurement', False)):
                    bad = {**records['native'], field: value}
                    write_json(root / 'native.json', bad)
                    report['records_sha256']['native'] = sha256(root / 'native.json')
                    receipt()
                    with self.assertRaises(ValueError):
                        fresh.verify_bundle(root, {}, 'wall', scenario, 'freeze', False)
                with self.assertRaises(ValueError):
                    fresh.verify_bundle(root, {}, 'wall', {'episode': 1}, 'freeze', False)

    def test_native_success_must_be_boolean_and_full_budget(self):
        row = {'episode': 0, 'initial_sha256': 'initial', 'goal_sha256': 'goal'}
        result = {'initial_sha256': 'initial', 'goal_sha256': 'goal', 'native_success': True,
            'native_state_distance': 1., 'native_reward': 0., 'expert_success': 1.,
            'elementary_steps': 30, 'published_candidate_count': 300,
            'planning_calls': [{'iterations': 30, 'returned_model_actions': 6}]}
        calls = [{'horizon': h, 'candidates': n, 'backend_calls': 1} for h, n in [(6, 300), (6, 1)] * 30]
        record = {'scenario': row, 'result': result, 'calls': calls}
        fresh.verify_record(record, row, 'wall')
        for key, bad in (('native_success', 1), ('native_reward', float('nan')), ('elementary_steps', 2)):
            changed = copy.deepcopy(record)
            changed['result'][key] = bad
            with self.assertRaises(ValueError):
                fresh.verify_record(changed, row, 'wall')


if __name__ == '__main__':
    unittest.main()
