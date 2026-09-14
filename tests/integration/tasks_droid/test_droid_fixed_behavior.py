"""DROID behavioral contracts; these do not count as completed GPU receiving tests."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from offline_study.tasks.droid import droid_fixed_response_behavior as behavior


def calls(edited):
    result = []
    for _ in range(15):
        for count in (300, 1):
            record = {'backend_calls': 1, 'response_probe_rollouts': 0, 'native_shadow_rollouts': 0, 'horizon': 3}
            if edited:
                record.update(active=[True] * count, requested_l2=[.5] * count,
                    realized_l2=[.5] * count, coefficients=[[.5, 0, 0, 0]] * count)
            result.append({'horizon': 3, 'candidates': count, 'record': record})
    return result


class DroidBehaviorTests(unittest.TestCase):
    def test_64_total_three_conditions_and_global_12_contrasts(self):
        protocol = behavior.protocol_for({'episodes': list(range(64))}, {})
        self.assertEqual(protocol['total_panel_evaluations'], 192)
        self.assertEqual(protocol['analysis']['family'], 12)
        self.assertEqual(protocol['robot_executions'], 0)
        self.assertFalse(protocol['legacy_solver'])
        self.assertFalse(protocol['historical_native_reuse'])
        self.assertFalse(protocol['candidate_admission_requires_offline_improvement'])
        with self.assertRaises(ValueError): behavior.protocol_for({'episodes': list(range(96))}, {})

    def test_full_cem_budget_and_delivered_control_dose(self):
        behavior.validate_calls(calls(False), 'native', .5)
        for arm in behavior.ARMS[1:]: behavior.validate_calls(calls(True), arm, .5)
        for issue in ('budget', 'work', 'dose', 'finite', 'native_edit', 'active'):
            data = calls(True)
            arm = 'matched_random_fixed_rank4'
            if issue == 'budget': data.pop()
            elif issue == 'work': data[0]['record']['backend_calls'] = 17
            elif issue == 'dose': data[0]['record']['realized_l2'][0] = 1.
            elif issue == 'finite': data[0]['record']['coefficients'][0][0] = float('nan')
            elif issue == 'active': data[0]['record']['active'][0] = False
            else: arm = 'native'
            with self.subTest(issue=issue), self.assertRaises(ValueError): behavior.validate_calls(data, arm, .5)

    def test_failed_native_episode_restores_planner(self):
        original = lambda *a, **k: None
        agent = SimpleNamespace(planner=SimpleNamespace(unroll=original))
        adapter = SimpleNamespace(last_record={})
        with tempfile.TemporaryDirectory() as tmp, patch.object(behavior, 'Intervention', return_value=adapter), \
                patch.object(behavior, 'full_episode', side_effect=RuntimeError('native failed')):
            with self.assertRaises(RuntimeError):
                behavior.episode(None, None, None, None, {'dose': .5}, 'native', Path(tmp),
                    {'episode': 0, 'environment_seed': 1}, agent, False)
        self.assertIs(agent.planner.unroll, original)
