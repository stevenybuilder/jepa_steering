"""CPU contract/negative tests; these do not count as GPU behavioral validation."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from offline_study.protocol import sha256, write_json
from offline_study import refined_task_behavior as behavior
from offline_study.refined_task_fit import selection, CHECKPOINTS
from offline_study.refined_fit_inputs import selected_files


def cohort():
    return {'task': 'pusht', 'dataset': 'pusht', 'fit': [dict(index=i, task='pusht',
        split='fit', lineage_group=f'g{i}', trajectory_id=f't{i}') for i in range(128)],
        'evaluation': [], 'all_protected_groups': [], 'all_author_validation_groups': [],
        'protected_official_validation': [], 'coverage': {'evaluated_rows': 0, 'official_rows': 0}}


def calls(edited=False):
    out = []
    for _ in range(30):
        for n in (300, 1):
            r = dict(backend_calls=1, response_probe_rollouts=0, native_shadow_rollouts=0)
            if edited:
                r.update(requested_l2=[.5] * n, realized_l2=[.5] * n,
                    coefficients=[[.5, 0, 0, 0]] * n, active=[True] * n)
            out.append(dict(horizon=6, candidates=n, record=r))
    return out


class RefinedCompletionTests(unittest.TestCase):
    def test_task_checkpoints_are_distinct_and_droid_not_six_block(self):
        self.assertEqual(len({CHECKPOINTS[t] for t in behavior.TASKS}), 3)
        self.assertTrue(all(len(CHECKPOINTS[t]) == 64 for t in behavior.TASKS))
        with self.assertRaises(ValueError):
            behavior.make_protocol('droid', {})

    def test_three_conditions_96_total_and_unchanged_global_contrast_family(self):
        for task in behavior.TASKS:
            p = behavior.make_protocol(task, {})
            self.assertEqual(len(p['episodes']) * len(p['arms']), 288)
            self.assertEqual(p['analysis']['family'], 12)
            self.assertFalse(p['legacy_solver'])
            self.assertFalse(p['historical_native_reuse'])

    def test_changed_science_or_source_rejected_even_with_rehashed_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = behavior.make_protocol('pusht', {})
            def save(value):
                write_json(root / 'protocol.json', value)
                write_json(root / 'FROZEN.json', {'protocol_sha256': sha256(root / 'protocol.json')})
            save(p); behavior.validate_protocol(root)
            for key, value in [('episodes_per_condition', 768), ('arms', ['native']),
                    ('source_sha256', 'bad'), ('protected_access', True), ('precision', 'bfloat16')]:
                save({**p, key: value})
                with self.subTest(key=key), self.assertRaises(ValueError):
                    behavior.validate_protocol(root)

    def test_fit_selection_and_input_extraction_have_no_validation_rows(self):
        c = cohort(); selected = selection(c)
        self.assertEqual([len(v) for v in selected.values()], [24, 8])
        self.assertFalse({r['index'] for r in selected['calibration']} &
                         {r['index'] for r in selected['response_audit']})
        files = selected_files(c)
        self.assertTrue(all(name.startswith('train/') for name in files))
        self.assertIn('train/obses/episode_000.mp4', files)
        self.assertIn('train/velocities.pth', files)
        bad = copy.deepcopy(c); bad['all_author_validation_groups'] = ['g0']
        with self.assertRaises(ValueError): selected_files(bad)
        bad = copy.deepcopy(c); bad['all_protected_groups'] = ['g0']
        with self.assertRaises(ValueError): selection(bad)

    def test_dose_and_work_checks_reject_mismatched_controls(self):
        behavior.validate_calls(calls(), 'native', .5)
        behavior.validate_calls(calls(True), 'fixed_rank4', .5)
        behavior.validate_calls(calls(True), 'matched_random_fixed_rank4', .5)
        for mode in ('extra_work', 'dose', 'truncate', 'nonfinite', 'native_edited'):
            data = calls(True)
            arm = 'matched_random_fixed_rank4'
            if mode == 'extra_work': data[0]['record']['backend_calls'] = 17
            elif mode == 'dose': data[0]['record']['realized_l2'][0] *= 2
            elif mode == 'truncate': data.pop()
            elif mode == 'nonfinite': data[0]['record']['coefficients'][0][0] = float('nan')
            else: arm = 'native'
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                behavior.validate_calls(data, arm, .5)

    def test_no_new_randomness_or_cross_scenario_pairing(self):
        row = dict(episode=0, logical_rank=0, local_seed=2, environment_seed=2,
            result={'initial_sha256': 'i', 'goal_sha256': 'g'})
        behavior.paired(row, row, 'wall')
        for key in ('episode', 'logical_rank', 'local_seed', 'environment_seed'):
            with self.assertRaises(ValueError):
                behavior.paired(row, {**row, key: 3}, 'wall')
        with self.assertRaises(ValueError):
            behavior.paired(row, {**row, 'result': {'initial_sha256': 'i', 'goal_sha256': 'changed'}}, 'wall')

    def test_observer_restores_original_methods_on_simulator_failure(self):
        original = lambda *a, **k: None
        agent = SimpleNamespace(act=original, planner=SimpleNamespace(unroll=original))
        adapter = SimpleNamespace(last_record={})
        row = dict(episode=0, logical_rank=0, local_seed=2, environment_seed=2)
        with tempfile.TemporaryDirectory() as tmp, patch.object(behavior, 'FixedResponseIntervention', return_value=adapter), \
                patch.object(behavior, 'run_episode', side_effect=RuntimeError('simulator failed')):
            with self.assertRaises(RuntimeError):
                behavior.one_episode(None, None, agent, None, row, 'native', {'dose': .5}, Path(tmp))
        self.assertIs(agent.act, original)
        self.assertIs(agent.planner.unroll, original)


if __name__ == '__main__': unittest.main()
