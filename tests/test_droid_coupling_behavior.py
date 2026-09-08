import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from offline_study.droid_coupling import ARMS, Intervention, arm_fields
from offline_study.droid_coupling_behavior import (
    IndependentFields, PAIRS, analyze, paired_inputs, reference_fields, validate_records, scientific_shard)
from offline_study.protocol import sha256, write_json
from offline_study.droid_contract import action_metrics
from offline_study.planning_contract import seed_schedule
import test_droid_coupling as fixtures


class DroidBehaviorTests(unittest.TestCase):
    def record(self, row, arm='native', error=.03):
        actions = torch.zeros(3, 7, dtype=torch.float64); actions[0, 0] = error
        result = {'initial_sha256': str(row['episode']), 'goal_sha256': 'goal-' + str(row['episode']),
            'dataset_sample': {'path': 'recording-' + str(row['episode'] % 15),
                               'raw_frame_indices': [row['episode'] + i for i in range(5)]},
            'planned_actions': actions.tolist(), 'recorded_actions': torch.zeros_like(actions).tolist(),
            'metrics': {k: float(v) for k, v in action_metrics(actions, torch.zeros_like(actions)).items()},
            'robot_executions': 0, 'dummy_success_intentionally_omitted': True,
            'unroll_calls': [[3, 300], [3, 1]] * 15}
        return {**row, 'arm': arm, 'result': result, 'seconds': 1.}

    def test_paired_native_streams_and_metric_integrity(self):
        rows = seed_schedule(1, 64, 8, 3)
        records = [self.record(r) for r in rows]
        validate_records(records, rows)
        for key in ('initial_sha256', 'goal_sha256', 'dataset_sample', 'recorded_actions'):
            other = copy.deepcopy(records[0]); other['result'][key] = 'changed'
            with self.assertRaisesRegex(ValueError, 'stimulus pairing'):
                paired_inputs(records[0], other)
        other = copy.deepcopy(records); other[0]['result']['metrics'] = {}
        with self.assertRaisesRegex(ValueError, 'metric'):
            validate_records(other, rows)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            validate_records(records[:-1], rows)
        with self.assertRaisesRegex(ValueError, 'stream order'):
            validate_records(list(reversed(records)), rows)

    def test_complete_clustered_native_score_analysis(self):
        episodes = seed_schedule(1, 64, 8, 3)
        panels = {arm: [self.record(row, arm, .03 if arm == 'native' else .02) for row in episodes] for arm in ARMS}
        report = analyze(panels, episodes)
        self.assertEqual(len(PAIRS), 15)
        self.assertEqual(len(report['contrasts']), 16)
        self.assertAlmostEqual(report['arm_summaries']['native']['checkpoint_score'], 56.)
        self.assertAlmostEqual(report['contrasts'][0]['score_difference'], 8.)
        self.assertEqual(report['arm_summaries']['native']['recordings'], 15)
        self.assertFalse(report['dummy_success_reported'])
        self.assertEqual(report['robot_executions'], 0)
        del panels['joint']
        with self.assertRaisesRegex(ValueError, 'nine'):
            analyze(panels, episodes)

    def test_reference_compiler_and_hooks_independently_match(self):
        bank = fixtures.DroidCouplingTests().bank()
        model = fixtures.Model()
        context = torch.zeros(2, 1, 1, 16, 16, 1024)
        acts = torch.zeros(3, 2, 7)
        for arm in ARMS:
            expected_fields = reference_fields(bank, arm, 'cpu')
            for actual, expected in zip(arm_fields(bank, arm, 'cpu'), expected_fields):
                self.assertTrue(actual is None and expected is None or torch.equal(actual, expected))
            with IndependentFields(model.model.predictor, *expected_fields):
                expected = model.unroll(context, act_suffix=acts)
            actual = Intervention(model, bank, arm)(context, act_suffix=acts)
            self.assertTrue(torch.equal(actual, expected), arm)
        with IndependentFields(model.model.predictor, *reference_fields(bank, 'zero_hook', 'cpu')):
            zero = model.unroll(context, act_suffix=acts)
        self.assertTrue(torch.equal(zero, model.unroll(context, act_suffix=acts)))

    def test_receiving_shard_requires_complete_bound_same_device_native(self):
        episodes = seed_schedule(1, 64, 8, 3)
        selected = [row for row in episodes if row['logical_rank'] == 0]
        protocol = {'planning': {'episodes': episodes}, 'source_sha256': 'source'}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            launch = {'freeze_sha256': 'freeze', 'source_sha256': 'source', 'role': 'scientific_development',
                'arm': 'native', 'logical_ranks': [0], 'device_uuid': 'gpu-a', 'engineering_report_sha256': 'proof'}
            write_json(root / 'protocol.json', launch)
            files = {}
            for row in selected:
                record = {**self.record(row), 'device_uuid': 'gpu-a'}
                for kind, payload in (('episode', record), ('trace', [])):
                    name = f"{kind}-{row['episode']:03d}.json"
                    write_json(root / name, payload); files[name] = sha256(root / name)
            report = {'status': 'droid_coupling_scientific_shard_complete', 'freeze_sha256': 'freeze',
                'protocol_sha256': sha256(root / 'protocol.json'), 'episodes': 8,
                'files_sha256': files, 'device_uuid': 'gpu-a', 'parameters_unchanged': True,
                'fresh_confirmation': False, 'robot_executions': 0}
            def bind(value):
                write_json(root / 'report.json', value)
                write_json(root / 'DONE.json', {'report_sha256': sha256(root / 'report.json')})
            bind(report)
            records, _, _ = scientific_shard(root, protocol, 'freeze', 0, 'native', 'gpu-a')
            self.assertEqual(len(records), 8)
            with self.assertRaisesRegex(ValueError, 'binding'):
                scientific_shard(root, protocol, 'freeze', 0, 'native', 'gpu-b')
            changed = copy.deepcopy(report); changed['files_sha256'].pop('episode-000.json'); bind(changed)
            with self.assertRaisesRegex(ValueError, 'binding'):
                scientific_shard(root, protocol, 'freeze', 0, 'native', 'gpu-a')
            bind(report)
            with self.assertRaisesRegex(ValueError, 'binding'):
                scientific_shard(root, {**protocol, 'source_sha256': 'different'}, 'freeze', 0, 'native', 'gpu-a')
            record_path = root / 'episode-000.json'
            changed_row = json.loads(record_path.read_text()); changed_row['device_uuid'] = 'gpu-b'
            write_json(record_path, changed_row)
            report['files_sha256'][record_path.name] = sha256(record_path); bind(report)
            with self.assertRaisesRegex(ValueError, 'mislabeled'):
                scientific_shard(root, protocol, 'freeze', 0, 'native', 'gpu-a')


if __name__ == '__main__':
    unittest.main()
