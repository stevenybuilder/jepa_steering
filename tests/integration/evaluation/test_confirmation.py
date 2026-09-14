import copy
import unittest
from pathlib import Path

from offline_study.evaluation.confirmation import SCENARIOS, ReservedGoalBank
from offline_study.evaluation.fixed_response_behavior import ARMS, TASKS
from offline_study.evaluation.fixed_response_behavior_analysis import analyze, validate_panel
from offline_study.evaluation.behavioral_development import schedule


def fixture(rows):
    return {task: {arm: [{**row, 'arm': arm, 'seconds': 301.,
        'initial_state_vector': [row['episode']], 'result': {'initial_sha256': str(row['episode']),
            'goal_sha256': str(row['episode']), 'elementary_steps': 100,
            'native_success': (row['episode'] % (2+ARMS.index(arm)) == 0),
            'native_state_distance': .2, 'native_reward': 0.}} for row in rows] for arm in ARMS} for task in TASKS}


class ConfirmationTests(unittest.TestCase):
    def test_exact_960_no_padding_or_development_overlap(self):
        self.assertEqual(len(SCENARIOS)*len(TASKS)*len(ARMS), 960)
        self.assertEqual(len({r['environment_seed'] for r in SCENARIOS}), 96)
        self.assertFalse({r['environment_seed'] for r in SCENARIOS} & {r['environment_seed'] for r in schedule()})
        for rank in range(8):
            self.assertEqual(sum(r['logical_rank'] == rank for r in SCENARIOS), 12)

    def test_development_and_confirmation_cannot_be_interchanged(self):
        panel = fixture(SCENARIOS)
        validate_panel(panel, SCENARIOS)
        with self.assertRaises(ValueError):
            validate_panel(panel)
        with self.assertRaises(ValueError):
            validate_panel(fixture(schedule()), SCENARIOS)

    def test_paired_completeness_and_endpoint(self):
        for mutation in ('missing', 'seed', 'goal', 'short', 'arm'):
            panel = fixture(SCENARIOS)
            row = panel['reach']['fixed_rank4'][0]
            if mutation == 'missing': panel['reach']['fixed_rank4'].pop()
            if mutation == 'seed': row['local_seed'] += 1
            if mutation == 'goal': row['result']['goal_sha256'] = 'changed'
            if mutation == 'short': row['result']['elementary_steps'] = 15
            if mutation == 'arm': del panel['reach']['coupling_only']
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_panel(panel, SCENARIOS)

    def test_same_frozen_statistics_no_test_switch(self):
        old, new = analyze(fixture(schedule())), analyze(fixture(SCENARIOS), SCENARIOS)
        self.assertEqual(old['contrasts'], new['contrasts'])
        self.assertEqual(len(new['contrasts']), 8)

    def test_real_reserved_inputs_all_hashes_and_schedule(self):
        root = Path(__file__).resolve().parents[3]/'artifacts/offline_study/primary-durable-20260907/planning-scenarios-20260907'
        if not root.is_dir():
            self.skipTest('Original private input tensors are not in a clean git clone')
        for task in TASKS:
            bank = ReservedGoalBank(root/(task+'-v1'), task)
            for row in SCENARIOS:
                metadata, tensors = bank.load(row)
                self.assertEqual(metadata['environment_seed'], row['environment_seed'])
            with self.assertRaises(ValueError):
                bank.load(schedule()[0])


if __name__ == '__main__':
    unittest.main()
