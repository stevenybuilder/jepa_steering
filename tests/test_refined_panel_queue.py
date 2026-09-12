"""Operational coverage tests; not model or behavioral efficacy results."""
import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts/vast'))
from refined_panel_queue import jobs
from component_boundary_job import validate_job
from stage_remaining_refined_behavior import navigation_reference_source


class PanelQueueTests(unittest.TestCase):
    def test_historical_navigation_source_is_nested_and_matches_its_receipt(self):
        base=Path(__file__).parents[1]/'artifacts/offline_study/navigation-coupling-reference-20260908-v1'
        for task in ('wall','pointmaze'):
            if not (base/task).exists():continue
            source=navigation_reference_source(base/task)
            self.assertEqual(source,base/task/'baseline-source/src/offline_study')

    def test_droid_uses_its_own_adapter_and_eight_original_streams(self):
        spec = {k: '/input/' + k for k in ('vendor','fit','assets','manifest','reference','native-engineering','encoder-source','encoder-root')}
        spec.update(task='droid', logical_ranks=list(range(8)))
        result = jobs(Path('/panel'), spec)
        self.assertEqual(len(result), 25)
        self.assertTrue(all(x[2][3] == 'offline_study.droid_fixed_response_behavior' for x in result))
        self.assertTrue(all('--checkpoint' not in x[2] for x in result))

    def test_pointmaze_interpreter_is_isolated_to_its_panel(self):
        spec = {k: '/input/' + k for k in ('vendor','checkpoint','fit','cohort','reference','reference-source','data-root')}
        spec.update(task='pointmaze', logical_ranks=list(range(8)), python='/panel/runtime/python/bin/python')
        self.assertEqual(jobs(Path('/panel'), spec)[0][2][0], spec['python'])
        with self.assertRaises(ValueError): jobs(Path('/panel'), {**spec, 'python':'/foreign/bin/python'})

    def test_all96_original_scenarios_each_arm_without_padding(self):
        spec = {k: '/input/' + k for k in ('vendor','checkpoint','fit','cohort','reference','reference-source','data-root')}
        spec.update(task='pusht', logical_ranks=list(range(8)))
        result = jobs(Path('/panel'), spec)
        self.assertEqual(len(result), 25)
        for rank in range(8):
            group = result[1+3*rank:4+3*rank]
            self.assertEqual([x[0] for x in group], [f'{a}-{rank}' for a in ('native','fixed_rank4','matched_random_fixed_rank4')])
            for _, _, command in group:
                self.assertEqual(command[command.index('--logical-ranks')+1], str(rank))
            for _, _, command in group[1:]:
                self.assertEqual(command[command.index('--native')+1], f'/panel/conditions/native/shard-{rank}')
        with self.assertRaises(ValueError): jobs(Path('/panel'), {**spec, 'logical_ranks': list(range(9))})

    def test_boundary_accepts_only_exact_owned_operational_command(self):
        root = Path('/panel')
        spec = {'kind':'refined_panel','task':'pusht','timeout_seconds':21600,'cwd':'/panel/code',
            'files':{},'command':['/workspace/component-python/bin/python','-u','/panel/ops/refined_panel_queue.py','--root','/panel']}
        validate_job(spec, root)
        with self.assertRaises(ValueError): validate_job({**spec,'command':spec['command']+['--skip-validation']},root)
        with self.assertRaises(ValueError): validate_job({**spec,'timeout_seconds':21601},root)


if __name__ == '__main__': unittest.main()
