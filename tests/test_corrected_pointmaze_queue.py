"""CPU-only safety checks for the corrected three-seed fresh-start queue."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / 'scripts/vast/corrected_pointmaze_queue.py'
SPEC = importlib.util.spec_from_file_location('corrected_pointmaze_queue', SOURCE)
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


def plan_fixture():
    plan = {'schema': 1, 'assignments': [], 'output_root': '/runtime/pointmaze-corrected-v1',
        'source_root': '/runtime/corrected-code', 'source_sha256': 'a' * 64,
        'vendor': '/vendor', 'python': '/runtime/python', 'overlay': '/runtime/overlay',
        'driver': '/runtime/libcuda.so', 'driver_sha256': 'b' * 64,
        'assets': '/runtime/assets', 'input_check': '/runtime/input-check',
        'input_receipt': '/runtime/input-receipt', 'data_root': '/runtime/data',
        'fixed_files': {'/runtime/corrected-code/tests/test_native_training_sampler.py': 'c' * 64},
        'execution_approval': {'path': '/runtime/approval.json', 'sha256': 'd' * 64}}
    for gpu, seed in enumerate(queue.SEEDS):
        plan['assignments'].append({'seed': seed, 'instance': 50259194, 'gpu': gpu,
            'gpu_uuid': 'GPU-test-' + str(gpu), 'predecessor': {'pid': 1000 + gpu,
                'start_ticks': 12345 + gpu, 'argv': ['python', '/nav.py', '--worker', 'tx' + str(gpu)],
                'done': '/nav/worker' + str(gpu) + '/DONE.json', 'failed': '/nav/worker' + str(gpu) + '/FAILED.json',
                'launch': '/nav/worker' + str(gpu) + '/LAUNCH.json', 'launch_sha256': 'e' * 64,
                'expected_done': {'status': 'navigation_redistribution_assignment_complete', 'endpoints': 96},
                'verifier': {'python': '/python', 'script': '/verify.py', 'script_sha256': 'f' * 64,
                    'args': ['verify', '--worker', 'tx' + str(gpu)], 'expected_result': {'status': 'verified', 'endpoints': 96}}}})
    return plan


def publish_report(root, report, protocol):
    root.mkdir(parents=True)
    (root / 'protocol.json').write_text(json.dumps(protocol))
    report['protocol_sha256'] = queue.digest(root / 'protocol.json')
    (root / 'report.json').write_text(json.dumps(report))
    (root / 'DONE.json').write_text(json.dumps({'report_sha256': queue.digest(root / 'report.json')}))


class CorrectedPointMazeQueueTests(unittest.TestCase):
    def test_all_three_seeds_start_fresh_then_resume_only_their_own_epoch(self):
        plan = plan_fixture()
        for seed in queue.SEEDS:
            self.assertEqual(queue.validate_plan(plan, seed)['seed'], seed)
            output = Path(plan['output_root']) / ('seed-' + str(seed))
            engineering, remaining = queue.training_arguments(plan, seed, output)
            self.assertIn('--engineering-only', engineering)
            self.assertNotIn('--resume-from', engineering)
            self.assertNotIn('--engineering-proof', engineering)
            for arguments in (engineering, remaining):
                self.assertEqual(arguments[arguments.index('--seed') + 1], str(seed))
                self.assertEqual(arguments[arguments.index('--pilot') + 1], str(output / 'receiving-pilot'))
            self.assertEqual(remaining[remaining.index('--resume-from') + 1], str(output / 'epoch-one-engineering/jepa-e0.pth.tar'))
            self.assertEqual(remaining[remaining.index('--engineering-proof') + 1], str(output / 'epoch-one-engineering'))

    def test_reject_missing_seed_device_collision_and_unbound_sampler_test(self):
        for mutate in (
            lambda p: p['assignments'].pop(),
            lambda p: p['assignments'][1].update(seed=234),
            lambda p: p['assignments'][1].update(gpu=0),
            lambda p: p['assignments'][0]['predecessor'].update(expected_done={}),
            lambda p: p.update(fixed_files={}),
            lambda p: p.update(output_root='/runtime/old-history'),
        ):
            plan = plan_fixture()
            mutate(plan)
            with self.assertRaises(ValueError):
                queue.validate_plan(plan, 234)

    def test_cpu_environment_hides_gpus_and_gpu_environment_selects_only_owner(self):
        plan = plan_fixture()
        self.assertEqual(queue.environment(plan)['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(queue.environment(plan, 2)['CUDA_VISIBLE_DEVICES'], '2')
        self.assertEqual(queue.environment(plan)['LD_PRELOAD'], plan['driver'])
        self.assertTrue(queue.environment(plan)['PYTHONPATH'].startswith('/runtime/corrected-code/src:'))

    def test_predecessor_exit_race_rechecks_done_and_never_bypasses_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            binding = plan_fixture()['assignments'][0]['predecessor']
            binding.update(done=str(root / 'DONE.json'), failed=str(root / 'FAILED.json'))
            def exits_after_done(*args):
                (root / 'DONE.json').write_text('{}')
                return None
            with patch.object(queue, 'process_identity', side_effect=exits_after_done):
                self.assertFalse(queue.predecessor_pending(binding))
            (root / 'FAILED.json').write_text('{}')
            with self.assertRaises(ValueError):
                queue.predecessor_pending(binding)

    def test_exact_predecessor_identity_required_not_just_a_live_pid(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            binding = plan_fixture()['assignments'][0]['predecessor']
            binding.update(done=str(root / 'DONE.json'), failed=str(root / 'FAILED.json'))
            identity = {'state': 'S', 'start_ticks': binding['start_ticks'], 'argv': binding['argv']}
            with patch.object(queue, 'process_identity', return_value=identity):
                self.assertTrue(queue.predecessor_pending(binding))
            for bad in (None, {**identity, 'state': 'Z'}, {**identity, 'start_ticks': 9}, {**identity, 'argv': ['unrelated']}):
                with patch.object(queue, 'process_identity', return_value=bad), self.assertRaises(ValueError):
                    queue.predecessor_pending(binding)

    def test_proc_parser_handles_spaces_and_parentheses_in_comm(self):
        with tempfile.TemporaryDirectory() as temp:
            proc = Path(temp)
            task = proc / '1000'
            task.mkdir()
            fields = ['S'] + ['0'] * 18 + ['54321'] + ['0'] * 10
            (task / 'stat').write_text('1000 (python (worker)) ' + ' '.join(fields))
            (task / 'cmdline').write_bytes(b'python\0/nav.py\0--worker\0tx0\0')
            self.assertEqual(queue.process_identity(1000, proc), {'state': 'S', 'start_ticks': 54321,
                'argv': ['python', '/nav.py', '--worker', 'tx0']})
            self.assertIsNone(queue.process_identity(1001, proc))

    def test_owned_device_identity_and_empty_compute_list_both_required(self):
        assignment = plan_fixture()['assignments'][0]
        with patch.object(queue.subprocess, 'check_output', side_effect=[assignment['gpu_uuid'] + '\n', '']):
            self.assertTrue(queue.gpu_empty(assignment))
        with patch.object(queue.subprocess, 'check_output', side_effect=[assignment['gpu_uuid'] + '\n', '123\n']):
            self.assertFalse(queue.gpu_empty(assignment))
        with patch.object(queue.subprocess, 'check_output', return_value='GPU-somebody-else\n'), self.assertRaises(ValueError):
            queue.gpu_empty(assignment)

    def test_old_pilot_sampler_is_rejected_even_when_report_hashes_are_valid(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'pilot'
            old = {**queue.POLICY, 'train_shuffle': True}
            publish_report(root, {'status': 'native_training_accumulation_pilot_passed', 'task': 'pointmaze',
                'updates_complete': 5, 'updates_per_epoch': 1139, 'training_clips': 145800,
                'sampler_policy': old}, {'sampler_policy': old})
            with self.assertRaisesRegex(ValueError, 'Receiving numerical pilot'):
                queue.verify_pilot(root)

    def test_seed_specific_first_epoch_rejects_different_seed_and_historical_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            pilot = base / 'pilot'
            pilot.mkdir()
            (pilot / 'report.json').write_text('{}')
            for seed, resume in ((234, None), (236, '/historical/seed-236/jepa-e2.pth.tar')):
                root = base / ('engineering-' + str(seed))
                publish_report(root, {'status': 'one_epoch_pointmaze_training_engineering_complete',
                    'seed': seed, 'checkpoints': 1, 'validation_events': 5},
                    {'seed': seed, 'sampler_policy': queue.POLICY, 'resume_checkpoint': resume,
                     'pilot_report_sha256': queue.digest(pilot / 'report.json')})
                with self.assertRaises(ValueError):
                    queue.verify_engineering(root, 236, pilot)

    def test_receipts_are_immutable(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'DONE.json'
            queue.write(path, {'first': True})
            with self.assertRaises(FileExistsError):
                queue.write(path, {'first': False})
            self.assertEqual(queue.read(path), {'first': True})

    def test_completion_requires_full_verified_assignment_not_any_done_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            binding = copy.deepcopy(plan_fixture()['assignments'][0]['predecessor'])
            binding.update(done=str(root / 'DONE.json'), failed=str(root / 'FAILED.json'), launch=str(root / 'LAUNCH.json'))
            (root / 'LAUNCH.json').write_text('{}')
            binding['launch_sha256'] = queue.digest(root / 'LAUNCH.json')
            (root / 'DONE.json').write_text(json.dumps({'status': 'navigation_redistribution_assignment_complete', 'endpoints': 12}))
            with patch.object(queue.subprocess, 'check_output') as called, self.assertRaises(ValueError):
                queue.verify_predecessor(binding, plan_fixture())
            called.assert_not_called()


if __name__ == '__main__':
    unittest.main()
