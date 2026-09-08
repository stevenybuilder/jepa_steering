"""Fail-closed CPU tests for the owned-worker history migration."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


def load(name):
    path = Path(__file__).resolve().parents[1] / 'scripts/vast' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


worker = load('pointmaze_tx_resume_worker')
stage = load('pointmaze_tx_resume_stage')


class PointMazeTXResumeTests(unittest.TestCase):
    def test_manifest_scope_rejects_path_escape(self):
        wanted = {'bytes': 3, 'sha256': 'abc'}
        stage.validate_manifest({'workspace/jepa-python/a.py': wanted}, ['workspace/jepa-python'])
        for path in ('/workspace/jepa-python/a.py', 'workspace/jepa-python/../.env',
                     'workspace/jepa-droid-python/a.py', 'root/.ssh/id_ed25519'):
            with self.assertRaises(ValueError):
                stage.validate_manifest({path: wanted}, ['workspace/jepa-python'])

    def test_receiving_checks_actual_bytes_and_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file = root / 'one'
            file.write_bytes(b'abc')
            manifest = {'one': {'bytes': 3, 'sha256': worker.digest(file)}}
            worker.verify_members(root, manifest)
            file.write_bytes(b'abd')
            with self.assertRaises(ValueError):
                worker.verify_members(root, manifest)
            (root / 'link').symlink_to('one')
            worker.verify_members(root, {'link': {'symlink': 'one'}})
            with self.assertRaises(ValueError):
                worker.verify_members(root, {'link': {'symlink': 'elsewhere'}})

    def test_checkpoint_requires_exact_epoch3_cursors_and_original_binding(self):
        binding = {'task': 'pointmaze', 'seed': 234, 'pilot_report_sha256': 'original'}
        data = {'epoch': 3, 'study_resume': {'epoch': 3, 'binding': binding,
            'epoch_boundary_only': True, 'scheduler_step': 3417, 'wd_step': 3417,
            'validation_events': 15, 'cpu_rngs': list(range(16)), 'cuda_rngs': list(range(16)),
            'checkpoint_history': [{'epoch': 1}, {'epoch': 2}]}}
        worker.verify_state_metadata(data, binding)
        for key, value in [('epoch', 4), ('scheduler_step', 1254), ('wd_step', 3418),
                           ('validation_events', 6), ('cpu_rngs', [1]), ('cuda_rngs', [1]),
                           ('epoch_boundary_only', False), ('checkpoint_history', [{'epoch': 2}])]:
            bad = copy.deepcopy(data)
            bad['study_resume'][key] = value
            with self.assertRaises(ValueError):
                worker.verify_state_metadata(bad, binding)
        with self.assertRaises(ValueError):
            worker.verify_state_metadata(data, {**binding, 'pilot_report_sha256': 'new_receiving_pilot'})

    def test_handoff_requires_entire_assigned_droid_streams_not_one_arm(self):
        arms = ['native', 'visual_only', 'action_condition_only', 'joint', 'joint_equal_standardized_energy',
                'permuted_visual', 'permuted_joint', 'matched_random', 'matched_random_equal_standardized_energy']
        done = {'completed_paired_streams': [0, 4], 'arms': arms, 'episodes': 144}
        launch = {'pid': 1296, 'instance': 50259194, 'device_uuid': worker.UUID, 'logical_ranks': [0, 4],
            'episodes_per_arm_global': 64, 'episodes_per_arm_on_device': 16,
            'source_sha256': '249a8cdb9326cd79f3b9180a830e409cdd0ffa4f2f3be2ac393c32e30ab81a54'}
        worker.verify_queue_done(done, launch)
        for key, value in [('completed_paired_streams', [0]), ('arms', ['native']), ('episodes', 16)]:
            with self.assertRaises(ValueError):
                worker.verify_queue_done({**done, key: value}, launch)
        for key, value in [('pid', 123), ('instance', 123), ('device_uuid', 'wrong'),
                           ('logical_ranks', [0, 1]), ('episodes_per_arm_global', 256)]:
            with self.assertRaises(ValueError):
                worker.verify_queue_done(done, {**launch, key: value})

    def test_task_environment_keeps_runtime_isolated_and_cuda_hidden_by_default(self):
        env = worker.environment()
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
        self.assertIn('python3.10/site-packages', env['PYTHONPATH'])
        self.assertNotIn('droid-coupling-code', env['PYTHONPATH'])
        self.assertEqual(env['LD_PRELOAD'], worker.DRIVER)
        self.assertEqual(worker.environment('0')['CUDA_VISIBLE_DEVICES'], '0')

    def test_nvidia_device_label_and_droid_record_uuid_are_distinct(self):
        self.assertEqual(worker.UUID, 'GPU-' + worker.DROID_RECORD_UUID)
        self.assertEqual(worker.DROID_RECORD_UUID, '99bc03dd-d8ee-d7c3-fb46-5771e42b95a6')

    def test_normalized_record_uuid_passes_real_droid_shard_verifier(self):
        from offline_study.droid_coupling_behavior import scientific_shard
        from offline_study.planning_contract import seed_schedule
        from test_droid_coupling_behavior import DroidBehaviorTests
        episodes = seed_schedule(1, 64, 8, 3)
        selected = [row for row in episodes if row['logical_rank'] == 0]
        protocol = {'planning': {'episodes': episodes}, 'source_sha256': 'fixture'}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker.write(root / 'protocol.json', {'freeze_sha256': 'freeze', 'source_sha256': 'fixture',
                'role': 'scientific_development', 'arm': 'native', 'logical_ranks': [0],
                'device_uuid': worker.DROID_RECORD_UUID, 'engineering_report_sha256': 'proof'})
            files = {}
            for row in selected:
                record = {**DroidBehaviorTests().record(row), 'device_uuid': worker.DROID_RECORD_UUID}
                for kind, payload in [('episode', record), ('trace', [])]:
                    name = f"{kind}-{row['episode']:03d}.json"
                    worker.write(root / name, payload)
                    files[name] = worker.digest(root / name)
            worker.write(root / 'report.json', {'status': 'droid_coupling_scientific_shard_complete',
                'freeze_sha256': 'freeze', 'protocol_sha256': worker.digest(root / 'protocol.json'),
                'episodes': 8, 'files_sha256': files, 'device_uuid': worker.DROID_RECORD_UUID,
                'parameters_unchanged': True, 'fresh_confirmation': False, 'robot_executions': 0})
            worker.write(root / 'DONE.json', {'report_sha256': worker.digest(root / 'report.json')})
            records, _, _ = scientific_shard(root, protocol, 'freeze', 0, 'native', worker.DROID_RECORD_UUID)
            self.assertEqual(len(records), 8)
            with self.assertRaises(ValueError):
                scientific_shard(root, protocol, 'freeze', 0, 'native', worker.UUID)

    def test_done_wins_over_process_exit_race_but_incomplete_exit_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = Path(directory)
            command = previous / 'cmdline'
            with self.assertRaises(ValueError):
                worker.predecessor_pending(previous, command)
            command.write_bytes(b'python\0/workspace/jepa-runtime/droid-coupling-code-20260908-v3/run_droid_parallel_queue.py\0--gpu\x000\0')
            self.assertTrue(worker.predecessor_pending(previous, command))
            def racing_read(path):
                worker.write(previous / 'DONE.json', {'complete': True})
                raise FileNotFoundError('Process exited after DONE')
            with patch.object(Path, 'read_bytes', racing_read):
                self.assertFalse(worker.predecessor_pending(previous, command))
            self.assertFalse(worker.predecessor_pending(previous, command))

    def test_predecessor_rejects_wrong_gpu_or_zombie_without_done(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = Path(directory)
            command = previous / 'cmdline'
            for content in (b'', b'python\0/workspace/jepa-runtime/droid-coupling-code-20260908-v3/run_droid_parallel_queue.py\0--gpu\x001\0'):
                command.write_bytes(content)
                with self.assertRaises(ValueError):
                    worker.predecessor_pending(previous, command)

    def test_lifecycle_receipts_cannot_overwrite_existing_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'READY.json'
            worker.write(path, {'attempt': 1})
            with self.assertRaises(FileExistsError):
                worker.write(path, {'attempt': 2})
            self.assertEqual(worker.read(path), {'attempt': 1})


if __name__ == '__main__':
    unittest.main()
