"""Seed236 is an independent initialization and a GPU1-only queued workload."""
import importlib.util
import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

path = Path(__file__).resolve().parents[1] / 'scripts/vast/pointmaze_seed236_worker.py'
spec = importlib.util.spec_from_file_location('pointmaze_seed236_worker', path)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class PointMaze236QueueTests(unittest.TestCase):
    def test_new_seed_engineering_never_inherits_seed234_checkpoint(self):
        vendor = Path('/vendor')
        pilot = Path('/original-pilot')
        engineering, remaining = worker.training_arguments(vendor, pilot)
        self.assertIn('--engineering-only', engineering)
        self.assertNotIn('--resume-from', engineering)
        self.assertNotIn('--engineering-proof', engineering)
        for arguments in (engineering, remaining):
            self.assertEqual(arguments[arguments.index('--seed') + 1], '236')
            self.assertEqual(arguments[arguments.index('--pilot') + 1], str(pilot))
            self.assertFalse(any('seed-234' in value for value in arguments))
        self.assertTrue(remaining[remaining.index('--resume-from') + 1].endswith('seed-236/epoch-one-engineering/jepa-e0.pth.tar'))
        self.assertTrue(remaining[remaining.index('--engineering-proof') + 1].endswith('seed-236/epoch-one-engineering'))
        self.assertTrue(remaining[remaining.index('--output') + 1].endswith('seed-236/remaining-epochs'))

    def test_full_gpu1_panel_required_without_episode_inflation(self):
        done = {'completed_paired_streams': [1, 5], 'arms': worker.ARMS, 'episodes': 144}
        launch = {'pid': 1297, 'instance': 50259194, 'device_uuid': worker.UUID, 'logical_ranks': [1, 5],
            'episodes_per_arm_global': 64, 'episodes_per_arm_on_device': 16, 'source_sha256': worker.DROID_SOURCE_SHA}
        worker.verify_predecessor(done, launch)
        for key, value in [('completed_paired_streams', [1]), ('arms', ['native']), ('episodes', 576)]:
            with self.assertRaises(ValueError):
                worker.verify_predecessor({**done, key: value}, launch)
        for key, value in [('pid', 1296), ('device_uuid', 'wrong'), ('logical_ranks', [0, 4]),
                           ('episodes_per_arm_global', 256), ('episodes_per_arm_on_device', 64)]:
            with self.assertRaises(ValueError):
                worker.verify_predecessor(done, {**launch, key: value})

    def test_predecessor_requires_exact_gpu1_argv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = root / 'cmdline'
            command.write_bytes(b'python\0/workspace/jepa-runtime/droid-coupling-code-20260908-v3/run_droid_parallel_queue.py\0--gpu\x001\0')
            self.assertTrue(worker.predecessor_pending(root, command))
            for content in (b'', b'python\0/workspace/jepa-runtime/droid-coupling-code-20260908-v3/run_droid_parallel_queue.py\0--gpu\x000\0'):
                command.write_bytes(content)
                with self.assertRaises(ValueError):
                    worker.predecessor_pending(root, command)

    def test_done_process_exit_race_is_not_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = root / 'cmdline'
            with self.assertRaises(ValueError):
                worker.predecessor_pending(root, command)
            def racing_read(path):
                (root / 'DONE.json').write_text('{}')
                raise FileNotFoundError('Process exited after completion')
            with patch.object(Path, 'read_bytes', racing_read):
                self.assertFalse(worker.predecessor_pending(root, command))

    def test_failed_or_contradictory_predecessor_cannot_advance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'FAILED.json').write_text('{}')
            for completed in (False, True):
                if completed:
                    (root / 'DONE.json').write_text('{}')
                with self.assertRaises(ValueError):
                    worker.predecessor_pending(root, root / 'missing-command')

    def test_torch_record_uuid_is_raw_and_nvidia_guard_keeps_prefix(self):
        self.assertEqual(worker.UUID, 'GPU-' + worker.RECORD_UUID)
        self.assertEqual(worker.RECORD_UUID, '732bb4c8-5057-668e-1126-c3cd87e6cbf9')
        self.assertEqual(worker.RANKS, [1, 5])
        self.assertEqual(worker.PID, 1297)

    def test_embedded_remote_python_programs_compile(self):
        # This file is present locally; receiving payload omits the local activator.
        paths = [path]
        activate = path.with_name('pointmaze_seed236_activate.py')
        if activate.exists():
            paths.append(activate)
        count = 0
        for source in paths:
            for node in ast.walk(ast.parse(source.read_text())):
                if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and
                        isinstance(node.value.value, str) and any(isinstance(t, ast.Name) and t.id in
                            ('validation', 'stage', 'launch') for t in node.targets)):
                    compile(node.value.value, str(source), 'exec')
                    count += 1
        self.assertGreaterEqual(count, 1)


if __name__ == '__main__':
    unittest.main()
