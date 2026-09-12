import importlib.util
from pathlib import Path
import sys
import unittest
import json
import signal
import tempfile
from types import SimpleNamespace
from unittest.mock import patch, Mock

OPS = Path(__file__).resolve().parents[1] / 'scripts/vast'
sys.path.insert(0, str(OPS))
import component_boundary_job as module


class BoundaryTests(unittest.TestCase):
    def test_driver_cleanup_lag_is_waited_out_not_an_immediate_failure(self):
        with patch.object(module.subprocess,'check_output',side_effect=[b'123\n',b'']),patch.object(module.time,'sleep') as sleep:
            module.wait_gpu_release()
        sleep.assert_called_once_with(1)

    def test_occupied_gpu_is_not_used_after_bounded_wait(self):
        with patch.object(module.subprocess,'check_output',return_value=b'123\n'),patch.object(module.time,'monotonic',side_effect=[0,31]):
            with self.assertRaises(ValueError):module.wait_gpu_release()

    def test_active_single_device_only(self):
        root = Path('/workspace/owned')
        args = ['python', '-u', '/ops/component_extension_queue.py', '--root', str(root), '--gpus', '1']
        module.validate_queue({'args': args, 'state': 'S'}, root)
        for state in ('T', 't', 'Z'):
            with self.assertRaises(ValueError):
                module.validate_queue({'args': args, 'state': state}, root)
        with self.assertRaises(ValueError):
            module.validate_queue({'args': args[:-1] + ['4'], 'state': 'S'}, root)
        with self.assertRaises(ValueError):
            module.validate_queue({'args': args, 'state': 'S'}, Path('/wrong'))

    def test_fit_only_bounded_contract(self):
        root = Path('/workspace/prerequisite')
        job = {'command': ['/workspace/component-python/bin/python', '-u', '-m',
            'offline_study.refined_task_fit', '--output', str(root / 'fit-v1')],
            'timeout_seconds': 3900, 'task': 'pusht', 'cwd': str(root / 'code'), 'files': {}}
        module.validate_job(job, root)
        with self.assertRaises(ValueError):
            module.validate_job({**job, 'timeout_seconds': 3901}, root)
        with self.assertRaises(ValueError):
            module.validate_job({**job, 'task': 'droid'}, root)
        with self.assertRaises(ValueError):
            module.validate_job({**job, 'cwd': '/another/project'}, root)
        with self.assertRaises(ValueError):
            module.validate_job({**job, 'files': {'../escape': {'bytes': 0}}}, root)

    def test_queue_resumes_after_fit_failure_without_signaling_scientific_child(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); component = root / 'component'; component.mkdir()
            scientific = component / 'results'; scientific.mkdir()
            (scientific / 'DONE.json').write_text('{}')
            job = {'gpu_uuid': 'GPU-test', 'command': ['fit'], 'cwd': str(root), 'timeout_seconds': 1}
            (root / 'JOB.json').write_text(json.dumps(job))
            queue = {'args': ['python', '-u', '/ops/component_extension_queue.py', '--root', str(component), '--gpus', '1'], 'state': 'S'}
            child = {'args': ['python', '-m', 'offline_study.metaworld_component_behavior', '--output', str(scientific)]}
            fit = Mock(); fit.wait.return_value = 1; fit.poll.return_value = 1
            with patch.object(module, 'validate_job'), \
                 patch.object(module, 'process', side_effect=lambda pid: queue if pid == 10 else {'state': 'Z'}), \
                 patch.object(module, 'direct_children', return_value=[(20, child)]), \
                 patch.object(module.signal, 'signal'), \
                 patch.object(module.os, 'kill') as signals, \
                 patch.object(module.subprocess, 'check_output', side_effect=['GPU-test', b'']), \
                 patch.object(module.subprocess, 'Popen', return_value=fit):
                module.main(SimpleNamespace(job=root / 'JOB.json', component_root=component, queue_pid=10))
            self.assertEqual([c.args for c in signals.call_args_list], [(10, signal.SIGSTOP), (10, signal.SIGCONT)])
            self.assertTrue((root / 'COMPONENTS_RESUMED.json').exists())
            self.assertEqual(json.loads((root / 'TERMINAL.json').read_text())['status'], 'incomplete_preserve_all')

    def test_droid_overlay_is_private_and_cannot_change_other_task_imports(self):
        root = Path('/workspace/refined-droid-prerequisite-20260911-v2')
        job = {'command': ['/workspace/component-python/bin/python', '-u', '-m',
            'offline_study.droid_fixed_response_fit', '--output', str(root / 'fit-v1')],
            'timeout_seconds': 3900, 'task': 'droid', 'cwd': str(root / 'code'),
            'files': {}, 'pythonpath_additions': [str(root / 'runtime-packages')]}
        module.validate_job(job, root)
        with self.assertRaises(ValueError):
            module.validate_job({**job, 'pythonpath_additions': ['/shared/environment']}, root)
        with self.assertRaises(ValueError):
            module.validate_job({**job, 'task': 'pusht'}, root)


if __name__ == '__main__':
    unittest.main()
