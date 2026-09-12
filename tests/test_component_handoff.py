"""Operational handoff tests: no SSH, provider, process signal or GPU calls."""
import importlib.util
import json
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('handoff', Path(__file__).resolve().parents[1] / 'scripts/vast/component_queue_handoff.py')
handoff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handoff)


class HandoffTests(unittest.TestCase):
    def setup_case(self, root, completed=True):
        (root / 'ops/handoff-v1').mkdir(parents=True)
        out = root / 'results/reach/native/shard-01'
        out.mkdir(parents=True)
        if completed:
            (out / 'DONE.json').write_text('{}')
        queue = {'args': ['python', '-u', str(root / 'ops/component_extension_queue.py'),
            '--root', str(root), '--deadline', '500'], 'state': 'S', 'parent': 1}
        child = {'args': ['python', '-m', 'offline_study.metaworld_component_behavior',
            'run', '--output', str(out), '--task', 'reach'], 'state': 'R', 'parent': 10}
        return queue, child

    def test_only_coordinator_signaled_and_complete_stream_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            queue, child = self.setup_case(root)
            def kill(pid, sig):
                self.assertEqual(pid, 10)
                if sig == signal.SIGKILL:
                    (root / 'TERMINAL.json').write_text('{"status":"operational_coordinator_exit"}')
            with patch.object(handoff, 'process', side_effect=lambda pid: queue if pid == 10 else {'state': 'Z'}), \
                 patch.object(handoff, 'direct_children', return_value=[(20, child)]), \
                 patch.object(handoff.time, 'time', return_value=100), \
                 patch.object(handoff.os, 'kill', side_effect=kill) as signals, \
                 patch.object(handoff.subprocess, 'Popen', return_value=SimpleNamespace(pid=30)) as launch:
                handoff.main(SimpleNamespace(root=root, queue_pid=10, deadline=10000))
            self.assertEqual([c.args[1] for c in signals.call_args_list], [signal.SIGSTOP, signal.SIGKILL])
            self.assertIn('--resume-verified', launch.call_args.args[0])
            self.assertFalse((root / 'TERMINAL.json').exists())
            self.assertTrue((root / 'ops/handoff-v1/PRIOR_TERMINAL.json').exists())
            result = json.loads((root / 'ops/handoff-v1/HANDOFF_DONE.json').read_text())
            self.assertFalse(result['partial_stream_reused'])

    def test_partial_child_is_not_skipped_and_original_coordinator_resumes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            queue, child = self.setup_case(root, completed=False)
            with patch.object(handoff, 'process', side_effect=lambda pid: queue if pid == 10 else {'state': 'Z'}), \
                 patch.object(handoff, 'direct_children', return_value=[(20, child)]), \
                 patch.object(handoff.time, 'time', return_value=100), \
                 patch.object(handoff.os, 'kill') as signals, \
                 patch.object(handoff.subprocess, 'Popen') as launch:
                with self.assertRaises(ValueError):
                    handoff.main(SimpleNamespace(root=root, queue_pid=10, deadline=10000))
            self.assertEqual([c.args[1] for c in signals.call_args_list], [signal.SIGSTOP, signal.SIGCONT])
            launch.assert_not_called()

    def test_other_workload_never_suspended(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(handoff, 'process', return_value={'args': ['python', '-u', 'other.py']}), \
                 patch.object(handoff.os, 'kill') as signals:
                with self.assertRaises(ValueError):
                    handoff.main(SimpleNamespace(root=root, queue_pid=10, deadline=10000))
            signals.assert_not_called()
