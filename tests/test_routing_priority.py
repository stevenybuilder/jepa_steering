"""CPU-only scheduler tests; no production processes are signaled."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

scripts = Path(__file__).resolve().parents[1] / 'scripts/vast'
if not (scripts / 'routing_priority_common.py').exists():
    scripts = Path(__file__).resolve().parent
sys.path.insert(0, str(scripts))
import routing_priority_common as c
import routing_priority_control as control


def fixture(state='S', pid=5202, starttime=123):
    return {'pid': pid, 'state': state, 'ppid': 1, 'starttime': starttime, 'command': ['python', 'original.py']}


class PriorityTests(unittest.TestCase):
    def test_proc_identity_accepts_spaces_and_parentheses(self):
        text = '5202 (python (worker)) ' + ' '.join(['S', '1'] + ['0'] * 17 + ['12345'] + ['0'] * 3)
        self.assertEqual(c.stat_fields(text), {'state': 'S', 'ppid': 1, 'starttime': 12345})

    def test_normal_exit_and_zombie_are_terminal_not_pid_reuse(self):
        original = fixture()
        self.assertFalse(c.alive(original, lambda pid: None))
        self.assertFalse(c.alive(original, lambda pid: {**original, 'state': 'Z', 'command': []}))
        self.assertTrue(c.alive(original, lambda pid: original))

    def test_pid_reuse_or_changed_command_never_authorizes_signal(self):
        original = fixture()
        for changed in ({**original, 'starttime': 124}, {**original, 'command': ['other']}):
            with self.assertRaises(ValueError):
                c.alive(original, lambda pid: changed)

    def test_new_process_waits_out_empty_exec_argv_with_same_starttime(self):
        final = fixture()
        from unittest.mock import Mock
        reader = Mock(side_effect=[{**final, 'command': []}, final])
        with patch.object(c.time, 'sleep') as sleep:
            self.assertEqual(c.started(final['pid'], final['command'], reader), final)
            sleep.assert_called_once()
        reader = Mock(side_effect=[{**final, 'command': []}, {**final, 'starttime': 999}])
        with patch.object(c.time, 'sleep'), self.assertRaises(ValueError):
            c.started(final['pid'], final['command'], reader)

    def test_proc_exit_during_snapshot_is_normal_terminal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / '5202').mkdir()
            with (root / '5202/stat').open('x') as stream:
                stream.write('5202 (python) ' + ' '.join(['S', '1'] + ['0'] * 17 + ['123']))
            self.assertIsNone(c.process(5202, root))

    def test_resume_gate_refuses_active_child_or_nonempty_gpu(self):
        parent, child = fixture('T'), fixture(pid=1234)
        reader = lambda pid: parent if pid == parent['pid'] else child
        self.assertFalse(c.safe_to_resume(parent, child, 0, reader, lambda gpu: []))
        reader = lambda pid: parent if pid == parent['pid'] else {**child, 'state': 'Z', 'command': []}
        self.assertFalse(c.safe_to_resume(parent, child, 0, reader, lambda gpu: [1234]))
        self.assertTrue(c.safe_to_resume(parent, child, 0, reader, lambda gpu: []))
        self.assertTrue(c.safe_to_resume(parent, None, 0, reader, lambda gpu: []))

    def test_failure_cleanup_waits_for_child_before_continuing_parent(self):
        parent, child = fixture('T'), fixture(pid=1234)
        with tempfile.TemporaryDirectory() as temporary, patch.object(c, 'alive', return_value=True), \
                patch.object(c, 'safe_to_resume', side_effect=[False, False, True]) as ready, \
                patch.object(c, 'process', return_value=parent), patch.object(c, 'send') as send, \
                patch.object(c.time, 'sleep') as sleep:
            c.resume_parent(parent, child, 0, Path(temporary))
            self.assertEqual(ready.call_count, 3)
            self.assertEqual(sleep.call_count, 2)
            send.assert_called_once_with(parent, c.signal.SIGCONT)
            self.assertTrue((Path(temporary) / 'PARENT_RESUMED.json').exists())

    def test_cleanup_never_resumes_while_gpu_stays_busy(self):
        parent = fixture('T')
        with tempfile.TemporaryDirectory() as temporary, patch.object(c, 'alive', return_value=True), \
                patch.object(c, 'safe_to_resume', return_value=False), patch.object(c, 'process', return_value=parent), \
                patch.object(c, 'send') as send, patch.object(c.time, 'monotonic', side_effect=[0, 2]):
            with self.assertRaises(TimeoutError):
                c.resume_parent(parent, None, 0, Path(temporary), timeout=1)
            send.assert_not_called()

    def test_nvml_prefix_normalization_is_explicit_and_strict(self):
        bare = 'd55b4747-7865-c32a-ca90-5364d5350346'
        self.assertEqual(c.normalize_nvml_uuid('GPU-' + bare), bare)
        for bad in (bare, 'GPU-other', 'MIG-' + bare):
            with self.assertRaises(ValueError):
                c.normalize_nvml_uuid(bad)

    @unittest.skipUnless(hasattr(signal, 'SIGSTOP'), 'POSIX-only supervisor lifecycle check')
    def test_completed_child_reaped_after_parent_paused_past_wait_timeout(self):
        # Only new, owned CPU test processes are signaled. Receiving research
        # Python must reap a completed child before checking an expired deadline.
        program = ('import subprocess,sys; '
            'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(.15)"]); '
            'print("ready",flush=True); '
            'result=p.wait(timeout=.4); print("completed",result,flush=True)')
        parent = subprocess.Popen([sys.executable, '-u', '-c', program],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            self.assertEqual(parent.stdout.readline().strip(), 'ready')
            binding = c.started(parent.pid, [sys.executable, '-u', '-c', program]) if sys.platform == 'linux' else None
            if binding is not None:
                self.assertTrue(c.send(binding, signal.SIGSTOP))
            else:
                os.kill(parent.pid, signal.SIGSTOP)
            time.sleep(.8)
            if binding is not None:
                self.assertTrue(c.send(binding, signal.SIGCONT))
            else:
                os.kill(parent.pid, signal.SIGCONT)
            output, errors = parent.communicate(timeout=5)
            self.assertEqual(parent.returncode, 0, errors)
            self.assertIn('completed 0', output)
        finally:
            if parent.poll() is None:
                os.kill(parent.pid, signal.SIGCONT)
                parent.terminate(); parent.wait(timeout=5)
            parent.stdout.close(); parent.stderr.close()

    def test_reference_gate_requires_only_three_complete_arms(self):
        self.assertEqual(c.REQUIRED, ('native', 'fixed_rank4', 'matched_random_fixed_rank4'))
        self.assertEqual(len(c.ORIGINAL_ARMS), 5)
        # Real loader is mocked, but its exact 24-row/device contract remains active.
        import types
        gpu = 0; ranks = [0, 4]
        with tempfile.TemporaryDirectory() as temporary:
            reference = Path(temporary)
            for arm in c.REQUIRED:
                root = reference / 'reach' / arm / 'shard-gpu0'; root.mkdir(parents=True)
                (root / 'DONE.json').touch()
            args = types.SimpleNamespace(reference=reference)
            rows = [{'device_uuid': 'device'} for _ in range(24)]
            fake = types.ModuleType('offline_study.routing_behavior')
            from unittest.mock import Mock
            fake.reference_records = Mock(return_value=(rows, {'source/report.json': 'sha'}))
            with patch.dict(sys.modules, {'offline_study.routing_behavior': fake}), patch.object(c, 'gpu_uuid', return_value='device'):
                self.assertEqual(c.reference_gate(gpu, args, {}), {'source/report.json': 'sha'})
                self.assertEqual(fake.reference_records.call_count, 3)
                for call in fake.reference_records.call_args_list:
                    self.assertEqual(call.args[-1], ranks)
            (reference / 'reach/native/shard-gpu0/DONE.json').unlink()
            with patch.dict(sys.modules, {'offline_study.routing_behavior': fake}):
                self.assertIsNone(c.reference_gate(gpu, args, {}))


if __name__ == '__main__':
    unittest.main()
