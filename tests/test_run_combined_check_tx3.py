import copy
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

PATH = Path(__file__).resolve().parents[1]/'scripts/vast/run_combined_check_tx3.py'
spec = importlib.util.spec_from_file_location('combined_launcher', PATH)
q = importlib.util.module_from_spec(spec); spec.loader.exec_module(q)


class LaunchTests(unittest.TestCase):
    def authority(self):
        return {'owner': 'rep_geometry_transcoder/root', 'instance': 50259194, 'gpu': 3,
            'gpu_uuid': q.UUID, 'source_sha256': q.SCIENCE, 'script_sha256': 'a'*64,
            'receiving_tests_sha256': 'b'*64, 'hourly_cap_usd': 7, 'outer_seconds': 1210,
            'operation': 'one_combined_numerical_check_after_tx3', 'automatic_retry': False,
            'behavioral_launch_authorized': False, 'issued_unix': 1000,
            'expires_unix': 6000, 'observed_account_hourly_usd': 6.962593}

    def test_exact_authority_budget_time_and_source(self):
        q.validate_authority(self.authority(), 'a'*64, 'b'*64, now=1100)
        for key, bad in [('gpu', 0), ('instance', 50205763), ('hourly_cap_usd', 10),
                         ('source_sha256','c'*64), ('observed_account_hourly_usd', 7.01),
                         ('expires_unix', 2310), ('issued_unix', 1200),
                         ('expires_unix', float('nan')), ('automatic_retry', True),
                         ('behavioral_launch_authorized', True)]:
            with self.subTest(key=key):
                value = self.authority(); value[key] = bad
                with self.assertRaises(ValueError): q.validate_authority(value,'a'*64,'b'*64,now=1100)

    def test_one_fixed_receiving_command_no_arms_or_sample_override(self):
        command = q.command()
        self.assertEqual(command[3], 'offline_study.fixed_combined_check')
        self.assertEqual(command[command.index('--task')+1], 'mw-reach')
        self.assertEqual(command[-1], str(q.ROOT/'check'))
        for flag in ('--episodes','--rank','--arm','--retry'): self.assertNotIn(flag, command)

    def fake(self, code=0):
        child = Mock(pid=123, returncode=code)
        child.poll.return_value = code
        child.wait.return_value = code
        c = types.SimpleNamespace(write=Mock(), started=Mock(return_value={'pid':123}))
        return child,c

    def test_success_has_one_child_and_no_signals(self):
        child,c = self.fake()
        with tempfile.TemporaryDirectory() as temp, patch.object(q.subprocess,'Popen',return_value=child) as popen:
            self.assertEqual(q.execute(c, {'CUDA_VISIBLE_DEVICES':'3'},Path(temp)), {'pid':123})
            popen.assert_called_once(); child.terminate.assert_not_called(); child.kill.assert_not_called()
            self.assertLessEqual(child.wait.call_args.kwargs['timeout'],1210)

    def test_nonzero_is_terminal_never_restarted(self):
        child,c = self.fake(1)
        with tempfile.TemporaryDirectory() as temp, patch.object(q.subprocess,'Popen',return_value=child) as popen:
            with self.assertRaisesRegex(ValueError,'no automatic retry'): q.execute(c,{},Path(temp))
            popen.assert_called_once(); child.terminate.assert_not_called()

    def test_timeout_cleans_only_held_child_before_returning(self):
        child,c = self.fake(); child.poll.return_value=None
        child.wait.side_effect=[subprocess.TimeoutExpired('fixture',1210),0]
        with tempfile.TemporaryDirectory() as temp, patch.object(q.subprocess,'Popen',return_value=child):
            with self.assertRaises(subprocess.TimeoutExpired): q.execute(c,{},Path(temp))
            child.terminate.assert_called_once(); child.kill.assert_not_called()
            self.assertEqual(c.write.call_args.args[1]['predecessor_signals'],0)

    def test_identity_capture_failure_retains_spawn_and_cleans_held_child(self):
        child,c = self.fake(); child.poll.return_value=None
        c.started.side_effect=RuntimeError('observation failure')
        with tempfile.TemporaryDirectory() as temp, patch.object(q.subprocess,'Popen',return_value=child):
            with self.assertRaisesRegex(RuntimeError,'observation failure'): q.execute(c,{},Path(temp))
            child.terminate.assert_called_once()
            self.assertEqual(c.write.call_args_list[0].args[0].name,'SPAWNED.json')

    def test_source_names_and_bytes_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/'a.py').write_text('x=1')
            before=q.source_digest(root); (root/'b.py').write_text('x=1')
            self.assertNotEqual(before,q.source_digest(root))


if __name__=='__main__':unittest.main()
