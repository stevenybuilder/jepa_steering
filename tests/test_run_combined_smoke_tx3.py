import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

PATH = Path(__file__).resolve().parents[1]/'scripts/vast/run_combined_smoke_tx3.py'
with patch.object(sys, 'path', [str(PATH.parent), *sys.path]):
    spec = importlib.util.spec_from_file_location('combined_smoke_launcher', PATH)
    q = importlib.util.module_from_spec(spec); spec.loader.exec_module(q)


class FullPlannerLaunchTests(unittest.TestCase):
    def authority(self):
        return {'owner': 'rep_geometry_transcoder/root', 'instance': 50259194, 'gpu': 3,
            'gpu_uuid': q.prior.UUID, 'source_sha256': q.SCIENCE, 'script_sha256': 'a'*64,
            'receiving_tests_sha256': 'b'*64, 'hourly_cap_usd': 7, 'outer_seconds': 3610,
            'operation': 'four_combined_full_planner_engineering_episodes', 'automatic_retry': False,
            'behavioral_launch_authorized': False, 'numerical_report_sha256': q.REPORT,
            'issued_unix': 1000, 'expires_unix': 6000, 'observed_account_hourly_usd': 6.962593}

    def test_exact_source_predecessor_budget_and_deadline(self):
        q.validate_authority(self.authority(), 'a'*64, 'b'*64, now=1100)
        for key, value in [('gpu', 0), ('numerical_report_sha256', 'c'*64),
                ('source_sha256', 'c'*64), ('hourly_cap_usd', 10), ('expires_unix', 4700),
                ('automatic_retry', True), ('behavioral_launch_authorized', True),
                ('observed_account_hourly_usd', 7.01), ('issued_unix', float('nan'))]:
            authority = self.authority(); authority[key] = value
            with self.assertRaises(ValueError): q.validate_authority(authority, 'a'*64, 'b'*64, now=1100)

    def test_single_frozen_command_no_episode_or_candidate_override(self):
        cmd = q.command()
        self.assertEqual(cmd[3], 'offline_study.fixed_combined_smoke')
        self.assertEqual(cmd[cmd.index('--numerical-report-sha256')+1], q.REPORT)
        self.assertEqual(cmd[-1], str(q.ROOT/'check'))
        for flag in ('--episodes', '--rank', '--arm', '--retry', '--candidates'): self.assertNotIn(flag, cmd)

    def test_explicit_metaworld_egl_on_owned_device_with_verified_host_driver(self):
        c = types.SimpleNamespace(environment=Mock(return_value={
            'CUDA_VISIBLE_DEVICES': '3', 'LD_PRELOAD': 'verified_driver', 'MUJOCO_PY_FORCE_CPU': '1'}))
        with patch.object(q.prior, 'sha', side_effect=lambda p: q.GL_LIBRARIES[p.name]):
            env = q.child_environment(c)
        c.environment.assert_called_once_with(50259194, 3)
        self.assertEqual(env['MUJOCO_GL'], 'egl')
        self.assertEqual(env['PYOPENGL_PLATFORM'], 'egl')
        self.assertEqual(env['MUJOCO_EGL_DEVICE_ID'], '3')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '3')
        self.assertEqual(env['LD_PRELOAD'], 'verified_driver')
        self.assertNotIn('MUJOCO_PY_FORCE_CPU', env)
        self.assertEqual(env['LD_LIBRARY_PATH'], str(q.ROOT/'lib')+':/opt/conda/lib')
        with patch.object(q.prior, 'sha', return_value='changed'):
            with self.assertRaisesRegex(ValueError, 'libraries changed'): q.child_environment(c)

    def fixture(self, code=0):
        child = Mock(pid=123, returncode=code)
        child.poll.return_value = code; child.wait.return_value = code
        return child, types.SimpleNamespace(write=Mock(), started=Mock(return_value={'pid': 123}))

    def test_success_one_child_bound_and_no_signals(self):
        child, c = self.fixture()
        with tempfile.TemporaryDirectory() as temp, patch.object(q.subprocess, 'Popen', return_value=child) as popen:
            self.assertEqual(q.execute(c, {}, Path(temp)), {'pid': 123})
            popen.assert_called_once(); child.terminate.assert_not_called()
            self.assertLessEqual(child.wait.call_args.kwargs['timeout'], 3610)

    def test_nonzero_terminal_no_automatic_retry(self):
        child, c = self.fixture(1)
        with tempfile.TemporaryDirectory() as temp, patch.object(q.subprocess, 'Popen', return_value=child) as popen:
            with self.assertRaisesRegex(ValueError, 'no automatic retry'): q.execute(c, {}, Path(temp))
            popen.assert_called_once(); child.terminate.assert_not_called()

    def test_timeout_kill_fallback_cleans_only_owned_child(self):
        child, c = self.fixture(); child.poll.return_value = None
        child.wait.side_effect = [subprocess.TimeoutExpired('fixture', 3610), subprocess.TimeoutExpired('fixture', 5), 0]
        with tempfile.TemporaryDirectory() as temp, patch.object(q.subprocess, 'Popen', return_value=child):
            with self.assertRaises(subprocess.TimeoutExpired): q.execute(c, {}, Path(temp))
            child.terminate.assert_called_once(); child.kill.assert_called_once()
            self.assertEqual(c.write.call_args.args[1]['predecessor_signals'], 0)

    def test_identity_failure_cleanup_keeps_spawn_evidence(self):
        child, c = self.fixture(); child.poll.return_value = None
        c.started.side_effect = RuntimeError('identity missing')
        with tempfile.TemporaryDirectory() as temp, patch.object(q.subprocess, 'Popen', return_value=child):
            with self.assertRaisesRegex(RuntimeError, 'identity missing'): q.execute(c, {}, Path(temp))
            child.terminate.assert_called_once()
            self.assertEqual(c.write.call_args_list[0].args[0].name, 'SPAWNED.json')


if __name__ == '__main__': unittest.main()
