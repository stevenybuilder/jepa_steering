"""CPU fixtures for explicit outage recovery; no scientific or GPU execution."""
import argparse
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

PATH = Path(__file__).resolve().parents[1] / 'scripts/vast/navigation_recovery.py'
if not PATH.is_file():
    PATH = Path(__file__).resolve().parents[1] / 'navigation_recovery.py'
SPEC = importlib.util.spec_from_file_location('navigation_recovery', PATH)
r = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(r)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def relative(job):
    return Path(job['task']) / 'conditions' / job['arm'] / ('shard-' + str(job['rank']))


def rows(rank):
    seed = 2026090721 + rank * 6000
    return [{'episode': e, 'logical_rank': rank, 'local_seed': seed,
        'environment_seed': (seed * seed + e * seed) % (2**32 - 2)} for e in range(rank * 12, (rank + 1) * 12)]


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.nav, self.root, self.gpu = self.base / 'original', self.base / 'recovery', self.base / 'gpu'
        for name, value in [('NAV', self.nav), ('ROOT', self.root), ('GPU_CHECK', self.gpu)]:
            hook = patch.object(r, name, value); hook.start(); self.addCleanup(hook.stop)
        self.root.mkdir()
        self.c = SimpleNamespace(SOURCE='a' * 64, FREEZES={'wall': 'b' * 64, 'pointmaze': 'c' * 64},
            TASKS=('wall', 'pointmaze'), PYTHON='/verified/python', PAYLOAD=self.nav / 'payload',
            relative=relative, expected_rows=rows, verify_shard=Mock(return_value='d' * 64),
            environment=Mock(return_value={'CUDA_VISIBLE_DEVICES': '0'}),
            started=Mock(return_value={'pid': 123, 'ppid': 99, 'starttime': 456, 'command': ['frozen']}),
            wait_gpu_release=Mock(return_value={'released': True}))
        self.control = SimpleNamespace(engineering_proof=Mock(return_value='e' * 64))

    def inventory_fixture(self):
        for task in self.c.TASKS:
            put(self.nav / 'workers/in0' / ('ENGINEERING-' + task + '.json'),
                {'path': '/frozen/' + task, 'report_sha256': 'e' * 64, 'gpu_uuid': r.UUID})
        for job in r.EXPECTED[:8]:
            put(self.nav / 'results' / relative(job) / 'DONE.json', {'fixture': True})
        partial = self.nav / 'results' / relative(r.EXPECTED[8])
        put(partial / 'protocol.json', {'task': 'pointmaze', 'arm': 'permuted_visual', 'logical_ranks': [4],
            'source_sha256': self.c.SOURCE, 'freeze_sha256': self.c.FREEZES['pointmaze'],
            'engineering_report_sha256': 'e' * 64, 'engineering_only': False, 'fresh_confirmation': False,
            'expected_episodes': rows(4)})
        put(partial / 'episode-048.json', {**rows(4)[0], 'arm': 'permuted_visual'})
        return partial

    def test_exact_eight_original_seven_recovered_no_pool_growth(self):
        self.assertEqual(len(r.EXPECTED), 15)
        self.assertEqual(sum(j['task'] == 'wall' for j in r.EXPECTED) * 12, 84)
        self.assertEqual(sum(j['task'] == 'pointmaze' for j in r.EXPECTED) * 12, 96)
        self.assertEqual(len({tuple(j.values()) for j in r.EXPECTED}), 15)
        self.assertEqual(r.EXPECTED[8], {'task': 'pointmaze', 'arm': 'permuted_visual', 'rank': 4})
        self.assertEqual(r.SECONDS, 7200)

    def test_receipts_atomic_append_only(self):
        path = self.root / 'receipt.json'; r.write(path, {'first': True})
        with self.assertRaises(FileExistsError): r.write(path, {'first': False})
        self.assertEqual(r.read(path), {'first': True})
        self.assertTrue(path.with_name(path.name + '.publishing').exists())

    def test_evidence_tree_rejects_symlink(self):
        (self.root / 'source').write_text('evidence')
        before = r.tree(self.root)
        self.assertEqual(set(before), {'source'})
        (self.root / 'link').symlink_to(self.root / 'source')
        with self.assertRaisesRegex(ValueError, 'symlink'): r.tree(self.root)

    def test_inventory_verifies_eight_only_preserves_partial(self):
        partial = self.inventory_fixture(); before = r.tree(partial)
        with patch.object(r, 'scientific_verify', return_value={'verified_endpoints': 96}) as science:
            state = r.inventory(self.c, self.control)
        self.assertEqual(len(state['completed']), 8)
        self.assertEqual(state['unstarted'], r.EXPECTED[9:])
        self.assertEqual(state['interrupted']['members'], before)
        self.assertEqual(self.c.verify_shard.call_count, 8)
        self.assertEqual(len(science.call_args.args[2]), 8)
        self.assertEqual(r.tree(partial), before)

    def test_inventory_refuses_changed_pristine_assignment(self):
        self.inventory_fixture()
        (self.nav / 'results' / relative(r.EXPECTED[9])).mkdir(parents=True)
        with patch.object(r, 'scientific_verify') as science:
            with self.assertRaisesRegex(ValueError, 'unstarted'): r.inventory(self.c, self.control)
            science.assert_not_called()

    def test_inventory_refuses_missing_extra_or_completed_partial(self):
        partial = self.inventory_fixture()
        put(partial / 'episode-049.json', {})
        with patch.object(r, 'scientific_verify'):
            with self.assertRaisesRegex(ValueError, 'one preserved'): r.inventory(self.c, self.control)
        (partial / 'episode-049.json').unlink()
        put(partial / 'DONE.json', {})
        with self.assertRaisesRegex(ValueError, 'complete/ambiguous'): r.inventory(self.c, self.control)

    def test_interrupted_protocol_cannot_change_seed_or_proof(self):
        partial = self.inventory_fixture()
        protocol = r.read(partial / 'protocol.json'); protocol['logical_ranks'] = [5]
        put(partial / 'protocol.json', protocol)
        with self.assertRaisesRegex(ValueError, 'Interrupted protocol'): r.inventory(self.c, self.control)

    def test_scientific_comparison_ignores_only_declared_timings(self):
        record = {'seconds': 10., 'result': {'planning_calls': [{'seconds': 9., 'iterations': 30}],
            'native_reward': 0.}, 'planned_actions': [[0., -0.]], 'edit_energy': [{'requested': 2.}]}
        other = copy.deepcopy(record); other['seconds'] = 20.; other['result']['planning_calls'][0]['seconds'] = 19.
        self.assertEqual(r.scientific_bytes(record), r.scientific_bytes(other))
        other['planned_actions'][0][1] = 0.
        self.assertNotEqual(r.scientific_bytes(record), r.scientific_bytes(other))
        other = copy.deepcopy(record); other['edit_energy'][0]['requested'] = 3.
        self.assertNotEqual(r.scientific_bytes(record), r.scientific_bytes(other))
        other = copy.deepcopy(record); other['new_field'] = True
        self.assertNotEqual(r.scientific_bytes(record), r.scientific_bytes(other))

    def test_prefix_raw_difference_not_misreported_as_byte_identical(self):
        old, new = self.root / 'old', self.root / 'new'
        record = {'seconds': 1., 'result': {'planning_calls': [{'seconds': 1.}], 'reward': 3.}}
        put(old / 'episode-048.json', record)
        changed = copy.deepcopy(record); changed['seconds'] = 2.
        put(new / 'episode-048.json', changed)
        plan = {'interrupted': {'output': str(old), 'completed_prefix_files': ['episode-048.json']},
                'recovered': [{'output': str(new)}]}
        value = r.compare_prefix(plan)
        self.assertFalse(value['records'][0]['raw_bytes_equal'])
        self.assertTrue(value['records'][0]['scientific_bytes_equal'])
        self.assertEqual(value['partial_records_in_analysis'], 0)
        changed['result']['reward'] = 4.; put(new / 'episode-048.json', changed)
        with self.assertRaisesRegex(ValueError, 'stop, never select'): r.compare_prefix(plan)

    def test_command_keeps_original_full_stream_seed_no_episode_skip(self):
        self.c.arguments = lambda task: ['--task', task, '--freeze', '/frozen/' + task]
        job = {**r.EXPECTED[8], 'output': str(self.root / 'results' / relative(r.EXPECTED[8]))}
        command = r.command(self.c, job, {'path': '/original/engineering'})
        self.assertIn('offline_study.navigation_coupling_behavior', command)
        self.assertEqual(command[command.index('--logical-ranks') + 1], '4')
        self.assertNotIn('--seed', command); self.assertNotIn('--episode-start', command)
        native = r.native_command(self.c)
        self.assertIn('offline_study.navigation_smoke', native)
        self.assertNotIn('--repetitions', native)

    def test_stopped_originals_never_signalled(self):
        put(self.gpu / 'ACTIVATED.json', {'pid': 20, 'start_ticks': 21, 'argv': ['old-waiter']})
        launch = {'identity': {'pid': 10, 'starttime': 11, 'command': ['original']}}
        self.c.alive = Mock(return_value=False)
        self.assertEqual(len(r.stopped(self.c, launch)), 2)
        self.c.alive.return_value = True
        with self.assertRaisesRegex(ValueError, 'still alive'): r.stopped(self.c, launch)

    def test_execute_records_spawn_before_identity_then_exit(self):
        child = Mock(pid=123); child.wait.return_value = 0
        def identity(pid, argv):
            self.assertTrue((self.root / 'stage-SPAWNED.json').exists())
            return {'pid': pid, 'ppid': 99, 'starttime': 456, 'command': argv}
        self.c.started.side_effect = identity
        with patch.object(r.subprocess, 'Popen', return_value=child):
            result = r.execute(self.c, ['frozen'], 'stage', self.root)
        self.assertEqual(result['pid'], 123)
        self.assertEqual(r.read(self.root / 'stage-EXIT.json')['returncode'], 0)
        child.terminate.assert_not_called(); child.kill.assert_not_called()

    def test_execute_timeout_only_terminates_owned_child_and_stops(self):
        child = Mock(pid=123, returncode=-9); child.poll.return_value = None
        child.wait.side_effect = [subprocess.TimeoutExpired('frozen', 7200), subprocess.TimeoutExpired('frozen', 5), -9]
        with patch.object(r.subprocess, 'Popen', return_value=child), patch.object(r.signal, 'signal', return_value=None):
            with self.assertRaises(subprocess.TimeoutExpired): r.execute(self.c, ['frozen'], 'stage', self.root)
        child.terminate.assert_called_once(); child.kill.assert_called_once()
        self.assertEqual(r.read(self.root / 'stage-CLEANUP.json')['original_process_signals'], 0)
        self.assertFalse((self.root / 'stage-EXIT.json').exists())

    def test_spawn_receipt_failure_retains_owned_handle_for_cleanup(self):
        child = Mock(pid=123, returncode=-15); child.poll.return_value = None
        original_write = r.write
        def write(path, value):
            if path.name.endswith('-SPAWNED.json'): raise OSError('fixture disk error')
            return original_write(path, value)
        with patch.object(r.subprocess, 'Popen', return_value=child), patch.object(r, 'write', side_effect=write), \
                patch.object(r.signal, 'signal', return_value=None):
            with self.assertRaisesRegex(OSError, 'disk error'): r.execute(self.c, ['frozen'], 'stage', self.root)
        child.terminate.assert_called_once()
        self.assertEqual(r.read(self.root / 'stage-CLEANUP.json')['pid'], 123)

    def test_authority_requires_explicit_whole_stream_restart_and_finite_expiry(self):
        plan = {'operations_sha256': 'a' * 64, 'runtime': {'sha256': 'b' * 64}}
        auth = {'owner': 'rep_geometry_transcoder/root', 'operation': 'explicit_navigation_outage_recovery',
            'plan_sha256': 'c' * 64, 'operations_sha256': plan['operations_sha256'],
            'runtime_sha256': 'b' * 64, 'instance': r.INSTANCE, 'gpu': r.GPU, 'gpu_uuid': r.UUID,
            'original_plan_sha256': r.PLAN_SHA, 'original_launch_sha256': r.LAUNCH_SHA,
            'whole_stream_restarts': r.EXPECTED[8:9], 'unstarted_streams': r.EXPECTED[9:],
            'maximum_child_seconds': 7200, 'native_repeat_episodes': 2, 'automatic_retry': False,
            'new_episode_identities': 0, 'issued_unix': 1000, 'expires_unix': 2000}
        path = self.root / 'AUTHORIZATION.json'; put(path, auth)
        args = argparse.Namespace(authorization=path, authorization_sha256=r.digest(path), plan_sha256='c' * 64)
        with patch.object(r.time, 'time', return_value=1500): r.authority(args, plan)
        with patch.object(r.time, 'time', return_value=1500):
            with self.assertRaisesRegex(ValueError, 'complete child and cleanup'):
                r.authority(args, plan, required_seconds=r.SECONDS + 5)
        for change in ({'automatic_retry': True}, {'whole_stream_restarts': []}, {'expires_unix': float('nan')}, {'expires_unix': 100000}):
            put(path, {**auth, **change}); args.authorization_sha256 = r.digest(path)
            with patch.object(r.time, 'time', return_value=1500), self.assertRaises(ValueError): r.authority(args, plan)

    def test_run_requires_hidden_cuda_before_importing_original_helpers(self):
        path = self.root / 'PLAN.json'; put(path, {})
        args = argparse.Namespace(plan=path, plan_sha256=r.digest(path), authorization=self.root / 'AUTHORIZATION.json')
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}), patch.object(r, 'frozen_ops') as load:
            with self.assertRaisesRegex(ValueError, 'CPU-hidden'): r.run(args)
            load.assert_not_called()

    def test_runtime_discloses_historical_gap_but_requires_current_hashes(self):
        python = self.root / 'python'; python.write_bytes(b'fixed-runtime')
        driver = self.root / 'driver'; driver.write_bytes(b'current-driver')
        self.c.PYTHON = str(python)
        put(self.nav / 'INPUTS.json', {'fixture': 'frozen payload manifest'})
        value = {'status': 'same_device_post_outage_runtime_verified',
            'owner': 'rep_geometry_transcoder/root', 'instance': r.INSTANCE, 'gpu': r.GPU,
            'gpu_uuid': r.UUID, 'original_launch_sha256': r.LAUNCH_SHA,
            'source_sha256': self.c.SOURCE, 'freeze_sha256': self.c.FREEZES,
            'pre_outage_payload_hashes_reverified': True, 'historical_host_libraries_bound': False,
            'unbound_historical_runtime_paths': [str(driver)],
            'input_manifest_sha256': r.digest(self.nav / 'INPUTS.json'),
            'files': {str(python): r.descriptor(python), str(driver): r.descriptor(driver)},
            'namespace': {'boot_id': 'fixture', 'pid1_starttime': 5}}
        path = self.root / 'RUNTIME.json'; put(path, value)
        with patch.object(r, 'namespace', return_value=value['namespace']):
            self.assertFalse(r.verify_runtime(path, r.digest(path), self.c)['historical_host_libraries_bound'])
            driver.write_bytes(b'changed-driver')
            with self.assertRaisesRegex(ValueError, 'Runtime bytes changed'):
                r.verify_runtime(path, r.digest(path), self.c)
        driver.write_bytes(b'current-driver')
        with patch.object(r, 'namespace', return_value={'boot_id': 'different'}):
            with self.assertRaisesRegex(ValueError, 'namespace changed'):
                r.verify_runtime(path, r.digest(path), self.c)

    def test_plan_rejects_added_identity_changed_output_or_omitted_prefix(self):
        self.inventory_fixture()
        with patch.object(r, 'scientific_verify'):
            state = r.inventory(self.c, self.control)
        put(self.gpu / 'ACTIVATED.json', {'pid': 20, 'start_ticks': 21, 'argv': ['old-waiter']})
        plan = {'schema': 1, 'root': str(self.root), 'instance': r.INSTANCE, 'gpu': r.GPU,
            'gpu_uuid': r.UUID, 'original_plan_sha256': r.PLAN_SHA, 'original_launch_sha256': r.LAUNCH_SHA,
            'source_sha256': self.c.SOURCE, 'freeze_sha256': self.c.FREEZES,
            'operations_sha256': r.digest(PATH), 'expected_endpoints': 180,
            'reused_streams': 8, 'recovered_streams': 7, 'partial_prefix_excluded_from_analysis': True,
            'whole_stream_restart_only': True, 'automatic_retry': False, 'new_episode_identities': 0,
            'original_worker_tree': r.tree(self.nav / 'workers/in0'),
            'original_waiter_launch': r.descriptor(self.gpu / 'ACTIVATED.json'),
            'runtime': {'path': 'fixture-runtime', 'sha256': 'x'}, 'namespace': {'fixture': True}, **state,
            'recovered': [{**j, 'output': str(self.root / 'results' / relative(j))} for j in r.EXPECTED[8:]]}
        with patch.object(r, 'original'), patch.object(r, 'verify_runtime'):
            r.verify_plan(plan, self.c, live=False)
            changed = copy.deepcopy(plan); changed['recovered'].append(changed['recovered'][0])
            with self.assertRaisesRegex(ValueError, 'assignment grew'): r.verify_plan(changed, self.c, live=False)
            changed = copy.deepcopy(plan); changed['recovered'][0]['output'] = '/overwrite/original'
            with self.assertRaisesRegex(ValueError, 'output path'): r.verify_plan(changed, self.c, live=False)
            changed = copy.deepcopy(plan); changed['interrupted']['completed_prefix_files'] = []
            with self.assertRaisesRegex(ValueError, 'cannot be omitted'): r.verify_plan(changed, self.c, live=False)

    def test_native_verification_program_compiles_without_model_execution(self):
        tree = ast.parse(PATH.read_text())
        programs = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and
            isinstance(n.value, str) and 'from offline_study.navigation_smoke import' in n.value]
        self.assertEqual(len(programs), 1)
        compile(programs[0], 'native-verifier', 'exec')
        self.assertIn('comparison_record', programs[0]); self.assertIn('range(2)', programs[0])


if __name__ == '__main__': unittest.main()
