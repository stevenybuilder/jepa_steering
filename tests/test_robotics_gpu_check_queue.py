"""CPU-only tests for the one-shot Indiana engineering handoff."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / 'scripts/vast/robotics_gpu_check_queue.py'
SPEC = importlib.util.spec_from_file_location('robotics_gpu_check_queue', SOURCE)
q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(q)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def descriptor(path):
    return {'bytes': path.stat().st_size, 'sha256': q.digest(path)}


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'isolated'
        self.root.mkdir()
        self.batch = Path(self.temp.name) / 'batch'
        self.nav = Path(self.temp.name) / 'nav'
        self.nav_status = self.nav / 'workers/in0'
        self.nav_status.mkdir(parents=True)
        for key, value in [('ROOT', self.root), ('BATCH', self.batch), ('NAV', self.nav)]:
            hook = patch.object(q, key, value)
            hook.start()
            self.addCleanup(hook.stop)
        self.plan = {'schema': 1, 'root': str(self.root), 'instance': q.INSTANCE,
            'gpu': 0, 'gpu_uuid': q.GPU_UUID, 'batch_root': str(self.batch), 'vendor': q.VENDOR,
            'source_sha256': 'a' * 64,
            'source_manifest': {'path': str(self.root / 'SOURCE_FILES.json'), 'sha256': 'b' * 64},
            'runtime': {'python': q.PYTHON, 'overlay': '/runtime/site-packages', 'ld_library_path': '/opt/conda/lib'},
            'fixed_files': {q.PYTHON: {'bytes': 1, 'sha256': 'c' * 64},
                **{str(self.batch / n): {'bytes': 1, 'sha256': h} for n, h in q.INPUTS.items()}}}

    def authorized(self, **changes):
        value = {'status': 'root_authorized_native32_gpu_check', 'owner': 'rep_geometry_transcoder/root',
            'plan_sha256': 'a' * 64, 'instance': q.INSTANCE, 'gpu': 0, 'gpu_uuid': q.GPU_UUID,
            'child_timeout_seconds': 1210, 'kill_grace_seconds': 5, 'full_history_authorized': False,
            'issued_unix': 1000, 'expires_unix': 1000 + 12 * 3600, **changes}
        path = self.root / 'AUTH.json'
        put(path, value)
        return path, q.digest(path)

    def arguments(self, prepare=False):
        path = self.root / 'PLAN.json'
        put(path, self.plan)
        return argparse.Namespace(plan=path, plan_sha256=q.digest(path),
            authorization=self.root / 'AUTH.json', authorization_sha256='c' * 64,
            prepare_only=prepare)

    def prepared(self):
        args = self.arguments(True)
        static = {'source_sha256': self.plan['source_sha256'],
            'source_manifest_sha256': self.plan['source_manifest']['sha256'],
            'fixed_files_verified': len(self.plan['fixed_files']), 'model_imports': 0, 'gpu_calls': 0}
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}), \
                patch.object(q, 'authorization'), patch.object(q, 'verify_static', return_value=static), \
                patch.object(q, 'verify_navigation_static', return_value={}), \
                patch.object(q, 'gpu_snapshot') as gpu, patch.object(q.subprocess, 'Popen') as launch:
            result = q.run(args)
            gpu.assert_not_called()
            launch.assert_not_called()
        self.assertFalse((self.root / 'queue').exists())
        self.assertFalse((self.root / 'check').exists())
        self.assertEqual(result['gpu_calls'], 0)
        return args

    def test_plan_requires_exact_owner_device_input_pins_and_runtime(self):
        q.validate_plan(self.plan)
        for key, bad in [('instance', 1), ('gpu', 1), ('gpu_uuid', 'GPU-other'), ('batch_root', '/elsewhere'),
                ('root', '/workspace'), ('source_sha256', 'unbound')]:
            plan = copy.deepcopy(self.plan)
            plan[key] = bad
            with self.assertRaises(ValueError):
                q.validate_plan(plan)
        for change in ('python', 'input', 'path'):
            plan = copy.deepcopy(self.plan)
            if change == 'python':
                plan['runtime']['python'] = '/python3.11'
            elif change == 'input':
                plan['fixed_files'][str(self.batch / 'batch.pt')]['sha256'] = 'd' * 64
            else:
                plan['runtime']['overlay'] = '/runtime::/other'
            with self.assertRaises(ValueError):
                q.validate_plan(plan)

    def test_environment_is_cpu_hidden_and_child_is_only_gpu_zero(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1', 'LD_PRELOAD': 'bad',
                'PYTHONHOME': '/other', 'TORCH_HOME': '/other', 'XDG_CACHE_HOME': '/other'}):
            cpu, child = q.environment(self.plan), q.environment(self.plan, gpu=True)
        self.assertEqual(cpu['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(child['CUDA_VISIBLE_DEVICES'], '0')
        self.assertEqual(cpu['PYTHONPATH'], str(self.root / 'src') + ':/runtime/site-packages')
        for key in ('LD_PRELOAD', 'PYTHONHOME', 'TORCH_HOME', 'XDG_CACHE_HOME'):
            self.assertNotIn(key, cpu)
        self.assertEqual(child['OMP_NUM_THREADS'], '1')

    def test_authorization_fresh_start_twelve_hour_expiry_and_no_histories(self):
        path, digest = self.authorized()
        q.authorization(path, digest, 'a' * 64, fresh=True, now=1200)
        q.authorization(path, digest, 'a' * 64, fresh=False, now=3000)
        for now, fresh in [(999, False), (1301, True), (44200, False)]:
            with self.assertRaises(ValueError):
                q.authorization(path, digest, 'a' * 64, fresh=fresh, now=now)
        for change in ({'expires_unix': 44201}, {'full_history_authorized': True},
                {'child_timeout_seconds': 1211}, {'kill_grace_seconds': 6},
                {'expires_unix': float('nan')}, {'owner': 'another-project'}):
            path, digest = self.authorized(**change)
            with self.assertRaises(ValueError):
                q.authorization(path, digest, 'a' * 64, now=1200)

    def test_receipts_are_atomic_and_immutable(self):
        path = self.root / 'receipt.json'
        q.write(path, {'first': True})
        with self.assertRaises(FileExistsError):
            q.write(path, {'first': False})
        self.assertTrue(q.read(path)['first'])
        self.assertTrue(path.with_name('receipt.json.publishing').exists())

    def test_runtime_symlink_requires_exact_target_and_bytes(self):
        real, link = self.root / 'python-real', self.root / 'python'
        real.write_bytes(b'fixed executable bytes')
        link.symlink_to(real)
        row = {**descriptor(real), 'symlink': str(real)}
        q.verify_file(link, row)
        for changed in ({**row, 'symlink': '/other'}, descriptor(real), {**row, 'sha256': '0' * 64}):
            with self.assertRaises(ValueError):
                q.verify_file(link, changed)

    def test_static_preflight_verifies_frozen_sources_tests_and_input_bytes(self):
        paths = ['src/offline_study/' + x for x in q.SOURCE_MEMBERS]
        paths += ['scripts/vast/robotics_gpu_check_queue.py', 'tests/test_robotics_gpu_check_queue.py']
        files = {}
        for name in paths:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# CPU fixture ' + name + '\n')
            files[name] = descriptor(path)
        put(self.root / 'SOURCE_FILES.json', files)
        runtime = self.root / 'python'
        runtime.write_bytes(b'pinned-test-runtime')
        pins = {}
        for name in q.INPUTS:
            put(self.batch / name, {'fixture': name})
            pins[name] = q.digest(self.batch / name)
        plan = copy.deepcopy(self.plan)
        plan['runtime']['python'] = str(runtime)
        plan['source_sha256'] = q.source_hash(self.root / 'src/offline_study')
        plan['source_manifest']['sha256'] = q.digest(self.root / 'SOURCE_FILES.json')
        plan['fixed_files'] = {str(runtime): descriptor(runtime),
            **{str(self.batch / n): descriptor(self.batch / n) for n in pins}}
        with patch.object(q, 'PYTHON', str(runtime)), patch.object(q, 'INPUTS', pins), \
                patch.object(q, '__file__', str(self.root / 'scripts/vast/robotics_gpu_check_queue.py')):
            self.assertEqual(q.verify_static(plan)['model_imports'], 0)
            (self.root / 'src/offline_study/model_loader.py').write_text('# changed\n')
            with self.assertRaisesRegex(ValueError, 'Pinned file changed'):
                q.verify_static(plan)

    def test_predecessor_uses_actual_pinned_identity(self):
        self.assertEqual(q.predecessor()['pid'], 10073)
        self.assertEqual(q.predecessor()['starttime'], 17416272)
        self.assertEqual(q.NAV_LAUNCH_SHA, 'ae9ac3397e0a0611e375d4e2657fe429041d2367a8fab3b64f0ebc544c3a2597')
        live = {**q.predecessor(), 'state': 'S'}
        self.assertTrue(q.predecessor_pending(lambda pid: live))
        for current in (None, {**live, 'state': 'Z'}, {**live, 'starttime': 1}, {**live, 'command': ['other']}):
            with self.assertRaises(ValueError):
                q.predecessor_pending(lambda pid: current)

    def test_normal_exit_race_rechecks_done_and_failure_takes_precedence(self):
        def exits(pid):
            put(self.nav_status / 'DONE.json', {})
            return None
        self.assertFalse(q.predecessor_pending(exits))
        put(self.nav_status / 'FAILED.json', {'error': 'preserved'})
        with self.assertRaises(ValueError):
            q.predecessor_pending(exits)

    def test_concurrent_failure_and_done_after_proc_read_is_not_success(self):
        def exits(pid):
            put(self.nav_status / 'DONE.json', {})
            put(self.nav_status / 'FAILED.json', {})
            return None
        with self.assertRaises(ValueError):
            q.predecessor_pending(exits)

    def test_process_parser_handles_parentheses_and_terminal_empty_argv(self):
        proc = self.root / 'proc'
        entry = proc / '123'
        entry.mkdir(parents=True)
        fields = ['Z'] + ['0'] * 18 + ['4567']
        (entry / 'stat').write_text('123 (a (b) python) ' + ' '.join(fields))
        (entry / 'cmdline').write_bytes(b'')
        result = q.process(123, proc)
        self.assertEqual(result, {'pid': 123, 'starttime': 4567, 'state': 'Z', 'command': []})
        self.assertFalse(q.same_live({'pid': 123, 'starttime': 4567, 'command': ['python']}, result))
        self.assertIsNone(q.process(124, proc))

    def test_whole_worker_verifier_rejects_partial_scope_before_subprocess(self):
        launch = {'identity': {**q.predecessor(), 'state': 'R', 'ppid': 1}}
        put(self.nav_status / 'DONE.json', {'status': 'navigation_redistribution_assignment_complete',
            'worker': 'in0', 'plan_sha256': q.NAV_PLAN_SHA, 'launch_sha256': q.NAV_LAUNCH_SHA,
            'identity': launch['identity'], 'endpoints': 12})
        with patch.object(q, 'verify_navigation_static', return_value=launch), \
                patch.object(q.subprocess, 'check_output') as called, self.assertRaises(ValueError):
            q.verify_predecessor(self.plan)
        called.assert_not_called()

    def test_full_180_endpoint_verifier_is_cpu_hidden_and_exactly_bound(self):
        identity = {**q.predecessor(), 'state': 'R', 'ppid': 1}
        put(self.nav_status / 'DONE.json', {'status': 'navigation_redistribution_assignment_complete',
            'worker': 'in0', 'plan_sha256': q.NAV_PLAN_SHA, 'launch_sha256': q.NAV_LAUNCH_SHA,
            'identity': identity, 'endpoints': 180})
        result = {'worker': 'in0', 'instance': q.INSTANCE, 'gpu': 0, 'gpu_uuid': q.GPU_UUID,
            'plan_sha256': q.NAV_PLAN_SHA, 'identity': identity, 'verified_endpoints': 180,
            'scientific_validators_unchanged': True, 'done_sha256': q.digest(self.nav_status / 'DONE.json')}
        with patch.object(q, 'verify_navigation_static', return_value={'identity': identity}), \
                patch.object(q.subprocess, 'check_output', return_value=json.dumps(result)) as call:
            self.assertEqual(q.verify_predecessor(self.plan), result)
        self.assertEqual(call.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(call.call_args.args[0][-3:], ['verify-completed', '--worker', 'in0'])
        with patch.object(q, 'verify_navigation_static', return_value={'identity': identity}), \
                patch.object(q.subprocess, 'check_output', return_value=json.dumps({**result, 'verified_endpoints': 96})), \
                self.assertRaises(ValueError):
            q.verify_predecessor(self.plan)

    def test_navigation_hash_failure_precedes_any_verifier_call(self):
        put(self.nav / 'PLAN.json', {'not': 'original'})
        with patch.object(q.subprocess, 'check_output') as called, self.assertRaises(ValueError):
            q.verify_navigation_static()
        called.assert_not_called()

    def test_gpu_release_allows_terminal_nvml_lag_then_two_empty_samples(self):
        snapshots = iter([(q.GPU_UUID, [999]), (q.GPU_UUID, []), (q.GPU_UUID, [])])
        result = q.wait_release(snapshot=lambda: next(snapshots), reader=lambda pid: None,
            sleeper=lambda seconds: None)
        self.assertEqual(result['stable_empty_samples'], 2)
        self.assertEqual(len(result['observations']), 3)

    def test_gpu_release_rejects_live_unknown_work_uuid_change_and_pid_reuse(self):
        with self.assertRaises(ValueError):
            q.wait_release(snapshot=lambda: (q.GPU_UUID, [999]), reader=lambda pid:
                None if pid == 10073 else {'state': 'R'}, sleeper=lambda seconds: None)
        with self.assertRaises(ValueError):
            q.wait_release(snapshot=lambda: ('GPU-other', []), reader=lambda pid: None)
        with self.assertRaises(ValueError):
            q.wait_release(snapshot=lambda: (q.GPU_UUID, []), reader=lambda pid:
                {**q.predecessor(), 'starttime': 42, 'state': 'S'})

    def test_empty_gpu_is_not_release_while_supervisor_is_live(self):
        time = [0.0]
        def sleep(seconds):
            time[0] += seconds
        with self.assertRaises(TimeoutError):
            q.wait_release(snapshot=lambda: (q.GPU_UUID, []),
                reader=lambda pid: {**q.predecessor(), 'state': 'S'},
                clock=lambda: time[0], sleeper=sleep, timeout=1)
        self.assertEqual(time[0], 1)

    def test_release_rechecks_predecessor_failure(self):
        put(self.nav_status / 'FAILED.json', {})
        with self.assertRaises(ValueError):
            q.wait_release(snapshot=lambda: (q.GPU_UUID, []), reader=lambda pid: None)

    def test_device_lock_prevents_two_independent_waiters_and_releases(self):
        path = self.root / 'shared-device.lock'
        with q.device_lock(path):
            with self.assertRaises(BlockingIOError):
                with q.device_lock(path):
                    self.fail('Duplicate device claim')
        with q.device_lock(path):
            pass

    def test_prepare_only_does_not_claim_gpu_and_cannot_overwrite_preparation(self):
        args = self.prepared()
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}), patch.object(q, 'authorization'), \
                patch.object(q, 'verify_static', return_value={}), patch.object(q, 'verify_navigation_static'), \
                self.assertRaises(FileExistsError):
            q.run(args)

    def test_waiting_failure_is_preserved_without_any_gpu_or_child_call(self):
        args = self.prepared()
        args.prepare_only = False
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}), patch.object(q, 'authorization'), \
                patch.object(q, 'verify_static', return_value={}), patch.object(q, 'verify_navigation_static'), \
                patch.object(q, 'predecessor_pending', side_effect=ValueError('failed predecessor')), \
                patch.object(q, 'gpu_snapshot') as gpu, patch.object(q.subprocess, 'Popen') as launch, \
                self.assertRaisesRegex(ValueError, 'failed predecessor'):
            q.run(args)
        gpu.assert_not_called()
        launch.assert_not_called()
        self.assertFalse(q.read(self.root / 'queue/FAILED.json')['automatic_retry'])
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}), patch.object(q, 'authorization'), \
                self.assertRaises(FileExistsError):
            q.run(args)

    def test_launch_refuses_changed_preparation_or_visible_cuda(self):
        args = self.prepared()
        args.prepare_only = False
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}), self.assertRaises(ValueError):
            q.run(args)
        prepared = q.read(self.root / 'PREPARED.json')
        prepared['authorization_sha256'] = 'd' * 64
        put(self.root / 'PREPARED.json', prepared)
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}), patch.object(q, 'authorization'), \
                self.assertRaises(ValueError):
            q.run(args)
        self.assertFalse((self.root / 'queue').exists())

    def test_child_command_has_no_history_seed_precision_or_count_override(self):
        command = q.child_command(self.plan)
        self.assertIn('offline_study.robotics_training_gpu_check', command)
        self.assertEqual(command[command.index('--output') + 1], str(self.root / 'check'))
        self.assertEqual(command[command.index('--expected-device-uuid') + 1], q.GPU_UUID)
        for flag in ('--epochs', '--seed', '--updates', '--resume-from', '--precision'):
            self.assertNotIn(flag, command)

    def test_owned_popen_timeout_uses_term_then_kill_only_after_five_seconds(self):
        child = FakeChild([subprocess.TimeoutExpired('owned', 5), 0])
        self.assertEqual(q.stop_owned_child(child), ['terminate_owned_popen', 'kill_owned_popen_after_five_seconds'])
        self.assertEqual(child.signals, ['TERM', 'KILL'])
        self.assertEqual(child.waits, [5, 5])
        self.assertEqual(q.stop_owned_child(child), [])

    def test_spawn_is_recorded_before_identity_and_timeout_preserves_child_receipts(self):
        queue = self.root / 'queue'
        queue.mkdir()
        command = q.child_command(self.plan)
        child = FakeChild([subprocess.TimeoutExpired(command, 1210), 0])
        def identity(pid):
            self.assertTrue((queue / 'CHILD_SPAWNED.json').exists())
            return {'pid': pid, 'starttime': 1234, 'state': 'R', 'command': command}
        with patch.object(q.subprocess, 'Popen', return_value=child) as launch, \
                patch.object(q, 'process', side_effect=identity), self.assertRaises(subprocess.TimeoutExpired):
            q.execute_child(self.plan, queue)
        self.assertEqual(launch.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '0')
        self.assertLessEqual(child.waits[0], 1210)
        self.assertEqual(child.signals, ['TERM'])
        self.assertEqual(q.read(queue / 'CHILD_LAUNCH.json')['starttime'], 1234)
        self.assertEqual(q.read(queue / 'CHILD_CLEANUP.json')['predecessor_signals'], 0)

    def test_spawn_receipt_failure_still_cleans_only_the_owned_popen(self):
        queue = self.root / 'queue'
        queue.mkdir()
        child = FakeChild([0])
        original = q.write
        def fails_once(path, value):
            if path.name == 'CHILD_SPAWNED.json':
                raise OSError('receipt disk failure')
            return original(path, value)
        with patch.object(q.subprocess, 'Popen', return_value=child), patch.object(q, 'write', side_effect=fails_once), \
                patch.object(q, 'process') as identity, self.assertRaisesRegex(OSError, 'receipt disk failure'):
            q.execute_child(self.plan, queue)
        identity.assert_not_called()
        self.assertEqual(child.signals, ['TERM'])
        self.assertEqual(q.read(queue / 'CHILD_CLEANUP.json')['pid'], child.pid)

    def test_child_timeout_is_measured_from_spawn_not_reset_after_identity(self):
        queue = self.root / 'queue'
        queue.mkdir()
        child = FakeChild([0])
        identity = {'pid': child.pid, 'starttime': 1234, 'state': 'R', 'command': q.child_command(self.plan)}
        with patch.object(q.subprocess, 'Popen', return_value=child), \
                patch.object(q, 'process', return_value=identity), \
                patch.object(q.time, 'monotonic', side_effect=[10.0, 10.0, 12.0, 13.0]):
            q.execute_child(self.plan, queue)
        self.assertEqual(child.waits, [1207.0])
        self.assertEqual(child.signals, [])
        self.assertEqual(q.read(queue / 'CHILD_EXIT.json')['returncode'], 0)

    def complete_child_fixture(self):
        root = self.root / 'check'
        root.mkdir()
        for name in q.SOURCE_MEMBERS:
            path = self.root / 'src/offline_study' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# immutable fixture\n')
        protocol = {'task': 'pusht', 'model_seed': 234, 'timed_updates': 4,
            'input_proof': str(self.batch), 'vendor': q.VENDOR, 'expected_device_uuid': q.GPU_UUID,
            'logical_ranks': 32, 'microbatch': 8, 'global_batch': 256,
            'numerical_tolerance': 'bitwise_no_relaxation', 'engineering_only': True,
            'validation_or_confirmation_access': False, 'history_launch_authorized': False,
            'source_sha256': {x: q.digest(self.root / 'src/offline_study' / x) for x in q.SOURCE_MEMBERS}}
        put(root / 'protocol.json', protocol)
        report = {'status': 'native32_pusht_disposable_gpu_arithmetic_check_complete',
            'protocol_sha256': q.digest(root / 'protocol.json'), 'engineering_only': True,
            'training_history_updates': 0, 'history_launch_authorized': False,
            'validation_or_confirmation_access': False,
            'live_model_rng_optimizer_scheduler_unchanged_by_verification': True,
            'native_loader_lpips_rng_or_resume_verified': False,
            'physical_ddp_allreduce_order_reproduced': False,
            'device': {'device_uuid': q.GPU_UUID.removeprefix('GPU-')},
            'inputs': {'root': str(self.batch), 'files_sha256': {k: v for k, v in q.INPUTS.items() if k != 'DONE.json'}},
            'timed_engineering_updates': [{'update': x} for x in range(1, 5)],
            'actual_call_timings': [{'completed': True} for _ in range(6)]}
        for name in ('constructor', 'numerical_proof', 'clone_state'):
            put(root / (name + '.json'), {'proof': name})
            report[name + '_sha256'] = q.digest(root / (name + '.json'))
        put(root / 'report.json', report)
        put(root / 'DONE.json', {'report_sha256': q.digest(root / 'report.json')})
        return root, report

    def test_child_completion_binds_raw_proof_device_source_and_exact_updates(self):
        root, original = self.complete_child_fixture()
        self.assertEqual(q.verify_child_result(self.plan)['timed_engineering_updates'], 4)
        mutations = [lambda r: r.update(training_history_updates=1),
            lambda r: r.update(history_launch_authorized=True),
            lambda r: r['device'].update(device_uuid='another'),
            lambda r: r['timed_engineering_updates'].pop(),
            lambda r: r['actual_call_timings'].pop(),
            lambda r: r['actual_call_timings'][0].update(completed=False)]
        for change in mutations:
            report = copy.deepcopy(original)
            change(report)
            put(root / 'report.json', report)
            put(root / 'DONE.json', {'report_sha256': q.digest(root / 'report.json')})
            with self.assertRaises(ValueError):
                q.verify_child_result(self.plan)
        put(root / 'report.json', original)
        put(root / 'DONE.json', {'report_sha256': q.digest(root / 'report.json')})
        (root / 'numerical_proof.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'raw proof changed'):
            q.verify_child_result(self.plan)

    def test_child_checkpoint_output_is_not_disposable_completion(self):
        root, _ = self.complete_child_fixture()
        (root / 'model.pth.tar').write_bytes(b'not allowed')
        with self.assertRaisesRegex(ValueError, 'checkpoints'):
            q.verify_child_result(self.plan)


class FakeChild:
    pid = 98765

    def __init__(self, waits):
        self.responses = list(waits)
        self.returncode = None
        self.signals, self.waits = [], []

    def poll(self):
        return self.returncode

    def terminate(self):
        self.signals.append('TERM')

    def kill(self):
        self.signals.append('KILL')

    def wait(self, timeout):
        self.waits.append(timeout)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        self.returncode = response
        return response


if __name__ == '__main__':
    unittest.main()
