"""CPU-only recovery collector fixtures; never contact a provider or worker."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

PATH = Path(__file__).resolve().parents[1] / 'scripts/vast/navigation_recovery_collect.py'
SPEC = importlib.util.spec_from_file_location('navigation_recovery_collect', PATH)
r = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(r)


def relative(job):
    return Path(job['task']) / 'conditions' / job['arm'] / ('shard-' + str(job['rank']))


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pins = {'files': r.FILES_SHA, 'original': r.PLAN_SHA, 'script': 'a' * 64, 'recovery': 'b' * 64}
        self.c = SimpleNamespace(SOURCE='c' * 64, FREEZES={'wall': 'd' * 64, 'pointmaze': 'e' * 64},
            validate_plan=Mock(), relative=relative, WORKERS={'in0': None})
        self.worker = {'instance': r.INSTANCE, 'gpu': 0, 'gpu_uuid': r.UUID, 'jobs': copy.deepcopy(r.EXPECTED)}
        self.plan = {'workers': {'in0': self.worker}, 'completed': []}
        self.recovery = {'schema': 1, 'root': str(r.RECOVERY), 'instance': r.INSTANCE, 'gpu': 0,
            'gpu_uuid': r.UUID, 'original_plan_sha256': r.PLAN_SHA, 'original_launch_sha256': r.LAUNCH_SHA,
            'operations_sha256': self.pins['script'], 'source_sha256': self.c.SOURCE,
            'freeze_sha256': self.c.FREEZES, 'expected_endpoints': 180, 'reused_streams': 8,
            'recovered_streams': 7, 'whole_stream_restart_only': True,
            'partial_prefix_excluded_from_analysis': True, 'automatic_retry': False, 'new_episode_identities': 0,
            'unstarted': copy.deepcopy(r.EXPECTED[9:]), 'interrupted': {'job': r.EXPECTED[8]}}
        self.rows = [{**j, 'output': str((r.NAV if i < 8 else r.RECOVERY) / 'results' / relative(j)),
                      'report_sha256': 'f' * 64} for i, j in enumerate(r.EXPECTED)]
        self.recovery.update(completed=copy.deepcopy(self.rows[:8]), recovered=copy.deepcopy(self.rows[8:]))
        self.proof = {'worker': 'in0', 'plan_sha256': r.PLAN_SHA, 'instance': r.INSTANCE,
            'gpu': 0, 'gpu_uuid': r.UUID, 'verified_endpoints': 180, 'scientific_validators_unchanged': True,
            'done_sha256': '1' * 64, 'identity': {'pid': 123, 'starttime': 456, 'command': ['new-recovery']},
            'recovery_plan_sha256': self.pins['recovery'], 'original_launch_sha256': r.LAUNCH_SHA,
            'recovery_root': str(r.RECOVERY), 'partial_records_in_analysis': 0, 'completed_jobs': self.rows}
        self.connections = {50231985: ['ne'], 50205763: ['in'], 50259194: ['tx']}
        self.authority = {'instances': list(self.connections), 'aggregate_usd_hour': 6.9,
            'checked_unix': time.time(), 'owner': 'rep_geometry_transcoder/root', 'gpu_calls': 0,
            'board_sha256': '2' * 64}
        self.stage = SimpleNamespace(connections=Mock(return_value=(self.connections, self.authority)), get=Mock())

    def test_exact_eight_plus_seven_union_and_worker_binding(self):
        r.validate_recovery(self.c, self.plan, self.recovery, self.pins['script'])
        r.validate_proof(self.c, 'in0', self.worker, self.proof, self.pins, self.recovery)
        self.assertEqual(sum(j['task'] == 'wall' for j in r.EXPECTED) * 12, 84)
        self.assertEqual(sum(j['task'] == 'pointmaze' for j in r.EXPECTED) * 12, 96)

    def test_changed_worker_gpu_original_source_or_recovery_pin_refused(self):
        for target, key, value in [(self.worker, 'gpu', 1), (self.worker, 'gpu_uuid', 'other'),
                (self.recovery, 'original_plan_sha256', '0' * 64),
                (self.recovery, 'operations_sha256', '0' * 64), (self.recovery, 'source_sha256', '0' * 64)]:
            with self.subTest(key=key):
                previous = target[key]; target[key] = value
                with self.assertRaises(ValueError): r.validate_recovery(self.c, self.plan, self.recovery, self.pins['script'])
                target[key] = previous

    def test_overlap_missing_order_and_partial_path_refused(self):
        for mutate in (lambda p: p['completed'].pop(),
                lambda p: p['recovered'].__setitem__(0, p['completed'][0]),
                lambda p: p['recovered'].reverse(),
                lambda p: p['recovered'][0].update(output=self.rows[7]['output'])):
            plan = copy.deepcopy(self.recovery); mutate(plan)
            with self.assertRaises(ValueError): r.validate_recovery(self.c, self.plan, plan, self.pins['script'])

    def test_no_fabricated_old_completion_or_partial_analysis(self):
        for key, value in [('recovery_plan_sha256', r.PLAN_SHA), ('partial_records_in_analysis', 1),
                           ('verified_endpoints', 181), ('identity', {'pid': 123}), ('worker', 'ne0')]:
            proof = copy.deepcopy(self.proof); proof[key] = value
            with self.assertRaises(ValueError): r.validate_proof(self.c, 'in0', self.worker, proof, self.pins, self.recovery)

    def test_reused_report_and_mixed_export_paths_must_match(self):
        for index, key, value in [(0, 'report_sha256', '0' * 64), (8, 'output', str(r.NAV / 'partial'))]:
            proof = copy.deepcopy(self.proof); proof['completed_jobs'][index][key] = value
            with self.assertRaises(ValueError): r.validate_proof(self.c, 'in0', self.worker, proof, self.pins, self.recovery)

    def test_provider_fresh_owned_us_budget(self):
        r.fresh_connections(self.stage)
        for key, value in [('aggregate_usd_hour', 7.01), ('aggregate_usd_hour', float('nan')),
                ('aggregate_usd_hour', float('inf')), ('checked_unix', time.time() - 61),
                ('owner', 'other'), ('instances', [r.INSTANCE]), ('gpu_calls', 1)]:
            with self.subTest(key=key, value=value):
                previous = self.authority[key]; self.authority[key] = value
                with self.assertRaises(ValueError): r.fresh_connections(self.stage)
                self.authority[key] = previous

    def test_transport_and_timeout_retryable_but_validator_failures_not(self):
        for error in (subprocess.CalledProcessError(255, ['ssh']), subprocess.TimeoutExpired(['ssh'], 600)):
            self.stage.get.side_effect = error
            with self.assertRaises(r.ObservationUnavailable): r.observe(self.stage, ['ssh'], self.pins, {'kind': 'status'})
        self.stage.get.side_effect = subprocess.CalledProcessError(1, ['ssh'])
        with self.assertRaises(subprocess.CalledProcessError): r.observe(self.stage, ['ssh'], self.pins, {'kind': 'status'})
        self.stage.get.side_effect = None; self.stage.get.return_value = 'not json'
        with self.assertRaises(json.JSONDecodeError): r.observe(self.stage, ['ssh'], self.pins, {'kind': 'status'})

    def test_atomic_receipt_never_overwrites(self):
        path = self.root / 'receipt.json'; r.write(path, {'first': True})
        with self.assertRaises(FileExistsError): r.write(path, {'first': False})
        self.assertEqual(json.loads(path.read_bytes()), {'first': True})
        self.assertTrue(path.with_name(path.name + '.publishing').exists())

    def remote_fixture(self):
        nav, recovery = self.root / 'nav', self.root / 'recovery'
        nav.mkdir(); recovery.mkdir()
        original = {'workers': {'in0': self.worker}, 'completed': []}
        (nav / 'PLAN.json').write_text(json.dumps(original))
        pins = {**self.pins, 'original': r.sha((nav / 'PLAN.json').read_bytes())}
        (nav / 'CUTOVER.json').write_text(json.dumps({'plan_sha256': pins['original']}))
        code = b'import json,sys;print(json.dumps({"argv":sys.argv,"file":__file__}))'
        (nav / 'navigation_redistribution_control.py').write_bytes(code)
        manifest = {'navigation_redistribution_control.py': {'bytes': len(code), 'sha256': r.sha(code)}}
        (nav / 'FILES.json').write_text(json.dumps(manifest)); pins['files'] = r.sha((nav / 'FILES.json').read_bytes())
        (recovery / 'navigation_recovery.py').write_bytes(code); pins['script'] = r.sha(code)
        (recovery / 'PLAN.json').write_text(json.dumps({'operations_sha256': pins['script'], 'original_plan_sha256': pins['original']}))
        pins['recovery'] = r.sha((recovery / 'PLAN.json').read_bytes())
        return nav, recovery, pins

    def remote(self, nav, recovery, pins, request):
        return subprocess.run([sys.executable, '-c', r.BOUND, str(nav), str(recovery), json.dumps(pins), json.dumps(request)],
            capture_output=True, text=True, timeout=10)

    def test_bound_remote_recovery_executes_exact_bytes_and_cli(self):
        nav, recovery, pins = self.remote_fixture()
        result = self.remote(nav, recovery, pins, {'kind': 'recovery', 'mode': 'verify-completed', 'worker': 'in0'})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['argv'], [str(recovery / 'navigation_recovery.py'), 'verify-completed', '--plan-sha256', pins['recovery']])

    def test_bound_remote_changed_source_plan_manifest_or_symlink_refused(self):
        nav, recovery, pins = self.remote_fixture()
        for path in (recovery / 'navigation_recovery.py', recovery / 'PLAN.json', nav / 'PLAN.json', nav / 'FILES.json'):
            original = path.read_bytes(); path.write_bytes(original + b' ')
            result = self.remote(nav, recovery, pins, {'kind': 'recovery', 'mode': 'verify-completed'})
            self.assertNotEqual(result.returncode, 0)
            path.write_bytes(original)
        target = nav / 'PLAN.json'; actual = nav / 'actual'; target.rename(actual); target.symlink_to(actual)
        self.assertNotEqual(self.remote(nav, recovery, pins, {'kind': 'original-plan'}).returncode, 0)

    def test_remote_wrapper_refuses_science_analysis_wrong_exports_and_new_importer(self):
        nav, recovery, pins = self.remote_fixture()
        for request in ({'kind': 'control', 'mode': 'run'}, {'kind': 'control', 'mode': 'analyze'},
                {'kind': 'recovery', 'mode': 'import-result', 'job': r.EXPECTED[0]},
                {'kind': 'recovery', 'mode': 'export-result', 'job': {'task': 'wall', 'arm': 'joint', 'rank': 0}},
                {'kind': 'control', 'mode': 'export-result', 'job': r.EXPECTED[0], 'original': True}):
            with self.subTest(request=request): self.assertNotEqual(self.remote(nav, recovery, pins, request).returncode, 0)

    def test_recovery_status_reads_new_root_not_old_done(self):
        nav, recovery, pins = self.remote_fixture()
        old = nav / 'workers/in0'; old.mkdir(parents=True); (old / 'DONE.json').write_text('{}')
        result = self.remote(nav, recovery, pins, {'kind': 'status', 'worker': 'in0'})
        self.assertEqual(json.loads(result.stdout), {'FAILED.json': False, 'DONE.json': False})
        (recovery / 'run').mkdir(); (recovery / 'run/DONE.json').write_text('{}')
        self.assertTrue(json.loads(self.remote(nav, recovery, pins, {'kind': 'status', 'worker': 'in0'}).stdout)['DONE.json'])

    def test_completion_publish_is_append_only_and_identical_retry_readback(self):
        nav, recovery, pins = self.remote_fixture()
        request = {'kind': 'publish', 'value': {'plan_sha256': pins['original'], 'workers': {'in0': self.proof}}}
        first = self.remote(nav, recovery, pins, request); second = self.remote(nav, recovery, pins, request)
        self.assertEqual(first.returncode, 0, first.stderr); self.assertEqual(second.stdout, first.stdout)
        original = (nav / 'COLLECTION_COMPLETE.json').read_bytes()
        request['value']['changed'] = True
        self.assertNotEqual(self.remote(nav, recovery, pins, request).returncode, 0)
        self.assertEqual((nav / 'COLLECTION_COMPLETE.json').read_bytes(), original)

    def test_poll_observation_failure_leaves_worker_pending(self):
        with patch.object(r, 'fresh_connections', return_value=(self.connections, self.authority)), \
                patch.object(r, 'observe', side_effect=[self.plan, self.recovery, r.ObservationUnavailable('connection reset')]):
            plan, verified, jobs = r.poll(self.c, self.stage, self.root, self.pins)
        self.assertEqual(plan, self.plan); self.assertFalse(verified); self.assertFalse(jobs)
        retry = list(self.root.glob('*OBSERVATION_RETRYABLE.json'))
        self.assertEqual(len(retry), 1)
        self.assertEqual(json.loads(retry[0].read_bytes())['scientific_restarts'], 0)

    def test_poll_recovery_uses_exact_union_not_original_done(self):
        with patch.object(r, 'fresh_connections', return_value=(self.connections, self.authority)), \
                patch.object(r, 'observe', side_effect=[self.plan, self.recovery,
                    {'FAILED.json': False, 'DONE.json': True}, self.proof]) as observe:
            _, verified, jobs = r.poll(self.c, self.stage, self.root, self.pins)
        self.assertEqual(verified['in0'], self.proof); self.assertEqual(jobs['in0'], self.rows)
        self.assertEqual(observe.call_args.args[-1]['kind'], 'recovery')

    def test_worker_failure_is_fatal_not_retry(self):
        with patch.object(r, 'fresh_connections', return_value=(self.connections, self.authority)), \
                patch.object(r, 'observe', side_effect=[self.plan, self.recovery, {'FAILED.json': True, 'DONE.json': True}]):
            with self.assertRaisesRegex(ValueError, 'Required worker failed'): r.poll(self.c, self.stage, self.root, self.pins)

    def test_transfer_routes_mixed_in0_through_recovery_and_frozen_importer(self):
        job = r.EXPECTED[8]
        with patch.object(r.subprocess, 'run') as run, patch.object(r.subprocess, 'check_output',
                return_value=json.dumps({'job': job, 'report_sha256': 'f' * 64})) as output:
            r.copy_job(self.stage, self.connections, self.pins, self.worker, job)
        self.assertEqual(run.call_args.args[0][0], 'in'); self.assertEqual(output.call_args.args[0][0], 'ne')
        self.assertIn('"kind": "recovery"', run.call_args.args[0][-1])
        self.assertIn('"mode": "import-result"', output.call_args.args[0][-1])

    def test_transfer_failure_is_not_observation_and_no_retry(self):
        with patch.object(r.subprocess, 'run', side_effect=subprocess.CalledProcessError(255, ['ssh'])) as run, \
                patch.object(r.subprocess, 'check_output') as output:
            with self.assertRaises(subprocess.CalledProcessError): r.copy_job(self.stage, self.connections, self.pins, self.worker, r.EXPECTED[8])
        run.assert_called_once(); output.assert_not_called()

    def test_existing_collection_receipt_does_not_skip_destination_verification(self):
        row = self.rows[0]; job = {k: row[k] for k in r.KEYS}
        receipt = self.root / ('-'.join(str(job[k]) for k in r.KEYS) + '.json')
        receipt.write_text(json.dumps({'job': job, 'report_sha256': row['report_sha256']}))
        with patch.object(r, 'fresh_connections', return_value=(self.connections, self.authority)) as fresh, \
                patch.object(r, 'copy_job', return_value={'job': job, 'report_sha256': row['report_sha256']}) as copied:
            r.collect_rows(self.stage, self.root, self.pins, self.worker, [row])
        fresh.assert_called_once(); copied.assert_called_once()

    def test_bad_sha_and_cpu_visibility_gate_before_helpers_or_network(self):
        with self.assertRaises(ValueError): r.valid_sha('not a hash')
        args = argparse.Namespace()
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}), patch.object(r, 'frozen_helpers') as helper:
            with self.assertRaises(ValueError): r.watch(args)
        helper.assert_not_called()

    def watcher_fixture(self):
        original = self.root / 'original'; (original / 'collection').mkdir(parents=True)
        self.stage.PROOF = original
        args = argparse.Namespace(worker='in0', recovery_root=r.RECOVERY,
            original_plan_sha256=r.PLAN_SHA, recovery_plan_sha256=self.pins['recovery'],
            recovery_script_sha256=self.pins['script'], collector_sha256=r.sha(PATH.read_bytes()), once=True)
        return args, self.root / 'collection'

    def test_watcher_transient_initial_observation_returns_once_without_transfer(self):
        args, folder = self.watcher_fixture()
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}), patch.object(r, 'LOCAL', folder), \
                patch.object(r, 'frozen_helpers', return_value=(self.c, self.stage, None)), \
                patch.object(r, 'poll', side_effect=r.ObservationUnavailable('connection reset')), \
                patch.object(r, 'copy_job') as transfer:
            r.watch(args)
        transfer.assert_not_called()
        self.assertEqual(len(list(folder.glob('*OBSERVATION_RETRYABLE.json'))), 1)
        self.assertFalse(list(folder.glob('*STOPPED_FOR_REVIEW.json')))

    def test_watcher_ambiguous_transfer_stops_and_preserves_failure(self):
        args, folder = self.watcher_fixture()
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}), patch.object(r, 'LOCAL', folder), \
                patch.object(r, 'frozen_helpers', return_value=(self.c, self.stage, None)), \
                patch.object(r, 'poll', return_value=(self.plan, {'in0': self.proof}, {'in0': self.rows})), \
                patch.object(r, 'collect_rows', side_effect=subprocess.CalledProcessError(255, ['ssh'])) as transfer:
            with self.assertRaises(subprocess.CalledProcessError): r.watch(args)
        transfer.assert_called_once()
        failed = list(folder.glob('*STOPPED_FOR_REVIEW.json')); self.assertEqual(len(failed), 1)
        self.assertTrue(json.loads(failed[0].read_bytes())['partial_evidence_preserved'])
        self.assertFalse(list(folder.glob('*OBSERVATION_RETRYABLE.json')))

    def test_watcher_completion_has_explicit_override_no_old_done_or_analysis(self):
        args, folder = self.watcher_fixture()
        def publication(connection, argv):
            request = json.loads(argv[-1]); self.assertEqual(request['kind'], 'publish')
            data = (json.dumps(request['value'], sort_keys=True, indent=2) + '\n').encode()
            return json.dumps({'sha256': r.sha(data), 'bytes': len(data)})
        self.stage.get.side_effect = publication
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}), patch.object(r, 'LOCAL', folder), \
                patch.object(r, 'frozen_helpers', return_value=(self.c, self.stage, None)), \
                patch.object(r, 'poll', return_value=(self.plan, {'in0': self.proof}, {'in0': self.rows})), \
                patch.object(r, 'collect_rows') as collect, \
                patch.object(r, 'fresh_connections', return_value=(self.connections, self.authority)):
            r.watch(args)
        self.assertEqual(collect.call_count, 2)
        complete = json.loads((folder / 'COLLECTION_COMPLETE.json').read_bytes())
        self.assertEqual(complete['recovery_override']['override_worker'], 'in0')
        self.assertEqual(complete['workers']['in0']['recovery_plan_sha256'], self.pins['recovery'])
        self.assertFalse(complete['recovery_override']['analysis_launched'])
        self.assertFalse(list(self.root.rglob('DONE.json')))
        self.stage.get.assert_called_once()

    def test_frozen_local_helpers_match_original_manifest_without_network(self):
        with patch.object(r.subprocess, 'check_output', side_effect=AssertionError('No network in fixture')):
            common, stage, collector = r.frozen_helpers()
        self.assertEqual(common.SOURCE, 'fc7578672eabd8e08469afcc1f7f4d9b1c2ed18c447f4f003539b61da7950f58')
        self.assertIs(collector.stage, stage)


if __name__ == '__main__': unittest.main()
