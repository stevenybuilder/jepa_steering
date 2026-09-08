"""Bounded operations tests; no production processes, GPUs or scientific launches."""
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

scripts = Path(__file__).resolve().parents[1] / 'scripts/vast'
if not (scripts / 'navigation_redistribution_common.py').exists():
    scripts = Path(__file__).resolve().parent
sys.path.insert(0, str(scripts))
import navigation_redistribution_common as c
import navigation_redistribution_control as control

UUID = 'GPU-99bc03dd-d8ee-d7c3-fb46-5771e42b95a6'


def binding(pid=4104, state='S'):
    return {'pid': pid, 'ppid': 1, 'state': state, 'starttime': 123,
            'command': ['python', 'owned-parent']}


def receiving():
    result = {str(number): {'instance': number, 'workers': {}} for number in (50231985, 50205763, 50259194)}
    for name, (number, gpu, task) in c.WORKERS.items():
        result[str(number)]['workers'][name] = {'instance': number, 'gpu': gpu, 'gpu_uuid': UUID}
    for name, pid in (('ne0', 4253), ('ne1', 4104)):
        result['50231985']['workers'][name]['parent'] = binding(pid)
    result['50231985']['watcher'] = binding(4384)
    return result


def remaining():
    return [j for j in c.jobs() if c.ARMS.index(j['arm']) > 2 or (j['arm'] == 'joint' and j['rank'] > 0)]


def plan_for(pristine):
    wanted = {c.key(j) for j in pristine}
    return {'source_sha256': c.SOURCE, 'freeze_sha256': c.FREEZES,
            'completed': [{**j, 'report_sha256': 'old'} for j in c.jobs() if c.key(j) not in wanted],
            'workers': c.allocation(pristine, receiving())}


def make_shard(path, job, proof='engineering'):
    path.mkdir(parents=True)
    task, arm, rank = c.key(job)
    launch = {'task': task, 'arm': arm, 'logical_ranks': [rank], 'expected_episodes': c.expected_rows(rank),
              'source_sha256': c.SOURCE, 'freeze_sha256': c.FREEZES[task],
              'engineering_only': False, 'fresh_confirmation': False, 'engineering_report_sha256': proof}
    c.write(path / 'protocol.json', launch)
    hashes = {}
    for row in c.expected_rows(rank):
        name = f"episode-{row['episode']:03d}.json"
        c.write(path / name, {**row, 'arm': arm, 'result': {'elementary_steps': 30}})
        hashes[name] = c.digest(path / name)
    report = {'status': 'complete_coupling_navigation_shard', 'episodes': 12, 'task': task, 'arm': arm,
              'source_sha256': c.SOURCE, 'freeze_sha256': c.FREEZES[task], 'parameters_unchanged': True,
              'fresh_confirmation': False, 'scientific_efficacy_measurement': True,
              'protocol_sha256': c.digest(path / 'protocol.json'), 'episode_files_sha256': hashes}
    c.write(path / 'report.json', report)
    c.write(path / 'DONE.json', {'report_sha256': c.digest(path / 'report.json')})


class RedistributionTests(unittest.TestCase):
    def test_47_stream_partition_is_exact_17_8_11_11_per_task(self):
        plan = plan_for(remaining()); c.validate_plan(plan)
        for name, count in {'ne0': 17, 'ne1': 17, 'in0': 16, 'tx0': 11, 'tx1': 11, 'tx2': 11, 'tx3': 11}.items():
            self.assertEqual(len(plan['workers'][name]['jobs']), count)
        for task in c.TASKS:
            self.assertEqual(sum(j['task'] == task for j in plan['workers']['in0']['jobs']), 8)

    def test_every_later_boundary_remains_exact_without_completed_reruns(self):
        original = c.jobs()
        for offset in range(65):
            pristine = [j for j in original if c.ARMS.index(j['arm']) * 8 + j['rank'] >= offset]
            c.validate_plan(plan_for(pristine))
        with self.assertRaises(ValueError): c.allocation([original[0], original[0]], receiving())

    def test_plan_rejects_missing_duplicate_and_wrong_device(self):
        for mutation in ('missing', 'duplicate', 'device'):
            plan = plan_for(remaining())
            if mutation == 'missing': plan['workers']['ne0']['jobs'].pop()
            elif mutation == 'duplicate': plan['workers']['ne0']['jobs'].append(plan['workers']['ne1']['jobs'][0])
            else: plan['workers']['tx0']['instance'] = 999
            with self.assertRaises(ValueError): c.validate_plan(plan)

    def test_stream_ids_are_contiguous_original_12_episode_rng_sequence(self):
        rows = [r for rank in range(8) for r in c.expected_rows(rank)]
        self.assertEqual([r['episode'] for r in rows], list(range(96)))
        self.assertEqual(len({r['environment_seed'] for r in rows}), 96)
        self.assertEqual(rows[12]['local_seed'], c.DEV_SEED + 6000)

    def test_shard_requires_complete_hash_bound_intact_stream_and_device_proof(self):
        job = c.jobs()[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'shard'; make_shard(path, job)
            self.assertEqual(c.verify_shard(path, job, 'engineering'), c.digest(path / 'report.json'))
            with self.assertRaises(ValueError): c.verify_shard(path, job, 'other GPU engineering')
            with self.assertRaises(ValueError): c.verify_shard(path, {**job, 'rank': 1})
            (path / 'episode-000.json').write_text('{}')
            with self.assertRaises(ValueError): c.verify_shard(path, job)

    def test_existing_partial_shard_is_never_pristine_or_automatically_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / c.relative(c.jobs()[0])).mkdir(parents=True)
            with self.assertRaises(FileNotFoundError): c.inventory(root)

    def test_pid_reuse_and_command_change_fail_closed(self):
        old = binding()
        for current in ({**old, 'starttime': 999}, {**old, 'command': ['unrelated']}):
            with self.assertRaises(ValueError): c.alive(old, lambda pid: current)
        self.assertFalse(c.alive(old, lambda pid: None))
        self.assertFalse(c.alive(old, lambda pid: {**old, 'state': 'Z', 'command': []}))

    def test_missing_review_authorization_cannot_pause_or_launch(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(c, 'CONTROL', Path(directory)), patch.object(c, 'send') as send:
            c.write(Path(directory) / 'READY.json', receiving()['50231985'])
            with self.assertRaises(FileNotFoundError): control.boundary()
            send.assert_not_called()

    def test_changed_ready_or_plan_authorization_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(c, 'CONTROL', Path(directory)):
            c.write(Path(directory) / 'RUN_AUTHORIZATION.json', {'owner': 'rep_geometry_transcoder/root', 'operation': 'run', 'plan_sha256': 'old'})
            with self.assertRaises(ValueError): c.require_authority('run', plan_hash='new')

    def test_uncommitted_boundary_failure_resumes_only_original_parents(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(c, 'CONTROL', Path(directory)):
            ready = receiving(); c.write(c.CONTROL / 'READY.json', ready['50231985']); c.write(c.CONTROL / 'ALL_READY.json', ready)
            parents = {p['parent']['pid']: p['parent'] for p in ready['50231985']['workers'].values()}
            calls = []
            def send(parent, sig):
                calls.append((parent['pid'], sig)); parents[parent['pid']]['state'] = 'T' if sig == signal.SIGSTOP else 'S'; return True
            with patch.object(c, 'require_authority'), patch.object(c, 'verify_source'), patch.object(c, 'gpu_uuid', return_value=UUID), \
                    patch.object(c, 'alive', return_value=True), patch.object(c, 'process', side_effect=lambda pid: parents[pid]), \
                    patch.object(c, 'send', side_effect=send), patch.object(control, 'children', return_value=[]), \
                    patch.object(c, 'gpu_processes', return_value=[]), patch.object(c, 'wait_gpu_release', return_value={'verified': True}), \
                    patch.object(c, 'inventory', side_effect=ValueError('partial shard')):
                with self.assertRaises(ValueError): control.boundary()
            self.assertEqual(calls, [(4253, signal.SIGSTOP), (4104, signal.SIGSTOP), (4253, signal.SIGCONT), (4104, signal.SIGCONT)])
            self.assertFalse((c.CONTROL / 'PLAN.json').exists())

    def test_terminal_child_nvml_lag_waits_for_two_empty_samples(self):
        child = binding(123)
        zombie = {**child, 'state': 'Z', 'command': []}
        query = Mock(side_effect=[[{'pid': 123, 'gpu_uuid': UUID}], [{'pid': 123, 'gpu_uuid': UUID}], [], []])
        times = iter([0, 0, .5, 1, 1.5])
        observations = []
        result = c.wait_gpu_release(0, UUID, [child], query=query, reader=lambda pid: zombie,
            clock=lambda: next(times), sleeper=Mock(), device=lambda gpu: UUID, observations=observations)
        self.assertEqual(result['stable_empty_samples'], 2)
        self.assertEqual(len(observations), 4)
        self.assertEqual(result['elapsed_seconds'], 1.5)

    def test_empty_nvml_does_not_skip_still_live_original_child(self):
        child = binding(123)
        reader = Mock(side_effect=[child, {**child, 'state': 'Z', 'command': []}, None])
        times = iter([0, 0, .5, 1])
        result = c.wait_gpu_release(0, UUID, [child], query=lambda gpu: [], reader=reader,
            clock=lambda: next(times), sleeper=Mock(), device=lambda gpu: UUID)
        self.assertFalse(result['observations'][0]['children_terminal'])
        self.assertEqual(result['elapsed_seconds'], 1)

    def test_unknown_live_gpu_process_fails_without_wait_or_signal(self):
        sleep = Mock(); observations = []
        with self.assertRaisesRegex(ValueError, 'Unknown live GPU process'):
            c.wait_gpu_release(0, UUID, query=lambda gpu: [{'pid': 888, 'gpu_uuid': UUID}],
                reader=lambda pid: binding(888), sleeper=sleep, device=lambda gpu: UUID, observations=observations)
        sleep.assert_not_called()
        self.assertEqual(observations[0]['nvml_processes'][0]['pid'], 888)

    def test_release_guard_pid_reuse_and_parent_resume_fail_closed(self):
        child = binding(123)
        with self.assertRaisesRegex(ValueError, 'PID reused'):
            c.wait_gpu_release(0, UUID, [child], query=lambda gpu: [],
                reader=lambda pid: {**child, 'starttime': 999}, device=lambda gpu: UUID)
        parent = binding(4253, 'T')
        with self.assertRaisesRegex(ValueError, 'parent resumed'):
            c.wait_gpu_release(0, UUID, paused_parent=parent, query=lambda gpu: [],
                reader=lambda pid: {**parent, 'state': 'S'}, device=lambda gpu: UUID)

    def test_release_deadline_preserves_stale_nvml_rows_for_diagnosis(self):
        times = iter([0, 0, .5, 1]); observations = []
        with self.assertRaises(TimeoutError):
            c.wait_gpu_release(0, UUID, timeout=1, query=lambda gpu: [{'pid': 123, 'gpu_uuid': UUID}],
                reader=lambda pid: None, clock=lambda: next(times), sleeper=Mock(), device=lambda gpu: UUID,
                observations=observations)
        self.assertEqual(len(observations), 3)
        self.assertIsNone(observations[-1]['nvml_processes'][0]['process_state'])

    def test_an_empty_interval_followed_by_nvml_lag_resets_stability(self):
        query = Mock(side_effect=[[], [{'pid': 123, 'gpu_uuid': UUID}], [], []])
        times = iter([0, 0, .5, 1, 1.5])
        result = c.wait_gpu_release(0, UUID, query=query, reader=lambda pid: None,
            clock=lambda: next(times), sleeper=Mock(), device=lambda gpu: UUID)
        self.assertEqual(len(result['observations']), 4)

    def test_early_engineering_binding_is_exact_task_device_source_and_ready(self):
        worker = receiving()['50205763']['workers']['in0']
        launch = {'status': 'early_original_wall_engineering', 'worker': 'in0', **worker,
            'source_sha256': c.SOURCE, 'freeze_sha256': c.FREEZES['wall'],
            'ready_sha256': 'ready', 'original_engineering_only': True}
        control.validate_early_binding(launch, worker, 'ready')
        for changes in ({'gpu': 1}, {'source_sha256': 'other'}, {'freeze_sha256': c.FREEZES['pointmaze']}, {'ready_sha256': 'stale'}):
            with self.assertRaises(ValueError): control.validate_early_binding({**launch, **changes}, worker, 'ready')

    def test_completed_early_suite_is_verified_and_reused_without_launch(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(c, 'CONTROL', Path(directory)):
            root = control.early_root(); root.mkdir(parents=True)
            worker = receiving()['50205763']['workers']['in0']
            launch = {'status': 'early_original_wall_engineering', 'worker': 'in0', **worker,
                'source_sha256': c.SOURCE, 'freeze_sha256': c.FREEZES['wall'], 'ready_sha256': 'ready',
                'original_engineering_only': True, 'identity': binding(999)}
            c.write(root / 'LAUNCH.json', launch)
            c.write(root / 'CHILD.json', {**binding(123), 'ppid': 999, 'command': control.early_command()})
            c.write(root / 'GPU_RELEASE.json', {'stable_empty_samples': 2})
            c.write(root / 'DONE.json', {'status': 'early_navigation_engineering_complete',
                'launch_sha256': c.digest(root / 'LAUNCH.json'), 'child_sha256': c.digest(root / 'CHILD.json'),
                'gpu_release_sha256': c.digest(root / 'GPU_RELEASE.json'), 'source_sha256': c.SOURCE,
                'freeze_sha256': c.FREEZES['wall'], 'gpu_uuid': UUID, 'engineering_report_sha256': 'proof'})
            with patch.object(control, 'wait_terminal'), patch.object(c, 'wait_gpu_release'), \
                    patch.object(control, 'engineering_proof', return_value='proof') as check, \
                    patch.object(control.subprocess, 'Popen') as launch_process:
                proof = control.wait_early_proof(worker, 'ready')
                self.assertEqual(proof['report_sha256'], 'proof')
                self.assertEqual(proof['path'], str(root / 'engineering'))
                check.assert_called_once(); launch_process.assert_not_called()

    def test_failed_early_suite_is_not_retried_or_accepted(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(c, 'CONTROL', Path(directory)):
            root = control.early_root(); root.mkdir(parents=True)
            worker = receiving()['50205763']['workers']['in0']
            c.write(root / 'LAUNCH.json', {'status': 'early_original_wall_engineering', 'worker': 'in0', **worker,
                'source_sha256': c.SOURCE, 'freeze_sha256': c.FREEZES['wall'], 'ready_sha256': 'ready',
                'original_engineering_only': True, 'identity': binding(999)})
            c.write(root / 'FAILED.json', {'error': 'preserved failure'})
            with patch.object(control.subprocess, 'Popen') as launch_process:
                with self.assertRaisesRegex(ValueError, 'failed'): control.wait_early_proof(worker, 'ready')
                launch_process.assert_not_called()

    def test_original_child_requires_entire_expected_argv(self):
        worker = receiving()['50231985']['workers']['ne0']
        with self.assertRaises(ValueError): control.validate_child({**binding(123), 'ppid': 4253, 'command': ['unknown']}, worker)

    def test_wait_terminal_does_not_treat_pid_reuse_as_completion(self):
        with patch.object(c, 'alive', side_effect=ValueError('PID reused')), patch.object(control.time, 'sleep') as sleep:
            with self.assertRaises(ValueError): control.wait_terminal(binding())
            sleep.assert_not_called()

    def test_droid_queue_requires_both_ranks_all_nine_arms_144_endpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            panel = Path(directory); queue = panel / 'queue-gpu2'; queue.mkdir(); (panel / 'freeze').mkdir()
            (panel / 'freeze/protocol.json').write_text('{}')
            done = {'completed_paired_streams': [2, 6], 'arms': ['native', *c.ARMS], 'episodes': 144}
            launch = {'pid': 1298, 'instance': 50259194, 'logical_ranks': [2, 6], 'device_uuid': UUID,
                      'source_sha256': c.DROID_SOURCE, 'episodes_per_arm_global': 64, 'episodes_per_arm_on_device': 16}
            c.write(queue / 'DONE.json', done); c.write(queue / 'LAUNCH.json', launch)
            with patch.object(c, 'DROID_FREEZE', c.digest(panel / 'freeze/protocol.json')):
                self.assertEqual(c.verify_droid_completion(panel, 2, {'pid': 1298, 'gpu_uuid': UUID}), [2, 6])
                for changes in ({'episodes': 128}, {'completed_paired_streams': [2]}, {'arms': list(c.ARMS)}):
                    with patch.object(c, 'read', side_effect=[{**done, **changes}, launch]):
                        with self.assertRaises(ValueError): c.verify_droid_completion(panel, 2, {'pid': 1298, 'gpu_uuid': UUID})

    def test_temporary_droid_gpu_gap_never_satisfies_whole_queue_gate(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(c, 'ROOT', Path(directory)), \
                patch.object(c, 'alive', return_value=False), patch.object(c, 'gpu_processes', return_value=[]) as gpu:
            with self.assertRaises(ValueError): control.droid_gate({'gpu': 0, 'predecessor': binding(), 'gpu_uuid': UUID})
            gpu.assert_not_called()

    def test_archive_path_traversal_and_unsafe_aliases_refused(self):
        for name in ('/root/secret', '../secret', 'payload/../outside', 'payload/._bad', 'payload/__pycache__/a'):
            with self.assertRaises(ValueError): c.safe_member(name)

    def test_manifest_mismatch_never_overwrites_existing_member(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'existing'; path.write_text('retained')
            with self.assertRaises(ValueError): c.verify_members(Path(directory), {'existing': {'bytes': 1, 'sha256': 'bad'}})
            self.assertEqual(path.read_text(), 'retained')

    def test_receiving_transfer_rejects_duplicate_member_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); data = b'fixed'; name = 'payload/code/a.py'
            manifest = {name: {'bytes': len(data), 'sha256': __import__('hashlib').sha256(data).hexdigest()}}
            stream = io.BytesIO()
            with tarfile.open(fileobj=stream, mode='w:gz') as archive:
                for filename, content in [('INPUTS.json', json.dumps(manifest).encode()), (name, data), (name, b'other')]:
                    info = tarfile.TarInfo(filename); info.size = len(content); archive.addfile(info, io.BytesIO(content))
            stream.seek(0)
            with patch.object(c, 'CONTROL', root), patch.object(c, 'PAYLOAD', root / 'payload'), patch.object(sys, 'stdin', type('Input', (), {'buffer': stream})()):
                with self.assertRaises(ValueError): control.receive_inputs()
            self.assertEqual((root / name).read_bytes(), data)

    def test_real_owned_cpu_process_lifecycle_uses_pidfd_on_receiving_linux(self):
        program = 'import time; time.sleep(10)'
        child = subprocess.Popen([sys.executable, '-c', program])
        try:
            if sys.platform == 'linux':
                owned = c.started(child.pid, [sys.executable, '-c', program])
                self.assertTrue(c.send(owned, signal.SIGSTOP)); self.assertTrue(c.send(owned, signal.SIGCONT))
                self.assertTrue(c.send(owned, signal.SIGTERM))
            else:
                os.kill(child.pid, signal.SIGSTOP); os.kill(child.pid, signal.SIGCONT); child.terminate()
            child.wait(timeout=5)
        finally:
            if child.poll() is None:
                os.kill(child.pid, signal.SIGCONT); child.terminate(); child.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
