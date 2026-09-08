"""Actual source packaging and real navigation handoff schema, CPU-only."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts/vast'))
import corrected_pointmaze_prepare as prepare
import navigation_redistribution_common as nav


def publish(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def nav_fixture(root, navigation_control=prepare.NAVIGATION):
    files = {'navigation_redistribution_control.py': {'bytes': 1, 'sha256': 'a' * 64}}
    publish(root / 'FILES.json', files)
    ready = {'instance': 50259194, 'files_sha256': prepare.q.digest(root / 'FILES.json')}
    publish(root / 'READY-50259194.json', ready)
    workers = {}
    for name, (instance, gpu, _) in nav.WORKERS.items():
        uuid = prepare.GPU_UUIDS[gpu] if name in ('tx0', 'tx1', 'tx2') else 'GPU-23f2a085-92f0-239e-c7ec-c67c11f2f571'
        workers[name] = {'instance': instance, 'gpu': gpu, 'gpu_uuid': uuid, 'jobs': []}
    for index, job in enumerate(nav.jobs()):
        workers[['tx0', 'tx1', 'tx2'][index % 3]]['jobs'].append(job)
    plan = {'status': 'intact_navigation_redistribution_frozen', 'source_sha256': prepare.NAV_SOURCE,
        'freeze_sha256': prepare.FREEZES, 'workers': workers, 'completed': [],
        'ready_sha256': {'50259194': prepare.q.digest(root / 'READY-50259194.json')},
        'total_candidate_streams': 128, 'episodes_per_task_condition': 96,
        'outcome_based_selection': False, 'old_children_completed_untouched': True}
    publish(root / 'PLAN.json', plan)
    plan_hash = prepare.q.digest(root / 'PLAN.json')
    publish(root / 'CUTOVER.json', {'plan_sha256': plan_hash, 'old_parents_terminal': True,
        'old_watcher_terminal': True, 'old_children_not_interrupted': True})
    for gpu in range(3):
        name = 'tx' + str(gpu)
        worker = workers[name]
        identity = {'pid': 8000 + gpu, 'starttime': 5555 + gpu, 'state': 'S', 'ppid': 1,
            'command': ['/usr/bin/python3', '-u', str(navigation_control / 'navigation_redistribution_control.py'), 'run', '--worker', name]}
        publish(root / 'workers' / name / 'LAUNCH.json', {'worker': name, 'identity': identity,
            'plan_sha256': plan_hash, **worker})
    return plan


class CorrectedPointMazePreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.members, cls.source_hash = prepare.frozen_source(REPO)

    def test_exact_pinned_scientific_source_retains_corrected_native_sampler(self):
        self.assertIn(b"shuffle=SAMPLER_POLICY['train_shuffle']", self.members['src/offline_study/pointmaze_training_history.py'])
        self.assertIn(b"historical extra-shuffled checkpoints cannot resume", self.members['src/offline_study/pointmaze_training_history.py'])
        self.assertIn('tests/test_native_training_sampler.py', self.members)
        self.assertEqual(len(self.source_hash), 64)
        self.assertNotIn('scripts/vast/pointmaze_seed236_worker.py', self.members)

    def test_input_bindings_keep_data_not_old_pilot_or_checkpoint(self):
        files = prepare.pinned_input_files(REPO / 'artifacts/offline_study/pointmaze-tx-resume-20260908-v1')
        self.assertIn('/workspace/jepa-runtime/pointmaze-training-inputs-20260908-v1/files.json', files)
        self.assertIn('/workspace/jepa-runtime/navigation-input-check-20260907-v1/report.json', files)
        self.assertFalse(any('pilot' in path or 'training-history' in path or 'history-code' in path for path in files))

    def make_preparation(self, output):
        with patch.object(prepare, 'frozen_source', return_value=(dict(self.members), self.source_hash)):
            return prepare.prepare(REPO, REPO / 'artifacts/offline_study/pointmaze-tx-resume-20260908-v1', output)

    def test_preparation_is_not_runnable_and_does_not_depend_on_cache_pilot(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'prepared'
            proposal = self.make_preparation(root)
            self.assertNotIn('assignments', proposal)
            self.assertFalse((root / 'PLAN.json').exists())
            approval = prepare.q.read(root / 'EXECUTION_APPROVAL.json')
            self.assertFalse(approval['cache_pilot_required'])
            self.assertEqual(approval['execution'], 'native_accumulation_uncached')
            self.assertNotIn('profiling_receipt', approval)
            with tarfile.open(root / 'source.tar.gz') as archive:
                self.assertEqual(archive.extractfile('code/src/offline_study/pointmaze_training_history.py').read(),
                    self.members['src/offline_study/pointmaze_training_history.py'])
                self.assertTrue(archive.getmember('code/vendor/jepa-wms').issym())
            with self.assertRaises(FileExistsError):
                self.make_preparation(root)

    def test_missing_actual_navigation_receipts_cannot_create_runnable_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_preparation(root / 'prepared')
            with self.assertRaises(FileNotFoundError):
                prepare.finalize(root / 'prepared', root / 'missing-navigation', root / 'final')
            self.assertFalse((root / 'final').exists())

    def test_finalize_binds_actual_full_assignment_counts_and_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_preparation(root / 'prepared')
            native = nav_fixture(root / 'navigation')
            plan = prepare.finalize(root / 'prepared', root / 'navigation', root / 'final')
            for gpu, assignment in enumerate(plan['assignments']):
                name = 'tx' + str(gpu)
                binding = assignment['predecessor']
                self.assertEqual(assignment['seed'], 234 + gpu)
                self.assertEqual(binding['pid'], 8000 + gpu)
                self.assertEqual(binding['start_ticks'], 5555 + gpu)
                self.assertEqual(binding['verifier']['args'], ['verify-completed', '--worker', name])
                self.assertEqual(binding['expected_done']['endpoints'], len(native['workers'][name]['jobs']) * 12)
                self.assertTrue(binding['verifier']['expected_result']['scientific_validators_unchanged'])
                self.assertNotIn('performance_approval', plan)
            self.assertFalse(prepare.q.read(root / 'final/FINALIZED.json')['remote_staging_or_activation_performed'])

    def test_incomplete_partition_or_false_cutover_cannot_finalize(self):
        for defect in ('missing-stream', 'cutover-not-terminal', 'old-droid-pid'):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                self.make_preparation(root / 'prepared')
                nav_fixture(root / 'navigation')
                if defect == 'missing-stream':
                    path = root / 'navigation/PLAN.json'
                    value = prepare.q.read(path)
                    value['workers']['tx0']['jobs'].pop()
                elif defect == 'cutover-not-terminal':
                    path = root / 'navigation/CUTOVER.json'
                    value = prepare.q.read(path)
                    value['old_parents_terminal'] = False
                else:
                    path = root / 'navigation/workers/tx0/LAUNCH.json'
                    value = prepare.q.read(path)
                    value['identity']['command'] = ['/python', '/droid-queue.py']
                publish(path, value)
                with self.assertRaises(ValueError):
                    prepare.finalize(root / 'prepared', root / 'navigation', root / 'final')
                self.assertFalse((root / 'final').exists())

    def test_reviewed_v2_paths_are_bound_from_real_launches_not_assumed_v1(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_preparation(root / 'prepared')
            control = prepare.REVIEWED_NAVIGATION_ROOTS[1]
            nav_fixture(root / 'navigation', control)
            plan = prepare.finalize(root / 'prepared', root / 'navigation', root / 'final')
            self.assertIn(str(control / 'PLAN.json'), plan['fixed_files'])
            self.assertNotIn(str(prepare.NAVIGATION / 'PLAN.json'), plan['fixed_files'])
            for assignment in plan['assignments']:
                self.assertTrue(assignment['predecessor']['done'].startswith(str(control) + '/'))
                self.assertEqual(assignment['predecessor']['verifier']['script'], str(control / 'navigation_redistribution_control.py'))


if __name__ == '__main__':
    unittest.main()
