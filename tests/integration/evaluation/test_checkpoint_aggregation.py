"""Numerical fixtures only; no real evaluation results or training are accessed."""
import ast
from collections import defaultdict
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import unittest

import numpy as np

from offline_study.evaluation import checkpoint_aggregation as a


def digest(label):
    return hashlib.sha256(label.encode()).hexdigest()


def make_rows(task='wall', condition='native', mode='late'):
    epochs = a.DROID_LATE_EPOCHS if task == 'droid' else a.SIMULATION_LATE_EPOCHS if mode == 'late' else (50,)
    rows = []
    for i, seed in enumerate(a.TRAINING_SEEDS):
        for j, epoch in enumerate(epochs):
            row = {'task': task, 'condition': condition, 'training_seed': seed, 'epoch': epoch,
                'checkpoint_sha256': digest(f'checkpoint-{task}-{seed}-{epoch}'),
                'evaluation_receipt_sha256': digest(f'evaluation-{task}-{condition}-{seed}-{epoch}'),
                'episode_count': 64 if task == 'droid' else 96}
            row['mean_xyz_error' if task == 'droid' else 'score'] = (.002 * (18 * i + j) if task == 'droid' else float(10 * i + j))
            rows.append(row)
    return rows


class Series(np.ndarray):
    """Small dataframe-compatible fixture; NOT a replacement production reader.

    Pandas is not a project dependency. NumPy's ndarray subclass supplies the
    precise column/iloc/values operations used by the unmodified upstream AST.
    If Pandas is available, parity tests use its actual DataFrame instead.
    """
    @property
    def iloc(self):
        return self

    @property
    def values(self):
        return np.asarray(self)

    def isnull(self):
        return np.isnan(np.asarray(self))


class Frame(dict):
    def __init__(self, columns):
        super().__init__({key: np.asarray(value).view(Series) for key, value in columns.items()})

    @property
    def columns(self):
        return list(self)


def frame(columns):
    try:
        import pandas as pd
    except ModuleNotFoundError as error:
        if error.name != 'pandas':
            raise
        return Frame(columns)
    return pd.DataFrame(columns)


class CheckpointAggregationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_path = Path(__file__).resolve().parents[3] / 'vendor/jepa-wms' / a.UPSTREAM_SOURCE
        cls.source = cls.source_path.read_bytes()
        tree = ast.parse(cls.source)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'aggregate_task_data_by_groups')
        namespace = {'np': np, 'os': os, 'defaultdict': defaultdict}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(cls.source_path), 'exec'), namespace)
        cls.native = staticmethod(namespace[function.name])

    def upstream(self, rows, task, *, final=False):
        data = {}
        for seed in a.TRAINING_SEEDS:
            chosen = sorted((row for row in rows if row['training_seed'] == seed), key=lambda row: row['epoch'])
            columns = {'epoch': [row['epoch'] for row in chosen]}
            columns['Act_err_xyz' if task == 'droid' else 'SR'] = [row['mean_xyz_error' if task == 'droid' else 'score'] for row in chosen]
            data[('/frozen-model', task, 'fixed-planner-no-selection', seed)] = frame(columns)
        with contextlib.redirect_stdout(io.StringIO()):
            result = self.native(data, {'fixed-condition': ['/frozen-model']}, {task: 'single-task'},
                last_n_epochs={task: 1 if final else 18 if task == 'droid' else 10},
                start_from_epoch={task: 215} if task == 'droid' else {},
                use_computed_best_eval_setup=False, filter_best_eval_setup=False)
        # Ignore the released function's extra cross-task Avg: we never emit it.
        return result['single-task']['fixed-condition']['fixed-planner-no-selection']

    def test_source_is_exact_pinned_bytes_and_simulation_percentage_conversion(self):
        self.assertEqual(hashlib.sha256(self.source).hexdigest(), a.UPSTREAM_SOURCE_SHA256)
        tree = ast.parse(self.source)
        loader = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_task_data')
        mapping = next(n for n in loader.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'column_mapping' for t in n.targets))
        namespace = {}; exec(compile(ast.Module(body=[mapping], type_ignores=[]), str(self.source_path), 'exec'), namespace)
        label, convert = namespace['column_mapping']['episode_success']
        self.assertEqual(label, 'SR'); self.assertEqual(convert(.625), 62.5)

    def test_exact_upstream_ast_mean_and_population_sd_all_six_tasks(self):
        for task in (*a.SIMULATION_TASKS, 'droid'):
            with self.subTest(task=task):
                rows = make_rows(task); result = a.aggregate_checkpoints(rows, task=task, condition='native')
                mean, std, count = self.upstream(rows, task)
                self.assertEqual(result['mean'], mean)
                self.assertEqual(result['released_code_population_sd'], std)
                self.assertEqual(result['checkpoint_evaluation_count'], count)

    def test_exact_upstream_ast_final_checkpoint_mode(self):
        rows = make_rows(mode='final')
        result = a.aggregate_checkpoints(rows, task='wall', condition='native', mode='final')
        mean, std, count = self.upstream(rows, 'wall', final=True)
        self.assertEqual((result['mean'], result['released_code_population_sd'], result['checkpoint_evaluation_count']), (mean, std, count))
        self.assertEqual(result['epochs'], [50]); self.assertEqual(count, 3)

    def test_fixed_grid_counts_and_droid_metadata_cadence(self):
        self.assertEqual(a.TRAINING_SEEDS, (234, 235, 236))
        self.assertEqual(a.SIMULATION_LATE_EPOCHS, tuple(range(41, 51)))
        self.assertEqual(a.DROID_LATE_EPOCHS, (217, 223, 229, 235, 241, 247, 253, 259, 265, 271, 277, 283, 289, 295, 301, 307, 313, 315))
        for task, evaluations, episodes in [('wall', 30, 2880), ('droid', 54, 3456)]:
            result = a.aggregate_checkpoints(make_rows(task), task=task, condition='native')
            self.assertEqual(result['checkpoint_evaluation_count'], evaluations)
            self.assertEqual(result['repeated_episode_evaluation_count'], episodes)
            self.assertEqual(result['training_seed_count'], 3)

    def test_input_order_does_not_change_statistics_or_bound_row_order(self):
        rows = make_rows()
        forward = a.aggregate_checkpoints(rows, task='wall', condition='native')
        reverse = a.aggregate_checkpoints(reversed(rows), task='wall', condition='native')
        self.assertEqual(forward, reverse)
        self.assertEqual([(r['training_seed'], r['epoch']) for r in forward['per_checkpoint']],
                         [(s, e) for s in a.TRAINING_SEEDS for e in a.SIMULATION_LATE_EPOCHS])

    def test_checkpoint_sd_seed_mean_sd_and_epoch_mean_sd_are_distinct_not_ci(self):
        rows = make_rows(); result = a.aggregate_checkpoints(rows, task='wall', condition='native')
        scores = np.asarray([r['score'] for r in rows]).reshape(3, 10)
        self.assertEqual(result['released_code_population_sd'], np.std(scores.reshape(-1), ddof=0))
        self.assertEqual(result['per_seed_means_population_sd'], np.std(np.mean(scores, axis=1), ddof=0))
        self.assertEqual(result['per_epoch_seed_means_population_sd'], np.std(np.mean(scores, axis=0), ddof=0))
        self.assertEqual(len({result[k] for k in ('released_code_population_sd', 'per_seed_means_population_sd', 'per_epoch_seed_means_population_sd')}), 3)
        self.assertNotEqual(result['released_code_population_sd'], np.std(scores, ddof=1))
        self.assertNotEqual(result['released_code_population_sd'], np.std(scores) / np.sqrt(scores.size))
        self.assertFalse(result['sd_definitions']['confidence_interval'])

    def test_droid_clips_evaluation_mean_not_individual_episodes(self):
        rows = make_rows('droid')
        episode_errors = np.array([0., .11] * 32)
        for row in rows: row['mean_xyz_error'] = float(np.mean(episode_errors))
        result = a.aggregate_checkpoints(rows, task='droid', condition='native')
        self.assertAlmostEqual(result['mean'], 36.)
        wrong_episode_clipping = np.mean(np.maximum(0, 800 * (.1 - episode_errors)))
        self.assertEqual(wrong_episode_clipping, 40.)
        self.assertNotEqual(result['mean'], wrong_episode_clipping)
        self.assertEqual(result['score_units'], 'recorded_plan_action_score_not_robot_success')

    def test_droid_transforms_before_averaging_checkpoints_not_after(self):
        rows = make_rows('droid')
        for index, row in enumerate(rows): row['mean_xyz_error'] = 0. if index % 2 == 0 else .11
        result = a.aggregate_checkpoints(rows, task='droid', condition='native')
        self.assertEqual(result['mean'], 40.)
        self.assertAlmostEqual(max(0., 800 * (.1 - np.mean([r['mean_xyz_error'] for r in rows]))), 36.)
        self.assertEqual((result['mean'], result['released_code_population_sd']), self.upstream(rows, 'droid')[:2])

    def test_droid_score_endpoints_and_error_preservation(self):
        for error, expected in [(0., 80.), (.1, 0.), (.5, 0.)]:
            rows = make_rows('droid')
            for row in rows: row['mean_xyz_error'] = error
            result = a.aggregate_checkpoints(rows, task='droid', condition='native')
            self.assertEqual(result['mean'], expected)
            self.assertEqual(result['per_checkpoint'][0]['mean_xyz_error'], error)

    def test_missing_rows_and_empty_input_never_return_complete(self):
        rows = make_rows()
        for partial in ([], rows[:-1], rows[10:], [r for r in rows if r['epoch'] != 45]):
            with self.assertRaisesRegex(ValueError, 'Incomplete numerical grid'):
                a.aggregate_checkpoints(partial, task='wall', condition='native')

    def test_duplicate_even_identical_row_is_rejected(self):
        rows = make_rows()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            a.aggregate_checkpoints(rows + [copy.deepcopy(rows[0])], task='wall', condition='native')

    def test_unexpected_seed_and_off_window_rows_not_silently_filtered(self):
        for key, value in [('training_seed', 233), ('training_seed', 237), ('epoch', 40), ('epoch', 51)]:
            rows = make_rows(); rows[0][key] = value
            with self.assertRaisesRegex(ValueError, 'outside the fixed'):
                a.aggregate_checkpoints(rows, task='wall', condition='native')
        for epoch in (215, 216, 218, 314, 316):
            rows = make_rows('droid'); rows[0]['epoch'] = epoch
            with self.assertRaises(ValueError): a.aggregate_checkpoints(rows, task='droid', condition='native')

    def test_mixed_conditions_tasks_and_cross_condition_counts_refused(self):
        for key, value in [('condition', 'candidate'), ('task', 'pusht'), ('episode_count', 95), ('episode_count', 64)]:
            rows = make_rows(); rows[3][key] = value
            with self.assertRaises(ValueError): a.aggregate_checkpoints(rows, task='wall', condition='native')
        rows = make_rows('droid'); rows[0]['episode_count'] = 96
        with self.assertRaises(ValueError): a.aggregate_checkpoints(rows, task='droid', condition='native')

    def test_repeated_checkpoint_or_receipt_cannot_fake_histories(self):
        for key in ('checkpoint_sha256', 'evaluation_receipt_sha256'):
            rows = make_rows(); rows[-1][key] = rows[0][key]
            with self.assertRaisesRegex(ValueError, 'stand in for another'):
                a.aggregate_checkpoints(rows, task='wall', condition='native')

    def test_hashes_are_required_structural_bindings(self):
        for value in ('', 'x' * 64, 'A' * 64, 'a' * 63, 123, None):
            rows = make_rows(); rows[0]['evaluation_receipt_sha256'] = value
            with self.assertRaises(ValueError): a.aggregate_checkpoints(rows, task='wall', condition='native')

    def test_metric_nonfinite_negative_boolean_string_or_array_rejected(self):
        for task in ('wall', 'droid'):
            for value in (float('nan'), float('inf'), -float('inf'), -.001, True, np.bool_(True), '1.0', None, [1.], np.array([1.])):
                with self.subTest(task=task, value=repr(value)):
                    rows = make_rows(task); rows[0]['mean_xyz_error' if task == 'droid' else 'score'] = value
                    with self.assertRaises(ValueError): a.aggregate_checkpoints(rows, task=task, condition='native')
        rows = make_rows(); rows[0]['score'] = 100.001
        with self.assertRaises(ValueError): a.aggregate_checkpoints(rows, task='wall', condition='native')

    def test_integral_metadata_cannot_be_coerced_from_float_string_or_boolean(self):
        for key in ('training_seed', 'epoch', 'episode_count'):
            for bad in (True, '234', 234.0):
                rows = make_rows(); rows[0][key] = bad
                with self.assertRaises(ValueError): a.aggregate_checkpoints(rows, task='wall', condition='native')

    def test_exact_metric_schema_prevents_preclipped_droid_or_extra_planner_rows(self):
        rows = make_rows('droid'); rows[0]['score'] = 36.
        with self.assertRaises(ValueError): a.aggregate_checkpoints(rows, task='droid', condition='native')
        rows = make_rows(); rows[0]['planner'] = 'chosen-best'
        with self.assertRaises(ValueError): a.aggregate_checkpoints(rows, task='wall', condition='native')
        rows = make_rows(); del rows[0]['evaluation_receipt_sha256']
        with self.assertRaises(ValueError): a.aggregate_checkpoints(rows, task='wall', condition='native')

    def test_final_mode_cannot_select_last_or_best_from_partial_or_late_input(self):
        for rows in (make_rows(), make_rows(mode='final')[:-1]):
            with self.assertRaises(ValueError): a.aggregate_checkpoints(rows, task='wall', condition='native', mode='final')
        rows = make_rows(mode='final'); rows[0]['epoch'] = 49
        with self.assertRaises(ValueError): a.aggregate_checkpoints(rows, task='wall', condition='native', mode='final')
        with self.assertRaises(ValueError): a.aggregate_checkpoints(make_rows('droid'), task='droid', condition='native', mode='final')

    def test_no_arbitrary_window_seed_task_or_best_mode_override(self):
        for mode in ('best', 'partial', 'last-five'):
            with self.assertRaises(ValueError): a.aggregate_checkpoints(make_rows(), task='wall', condition='native', mode=mode)
        for task in ('robocasa', 'all', ['wall', 'pointmaze']):
            with self.assertRaises(ValueError): a.aggregate_checkpoints(make_rows(), task=task, condition='native')
        with self.assertRaises(TypeError): a.aggregate_checkpoints(make_rows(), task='wall', condition='native', epochs=[50])

    def test_json_output_preserves_metadata_without_mutating_input_or_claiming_provenance(self):
        rows = make_rows(); original = copy.deepcopy(rows)
        result = a.aggregate_checkpoints(rows, task='wall', condition='native')
        self.assertEqual(rows, original); json.dumps(result, allow_nan=False)
        self.assertEqual(result['per_checkpoint'], original)
        self.assertEqual(result['status'], 'complete_fixed_numerical_grid_only')
        self.assertTrue(result['claim_limits']['caller_must_verify_raw_receipts_scenarios_and_training_histories'])
        for name in ('raw_provenance_verification_performed', 'checkpoint_evaluations_are_independent_training_seeds',
                     'completed_study_claimed', 'confidence_interval_computed'):
            self.assertFalse(result['claim_limits'][name])
        self.assertNotIn('Avg', result)


if __name__ == '__main__': unittest.main()
