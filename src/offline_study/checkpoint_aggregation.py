"""Fixed-window numerical summaries, not raw-evidence or training verification.

The simulation primary window is epochs 41..50 for reproduction training seeds
234/235/236. Its separate final-checkpoint mode uses epoch 50 only. DROID uses
the registered 18 native-cadence checkpoints 217,223,...,313,315.

Match the pinned released plotting code: concatenate each seed's checkpoint
scores, then NumPy mean and population standard deviation (ddof=0). This SD is
not a confidence interval; 30 simulation checkpoint evaluations are not 30
independent training seeds. No task, condition, epoch or planner selection occurs.

Callers MUST first validate complete raw evaluation receipts, exact scenario and
planner bindings, checkpoint/training histories, and authorized data access. Hash
strings and counts here are structural bindings, not proof their contents exist.
This module performs no filesystem/network/model operations and has no CLI.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from numbers import Integral, Real
import re

import numpy as np

TRAINING_SEEDS = (234, 235, 236)
SIMULATION_TASKS = ('mw-reach', 'mw-reach-wall', 'pusht', 'pointmaze', 'wall')
SIMULATION_LATE_EPOCHS = tuple(range(41, 51))
SIMULATION_FINAL_EPOCHS = (50,)
DROID_LATE_EPOCHS = (*range(217, 314, 6), 315)
UPSTREAM_COMMIT = '13cf1d9c7e476f53c17714d2e0f1dc239a883ce0'
UPSTREAM_SOURCE = 'app/plan_common/plot/logs_plan_joint_per_design_choice.py'
UPSTREAM_SOURCE_SHA256 = '2c2aa806022bc5c0c5b2f8a63027873778ebc88c341e193d36c2f19ed5686328'
BASE_FIELDS = frozenset(('task', 'condition', 'training_seed', 'epoch',
    'checkpoint_sha256', 'evaluation_receipt_sha256', 'episode_count'))


def _integer(value, label):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(label + ' must be an integer, not a coercible value')
    return int(value)


def _hash(value, label):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError(label + ' must be an explicit lowercase SHA256')
    return value


def _number(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(label + ' must be finite real numerical data')
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(label + ' must be representable as finite float64')
    return value


def aggregate_checkpoints(rows: Iterable[Mapping], *, task: str, condition: str,
                          mode: str = 'late') -> dict:
    """Summarize exactly one task/condition and its complete fixed seed/epoch grid.

    Each row has exactly task, condition, training_seed, epoch, checkpoint_sha256,
    evaluation_receipt_sha256, episode_count and one metric. Simulation ``score``
    is the native SR in percentage points (0..100), not a success fraction. DROID
    ``mean_xyz_error`` is the COMPLETE 64-episode evaluation mean: transform it
    to max(0,800*(.1-error)) once per evaluation BEFORE aggregating checkpoints.
    DROID never accepts a pre-clipped score or a list of episode scores/errors.

    Every seed has all ten simulation late checkpoints (96 episodes each), or
    all 18 DROID late checkpoints (64 each). ``mode='final'`` is separate and
    simulation-only: exactly one epoch-50 evaluation per training seed. Inputs
    outside the selected fixed window are errors, not silently filtered data.

    The primary mean/SD is over seed-major, epoch-minor float64 checkpoint scores.
    Secondary SDs of three seed means and of epoch-wise means across the three
    seeds are labelled separately; none is an inferential CI or standard error.
    Distinct checkpoint and evaluation receipt hashes are required within this
    one scope; they do not themselves establish independent training histories.
    """
    if not isinstance(task, str) or task not in (*SIMULATION_TASKS, 'droid'):
        raise ValueError('Require one registered task; task pooling is not supported')
    if not isinstance(condition, str) or re.fullmatch('[A-Za-z0-9_.-]+', condition) is None:
        raise ValueError('Require one explicit condition identifier')
    if mode not in ('late', 'final'):
        raise ValueError('Only fixed late or separate final mode is supported')
    if task == 'droid':
        if mode != 'late':
            raise ValueError('DROID requires the registered 18-checkpoint late window')
        epochs, count, metric = DROID_LATE_EPOCHS, 64, 'mean_xyz_error'
    else:
        epochs = SIMULATION_LATE_EPOCHS if mode == 'late' else SIMULATION_FINAL_EPOCHS
        count, metric = 96, 'score'

    expected = {(seed, epoch) for seed in TRAINING_SEEDS for epoch in epochs}
    indexed, checkpoint_hashes, receipt_hashes = {}, set(), set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != BASE_FIELDS | {metric}:
            raise ValueError('Require exact evaluation-row fields and the task-specific metric')
        if row['task'] != task or row['condition'] != condition:
            raise ValueError('Mixed task/condition rows cannot be pooled or silently dropped')
        seed, epoch = _integer(row['training_seed'], 'training_seed'), _integer(row['epoch'], 'epoch')
        key = (seed, epoch)
        if key not in expected:
            raise ValueError('Training seed or epoch is outside the fixed aggregation grid')
        if key in indexed:
            raise ValueError('Duplicate seed/checkpoint evaluation, even if values agree')
        if _integer(row['episode_count'], 'episode_count') != count:
            raise ValueError('Every checkpoint/condition requires its complete fixed episode count')
        checkpoint = _hash(row['checkpoint_sha256'], 'checkpoint_sha256')
        receipt = _hash(row['evaluation_receipt_sha256'], 'evaluation_receipt_sha256')
        if checkpoint in checkpoint_hashes or receipt in receipt_hashes:
            raise ValueError('A checkpoint or evaluation receipt cannot stand in for another grid cell')
        value = _number(row[metric], metric)
        if value < 0 or (metric == 'score' and value > 100):
            raise ValueError('Require nonnegative XYZ error or simulation SR percentage in [0,100]')
        score = value if metric == 'score' else max(0.0, 800 * (0.1 - value))
        indexed[key] = {**row, 'training_seed': seed, 'epoch': epoch,
                        'episode_count': count, metric: value, 'score': score}
        checkpoint_hashes.add(checkpoint); receipt_hashes.add(receipt)
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        raise ValueError('Incomplete numerical grid; missing seed/epoch evaluations: ' + repr(missing))

    ordered = [indexed[(seed, epoch)] for seed in TRAINING_SEEDS for epoch in epochs]
    scores = np.asarray([row['score'] for row in ordered], dtype=np.float64)
    matrix = scores.reshape(len(TRAINING_SEEDS), len(epochs))
    seed_means, epoch_means = np.mean(matrix, axis=1), np.mean(matrix, axis=0)
    return {
        'schema_version': 1, 'status': 'complete_fixed_numerical_grid_only',
        'task': task, 'condition': condition, 'mode': mode,
        'training_seeds': list(TRAINING_SEEDS), 'epochs': list(epochs),
        'training_seed_count': len(TRAINING_SEEDS), 'checkpoint_evaluation_count': len(ordered),
        'episodes_per_checkpoint_condition': count,
        'repeated_episode_evaluation_count': count * len(ordered),
        'score_units': 'recorded_plan_action_score_not_robot_success' if task == 'droid' else 'success_percentage_points',
        'mean': float(np.mean(scores)),
        'released_code_population_sd': float(np.std(scores, ddof=0)),
        'per_seed_means_population_sd': float(np.std(seed_means, ddof=0)),
        'per_epoch_seed_means_population_sd': float(np.std(epoch_means, ddof=0)),
        'sd_definitions': {
            'released_code_population_sd': 'SD of concatenated per-training-seed checkpoint scores',
            'per_seed_means_population_sd': 'SD across three training-seed means over the fixed epochs',
            'per_epoch_seed_means_population_sd': 'SD across epochs after averaging the three seeds at each epoch',
            'ddof': 0, 'confidence_interval': False, 'standard_error': False},
        'per_seed': [{'training_seed': seed, 'checkpoint_count': len(epochs),
            'mean': float(seed_means[i]), 'within_seed_checkpoint_population_sd': float(np.std(matrix[i], ddof=0))}
            for i, seed in enumerate(TRAINING_SEEDS)],
        'per_epoch': [{'epoch': epoch, 'training_seed_count': len(TRAINING_SEEDS),
            'mean_across_training_seeds': float(epoch_means[i]),
            'across_training_seeds_population_sd': float(np.std(matrix[:, i], ddof=0))}
            for i, epoch in enumerate(epochs)],
        'per_checkpoint': ordered,
        'method_reference': {'vendor_commit': UPSTREAM_COMMIT, 'source': UPSTREAM_SOURCE,
            'source_sha256': UPSTREAM_SOURCE_SHA256, 'function': 'aggregate_task_data_by_groups',
            'best_epoch_or_planner_selection': False, 'task_pooling': False,
            'window_basis': ('released_code_sparse_droid_window' if task == 'droid' else
                'paper_simulation_last_ten' if mode == 'late' else 'separate_simulation_final_checkpoint')},
        'claim_limits': {'raw_provenance_verification_performed': False,
            'caller_must_verify_raw_receipts_scenarios_and_training_histories': True,
            'checkpoint_evaluations_are_independent_training_seeds': False,
            'repeated_episodes_are_independent_population_units': False,
            'authors_exact_training_seed_triplet_claimed': False,
            'completed_study_claimed': False, 'confidence_interval_computed': False},
    }
