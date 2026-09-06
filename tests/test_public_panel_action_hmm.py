from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "cgs_pilot"))

from public_panel_action_hmm import (  # noqa: E402
    control_features,
    fit_action_conditioned_hmm,
    hmm_filter_conditioned,
    hmm_forward_backward_conditioned,
    model_features,
    transition_matrices,
)


def _controlled_sequences(seed: int = 0, n: int = 18) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    sequences, actions = [], []
    for _ in range(n):
        state = int(rng.integers(0, 2))
        hidden, control = [], []
        for _time in range(9):
            action = 1.0 if rng.random() > 0.5 else -1.0
            control.append([action])
            if action > 0:
                state = 1
            else:
                state = 0
            # Panel-P capture stores the selected-rollout endpoint, so this
            # chunk leads into the hidden row with the same index.
            hidden.append(rng.normal(3.0 if state else -3.0, 0.15, size=2))
        sequences.append(np.asarray(hidden))
        actions.append(np.asarray(control))
    return sequences, actions


def test_action_transition_matrices_are_row_stochastic_and_control_dependent() -> None:
    coef = np.zeros((2, 2, 2))
    coef[0, :, 1] = [-3.0, 3.0]
    coef[1, :, 1] = [-3.0, 3.0]
    feature = np.asarray([[1.0, -1.0], [1.0, 1.0]])
    transition = transition_matrices(feature, coef)
    np.testing.assert_allclose(transition.sum(2), 1.0)
    assert transition[0, 0, 0] > 0.99
    assert transition[1, 0, 1] > 0.99


def test_action_conditioned_filter_is_prefix_causal() -> None:
    sequences, actions = _controlled_sequences(n=12)
    model = fit_action_conditioned_hmm(sequences, actions, 2, seed=3, iterations=8)
    first = sequences[0].copy()
    second = first.copy()
    second[5:] *= -1.0
    feature = model_features(first, actions[0], model)
    np.testing.assert_allclose(
        hmm_filter_conditioned(first, feature, model)[:5],
        hmm_filter_conditioned(second, feature, model)[:5],
        atol=1e-10,
    )


def test_fitted_action_hmm_scores_bound_controls_above_shifted_controls() -> None:
    sequences, actions = _controlled_sequences(n=20)
    model = fit_action_conditioned_hmm(sequences[:16], actions[:16], 2, seed=9, iterations=12)
    correct, shifted = [], []
    for sequence, action in zip(sequences[16:], actions[16:]):
        correct_feature = model_features(sequence, action, model)
        shifted_feature = model_features(sequence, np.roll(action, 1, axis=0), model)
        correct.append(hmm_forward_backward_conditioned(sequence, correct_feature, model)[0])
        shifted.append(hmm_forward_backward_conditioned(sequence, shifted_feature, model)[0])
    assert np.mean(correct) > np.mean(shifted)


def test_control_feature_modes_keep_progress_explicit() -> None:
    action = np.arange(8, dtype=float)[:, None]
    mean, scale = np.zeros(1), np.ones(1)
    fixed = control_features(action, 8, mean, scale, mode="fixed")
    progress = control_features(action, 8, mean, scale, mode="progress")
    both = control_features(action, 8, mean, scale, mode="action_progress")
    assert fixed.shape == (7, 1)
    assert progress.shape == (7, 3)
    assert both.shape == (7, 4)
    np.testing.assert_array_equal(both[:, 1], action[1:, 0])


def test_action_transition_fit_can_freeze_shared_emissions() -> None:
    sequences, actions = _controlled_sequences(n=12)
    shared = {
        "initial": np.asarray([0.4, 0.6]),
        "transition": np.asarray([[0.8, 0.2], [0.3, 0.7]]),
        "means": np.asarray([[-2.5, -2.5], [2.5, 2.5]]),
        "var": np.asarray([[0.5, 0.5], [0.7, 0.7]]),
    }
    model = fit_action_conditioned_hmm(
        sequences, actions, 2, seed=15, iterations=8, fixed_state_model=shared
    )
    np.testing.assert_array_equal(model["initial"], shared["initial"])
    np.testing.assert_array_equal(model["means"], shared["means"])
    np.testing.assert_array_equal(model["var"], shared["var"])
    assert model["state_model_frozen"] is True
