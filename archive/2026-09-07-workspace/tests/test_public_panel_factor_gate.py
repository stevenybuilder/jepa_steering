from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "cgs_pilot"))

from public_panel_factor_gate import (  # noqa: E402
    factor_cv,
    fit_diag_gmm,
    fit_diag_hmm,
    gmm_log_prob,
    hmm_filter,
    hmm_forward_backward,
    log_gaussian_diag,
    markov_history_gate,
    regime_cv,
    summarize_action_chunks,
)


def test_diagonal_gaussian_shapes_and_finite_models() -> None:
    rng = np.random.default_rng(0)
    rows = np.r_[rng.normal(-2, 0.2, (50, 3)), rng.normal(2, 0.2, (50, 3))]
    model = fit_diag_gmm(rows, 2, seed=0)
    assert log_gaussian_diag(rows, model["means"], model["var"]).shape == (100, 2)
    assert np.isfinite(gmm_log_prob(rows, model)).all()
    hmm = fit_diag_hmm([rows[:25], rows[25:50], rows[50:75], rows[75:]], 2, seed=0)
    likelihood, posterior, transitions = hmm_forward_backward(rows[:25], hmm)
    assert np.isfinite(likelihood)
    assert posterior.shape == (25, 2) and transitions.shape == (24, 2, 2)
    np.testing.assert_allclose(posterior.sum(1), 1.0, atol=1e-6)


def test_sequence_hmm_beats_static_mixture_on_persistent_states() -> None:
    rng = np.random.default_rng(1)
    sequences = []
    for _ in range(30):
        left = rng.normal(-2, 0.25, (5, 2))
        right = rng.normal(2, 0.25, (5, 2))
        sequences.append(np.r_[left, right])
    report = regime_cv(sequences, n_folds=5, k=2, seed=3)
    assert report["hmm_minus_static_gmm"]["mean"] > 0
    assert report["full_hmm"]["min_emission_variance"] > 0


def test_causal_filter_prefix_is_invariant_to_future_observations() -> None:
    model = {
        "initial": np.array([0.6, 0.4]),
        "transition": np.array([[0.9, 0.1], [0.2, 0.8]]),
        "means": np.array([[-1.0], [1.0]]),
        "var": np.ones((2, 1)),
    }
    first = np.array([[-1.0], [-0.5], [0.0], [10.0]])
    second = np.array([[-1.0], [-0.5], [0.0], [-10.0]])
    np.testing.assert_allclose(hmm_filter(first, model)[:3], hmm_filter(second, model)[:3], atol=1e-12)


def test_unimodal_iid_data_does_not_force_a_local_regime() -> None:
    rng = np.random.default_rng(11)
    sequences = [rng.normal(size=(6, 2)) for _ in range(24)]
    report = regime_cv(sequences, n_folds=4, k=2, seed=8, stability_draws=20)
    assert report["selected_regime_source"] == "one_gaussian"
    assert not report["regime_locality_eligible"]


def test_factor_cv_is_outcome_free_and_returns_a_registered_choice() -> None:
    rng = np.random.default_rng(4)
    dynamics = np.diag([0.95, 0.8, 0.2, -0.1])
    sequences = []
    for _ in range(20):
        x = rng.normal(size=4)
        seq = [x]
        for _step in range(7):
            x = x @ dynamics + rng.normal(scale=0.1, size=4)
            seq.append(x)
        sequences.append(np.stack(seq))
    report = factor_cv(
        sequences, rank=4, block_size=2, n_folds=5, n_random=5, ridge=1.0, seed=5
    )
    assert report["selected_representation"] in {"global", "pca_blocks", "predictive_factors"}
    assert all(np.isfinite(value) for value in report["cv_normalized_mse"].values())
    assert report["pca_diagnostics"]["rank"] == 4
    if report["selected_representation"] == "predictive_factors":
        assert report["relative_gain_predictive_vs_global"] >= 0.02


def test_history_gate_requires_actions() -> None:
    sequences = [np.zeros((5, 2)) for _ in range(6)]
    result = markov_history_gate(sequences, None, n_folds=3, ridge=1.0, seed=0)
    assert not result["decidable"]
    assert "action" in result["reason"]


def test_executed_action_chunk_summary_is_causal_and_fixed_width() -> None:
    actions = np.asarray(
        [[1.0, 0.0], [2.0, 1.0], [4.0, 1.0], [3.0, 2.0], [3.0, 4.0]], dtype=float
    )
    features = summarize_action_chunks(actions, np.asarray([0, 2, 5]))
    assert features.shape == (2, 9)
    np.testing.assert_allclose(features[0, :2], [3.0, 1.0])
    np.testing.assert_allclose(features[0, 2:4], [1.5, 0.5])
    np.testing.assert_allclose(features[0, 4:6], [1.0, 1.0])
    assert features[0, -1] == 2
