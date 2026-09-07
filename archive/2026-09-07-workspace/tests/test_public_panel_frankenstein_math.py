from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "cgs_pilot"))

from public_panel_frankenstein_math import (  # noqa: E402
    apply_orthogonal_transport,
    attention_probabilities,
    attention_retrieval_gate,
    fit_orthogonal_transport,
    fit_sparse_dictionary,
    mahalanobis_squared,
    matched_lure_pairs,
    pattern_preserving_delta,
    probability_geometry_gate,
    restore_attention_output,
    sparse_reconstruct,
    symmetric_gaussian_kl,
    transport_direction,
    transport_operator,
)


def test_probability_geometry_compares_families_and_calibrates_empirically() -> None:
    rng = np.random.default_rng(2)
    sequences, responsibilities = [], []
    for episode in range(24):
        regime = np.array([0, 0, 0, 1, 1, 1]) if episode % 2 else np.array([1, 1, 1, 0, 0, 0])
        rows = np.stack([
            rng.normal([-2.0, 0.0], [0.25, 0.8]) if value == 0
            else rng.normal([2.0, 0.0], [0.8, 0.25])
            for value in regime
        ])
        sequences.append(rows)
        responsibilities.append(np.eye(2)[regime])
    report = probability_geometry_gate(sequences, responsibilities, blocks=[(0, 2)], n_folds=4, seed=4)
    assert report["blocks"][0]["selected_family"] in {"isotropic", "diagonal", "shrinkage_full"}
    assert report["at_least_one_regime_distribution_difference"]
    assert set(report["blocks"][0]["empirical_tail_coverage"]) == {"0.9", "0.95"}


def test_sparse_dictionary_round_trip_beats_zero_reconstruction() -> None:
    rng = np.random.default_rng(3)
    dictionary = rng.normal(size=(10, 6))
    dictionary /= np.linalg.norm(dictionary, axis=1, keepdims=True)
    code = np.zeros((200, 10))
    for row in range(len(code)):
        active = rng.choice(10, size=2, replace=False)
        code[row, active] = rng.normal(size=2)
    rows = code @ dictionary + 0.01 * rng.normal(size=(200, 6))
    model = fit_sparse_dictionary(rows, n_components=10, alpha=0.10, seed=5, max_iter=80)
    reconstructed, fitted_code = sparse_reconstruct(rows, model)
    assert reconstructed.shape == rows.shape and fitted_code.shape == code.shape
    assert np.mean(np.square(rows - reconstructed)) < 0.2 * np.mean(np.square(rows - rows.mean(0)))
    assert np.mean(np.abs(fitted_code) > 1e-8) < 0.5


def test_lure_matching_is_cross_episode_and_control_distinct() -> None:
    nuisance = np.arange(12, dtype=float)[:, None]
    labels = np.arange(12) % 2
    episodes = np.repeat(np.arange(6), 2)
    pairs = matched_lure_pairs(nuisance, labels, episodes)
    assert len(pairs)
    assert np.all(episodes[pairs[:, 0]] != episodes[pairs[:, 1]])
    assert np.all(labels[pairs[:, 0]] != labels[pairs[:, 1]])
    basis = np.eye(3)[:, :1]
    delta = np.asarray([[1.0, 2.0, 3.0]])
    protected = pattern_preserving_delta(delta, basis)
    np.testing.assert_allclose(protected, [[0.0, 2.0, 3.0]])


def test_procrustes_row_convention_transports_directions_and_operators() -> None:
    rng = np.random.default_rng(7)
    rotation, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    source = rng.normal(size=(100, 5))
    target = source @ rotation + np.arange(5)
    model = fit_orthogonal_transport(source, target)
    np.testing.assert_allclose(apply_orthogonal_transport(source, model), target, atol=1e-10)
    assert model["relative_residual"] < 1e-10
    direction = rng.normal(size=5)
    np.testing.assert_allclose(transport_direction(direction, rotation), direction @ rotation)
    operator = rng.normal(size=(5, 5))
    source_edited = source @ operator.T
    target_edited_via_source = source_edited @ rotation
    target_centerless = source @ rotation
    transported = transport_operator(operator, rotation)
    np.testing.assert_allclose(target_centerless @ transported.T, target_edited_via_source, atol=1e-10)


def test_attention_probability_retrieval_and_restoration_are_distinct() -> None:
    query = np.asarray([[[[1.0, 0.0], [0.0, 1.0]]]])
    key = query.copy()
    probability = attention_probabilities(query, key)
    np.testing.assert_allclose(probability.sum(-1), 1.0)
    assert probability[0, 0, 0, 0] > probability[0, 0, 0, 1]
    steered = np.ones((2, 3))
    clean = np.zeros((2, 3))
    np.testing.assert_allclose(restore_attention_output(steered, clean, 0.25), 0.75)


def test_attention_gate_detects_qk_regime_amplification() -> None:
    rng = np.random.default_rng(11)
    n, heads, tokens, width = 40, 2, 3, 2
    labels = np.repeat([0, 1], n // 2)
    residual = rng.normal(scale=1.0, size=(n, 4))
    query = rng.normal(scale=0.2, size=(n, heads, tokens, width))
    key = rng.normal(scale=0.2, size=(n, heads, tokens, width))
    query[labels == 1, :, :, 0] += 2.0
    key[labels == 1, :, :, 0] += 2.0
    report = attention_retrieval_gate(residual, query, key, labels)
    assert report["decidable"] and report["eligible"]
    assert report["normalized_regime_energy_distance"]["query"] > report["normalized_regime_energy_distance"]["residual"]
