from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "cgs_pilot"))

from public_panel_sonar_fit import (  # noqa: E402
    action_metric,
    bootstrap_density_stability,
    coordinate_maps,
    fit_outcome_models_from_episodes,
    local_reach_progress_labels,
    routed_local_quality_cv,
    static_responsibilities,
    transform,
)
from public_panel_action_hmm import hmm_filter_conditioned, model_features  # noqa: E402


def test_coordinate_map_decodes_factor_displacements() -> None:
    theta = 0.4
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    coordinate = {
        "mean": np.array([1.0, 2.0, 3.0]),
        "basis": np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
        "scale": np.array([2.0, 4.0]),
        "factor_rotation": rotation,
        "representation": np.asarray("predictive_factors"),
    }
    encoder, decoder = coordinate_maps(coordinate)
    hidden = np.array([[2.0, -3.0, 9.0]])
    factors = transform(hidden, coordinate)
    assert np.allclose((hidden - coordinate["mean"]) @ encoder, factors)
    assert np.allclose((factors + np.array([[0.3, -0.2]])) @ decoder - factors @ decoder,
                       np.array([[0.3, -0.2]]) @ decoder)


def test_static_regime_responsibilities_normalize() -> None:
    model = {
        "weights": np.array([0.3, 0.7]),
        "means": np.array([[-2.0, 0.0], [2.0, 0.0]]),
        "var": np.ones((2, 2)),
    }
    responsibility = static_responsibilities(np.array([[-3.0, 0.0], [3.0, 0.0]]), model)
    assert np.allclose(responsibility.sum(1), 1.0)
    assert responsibility[0, 0] > 0.99 and responsibility[1, 1] > 0.99


def test_action_metric_uses_exact_executed_control_summaries() -> None:
    sequences = []
    action_sequences = []
    rng = np.random.default_rng(3)
    for _episode in range(5):
        latent = rng.normal(size=(3, 2))
        action = np.zeros((3, 2, 1), dtype=np.float32)
        action[:, 0, 0] = latent[:, 0]
        action_sequences.append(action.reshape(3, -1))
        sequences.append(latent)
    gram, report = action_metric(
        action_sequences,
        sequences,
        (0, 3),
        ridge=1e-4,
        action_source="exact_executed_control_chunk_summary",
    )
    assert report["available"]
    assert report["in_sample_variance_explained_descriptive_only"] > 0.9
    assert gram[0, 0] > gram[1, 1]


def test_action_metric_fails_closed_without_exact_executed_controls() -> None:
    gram, report = action_metric(
        [np.ones((2, 4))],
        [np.ones((2, 3))],
        (0, 2),
        ridge=1.0,
        action_source="selected_planner_action_fallback",
    )
    assert not report["available"]
    assert np.count_nonzero(gram) == 0


def test_density_and_ot_fields_are_stable_under_episode_bootstrap() -> None:
    rng = np.random.default_rng(21)
    sequences, labels, regimes = [], [], []
    for outcome in (0, 1):
        for _episode in range(12):
            sequences.append(rng.normal(loc=2.0 * outcome - 1.0, scale=0.15, size=(4, 2)))
            labels.append(outcome)
            regimes.append(np.ones((4, 1)))
    label_sequences = [np.full(len(sequence), label, dtype=np.int8) for sequence, label in zip(sequences, labels)]
    fitted, _effective = fit_outcome_models_from_episodes(
        sequences,
        label_sequences,
        regimes,
        shrinkage=0.25,
        covariance_floor=1e-3,
    )
    report = bootstrap_density_stability(
        sequences,
        label_sequences,
        regimes,
        fitted,
        shrinkage=0.25,
        covariance_floor=1e-3,
        draws=20,
        seed=4,
        minimum_cosine=0.8,
    )
    assert report["energy_eligible"]
    assert report["ot_eligible"]
    assert report["precision_operator_cosine"]["p05"] > 0.8


def test_routed_local_quality_comparison_is_episode_held_out_and_finite() -> None:
    rng = np.random.default_rng(31)
    regime = {
        "weights": np.array([0.5, 0.5]),
        "means": np.array([[-1.0, 0.0], [1.0, 0.0]]),
        "var": np.full((2, 2), 0.25),
    }
    transition = np.array([[0.9, 0.1], [0.1, 0.9]])
    sequences, labels = [], []
    for outcome in (0, 1):
        for episode in range(8):
            states = np.array([0, 0, 1, 1]) if episode % 2 == 0 else np.array([1, 1, 0, 0])
            mean = regime["means"][states].copy()
            mean[:, 1] += (2 * outcome - 1) * 0.25
            sequences.append(mean + rng.normal(scale=0.08, size=mean.shape))
            labels.append(outcome)
    transition_coef = np.zeros((2, 2, 4))
    transition_coef[:, :, 0] = np.log(transition)
    action_hmm = {
        "initial": np.array([0.5, 0.5]),
        "transition_coef": transition_coef,
        "action_mean": np.zeros(1),
        "action_scale": np.ones(1),
        "means": regime["means"],
        "var": regime["var"],
        "feature_mode": "action_progress",
    }
    label_sequences = [
        np.full(len(sequence), label, dtype=np.int8)
        for sequence, label in zip(sequences, labels)
    ]
    report = routed_local_quality_cv(
        sequences,
        label_sequences,
        regime,
        actions=[np.zeros((len(sequence), 1)) for sequence in sequences],
        action_hmm=action_hmm,
        cross_fitted_hmm_route=np.stack([
            hmm_filter_conditioned(
                sequence,
                model_features(sequence, np.zeros((len(sequence), 1)), action_hmm),
                action_hmm,
            )
            for sequence in sequences
        ]),
        shrinkage=0.25,
        covariance_floor=1e-3,
        seed=7,
    )
    assert report["decidable"]
    assert report["causal_filter_no_future"]
    assert not report["terminal_episode_outcome_used"]
    assert np.isfinite(list(report["mean_log_loss"].values())).all()
    assert all(
        0.0 < value < 1.0
        for values in report["per_episode_mean_probability"].values()
        for value in values
    )


def test_reach_labels_use_each_replan_progress_not_terminal_outcome(tmp_path: Path) -> None:
    episodes = []
    index = {"episodes": []}
    hands = [
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.4, 0.0, 0.0]],
        [[0.2, 0.0, 0.0], [0.2, 0.0, 0.0], [0.8, 0.0, 0.0]],
    ]
    for episode_id, positions in enumerate(hands):
        states = np.zeros((3, 39), dtype=np.float64)
        states[:, :3] = positions
        states[:, -3:] = [1.0, 0.0, 0.0]
        goal = np.zeros(39, dtype=np.float64)
        goal[-3:] = [1.0, 0.0, 0.0]
        trace = Path("behavior_traces") / f"ep{episode_id:03d}.npz"
        (tmp_path / trace.parent).mkdir(exist_ok=True)
        np.savez_compressed(
            tmp_path / trace,
            simulator_states=states,
            goal_state=goal,
            plan_boundaries=np.array([0, 1, 2]),
        )
        # Both episodes deliberately have the same terminal label. Their local
        # labels must still differ because fitting ignores that endpoint label.
        episodes.append({"ep": episode_id, "success": True, "behavior_trace": str(trace)})
        index["episodes"].append({"ep": episode_id, "success": True})
    (tmp_path / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in episodes)
    )
    labels, progress, report = local_reach_progress_labels(
        tmp_path, index, (0, 2), minimum_progress=1e-4
    )
    np.testing.assert_array_equal(labels[0], [1, 0])
    np.testing.assert_array_equal(labels[1], [0, 1])
    np.testing.assert_allclose(progress[0], [0.5, -0.1])
    np.testing.assert_allclose(progress[1], [0.0, 0.6])
    assert not report["terminal_episode_outcome_used_for_fit"]
    assert report["transition_counts"] == {"0": 2, "1": 2}
