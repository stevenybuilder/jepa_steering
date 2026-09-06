from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "cgs_pilot"))

from public_panel_coordinate_gate import (  # noqa: E402
    REACH_WALL_CENTER,
    coordinate_cv,
    fit_native_readout,
    metaworld_coordinate_targets,
    metaworld_reach_coordinate_targets,
    selected_replan_indices,
    signed_box_distance,
)


def test_reach_wall_coordinate_signs_and_audited_wall() -> None:
    states = np.zeros((2, 39))
    states[:, :3] = [[1, 2, 3], [2, 2, 3]]
    states[:, -3:] = [4, 2, 3]
    goal = np.zeros(39)
    goal[-3:] = [4, 2, 3]
    targets = metaworld_coordinate_targets(states, goal)
    np.testing.assert_allclose(targets["relative_hand_to_goal"], [[3, 0, 0], [2, 0, 0]])
    np.testing.assert_allclose(
        targets["relative_hand_to_fixed_wall"], REACH_WALL_CENTER[None, :] - states[:, :3]
    )
    assert "relative_hand_to_obstacle" not in targets


def test_obstacle_free_reach_adapter_uses_hand_and_goal_only() -> None:
    states = np.zeros((2, 39))
    states[:, :3] = [[0.0, 0.4, 0.1], [0.1, 0.5, 0.2]]
    states[:, -3:] = [0.2, 0.7, 0.3]
    goal = states[0].copy()
    targets = metaworld_reach_coordinate_targets(states, goal)
    np.testing.assert_allclose(targets["relative_hand_to_goal"][0], [0.2, 0.3, 0.2])
    assert "relative_hand_to_fixed_wall" not in targets


def test_adapter_rejects_stale_initial_goal_and_keeps_valid_first_endpoint() -> None:
    states = np.zeros((1, 39))
    goal = np.zeros(39)
    states[0, -3:] = [1, 2, 3]
    with pytest.raises(ValueError, match="does not match"):
        metaworld_coordinate_targets(states, goal)
    np.testing.assert_array_equal(selected_replan_indices((0, 5)), [0, 1, 2, 3, 4])
    with pytest.raises(ValueError, match="fewer than three"):
        selected_replan_indices((0, 2))


def test_signed_wall_distance_is_negative_inside_positive_outside() -> None:
    values = signed_box_distance(np.asarray([REACH_WALL_CENTER, [1.0, 1.0, 1.0]]))
    assert values[0] < 0.0
    assert values[1] > 0.0


def test_coordinate_cv_uses_matched_and_goal_binding_controls() -> None:
    rng = np.random.default_rng(4)
    factors, targets = [], []
    wall = REACH_WALL_CENTER
    for episode in range(20):
        hand = rng.normal(scale=0.3, size=(5, 3))
        goal = rng.normal(scale=1.5, size=3)
        relation = goal[None, :] - hand
        # Hidden state contains the episode-bound relation, not absolute hand or
        # the goal belonging to a different episode.
        factors.append(np.c_[relation, rng.normal(scale=0.01, size=(5, 1))])
        targets.append({
            "absolute_hand_world": hand,
            "relative_hand_to_goal": relation,
            "relative_hand_to_fixed_wall": wall[None, :] - hand,
            "absolute_goal_world": np.broadcast_to(goal, hand.shape),
            "distance_hand_to_goal": np.linalg.norm(relation, axis=1, keepdims=True),
            "signed_hand_wall_clearance": rng.normal(size=(5, 1)),
            "straight_path_wall_clearance": rng.normal(size=(5, 1)),
        })
    report = coordinate_cv(factors, targets, n_folds=5, ridge=0.1, seed=9)
    assert report["model_native_relational_candidate"]
    assert report["primary_matched_comparison"]["candidate_better"]
    assert report["goal_binding_control"]["candidate_better"]
    assert report["native_readout_rowspace_stability"]["stable"]
    # A fixed translation/sign of the absolute target is mathematically
    # equivalent for a centered linear readout.
    assert abs(
        report["fixed_wall_affine_equivalence_control"]["candidate_minus_reference_normalized_mse"]
    ) < 1e-10


def test_native_readout_serializes_goal_relative_rowspace() -> None:
    rng = np.random.default_rng(8)
    factors = [rng.normal(size=(5, 4)) for _ in range(8)]
    weights = np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]])
    targets = [
        {"relative_hand_to_goal": value @ weights + [0.1, -0.2, 0.3]} for value in factors
    ]
    fitted = fit_native_readout(factors, targets, ridge=1e-6)
    assert fitted["rank"] == 3
    projection = fitted["rowspace_basis"] @ fitted["rowspace_basis"].T
    np.testing.assert_allclose(projection[3], np.zeros(4), atol=1e-6)
