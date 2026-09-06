"""Physical decoder split integrity, time layout, and causal measurement semantics."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "geometry_map"))
# Remote CPU execution places the isolated scripts in a sibling code directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from analyze_causal_forks import (action_metrics, checked_load, cost_metrics, decode,
                                 fit_and_validate, load_discovery, path_crossings,
                                 pool_native_prediction, prediction_arrays, realized_metrics)


def test_heldout_receipts_never_open_or_hash_tensors(tmp_path, monkeypatch):
    (tmp_path / "DONE.json").write_text(json.dumps({"complete": True, "outputs": [
        {"seed": 1, "episode": 50, "path": "validation.pt", "sha256": "unused"},
        {"seed": 1, "episode": 75, "path": "confirmation.pt", "sha256": "unused"}]}))
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("Held-out tensor opened"))
    data, _ = load_discovery([tmp_path], require_complete=False)
    assert data == []


def test_hash_mismatch_blocks_deserialization(tmp_path, monkeypatch):
    path = tmp_path / "bad.pt"
    path.write_bytes(b"not a tensor")
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("Bad hash was loaded"))
    with pytest.raises(RuntimeError, match="Hash mismatch before"):
        checked_load(tmp_path, {"path": path.name, "sha256": "wrong"})


def test_native_pooling_removes_context_preserves_time_candidates():
    x = torch.arange(3*6*4*384).reshape(3, 6, 4, 384).float()
    pooled = pool_native_prediction({"visual": x}, 2, 6)
    np.testing.assert_allclose(pooled, x[1:].mean(2).numpy())
    with pytest.raises(RuntimeError, match="native visual"):
        pool_native_prediction({"visual": x.transpose(0, 1)}, 2, 6)


def test_point_path_margin_and_goal_line_are_distinct():
    # A path over the wall top clears the exact wall but enters the 3 cm margin.
    points = np.array([[.1, .7, .14], [.1, .8, .14]])
    result = path_crossings(points, np.array([.1, .9, .14]))
    assert not result["point_margin_0"]["any_segment_intersects"]
    assert result["point_margin_0_03"]["any_segment_intersects"]
    assert result["body_collision_validated"] is False
    # An actual sideways path clears, though the hand-to-goal proxy hits the wall.
    points = np.array([[.1, .7, .08], [-.1, .7, .08]])
    result = path_crossings(points, np.array([.1, .8, .08]))
    assert not result["point_margin_0"]["any_segment_intersects"]
    assert result["point_margin_0"]["hand_to_goal_straight_line_proxy"][0]


def test_action_cancellation_separates_chunk_norm_from_net():
    action = np.zeros((5, 4))
    action[0, 0], action[1, 0] = 1, -1
    result = action_metrics(action, np.zeros_like(action))
    assert result["translation_net"]["norm"] == 0
    assert result["translation_net"]["unit_direction_xyz"] is None
    assert result["translation_frobenius_norm"] == pytest.approx(np.sqrt(2))


def test_tied_costs_use_tie_aware_ranks_and_identity_is_exact():
    base = np.array([1., 1., 2., 3.])
    result = cost_metrics(base, base)
    assert result["ranks"] == [.5, .5, 2., 3.]
    assert result["rank_changes"] == 0
    assert result["cost_l2_change"] == 0
    changed = cost_metrics(base[::-1], base)
    assert changed["argmin_changed"] and changed["rank_changes"] == 4


def test_realized_sampling_is_five_raw_steps_and_includes_start():
    start, goal = np.zeros(3), np.ones(3)
    states = np.zeros((15, 9))
    states[:, 0] = np.arange(1, 16) / 100
    states[:, -3:] = goal
    fork = {"states": states, "end_success": False}
    result = realized_metrics(fork, fork, start, goal)
    assert len(result["hand_xyz_path_including_start"]) == 16
    np.testing.assert_allclose(np.array(result["model_step_hand_xyz"])[:, 0], [.05, .1, .15])
    assert result["endpoint_change_from_unsteered"]["norm"] == 0


def test_prediction_steps_are_incremental_but_displacement_is_start_relative():
    xyz = np.array([[[.2, .6, .3]], [[.3, .6, .3]]])
    start, goal = np.array([.1, .6, .3]), np.array([.5, .6, .3])
    result = prediction_arrays(xyz, start, goal)
    np.testing.assert_allclose(result["decoded_step_displacement_xyz"][:, 0, 0], [.1, .1])
    np.testing.assert_allclose(result["decoded_displacement_xyz"][:, 0, 0], [.1, .2])
    np.testing.assert_allclose(result["decoded_goal_progress"][:, 0], [.1, .2])


def test_decoder_uses_natural_features_and_next_state_alignment():
    rng = np.random.default_rng(12)
    trajectories = []
    for seed in (1, 2, 3):
        for episode in range(4):
            x = rng.normal(size=(20, 6)) + seed
            xyz = x[:, :3] * .1
            trajectories.append({"seed": seed, "episode": episode, "x": x,
                                 "future": x[1:].copy(), "xyz": xyz})
    readout, report = fit_and_validate(trajectories)
    assert report["final_fit_states"] == 240
    assert report["natural_state_decoding"]["n"] == 240
    assert report["unsteered_jepa_predicted_final_to_actual_next_hand"]["n"] == 228
    for fold in report["folds"]:
        assert fold["train_trajectories"] == 8 and fold["test_trajectories"] == 4
    assert decode(readout, trajectories[0]["x"]).shape == (20, 3)
    np.testing.assert_allclose(readout["mean"], np.concatenate([r["x"] for r in trajectories]).mean(0))
