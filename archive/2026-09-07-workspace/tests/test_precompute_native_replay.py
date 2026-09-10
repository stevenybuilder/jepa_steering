from pathlib import Path
import sys
import os
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "geometry_map"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
if os.environ.get("GEOMETRY_REPLAY_CODE_DIR"):
    sys.path.insert(0, os.environ["GEOMETRY_REPLAY_CODE_DIR"])
from precompute_native_replay import exact_array, safe_rows, ReachControlCapture, PushContactCapture, restore_goal_encodings
from precompute_native_replay import check_reach_waypoint


def test_development_gate_rejects_held_before_loading():
    index = {"banks": [{"task": "reach_wall", "episode": 12, "sealed": True, "split": "evaluation"}]}
    with pytest.raises(RuntimeError, match="non-development"):
        safe_rows(index, "reach_wall", [12])


def test_waypoint_checks_do_not_relax_pixel_or_physics_equality():
    assert exact_array(np.zeros(3), np.zeros(3), "same", 0) == 0
    with pytest.raises(RuntimeError, match="error"):
        exact_array(np.array([1]), np.array([0]), "pixels", 0)
    with pytest.raises(RuntimeError, match="layout"):
        exact_array(np.zeros((2, 2)), np.zeros(4), "state")


def test_waypoint_failure_reports_physics_even_when_pixels_fail(monkeypatch):
    import types
    now = {"time": 1., "elapsed_steps": 1, "physics": {"qpos": np.array([.01])}, "task_state": {}}
    monkeypatch.setitem(sys.modules, "collect_on_policy_bank", types.SimpleNamespace(physics_snapshot=lambda env: now))
    snapshot = {"observation_visual": np.array([0]), "observation_proprio": np.array([0.]),
                "elapsed_steps": 1, "simulator": {**now, "physics": {"qpos": np.array([0.])}}}
    with pytest.raises(RuntimeError) as error:
        check_reach_waypoint(None, np.array([48]), np.array([0.]), snapshot)
    assert '"pixels"' in str(error.value) and '"qpos"' in str(error.value)
    assert '"proprio": 0.0' in str(error.value)


class FakeReach:
    def __init__(self):
        self.data = SimpleNamespace(mocap_pos=np.zeros((1, 3)))
    def set_xyz_action(self, action):
        self.data.mocap_pos += np.clip(action, -1, 1) * .01
        return 17


def test_control_observer_preserves_native_clipping_and_restores_method():
    base = FakeReach()
    original = base.set_xyz_action
    with ReachControlCapture(base) as capture:
        assert base.set_xyz_action(np.array([2., 0., -3.])) == 17
        np.testing.assert_allclose(base.data.mocap_pos, [[.01, 0., -.01]])
        assert len(capture.records) == 1
        np.testing.assert_allclose(capture.records[0]["xyz_argument_to_native_setter"], [2., 0., -3.])
    assert base.set_xyz_action == original


def test_contact_observer_preserves_existing_callback_and_body_identity():
    calls = []
    original = lambda *args: calls.append(args) or 13
    agent, block, wall = object(), object(), object()
    base = SimpleNamespace(agent=agent, block=block, space=SimpleNamespace(static_body=wall),
                           collision_handeler=SimpleNamespace(post_solve=original))
    arbiter = SimpleNamespace(shapes=[SimpleNamespace(body=agent), SimpleNamespace(body=block)],
                              contact_point_set=SimpleNamespace(normal=(1, 0), points=[]), total_impulse=(2, 3))
    with PushContactCapture(base) as capture:
        assert base.collision_handeler.post_solve(arbiter, base.space, {}) == 13
        assert capture.records[0]["agent_block_contact"]
        assert len(calls) == 1
    assert base.collision_handeler.post_solve is original


def test_missing_contact_api_does_not_fabricate_contact():
    with PushContactCapture(SimpleNamespace()) as capture:
        assert not capture.available and capture.records == []


def test_cached_stimulus_is_restored_and_fresh_encoding_error_is_reported():
    import torch
    cached = {"encoded_visual": torch.tensor([1., 2.], dtype=torch.float16),
              "encoded_proprio": torch.tensor([3.], dtype=torch.float16)}
    agent = SimpleNamespace(device="cpu", goal_state_enc={"visual": torch.tensor([1.01, 2.]),
                                                        "proprio": torch.tensor([3.])})
    errors = restore_goal_encodings(agent, cached)
    assert errors["visual"] > 0 and errors["proprio"] == 0
    assert torch.equal(agent.goal_state_enc["visual"], cached["encoded_visual"])
