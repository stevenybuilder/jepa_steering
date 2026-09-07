"""Causal diagnostic controls: metric, time scope, split boundary, replay layout."""
from pathlib import Path
import json
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "geometry_map"))
from subspace_control import FirstStepSubspacePatch, projected_delta
from causal_planner_forks import counterfactual_actions, execute_fork, load_development_bank, replay_action_count, visual_pooled
from fit_causal_subspace import fit_basis


def fixture_coordinates():
    generator = torch.Generator().manual_seed(19)
    basis = torch.linalg.qr(torch.randn(12, 3, generator=generator, dtype=torch.float64))[0]
    scale = torch.linspace(0.4, 2.0, 12, dtype=torch.float64)
    current = torch.randn(4, 12, generator=generator, dtype=torch.float64)
    donor = torch.randn(1, 12, generator=generator, dtype=torch.float64)
    return current, donor, basis, scale


def test_projection_reaches_donor_coordinates_and_preserves_complement():
    current, donor, basis, scale = fixture_coordinates()
    delta = projected_delta(current, donor, basis, scale, 1.0)
    torch.testing.assert_close(((current + delta - donor) / scale) @ basis, torch.zeros(4, 3, dtype=torch.float64), atol=1e-12, rtol=0)
    complement = torch.eye(12, dtype=torch.float64) - basis @ basis.T
    torch.testing.assert_close((delta / scale) @ complement, torch.zeros_like(delta), atol=1e-12, rtol=0)


def test_beta_zero_exact_identity_and_hook_removal():
    current, donor, basis, scale = fixture_coordinates()
    block = torch.nn.Identity()
    value = current.float().unsqueeze(1).repeat(1, 256, 1).to(torch.float16)
    patch = FirstStepSubspacePatch(block, block, basis.float(), scale.float(), donor.float(), beta=0)
    with patch:
        assert patch.unroll(value) is value
        assert patch.unroll(value) is value
    assert not block._forward_hooks
    assert patch.calls == 2


def test_first_step_only_resets_each_rollout_and_preserves_token_residuals():
    current, donor, basis, scale = fixture_coordinates()
    block = torch.nn.Identity()
    value = current.float().unsqueeze(1) + torch.linspace(-0.1, 0.1, 256).reshape(1, 256, 1)

    def rollout(x):
        return block(x), block(x), block(x)

    patch = FirstStepSubspacePatch(block, rollout, basis.float(), scale.float(), donor.float())
    with patch:
        first = patch.unroll(value)
        second = patch.unroll(value)
    assert not torch.equal(first[0], value)
    assert first[1] is value and first[2] is value
    torch.testing.assert_close(first[0], second[0], atol=0, rtol=0)
    torch.testing.assert_close(first[0][:, 1:] - first[0][:, :-1], value[:, 1:] - value[:, :-1], atol=3e-7, rtol=0)
    assert patch.calls == 2


def test_sham_matches_per_candidate_raw_edit_norm():
    current, donor, basis, scale = fixture_coordinates()
    value = current.float().unsqueeze(1).repeat(1, 256, 1)
    edits = []
    for sham in (False, True):
        block = torch.nn.Identity()
        patch = FirstStepSubspacePatch(block, block, basis.float(), scale.float(), donor.float(), sham=sham)
        with patch:
            patch.unroll(value)
        edits.append(patch.first_edit)
    torch.testing.assert_close(edits[0].norm(dim=1), edits[1].norm(dim=1), atol=1e-6, rtol=1e-6)
    assert not torch.allclose(edits[0], edits[1])


def test_confirmation_receipt_rejected_before_tensor_load(tmp_path, monkeypatch):
    (tmp_path / "DONE.json").write_text(json.dumps({"complete": True, "outputs": [{"path": "episode-012.pt", "episode": 12, "sha256": "unused"}]}))
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("Held-out tensor was opened"))
    with pytest.raises(RuntimeError, match="confirmation"):
        load_development_bank(tmp_path / "episode-012.pt")


def test_physical_counterfactuals_preserve_gripper():
    baseline = torch.arange(60).reshape(15, 4).float() / 60
    raw = counterfactual_actions({"planned_actions_raw": baseline})
    torch.testing.assert_close(raw[:, :, 3], baseline[:5, 3].expand(6, 5), atol=0, rtol=0)
    torch.testing.assert_close(raw[0], baseline[:5], atol=0, rtol=0)
    assert raw[4, :, 2].eq(1).all() and raw[5, :, 2].eq(-1).all()


def test_replay_prefix_excludes_environment_warmup_step():
    bank = {"replans": [{"elapsed_steps": 1, "executed_action_count": 15},
                        {"elapsed_steps": 16, "executed_action_count": 15}]}
    assert replay_action_count(bank, 0) == 0
    assert replay_action_count(bank, 1) == 15


def test_prediction_pooling_keeps_time_and_candidate_axes():
    visual = torch.arange(3 * 6 * 4 * 3).reshape(3, 6, 4, 3).float()
    torch.testing.assert_close(visual_pooled({"visual": visual}, torch.zeros(2, 6, 20)), visual[1:].mean(2))
    with pytest.raises(RuntimeError, match="time,batch"):
        visual_pooled({"visual": visual.transpose(0, 1)}, torch.zeros(2, 6, 20))


def test_simulator_fork_records_real_list_api():
    class Env:
        def step_multiple(self, actions):
            return [torch.zeros(2, 2, 3, dtype=torch.uint8)] * 5, [0.] * 5, [False] * 5, [{"state": np.arange(9), "success": False}] * 5
    result = execute_fork(Env(), torch.zeros(5, 4))
    assert result["frames"].shape == (5, 2, 2, 3)
    assert result["steps"] == 5 and result["states"].shape == (5, 9)


def test_iterative_fit_returns_orthonormal_saved_metric():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(120, 12)) * np.linspace(.1, 3, 12)
    y = x[:, :2] + rng.normal(size=(120, 2)) * .01
    mean, scale, basis = fit_basis(x, y, np.ones(120, dtype=bool), iterations=3)
    np.testing.assert_allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-12)
    assert 0 < basis.shape[1] <= 6
    np.testing.assert_allclose(mean, x.mean(0))
    np.testing.assert_allclose(scale, x.std(0))
