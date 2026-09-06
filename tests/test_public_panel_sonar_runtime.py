from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "cgs_pilot"))

from public_panel_sonar_fit import coordinate_maps, transform  # noqa: E402
from public_panel_frankenstein_math import encode_sparse  # noqa: E402
from public_panel_steer import (  # noqa: E402
    AttentionOutputMediator,
    SonarSteerer,
    planner_to_executed_action_rows,
)


def _operator() -> dict[str, torch.Tensor]:
    covariance = torch.eye(2).reshape(1, 2, 2)
    global_conceptor = torch.diag(torch.tensor([0.5, 0.25]))
    return {
        "regime_weights": torch.ones(1),
        "regime_means": torch.zeros(1, 2),
        "regime_var": torch.ones(1, 2),
        "support_thresholds": torch.tensor([100.0]),
        "trust_metrics": torch.eye(2).reshape(1, 2, 2),
        "success_weights": torch.ones(1),
        "success_means": torch.tensor([[1.0, 0.0]]),
        "success_covariances": covariance,
        "success_precision": covariance,
        "success_logdet": torch.zeros(1),
        "failure_weights": torch.ones(1),
        "failure_means": torch.tensor([[-1.0, 0.0]]),
        "failure_covariances": covariance,
        "failure_precision": covariance,
        "failure_logdet": torch.zeros(1),
        "ot_affine": covariance.clone(),
        "ot_offset": torch.tensor([[2.0, 0.0]]),
        "sham_rotation": torch.tensor([[0.0, -1.0], [1.0, 0.0]]),
        "global_contrastive_conceptor": global_conceptor,
        "local_contrastive_conceptors": global_conceptor.reshape(1, 2, 2),
        "local_matched_spectrum_conceptors": torch.diag(torch.tensor([0.25, 0.5])).reshape(1, 2, 2),
        "local_label_shuffled_conceptors": torch.diag(torch.tensor([0.1, 0.9])).reshape(1, 2, 2),
        "hidden_mean": torch.zeros(2),
        "encoder": torch.eye(2),
        "decoder": torch.eye(2),
    }


def _steerer(arm="sonar_energy", mode="joint", beta=1.0) -> SonarSteerer:
    steerer = SonarSteerer.__new__(SonarSteerer)
    steerer.op = _operator()
    steerer.arm = arm
    steerer.beta = beta
    steerer.radius = 0.25
    steerer.displacement_mode = mode
    steerer.n_spatial = 2
    steerer.n_applied = 0
    steerer.n_abstained_support = 0
    steerer.metric_norm_sum = 0.0
    steerer.metric_norm_max = 0.0
    steerer.n_supported = 0
    steerer.n_cap_hits = 0
    steerer.n_boundary_normalized = 0
    steerer.n_zero_mode_components = 0
    steerer.use_frankenstein = False
    steerer.frankenstein_ablate = set()
    steerer.model_native_mode = "rowspace"
    steerer.n_abstained_sparse = 0
    steerer.pattern_removed_norm_sum = 0.0
    steerer.model_native_removed_norm_sum = 0.0
    steerer.hmm_belief = None
    steerer.hmm_pending_control = None
    steerer.hmm_transition_index = 0
    steerer.observe_only = False
    steerer.hmm_observation = None
    steerer.n_hmm_observation_forwards = 0
    steerer.n_hmm_belief_updates = 0
    return steerer


def test_energy_runtime_moves_toward_success_at_exact_trust_radius() -> None:
    steerer = _steerer()
    delta = steerer._factor_delta(torch.tensor([[0.0, 0.0]]))
    assert torch.allclose(delta, torch.tensor([[0.25, 0.0]]), atol=1e-6)
    assert np.isclose(steerer.metric_norm_max, 0.25)
    assert steerer.n_cap_hits == 1


def test_primary_energy_preserves_magnitude_below_cap() -> None:
    steerer = _steerer(beta=0.01)
    delta = steerer._factor_delta(torch.tensor([[0.0, 0.0]]))
    assert torch.allclose(delta, torch.tensor([[0.02, 0.0]]), atol=1e-6)
    assert steerer.n_cap_hits == 0


def test_boundary_energy_is_logged_as_boundary_normalized_not_cap_hit() -> None:
    steerer = _steerer(arm="sonar_energy_boundary")
    delta = steerer._factor_delta(torch.tensor([[0.0, 0.0]]))
    assert torch.allclose(delta, torch.tensor([[0.25, 0.0]]), atol=1e-6)
    assert steerer.n_boundary_normalized == 1
    assert steerer.n_cap_hits == 0


def test_ot_runtime_is_dosed_and_capped() -> None:
    steerer = _steerer(arm="sonar_ot")
    delta = steerer._factor_delta(torch.tensor([[0.0, 0.0]]))
    assert torch.allclose(delta, torch.tensor([[0.25, 0.0]]), atol=1e-6)


def test_sham_preserves_metric_norm_and_support_gate_abstains() -> None:
    steerer = _steerer(arm="sonar_sham")
    main = _steerer()._factor_delta(torch.tensor([[0.0, 0.0]]))
    sham = steerer._factor_delta(torch.tensor([[0.0, 0.0]]))
    assert torch.allclose(torch.linalg.vector_norm(main), torch.linalg.vector_norm(sham))
    assert not torch.allclose(main, sham)
    steerer.op["support_thresholds"][:] = 0.1
    rejected = steerer._factor_delta(torch.tensor([[10.0, 10.0]]))
    assert torch.count_nonzero(rejected) == 0
    assert steerer.n_abstained_support == 1


def test_factor_space_conceptor_uses_origin_centered_multiplicative_gate() -> None:
    x = torch.tensor([[2.0, 1.0]])
    expected = torch.tensor([[-0.4, -0.3]])
    global_steerer = _steerer(arm="sonar_coast_global", beta=0.4)
    static_steerer = _steerer(arm="sonar_coast_static", beta=0.4)
    global_steerer.radius = 2.0
    static_steerer.radius = 2.0
    global_delta = global_steerer._factor_delta(x)
    static_delta = static_steerer._factor_delta(x)
    assert torch.allclose(global_delta, expected, atol=1e-6)
    assert torch.allclose(static_delta, expected, atol=1e-6)


def test_final_trust_cap_applies_to_conceptor_proposal() -> None:
    steerer = _steerer(arm="sonar_coast_global", beta=0.4)
    delta = steerer._factor_delta(torch.tensor([[2.0, 1.0]]))
    assert np.isclose(float(torch.linalg.vector_norm(delta)), steerer.radius)
    assert steerer.n_cap_hits == 1
    assert steerer.metric_norm_max <= steerer.radius + 1e-7


def test_angular_and_radial_modes_match_joint_metric_dose_when_nonzero() -> None:
    x = torch.tensor([[1.0, 1.0]])
    joint = _steerer(mode="joint", beta=0.01)._factor_delta(x)
    angular = _steerer(mode="angular", beta=0.01)._factor_delta(x)
    radial = _steerer(mode="radial", beta=0.01)._factor_delta(x)
    joint_norm = torch.linalg.vector_norm(joint)
    assert torch.allclose(torch.linalg.vector_norm(angular), joint_norm, atol=1e-6)
    assert torch.allclose(torch.linalg.vector_norm(radial), joint_norm, atol=1e-6)
    assert abs(float((angular * x).sum())) < 1e-6


def test_coordinate_encoder_decoder_round_trip_and_hidden_edit() -> None:
    coordinate = {
        "mean": np.array([1.0, 2.0, 3.0]),
        "basis": np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
        "scale": np.array([2.0, 4.0]),
        "factor_rotation": np.eye(2),
        "representation": np.asarray("global"),
    }
    encoder, decoder = coordinate_maps(coordinate)
    rows = np.array([[3.0, 6.0, 9.0]])
    factors = transform(rows, coordinate)
    assert np.allclose((rows - coordinate["mean"]) @ encoder, factors)
    assert np.allclose(factors @ decoder, np.array([[2.0, 4.0, 0.0]]))


def test_frankenstein_pattern_projection_preserves_registered_axis() -> None:
    steerer = _steerer()
    steerer.use_frankenstein = True
    steerer.op["frankenstein_pattern_eligible"] = torch.tensor(1.0)
    steerer.op["frankenstein_pattern_basis"] = torch.tensor([[1.0], [0.0]])
    protected = steerer._preserve_pattern(torch.tensor([[2.0, 3.0]]))
    assert torch.allclose(protected, torch.tensor([[0.0, 3.0]]))


def test_sparse_runtime_ista_matches_offline_encoder() -> None:
    steerer = _steerer()
    dictionary = np.asarray([[1.0, 0.0], [0.0, 1.0], [2 ** -0.5, 2 ** -0.5]])
    rows = np.asarray([[1.0, -0.4], [0.2, 0.8]])
    steerer.op.update({
        "frankenstein_sparse_dictionary": torch.tensor(dictionary, dtype=torch.float32),
        "frankenstein_sparse_mean": torch.tensor([0.1, -0.2]),
        "frankenstein_sparse_alpha": torch.tensor(0.15),
        "frankenstein_sparse_ista_iterations": torch.tensor(100),
    })
    online = steerer._sparse_code(torch.tensor(rows, dtype=torch.float32)).numpy()
    offline = encode_sparse(rows - [0.1, -0.2], dictionary, 0.15, iterations=100)
    np.testing.assert_allclose(online, offline, atol=2e-5, rtol=2e-5)


def test_model_native_projection_preserves_metric_dose() -> None:
    steerer = _steerer()
    steerer.use_frankenstein = True
    steerer.op["frankenstein_model_native_eligible"] = torch.tensor(1.0)
    steerer.op["frankenstein_model_native_rowspace_basis"] = torch.tensor([[1.0], [0.0]])
    metric = torch.eye(2).reshape(1, 2, 2)
    projected = steerer._model_native_delta(torch.tensor([[1.0, 2.0]]), metric)
    assert torch.allclose(torch.linalg.vector_norm(projected), torch.sqrt(torch.tensor(5.0)))
    assert projected[0, 1] == 0


def test_module_ablation_and_model_native_complement_are_executable_controls() -> None:
    steerer = _steerer()
    steerer.use_frankenstein = True
    steerer.op["frankenstein_model_native_eligible"] = torch.tensor(1.0)
    steerer.op["frankenstein_model_native_rowspace_basis"] = torch.tensor([[1.0], [0.0]])
    metric = torch.eye(2).reshape(1, 2, 2)
    steerer.model_native_mode = "complement"
    complement = steerer._model_native_delta(torch.tensor([[1.0, 2.0]]), metric)
    assert complement[0, 0] == 0
    assert torch.allclose(torch.linalg.vector_norm(complement), torch.sqrt(torch.tensor(5.0)))
    steerer.frankenstein_ablate = {"model_native_coordinates"}
    unchanged = steerer._model_native_delta(torch.tensor([[1.0, 2.0]]), metric)
    assert torch.allclose(unchanged, torch.tensor([[1.0, 2.0]]))


def test_action_hmm_belief_is_candidate_specific_and_resettable() -> None:
    steerer = _steerer(arm="sonar_coast_hmm")
    steerer.op.update({
        "regime_weights": torch.tensor([0.5, 0.5]),
        "regime_means": torch.tensor([[-2.0, 0.0], [2.0, 0.0]]),
        "regime_var": torch.ones(2, 2),
        "action_hmm_initial": torch.tensor([0.5, 0.5]),
        "action_hmm_transition_coef": torch.zeros(2, 2, 4),
        "action_hmm_action_mean": torch.zeros(1),
        "action_hmm_action_scale": torch.ones(1),
        "action_hmm_sequence_length": torch.tensor(5.0),
        "action_hmm_feature_mode": "action_progress",
    })
    belief, _ = steerer._hmm_regime(torch.tensor([[-2.0, 0.0], [2.0, 0.0]]))
    assert belief[0, 0] > 0.95 and belief[1, 1] > 0.95
    routed, _ = steerer._regime(torch.tensor([[-2.0, 0.0], [2.0, 0.0]]))
    assert torch.allclose(routed[0], routed[1])
    steerer.hmm_belief = torch.tensor([0.1, 0.9])
    steerer.hmm_pending_control = torch.tensor([1.0])
    steerer.reset_episode_belief()
    assert steerer.hmm_belief is None and steerer.hmm_pending_control is None


def test_planner_control_summary_uses_exact_evaluator_action_units() -> None:
    class Preprocessor:
        @staticmethod
        def denormalize_actions(actions):
            return 2.0 * actions + torch.tensor([1.0, -1.0])

    planned = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    rows = planner_to_executed_action_rows(
        planned,
        environment_action_dim=2,
        repeat_actskip=True,
        action_skip=2,
        preprocessor=Preprocessor(),
    )
    expected = np.repeat(
        2.0 * np.asarray([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=float) + [1, -1],
        2,
        axis=0,
    )
    np.testing.assert_allclose(rows, expected)


def test_action_hmm_commits_once_after_multistep_chosen_action_replay() -> None:
    steerer = _steerer(arm="sonar_coast_hmm")
    steerer.op.update({
        "regime_weights": torch.tensor([0.5, 0.5]),
        "regime_means": torch.tensor([[-2.0, 0.0], [2.0, 0.0]]),
        "regime_var": torch.ones(2, 2),
        "action_hmm_initial": torch.tensor([0.5, 0.5]),
        "action_hmm_transition_coef": torch.zeros(2, 2, 4),
        "action_hmm_action_mean": torch.zeros(1),
        "action_hmm_action_scale": torch.ones(1),
        "action_hmm_sequence_length": torch.tensor(5.0),
        "action_hmm_feature_mode": "action_progress",
    })
    steerer.begin_hmm_observation()
    steerer.set_pending_control([0.25])
    steerer._edit(torch.tensor([[[-2.0, 0.0], [-2.0, 0.0]]]))
    steerer._edit(torch.tensor([[[2.0, 0.0], [2.0, 0.0]]]))
    assert steerer.n_hmm_belief_updates == 0
    steerer.finish_hmm_observation()
    assert steerer.n_hmm_observation_forwards == 2
    assert steerer.n_hmm_belief_updates == 1
    assert steerer.hmm_transition_index == 0
    assert steerer.hmm_belief[1] > 0.95
    steerer.set_pending_control([-0.25])
    steerer.begin_hmm_observation()
    steerer._edit(torch.tensor([[[-2.0, 0.0], [-2.0, 0.0]]]))
    steerer.finish_hmm_observation()
    assert steerer.hmm_transition_index == 1
    assert steerer.n_hmm_belief_updates == 2


def test_attention_mediator_restores_exact_paired_clean_output() -> None:
    class Predictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            attention = torch.nn.Linear(2, 2, bias=False)
            attention.weight.data.copy_(torch.eye(2))
            feed_forward = torch.nn.Identity()
            self.transformer = type("Transformer", (), {})()
            self.transformer.layers = [[attention, feed_forward]]
            self.attention = attention

        def forward(self, x):
            return self.attention(x)

    predictor = Predictor()
    state = type("SteererState", (), {"enabled": True})()

    def steering_hook(_module, _args, output):
        return output + 2.0 if state.enabled else output

    steering_handle = predictor.attention.register_forward_hook(steering_hook)
    with AttentionOutputMediator(predictor, "L00.attn_out", state):
        restored = predictor(torch.tensor([[1.0, 3.0]]))
    steering_handle.remove()
    assert torch.allclose(restored, torch.tensor([[1.0, 3.0]]))
