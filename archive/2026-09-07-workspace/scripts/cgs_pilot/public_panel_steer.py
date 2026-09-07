#!/usr/bin/env python3
"""Panel P stage 3 steering driver: rerun official-evaluator episodes with ONE operator applied at
ONE predictor site (Label-first principle, amendment 3; COAST eq. 5). Everything else is the
released evaluator, planner and label. The operator acts on the last-frame (predicted) tokens of
every predictor forward (candidate batches and the final unroll alike), so the planner's own cost
ranking is what changes.

Arms (``--arm``):
  unsteered   no hook
  identity    hook installed, returns the tensor unchanged
  coast       h' = h M^T,  M = (1-beta) I + beta C_steer                 (equation reference)
  sham        same with the matched-spectrum random conceptor (COAST's control)
  caa         h' = h + beta * caa_norm * caa_direction                          (rank-one additive)
  wrong_site  coast operator of --operator-site applied at --site (a different block)
  outcome_success  M = (1-beta) I + beta C_success (positive-only conceptor, COAST's fallback)
  sonar_energy     proper mixture-energy metric trust-region edit
  sonar_energy_boundary boundary-normalized energy ablation
  sonar_ot         responsibility-weighted Gaussian-OT factor transport
  sonar_reverse    sign-reversed energy edit
  sonar_sham       metric-norm-matched random-coordinate energy edit
  sonar_coast_global factor-space global contrastive conceptor
  sonar_coast_static emission-routed regime-local contrastive conceptors
  sonar_coast_sham  same static beliefs/spectra with randomized orientations
  sonar_coast_label_shuffled episode-label-shuffled local conceptors

Operators come from ``public_panel_coast.py`` (``operators/<site>.npz``). Episode selection as in
``public_panel_capture.py`` (``--episodes`` prefix or ``--subset a-b``; every arm of a comparison
uses the same selection). Development sweeps run on fitting episodes only; the held-out opening is
a separate, authorised launch.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from public_panel_capture import infer_n_spatial, site_specs  # noqa: E402
from public_panel_eval import build_parser, install_logging_hooks, install_realization_hashing, resolve_config  # noqa: E402
from public_panel_factor_gate import summarize_action_chunks  # noqa: E402


def planner_to_executed_action_rows(
    actions,
    *,
    environment_action_dim: int,
    repeat_actskip: bool,
    action_skip: int,
    preprocessor,
) -> np.ndarray:
    """Mirror PlanEvaluator's exact action reshape/repeat/denormalization path."""
    rows = actions.detach().cpu() if hasattr(actions, "detach") else torch.as_tensor(actions)
    if rows.ndim == 1:
        rows = rows[None, :]
    if rows.ndim != 2 or rows.shape[-1] % int(environment_action_dim) != 0:
        raise ValueError(
            f"planner actions {tuple(rows.shape)} cannot be split into environment dim "
            f"{environment_action_dim}"
        )
    rows = rows.reshape(-1, int(environment_action_dim))
    if repeat_actskip:
        rows = rows.repeat_interleave(int(action_skip), dim=0)
    if preprocessor is not None:
        rows = preprocessor.denormalize_actions(rows)
    return rows.detach().float().cpu().numpy()


def load_operator(npz: Path, arm: str, beta: float, device, dtype) -> dict:
    z = np.load(npz)
    d = z["mean"].shape[0]
    eye = np.eye(d)
    if arm in ("coast", "wrong_site"):
        V, mu = z["steer_eigvecs"], z["steer_mu"]
    elif arm == "sham":
        V, mu = z["sham_eigvecs"], z["sham_mu"]
    elif arm == "outcome_success":
        V, mu = z["success_eigvecs"], z["success_mu"]
    elif arm == "caa":
        V, mu = None, None
    else:
        raise ValueError(arm)
    op = {"mean": torch.tensor(z["mean"], device=device, dtype=dtype)}
    if V is not None:
        C = (V.T * mu) @ V
        M = (1.0 - beta) * eye + beta * C
        op["M"] = torch.tensor(M, device=device, dtype=dtype)
        op["quota"] = float(np.sum(mu)) / d
    else:
        op["delta"] = torch.tensor(beta * float(z["caa_norm"]) * z["caa_direction"], device=device, dtype=dtype)
    return op


def load_sonar_operator(npz: Path, device) -> dict:
    z = np.load(npz)
    required = {
        "hidden_mean", "encoder", "decoder", "regime_weights", "regime_means", "regime_var",
        "success_weights", "success_means", "success_covariances", "failure_weights",
        "failure_means", "failure_covariances", "trust_metrics", "support_thresholds",
        "ot_affine", "ot_offset", "sham_rotation",
    }
    missing = sorted(required - set(z.files))
    if missing:
        raise SystemExit(f"Sonar operator {npz} is missing {missing}")
    out = {key: torch.as_tensor(z[key], device=device, dtype=torch.float32) for key in required}
    optional_tensors = {
        "global_contrastive_conceptor", "local_contrastive_conceptors",
        "local_matched_spectrum_conceptors", "global_label_shuffled_conceptor",
        "local_label_shuffled_conceptors", "hmm_initial", "hmm_transition",
        "action_hmm_initial", "action_hmm_transition_coef", "action_hmm_action_mean",
        "action_hmm_action_scale", "action_hmm_sequence_length", "action_hmm_means",
        "action_hmm_var",
    }
    out.update({
        key: torch.as_tensor(z[key], device=device, dtype=torch.float32)
        for key in optional_tensors if key in z.files
    })
    for key in z.files:
        if not key.startswith("frankenstein_"):
            continue
        value = np.asarray(z[key])
        if value.dtype.kind in "biufc":
            out[key] = torch.as_tensor(value, device=device, dtype=torch.float32)
        else:
            out[key] = str(value.item()) if value.ndim == 0 else value.astype(str).tolist()
    out["hmm_deployment_eligible"] = bool(
        z["hmm_deployment_eligible"].item() if "hmm_deployment_eligible" in z.files else False
    )
    routing_key = (
        "hmm_local_quality_routing_eligible"
        if "hmm_local_quality_routing_eligible" in z.files
        else "hmm_outcome_routing_eligible"
    )
    out["hmm_local_quality_routing_eligible"] = bool(
        z[routing_key].item() if routing_key in z.files else False
    )
    out["offline_hmm_gates_pass"] = bool(
        z["offline_hmm_gates_pass"].item() if "offline_hmm_gates_pass" in z.files else False
    )
    out["action_hmm_feature_mode"] = (
        str(z["action_hmm_feature_mode"].item()) if "action_hmm_feature_mode" in z.files else "unavailable"
    )
    for family in ("conceptor", "energy", "ot"):
        key = f"{family}_deployment_eligible"
        out[key] = bool(z[key].item()) if key in z.files else False
    out["regime_source"] = str(z["regime_source"].item()) if "regime_source" in z.files else "legacy"
    for outcome in ("success", "failure"):
        covariance = out[f"{outcome}_covariances"]
        out[f"{outcome}_precision"] = torch.linalg.inv(covariance)
        sign, logdet = torch.linalg.slogdet(covariance)
        if not bool(torch.all(sign > 0)):
            raise SystemExit(f"{outcome} covariance is not positive definite")
        out[f"{outcome}_logdet"] = logdet
    return out


class SonarSteerer:
    """Online factor-coordinate Sonar edit; fitting/selection never occurs here."""

    def __init__(
        self,
        predictor,
        n_spatial: int,
        site: str,
        arm: str,
        op: dict,
        beta: float,
        radius: float,
        displacement_mode: str,
        use_frankenstein: bool = False,
        frankenstein_ablate: set[str] | None = None,
        model_native_mode: str = "rowspace",
    ):
        self.predictor, self.n_spatial, self.site, self.arm = predictor, n_spatial, site, arm
        self.op, self.beta, self.radius, self.displacement_mode = op, float(beta), float(radius), displacement_mode
        self.use_frankenstein = bool(use_frankenstein)
        self.frankenstein_ablate = set(frankenstein_ablate or ())
        self.model_native_mode = str(model_native_mode)
        if self.model_native_mode not in {"rowspace", "complement"}:
            raise ValueError(f"invalid model-native mode {self.model_native_mode!r}")
        specs = {spec[0]: (spec[1], spec[2]) for spec in site_specs(predictor)}
        if site not in specs:
            raise SystemExit(f"unknown site {site}; known: {sorted(specs)}")
        self.module, self.kind = specs[site]
        self._handles = []
        self.n_applied = 0
        self.n_abstained_support = 0
        self.metric_norm_sum = 0.0
        self.metric_norm_max = 0.0
        self.n_supported = 0
        self.n_cap_hits = 0
        self.n_boundary_normalized = 0
        self.n_zero_mode_components = 0
        self.n_abstained_sparse = 0
        self.pattern_removed_norm_sum = 0.0
        self.model_native_removed_norm_sum = 0.0
        self.hmm_belief = None
        self.hmm_pending_control = None
        self.hmm_transition_index = 0
        self.observe_only = False
        self.hmm_observation = None
        self.n_hmm_observation_forwards = 0
        self.n_hmm_belief_updates = 0
        self.enabled = True

    def _frankenstein_enabled(self, module: str) -> bool:
        return self.use_frankenstein and module not in self.frankenstein_ablate

    def reset_episode_belief(self):
        self.hmm_belief = None
        self.hmm_pending_control = None
        self.hmm_transition_index = 0
        self.observe_only = False
        self.hmm_observation = None

    def begin_hmm_observation(self):
        if self.observe_only:
            raise RuntimeError("nested HMM observation replay")
        self.hmm_observation = None
        self.observe_only = True

    def finish_hmm_observation(self):
        self.observe_only = False
        if self.hmm_observation is None:
            raise RuntimeError("chosen-action replay produced no registered HMM activation")
        had_previous_belief = self.hmm_belief is not None
        belief, _mahalanobis = self._hmm_regime(self.hmm_observation)
        self.hmm_belief = belief.mean(0).detach()
        if had_previous_belief and self.hmm_pending_control is not None:
            self.hmm_transition_index += 1
        self.n_hmm_belief_updates += 1
        self.hmm_observation = None

    def cancel_hmm_observation(self):
        self.observe_only = False
        self.hmm_observation = None

    def set_pending_control(self, control):
        pending = torch.as_tensor(
            control, device=self.op["hidden_mean"].device, dtype=torch.float32
        ).reshape(-1)
        expected = int(self.op["action_hmm_action_mean"].numel())
        if pending.numel() != expected:
            raise ValueError(
                f"runtime control summary width {pending.numel()} != fitted HMM width {expected}"
            )
        self.hmm_pending_control = pending

    def _action_transition(self, batch_size):
        coefficient = self.op["action_hmm_transition_coef"]
        prior = self.op["action_hmm_initial"] if self.hmm_belief is None else self.hmm_belief
        prior = prior.reshape(1, -1).expand(batch_size, -1)
        if self.hmm_belief is None or self.hmm_pending_control is None:
            return prior
        if self.op.get("action_hmm_feature_mode") != "action_progress":
            raise RuntimeError("online HMM requires the registered action_progress feature map")
        scaled = (
            self.hmm_pending_control - self.op["action_hmm_action_mean"]
        ) / self.op["action_hmm_action_scale"]
        length = max(int(self.op["action_hmm_sequence_length"].item()) - 1, 1)
        progress = min(self.hmm_transition_index / length, 1.0)
        feature = torch.cat([
            torch.ones(1, device=scaled.device), scaled,
            torch.tensor([progress, progress * progress], device=scaled.device),
        ])
        logits = torch.einsum("ijp,p->ij", coefficient, feature)
        transition = torch.softmax(logits, dim=1)
        return prior @ transition

    def _hmm_regime(self, x):
        emission_x = x[:, : self.op["regime_means"].shape[1]]
        difference = emission_x[:, None, :] - self.op["regime_means"][None, :, :]
        variance = self.op["regime_var"]
        log_emission = -0.5 * (
            torch.log(2.0 * np.pi * variance)[None, :, :]
            + difference.square() / variance[None, :, :]
        ).sum(dim=2)
        prediction = self._action_transition(len(x))
        log_belief = torch.log(prediction + 1e-30) + log_emission
        belief = torch.softmax(log_belief, dim=1)
        mahalanobis = (difference.square() / variance[None, :, :]).sum(dim=2)
        return belief, mahalanobis

    def _sparse_support(self, x, regime):
        required = {
            "frankenstein_sparse_mean", "frankenstein_sparse_dictionary",
            "frankenstein_sparse_alpha", "frankenstein_sparse_active_count_bounds",
            "frankenstein_sparse_active_epsilon", "frankenstein_sparse_ista_iterations",
            "frankenstein_sparse_activation_probability_by_regime",
            "frankenstein_sparse_support_nll_q99_by_regime", "frankenstein_sparse_eligible",
        }
        if not self._frankenstein_enabled("sparse_superposition") or not required.issubset(self.op):
            return torch.ones(len(x), device=x.device, dtype=torch.bool)
        if not bool(self.op["frankenstein_sparse_eligible"].item()):
            return torch.ones(len(x), device=x.device, dtype=torch.bool)
        code = self._sparse_code(x)
        active_epsilon = self.op["frankenstein_sparse_active_epsilon"]
        count = (torch.abs(code) > active_epsilon).sum(1)
        active = torch.abs(code) > active_epsilon
        bounds = self.op["frankenstein_sparse_active_count_bounds"]
        supported = (count >= torch.floor(bounds[0])) & (count <= torch.ceil(bounds[1]))
        top = regime.argmax(1)
        probability = torch.clamp(
            self.op["frankenstein_sparse_activation_probability_by_regime"][top],
            min=1e-6,
            max=1.0 - 1e-6,
        )
        support_nll = -torch.sum(
            active * torch.log(probability) + (~active) * torch.log(1.0 - probability), dim=1
        )
        threshold_nll = self.op["frankenstein_sparse_support_nll_q99_by_regime"][top]
        supported = supported & (support_nll <= threshold_nll)
        self.n_abstained_sparse += int((~supported).sum().item())
        return supported

    def _sparse_code(self, x):
        """Exact torch counterpart of ``public_panel_frankenstein_math.encode_sparse``."""
        atoms = self.op["frankenstein_sparse_dictionary"]
        centered = x - self.op["frankenstein_sparse_mean"]
        gram = atoms @ atoms.T
        lipschitz = torch.clamp(torch.linalg.eigvalsh((gram + gram.T) / 2.0).max(), min=1e-8)
        step = 1.0 / lipschitz
        threshold = self.op["frankenstein_sparse_alpha"] * step
        code = torch.zeros((len(x), len(atoms)), device=x.device, dtype=x.dtype)
        for _ in range(int(self.op["frankenstein_sparse_ista_iterations"].item())):
            gradient = (code @ atoms - centered) @ atoms.T
            proposal = code - step * gradient
            code = torch.sign(proposal) * torch.clamp(torch.abs(proposal) - threshold, min=0.0)
        return code

    def _preserve_pattern(self, delta):
        required = {"frankenstein_pattern_basis", "frankenstein_pattern_eligible"}
        if not self._frankenstein_enabled("pattern_separation") or not required.issubset(self.op):
            return delta
        if not bool(self.op["frankenstein_pattern_eligible"].item()):
            return delta
        basis = self.op["frankenstein_pattern_basis"]
        if basis.numel() == 0:
            return delta
        component = (delta @ basis) @ basis.T
        self.pattern_removed_norm_sum += float(torch.linalg.vector_norm(component, dim=1).sum().item())
        return delta - component

    def _probability_support(self, x, regime):
        required = {
            "frankenstein_probability_geometry_eligible",
            "frankenstein_density_n_blocks",
            "frankenstein_density_block_bounds",
        }
        if not self._frankenstein_enabled("probability_geometry") or not required.issubset(self.op):
            return torch.ones(len(x), device=x.device, dtype=torch.bool)
        if not bool(self.op["frankenstein_probability_geometry_eligible"].item()):
            return torch.ones(len(x), device=x.device, dtype=torch.bool)
        top = regime.argmax(1)
        row = torch.arange(len(x), device=x.device)
        supported = torch.ones(len(x), device=x.device, dtype=torch.bool)
        n_blocks = int(self.op["frankenstein_density_n_blocks"].item())
        bounds = self.op["frankenstein_density_block_bounds"].to(dtype=torch.int64)
        for block in range(n_blocks):
            means = self.op[f"frankenstein_density_block{block}_means"]
            covariance = self.op[f"frankenstein_density_block{block}_covariances"]
            threshold = self.op[f"frankenstein_density_block{block}_support_q99"]
            start, stop = int(bounds[block, 0].item()), int(bounds[block, 1].item())
            difference = x[:, start:stop] - means[top]
            precision = torch.linalg.inv(covariance[top])
            mahalanobis = torch.einsum("bi,bij,bj->b", difference, precision, difference)
            supported = supported & (mahalanobis <= threshold[top])
        return supported

    def _model_native_delta(self, delta, metric):
        required = {"frankenstein_model_native_eligible", "frankenstein_model_native_rowspace_basis"}
        if not self._frankenstein_enabled("model_native_coordinates") or not required.issubset(self.op):
            return delta
        if not bool(self.op["frankenstein_model_native_eligible"].item()):
            return delta
        basis = self.op["frankenstein_model_native_rowspace_basis"]
        rowspace = (delta @ basis) @ basis.T
        projected = rowspace if self.model_native_mode == "rowspace" else delta - rowspace
        removed = delta - projected
        self.model_native_removed_norm_sum += float(torch.linalg.vector_norm(removed, dim=1).sum().item())
        # Preserve the already selected latent metric dose when the native
        # component is nonzero; zero overlap remains a principled abstention.
        target_norm = self._metric_norm(delta, metric)
        projected_norm = self._metric_norm(projected, metric)
        live = projected_norm > 1e-10
        scale = torch.where(
            live,
            target_norm / torch.clamp(projected_norm, min=1e-30),
            torch.zeros_like(projected_norm),
        )
        return projected * scale[:, None]

    @staticmethod
    def _log_gaussian(x, means, precision, logdet):
        difference = x[:, None, :] - means[None, :, :]
        quadratic = torch.einsum("bkd,kde,bke->bk", difference, precision, difference)
        dimension = x.shape[1]
        return -0.5 * (dimension * np.log(2.0 * np.pi) + logdet[None, :] + quadratic)

    def _outcome_gradient(self, x, name):
        means = self.op[f"{name}_means"]
        precision = self.op[f"{name}_precision"]
        log_joint = (
            torch.log(self.op[f"{name}_weights"] + 1e-30)[None, :]
            + self._log_gaussian(x, means, precision, self.op[f"{name}_logdet"])
        )
        responsibility = torch.softmax(log_joint, dim=1)
        difference = x[:, None, :] - means[None, :, :]
        component_gradient = torch.einsum("bkd,kde->bke", difference, precision)
        return torch.einsum("bk,bkd->bd", responsibility, component_gradient)

    def _regime(self, x):
        if self.arm == "sonar_coast_hmm":
            # One causal belief routes the complete CEM computation at a physical
            # replan. Candidate optimizer samples are not chronological HMM
            # observations and cannot advance or independently redefine b_t.
            prior = self.op["action_hmm_initial"] if self.hmm_belief is None else self.hmm_belief
            responsibility = prior.reshape(1, -1).expand(len(x), -1)
            emission_x = x[:, : self.op["regime_means"].shape[1]]
            difference = emission_x[:, None, :] - self.op["regime_means"][None, :, :]
            mahalanobis = (difference.square() / self.op["regime_var"][None, :, :]).sum(dim=2)
            return responsibility, mahalanobis
        emission_x = x[:, : self.op["regime_means"].shape[1]]
        difference = emission_x[:, None, :] - self.op["regime_means"][None, :, :]
        variance = self.op["regime_var"]
        log_joint = (
            torch.log(self.op["regime_weights"] + 1e-30)[None, :]
            - 0.5 * (
                torch.log(2.0 * np.pi * variance)[None, :, :]
                + difference.square() / variance[None, :, :]
            ).sum(dim=2)
        )
        responsibility = torch.softmax(log_joint, dim=1)
        mahalanobis = (difference.square() / variance[None, :, :]).sum(dim=2)
        return responsibility, mahalanobis

    @staticmethod
    def _metric_norm(delta, metric):
        return torch.sqrt(torch.clamp(torch.einsum("bi,bij,bj->b", delta, metric, delta), min=1e-30))

    def _factor_delta(self, x):
        regime, mahalanobis = self._regime(x)
        top = regime.argmax(dim=1)
        row = torch.arange(len(x), device=x.device)
        supported = mahalanobis[row, top] <= self.op["support_thresholds"][top]
        metric = torch.einsum("bk,kde->bde", regime, self.op["trust_metrics"])
        if self.arm.startswith("sonar_coast_"):
            key = {
                "sonar_coast_global": "global_contrastive_conceptor",
                "sonar_coast_static": "local_contrastive_conceptors",
                "sonar_coast_hmm": "local_contrastive_conceptors",
                "sonar_coast_sham": "local_matched_spectrum_conceptors",
                "sonar_coast_label_shuffled": "local_label_shuffled_conceptors",
            }.get(self.arm)
            if key is None:
                raise RuntimeError(f"unsupported or ineligible conceptor arm {self.arm}")
            if key not in self.op:
                raise RuntimeError(f"Sonar artifact has no {key}")
            if self.arm == "sonar_coast_global":
                operator = self.op[key].expand(len(x), -1, -1)
            else:
                operator = torch.einsum("bk,kde->bde", regime, self.op[key])
            eye = torch.eye(x.shape[1], device=x.device, dtype=x.dtype).expand(len(x), -1, -1)
            gate = (1.0 - self.beta) * eye + self.beta * operator
            edited = torch.einsum("bi,bji->bj", x, gate)
            delta = edited - x
        elif self.arm == "sonar_ot":
            # ``ot_offset`` already contains target_mean - A*source_mean; use
            # x*A^T + offset directly to avoid an implicit regime-center change.
            transported = torch.einsum("bd,ked->bke", x, self.op["ot_affine"]) + self.op["ot_offset"][None, :, :]
            delta = self.beta * torch.einsum("bk,bkd->bd", regime, transported - x[:, None, :])
        else:
            gradient = self._outcome_gradient(x, "success") - self._outcome_gradient(x, "failure")
            inverse_gradient = torch.linalg.solve(metric, gradient.unsqueeze(-1)).squeeze(-1)
            denominator = torch.sqrt(torch.clamp((gradient * inverse_gradient).sum(dim=1), min=1e-30))
            if self.arm == "sonar_energy_boundary":
                delta = -self.radius * inverse_gradient / denominator[:, None]
                delta = self.beta * delta
                self.n_boundary_normalized += int(supported.sum().item())
            else:
                delta = -self.beta * inverse_gradient

        if not self.arm.startswith("sonar_coast_"):
            target_norm = self._metric_norm(delta, metric)
            norm_x = torch.clamp(torch.linalg.vector_norm(x, dim=1, keepdim=True), min=1e-8)
            unit = x / norm_x
            radial = (unit * delta).sum(dim=1, keepdim=True) * unit
            if self.displacement_mode == "radial":
                delta = radial
            elif self.displacement_mode == "angular":
                delta = delta - radial
            if self.displacement_mode != "joint":
                component_norm = self._metric_norm(delta, metric)
                live = component_norm > 1e-10
                self.n_zero_mode_components += int((~live).sum().item())
                scale = torch.where(
                    live,
                    target_norm / torch.clamp(component_norm, min=1e-30),
                    torch.zeros_like(component_norm),
                )
                delta = delta * scale[:, None]
        if self.arm == "sonar_reverse":
            delta = -delta
        elif self.arm == "sonar_sham":
            target_norm = self._metric_norm(delta, metric)
            delta = delta @ self.op["sham_rotation"]
            sham_norm = self._metric_norm(delta, metric)
            delta = delta * (target_norm / torch.clamp(sham_norm, min=1e-30))[:, None]
        if self.use_frankenstein:
            supported = supported & self._probability_support(x, regime) & self._sparse_support(x, regime)
            delta = self._model_native_delta(delta, metric)
            delta = self._preserve_pattern(delta)
        delta = delta * supported[:, None]
        # One universal cap is applied to the *final* proposal, after every
        # support filter, factorization, rotation, and Frankenstein modifier.
        # This makes cap statistics comparable across energy, OT, and conceptor
        # arms and prevents an upstream component audit from hiding saturation
        # introduced by the actual runtime composition.
        proposal_norm = self._metric_norm(delta, metric)
        self.n_cap_hits += int(((proposal_norm > self.radius) & supported).sum().item())
        cap = torch.clamp(self.radius / torch.clamp(proposal_norm, min=1e-30), max=1.0)
        delta = delta * cap[:, None]
        self.n_supported += int(supported.sum().item())
        self.n_abstained_support += int((~supported).sum().item())
        final_norm = self._metric_norm(delta, metric)
        self.metric_norm_sum += float(final_norm.sum().item())
        self.metric_norm_max = max(self.metric_norm_max, float(final_norm.max().item()))
        return delta

    def _edit(self, hidden):
        n = min(self.n_spatial, hidden.shape[1])
        tail = hidden[:, -n:]
        pooled = tail.float().mean(dim=1)
        x = (pooled - self.op["hidden_mean"]) @ self.op["encoder"]
        if self.observe_only:
            # DINO-WM calls the predictor once per imagined action step.  Match
            # BlockCapture.end_call by retaining only the last predictor output,
            # then commit one causal belief after the whole physical-plan replay.
            self.hmm_observation = x.mean(0, keepdim=True).detach()
            self.n_hmm_observation_forwards += 1
            return hidden
        delta_factor = self._factor_delta(x)
        delta_hidden = delta_factor @ self.op["decoder"]
        edited = tail + delta_hidden[:, None, :].to(tail.dtype)
        self.n_applied += int(len(hidden))
        return torch.cat([hidden[:, :-n], edited], dim=1)

    def _hook(self, module, args, out):
        if not self.enabled:
            return out
        tensor = out[0] if isinstance(out, (tuple, list)) else out
        if not hasattr(tensor, "shape") or tensor.shape[-1] != self.op["hidden_mean"].shape[0]:
            return out
        if self.kind == "ff_resid":
            source = args[0]
            edited = self._edit((tensor + source).reshape(tensor.shape[0], -1, tensor.shape[-1])).reshape(tensor.shape) - source
        else:
            edited = self._edit(tensor.reshape(tensor.shape[0], -1, tensor.shape[-1])).reshape(tensor.shape)
        if isinstance(out, tuple):
            return (edited,) + tuple(out[1:])
        if isinstance(out, list):
            return [edited] + list(out[1:])
        return edited

    def __enter__(self):
        self._handles.append(self.module.register_forward_hook(self._hook))
        return self

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()


class AttentionOutputMediator:
    """Paired clean-attention restoration for a deterministic predictor forward.

    A clean duplicate forward is run on the exact same predictor inputs with the
    Sonar hook disabled.  The selected attention output from the actual steered
    forward is then replaced by the paired clean value.  This is a mediation
    ablation, not an attention weighting heuristic.
    """

    def __init__(self, predictor, site: str, steerer: SonarSteerer, strength: float = 1.0):
        if not 0.0 <= strength <= 1.0:
            raise ValueError("attention restoration strength must lie in [0,1]")
        specs = {name: module for name, module, _kind in site_specs(predictor)}
        if site not in specs or not site.endswith(".attn_out"):
            raise ValueError(f"attention restore site must be an attn_out site, got {site!r}")
        self.predictor = predictor
        self.module = specs[site]
        self.steerer = steerer
        self.strength = float(strength)
        self.clean_phase = False
        self.recursing = False
        self.clean_output = None
        self.n_restored = 0
        self._handles = []

    def _attention_hook(self, _module, _args, output):
        tensor = output[0] if isinstance(output, (tuple, list)) else output
        if self.clean_phase:
            self.clean_output = tensor.detach()
            return output
        if self.clean_output is None:
            raise RuntimeError("steered attention forward has no paired clean output")
        restored = tensor + self.strength * (self.clean_output.to(tensor.dtype) - tensor)
        self.clean_output = None
        self.n_restored += 1
        if isinstance(output, tuple):
            return (restored,) + tuple(output[1:])
        if isinstance(output, list):
            return [restored] + list(output[1:])
        return restored

    def _predictor_pre_hook(self, module, args):
        if self.recursing:
            return None
        self.recursing = True
        self.clean_phase = True
        previous_enabled = self.steerer.enabled
        self.steerer.enabled = False
        try:
            with torch.no_grad():
                module(*args)
        finally:
            self.steerer.enabled = previous_enabled
            self.clean_phase = False
            self.recursing = False
        if self.clean_output is None:
            raise RuntimeError("clean duplicate forward did not reach the requested attention site")
        return None

    def __enter__(self):
        # Sonar's hook is registered first, so this hook observes and replaces
        # its downstream effect when both target the same module.
        self._handles.append(self.module.register_forward_hook(self._attention_hook))
        self._handles.append(self.predictor.register_forward_pre_hook(self._predictor_pre_hook))
        return self

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        self._handles = []


class Steerer:
    def __init__(self, predictor, n_spatial: int, site: str, arm: str, op: dict | None):
        self.predictor, self.n_spatial, self.site, self.arm, self.op = predictor, n_spatial, site, arm, op
        specs = {s[0]: (s[1], s[2]) for s in site_specs(predictor)}
        if site not in specs:
            raise SystemExit(f"unknown site {site}; known: {sorted(specs)}")
        self.module, self.kind = specs[site]
        self._handles = []
        self.n_applied = 0

    def _edit(self, h: torch.Tensor) -> torch.Tensor:
        """h [B, N, d] -> edited on the last n_spatial tokens."""
        if self.arm == "identity":
            return h
        n = self.n_spatial
        tail = h[:, -n:]
        if "M" in self.op:
            new = tail @ self.op["M"].T
        else:
            new = tail + self.op["delta"]
        self.n_applied += 1
        return torch.cat([h[:, :-n], new.to(h.dtype)], dim=1)

    def _hook(self, module, args, out):
        if self.arm == "identity":
            return None
        if self.kind == "ff_resid":  # resid_post of a DINO-WM block = ff(x) + x; edit the residual, return ff part
            x = args[0]
            resid = out + x
            shape = resid.shape
            flat = resid.reshape(shape[0], -1, shape[-1])
            return (self._edit(flat).reshape(shape) - x)
        shape = out.shape
        flat = out.reshape(shape[0], -1, shape[-1])
        return self._edit(flat).reshape(shape)

    def __enter__(self):
        self._handles.append(self.module.register_forward_hook(self._hook))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()


def main() -> int:
    ap = build_parser(__doc__)
    ap.add_argument("--arm", required=True, choices=[
        "unsteered", "identity", "coast", "sham", "caa", "wrong_site", "outcome_success",
        "sonar_energy", "sonar_energy_boundary", "sonar_ot", "sonar_reverse", "sonar_sham",
        "sonar_coast_global", "sonar_coast_static", "sonar_coast_hmm", "sonar_coast_sham",
        "sonar_coast_label_shuffled",
    ])
    ap.add_argument("--site", default=None, help="site the hook is installed at")
    ap.add_argument("--operator-dir", type=Path, default=None, help="public_panel_coast.py operators/ dir")
    ap.add_argument("--operator-site", default=None, help="site whose operator is used (default --site; differs for wrong_site)")
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--sonar-operator", type=Path, default=None)
    ap.add_argument("--trust-radius", type=float, default=0.25)
    ap.add_argument("--displacement-mode", choices=["angular", "radial", "joint"], default="joint")
    ap.add_argument(
        "--frankenstein-mode", choices=["off", "admitted"], default="off",
        help="apply only sparse-support/pattern modules whose serialized development gates passed",
    )
    ap.add_argument(
        "--frankenstein-ablate",
        action="append",
        default=[],
        choices=[
            "probability_geometry",
            "sparse_superposition",
            "pattern_separation",
            "model_native_coordinates",
        ],
        help="development mechanism ablation; repeat to disable multiple admitted modules",
    )
    ap.add_argument(
        "--model-native-mode",
        choices=["rowspace", "complement"],
        default="rowspace",
        help="use the admitted relational row space or its dose-matched complement control",
    )
    ap.add_argument(
        "--attention-restore-site", default=None,
        help="development mediation ablation: restore this attn_out from an exact clean duplicate forward",
    )
    ap.add_argument("--attention-restore-strength", type=float, default=1.0)
    ap.add_argument("--subset", default=None)
    args = ap.parse_args()
    args.frankenstein_ablate = sorted(set(args.frankenstein_ablate))
    sonar_arm = args.arm.startswith("sonar_")
    if args.attention_restore_site is not None and not sonar_arm:
        raise SystemExit("attention restoration is defined only as a Sonar mediation ablation")
    if args.attention_restore_site is not None and args.frankenstein_mode != "admitted":
        raise SystemExit("attention restoration requires --frankenstein-mode admitted")
    if args.frankenstein_ablate and args.frankenstein_mode != "admitted":
        raise SystemExit("--frankenstein-ablate requires --frankenstein-mode admitted")
    if args.model_native_mode == "complement" and args.frankenstein_mode != "admitted":
        raise SystemExit("the model-native complement is defined only for admitted Frankenstein arms")
    if sonar_arm:
        if args.sonar_operator is None:
            raise SystemExit("Sonar arms require --sonar-operator")
        with np.load(args.sonar_operator) as sonar_meta:
            artifact_site = str(sonar_meta["site"].item())
        if args.site is None:
            args.site = artifact_site
        if args.site != artifact_site:
            raise SystemExit(f"Sonar artifact site {artifact_site} does not match --site {args.site}")
    if args.episode_manifest is not None and args.subset:
        raise SystemExit("--subset and --episode-manifest are mutually exclusive")
    subset = None
    if args.subset:
        a, b = args.subset.split("-")
        subset = (int(a), int(b))
    extra = [{"field": "steering arm", "from": "none", "to": f"{args.arm} site={args.site} op_site={args.operator_site or args.site} beta={args.beta} radius={args.trust_radius} mode={args.displacement_mode} frankenstein={args.frankenstein_mode} ablate={args.frankenstein_ablate} native={args.model_native_mode}", "reason": "stage-3 operator table (development); evaluator unchanged"}]
    if subset:
        extra.append({"field": "episode selection", "from": "all", "to": f"subset {subset}", "reason": "held-out / fitting episode family"})
    repo, params, meta, out, jsonl_path = resolve_config(args, extra_deviations=extra)
    meta.update({"arm": args.arm, "site": args.site, "operator_site": args.operator_site or args.site, "beta": args.beta,
                 "trust_radius": args.trust_radius, "displacement_mode": args.displacement_mode,
                 "frankenstein_mode": args.frankenstein_mode,
                 "frankenstein_ablate": args.frankenstein_ablate,
                 "model_native_mode": args.model_native_mode,
                 "attention_restore_site": args.attention_restore_site,
                 "attention_restore_strength": args.attention_restore_strength})

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    log = logging.getLogger("panelP.steer")
    from src.utils.distributed import init_distributed  # noqa: E402
    from evals.scaffold import main as eval_main  # noqa: E402
    import evals.simu_env_planning.planning.plan_evaluator as pe  # noqa: E402

    episode_rows = getattr(args, "_episode_rows", None)
    install_realization_hashing(pe)
    counter = install_logging_hooks(pe, jsonl_path, meta, episode_rows=episode_rows)
    holder: dict = {}
    orig_init = pe.PlanEvaluator.__init__
    orig_eval = pe.PlanEvaluator.eval

    def init(self, cfg, agent):
        orig_init(self, cfg, agent)
        if args.arm == "unsteered":
            return
        predictor = getattr(agent.model, "predictor", None)
        if predictor is None:
            predictor = getattr(getattr(agent.model, "model", None), "predictor", None)
        if predictor is None:
            raise SystemExit("agent model exposes no predictor")
        gh, gw = getattr(predictor, "grid_height", None), getattr(predictor, "grid_width", None)
        n_spatial = int(gh * gw) if gh and gw else infer_n_spatial(predictor)
        p = next(predictor.parameters())
        op = None
        if sonar_arm:
            op = load_sonar_operator(args.sonar_operator, p.device)
            if args.frankenstein_mode == "admitted":
                required = {
                    "frankenstein_probability_geometry_eligible",
                    "frankenstein_sparse_eligible",
                    "frankenstein_pattern_eligible",
                    "frankenstein_attention_hopfield_eligible",
                    "frankenstein_model_native_eligible",
                }
                missing = sorted(required - set(op))
                if missing:
                    raise SystemExit(
                        f"Frankenstein mode requires a bound module artifact; missing {missing}"
                    )
            family = (
                "conceptor" if args.arm.startswith("sonar_coast_")
                else "ot" if args.arm == "sonar_ot"
                else "energy"
            )
            if not op[f"{family}_deployment_eligible"]:
                raise SystemExit(
                    f"{args.arm} is fail-closed: the serialized {family} stability gate did not pass"
                )
            if args.arm == "sonar_coast_hmm":
                binding_keys = {"action_hmm_means", "action_hmm_var"}
                if not binding_keys.issubset(op):
                    raise SystemExit("sonar_coast_hmm lacks its shared-emission binding")
                if not (
                    torch.allclose(op["action_hmm_means"], op["regime_means"], atol=1e-6, rtol=1e-6)
                    and torch.allclose(op["action_hmm_var"], op["regime_var"], atol=1e-6, rtol=1e-6)
                ):
                    raise SystemExit("sonar_coast_hmm transition and local-operator regimes do not match")
                if not (
                    op["hmm_deployment_eligible"]
                    and op["offline_hmm_gates_pass"]
                    and op["hmm_local_quality_routing_eligible"]
                ):
                    raise SystemExit(
                        "sonar_coast_hmm is fail-closed: the action-conditioned, progress, "
                        "Chapman-Kolmogorov, local-quality-routing, and treatment-separation gates have not passed"
                    )
            if args.arm.startswith("sonar_coast_") and args.displacement_mode != "joint":
                raise SystemExit("conceptor arms require --displacement-mode joint")
            if args.attention_restore_site is not None:
                expected_attention_site = op.get("frankenstein_attention_restore_site")
                if expected_attention_site != args.attention_restore_site:
                    raise SystemExit(
                        f"attention restoration site {args.attention_restore_site!r} does not match "
                        f"the gated site {expected_attention_site!r}"
                    )
                if not bool(op["frankenstein_attention_hopfield_eligible"]):
                    raise SystemExit("attention/Hopfield mediation is fail-closed: its Q/K gate did not pass")
            holder["steerer"] = SonarSteerer(
                predictor, n_spatial, args.site, args.arm, op, args.beta, args.trust_radius,
                args.displacement_mode, use_frankenstein=args.frankenstein_mode == "admitted",
                frankenstein_ablate=set(args.frankenstein_ablate),
                model_native_mode=args.model_native_mode,
            ).__enter__()
            if args.attention_restore_site is not None:
                holder["attention_mediator"] = AttentionOutputMediator(
                    predictor,
                    args.attention_restore_site,
                    holder["steerer"],
                    args.attention_restore_strength,
                ).__enter__()
            log.info(
                f"[panelP] arm={args.arm} site={args.site} beta={args.beta} "
                f"radius={args.trust_radius} mode={args.displacement_mode} n_spatial={n_spatial}"
            )
            return
        if args.arm != "identity":
            npz = args.operator_dir / f"{args.operator_site or args.site}.npz"
            op = load_operator(npz, args.arm, args.beta, p.device, p.dtype)
        holder["steerer"] = Steerer(predictor, n_spatial, args.site, args.arm, op).__enter__()
        log.info(f"[panelP] arm={args.arm} site={args.site} beta={args.beta} quota={op.get('quota') if op else None} n_spatial={n_spatial}")

    def eval_ep(self, cfg, agent, env, task_idx=-1, ep=0):
        if subset is not None and not (subset[0] <= int(ep) <= subset[1]):
            return (0, 0, float("nan"), float("nan"), float("nan"), -1.0, -1.0, -1.0, float("nan"), -1.0, -1.0)
        if args.arm == "sonar_coast_hmm" and "steerer" in holder:
            holder["steerer"].reset_episode_belief()
        return orig_eval(self, cfg, agent, env, task_idx=task_idx, ep=ep)

    pe.PlanEvaluator.__init__ = init
    pe.PlanEvaluator.eval = eval_ep
    if args.arm == "sonar_coast_hmm":
        from evals.simu_env_planning.planning.gc_agent import GC_Agent

        original_plan = GC_Agent.plan

        def plan_with_belief_commit(self, z, steps_left=None):
            actions = original_plan(self, z, steps_left=steps_left)
            steerer = holder.get("steerer")
            if steerer is None:
                raise RuntimeError("HMM steerer was not installed before the first plan call")
            action_tensor = actions
            if action_tensor.dim() == 2:
                action_tensor = action_tensor.unsqueeze(1)
            elif action_tensor.dim() == 1:
                action_tensor = action_tensor.view(1, 1, -1)
            action_rows = planner_to_executed_action_rows(
                actions,
                environment_action_dim=int(self.cfg.action_dim),
                repeat_actskip=bool(self.cfg.planner.repeat_actskip),
                action_skip=int(self.model.action_skip),
                preprocessor=self.preprocessor,
            )
            control = summarize_action_chunks(
                action_rows, np.asarray([0, len(action_rows)], dtype=np.int64)
            )[0]
            # The selected chunk leads into the endpoint activation captured by
            # the replay below. The initial endpoint still uses the initial prior.
            steerer.set_pending_control(control)
            steerer.begin_hmm_observation()
            try:
                with torch.no_grad():
                    self.model.unroll(z, action_tensor)
            except Exception:
                steerer.cancel_hmm_observation()
                raise
            steerer.finish_hmm_observation()
            return actions

        GC_Agent.plan = plan_with_belief_commit
    if episode_rows is not None:
        from public_panel_eval import install_manifest_seeding

        install_manifest_seeding(pe, episode_rows)
    init_distributed(rank_and_world_size=(0, 1))
    t0 = time.time()
    eval_main(params["eval_name"], args_eval=params)
    if "steerer" in holder:
        holder["steerer"].__exit__()
    if "attention_mediator" in holder:
        holder["attention_mediator"].__exit__()
    expected_total = len(episode_rows) if episode_rows is not None else int(params["meta"]["eval_episodes"])
    expected_logged = (
        sum(subset[0] <= episode < min(subset[1] + 1, expected_total) for episode in range(expected_total))
        if subset is not None else expected_total
    )
    done = {
        "finished_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(), "arm": args.arm, "site": args.site, "beta": args.beta,
        "operator_site": args.operator_site or args.site, "subset": subset, "n_logged": counter["n"], "n_success": counter["n_success"],
        "expected_episodes": expected_logged,
        "complete": counter["n"] == expected_logged,
        "n_hook_applications": holder["steerer"].n_applied if "steerer" in holder else 0, "wall_s": round(time.time() - t0, 1),
    }
    if sonar_arm and "steerer" in holder:
        steerer = holder["steerer"]
        done.update({
            "n_abstained_support": steerer.n_abstained_support,
            "n_supported": steerer.n_supported,
            "n_cap_hits": steerer.n_cap_hits,
            "cap_hit_fraction_supported": steerer.n_cap_hits / max(steerer.n_supported, 1),
            "n_boundary_normalized": steerer.n_boundary_normalized,
            "n_zero_mode_components": steerer.n_zero_mode_components,
            "n_abstained_sparse": steerer.n_abstained_sparse,
            "mean_pattern_component_removed": steerer.pattern_removed_norm_sum / max(steerer.n_applied, 1),
            "mean_model_native_removed_component": steerer.model_native_removed_norm_sum / max(steerer.n_applied, 1),
            "frankenstein_mode": args.frankenstein_mode,
            "frankenstein_ablate": args.frankenstein_ablate,
            "model_native_mode": args.model_native_mode,
            "n_hmm_belief_updates": steerer.n_hmm_belief_updates,
            "n_hmm_observation_forwards": steerer.n_hmm_observation_forwards,
            "hmm_belief_commit": (
                "one unedited chosen-action unroll after each physical plan call"
                if args.arm == "sonar_coast_hmm" else "not_applicable"
            ),
            "mean_metric_edit_norm": steerer.metric_norm_sum / max(steerer.n_applied, 1),
            "max_metric_edit_norm": steerer.metric_norm_max,
            "trust_radius": args.trust_radius,
            "displacement_mode": args.displacement_mode,
            "attention_restore_site": args.attention_restore_site,
            "n_attention_outputs_restored": (
                holder["attention_mediator"].n_restored if "attention_mediator" in holder else 0
            ),
        })
    (out / ("DONE.json" if done["complete"] else "INCOMPLETE.json")).write_text(json.dumps(done, indent=2))
    print(json.dumps(done))
    return 0 if done["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
