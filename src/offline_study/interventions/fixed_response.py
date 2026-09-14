"""Design 1 successor: fit the response inverse once, apply one thin map.

Not an equivalent implementation of the legacy candidate-specific support solver.
No online probes, native shadow, future targets, or coupled-path features.
"""
from __future__ import annotations

import math
import json
from pathlib import Path

import torch


METHOD = "fixed_response_rank4_v1"
ARMS = ("native", "zero_dose", "fixed_rank4", "matched_random_fixed_rank4")
ZERO_THRESHOLD = 1e-10


def load_fitted_bank(directory, *, task, checkpoint_sha256):
    """Require a completed, hash-bound fitting artifact; not a planning gate."""
    from offline_study.core.protocol import sha256
    directory = Path(directory)
    done = json.loads((directory / "DONE.json").read_text())
    files = ("contract.json", "cohort.json", "operator_bank.pt", "mean_responses.pt",
             "PARITY.json", "audit_metrics.json", "report.json")
    if (directory / "FAILED.json").exists() or done.get("status") != "fit_only_complete_not_behavioral_clearance":
        raise ValueError("Incomplete/failed successor fitting artifact")
    for name in files:
        if done.get(name) != sha256(directory / name):
            raise ValueError("Successor artifact checksum mismatch: " + name)
    report = json.loads((directory / "report.json").read_text())
    contract = json.loads((directory / "contract.json").read_text())
    bank = torch.load(directory / "operator_bank.pt", map_location="cpu", weights_only=True)
    binding = bank["binding"]
    if (binding != report["binding"] or binding != contract["binding"] or
            binding.get("method") != METHOD or binding.get("task") != task or
            binding.get("checkpoint_sha256") != checkpoint_sha256 or
            binding.get("cohort_sha256") != sha256(directory / "cohort.json") or
            report.get("development_or_protected_outcomes_accessed") is not False):
        raise ValueError("Successor model/task/input binding mismatch")
    validate_bank(bank)
    return bank


def compose_map(mean_response, error_weight):
    """R is [4, output], E is [4 features, output]; returns A [4, 4]."""
    if (mean_response.ndim != 2 or error_weight.shape != mean_response.shape or
            mean_response.shape[0] != 4 or mean_response.shape[1] < 1):
        raise ValueError("Expected four response directions and intercept-plus-three readout features")
    if not torch.isfinite(mean_response).all() or not torch.isfinite(error_weight).all():
        raise ValueError("Nonfinite response/readout")
    response, weight = mean_response.double(), error_weight.double()
    gram = response @ response.T
    damping = .01 * gram.diagonal().mean().clamp_min(1e-18)
    result = torch.linalg.solve(gram + damping * torch.eye(4, device=gram.device),
                                response @ weight.T)
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite fixed coefficient map")
    return result.float()


def make_bank(source, responses, dose, binding):
    """Responses are family-balanced means, NOT averages of fitted inverses."""
    target = source["support_bank"]["target"]
    result = {"schema_version": 1, "method": METHOD, "binding": binding,
              "dose": float(dose), "zero_threshold": ZERO_THRESHOLD,
              "mean": target["mean"].float().cpu(),
              "projection": target["basis"].float().cpu(),
              "scale": target["scale"].float().cpu(), "operators": {}}
    for arm, key in (("fixed_rank4", "rank"), ("matched_random_fixed_rank4", "rank_random")):
        spec = source["support_bank"]["bases"][key]
        if spec["blocks"] != [3] or spec["positions"] != list(range(256)):
            raise ValueError("Successor requires the original all-patch B3 support")
        basis = spec["basis"][:4, 0].float().cpu()
        result["operators"][arm] = {"basis": basis,
            "map": compose_map(responses[key], target["weight"].to(responses[key].device)).cpu()}
    validate_bank(result)
    return result


def validate_bank(bank):
    if bank.get("method") != METHOD or bank.get("schema_version") != 1:
        raise ValueError("Not a Design 1 bank")
    if (not math.isfinite(bank["dose"]) or bank["dose"] <= ZERO_THRESHOLD or
            bank.get("zero_threshold") != ZERO_THRESHOLD):
        raise ValueError("Invalid frozen dose or zero rule")
    if set(bank["operators"]) != set(ARMS[2:]):
        raise ValueError("Missing/extra registered operator")
    width = bank["mean"].numel()
    if (bank["mean"].shape != (width,) or width != 256 * 400 or
            bank["projection"].shape != (3, width) or bank["scale"].shape != (3,)):
        raise ValueError("Expected the MetaWorld B3 field and three-score projection")
    tensors = [bank[key] for key in ("mean", "projection", "scale")]
    for spec in bank["operators"].values():
        if spec["basis"].shape != (4, 256, 400) or spec["map"].shape != (4, 4):
            raise ValueError("Wrong thin-map dimensions")
        basis = spec["basis"].double().flatten(1)
        if not torch.allclose(basis @ basis.T, torch.eye(4, device=basis.device).double(), atol=2e-5, rtol=2e-5):
            raise ValueError("Output basis lost rank/orthonormality")
        tensors.extend(spec.values())
    if any(not torch.isfinite(t).all() for t in tensors) or not (bank["scale"] > 0).all():
        raise ValueError("Invalid fitted tensor")


class FixedResponseHook:
    """Read native H3 in the same rollout; edit B3 newest patches once.

    Banks must be staged once by FixedResponseIntervention, not copied per call.
    Energy tensors stay on-device until the caller requests an audit summary.
    """
    def __init__(self, predictor, bank, arm):
        self.predictor, self.bank, self.arm = predictor, bank, arm
        self.horizon, self.applications, self.handles = 0, 0, []
        self.record = None

    def __enter__(self):
        if len(self.predictor.predictor_blocks) != 6 or self.arm not in ARMS[2:]:
            raise ValueError("Unsupported predictor or active arm")
        # Coupling would invalidate the native-feature contract. Integration must
        # be standalone; do not silently compose this hook with another edit.
        if (self.predictor._forward_pre_hooks or self.predictor._forward_hooks or
                any(m._forward_hooks or m._forward_pre_hooks for m in self.predictor.modules()
                    if m is not self.predictor)):
            raise ValueError("Standalone native-feature map cannot coexist with predictor hooks")
        self.handles.append(self.predictor.register_forward_pre_hook(self._input))
        self.handles.append(self.predictor.predictor_blocks[3].register_forward_hook(self._output))
        return self

    def _input(self, module, args):
        self.horizon += 1

    def _output(self, module, args, output):
        if self.horizon != 3:
            return output
        if self.applications or output.ndim != 3 or output.shape[1] < 256 or output.shape[2] != 400:
            raise ValueError("Unexpected H3 field or repeated application")
        self.applications += 1
        spec = self.bank["operators"][self.arm]
        # Do not let the model's BF16 autocast silently change the coefficient
        # map arithmetic. Model output still returns in its original dtype.
        with torch.autocast(device_type=output.device.type, enabled=False):
            field = output[:, -256:]
            score = ((field.float().flatten(1) - self.bank["mean"]) @
                     self.bank["projection"].T) / self.bank["scale"]
            phi = torch.cat([torch.ones_like(score[:, :1]), score], 1)
            raw = phi @ spec["map"].T
            delta = (raw @ spec["basis"].flatten(1)).reshape_as(field)
            norm = delta.flatten(1).norm(dim=1)
            if not torch.isfinite(norm).all():
                raise ValueError("Nonfinite correction; no candidate may be discarded")
            active = norm > self.bank["zero_threshold"]
            factor = torch.where(active, self.bank["dose"] / norm.clamp_min(ZERO_THRESHOLD), 0.)
            delta = delta * factor[:, None, None]
            # Match the source probe hook's addition order: cast the field edit
            # to activation dtype before adding (including BF16 rounding).
            changed = field + delta.to(output.dtype)
            # Includes exact zero treatment even for negative zero/native BF16.
            changed = torch.where(active[:, None, None], changed, field)
            result = output.clone()
            result[:, -256:] = changed
            self.record = {"coefficients": (raw * factor[:, None]).detach(),
                "requested_l2": delta.flatten(1).norm(dim=1).detach(),
                "realized_l2": (changed.float() - field.float()).flatten(1).norm(dim=1).detach(),
                "active": active.detach()}
        return result

    def __exit__(self, kind, exc, traceback):
        for handle in self.handles:
            handle.remove()
        if kind is None and (self.horizon != 6 or self.applications != 1):
            raise ValueError("Expected exactly one B3/H3 edit in a full H6 rollout")


class FixedResponseIntervention:
    """One backend call per forecast; works with author and planning adapters.

    Runtime model/task/fit binding must be checked by the launcher's receipt
    validation. This class checks the tensor contract and does not grant access.
    """
    def __init__(self, backend, bank, arm):
        validate_bank(bank)
        if arm not in ARMS:
            raise ValueError("Unregistered Design 1 arm")
        if getattr(backend, "allow_tf32", False):
            raise ValueError("TF32 is not part of this contract")
        def stage(value):
            if isinstance(value, torch.Tensor):
                return value.to(device=backend.device, dtype=torch.float32)
            if isinstance(value, dict):
                return {k: stage(v) for k, v in value.items()}
            return value
        self.backend, self.bank, self.arm = backend, stage(bank), arm
        self.calls, self.last_record = 0, None

    @torch.no_grad()
    def __call__(self, context, act_suffix=None, **kwargs):
        if (kwargs or act_suffix is None or act_suffix.ndim != 3 or
                not 1 <= act_suffix.shape[0] <= 6 or act_suffix.shape[1] < 1):
            raise ValueError("Require explicit H1-H6 actions and unchanged rollout defaults")
        self.calls += 1
        self.last_record = {"response_probe_rollouts": 0, "native_shadow_rollouts": 0,
                            "backend_calls": 1, "horizon": len(act_suffix)}
        if len(act_suffix) < 6 or self.arm in ARMS[:2]:
            return self.backend.predict(context, act_suffix)
        with FixedResponseHook(self.backend.predictor, self.bank, self.arm) as hook:
            result = self.backend.predict(context, act_suffix)
        self.last_record.update(hook.record)
        if any(not torch.isfinite(result[key]).all() for key in ("visual", "proprio")):
            raise ValueError("Nonfinite predictions; no candidate may be discarded")
        return result
