"""Frozen coordinate-calibrated Sonar pilot, not native-coordinate steering.

The observer and predictor-coordinate readout are independent learned maps.
The low-rank covariance-weighted residual solve is followed by a Euclidean
radial cap; this cap is NOT an exact constrained trust-region solution.
"""
from __future__ import annotations

import torch


def calibration_matrix(covariance, jacobian, regularization=.1):
    """K = Sigma J.T (J Sigma J.T + lambda I)^-1 in RAW visual units."""
    if regularization <= 0 or covariance.ndim != 2 or jacobian.ndim != 2:
        raise ValueError("Positive regularization and matrix inputs required")
    if covariance.shape[0] != covariance.shape[1] or covariance.shape[0] != jacobian.shape[1]:
        raise ValueError("Covariance/Jacobian shape mismatch")
    covariance = (covariance + covariance.T) / 2
    # Cholesky both verifies a valid precision metric and avoids explicit inverses.
    torch.linalg.cholesky(covariance)
    cross = covariance @ jacobian.T
    system = jacobian @ cross + regularization * torch.eye(jacobian.shape[0], dtype=jacobian.dtype, device=jacobian.device)
    return torch.cholesky_solve(cross.T, torch.linalg.cholesky(system)).T


def radial_cap(delta, radius):
    if radius <= 0:
        raise ValueError("Positive fixed cap radius required")
    norms = delta.norm(dim=-1)
    scale = torch.clamp(radius / norms.clamp_min(torch.finfo(delta.dtype).tiny), max=1)
    return delta * scale[..., None], norms > radius


def rotated_sham(delta):
    """Fixed diagonal orthogonal rotation; 192 sign flips for384 visual channels.

    Its determinant is+1 and its coordinatewise squares (hence raw norm) are
    bit-exact, avoiding a changed floating-point reduction order from shuffling.
    """
    return delta * torch.where(
        torch.arange(delta.shape[-1], device=delta.device) % 2 == 0,
        torch.ones((), device=delta.device, dtype=delta.dtype),
        -torch.ones((), device=delta.device, dtype=delta.dtype))


class CoordinateCalibration:
    def __init__(self, artifact, device="cpu", dtype=torch.float32):
        names = ("p3_mean", "p3_scale", "p3_coef", "p3_intercept", "q_mean", "q_scale",
                 "observer_mean", "observer_scale", "observer_coef", "observer_intercept", "correction_matrix")
        self.values = {name: torch.as_tensor(artifact[name], dtype=dtype, device=device) for name in names}
        self.radius, self.beta = float(artifact["cap_radius"]), float(artifact["beta"])
        if self.beta < 0 or self.radius <= 0 or any(not torch.isfinite(x).all() for x in self.values.values()):
            raise ValueError("Malformed frozen calibration")
        if any((self.values[key] <= 0).any() for key in ("p3_scale", "q_scale", "observer_scale")):
            raise ValueError("Frozen scales must be positive")

    def observer_physical(self, visual):
        v = self.values
        return ((visual-v["observer_mean"])/v["observer_scale"]) @ v["observer_coef"].T + v["observer_intercept"]

    def delta(self, p3, visual, *, beta=None, sham=False):
        if p3.ndim != 2 or visual.ndim != 2 or len(p3) != len(visual):
            raise ValueError("Expected aligned [batch,pooled channels] inputs")
        v = self.values
        predicted_q_std = ((p3-v["p3_mean"])/v["p3_scale"]) @ v["p3_coef"].T + v["p3_intercept"]
        observed_q_std = (self.observer_physical(visual)-v["q_mean"])/v["q_scale"]
        residual = predicted_q_std-observed_q_std
        beta = self.beta if beta is None else beta
        requested = beta * (residual @ v["correction_matrix"].T)
        delta, saturated = radial_cap(requested, self.radius)
        return (rotated_sham(delta) if sham else delta), saturated, residual


class CoordinateCalibrationHook:
    """Retain newest P3 tokens and edit only returned predicted visual features.

    Every imagined step is calibrated, including recursive CEM predictions.
    No raw observations, goals, simulator data, or future labels enter the hook.
    """
    def __init__(self, wm, block, calibration, *, beta=None, sham=False):
        self.wm = wm if hasattr(wm, "forward_pred") else wm.model
        self.block, self.calibration = block, calibration
        self.beta, self.sham = beta, sham
        self.original = self.wm.forward_pred
        self.pending, self.handle = None, None
        self.calls = self.candidates = self.saturated = 0
        self.norm_sum = self.max_norm = self.residual_sum = 0.

    def capture(self, module, inputs, output):
        if not torch.is_tensor(output) or output.ndim != 3 or output.shape[-1] != 400 or output.shape[1] % 256:
            raise RuntimeError("Expected native P3 [batch,time*256,400]")
        self.pending = output[:, -256:].float().mean(1)

    def forward(self, *args, **kwargs):
        self.pending = None
        result = self.original(*args, **kwargs)
        if self.pending is None:
            raise RuntimeError("Native forward_pred did not call P3")
        if not isinstance(result, tuple) or len(result) != 3:
            raise RuntimeError("Expected native (predicted visual,action,proprio) tuple")
        visual = result[0]
        if visual.ndim != 6 or visual.shape[0] != len(self.pending) or visual.shape[2:] != (1,16,16,384):
            raise RuntimeError(f"Unexpected returned native visual layout: {visual.shape}")
        if self.beta == 0:
            return result  # exact identity: do not clone/repack a beta-zero output
        # Returned layout is checked by the adapter's native canary before launch.
        flat = visual[:, -1].reshape(len(self.pending), 256, 384)
        pooled = flat.float().mean(1)
        delta, saturated, residual = self.calibration.delta(self.pending, pooled, beta=self.beta, sham=self.sham)
        edited = visual.clone()
        edited[:, -1] = (flat + delta[:, None, :].to(flat)).reshape_as(visual[:, -1])
        output = (edited, result[1], result[2])
        self.calls += 1
        self.candidates += len(delta)
        self.saturated += int(saturated.sum())
        norms = delta.norm(dim=-1)
        self.norm_sum += float(norms.sum())
        self.max_norm = max(self.max_norm, float(norms.max()))
        self.residual_sum += float(residual.norm(dim=-1).sum())
        return output

    def __enter__(self):
        self.handle = self.block.register_forward_hook(self.capture)
        self.wm.forward_pred = self.forward
        return self

    def __exit__(self, *_):
        self.wm.forward_pred = self.original
        self.handle.remove()
        self.handle = None
        self.pending = None

    def statistics(self):
        return {"forward_calls": self.calls, "candidate_steps": self.candidates,
                "cap_saturated": self.saturated,
                "cap_saturation_fraction": self.saturated/max(self.candidates, 1),
                "mean_applied_raw_l2": self.norm_sum/max(self.candidates, 1), "max_applied_raw_l2": self.max_norm,
                "mean_pre_edit_standardized_coordinate_residual_l2": self.residual_sum/max(self.candidates, 1)}
