"""Low-rank counterfactual patches in an explicit, saved coordinate metric.

This is a causal diagnostic, not a performance-steering recipe. Only the pooled
coordinate changes; token-to-token residuals and the orthogonal complement in
the fitted standardized metric are preserved. The sham rotates the raw edit,
preserving its exact Euclidean magnitude for each candidate.
"""
from __future__ import annotations

import torch


def projected_delta(current, donor, basis, scale, beta):
    if beta == 0:
        return torch.zeros_like(current)
    difference = (donor - current) / scale
    return beta * ((difference @ basis) @ basis.T) * scale


class FirstStepSubspacePatch:
    def __init__(self, block, unroll, basis, scale, donor, beta=0.5, sham=False, seed=905):
        self.block, self.original_unroll = block, unroll
        self.basis, self.scale, self.donor = basis, scale, donor
        if basis.ndim != 2 or not 0 < basis.shape[1] <= basis.shape[0] or scale.shape != (basis.shape[0],):
            raise ValueError("Expected a nonempty [dim,rank] basis and [dim] scale")
        if not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError("Coordinate scales must be finite and positive")
        if not torch.allclose(basis.T @ basis, torch.eye(basis.shape[1], device=basis.device, dtype=basis.dtype), atol=1e-5):
            raise ValueError("Basis must be orthonormal in the standardized metric")
        self.beta, self.sham = beta, sham
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.permutation = torch.randperm(scale.numel(), generator=generator).to(scale.device)
        self.sign = (2 * torch.randint(2, (scale.numel(),), generator=generator) - 1).to(scale)
        self.step = 0
        self.calls = 0
        self.first_edit = None
        self.handle = None

    def hook(self, module, args, output):
        step = self.step
        self.step += 1
        if step != 0:
            return output
        if not isinstance(output, torch.Tensor) or output.ndim != 3 or output.shape[1] != 256:
            raise RuntimeError(f"Expected first-step [batch,256,dim], got {getattr(output, 'shape', None)}")
        self.calls += 1
        if self.beta == 0:
            return output  # exact identity, including dtype
        pooled = output.float().mean(dim=1)
        delta = projected_delta(pooled, self.donor, self.basis, self.scale, self.beta)
        if self.sham:
            delta = delta[:, self.permutation] * self.sign
        if self.first_edit is None:
            self.first_edit = delta.detach().cpu()
        edited = output + delta.to(output.dtype).unsqueeze(1)
        if not torch.isfinite(edited).all():
            raise RuntimeError("Non-finite subspace edit")
        return edited

    def unroll(self, *args, **kwargs):
        self.step = 0
        return self.original_unroll(*args, **kwargs)

    def __enter__(self):
        self.handle = self.block.register_forward_hook(self.hook)
        return self

    def __exit__(self, *exc):
        self.handle.remove()
