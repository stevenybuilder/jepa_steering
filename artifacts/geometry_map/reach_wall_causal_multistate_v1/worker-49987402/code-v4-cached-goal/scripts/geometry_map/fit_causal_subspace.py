#!/usr/bin/env python3
"""Freeze the existing block-3 realized-XZ-direction candidate on discovery only.

Five iterative two-output ridge probes define a candidate of at most ten
dimensions, not an estimate of intrinsic dimension or evidence of causal use.
The common scaling is saved and reused exactly when patching native activations.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import Ridge
from screen_pooled import load_discovery, TARGETS
from protocol import file_sha256, write_json_atomic


def fit_basis(x, y, mask, alpha=100.0, iterations=5):
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-6] = 1.0
    original = (x - mean) / scale
    residual = original[mask].copy()
    basis = np.empty((x.shape[1], 0))
    for _ in range(iterations):
        model = Ridge(alpha=alpha, solver="cholesky").fit(residual, y[mask])
        weights = np.atleast_2d(model.coef_).T
        weights -= basis @ (basis.T @ weights)
        q, singular, _ = np.linalg.svd(weights, full_matrices=False)
        q = q[:, singular > 1e-9]
        if not q.shape[1]:
            break
        basis = np.linalg.qr(np.concatenate((basis, q), axis=1))[0]
        residual = original[mask] - (original[mask] @ basis) @ basis.T
    return mean, scale, basis


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture-dirs", nargs="+", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    data = load_discovery(args.capture_dirs)
    names = [item[0] for item in TARGETS]
    motion = data["targets"][:, [names.index("realized_dx"), names.index("realized_dz")]]
    magnitude = np.linalg.norm(motion, axis=1)
    threshold = np.quantile(magnitude, 0.25)
    y = motion / np.maximum(magnitude[:, None], 1e-8)
    mean, scale, basis = fit_basis(data["features"]["predictor"][:, 3].astype(float), y, magnitude > threshold)
    if not basis.shape[1] or not np.isfinite(basis).all():
        raise RuntimeError("Discovery fit did not produce a finite nonempty candidate")
    output = args.output_dir / "candidate.pt"
    torch.save({"mean": torch.tensor(mean, dtype=torch.float32),
                "scale": torch.tensor(scale, dtype=torch.float32),
                "basis": torch.tensor(basis, dtype=torch.float32),
                "block": 3, "target": "realized_xz_direction", "imagined_step": 0,
                "status": "discovery_candidate_not_causally_validated"}, output)
    write_json_atomic(args.output_dir / "DONE.json", {
        "complete": True, "scope": "discovery-fit only; no behavior or confirmation used",
        "rank": basis.shape[1], "episodes": len(set(zip(data["seeds"], data["episodes"]))),
        "sample_count": len(y), "motion_mask_threshold": float(threshold),
        "metric": "one discovery mean/std shared by all target directions; save and reuse scaling",
        "ridge_alpha": 100.0, "probe_iterations": 5, "sources": data["sources"],
        "collection_seeds": sorted(int(seed) for seed in np.unique(data["seeds"])),
        "script_sha256": file_sha256(Path(__file__)),
        "sha256": file_sha256(output), "path": output.name,
    })


if __name__ == "__main__":
    main()
