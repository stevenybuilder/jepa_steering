"""Summarize existing fitted tensors for a figure; no fitting or model execution."""
from pathlib import Path
import hashlib
import json
import torch

ROOT = Path(__file__).resolve().parents[2]
out = {"new_model_evaluations": 0, "tasks": {}}
for task in ("reach", "reach-wall"):
    path = ROOT / f"artifacts/offline_study/fixed-response-20260908-v1/fits/{task}/operator_bank.pt"
    receipt = json.loads(path.with_name("DONE.json").read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == receipt[path.name]
    bank = torch.load(path, map_location="cpu", weights_only=True)
    basis = bank["operators"]["fixed_rank4"]["basis"].double()
    flat = basis.flatten(1)
    assert tuple(basis.shape) == (4, 256, 400)
    assert torch.allclose(flat @ flat.T, torch.eye(4).double(), atol=2e-5, rtol=2e-5)
    # Each unit basis vector assigns total squared loading mass one across patches.
    energy = basis.square().sum(-1)
    mean_energy = energy.mean(0)
    out["tasks"][task] = {
        "source": str(path.relative_to(ROOT)), "source_sha256": digest,
        "binding": bank["binding"],
        "patch_squared_loadings": energy.tolist(),
        "subspace_mean_patch_squared_loadings": mean_energy.tolist(),
        "coefficient_map": bank["operators"]["fixed_rank4"]["map"].tolist(),
        "basis_orthogonality_max_abs_error": float((flat @ flat.T - torch.eye(4)).abs().max()),
        "note": "Descriptive fitted-basis visualization; no semantic or causal attribution."
    }
path = ROOT / "paper/data/basis_spatial_summary.json"
path.write_text(json.dumps(out, indent=2) + "\n")
print(f"Exported two existing, checksum-verified banks to {path}")
