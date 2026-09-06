#!/usr/bin/env python3
"""Perturbed-candidate rollouts for the planner-currency ranking test (GPU, small).

For every complete scene, sample ``--n-candidates`` CEM-style candidates around
each of a0 and a1 (Gaussian jitter with ``--var-scale`` * released max_norms
[0.1 for dims 0-5, 0.75 for dim 6], then clipped to those max norms per step,
as the released CEM does with ``max_norms``/``max_norm_dims``), roll every
candidate out from the H0, H1 (and H0' when h2 cells exist) contexts, and save
the last-step predicted latents plus the per-level goal (true future under a0)
so ``counterfactual_validity_gate.py`` can compute the Spearman rank correlation
of the planner costs between hazard levels (a hazard-insensitive planner gives
rho ~ 1). The SAME candidate list is used under every hazard level of a scene.

Run only when the GPU is free (after the queued Jacobian/localization jobs).
Output: ``<artifacts>/planner_perturbed_costs.npz``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from protocol import CELL_ORDER  # noqa: E402

MAX_NORMS = np.array([0.1] * 6 + [0.75], dtype=np.float32)
LEVEL = {0: "h0", 1: "h1", 2: "h0prime"}


def sample_candidates(a0: np.ndarray, a1: np.ndarray, n: int, var_scale: float, rng: np.random.RandomState) -> np.ndarray:
    """[2n, T, 7]: n jittered copies of a0 then n of a1, clipped to the released max norms."""

    out = []
    for base in (a0, a1):
        noise = rng.randn(n, *base.shape).astype(np.float32) * (var_scale * MAX_NORMS)
        cand = base[None] + noise
        xyz = cand[..., :6]
        nrm = np.linalg.norm(xyz, axis=-1, keepdims=True)
        cand[..., :6] = np.where(nrm > 0.1, xyz * (0.1 / np.maximum(nrm, 1e-9)), xyz)
        cand[..., 6] = np.clip(cand[..., 6], -0.75, 0.75)
        out.append(cand)
    return np.concatenate(out, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--model-name", default="jepa_wm_droid")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-candidates", type=int, default=32, help="per base action (a0 and a1)")
    ap.add_argument("--var-scale", type=float, default=0.1, help="released CEM var_scale")
    ap.add_argument("--seeds-file", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    import torch

    from model_action_sensitivity import encode_frames, load_model

    rows = [json.loads(l) for l in (args.artifacts / "manifest.jsonl").read_text().splitlines() if l.strip()]
    if args.seeds_file is not None:
        keep = {int(s) for s in args.seeds_file.read_text().split()}
        rows = [r for r in rows if int(r["seed"]) in keep]
    grouped: dict[str, dict[tuple[int, int], dict]] = {}
    for r in rows:
        grouped.setdefault(r["pair_id"], {})[(int(r["hazard"]), int(r["candidate_action"]))] = r
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    model, _ = load_model(args.repo, args.config, args.checkpoint, args.model_name, str(device))

    arrays: dict[str, np.ndarray] = {}
    meta = {"n_candidates_per_base": args.n_candidates, "var_scale": args.var_scale, "max_norms": MAX_NORMS.tolist(), "scenes": {}, "candidate_order": "first n = jitter(a0), next n = jitter(a1)"}
    for pair_id, cells in grouped.items():
        if not set(CELL_ORDER) <= set(cells):
            continue
        with np.load(args.artifacts / cells[(1, 0)]["artifact"]) as c:
            a0 = np.asarray(c["model_actions"], np.float32)
        with np.load(args.artifacts / cells[(1, 1)]["artifact"]) as c:
            a1 = np.asarray(c["model_actions"], np.float32)
        cands = sample_candidates(a0, a1, args.n_candidates, args.var_scale, rng)
        levels = [h for h in (0, 1, 2) if (h, 0) in cells and (h, 1) in cells]
        meta["scenes"][pair_id] = [LEVEL[h] for h in levels]
        for h in levels:
            with np.load(args.artifacts / cells[(h, 0)]["artifact"]) as c:
                ctx = np.asarray(c["context_frames"])[-1:]
                goal = np.asarray(c["true_future_frames"])[-1:]
            z_ctx = encode_frames(model, ctx, device)
            z_goal = encode_frames(model, goal, device)
            D = int(z_ctx.shape[-1])
            preds = []
            with torch.inference_mode():
                for cand in cands:
                    act = torch.from_numpy(cand).to(device=device, dtype=torch.float32).unsqueeze(1)
                    roll = model.unroll(z_ctx, act_suffix=act)
                    preds.append(roll[-1][0, -1].float().reshape(16, 16, D).cpu().numpy().astype(np.float16))
            arrays[f"{pair_id}__{LEVEL[h]}__pred"] = np.stack(preds)
            arrays[f"{pair_id}__{LEVEL[h]}__goal"] = z_goal[0, -1].float().reshape(16, 16, D).cpu().numpy().astype(np.float16)
        arrays[f"{pair_id}__candidates"] = cands
        print(pair_id, "levels", meta["scenes"][pair_id], flush=True)
    out = args.output or (args.artifacts / "planner_perturbed_costs.npz")
    np.savez_compressed(out, meta=json.dumps(meta), **arrays)
    print("WROTE", out, "scenes", len(meta["scenes"]))


if __name__ == "__main__":
    main()
