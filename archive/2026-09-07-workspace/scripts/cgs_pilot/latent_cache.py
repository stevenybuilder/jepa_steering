#!/usr/bin/env python3
"""Build the per-cell latent cache used by the CPU-side analyses.

One GPU pass with the real JEPA-WM model (same encode/unroll path as
``model_action_sensitivity.py``), then ``retrieval_baseline.py``,
``counterfactual_validity_gate.py`` and ``audit_contamination.py`` run from the
cache anywhere.

``<cache>.npz`` (all latent grids are float16, token grid is 16 x 16, D = 1024):

- ``cell_id``        str  [n]
- ``context``        [n, 16, 16, D]     last context frame
- ``target``         [n, T, 16, 16, D]  true future frames (T = horizon)
- ``prediction``     [n, T, 16, 16, D]  model rollout with the cell's actions
- ``zero_prediction``[n, T, 16, 16, D]  model rollout with all-zero actions
- ``actions``        [n, T, 7]
- ``token_mask``     bool [n, T + 1, 16, 16]  egg|gripper patch mask per frame
                     (index 0 = context, 1..T = future) when
                     ``<stimulus_dir>/masks/<cell_id>.npz`` exists, else all True
- ``has_mask``       bool [n]
- ``droid_*``        optional DROID transition latents when ``--droid-reference``
                     is given: ``droid_pooled_t`` [K, D], ``droid_delta`` [K, 16, 16, D]
                     float16, ``droid_actions`` [K, 3, 7]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from protocol import sha256_file  # noqa: E402

GRID = 16


def pixel_mask_to_tokens(mask: np.ndarray, grid: int = GRID) -> np.ndarray:
    """bool [..., H, W] -> bool [..., grid, grid]; a patch is on if any pixel is on."""

    m = np.asarray(mask, dtype=bool)
    h, w = m.shape[-2:]
    ph, pw = h // grid, w // grid
    m = m[..., : ph * grid, : pw * grid]
    return m.reshape(*m.shape[:-2], grid, ph, grid, pw).any(axis=(-1, -3))


def load_token_mask(stimulus_dir: Path, cell_id: str, n_frames: int) -> tuple[np.ndarray, bool]:
    path = stimulus_dir / "masks" / f"{cell_id}.npz"
    if not path.exists():
        return np.ones((n_frames, GRID, GRID), dtype=bool), False
    with np.load(path) as z:
        egg = np.asarray(z["egg_mask"], dtype=bool)
        robot = np.asarray(z["robot_mask"], dtype=bool)
    tok = pixel_mask_to_tokens(egg | robot)
    if tok.shape[0] != n_frames:
        raise ValueError(f"mask {path} has {tok.shape[0]} frames, expected {n_frames}")
    return tok, True


def build(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from model_action_sensitivity import encode_frames, load_model

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model, _ = load_model(args.repo, args.config, args.checkpoint, args.model_name, str(device))
    rows = [json.loads(ln) for ln in (args.artifacts / "manifest.jsonl").read_text().splitlines() if ln.strip()]

    out: dict[str, list] = {k: [] for k in ("cell_id", "context", "target", "prediction", "zero_prediction", "actions", "token_mask", "has_mask")}
    for r in rows:
        with np.load(args.artifacts / r["artifact"]) as cell:
            context = np.asarray(cell["context_frames"])
            future = np.asarray(cell["true_future_frames"])
            actions = np.asarray(cell["model_actions"], dtype=np.float32)
        z_ctx = encode_frames(model, context[-1:], device)  # [1, 1, 16, 16, D]
        z_fut = encode_frames(model, future, device)  # [1, T, 16, 16, D]
        act = torch.from_numpy(actions).to(device=device, dtype=torch.float32).unsqueeze(1)
        with torch.inference_mode():
            roll = model.unroll(z_ctx, act_suffix=act)
            zero = model.unroll(z_ctx, act_suffix=torch.zeros_like(act))
        n_ctx = int(z_ctx.shape[1])
        D = int(z_ctx.shape[-1])

        def grid(t):  # squeeze any singleton view/time axes -> [16, 16, D]
            return t.float().reshape(GRID, GRID, D).cpu().numpy().astype(np.float16)

        pred = np.stack([grid(roll[i][0, -1]) for i in range(n_ctx, len(roll))])  # [T, 16, 16, D]
        zpred = np.stack([grid(zero[i][0, -1]) for i in range(n_ctx, len(zero))])
        mask, has = load_token_mask(args.artifacts, r["cell_id"], len(future) + 1)
        out["cell_id"].append(r["cell_id"])
        out["context"].append(grid(z_ctx[0, -1]))
        out["target"].append(np.stack([grid(z_fut[0, t]) for t in range(int(z_fut.shape[1]))]))
        out["prediction"].append(pred)
        out["zero_prediction"].append(zpred)
        out["actions"].append(actions)
        out["token_mask"].append(mask)
        out["has_mask"].append(has)

    arrays: dict[str, np.ndarray] = {k: np.stack(v) if k != "cell_id" else np.asarray(v) for k, v in out.items()}

    if args.droid_reference is not None and args.droid_reference.exists():
        with np.load(args.droid_reference) as z:
            ft, ft1, ta = z["trans_frame_t"], z["trans_frame_t1"], z["trans_actions"]
        pooled, delta = [], []
        for i in range(len(ft)):
            a = encode_frames(model, ft[i : i + 1], device)[0, -1].float().reshape(GRID, GRID, -1)
            b = encode_frames(model, ft1[i : i + 1], device)[0, -1].float().reshape(GRID, GRID, -1)
            pooled.append(a.reshape(-1, a.shape[-1]).mean(0).cpu().numpy())
            delta.append((b - a).cpu().numpy().astype(np.float16))
        arrays["droid_pooled_t"] = np.stack(pooled) if pooled else np.zeros((0, 1), np.float32)
        arrays["droid_delta"] = np.stack(delta) if delta else np.zeros((0, GRID, GRID, 1), np.float16)
        arrays["droid_actions"] = np.asarray(ta, dtype=np.float32)

    meta = {
        "model_name": args.model_name,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "config_sha256": sha256_file(args.config),
        "manifest_sha256": sha256_file(args.artifacts / "manifest.jsonl"),
        "droid_reference": str(args.droid_reference) if args.droid_reference else None,
        "torch": torch.__version__,
        "n_cells": len(rows),
        "n_with_masks": int(arrays["has_mask"].sum()),
    }
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.cache, meta=json.dumps(meta), **arrays)
    print(json.dumps(meta, indent=2))
    return meta


def load_cache(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d["meta"]))
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--model-name", default="jepa_wm_droid")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--droid-reference", type=Path, default=None, help="droid_100_reference.npz")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
