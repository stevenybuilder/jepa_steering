#!/usr/bin/env python3
"""Smoke-test ``vjepa2_ac_droid`` through the pilot's own loader.

Loads the model exactly as ``latent_cache.py`` does
(``model_action_sensitivity.load_model`` -> vendor ``hubconf._load_model_with_config``
with ``pretrained=False`` and a local checkpoint), then on one stimulus cell
checks: ``model.encode`` on a single frame (dup_image path), ``model.unroll``
per-step latents, the ``latent_cache.grid()`` reshape to 16 x 16 x D, finiteness,
and VRAM. Writes a JSON report.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from latent_cache import GRID  # noqa: E402
from model_action_sensitivity import encode_frames, load_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--model-name", default="vjepa2_ac_droid")
    ap.add_argument("--cell", type=Path, required=True, help="one stimulus cell .npz")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
    t0 = time.time()
    model, _ = load_model(args.repo, args.config, args.checkpoint, args.model_name, str(device))
    t_load = time.time() - t0
    inner = model.model
    enc = inner.encoder
    info = {
        "model_name": args.model_name,
        "load_s": t_load,
        "wrapper": type(model).__name__,
        "enc_type": inner.enc_type,
        "encoder_class": type(enc).__name__,
        "encoder_patch_size": int(enc.patch_size),
        "encoder_embed_dim": int(enc.embed_dim),
        "encoder_depth": len(enc.blocks),
        "encoder_params": sum(p.numel() for p in enc.parameters()),
        "encoder_dtype": str(next(enc.parameters()).dtype),
        "predictor_class": type(inner.predictor).__name__,
        "predictor_params": sum(p.numel() for p in inner.predictor.parameters()),
        "dup_image": bool(inner.dup_image),
        "batchify_video": bool(inner.batchify_video),
        "normalize_reps": bool(inner.normalize_reps),
        "grid_size": int(model.grid_size),
        "ctxt_window": int(model.ctxt_window),
        "use_proprio": bool(inner.use_proprio),
        "use_action": bool(inner.use_action),
        "action_dim": model.action_dim,
    }
    with np.load(args.cell) as cell:
        context = np.asarray(cell["context_frames"])
        future = np.asarray(cell["true_future_frames"])
        actions = np.asarray(cell["model_actions"], dtype=np.float32)
    info["frames"] = {"context": list(context.shape), "future": list(future.shape), "actions": list(actions.shape), "dtype": str(context.dtype)}

    z_ctx = encode_frames(model, context[-1:], device)
    z_fut = encode_frames(model, future, device)
    info["encode"] = {"z_ctx_shape": list(z_ctx.shape), "z_fut_shape": list(z_fut.shape), "z_ctx_dtype": str(z_ctx.dtype)}
    act = torch.from_numpy(actions).to(device=device, dtype=torch.float32).unsqueeze(1)
    with torch.inference_mode():
        roll = model.unroll(z_ctx, act_suffix=act)
        zero = model.unroll(z_ctx, act_suffix=torch.zeros_like(act))
    n_ctx = int(z_ctx.shape[1])
    D = int(z_ctx.shape[-1])
    info["unroll"] = {"roll_shape": list(roll.shape), "n_ctx": n_ctx, "T": int(act.shape[0]), "D": D}

    def grid(t):
        return t.float().reshape(GRID, GRID, D).cpu().numpy()

    steps = [grid(roll[i][0, -1]) for i in range(n_ctx, len(roll))]
    zsteps = [grid(zero[i][0, -1]) for i in range(n_ctx, len(zero))]
    tgt = [grid(z_fut[0, t]) for t in range(int(z_fut.shape[1]))]
    tokens_per_frame = int(np.prod(z_ctx.shape[2:-1]))
    info["grid_check"] = {
        "tokens_per_frame": tokens_per_frame,
        "expected": GRID * GRID,
        "ok": tokens_per_frame == GRID * GRID,
        "per_step_shapes": [list(s.shape) for s in steps],
    }
    info["finite"] = {
        "context": bool(np.isfinite(grid(z_ctx[0, -1])).all()),
        "target": bool(all(np.isfinite(s).all() for s in tgt)),
        "prediction": bool(all(np.isfinite(s).all() for s in steps)),
        "zero_prediction": bool(all(np.isfinite(s).all() for s in zsteps)),
    }
    rms = lambda a: float(np.sqrt(np.mean(np.square(a))))  # noqa: E731
    info["magnitudes"] = {
        "context_rms": rms(grid(z_ctx[0, -1])),
        "pred_step_rms": [rms(s) for s in steps],
        "pred_minus_zero_rms": [rms(s - z) for s, z in zip(steps, zsteps)],
        "pred_minus_target_rms": [rms(s - t) for s, t in zip(steps, tgt)],
        "target_minus_context_rms": [rms(t - grid(z_ctx[0, -1])) for t in tgt],
        "steps_distinct_rms": [rms(steps[i + 1] - steps[i]) for i in range(len(steps) - 1)],
    }
    if device.type == "cuda":
        info["vram"] = {
            "allocated_gb": torch.cuda.memory_allocated() / 2**30,
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
            "reserved_gb": torch.cuda.memory_reserved() / 2**30,
        }
    info["elapsed_s"] = time.time() - t0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
