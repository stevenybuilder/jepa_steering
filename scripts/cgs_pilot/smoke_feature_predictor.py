#!/usr/bin/env python3
"""Smoke-test a ``train_feature_predictor.py`` output directory through the pilot's own
UNCHANGED loader.

Loads ``<out>/eval_config.yaml`` + ``<out>/<checkpoint>`` with
``model_action_sensitivity.load_model(repo, config, checkpoint, "jepa_wm_driving", device)``
(vendor ``hubconf._load_model_with_config(pretrained=False)``; DINOv3 ViT-L encoder loaded
from ``JEPAWM_OSSCKPT``), then checks

1. ``model.encode`` on one random uint8 frame -> ``[1, 1, 1, G, G, D]``,
2. ``model.unroll`` on the trainer's ``probe.npz`` context/actions reproduces the trainer's
   fp32 predictions (max |diff| and RMS),
3. ``predictor_hooks.PredictorRecorder`` capture at one site for every imagined step,
4. VRAM and timings,

and writes a JSON report.  Requires the ``driving`` DATA_STATS entry
(``vendor_patches/0001-driving-data-stats.patch``); sets ``JEPAWM_DRIVING_ACTION_DIM`` from the
eval yaml before importing the vendor.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_action_sensitivity import encode_frames, load_model  # noqa: E402
from predictor_hooks import PredictorRecorder, Site, n_layers_of, predictor_of, spatial_tokens_of, unroll_from_latents  # noqa: E402
from protocol import sha256_file  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True, help="train_feature_predictor.py --out directory")
    ap.add_argument("--checkpoint", default="jepa-latest.pth.tar", help="file name inside --out-dir (probe.npz was made from jepa-latest)")
    ap.add_argument("--model-name", default="jepa_wm_driving")
    ap.add_argument("--site", default="L03.mlp_out")
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    config = args.out_dir / "eval_config.yaml"
    checkpoint = args.out_dir / args.checkpoint
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    action_dim = int(cfg.get("cgs_pilot", {}).get("action_dim", 2))
    os.environ.setdefault("JEPAWM_DRIVING_ACTION_DIM", str(action_dim))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.empty(0, device=device)  # create the context before querying memory stats
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(0)
    t0 = time.time()
    model, _ = load_model(args.repo, config, checkpoint, args.model_name, str(device))
    t_load = time.time() - t0
    inner = model.model
    predictor = predictor_of(model)
    vram_after_load = torch.cuda.memory_allocated(device) / 1e9 if device.type == "cuda" else None

    report: dict = {
        "model_name": args.model_name,
        "config": str(config),
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint),
        "load_s": t_load,
        "wrapper": type(model).__name__,
        "encoder_class": type(inner.encoder).__name__,
        "encoder_name": getattr(inner.encoder, "name", None),
        "encoder_params": sum(p.numel() for p in inner.encoder.parameters()),
        "encoder_patch_size": int(inner.encoder.patch_size),
        "predictor_class": type(predictor).__name__,
        "predictor_params": sum(p.numel() for p in predictor.parameters()),
        "predictor_depth": n_layers_of(predictor),
        "predictor_width": int(predictor.predictor_embed_dim),
        "action_encoder_in_features": int(predictor.action_encoder.in_features),
        "ctxt_window": int(model.ctxt_window),
        "grid_size": int(model.grid_size),
        "vram_after_load_gb": vram_after_load,
    }

    # 1. encoder path on one random frame
    grid = int(model.grid_size)
    img = grid * int(inner.encoder.patch_size)
    frame = np.random.default_rng(0).integers(0, 256, size=(1, img, img, 3), dtype=np.uint8)
    t0 = time.time()
    z = encode_frames(model, frame, device)
    report["encode"] = {"input": list(frame.shape), "output": list(z.shape), "dtype": str(z.dtype), "s": time.time() - t0, "finite": bool(torch.isfinite(z).all())}

    # 2. unroll on the trainer probe and compare
    probe = np.load(args.out_dir / "probe.npz")
    z_ctx = torch.from_numpy(probe["context"]).to(device=device, dtype=torch.float32)
    act = torch.from_numpy(probe["actions"]).to(device=device, dtype=torch.float32)
    t0 = time.time()
    with torch.inference_mode():
        roll = model.unroll(z_ctx, act_suffix=act)
    t_unroll = time.time() - t0
    pred = roll[int(z_ctx.shape[1]) :].float().cpu().numpy()
    ref = probe["prediction"].astype(np.float32)
    target = probe["target"].astype(np.float32)  # [1, H, 1, G, G, D]
    diff = pred - ref
    per_step_l2 = [float(np.mean((pred[h, 0] - target[0, h]) ** 2)) for h in range(pred.shape[0])]
    copy_l2 = [float(np.mean((probe["context"][0, 0].astype(np.float32) - target[0, h]) ** 2)) for h in range(pred.shape[0])]
    report["unroll"] = {
        "z_ctx": list(z_ctx.shape), "act_suffix": list(act.shape), "rollout": list(roll.shape), "prediction": list(pred.shape), "s": t_unroll,
        "max_abs_diff_vs_trainer_probe": float(np.abs(diff).max()), "rms_diff_vs_trainer_probe": float(np.sqrt(np.mean(diff**2))),
        "rms_prediction": float(np.sqrt(np.mean(ref**2))), "finite": bool(np.isfinite(pred).all()),
        "unroll_l2_vs_target_per_step": per_step_l2, "copy_baseline_l2_per_step": copy_l2,
        "probe_checkpoint": str(probe["checkpoint"]) if "checkpoint" in probe.files else None,
    }
    report["unroll"]["matches_trainer"] = bool(report["unroll"]["max_abs_diff_vs_trainer_probe"] < 1e-2 * max(1.0, report["unroll"]["rms_prediction"]))

    # 3. hook capture at one site
    site = Site.parse(args.site)
    rec = PredictorRecorder(predictor, [site], spatial_tokens_of(model))
    hooked = unroll_from_latents(model, z_ctx, act, recorder=rec)
    report["hooks"] = {
        "site": site.site_id, "n_steps": rec.n_steps,
        "shapes_per_step": {str(s): list(t.shape) for s, t in rec.acts[site.site_id].items()},
        "hooked_unroll_equals_plain": bool(torch.allclose(hooked.float().cpu(), roll[int(z_ctx.shape[1]) :].float().cpu(), atol=1e-4, rtol=1e-4)),
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        report["peak_vram_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
    report["torch"] = torch.__version__
    report["device"] = str(device)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
