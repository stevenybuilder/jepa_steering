#!/usr/bin/env python3
"""Reload-equivalence check for a feature-trained predictor directory (AdaLN or Panel B).

Proves that the UNCHANGED eval loader (``model_action_sensitivity.load_model`` ->
``hubconf._load_model_with_config`` -> ``init_module`` -> ``init_video_model`` -> ``VideoWM`` /
``EncPredWM``), fed ``<model-dir>/eval_config.yaml`` + the checkpoint, builds the same module
the trainer trained and predicts the same latents:

1. class + state dict: every tensor of the loader's predictor equals the checkpoint's
   (the vendor loads with ``strict=False``, so missing/unexpected keys are checked here);
2. ``probe.npz``: ``EncPredWM.unroll(context, act_suffix)`` (the call ``latent_cache.py`` /
   ``model_action_sensitivity.predict_cell`` make) reproduces the trainer's stored unroll;
3. random features: ``VideoWM.forward_pred`` == ``train_feature_predictor.forward_pred`` and
   ``EncPredWM.unroll`` == ``train_feature_predictor.unroll_eval`` on fresh random inputs.

Usage::

    JEPAWM_HOME=... JEPAWM_OSSCKPT=... PYTHONPATH=$REPO:$CODE python check_panelb_loader.py \
        --repo $REPO --model-dir $MD [--checkpoint jepa-latest.pth.tar] [--out check.json]

Exit status 1 when any comparison exceeds ``--atol`` (default 1e-4 absolute on fp32 latents).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", default="jepa-latest.pth.tar")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="JSON report (default <model-dir>/loader_check.json)")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(args.repo.resolve()))
    import train_feature_predictor as tfp  # noqa: E402
    from model_action_sensitivity import load_model  # noqa: E402

    md = args.model_dir
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    meta = json.loads((md / "jepa-latest.meta.json").read_text(encoding="utf-8"))
    mcfg = {k: v for k, v in meta["model"].items() if k in tfp.ModelCfg.__dataclass_fields__}
    cfg = tfp.ModelCfg(**mcfg)
    report: dict = {"model_dir": str(md), "checkpoint": args.checkpoint, "pred_type": cfg.pred_type, "atol": args.atol}

    # trainer-side predictor from the checkpoint (strict load)
    sd = torch.load(md / args.checkpoint, map_location="cpu", weights_only=True)
    torch.manual_seed(args.seed)
    trainer_pred, _ = tfp.build_vendor_predictor(cfg, device)
    trainer_pred.load_state_dict(sd["predictor"], strict=True)
    trainer_pred.eval()

    # eval-stack model through the unchanged loader
    model, _ = load_model(args.repo, md / "eval_config.yaml", md / args.checkpoint, tfp.MODEL_NAME, str(device))
    loader_pred = model.model.predictor
    report["trainer_class"] = type(trainer_pred).__name__
    report["loader_class"] = type(loader_pred).__name__
    report["loader_pred_type"] = model.model.pred_type
    report["loader_action_conditioning"] = model.model.action_conditioning
    report["loader_use_proprio"] = bool(model.model.use_proprio)
    report["loader_ctxt_window"] = int(model.ctxt_window)
    a = {k: v.detach().cpu() for k, v in loader_pred.state_dict().items()}
    b = {k: v.detach().cpu() for k, v in sd["predictor"].items()}
    report["state_dict"] = {
        "n_tensors_loader": len(a), "n_tensors_checkpoint": len(b),
        "missing_in_loader": sorted(set(b) - set(a)), "unexpected_in_loader": sorted(set(a) - set(b)),
        "all_equal": bool(set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)),
    }
    ok = report["state_dict"]["all_equal"] and type(trainer_pred).__name__ == type(loader_pred).__name__

    # probe reproduction (fp32, no autocast, as the trainer wrote it)
    with np.load(md / "probe.npz") as pr:
        ctx = torch.from_numpy(np.asarray(pr["context"], dtype=np.float32)).to(device)  # [1, 1, 1, G, G, D]
        act = torch.from_numpy(np.asarray(pr["actions"], dtype=np.float32)).to(device)  # [H, 1, A]
        want = torch.from_numpy(np.asarray(pr["prediction"], dtype=np.float32))  # [H, 1, 1, G, G, D]
        ctxt_window = int(pr["ctxt_window"])
        probe_ckpt = str(pr["checkpoint"])
    with torch.inference_mode():
        got = model.unroll(ctx, act_suffix=act)[ctx.shape[1]:].float().cpu()  # loader path
        mine = tfp.unroll_eval(trainer_pred, ctx, act.permute(1, 0, 2), ctxt_window).permute(1, 0, 2, 3, 4, 5).float().cpu()
    report["probe"] = {"checkpoint_in_probe": probe_ckpt, "ctxt_window": ctxt_window, "shape": list(got.shape),
                       "loader_vs_stored_max_abs": float((got - want).abs().max()), "trainer_vs_stored_max_abs": float((mine - want).abs().max()),
                       "loader_vs_trainer_max_abs": float((got - mine).abs().max()), "stored_rms": float(want.pow(2).mean().sqrt())}
    if probe_ckpt == args.checkpoint:
        ok = ok and report["probe"]["loader_vs_stored_max_abs"] <= args.atol
    ok = ok and report["probe"]["loader_vs_trainer_max_abs"] <= args.atol

    # random features: forward_pred and unroll equivalence
    g = torch.Generator().manual_seed(args.seed)
    t = cfg.num_frames_pred
    feats = torch.randn(2, t, 1, cfg.grid_size, cfg.grid_size, cfg.embed_dim, generator=g).to(device)
    acts = torch.randn(2, t, cfg.action_dim, generator=g).to(device)
    with torch.inference_mode():
        p_train = tfp.forward_pred(trainer_pred, feats, acts).float().cpu()
        p_load, _, _ = model.model.forward_pred(feats, acts, None)
        p_load = p_load.float().cpu()
        u_train = tfp.unroll_eval(trainer_pred, feats[:, :1], acts[:, :-1], int(model.ctxt_window)).float().cpu()
        u_load = model.unroll(feats[:, :1], act_suffix=acts[:, :-1].permute(1, 0, 2))[1:].permute(1, 0, 2, 3, 4, 5).float().cpu()
    report["random"] = {"forward_pred_max_abs": float((p_train - p_load).abs().max()), "unroll_max_abs": float((u_train - u_load).abs().max()),
                        "forward_pred_shape": list(p_load.shape), "unroll_shape": list(u_load.shape), "pred_rms": float(p_load.pow(2).mean().sqrt())}
    ok = ok and report["random"]["forward_pred_max_abs"] <= args.atol and report["random"]["unroll_max_abs"] <= args.atol
    ok = ok and list(p_load.shape) == list(feats.shape)
    report["ok"] = bool(ok)
    out = args.out or (md / "loader_check.json")
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print("LOADER_CHECK", "PASS" if ok else "FAIL", json.dumps({k: report[k] for k in ("pred_type", "trainer_class", "loader_class", "state_dict", "probe", "random")}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
