#!/usr/bin/env python3
"""Convert the HF ``facebook/vjepa2-vitg-fpc64-256`` encoder into the native
V-JEPA 2 ``vit_giant_xformers`` state dict that JEPA-WM's ``init_video_model``
loads for ``enc_version: v2_open`` (``vjepa2_ac_droid`` / ``vjepa2_ac_oss``).

Why: the vj2ac configs expect ``${JEPAWM_OSSCKPT}/vjepa2_opensource/vjepa2_vit_giant.pth``
with key ``encoder`` (see ``pretrain_enc_ckpt_key``). Meta's official file
(``https://dl.fbaipublicfiles.com/vjepa2/vitg.pt``) is 16.5 GB (encoder +
predictor + optimizer state) and does not fit on the box; the HF transformers
checkpoint (``model.safetensors``, 4.14 GB) carries the same encoder weights.

This script inverts transformers' ``convert_vjepa2_to_hf.py::convert_encoder_keys``
(``blocks.i.attn.qkv`` <- ``encoder.layer.i.attention.{query,key,value}``,
``patch_embed`` <- ``encoder.embeddings.patch_embeddings``, ``norm`` <-
``encoder.layernorm``; RoPE, so there is no ``pos_embed``), drops the HF predictor,
and refuses to write unless the native encoder reproduces the HF encoder:

1. random 16-frame 256x256 clip (fpc64 model; any multiple of tubelet_size=2
   works because positions are RoPE, not learned), fp32 vs fp32;
2. the ``dup_image`` case JEPA-WM actually uses (one frame duplicated to 2);
3. the same with fp16-rounded weights (what gets written), to quantify the
   rounding contribution separately from the mapping.

The output is saved in fp16 (frozen encoder; ``load_state_dict`` upcasts into the
fp32 module at load time) as ``{"encoder": state_dict}``; the vendor loader strips
``module.backbone.`` (absent here) and loads with ``strict=False``, so the
round-trip check asserts no missing/unexpected keys through that exact path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import torch

HF_REPO = "facebook/vjepa2-vitg-fpc64-256"
# Vendor kwargs for vit_giant_xformers under the vj2ac config (utils.init_video_model, v2_open branch).
NATIVE_KWARGS = dict(
    num_frames=512,
    img_size=256,
    tubelet_size=2,
    uniform_power=True,
    use_sdpa=True,
    use_activation_checkpointing=False,
    use_rope=True,
    use_silu=False,
    wide_silu=True,
    local_window=(8, -1, -1),
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def convert(hf_state: dict[str, torch.Tensor], template: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """HF VJEPA2Model encoder keys -> native VisionTransformer (v2) keys. Predictor keys are ignored."""

    out: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()

    def take(key: str) -> torch.Tensor:
        consumed.add(key)
        return hf_state[key]

    out["patch_embed.proj.weight"] = take("encoder.embeddings.patch_embeddings.proj.weight")
    out["patch_embed.proj.bias"] = take("encoder.embeddings.patch_embeddings.proj.bias")
    out["norm.weight"] = take("encoder.layernorm.weight")
    out["norm.bias"] = take("encoder.layernorm.bias")

    layer_ids = sorted({int(m.group(1)) for k in hf_state for m in [re.match(r"encoder\.layer\.(\d+)\.", k)] if m})
    for i in layer_ids:
        p = f"encoder.layer.{i}."
        b = f"blocks.{i}."
        for suffix in ("weight", "bias"):
            out[b + f"attn.qkv.{suffix}"] = torch.cat(
                [take(p + f"attention.{n}.{suffix}") for n in ("query", "key", "value")], dim=0
            )
        out[b + "attn.proj.weight"] = take(p + "attention.proj.weight")
        out[b + "attn.proj.bias"] = take(p + "attention.proj.bias")
        for n in ("norm1", "norm2", "mlp.fc1", "mlp.fc2"):
            out[b + f"{n}.weight"] = take(p + f"{n}.weight")
            out[b + f"{n}.bias"] = take(p + f"{n}.bias")

    unused_encoder = sorted(k for k in set(hf_state) - consumed if k.startswith("encoder."))
    if unused_encoder:
        raise RuntimeError(f"unconsumed HF encoder keys: {unused_encoder[:10]} (+{max(0, len(unused_encoder) - 10)})")
    missing = sorted(set(template) - set(out))
    extra = sorted(set(out) - set(template))
    if missing or extra:
        raise RuntimeError(f"key mismatch vs native template: missing={missing[:10]} extra={extra[:10]}")
    for k, v in out.items():
        if tuple(v.shape) != tuple(template[k].shape):
            raise RuntimeError(f"shape mismatch {k}: {tuple(v.shape)} vs {tuple(template[k].shape)}")
    dropped = sorted(k for k in hf_state if not k.startswith("encoder."))
    print(f"converted {len(out)} encoder tensors; dropped {len(dropped)} non-encoder HF tensors (predictor)")
    return {k: v.contiguous().to(torch.float32) for k, v in out.items()}


def compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a = a.float()
    b = b.float()
    diff = a - b
    ln = torch.nn.functional.layer_norm  # JEPA-WM applies normalize_reps (LayerNorm over D) to encoder tokens
    a_ln, b_ln = ln(a, (a.shape[-1],)), ln(b, (b.shape[-1],))
    return {
        "max_abs": float(diff.abs().max().item()),
        "max_abs_over_ref_max": float(diff.abs().max().item() / b.abs().max().item()),
        "rel_fro": float(diff.norm().item() / b.norm().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "ref_abs_max": float(b.abs().max().item()),
        "min_token_cosine": float(
            torch.nn.functional.cosine_similarity(a.flatten(0, -2), b.flatten(0, -2), dim=-1).min().item()
        ),
        "after_layernorm_max_abs": float((a_ln - b_ln).abs().max().item()),
        "after_layernorm_rel_fro": float((a_ln - b_ln).norm().item() / b_ln.norm().item()),
    }


def gpu_mem() -> dict[str, float] | None:
    if not torch.cuda.is_available():
        return None
    return {
        "allocated_gb": torch.cuda.memory_allocated() / 2**30,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True, help="jepa-wms checkout (vendor/jepa-wms)")
    ap.add_argument("--hf-dir", type=Path, required=True, help="dir with config.json + model.safetensors")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--provenance-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--frames", type=int, default=16)
    # V-JEPA 2 ViT-g tokens reach |x| ~ 50-60 before JEPA-WM's normalize_reps LayerNorm, so a raw
    # max-abs bound is the wrong unit; bounds are relative (Frobenius) and scale-normalised max-abs.
    ap.add_argument("--rel-fp32", type=float, default=1e-4, help="relative Frobenius bound, fp32 mapping check")
    ap.add_argument("--maxabs-norm-fp32", type=float, default=1e-3, help="max_abs / max|ref| bound, fp32 mapping check")
    ap.add_argument("--rel-fp16", type=float, default=2e-2, help="relative Frobenius bound, fp16-rounded weights")
    ap.add_argument("--hf-attn", default="eager", help="HF attention impl; native takes its eager path when attn_mask is None")
    ap.add_argument("--save-dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("--delete-source", action="store_true", help="delete model.safetensors after a verified write")
    ap.add_argument("--expected-source-sha256", default=None, help="LFS sha256 from the HF API, if known")
    args = ap.parse_args()

    sys.path.insert(0, str(args.repo))
    import src.models.vision_transformer_v2 as vit_v2_open  # type: ignore
    from safetensors.torch import load_file
    from transformers import VJEPA2Model

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    t0 = time.time()

    src_path = args.hf_dir / "model.safetensors"
    cfg_path = args.hf_dir / "config.json"
    hf_cfg = json.loads(cfg_path.read_text())
    assert hf_cfg["hidden_size"] == 1408 and hf_cfg["num_hidden_layers"] == 40 and hf_cfg["patch_size"] == 16
    assert hf_cfg["tubelet_size"] == 2 and not hf_cfg.get("use_SiLU", False), hf_cfg

    print("hashing source ...", flush=True)
    source_sha = sha256_file(src_path)
    if args.expected_source_sha256 and source_sha != args.expected_source_sha256:
        raise SystemExit(f"source sha mismatch: {source_sha} != {args.expected_source_sha256}")

    print("building native vit_giant_xformers ...", flush=True)
    native = vit_v2_open.vit_giant_xformers(**NATIVE_KWARGS).eval()
    template = native.state_dict()
    hf_state = load_file(str(src_path))
    converted = convert(hf_state, template)
    del hf_state
    native.load_state_dict(converted, strict=True)

    # Inputs: HF wants [B, T, C, H, W]; native wants [B, C, T, H, W].
    clip = torch.randn(1, args.frames, 3, 256, 256)
    one = torch.randn(1, 1, 3, 256, 256)
    dup = one.repeat(1, 2, 1, 1, 1)  # what VideoWM.dup_image feeds the encoder
    inputs = {"random_clip": clip, "dup_image": dup}

    print("running HF VJEPA2Model encoder (fp32) ...", flush=True)
    hf = VJEPA2Model.from_pretrained(str(args.hf_dir), torch_dtype=torch.float32, attn_implementation=args.hf_attn)
    hf = hf.eval().to(device)
    hf_out: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for name, x in inputs.items():
            hf_out[name] = hf.get_vision_features(x.to(device)).cpu()
    n_layers_hf = len(hf.encoder.layer)
    del hf
    if device.type == "cuda":
        torch.cuda.empty_cache()

    def run_native(model) -> dict[str, torch.Tensor]:
        res = {}
        with torch.inference_mode():
            for name, x in inputs.items():
                res[name] = model(x.permute(0, 2, 1, 3, 4).to(device)).cpu()
        return res

    print("running native encoder, fp32 weights ...", flush=True)
    native.to(device)
    nat_fp32 = run_native(native)
    checks: dict[str, dict] = {"fp32_native_vs_hf": {}, "fp16_rounded_native_vs_hf": {}, "fp16_rounded_vs_fp32_native": {}}
    for name in inputs:
        checks["fp32_native_vs_hf"][name] = {
            **compare(nat_fp32[name], hf_out[name]),
            "shape_native": list(nat_fp32[name].shape),
            "shape_hf": list(hf_out[name].shape),
        }

    rounded = {k: v.to(torch.float16) for k, v in converted.items()}
    if args.save_dtype == "float16":
        print("running native encoder, fp16-rounded weights ...", flush=True)
        native.load_state_dict({k: v.float() for k, v in rounded.items()}, strict=True)
        nat_fp16 = run_native(native)
        for name in inputs:
            checks["fp16_rounded_native_vs_hf"][name] = compare(nat_fp16[name], hf_out[name])
            checks["fp16_rounded_vs_fp32_native"][name] = compare(nat_fp16[name], nat_fp32[name])
        weight_round = max(float((rounded[k].float() - converted[k]).abs().max()) for k in converted)
        checks["fp16_weight_rounding_max_abs"] = weight_round
    native.cpu()

    d = int(nat_fp32["dup_image"].shape[-1])
    n_tok = int(nat_fp32["dup_image"].shape[1])
    grid = {"tokens_per_frame": n_tok, "grid": int(round(n_tok**0.5)), "embed_dim": d, "patch_size": int(native.patch_size)}
    ok = all(
        c["rel_fro"] <= args.rel_fp32 and c["max_abs_over_ref_max"] <= args.maxabs_norm_fp32
        for c in checks["fp32_native_vs_hf"].values()
    )
    if args.save_dtype == "float16":
        ok = ok and all(c["rel_fro"] <= args.rel_fp16 for c in checks["fp16_rounded_native_vs_hf"].values())
    ok = ok and grid["grid"] ** 2 == n_tok and grid["grid"] == 16 and d == 1408
    ok = ok and all(torch.isfinite(v).all() for v in nat_fp32.values())
    report = {
        "checks": checks,
        "criteria": {"rel_fp32": args.rel_fp32, "maxabs_norm_fp32": args.maxabs_norm_fp32, "rel_fp16": args.rel_fp16, "hf_attn": args.hf_attn},
        "grid": grid,
        "hf_num_layers": n_layers_hf,
        "equivalence_ok": bool(ok),
        "gpu": gpu_mem(),
    }
    print(json.dumps(report, indent=2))
    if not ok:
        raise SystemExit("conversion failed equivalence checks; nothing written")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    to_save = rounded if args.save_dtype == "float16" else {k: v.cpu() for k, v in converted.items()}
    torch.save({"encoder": {k: v.cpu().contiguous() for k, v in to_save.items()}}, args.output)

    # Round trip through the exact vendor path (utils.init_video_model, v2_open branch).
    ck = torch.load(args.output, map_location="cpu")
    ck_sd = {k.replace("module.backbone.", ""): v for k, v in ck["encoder"].items()}
    fresh = vit_v2_open.vit_giant_xformers(**NATIVE_KWARGS).eval()
    msg = fresh.load_state_dict(ck_sd, strict=False)
    if msg.missing_keys or msg.unexpected_keys:
        raise SystemExit(f"round-trip load_state_dict(strict=False) not clean: {msg}")
    rt_max = max(float((fresh.state_dict()[k].float() - to_save[k].float()).abs().max()) for k in to_save)
    if rt_max != 0.0:
        raise SystemExit(f"round-trip mismatch {rt_max}")
    report["roundtrip_max_abs"] = rt_max
    report["roundtrip_load_msg"] = "clean (no missing/unexpected keys)"
    del fresh, ck, ck_sd

    provenance = {
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "output_bytes": args.output.stat().st_size,
        "output_dtype": args.save_dtype,
        "output_format": {"top_level_key": "encoder", "prefix": "", "n_tensors": len(to_save)},
        "source_repo": HF_REPO,
        "source_file": "model.safetensors",
        "source_sha256": source_sha,
        "source_sha256_expected_from_hf_lfs": args.expected_source_sha256,
        "source_bytes": src_path.stat().st_size,
        "source_config_sha256": sha256_file(cfg_path),
        "source_config": hf_cfg,
        "native_kwargs": {k: (list(v) if isinstance(v, tuple) else v) for k, v in NATIVE_KWARGS.items()},
        "mapping": "inverse of transformers convert_vjepa2_to_hf.py::convert_encoder_keys; HF predictor dropped",
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "elapsed_s": time.time() - t0,
        "report": report,
    }
    args.provenance_dir.mkdir(parents=True, exist_ok=True)
    prov_path = args.provenance_dir / "vjepa2_vit_giant.provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print("WROTE", args.output, provenance["output_sha256"], provenance["output_bytes"])
    print("PROVENANCE", prov_path)
    if args.delete_source:
        os.remove(src_path)
        print("DELETED", src_path)


if __name__ == "__main__":
    main()
