#!/usr/bin/env python3
"""Convert the gated HF `facebook/dinov3-vitl16-pretrain-lvd1689m` safetensors
into the native DINOv3 ``DinoVisionTransformer`` state dict that JEPA-WM's
``DinoEncoder`` loads through ``torch.hub.load(<dinov3 repo>, "dinov3_vitl16",
weights=<path>)``.

Meta only publishes the native ``.pth`` through its own download form; the HF
repo carries the same weights in transformers layout. This script inverts the
key mapping used by transformers' own ``convert_dinov3_vit_to_hf.py`` and then
verifies the result three ways before writing anything:

1. native(converted) vs HF ``DINOv3ViTModel`` on random inputs at 224 and 256;
2. native(converted) vs HF on the COCO reference image used by transformers;
3. HF and native outputs vs the reference slices Meta/HF published for
   ``vitl16_lvd1689m`` (these were computed from the original ``.pth``).

Filename note: JEPA-WM's ``dino.py`` hardcodes the suffix ``-7c1da9a5.pth`` for
``dinov3_vitl16``; in the DINOv3 release that hash belongs to ViT-H+/16, while the
ViT-L/16 LVD-1689M file is ``-8aa4cbdd``. The suffix is only parsed to detect the
SAT-493M variant, and the architecture is fixed by ``dinov3_vitl16`` (1024-d,
depth 24), so the file is written under JEPA-WM's expected name and the
provenance JSON records the real origin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from io import BytesIO
from pathlib import Path

import torch


REFERENCE_IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
# From transformers/src/transformers/models/dinov3_vit/convert_dinov3_vit_to_hf.py
EXPECTED_CLS = [0.484527, -0.582214, 0.480636, 0.592040, 0.945166]
EXPECTED_PATCH = [-0.211367, -0.490863, -0.257131, 0.101763, 0.154511]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def convert(hf_state: dict[str, torch.Tensor], template: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()

    def take(key: str) -> torch.Tensor:
        consumed.add(key)
        return hf_state[key]

    out["cls_token"] = take("embeddings.cls_token")
    out["mask_token"] = take("embeddings.mask_token").squeeze(1)
    out["storage_tokens"] = take("embeddings.register_tokens")
    out["patch_embed.proj.weight"] = take("embeddings.patch_embeddings.weight")
    out["patch_embed.proj.bias"] = take("embeddings.patch_embeddings.bias")
    out["norm.weight"] = take("norm.weight")
    out["norm.bias"] = take("norm.bias")

    layer_ids = sorted({int(m.group(1)) for k in hf_state for m in [re.match(r"layer\.(\d+)\.", k)] if m})
    for i in layer_ids:
        p = f"layer.{i}."
        b = f"blocks.{i}."
        q_w, k_w, v_w = (take(p + f"attention.{n}_proj.weight") for n in ("q", "k", "v"))
        out[b + "attn.qkv.weight"] = torch.cat([q_w, k_w, v_w], dim=0)
        q_b = take(p + "attention.q_proj.bias")
        v_b = take(p + "attention.v_proj.bias")
        # HF has key_bias=False; native keeps a k-bias slot that is zeroed by
        # ``qkv.bias_mask`` (mask_k_bias=True). Store an explicit zero.
        out[b + "attn.qkv.bias"] = torch.cat([q_b, torch.zeros_like(q_b), v_b], dim=0)
        out[b + "attn.proj.weight"] = take(p + "attention.o_proj.weight")
        out[b + "attn.proj.bias"] = take(p + "attention.o_proj.bias")
        out[b + "ls1.gamma"] = take(p + "layer_scale1.lambda1")
        out[b + "ls2.gamma"] = take(p + "layer_scale2.lambda1")
        out[b + "mlp.fc1.weight"] = take(p + "mlp.up_proj.weight")
        out[b + "mlp.fc1.bias"] = take(p + "mlp.up_proj.bias")
        out[b + "mlp.fc2.weight"] = take(p + "mlp.down_proj.weight")
        out[b + "mlp.fc2.bias"] = take(p + "mlp.down_proj.bias")
        for n in ("norm1", "norm2"):
            out[b + f"{n}.weight"] = take(p + f"{n}.weight")
            out[b + f"{n}.bias"] = take(p + f"{n}.bias")
        # Deterministic buffers from the freshly built native model.
        out[b + "attn.qkv.bias_mask"] = template[b + "attn.qkv.bias_mask"].clone()
    out["rope_embed.periods"] = template["rope_embed.periods"].clone()

    unused = sorted(set(hf_state) - consumed)
    if unused:
        raise RuntimeError(f"unconsumed HF keys: {unused[:10]} (+{max(0, len(unused) - 10)})")
    missing = sorted(set(template) - set(out))
    extra = sorted(set(out) - set(template))
    if missing or extra:
        raise RuntimeError(f"key mismatch vs native template: missing={missing[:10]} extra={extra[:10]}")
    for k, v in out.items():
        if tuple(v.shape) != tuple(template[k].shape):
            raise RuntimeError(f"shape mismatch {k}: {tuple(v.shape)} vs {tuple(template[k].shape)}")
    return {k: v.contiguous().to(torch.float32) for k, v in out.items()}


def native_outputs(model, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    feats = model.forward_features(x)
    return feats["x_norm_clstoken"], feats["x_norm_patchtokens"]


def hf_outputs(model, x: torch.Tensor, n_reg: int) -> tuple[torch.Tensor, torch.Tensor]:
    out = model(pixel_values=x)
    return out.pooler_output, out.last_hidden_state[:, n_reg + 1 :]


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dinov3-repo", type=Path, required=True)
    ap.add_argument("--hf-dir", type=Path, required=True, help="dir with config.json + model.safetensors")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--atol", type=float, default=1e-4)
    args = ap.parse_args()

    sys.path.insert(0, str(args.dinov3_repo))
    from dinov3.hub.backbones import dinov3_vitl16  # type: ignore
    from safetensors.torch import load_file
    from transformers import DINOv3ViTModel

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    native = dinov3_vitl16(pretrained=False).eval()
    template = native.state_dict()
    hf_state = load_file(str(args.hf_dir / "model.safetensors"))
    converted = convert(hf_state, template)
    native.load_state_dict(converted, strict=True)
    native.to(device)

    hf = DINOv3ViTModel.from_pretrained(str(args.hf_dir), torch_dtype=torch.float32).eval().to(device)
    n_reg = int(hf.config.num_register_tokens)

    report: dict[str, object] = {}
    # Buffer sanity: native rope periods vs HF inv_freq (if exposed).
    inv = None
    for name, buf in hf.named_buffers():
        if name.endswith("inv_freq"):
            inv = buf
    if inv is not None:
        report["rope_periods_vs_hf_inv_freq_max_abs"] = max_abs(
            converted["rope_embed.periods"].to(inv.device), inv.flatten()[: converted["rope_embed.periods"].numel()]
        )

    checks = {}
    with torch.inference_mode():
        for res in (224, 256):
            x = torch.randn(2, 3, res, res, device=device)
            nc, np_ = native_outputs(native, x)
            hc, hp = hf_outputs(hf, x, n_reg)
            checks[f"random_{res}"] = {
                "cls_max_abs": max_abs(nc, hc),
                "patch_max_abs": max_abs(np_, hp),
                "patch_shape_native": list(np_.shape),
                "patch_shape_hf": list(hp.shape),
            }

        # Reference image (same preprocessing as transformers' converter).
        try:
            import httpx
            from PIL import Image
            from torchvision import transforms

            with httpx.stream("GET", REFERENCE_IMAGE_URL, timeout=30) as r:
                img = Image.open(BytesIO(r.read())).convert("RGB")
            tf = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((224, 224), antialias=True),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
            x = tf(img).unsqueeze(0).to(device)
            nc, np_ = native_outputs(native, x)
            hc, hp = hf_outputs(hf, x, n_reg)
            checks["reference_image"] = {
                "cls_max_abs_native_vs_hf": max_abs(nc, hc),
                "patch_max_abs_native_vs_hf": max_abs(np_, hp),
                "native_cls_first5": [round(v, 6) for v in nc[0, :5].tolist()],
                "native_patch_first5": [round(v, 6) for v in np_[0, 0, :5].tolist()],
                "expected_cls_first5": EXPECTED_CLS,
                "expected_patch_first5": EXPECTED_PATCH,
                "native_vs_published_cls_max_abs": max_abs(nc[0, :5].cpu(), torch.tensor(EXPECTED_CLS)),
                "native_vs_published_patch_max_abs": max_abs(np_[0, 0, :5].cpu(), torch.tensor(EXPECTED_PATCH)),
            }
        except Exception as exc:  # network failure should not hide the other checks
            checks["reference_image"] = {"error": repr(exc)}

    report["checks"] = checks
    ok = all(
        c["cls_max_abs"] <= args.atol and c["patch_max_abs"] <= args.atol
        for k, c in checks.items()
        if k.startswith("random_")
    )
    ref = checks.get("reference_image", {})
    if "error" not in ref:
        ok = ok and ref["cls_max_abs_native_vs_hf"] <= args.atol and ref["patch_max_abs_native_vs_hf"] <= args.atol
        ok = ok and ref["native_vs_published_cls_max_abs"] <= 1e-3 and ref["native_vs_published_patch_max_abs"] <= 1e-3
    report["equivalence_ok"] = bool(ok)
    print(json.dumps(report, indent=2))
    if not ok:
        raise SystemExit("conversion failed equivalence checks; nothing written")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in converted.items()}, args.output)
    # Round-trip through the exact loader path JEPA-WM uses.
    reloaded = dinov3_vitl16(pretrained=True, weights=str(args.output)).eval()
    rt = reloaded.state_dict()
    rt_max = max(max_abs(rt[k], converted[k].cpu()) for k in converted)
    report["roundtrip_max_abs"] = rt_max
    if rt_max != 0.0:
        raise SystemExit(f"round-trip mismatch {rt_max}")

    provenance = {
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "output_bytes": args.output.stat().st_size,
        "source_repo": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "source_file": "model.safetensors",
        "source_sha256": sha256_file(args.hf_dir / "model.safetensors"),
        "source_config_sha256": sha256_file(args.hf_dir / "config.json"),
        "filename_note": (
            "Written under JEPA-WM's hardcoded name '-7c1da9a5.pth'. In the DINOv3 release that "
            "suffix belongs to ViT-H+/16; ViT-L/16 LVD-1689M is '-8aa4cbdd'. Suffix only selects the "
            "SAT-493M variant; architecture is fixed by dinov3_vitl16."
        ),
        "torch": torch.__version__,
        "report": report,
    }
    prov_path = args.output.with_suffix(".provenance.json")
    prov_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print("WROTE", args.output, provenance["output_sha256"])
    print("PROVENANCE", prov_path)


if __name__ == "__main__":
    main()
