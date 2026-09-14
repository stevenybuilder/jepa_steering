"""Prepare official HF DINOv3 weights for the authors' native PyTorch implementation.

This reverses the documented Transformers tensor-name/QKV conversion, not the
encoder algorithm. All learned tensors are accounted for exactly; derived RoPE
and key-bias masks remain the native constructor's deterministic buffers. Verify
feature agreement on fixed synthetic images before permitting model use.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

import torch

from offline_study.core.protocol import sha256, write_json


REPO = "facebook/dinov3-vitl16-pretrain-lvd1689m"
REVISION = "ea8dc2863c51be0a264bab82070e3e8836b02d51"
WEIGHT_SHA256 = "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179"
SOURCE_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"
CONVERSION_REFERENCE = "https://github.com/huggingface/transformers/blob/main/src/transformers/models/dinov3_vit/convert_dinov3_vit_to_hf.py"


def hf_key(native_key):
    exact = {"cls_token": "embeddings.cls_token", "mask_token": "embeddings.mask_token",
             "storage_tokens": "embeddings.register_tokens"}
    if native_key in exact:
        return exact[native_key]
    key = native_key.replace("patch_embed.proj", "embeddings.patch_embeddings")
    key = re.sub(r"blocks\.(\d+)\.attn\.proj", r"layer.\1.attention.o_proj", key)
    key = re.sub(r"blocks\.(\d+)\.attn\.", r"layer.\1.attention.", key)
    key = re.sub(r"blocks\.(\d+)\.ls(\d+)\.gamma", r"layer.\1.layer_scale\2.lambda1", key)
    key = re.sub(r"blocks\.(\d+)\.mlp\.fc1", r"layer.\1.mlp.up_proj", key)
    key = re.sub(r"blocks\.(\d+)\.mlp\.fc2", r"layer.\1.mlp.down_proj", key)
    return re.sub(r"blocks\.(\d+)\.norm", r"layer.\1.norm", key)


def native_state(hf, template):
    consumed, result, derived = set(), {}, []

    def fetch(key):
        # Official HF release has layer.*, newer library conversions use model.layer.*.
        if key not in hf and key.startswith("layer.") and "model." + key in hf:
            key = "model." + key
        if key not in hf or key in consumed:
            raise ValueError("Missing or reused learned HF tensor: " + key)
        consumed.add(key)
        return hf[key]

    for name, expected in template.items():
        key = hf_key(name)
        if name == "rope_embed.periods" or re.fullmatch(r"blocks\.\d+\.attn\.qkv\.bias_mask", name):
            value = expected.clone()
            derived.append(name)
        elif ".attn.qkv." in name:
            values = []
            for projection in ("q_proj", "k_proj", "v_proj"):
                source = key.replace("qkv", projection)
                if projection == "k_proj" and name.endswith(".bias") and source not in hf and "model." + source not in hf:
                    # Native masks the key-bias slice; HF intentionally omits it.
                    value = torch.zeros_like(expected[:expected.shape[0] // 3])
                    derived.append(name + ":masked_key_bias_zero")
                else:
                    value = fetch(source)
                values.append(value)
            value = torch.cat(values, dim=0)
        else:
            value = fetch(key)
            if name == "mask_token":
                value = value.reshape(expected.shape)
        if value.shape != expected.shape or value.dtype != expected.dtype or not torch.isfinite(value).all():
            raise ValueError("Unexpected native tensor shape/dtype/values: " + name)
        result[name] = value
    if consumed != set(hf):
        raise ValueError("Unaccounted official HF tensors: " + str(sorted(set(hf) - consumed)))
    return result, derived


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()
    if commit != SOURCE_COMMIT or subprocess.check_output(["git", "-C", str(args.source), "status", "--porcelain"], text=True).strip():
        raise ValueError("DINOv3 source must be the clean pinned official revision")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import DINOv3ViTConfig, DINOv3ViTModel

        write_json(args.output / "protocol.json", {"role": "encoder_loading_and_numerical_parity_not_task_evaluation",
            "hf_repo": REPO, "hf_revision": REVISION, "weights_sha256": WEIGHT_SHA256,
            "native_source_commit": commit, "conversion_reference": CONVERSION_REFERENCE,
            "source_sha256": sha256(Path(__file__)), "synthetic_seed": 2026090723,
            "feature_parity_atol": .001, "feature_parity_rtol": .001,
            "precision": "strict_float32_no_tf32", "outcome_access": False,
            "checkpoint_is_reconstructed_native_serialization_not_original_pth_bytes": True})
        paths = {}
        for name in ("config.json", "model.safetensors", "preprocessor_config.json", "LICENSE.md"):
            paths[name] = Path(hf_hub_download(REPO, name, revision=REVISION, local_dir=args.output / "official_hf"))
        if sha256(paths["model.safetensors"]) != WEIGHT_SHA256:
            raise ValueError("Official encoder checksum mismatch")
        cfg = DINOv3ViTConfig.from_pretrained(str(args.output / "official_hf"), local_files_only=True)
        if (cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads, cfg.patch_size, cfg.num_register_tokens) != (1024, 24, 16, 16, 4):
            raise ValueError("Wrong official DINOv3-L architecture")
        torch.manual_seed(2026090723)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.cuda.set_device(0)
        native = torch.hub.load(str(args.source), "dinov3_vitl16", source="local", pretrained=False).eval()
        weights = load_file(str(paths["model.safetensors"]), device="cpu")
        state, derived = native_state(weights, native.state_dict())
        native.load_state_dict(state, strict=True)
        serialized = args.output / "native_state_from_official_hf.pth"
        torch.save(state, serialized)
        reloaded = torch.load(serialized, map_location="cpu", weights_only=True)
        if set(reloaded) != set(state) or any(not torch.equal(reloaded[k], state[k]) for k in state):
            raise ValueError("Native serialization changed a tensor")
        hf = DINOv3ViTModel.from_pretrained(str(args.output / "official_hf"), local_files_only=True,
                                          attn_implementation="sdpa").eval().cuda()
        native = native.cuda()
        generator = torch.Generator().manual_seed(2026090723)
        fixtures = torch.rand(2, 3, 256, 256, generator=generator).cuda()
        torch.cuda.reset_peak_memory_stats()
        before = time.monotonic()
        native_features = native.forward_features(fixtures)
        expected = hf(pixel_values=fixtures).last_hidden_state
        checks = {}
        for name, actual, reference in (("patch_tokens", native_features["x_norm_patchtokens"], expected[:, 5:]),
                                        ("class_token", native_features["x_norm_clstoken"], expected[:, 0])):
            torch.testing.assert_close(actual, reference, atol=.001, rtol=.001)
            checks[name] = {"shape": list(actual.shape), "max_abs_difference": (actual-reference).abs().max().item(),
                            "mean_abs_difference": (actual-reference).abs().mean().item()}
        torch.cuda.synchronize()
        write_json(args.output / "report.json", {"status": "native_dinov3_encoder_weight_mapping_and_feature_parity_verified",
            "protocol_sha256": sha256(args.output / "protocol.json"),
            "official_files_sha256": {name: sha256(path) for name, path in paths.items()},
            "native_weights_sha256": sha256(serialized), "native_weights_path": str(serialized),
            "native_source_commit": commit, "all_hf_learned_tensors_consumed_once": True,
            "strict_native_load": True, "serialization_tensor_identity": True,
            "derived_native_buffers": derived, "feature_checks": checks,
            "gpu_forward_seconds": time.monotonic() - before, "seconds": time.monotonic() - started,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "task_evaluations": 0, "original_pth_file_sha256_known": False})
        write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    except Exception as exc:
        message = str(exc)
        if os.environ.get("HF_TOKEN"):
            message = message.replace(os.environ["HF_TOKEN"], "[redacted]")
        write_json(args.output / "FAILED.json", {"error": message, "task_evaluations": 0})
        raise RuntimeError("Encoder preparation failed; inspect the redacted receipt") from None


if __name__ == "__main__":
    main()
