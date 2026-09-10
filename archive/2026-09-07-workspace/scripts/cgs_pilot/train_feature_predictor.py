#!/usr/bin/env python3
"""Train a JEPA-WM-style action-conditioned latent predictor on PRECOMPUTED frozen-encoder
features (single GPU), emitting artefacts that load through the UNCHANGED pilot loader.

Why
---
The driving substrate (``DRIVING_SUBSTRATE_PLAN.md``) trains the DROID JEPA-WM recipe
(frozen DINOv3 ViT-L/16 tokens + ``VisionTransformerAdaLN`` predictor) from scratch on our
own clips, in matched arms (0 / 10 / 50 % collision futures).  The vendor trainer
(``app/vjepa_wm/train.py``) runs the encoder in the loop and needs its dataset classes;
this script consumes a feature store instead and writes

* ``<out>/jepa-best.pth.tar``, ``<out>/jepa-latest.pth.tar``  vendor checkpoint format
  ``{"predictor": state_dict, "opt": state_dict | None, "scaler": None, "epoch": int}``
  (+ an extra ``"cgs_meta"`` key the vendor loader ignores),
* ``<out>/jepa-{best,latest}.meta.json``  data-manifest SHA-256s, arm label, hazard
  fraction, every hyper-parameter, per-epoch losses, timing and VRAM,
* ``<out>/eval_config.yaml``  hubconf-compatible eval yaml (mirrors the DROID eval yaml:
  encoder ``dinov3_vitl16`` through ``JEPAWM_OSSCKPT``/``JEPAWM_HOME``, decoder disabled),
* ``<out>/train_config.yaml``  vendor-style training yaml for provenance,
* ``<out>/probe.npz``  one eval-style unroll (context, actions, prediction) to verify the
  reloaded checkpoint reproduces the trainer's predictions.

Loading (unchanged code path)::

    model, _ = model_action_sensitivity.load_model(repo, out/"eval_config.yaml",
                                                   out/"jepa-best.pth.tar",
                                                   "jepa_wm_driving", "cuda:0")
    z = model.encode(frames); model.unroll(z, act_suffix=actions[T, B, A])

``hubconf._load_model_with_config`` resolves ``action_dim`` from
``app.plan_common.datasets.DATA_STATS[model_name.split("_")[-1]]``; the ``driving`` entry
is added by ``scripts/cgs_pilot/vendor_patches/0001-driving-data-stats.patch`` (additive,
DROID untouched; action dim via ``JEPAWM_DRIVING_ACTION_DIM``, default 2).

Vendor facts this script is built on (jepa-wms @ 13cf1d9)
---------------------------------------------------------
* Predictor: ``app.plan_common.models.AdaLN_vit.vit_predictor_AdaLN`` built exactly as in
  ``app.vjepa_wm.utils.init_video_model`` (pred_type ``AdaLN``): RoPE attention, block-causal
  attention mask over ``num_frames_pred`` frames of ``grid_size**2`` tokens, action enters
  only through ``action_encoder`` (Linear(A, W)) -> ``adaLN_modulation`` per block.  No
  action/proprio tokens; proprio disabled (``proprio_encoding: none``).
* Loss (training yaml ``..._2roll_4n``): under bf16 autocast, teacher-forced one-step L2
  over all ``T-1`` transitions divided by ``rollout_steps + 1``, plus a sequential rollout
  from ONE random prefix ``t`` (context = encoder frames ``0..t`` + the teacher-forced
  prediction of ``t+1``; further predictions appended with stop-gradient; window
  ``ctxt_window_train_rollout``) with per-step weight ``1 / rollout_steps``.
  ``rollout_steps=2`` -> ``loss = L_tf/3 + L_roll/2``.  ``--rollout-steps 3`` with ``T=4``
  turns this into a full unroll from the first frame.
* Optimiser: AdamW, betas (0.9, 0.995), eps 1e-8, weight decay excluded for biases and 1-D
  params, grad-clip 1.0, per-iteration warm-up + cosine LR and cosine WD schedules.
* ``EncPredWM.unroll`` seeds from the context frames and slides a ``ctxt_window`` (2)
  over [frames, actions]; ``unroll_eval`` below replicates it for validation and the probe.

Feature store
-------------
``--features`` is either an ``.npz`` (loaded in RAM) or a directory of ``.npy`` files
(memory-mapped; use this for >= 10k clips):

* ``features``  [N, T, G, G, D] float16  frozen encoder tokens per frame (G=16, D=1024)
* ``actions``   [N, T-1, A] float        action taken between frame t and t+1 (raw units;
                                          ``normalize_action: false`` as for DROID)
* ``proprio``   [N, T, P] optional        stored in metadata only (predictor is no-proprio)

``--meta`` JSON: a list (or ``{"clips": [...]}``) of ``{"clip_id": str, "hazard": bool,
"split": "train"|"val" (optional)}`` with one entry per clip.

Synthetic data (``--synthetic N,T,A``) has a learnable action-dependent drift
``z_{t+1} = z_t + sum_i a_i U_i + 0.1 eps`` inside a rank-128 token subspace, so the loss
must drop well below the copy-last-frame baseline.

Panel B: action-as-token / action-as-feature predictors (``--pred-type token|feature``)
---------------------------------------------------------------------------------------
``cross model design jepa.md`` Panel B trains a second JEPA predictor architecture on the
SAME cached features and clip multiset: ``src/models/ac_predictor.py::
VisionTransformerPredictorAC`` (the class behind the authors' V-JEPA-2-AC baseline, vendor
``pred_type: vjepa2_ac``), built exactly as ``init_video_model`` builds it:

* ``token``   one action token per frame (``action_encoder`` Linear(A, W)) prepended to the
  frame's ``G*G`` patch tokens; block-causal mask over ``num_frames`` frames of
  ``1 + G*G`` tokens (``build_action_block_causal_attention_mask``, full causal history, no
  local window); RoPE (the action token is rotated along the frame axis only).  The
  conditioning tokens are stripped from the output before ``predictor_proj``.
* ``feature`` the encoded action (Linear(A, ``--action-emb-dim``, default 64)) is concatenated
  to every patch token; the block width becomes ``W + action_emb_dim`` (must divide by the
  head count, head dim kept a multiple of 8 for the fused SDPA kernels); no extra tokens; the
  action slice is stripped from the output.

State/proprio token: NONE (``proprio_tokens: 0``), which is the vendor's own DROID
``vj2ac ... noprop`` recipe (``configs/vjepa_wm/droid_final_sweep/
droid_8fpcs_fps4_r256_vj2acvitgL1_repro_2roll_noprop_4n_asp1_wsd.yaml``).  The feature
store has no proprio (``has_proprio: false``) and the unchanged eval stack
(``EncPredWM.unroll`` with a Tensor context) passes ``proprio_features=None``, so a
zero-filled state token would (a) be a learned constant token, not state, and (b) break the
frozen loader.  ``eval_config.yaml`` mirrors the vendor vj2ac eval yaml (``pred_type:
vjepa2_ac``, ``action_conditioning``/``proprio_encoding`` = token|feature,
``proprio_encoder_inpred: true``) so ``model_action_sensitivity.load_model`` builds the
identical module; ``check_panelb_loader.py`` verifies reload equivalence.  Loss, optimiser,
schedules, split, checkpoint format and metadata are shared with the AdaLN path
(``forward_pred`` dispatches on the predictor class; the AdaLN code path is unchanged).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

MODEL_NAME = "jepa_wm_driving"  # hubconf env name -> DATA_STATS["driving"]
VENDOR_COMMIT = "13cf1d9c7e476f53c17714d2e0f1dc239a883ce0"
DROID_TRAIN_YAML = (
    "configs/vjepa_wm/droid_final_sweep/"
    "droid_4fpcs_fps4_r256_dv3vitl_asp1_pred_AdaLN_depth12_noprop_repro_2roll_4n.yaml"
)
DROID_EVAL_YAML = (
    "configs/evals/simu_env_planning/droid/jepa-wm/"
    "droid_L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r256_alpha0_ep64_decode.yaml"
)
IMAGENET_NORMALIZE = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class ModelCfg:
    pred_depth: int = 6
    pred_embed_dim: int = 512
    pred_num_heads: int = 8
    embed_dim: int = 1024  # frozen-encoder token dim D
    grid_size: int = 16
    img_size: int = 256
    num_frames_pred: int = 4  # T; sets the block-causal attention mask size
    action_dim: int = 2
    local_window_time: int = 3
    init_scale_factor_adaln: int = 10
    # Panel B (defaults reproduce the AdaLN predictor exactly)
    pred_type: str = "AdaLN"  # AdaLN | token | feature
    action_emb_dim: int = 0  # feature conditioning only: width of the per-patch action slice


PRED_TYPES = ("AdaLN", "token", "feature")
VENDOR_PRED_TYPE = {"AdaLN": "AdaLN", "token": "vjepa2_ac", "feature": "vjepa2_ac"}
AC_PREDICTOR_CLASS = "VisionTransformerPredictorAC"
VJ2AC_TRAIN_YAML = (
    "configs/vjepa_wm/droid_final_sweep/"
    "droid_8fpcs_fps4_r256_vj2acvitgL1_repro_2roll_noprop_4n_asp1_wsd.yaml"
)
VJ2AC_EVAL_YAML = (
    "configs/evals/simu_env_planning/droid/vj2ac/"
    "droid_L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r256_alpha0_ep64_decode.yaml"
)


def default_heads(width: int) -> int:
    return 16 if width >= 1024 else max(1, width // 64)


def predictor_kwargs(cfg: ModelCfg) -> dict[str, Any]:
    """Exactly the kwargs ``init_video_model`` passes to ``vit_predictor_AdaLN`` for the
    DROID JEPA-WM (pred_type AdaLN, action_encoder_inpred, no proprio)."""

    return dict(
        img_size=cfg.img_size,
        patch_size=cfg.img_size // cfg.grid_size,  # == encoder.patch_size (16 for dinov3_vitl16)
        num_frames=cfg.num_frames_pred,
        tubelet_size=1,
        embed_dim=cfg.embed_dim,
        predictor_embed_dim=cfg.pred_embed_dim,
        depth=cfg.pred_depth,
        num_heads=cfg.pred_num_heads,
        use_silu=None,
        use_rope=True,
        local_window=(cfg.local_window_time, -1, -1),
        use_activation_checkpointing=False,
        action_dim=cfg.action_dim,
        proprio_dim=None,
        act_mlp=False,
        prop_mlp=False,
        proprio_encoder_inpred=False,
        action_encoder_inpred=True,
        proprio_encoding="none",
        proprio_emb_dim=0,
        proprio_tokens=0,
        init_scale_factor_adaln=cfg.init_scale_factor_adaln,
    )


def conditioning_fields(cfg: ModelCfg) -> dict[str, Any]:
    """The ``model`` yaml fields that differ between the AdaLN predictor and the Panel B
    ``vjepa2_ac`` predictors (token / feature).  AdaLN values are the historical ones."""

    if cfg.pred_type == "AdaLN":
        return {
            "action_conditioning": "token",
            "proprio_encoding": "none",
            "action_encoder": {"action_tokens": 1, "action_emb_dim": 0, "act_mlp": False, "action_encoder_inpred": True},
            "proprio_encoder": {"proprio_tokens": 0, "proprio_emb_dim": 0, "prop_mlp": False, "proprio_encoder_inpred": False},
            "pred_type": "AdaLN",
        }
    if cfg.pred_type == "token":  # vendor DROID vj2ac noprop: action token only, no state token
        return {
            "action_conditioning": "token",
            "proprio_encoding": "token",
            "action_encoder": {"action_tokens": 1, "action_emb_dim": 0, "act_mlp": False, "action_encoder_inpred": True},
            "proprio_encoder": {"proprio_tokens": 0, "proprio_emb_dim": 0, "prop_mlp": False, "proprio_encoder_inpred": True},
            "pred_type": "vjepa2_ac",
        }
    if cfg.pred_type == "feature":  # vendor mw ..._predvj2acftcond_noprop: action slice on every patch token
        return {
            "action_conditioning": "feature",
            "proprio_encoding": "feature",
            "action_encoder": {"action_tokens": 0, "action_emb_dim": cfg.action_emb_dim, "act_mlp": False, "action_encoder_inpred": True},
            "proprio_encoder": {"proprio_tokens": 0, "proprio_emb_dim": 0, "prop_mlp": False, "proprio_encoder_inpred": True},
            "pred_type": "vjepa2_ac",
        }
    raise ValueError(f"unknown pred_type {cfg.pred_type!r} (expected one of {PRED_TYPES})")


def ac_predictor_kwargs(cfg: ModelCfg) -> dict[str, Any]:
    """Exactly the kwargs ``init_video_model`` passes to ``vit_ac_predictor`` (pred_type
    ``vjepa2_ac``) for ``conditioning_fields(cfg)`` with ``use_proprio=False`` (the eval
    loader's ``init_module`` sets ``proprio_dim=None`` then)."""

    fields = conditioning_fields(cfg)
    return dict(
        img_size=cfg.img_size,
        patch_size=cfg.img_size // cfg.grid_size,
        num_frames=cfg.num_frames_pred,
        tubelet_size=1,
        uniform_power=True,  # swallowed by **kwargs in the vendor class
        embed_dim=cfg.embed_dim,
        predictor_embed_dim=cfg.pred_embed_dim,
        action_dim=cfg.action_dim,
        proprio_dim=None,
        action_emb_dim=fields["action_encoder"]["action_emb_dim"],
        proprio_emb_dim=fields["proprio_encoder"]["proprio_emb_dim"],
        proprio_encoder_inpred=fields["proprio_encoder"]["proprio_encoder_inpred"],
        action_encoder_inpred=fields["action_encoder"]["action_encoder_inpred"],
        action_conditioning=fields["action_conditioning"],
        proprio_tokens=fields["proprio_encoder"]["proprio_tokens"],
        depth=cfg.pred_depth,
        is_frame_causal=True,
        num_heads=cfg.pred_num_heads,
        use_rope=True,
        use_sdpa=True,  # swallowed by **kwargs (the vendor Attention always uses SDPA when a mask is given)
        use_silu=None,  # -> nn.GELU, as use_SiLU: null in the vendor yamls
        wide_silu=True,
        use_extrinsics=False,
        use_activation_checkpointing=False,
    )


def block_width(cfg: ModelCfg) -> int:
    """Transformer block width: ``W`` for AdaLN/token, ``W + action_emb_dim`` for feature."""

    return cfg.pred_embed_dim + (cfg.action_emb_dim if cfg.pred_type == "feature" else 0)


def build_ac_predictor(cfg: ModelCfg, device: torch.device, with_encoder: bool = False):
    """Panel B: build ``VisionTransformerPredictorAC`` with the vendor factory.  ``with_encoder``
    goes through ``init_video_model(pred_type='vjepa2_ac')`` (frozen DINOv3 on CPU, never run)
    and asserts the direct build has identical parameter names/shapes."""

    import src.models.ac_predictor as vit_ac_pred  # type: ignore

    if block_width(cfg) % cfg.pred_num_heads:
        raise ValueError(f"block width {block_width(cfg)} not divisible by {cfg.pred_num_heads} heads")
    predictor = vit_ac_pred.vit_ac_predictor(**ac_predictor_kwargs(cfg))
    assert type(predictor).__name__ == AC_PREDICTOR_CLASS
    encoder = None
    if with_encoder:
        from app.vjepa_wm.utils import init_video_model  # type: ignore

        fields = conditioning_fields(cfg)
        vendor_pred, encoder, act_enc, prop_enc = init_video_model(
            device=torch.device("cpu"),
            img_size=cfg.img_size,
            num_frames_pred=cfg.num_frames_pred,
            enc_type="dino",
            enc_version="dinov3_vitl16",
            embed_dim=cfg.embed_dim,
            uniform_power=True,
            pred_type="vjepa2_ac",
            pred_depth=cfg.pred_depth,
            pred_embed_dim=cfg.pred_embed_dim,
            pred_num_heads=cfg.pred_num_heads,
            pred_use_extrinsics=False,
            tubelet_size=1,
            use_rope=True,
            use_SiLU=None,
            cfgs_attn_pattern={"local_window_time": cfg.local_window_time, "local_window_h": -1, "local_window_w": -1},
            use_activation_checkpointing=False,
            action_dim=cfg.action_dim,
            action_conditioning=fields["action_conditioning"],
            use_action=True,
            proprio_dim=None,
            proprio_encoding=fields["proprio_encoding"],
            use_proprio=False,
            **fields["action_encoder"],
            **fields["proprio_encoder"],
        )
        assert act_enc is None and prop_enc is None
        assert int(encoder.patch_size) == cfg.img_size // cfg.grid_size, "encoder patch size != img_size/grid"
        assert int(encoder.emb_dim) == cfg.embed_dim, "encoder dim != embed_dim"
        a = {k: tuple(v.shape) for k, v in predictor.state_dict().items()}
        b = {k: tuple(v.shape) for k, v in vendor_pred.state_dict().items()}
        assert a == b, "direct vit_ac_predictor build differs from init_video_model build"
        predictor = vendor_pred
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad_(False)
    return predictor.to(device), encoder


def build_vendor_predictor(cfg: ModelCfg, device: torch.device, with_encoder: bool = False):
    """Build the predictor with the vendor code.  ``with_encoder`` goes through
    ``init_video_model`` (instantiates the frozen DINOv3 encoder on CPU, never run here) and
    asserts the direct build has identical parameter names/shapes."""

    if cfg.pred_type != "AdaLN":  # Panel B
        return build_ac_predictor(cfg, device, with_encoder=with_encoder)

    from app.plan_common.models.AdaLN_vit import vit_predictor_AdaLN  # type: ignore

    predictor = vit_predictor_AdaLN(**predictor_kwargs(cfg))
    encoder = None
    if with_encoder:
        from app.vjepa_wm.utils import init_video_model  # type: ignore

        vendor_pred, encoder, act_enc, prop_enc = init_video_model(
            device=torch.device("cpu"),
            img_size=cfg.img_size,
            num_frames_pred=cfg.num_frames_pred,
            enc_type="dino",
            enc_version="dinov3_vitl16",
            embed_dim=cfg.embed_dim,
            uniform_power=True,
            pred_type="AdaLN",
            pred_depth=cfg.pred_depth,
            pred_embed_dim=cfg.pred_embed_dim,
            pred_num_heads=cfg.pred_num_heads,
            tubelet_size=1,
            use_rope=True,
            use_SiLU=None,
            cfgs_attn_pattern={"local_window_time": cfg.local_window_time, "local_window_h": -1, "local_window_w": -1},
            use_activation_checkpointing=False,
            action_dim=cfg.action_dim,
            action_conditioning="token",
            action_tokens=1,
            action_emb_dim=0,
            action_encoder_inpred=True,
            act_mlp=False,
            use_action=True,
            proprio_dim=None,
            proprio_encoding="none",
            proprio_tokens=0,
            proprio_emb_dim=0,
            proprio_encoder_inpred=False,
            prop_mlp=False,
            use_proprio=False,
        )
        assert act_enc is None and prop_enc is None
        assert int(encoder.patch_size) == cfg.img_size // cfg.grid_size, "encoder patch size != img_size/grid"
        assert int(encoder.emb_dim) == cfg.embed_dim, "encoder dim != embed_dim"
        a = {k: tuple(v.shape) for k, v in predictor.state_dict().items()}
        b = {k: tuple(v.shape) for k, v in vendor_pred.state_dict().items()}
        assert a == b, "direct vit_predictor_AdaLN build differs from init_video_model build"
        predictor = vendor_pred  # use the vendor-built instance
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad_(False)
    return predictor.to(device), encoder


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def sha256_bytes(data: bytes | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class FeatureStore:
    features: Any  # [N, T, G, G, D] float16 (np.ndarray or memmap)
    actions: np.ndarray  # [N, T-1, A] float32
    clips: list[dict[str, Any]]
    sha256: dict[str, str]
    source: str
    proprio: np.ndarray | None = None

    @property
    def n(self) -> int:
        return int(self.features.shape[0])

    @property
    def t(self) -> int:
        return int(self.features.shape[1])

    @property
    def grid(self) -> int:
        return int(self.features.shape[2])

    @property
    def dim(self) -> int:
        return int(self.features.shape[-1])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[-1])

    def validate(self) -> None:
        f, a = self.features, self.actions
        if f.ndim != 5 or f.shape[2] != f.shape[3]:
            raise ValueError(f"features must be [N, T, G, G, D], got {tuple(f.shape)}")
        if a.ndim != 3 or a.shape[0] != f.shape[0] or a.shape[1] != f.shape[1] - 1:
            raise ValueError(f"actions must be [N, T-1, A] for features {tuple(f.shape)}, got {tuple(a.shape)}")
        if len(self.clips) != f.shape[0]:
            raise ValueError(f"meta has {len(self.clips)} clips, features has {f.shape[0]}")
        if f.shape[1] < 2:
            raise ValueError("need at least 2 frames per clip")

    def hazard_fraction(self, idx: np.ndarray | None = None) -> float:
        rows = self.clips if idx is None else [self.clips[int(i)] for i in idx]
        if not rows:
            return float("nan")
        return float(np.mean([bool(r.get("hazard", False)) for r in rows]))


def load_meta(path: Path | None, n: int) -> list[dict[str, Any]]:
    if path is None:
        return [{"clip_id": f"clip_{i:06d}", "hazard": False} for i in range(n)]
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    clips = raw["clips"] if isinstance(raw, dict) else raw
    if not isinstance(clips, list):
        raise ValueError("meta JSON must be a list of clip dicts or {'clips': [...]}")
    return [dict(c) for c in clips]


def load_store(features: Path, meta: Path | None) -> FeatureStore:
    features = Path(features)
    shas: dict[str, str] = {}
    proprio = None
    if features.is_dir():
        feats = np.load(features / "features.npy", mmap_mode="r")
        acts = np.asarray(np.load(features / "actions.npy"), dtype=np.float32)
        shas["features.npy"] = sha256_path(features / "features.npy")
        shas["actions.npy"] = sha256_path(features / "actions.npy")
        if (features / "proprio.npy").exists():
            proprio = np.asarray(np.load(features / "proprio.npy"), dtype=np.float32)
            shas["proprio.npy"] = sha256_path(features / "proprio.npy")
        if meta is None and (features / "clips.json").exists():
            meta = features / "clips.json"
    else:
        with np.load(features) as z:
            feats = np.asarray(z["features"])
            acts = np.asarray(z["actions"], dtype=np.float32)
            proprio = np.asarray(z["proprio"], dtype=np.float32) if "proprio" in z.files else None
        shas[features.name] = sha256_path(features)
        if meta is None and features.with_suffix(".clips.json").exists():
            meta = features.with_suffix(".clips.json")
    clips = load_meta(meta, int(feats.shape[0]))
    if meta is not None:
        shas[Path(meta).name] = sha256_path(Path(meta))
    store = FeatureStore(feats, acts, clips, shas, source=str(features), proprio=proprio)
    store.validate()
    return store


def make_synthetic(n: int, t: int, grid: int, dim: int, action_dim: int, seed: int, hazard_fraction: float = 0.25, rank: int | None = None) -> FeatureStore:
    """Random tokens with a learnable action-dependent drift (see module docstring).

    Tokens live in a ``rank``-dimensional subspace of R^D (default ``min(128, D)``, unit
    per-dim variance) like real encoder features do; a predictor narrower than ``D``
    (``predictor_embed`` D -> W) could not even represent the identity on white noise."""

    rng = np.random.default_rng(seed)
    rank = min(rank or 128, dim)
    basis = np.linalg.qr(rng.normal(size=(dim, rank)))[0].T.astype(np.float32) * math.sqrt(dim / rank)  # [rank, D], rows orthogonal
    drift = rng.normal(0.0, 0.6, size=(action_dim, grid, grid, rank)).astype(np.float32)
    actions = rng.uniform(-1.0, 1.0, size=(n, t - 1, action_dim)).astype(np.float32)
    feats = np.empty((n, t, grid, grid, dim), dtype=np.float16)
    c = rng.normal(0.0, 1.0, size=(n, grid, grid, rank)).astype(np.float32)
    feats[:, 0] = c @ basis
    for step in range(t - 1):
        c = c + np.einsum("na,aghr->nghr", actions[:, step], drift) + 0.1 * rng.normal(size=c.shape).astype(np.float32)
        feats[:, step + 1] = c @ basis
    hazard = rng.uniform(size=n) < hazard_fraction
    clips = [{"clip_id": f"synthetic_{i:06d}", "hazard": bool(hazard[i])} for i in range(n)]
    shas = {"features(synthetic)": sha256_bytes(feats.tobytes()), "actions(synthetic)": sha256_bytes(actions.tobytes())}
    store = FeatureStore(feats, actions, clips, shas, source=f"synthetic(n={n},t={t},grid={grid},dim={dim},rank={rank},A={action_dim},seed={seed})")
    store.validate()
    return store


def write_store(store: FeatureStore, path: Path) -> None:
    """Write a store as ``.npz`` (file) or as a memmap-able ``.npy`` directory."""

    path = Path(path)
    if path.suffix == ".npz":
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, features=np.asarray(store.features), actions=store.actions)
    else:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "features.npy", np.asarray(store.features))
        np.save(path / "actions.npy", store.actions)
    meta_path = path.with_suffix(".clips.json") if path.suffix == ".npz" else path / "clips.json"
    meta_path.write_text(json.dumps(store.clips, indent=1) + "\n", encoding="utf-8")


def split_indices(store: FeatureStore, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray, str]:
    splits = [c.get("split") for c in store.clips]
    if all(s in ("train", "val") for s in splits):
        idx = np.arange(store.n)
        return idx[[s == "train" for s in splits]], idx[[s == "val" for s in splits]], "meta.split"
    perm = np.random.default_rng(seed).permutation(store.n)
    n_val = int(round(val_fraction * store.n))
    if n_val == 0:
        return perm, perm[: min(store.n, 32)], "val_is_train_subset"
    return np.sort(perm[n_val:]), np.sort(perm[:n_val]), f"random(seed={seed},val_fraction={val_fraction})"


def fetch_batch(store: FeatureStore, idx: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``feats [B, T, 1, G, G, D] float32`` and ``acts [B, T, A] float32`` (last action
    zero-padded; it only influences the discarded prediction of frame T)."""

    idx = np.sort(np.asarray(idx))
    feats = torch.from_numpy(np.ascontiguousarray(store.features[idx]))
    acts = torch.from_numpy(np.ascontiguousarray(store.actions[idx]))
    feats = feats.to(device, non_blocking=True).float().unsqueeze(2)
    acts = acts.to(device, non_blocking=True).float()
    acts = torch.cat([acts, torch.zeros_like(acts[:, :1])], dim=1)
    return feats, acts


# --------------------------------------------------------------------------- #
# Model interface, loss, eval-style unroll
# --------------------------------------------------------------------------- #


def forward_pred_ac(predictor: torch.nn.Module, feats: torch.Tensor, acts: torch.Tensor) -> torch.Tensor:
    """``VideoWM.forward_pred`` for pred_type ``vjepa2_ac`` (``normalize_reps: false``):
    ``predictor(feats.flatten(1, 4), actions [B, T, A], states=None)`` -> ``[B, T*(G*G), D]``
    with the action (+ proprio) tokens / feature slice already stripped by the vendor class."""

    b, t, v, g, _, d = feats.shape
    out, _, _ = predictor(feats.flatten(1, 4), acts, None)
    return out.view(b, t, v, g, g, d)


def forward_pred(predictor: torch.nn.Module, feats: torch.Tensor, acts: torch.Tensor) -> torch.Tensor:
    """``VideoWM.forward_pred`` for pred_type AdaLN: ``[B, T, 1, G, G, D]`` -> same shape.
    Panel B predictors (``VisionTransformerPredictorAC``) dispatch to ``forward_pred_ac``."""

    if type(predictor).__name__ == AC_PREDICTOR_CLASS:
        return forward_pred_ac(predictor, feats, acts)
    out, _, _ = predictor(feats, acts)
    b, t, n, d = out.shape
    g = int(math.isqrt(n))
    return out.view(b, t, 1, g, g, d)


def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(a.float(), b.float())


def jepa_wm_loss(
    predictor: torch.nn.Module,
    feats: torch.Tensor,
    acts: torch.Tensor,
    rollout_steps: int,
    prefixes: str,
    ctxt_window: int,
    stop_gradient: bool,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Vendor ``app/vjepa_wm/train.py`` step for the transition model (see module docstring)."""

    pred = forward_pred(predictor, feats, acts)
    tf_loss = mse(pred[:, :-1], feats[:, 1:])
    total = tf_loss / (rollout_steps + 1)
    stats: dict[str, float] = {"tf_l2": float(tf_loss.detach())}
    if rollout_steps > 1:
        n_frames = feats.shape[1]
        if n_frames - rollout_steps < 1:
            raise ValueError(f"rollout_steps={rollout_steps} needs T >= rollout_steps + 1 (T={n_frames})")
        if prefixes == "random":
            starts = [int(torch.randint(n_frames - rollout_steps, (1,), generator=generator))]
        elif prefixes == "first":
            starts = [0]
        elif prefixes == "all":
            starts = list(range(n_frames - rollout_steps))
        else:
            raise ValueError(f"unknown prefixes mode {prefixes}")
        weight = (1.0 / len(starts)) / rollout_steps  # vendor: (1/len(prefixes)) / ((rollout_steps-1)+1)
        for t in starts:
            vid = torch.cat([feats[:, : t + 1], pred[:, t : t + 1]], dim=1)
            act = acts[:, : t + 1]
            for h in range(rollout_steps - 1):
                act = torch.cat([act, acts[:, t + 1 + h : t + 2 + h]], dim=1)
                ctx = vid[:, -ctxt_window:]
                if stop_gradient:
                    ctx = ctx.detach()
                nxt = forward_pred(predictor, ctx, act[:, -ctxt_window:])[:, -1:]
                step_loss = mse(nxt, feats[:, t + 2 + h : t + 3 + h])
                total = total + step_loss * weight
                key = f"roll_l2_{h + 2}"
                stats[key] = stats.get(key, 0.0) + float(step_loss.detach()) / len(starts)
                vid = torch.cat([vid.detach() if stop_gradient else vid, nxt], dim=1)
    return total, stats


def unroll_eval(predictor: torch.nn.Module, z_ctx: torch.Tensor, acts: torch.Tensor, ctxt_window: int) -> torch.Tensor:
    """Replicates ``EncPredWM.unroll``: ``z_ctx [B, tau, 1, G, G, D]``, ``acts [B, H, A]`` ->
    predicted latents ``[B, H, 1, G, G, D]`` (context stripped)."""

    vid = z_ctx
    act: torch.Tensor | None = None
    for h in range(acts.shape[1]):
        new = acts[:, h : h + 1]
        act = new if act is None else torch.cat([act, new], dim=1)
        pred = forward_pred(predictor, vid[:, -ctxt_window:], act[:, -ctxt_window:])
        vid = torch.cat([vid, pred[:, -1:]], dim=1)
    return vid[:, z_ctx.shape[1] :]


@torch.no_grad()
def evaluate(predictor, store, idx, batch_size, device, ctxt_window, autocast_ctx) -> dict[str, float]:
    """Eval-style metrics: unroll from ONE frame for H = T-1 steps (as the planner and
    ``model_action_sensitivity.predict_cell`` do) plus the teacher-forced one-step L2 and the
    copy-last-frame baseline."""

    predictor.eval()
    sums: dict[str, float] = {}
    count = 0
    for start in range(0, len(idx), batch_size):
        feats, acts = fetch_batch(store, idx[start : start + batch_size], device)
        b = feats.shape[0]
        with autocast_ctx():
            pred_tf = forward_pred(predictor, feats, acts)
            roll = unroll_eval(predictor, feats[:, :1], acts[:, :-1], ctxt_window)
        per = {"tf_l2": float(mse(pred_tf[:, :-1], feats[:, 1:]))}
        horizon = roll.shape[1]
        step_losses = [float(mse(roll[:, h : h + 1], feats[:, h + 1 : h + 2])) for h in range(horizon)]
        per["unroll_l2"] = float(np.mean(step_losses))
        for h, value in enumerate(step_losses):
            per[f"unroll_l2_h{h + 1}"] = value
        per["copy_baseline_l2"] = float(np.mean([float(mse(feats[:, :1], feats[:, h + 1 : h + 2])) for h in range(horizon)]))
        for k, v in per.items():
            sums[k] = sums.get(k, 0.0) + v * b
        count += b
    predictor.train()
    return {k: v / max(count, 1) for k, v in sums.items()}


# --------------------------------------------------------------------------- #
# Optimiser and schedules (vendor init_opt / WarmupCosineSchedule / CosineWDSchedule)
# --------------------------------------------------------------------------- #


def make_optimizer(predictor: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    decay = [p for n, p in predictor.named_parameters() if p.requires_grad and "bias" not in n and p.ndim != 1]
    no_decay = [p for n, p in predictor.named_parameters() if p.requires_grad and ("bias" in n or p.ndim == 1)]
    groups = [
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0, "WD_exclude": True},
    ]
    return torch.optim.AdamW(groups, lr=args.start_lr, betas=(args.beta1, args.beta2), eps=args.eps)


class Schedules:
    def __init__(self, optimizer, total_steps: int, warmup_steps: int, args: argparse.Namespace) -> None:
        self.opt = optimizer
        self.total = max(1, total_steps)
        self.warmup = warmup_steps
        self.args = args
        self.step_count = 0

    def step(self) -> tuple[float, float]:
        self.step_count += 1
        a = self.args
        if self.step_count < self.warmup:
            lr = a.start_lr + (a.ref_lr - a.start_lr) * self.step_count / max(1, self.warmup)
        else:
            progress = (self.step_count - self.warmup) / max(1, self.total - self.warmup)
            lr = max(a.final_lr, a.final_lr + (a.ref_lr - a.final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)))
        progress_wd = self.step_count / self.total
        wd = a.final_weight_decay + (a.weight_decay - a.final_weight_decay) * 0.5 * (1.0 + math.cos(math.pi * progress_wd))
        wd = max(wd, a.final_weight_decay)
        for group in self.opt.param_groups:
            group["lr"] = lr
            if not group.get("WD_exclude", False):
                group["weight_decay"] = wd
        return lr, wd


# --------------------------------------------------------------------------- #
# Artefacts
# --------------------------------------------------------------------------- #


def eval_config(cfg: ModelCfg, out_dir: Path, checkpoint_name: str, ctxt_window: int, rollout: dict[str, Any]) -> dict[str, Any]:
    """Mirror of the DROID eval yaml with our predictor size, ``datasets: [driving]`` and the
    image decoder disabled (``heads_cfg.architectures: {}``).  Panel B predictors mirror the
    vendor vj2ac eval yaml (``pred_type: vjepa2_ac``) in the conditioning fields only."""

    fields = conditioning_fields(cfg)
    cgs_block: dict[str, Any] = {
        "model_name": MODEL_NAME,
        "note": (
            "Feature-trained driving predictor (train_feature_predictor.py). Load with "
            "model_action_sensitivity.load_model(repo, this_yaml, checkpoint, 'jepa_wm_driving', device); "
            "action_dim comes from DATA_STATS['driving'] (vendor_patches/0001-driving-data-stats.patch), "
            f"set JEPAWM_DRIVING_ACTION_DIM={cfg.action_dim} when it is not the default."
        ),
        "action_dim": cfg.action_dim,
        "template": DROID_EVAL_YAML,
    }
    if cfg.pred_type != "AdaLN":
        cgs_block["panel_b"] = {"pred_type": cfg.pred_type, "vendor_pred_type": fields["pred_type"], "predictor_class": AC_PREDICTOR_CLASS,
                                "conditioning_template": VJ2AC_EVAL_YAML, "state_token": "none (proprio_tokens 0, vendor noprop recipe)",
                                "block_width": block_width(cfg), "loader_check": "scripts/cgs_pilot/check_panelb_loader.py"}
    return {
        "nodes": 1,
        "tasks_per_node": 1,
        "cpus_per_task": 8,
        "copy_code": False,
        "folder": str(out_dir),
        "tag": f"cgs_pilot/{MODEL_NAME}/feature_trained",
        "eval_name": "simu_env_planning",
        "meta": {"quick_debug": False, "seed": 1, "eval_episodes": 0},
        "cgs_pilot": cgs_block,
        "model_kwargs": {
            "module_name": "app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds",
            "checkpoint": checkpoint_name,
            "pretrain_kwargs": {
                "grid_size": cfg.grid_size,
                "tubelet_size_enc": 1,
                "use_activation_checkpointing": False,
                "action_conditioning": fields["action_conditioning"],
                "proprio_encoding": fields["proprio_encoding"],
                "num_frames_pred": cfg.num_frames_pred,
                "visual_encoder": {
                    "enc_type": "dino",
                    "enc_version": "dinov3_vitl16",
                    "pretrain_enc_path": None,
                    "pretrain_enc_ckpt_key": None,
                    "embed_dim": cfg.embed_dim,
                    "enc_use_rope": None,
                    "enc_name": None,
                    "use_sdpa_enc": None,
                    "num_frames_enc": None,
                    "uniform_power": True,
                },
                "action_encoder": dict(fields["action_encoder"]),
                "proprio_encoder": dict(fields["proprio_encoder"]),
                "predictor": {
                    "tubelet_size": 1,
                    "pred_num_heads": cfg.pred_num_heads,
                    "pred_depth": cfg.pred_depth,
                    "pred_embed_dim": cfg.pred_embed_dim,
                    "pred_use_extrinsics": False,
                    "pred_type": fields["pred_type"],
                    "act_pred_projector": None,
                    "use_SiLU": None,
                    "use_rope": True,
                },
                "wm_encoding": {"batchify_video": True, "dup_image": False, "normalize_reps": False},
                "rollout_cfg": rollout,
                "attn": {"local_window_time": cfg.local_window_time, "local_window_h": -1, "local_window_w": -1},
                "heads_cfg": {"architectures": {}, "pretrain_dec_path": {}},
            },
            "data": {
                "dataset_type": "custom",
                "datasets": ["driving"],
                "datasets_weights": None,
                "seed": 234,
                "img_size": cfg.img_size,
                "custom": {"split_ratio": None, "frameskip": 1, "action_skip": 1, "state_skip": 1, "normalize_action": False, "traj_subset": True},
            },
            "data_aug": {
                "auto_augment": False,
                "random_horizontal_flip": False,
                "motion_shift": False,
                "random_resize_aspect_ratio": [1.0, 1.0],
                "random_resize_scale": [1.777, 1.777],
                "reprob": 0.0,
                "normalize": IMAGENET_NORMALIZE,
            },
            "wrapper_kwargs": {"ctxt_window": ctxt_window, "proprio_mode": "predict_proprio"},
        },
    }


def train_config(cfg: ModelCfg, args: argparse.Namespace, out_dir: Path, rollout: dict[str, Any], total_steps: int) -> dict[str, Any]:
    fields = conditioning_fields(cfg)
    cgs_block: dict[str, Any] = {
        "trainer": "scripts/cgs_pilot/train_feature_predictor.py",
        "template": DROID_TRAIN_YAML,
        "arm": args.arm,
        "note": "trained on precomputed frozen-encoder features; encoder never run during training",
    }
    if cfg.pred_type != "AdaLN":
        cgs_block["panel_b"] = {"pred_type": cfg.pred_type, "vendor_pred_type": fields["pred_type"], "conditioning_template": VJ2AC_TRAIN_YAML,
                                "note": "Panel B: same loss/optimiser/schedule/split as the AdaLN arm; only the predictor architecture differs"}
    return {
        "app": "vjepa_wm",
        "cgs_pilot": cgs_block,
        "nodes": 1,
        "tasks_per_node": 1,
        "folder": str(out_dir),
        "data": {"dataset_type": "feature_store", "datasets": ["driving"], "features": str(args.features) if args.features else args.synthetic, "meta": str(args.meta) if args.meta else None, "img_size": cfg.img_size,
                 "loader": {"batch_size": args.batch_size}, "custom": {"frameskip": 1, "action_skip": 1, "state_skip": 1, "normalize_action": False}},
        "loss": {"cos_loss_weight": 0.0, "l1_loss_weight": 0.0, "l2_loss_weight": 1.0, "smooth_l1_loss_weight": 0.0},
        "meta": {"freeze_encoder": True, "seed": args.seed, "dtype": args.dtype, "deterministic": True},
        "model": {
            "grid_size": cfg.grid_size,
            "tubelet_size_enc": 1,
            "use_activation_checkpointing": False,
            "action_conditioning": fields["action_conditioning"],
            "proprio_encoding": fields["proprio_encoding"],
            "num_frames_pred": cfg.num_frames_pred,
            "visual_encoder": {"enc_type": "dino", "enc_version": "dinov3_vitl16", "embed_dim": cfg.embed_dim, "uniform_power": True},
            "action_encoder": dict(fields["action_encoder"]),
            "proprio_encoder": dict(fields["proprio_encoder"]),
            "predictor": {"tubelet_size": 1, "pred_num_heads": cfg.pred_num_heads, "pred_depth": cfg.pred_depth, "pred_embed_dim": cfg.pred_embed_dim, "pred_use_extrinsics": False, "pred_type": fields["pred_type"], "use_rope": True},
            "wm_encoding": {"batchify_video": True, "dup_image": False, "normalize_reps": False},
            "rollout_cfg": rollout,
            "attn": {"local_window_time": cfg.local_window_time, "local_window_h": -1, "local_window_w": -1},
            "heads_cfg": {"architectures": {}, "pretrain_dec_path": {}},
        },
        "optimization": {
            "main_optimizer": "transition_model",
            "transition_model": {
                "iterations_per_epoch": int(math.ceil(total_steps / max(1, args.epochs))),
                "clip_grad": args.clip_grad,
                "betas": [args.beta1, args.beta2],
                "eps": args.eps,
                "weight_decay": args.weight_decay,
                "final_weight_decay": args.final_weight_decay,
                "num_epochs": args.epochs,
                "warmup": args.warmup_epochs,
                "start_lr": args.start_lr,
                "ref_lr": args.ref_lr,
                "final_lr": args.final_lr,
                "mixed_precision": args.dtype != "float32",
            },
        },
    }


def save_checkpoint(path: Path, predictor, optimizer, epoch: int, meta: dict[str, Any], with_opt: bool) -> None:
    meta = json.loads(json.dumps(meta, sort_keys=True, default=str))  # primitives only: vendor fetch_checkpoint uses torch.load(weights_only=True) on torch>=2.6
    payload = {
        "predictor": {k: v.detach().cpu() for k, v in predictor.state_dict().items()},
        "opt": optimizer.state_dict() if (with_opt and optimizer is not None) else None,
        "scaler": None,
        "epoch": int(epoch),
        "cgs_meta": meta,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    path.with_name(path.name.replace(".pth.tar", ".meta.json")).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def vendor_commit(repo: Path | None) -> str | None:
    if repo is None:
        return None
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def set_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def autocast_factory(device: torch.device, dtype_name: str) -> Callable[[], Any]:
    import contextlib

    if device.type != "cuda" or dtype_name == "float32":
        return contextlib.nullcontext
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]
    return lambda: torch.autocast("cuda", dtype=dtype)


def train(args: argparse.Namespace, predictor: torch.nn.Module, store: FeatureStore, device: torch.device, cfg: ModelCfg, encoder=None) -> dict[str, Any]:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    autocast_ctx = autocast_factory(device, args.dtype)
    train_idx, val_idx, split_rule = split_indices(store, args.val_fraction, args.seed)
    if len(train_idx) == 0:
        raise ValueError("empty training split")
    horizon = store.t - 1
    steps_per_epoch = int(math.ceil(len(train_idx) / args.batch_size))
    total_steps = steps_per_epoch * args.epochs
    optimizer = make_optimizer(predictor, args)
    schedules = Schedules(optimizer, total_steps, int(args.warmup_epochs * steps_per_epoch), args)
    generator = torch.Generator().manual_seed(args.seed)
    rollout = {
        "rollout_steps": args.rollout_steps,
        "train_rollout_prefixes": args.rollout_prefixes,
        "rollout_stop_gradient": not args.no_rollout_stop_gradient,
        "ctxt_window_train_rollout": args.ctxt_window_train_rollout,
        "do_parallel_rollout": None,
        "do_sequential_rollout": True,
        "prepend_gt": None,
        "sampling_scheduler": {"type": "linear", "start": 0.0, "end": 0.0},
    }
    import yaml

    (out_dir / "eval_config.yaml").write_text(yaml.safe_dump(eval_config(cfg, out_dir, "jepa-best.pth.tar", args.eval_ctxt_window, rollout), sort_keys=False), encoding="utf-8")
    (out_dir / "train_config.yaml").write_text(yaml.safe_dump(train_config(cfg, args, out_dir, rollout, total_steps), sort_keys=False), encoding="utf-8")

    n_params = sum(p.numel() for p in predictor.parameters())
    hyper = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    meta: dict[str, Any] = {
        "model_name": MODEL_NAME,
        "arm": args.arm,
        "checkpoint_format": "vendor jepa-wms {predictor, opt, scaler, epoch} + cgs_meta",
        "data": {"source": store.source, "sha256": store.sha256, "meta_json": str(args.meta) if args.meta else None,
                 "n_clips": store.n, "n_train": int(len(train_idx)), "n_val": int(len(val_idx)), "split_rule": split_rule,
                 "frames_per_clip": store.t, "horizon": horizon, "grid": store.grid, "feature_dim": store.dim, "action_dim": store.action_dim,
                 "hazard_fraction_all": store.hazard_fraction(), "hazard_fraction_train": store.hazard_fraction(train_idx), "hazard_fraction_val": store.hazard_fraction(val_idx),
                 "action_mean": np.asarray(store.actions, dtype=np.float64).reshape(-1, store.action_dim).mean(0).tolist(),
                 "action_std": np.asarray(store.actions, dtype=np.float64).reshape(-1, store.action_dim).std(0).tolist(),
                 "actions_normalized": False, "has_proprio": store.proprio is not None},
        "model": {**asdict(cfg), "params": int(n_params), "predictor_class": type(predictor).__name__, "encoder_present": encoder is not None,
                  "encoder_class": type(encoder).__name__ if encoder is not None else None, "encoder_run_during_training": False},
        "rollout_cfg": rollout,
        "hyperparameters": hyper,
        "schedule": {"steps_per_epoch": steps_per_epoch, "total_steps": total_steps, "warmup_steps": schedules.warmup},
        "vendor": {"repo": str(args.repo) if args.repo else None, "commit": vendor_commit(args.repo) or VENDOR_COMMIT, "train_template": DROID_TRAIN_YAML, "eval_template": DROID_EVAL_YAML},
        "env": {"torch": str(torch.__version__), "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "python": sys.version.split()[0]},
        "history": [],
        "best": None,
    }
    if cfg.pred_type != "AdaLN":
        meta["vendor"]["panel_b_templates"] = {"train": VJ2AC_TRAIN_YAML, "eval": VJ2AC_EVAL_YAML}
        meta["model"]["block_width"] = block_width(cfg)
        meta["model"]["state_token"] = "none (proprio_tokens 0, vendor noprop recipe)"
    print(json.dumps({k: meta[k] for k in ("model_name", "arm", "model", "schedule")}, indent=None, default=str))
    print(f"data: {meta['data']}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best_val = float("inf")
    t_start = time.time()
    predictor.train()
    for epoch in range(args.epochs):
        t_epoch = time.time()
        perm = train_idx[torch.randperm(len(train_idx), generator=generator).numpy()]
        loss_sum, stat_sums, n_seen = 0.0, {}, 0
        lr = wd = float("nan")
        for start in range(0, len(perm), args.batch_size):
            batch = perm[start : start + args.batch_size]
            feats, acts = fetch_batch(store, batch, device)
            lr, wd = schedules.step()
            with autocast_ctx():
                loss, stats = jepa_wm_loss(predictor, feats, acts, args.rollout_steps, args.rollout_prefixes,
                                           args.ctxt_window_train_rollout, not args.no_rollout_stop_gradient, generator)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), args.clip_grad)
            optimizer.step()
            b = len(batch)
            loss_sum += float(loss.detach()) * b
            for k, v in stats.items():
                stat_sums[k] = stat_sums.get(k, 0.0) + v * b
            n_seen += b
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_time = time.time() - t_epoch
        val = evaluate(predictor, store, val_idx, args.batch_size, device, args.eval_ctxt_window, autocast_ctx)
        row = {"epoch": epoch + 1, "train_loss": loss_sum / n_seen, **{f"train_{k}": v / n_seen for k, v in stat_sums.items()},
               **{f"val_{k}": v for k, v in val.items()}, "lr": lr, "wd": wd, "epoch_train_s": train_time, "epoch_total_s": time.time() - t_epoch,
               "ms_per_iter": 1000.0 * train_time / steps_per_epoch}
        if device.type == "cuda":
            row["peak_vram_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
        meta["history"].append(row)
        print(json.dumps(row), flush=True)
        is_best = val["unroll_l2"] < best_val
        if is_best:
            best_val = val["unroll_l2"]
            meta["best"] = {"epoch": epoch + 1, "val_unroll_l2": best_val, "val_tf_l2": val["tf_l2"]}
        meta["timing"] = {"elapsed_s": time.time() - t_start, "epoch_train_s_mean": float(np.mean([r["epoch_train_s"] for r in meta["history"]])),
                          "ms_per_iter_mean": float(np.mean([r["ms_per_iter"] for r in meta["history"]]))}
        save_checkpoint(out_dir / "jepa-latest.pth.tar", predictor, optimizer, epoch + 1, meta, with_opt=not args.no_save_opt)
        if is_best:
            save_checkpoint(out_dir / "jepa-best.pth.tar", predictor, optimizer, epoch + 1, meta, with_opt=False)
        if args.max_minutes and (time.time() - t_start) > 60 * args.max_minutes:
            print(f"stopping: --max-minutes {args.max_minutes} reached after epoch {epoch + 1}")
            break

    # Probe for reload verification: eval-style unroll from one frame of the first val clip.
    predictor.eval()
    feats, acts = fetch_batch(store, val_idx[:1], device)
    with torch.no_grad():
        roll = unroll_eval(predictor, feats[:, :1], acts[:, :-1], args.eval_ctxt_window)
    np.savez(out_dir / "probe.npz", context=feats[:, :1].cpu().numpy(), actions=acts[:, :-1].permute(1, 0, 2).cpu().numpy(),
             prediction=roll.permute(1, 0, 2, 3, 4, 5).cpu().numpy(), target=feats[:, 1:].cpu().numpy(),
             clip_index=np.asarray(int(val_idx[0])), ctxt_window=np.asarray(args.eval_ctxt_window), checkpoint=np.asarray("jepa-latest.pth.tar"))
    meta["probe"] = {"file": "probe.npz", "checkpoint": "jepa-latest.pth.tar", "clip_index": int(val_idx[0]), "ctxt_window": args.eval_ctxt_window,
                     "layout": "context [B=1, 1, 1, G, G, D]; actions [H, B, A] (EncPredWM act_suffix layout); prediction [H, B, 1, G, G, D] fp32"}
    (out_dir / "jepa-latest.meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta


def bench(args: argparse.Namespace, predictor: torch.nn.Module, store: FeatureStore, device: torch.device) -> dict[str, Any]:
    """Time fwd+bwd(+step) iterations for a per-epoch estimate at ``--bench-n`` clips."""

    autocast_ctx = autocast_factory(device, args.dtype)
    optimizer = None if args.bench_skip_optimizer else make_optimizer(predictor, args)
    generator = torch.Generator().manual_seed(args.seed)
    idx = np.arange(min(args.batch_size, store.n))
    feats, acts = fetch_batch(store, idx, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for i in range(args.bench_iters + 3):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.time()
        with autocast_ctx():
            loss, _ = jepa_wm_loss(predictor, feats, acts, args.rollout_steps, args.rollout_prefixes, args.ctxt_window_train_rollout, True, generator)
        loss.backward()
        if optimizer is not None:
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), args.clip_grad)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        else:
            predictor.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if i >= 3:
            times.append(time.time() - t0)
    ms = 1000.0 * float(np.mean(times))
    b = int(feats.shape[0])
    out = {"batch_size": b, "ms_per_iter": ms, "ms_per_clip": ms / b, "iters_for_bench_n": int(math.ceil(args.bench_n / b)),
           "epoch_s_at_bench_n": ms / 1000.0 * args.bench_n / b, "bench_n": args.bench_n, "optimizer_step_included": optimizer is not None,
           "params": sum(p.numel() for p in predictor.parameters())}
    if device.type == "cuda":
        out["peak_vram_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
    print(json.dumps(out))
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    data = ap.add_argument_group("data")
    data.add_argument("--features", type=Path, default=None, help=".npz file or directory of .npy (memmap)")
    data.add_argument("--meta", type=Path, default=None, help="clips JSON (clip_id, hazard, optional split)")
    data.add_argument("--synthetic", default=None, help="N,T,A  synthetic store with action-dependent drift (D and grid from --embed-dim/--grid)")
    data.add_argument("--write-synthetic", type=Path, default=None, help="also write the synthetic store here (.npz or dir)")
    data.add_argument("--val-fraction", type=float, default=0.1)
    data.add_argument("--arm", default="unlabelled", help="training-arm label recorded in the metadata (e.g. hazard50)")
    model = ap.add_argument_group("model")
    model.add_argument("--repo", type=Path, default=None, help="vendor jepa-wms checkout (needed unless a predictor is injected)")
    model.add_argument("--pred-depth", type=int, default=6)
    model.add_argument("--pred-embed-dim", type=int, default=512)
    model.add_argument("--pred-heads", type=int, default=None, help="default 16 for width>=1024 else width//64")
    model.add_argument("--pred-type", choices=PRED_TYPES, default="AdaLN",
                       help="AdaLN (Panel A, vendor vit_predictor_AdaLN) | token / feature (Panel B, vendor VisionTransformerPredictorAC = pred_type vjepa2_ac)")
    model.add_argument("--action-emb-dim", type=int, default=64,
                       help="--pred-type feature only: per-patch action slice width; block width W + this must divide by the heads and the head dim should stay a multiple of 8 "
                            "(512 + 64 = 576 / 8 heads = 72; 16 -> head dim 66 forces the SDPA math kernel: 2.5x VRAM, 1.6x time)")
    model.add_argument("--embed-dim", type=int, default=1024)
    model.add_argument("--grid", type=int, default=16)
    model.add_argument("--img-size", type=int, default=256)
    model.add_argument("--with-encoder", action="store_true", help="instantiate the frozen DINOv3 encoder on CPU via vendor init_video_model (never run)")
    loss = ap.add_argument_group("loss (vendor rollout_cfg)")
    loss.add_argument("--rollout-steps", type=int, default=2, help="vendor 2roll: 1 teacher-forced + (rollout_steps-1) sequential steps")
    loss.add_argument("--rollout-prefixes", choices=("random", "first", "all"), default="random")
    loss.add_argument("--ctxt-window-train-rollout", type=int, default=3)
    loss.add_argument("--no-rollout-stop-gradient", action="store_true")
    loss.add_argument("--eval-ctxt-window", type=int, default=2, help="EncPredWM ctxt_window used for validation/probe unrolls")
    opt = ap.add_argument_group("optimisation")
    opt.add_argument("--epochs", type=int, default=20)
    opt.add_argument("--batch-size", type=int, default=16)
    opt.add_argument("--ref-lr", type=float, default=5e-4)
    opt.add_argument("--start-lr", type=float, default=1e-6)
    opt.add_argument("--final-lr", type=float, default=1e-5)
    opt.add_argument("--warmup-epochs", type=float, default=1.0)
    opt.add_argument("--weight-decay", type=float, default=1e-7)
    opt.add_argument("--final-weight-decay", type=float, default=1e-6)
    opt.add_argument("--beta1", type=float, default=0.9)
    opt.add_argument("--beta2", type=float, default=0.995)
    opt.add_argument("--eps", type=float, default=1e-8)
    opt.add_argument("--clip-grad", type=float, default=1.0)
    opt.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    opt.add_argument("--seed", type=int, default=0)
    opt.add_argument("--max-minutes", type=float, default=0.0, help="stop after the epoch that crosses this wall-clock budget")
    io = ap.add_argument_group("output")
    io.add_argument("--out", type=Path, required=True)
    io.add_argument("--no-save-opt", action="store_true", help="omit optimizer state from jepa-latest.pth.tar")
    io.add_argument("--device", default="cuda:0")
    b = ap.add_argument_group("benchmark")
    b.add_argument("--bench-iters", type=int, default=0, help="time this many iterations instead of training")
    b.add_argument("--bench-n", type=int, default=10000)
    b.add_argument("--bench-skip-optimizer", action="store_true")
    return ap


def resolve_store(args: argparse.Namespace) -> FeatureStore:
    if args.synthetic:
        n, t, a = (int(x) for x in args.synthetic.split(","))
        store = make_synthetic(n, t, args.grid, args.embed_dim, a, args.seed)
        if args.write_synthetic:
            write_store(store, args.write_synthetic)
        return store
    if args.features is None:
        raise SystemExit("--features or --synthetic is required")
    return load_store(args.features, args.meta)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    set_determinism(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    store = resolve_store(args)
    if store.grid != args.grid or store.dim != args.embed_dim:
        raise SystemExit(f"store grid/dim {store.grid}/{store.dim} != --grid/--embed-dim {args.grid}/{args.embed_dim}")
    cfg = ModelCfg(pred_depth=args.pred_depth, pred_embed_dim=args.pred_embed_dim,
                   pred_num_heads=args.pred_heads or default_heads(args.pred_embed_dim), embed_dim=args.embed_dim,
                   grid_size=args.grid, img_size=args.img_size, num_frames_pred=store.t, action_dim=store.action_dim,
                   pred_type=args.pred_type, action_emb_dim=args.action_emb_dim if args.pred_type == "feature" else 0)
    if cfg.pred_type == "feature" and cfg.action_emb_dim <= 0:
        raise SystemExit("--pred-type feature needs --action-emb-dim > 0")
    if block_width(cfg) % cfg.pred_num_heads:
        raise SystemExit(f"block width {block_width(cfg)} (W{' + action_emb_dim' if cfg.pred_type == 'feature' else ''}) must divide by {cfg.pred_num_heads} heads")
    if (block_width(cfg) // cfg.pred_num_heads) % 8:
        print(f"WARNING: head dim {block_width(cfg) // cfg.pred_num_heads} is not a multiple of 8; SDPA falls back to the math kernel (much more VRAM/time)", flush=True)
    if args.repo is None:
        raise SystemExit("--repo is required to build the vendor predictor")
    sys.path.insert(0, str(args.repo.resolve()))
    torch.manual_seed(args.seed)
    predictor, encoder = build_vendor_predictor(cfg, device, with_encoder=args.with_encoder)
    if args.bench_iters > 0:
        bench(args, predictor, store, device)
        return
    train(args, predictor, store, device, cfg, encoder=encoder)


if __name__ == "__main__":
    main()
