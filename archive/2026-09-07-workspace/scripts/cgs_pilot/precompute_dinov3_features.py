#!/usr/bin/env python3
"""Turn rendered driving clips into frozen DINOv3 ViT-L/16 token features for
``train_feature_predictor.py``.

Why
---
The driving substrate trains the DROID JEPA-WM predictor from scratch on cached encoder
tokens (``DRIVING_SUBSTRATE_PLAN.md``).  The features must be IDENTICAL to what the pilot
computes at test time, so the encoder is loaded through the pilot's own
``model_action_sensitivity.load_model(repo, config, checkpoint, "jepa_wm_droid", device)``
(DROID eval yaml + checkpoint, exactly as ``latent_cache.py``) and frames are pushed through
the pilot's own ``encode_frames`` -> ``EncPredWM.encode`` (``/255`` -> preprocessor transform
-> DINOv3 -> ``[B, T, 1, 16, 16, 1024]``).  A per-run assertion checks that batched encoding,
whole-clip ``model.encode`` and single-frame encoding agree, and a per-shard self-check
re-encodes stored frames through ``encode_frames`` and compares against what was written to
disk (within fp16/bf16 half-ulp rounding plus the fp32 encoder's batch-shape nondeterminism,
measured at <= 1.9e-5 on the 4090 with TF32 off).

Input
-----
``--shards GLOB [GLOB ...]``  MetaDrive generator shards; each ``.npz`` holds
``frames [n, T, H, W, 3] uint8 RGB`` and ``actions [n, T-1, A] float32`` with a meta JSON next
to it (``<stem>.meta.json`` / ``<stem>_meta.json`` / ``<stem>.json`` / ``meta.json`` in the same
directory; a list of dicts, or ``{"clips": [...]}``, one per clip: ``clip_id``, ``seed``, ``arm``,
hazard kind/pose, contact flag/step, min distance).  Field names are matched tolerantly (see
``hazard_flag`` / ``clip_record``); ``--hazard-key`` forces the bool source.

``--frames PATH[:key]``  single-array fallback (``.npy`` / ``.npz``) with optional ``--actions``
(zeros are written and flagged in the metadata when absent) and optional ``--meta``.

Output (``--out DIR``; the trainer's memmap directory layout)
-------------------------------------------------------------
* ``features.npy``       ``[N, T, G, G, D] float16`` written INCREMENTALLY through a memmap
  (never held in RAM); with ``--dtype bfloat16`` the file is ``features.bf16.npy`` (uint16
  bit patterns, decode with ``load_features``) so the trainer's float16 contract cannot be
  silently misread.
* ``actions.npy``        ``[N, T-1, A] float32`` copied from the shards.
* ``clips.json``         trainer meta format, one dict per clip: ``clip_id``, ``hazard`` (bool),
  ``arm``, ``contact``, ``contact_step``, ``min_distance``, ``seed``, ``split`` (stable hash of the
  seed: ``sha256(f"{split_seed}:{seed}")[0] < 128`` -> discovery else confirmation, as
  ``merge_heldout.split_seeds``), plus shard provenance.
* ``features.meta.json`` checkpoint/config/shard SHA-256s, N/T/G/D, dtype, pooling, timing,
  VRAM, verification results and the per-1k-clip size table for every pool x dtype option.
* ``progress.json``      per-shard completion ledger (``--resume`` skips finished shards).

``--pool patch2`` average-pools the 16x16 grid 2x2 -> 8x8 (D unchanged).  NOTE: pooled
features are no longer what ``model.encode`` returns at test time; the eval path would need
the same pooling.  Size per 1k clips at T=4, D=1024, 2-byte dtype: 2.10 GB (none) / 0.52 GB
(patch2); the dtype does not change the size, only the rounding (fp16: 11-bit mantissa,
range 65504; bf16: 8-bit mantissa, fp32 range).
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")  # two model jobs on the 64-core box oversubscribe BLAS

import argparse
import glob
import hashlib
import json
import math
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from model_action_sensitivity import encode_frames, load_model  # noqa: E402
from protocol import sha256_file  # noqa: E402

GRID = 16
DIM = 1024
POOLS = ("none", "patch2")
DTYPES = ("float16", "bfloat16")
FEATURE_FILE = {"float16": "features.npy", "bfloat16": "features.bf16.npy"}
HALF_ULP_REL = {"float16": 2.0**-11, "bfloat16": 2.0**-8}
NONE_KINDS = (None, "", "none", "null", "no_hazard", "ghost", "empty", False, 0)


# --------------------------------------------------------------------------- #
# Shards and metadata
# --------------------------------------------------------------------------- #


def npz_member_shape(path: Path, key: str) -> tuple[tuple[int, ...], np.dtype]:
    """Read ``key``'s shape/dtype from an ``.npz`` WITHOUT decompressing the member."""

    with zipfile.ZipFile(path) as zf:
        name = key if key in zf.namelist() else f"{key}.npy"
        with zf.open(name) as fh:
            version = np.lib.format.read_magic(fh)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(fh)
            else:
                shape, _, dtype = np.lib.format.read_array_header_2_0(fh)
    return tuple(int(s) for s in shape), np.dtype(dtype)


def find_meta(shard: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    stem = shard.with_suffix("")
    for cand in (Path(f"{stem}.meta.json"), Path(f"{stem}_meta.json"), Path(f"{stem}.json"), shard.parent / "meta.json"):
        if cand.exists():
            return cand
    return None


def load_meta_list(path: Path | None, n: int, shard: Path) -> list[dict[str, Any]]:
    if path is None:
        print(f"WARNING: no meta JSON for {shard.name}; clip_id/hazard fall back to defaults", file=sys.stderr)
        return [{} for _ in range(n)]
    raw = json.loads(path.read_text(encoding="utf-8"))
    clips = raw["clips"] if isinstance(raw, dict) and "clips" in raw else raw
    if not isinstance(clips, list):
        raise ValueError(f"{path}: meta must be a list of dicts or {{'clips': [...]}}")
    if len(clips) != n:
        raise ValueError(f"{path}: {len(clips)} meta entries but {shard.name} has {n} clips")
    return [dict(c) for c in clips]


def first_present(rec: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in rec and rec[k] is not None:
            return rec[k]
    return default


def hazard_flag(rec: dict[str, Any], key: str | None) -> bool:
    """Bool hazard label: a hazard object of a real kind is present.  ``key`` forces the
    source field; otherwise ``hazard`` (bool / dict with ``kind`` / str kind) then
    ``hazard_kind`` / ``hazard_type`` are tried, defaulting to False."""

    val = rec.get(key) if key else first_present(rec, "hazard", "hazard_kind", "hazard_type", "hazard_present")
    if isinstance(val, dict):
        val = first_present(val, "kind", "type", "present", "name")
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, float, np.integer, np.floating)):
        return bool(val)
    if isinstance(val, str):
        return val.strip().lower() not in {str(k).lower() for k in NONE_KINDS if isinstance(k, str)}
    return val not in NONE_KINDS


def split_of(seed: Any, split_seed: int, names: tuple[str, str]) -> str:
    """Stable per-seed membership as in ``merge_heldout.split_seeds``."""

    h = hashlib.sha256(f"{split_seed}:{seed}".encode()).digest()[0]
    return names[0] if h < 128 else names[1]


def clip_record(rec: dict[str, Any], shard: Path, index: int, global_index: int, args: argparse.Namespace) -> dict[str, Any]:
    hazard = rec.get("hazard")
    seed = first_present(rec, "seed", "env_seed", "scene_seed", default=None)
    if seed is None:
        seed = f"{shard.stem}:{index}"  # still deterministic, flagged below
    out = {
        "clip_id": str(first_present(rec, "clip_id", "id", "name", default=f"{shard.stem}_{index:05d}")),
        "hazard": hazard_flag(rec, args.hazard_key),
        "arm": first_present(rec, "arm", "condition", default=None),
        "hazard_kind": first_present(hazard if isinstance(hazard, dict) else rec, "kind", "hazard_kind", "hazard_type", default=hazard if isinstance(hazard, str) else None),
        "hazard_pose": first_present(hazard if isinstance(hazard, dict) else rec, "pose", "hazard_pose", default=None),
        "contact": bool(first_present(rec, "contact", "collision", "contact_flag", "collided", default=False)),
        "contact_step": first_present(rec, "contact_step", "collision_step", "first_contact_step", default=None),
        "min_distance": first_present(rec, "min_distance", "min_dist", "min_distance_m", default=None),
        "seed": seed,
        "seed_from_meta": "seed" in rec or "env_seed" in rec or "scene_seed" in rec,
        "split": split_of(seed, args.split_seed, args.split_names),
        "shard": shard.name,
        "index_in_shard": index,
        "index": global_index,
    }
    return json.loads(json.dumps(out, default=str))  # primitives only


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #


def pool_tokens(feats: torch.Tensor, pool: str) -> torch.Tensor:
    """``[n, G, G, D]`` -> ``[n, G', G', D]`` (fp32, on device)."""

    if pool == "none":
        return feats
    if pool == "patch2":
        n, g, _, d = feats.shape
        x = feats.permute(0, 3, 1, 2)  # [n, D, G, G]
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x.permute(0, 2, 3, 1).contiguous()
    raise ValueError(f"unknown pool {pool}")


def grid_after(pool: str, grid: int = GRID) -> int:
    return grid if pool == "none" else grid // 2


def to_storage(feats: torch.Tensor, dtype: str) -> np.ndarray:
    """fp32 tensor -> numpy array in the on-disk representation."""

    if dtype == "float16":
        return feats.to(torch.float16).cpu().numpy()
    if dtype == "bfloat16":
        return feats.to(torch.bfloat16).view(torch.int16).cpu().numpy().view(np.uint16)
    raise ValueError(dtype)


def from_storage(arr: np.ndarray, dtype: str) -> np.ndarray:
    """On-disk representation -> float32 numpy."""

    if dtype == "float16":
        return np.asarray(arr, dtype=np.float32)
    if dtype == "bfloat16":
        u16 = np.ascontiguousarray(np.asarray(arr, dtype=np.uint16))
        return torch.from_numpy(u16.view(np.int16)).view(torch.bfloat16).float().numpy()
    raise ValueError(dtype)


def load_features(out_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Reopen a store written by this script (memmap) and decode to float32 lazily via
    ``from_storage``; returns the raw memmap and the meta dict."""

    out_dir = Path(out_dir)
    meta = json.loads((out_dir / "features.meta.json").read_text(encoding="utf-8"))
    return np.load(out_dir / meta["features_file"], mmap_mode="r"), meta


def encode_batch(model, frames: np.ndarray, device: torch.device) -> torch.Tensor:
    """``frames [n, T, H, W, 3] uint8`` -> fp32 ``[n, T, G, G, D]`` through the pilot's
    ``encode_frames`` (frames are independent under ``batchify_video``, so the flattened
    ``[1, n*T, ...]`` call is the same computation as ``[n, T, ...]``)."""

    n, t = frames.shape[:2]
    flat = np.ascontiguousarray(frames.reshape(n * t, *frames.shape[2:]))
    z = encode_frames(model, flat, device)  # [1, n*T, 1, G, G, D]
    assert z.shape[0] == 1 and z.shape[1] == n * t and z.shape[2] == 1, tuple(z.shape)
    return z[0, :, 0].float().reshape(n, t, *z.shape[3:])


def verify_encoding(model, frames: np.ndarray, batched: torch.Tensor, device: torch.device, n_clips: int, atol: float, rtol: float) -> dict[str, Any]:
    """Batched vs whole-clip ``model.encode`` vs single-frame ``encode_frames`` on the first
    ``n_clips`` clips; raises if any pair exceeds ``atol + rtol * |ref|``."""

    stats: dict[str, Any] = {"n_clips": int(min(n_clips, frames.shape[0])), "atol": atol, "rtol": rtol, "pairs": {}}
    worst = {"batched_vs_clip": 0.0, "batched_vs_single": 0.0, "clip_vs_single": 0.0}
    worst_rel = dict.fromkeys(worst, 0.0)
    for c in range(stats["n_clips"]):
        clip = np.ascontiguousarray(frames[c])
        visual = torch.from_numpy(clip).permute(0, 3, 1, 2).unsqueeze(0).to(device)
        with torch.inference_mode():
            whole = model.encode(visual)[0, :, 0].float()  # [T, G, G, D]
        single = torch.stack([encode_frames(model, clip[t : t + 1], device)[0, 0, 0].float() for t in range(clip.shape[0])])
        b = batched[c]
        for name, (x, y) in {"batched_vs_clip": (b, whole), "batched_vs_single": (b, single), "clip_vs_single": (whole, single)}.items():
            diff = (x - y).abs()
            worst[name] = max(worst[name], float(diff.max()))
            worst_rel[name] = max(worst_rel[name], float((diff / (y.abs() + 1e-6)).max()))
            if not torch.all(diff <= atol + rtol * y.abs()):
                raise AssertionError(f"encoding mismatch ({name}) on clip {c}: max|diff|={float(diff.max()):.3e} > atol {atol} + rtol {rtol}*|ref|")
    stats["max_abs_diff"] = worst
    stats["max_rel_diff"] = worst_rel
    stats["feature_rms"] = float(batched[: stats["n_clips"]].pow(2).mean().sqrt())
    stats["feature_max_abs"] = float(batched[: stats["n_clips"]].abs().max())
    return stats


def rounding_check(stored: np.ndarray, reference: torch.Tensor, dtype: str) -> dict[str, Any]:
    """Stored (pooled, cast) values must equal the fp32 reference within half an ulp of the
    storage dtype; also counts fp16 overflows."""

    ref = reference.cpu().numpy().astype(np.float32)
    dec = from_storage(stored, dtype)
    finite = np.isfinite(dec)
    n_inf = int((~finite).sum())
    err = np.abs(dec[finite] - ref[finite])
    bound = HALF_ULP_REL[dtype] * np.abs(ref[finite]) + 1e-6
    ok = bool(np.all(err <= bound)) and n_inf == 0
    out = {"dtype": dtype, "ok": ok, "n_nonfinite": n_inf, "max_abs_err": float(err.max()) if err.size else 0.0,
           "max_rel_err": float((err / (np.abs(ref[finite]) + 1e-6)).max()) if err.size else 0.0,
           "ref_max_abs": float(np.abs(ref).max()), "n_values": int(ref.size)}
    if not ok:
        raise AssertionError(f"storage rounding check failed: {out}")
    return out


# --------------------------------------------------------------------------- #
# Sizes
# --------------------------------------------------------------------------- #


def size_table(t: int, d: int, project_n: int) -> dict[str, Any]:
    rows = {}
    for pool in POOLS:
        g = grid_after(pool)
        for dtype in DTYPES:
            per_clip = t * g * g * d * 2
            rows[f"{pool}/{dtype}"] = {"grid": g, "dim": d, "frames_per_clip": t, "bytes_per_clip": per_clip,
                                       "gb_per_1k_clips": per_clip * 1000 / 1e9, f"gb_at_{project_n}_clips": per_clip * project_n / 1e9}
    return rows


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def resolve_inputs(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Return ``[{path, kind, n, t, h, w, action_dim, meta_path}]`` from shard globs or the
    single-array fallback, in sorted order, without loading pixel data."""

    items: list[dict[str, Any]] = []
    if args.shards:
        paths = sorted({Path(p).resolve() for pattern in args.shards for p in glob.glob(pattern)})
        if not paths:
            raise SystemExit(f"no shards match {args.shards}")
        for p in paths:
            if p.suffix != ".npz":
                raise SystemExit(f"shard {p} must be .npz (frames + actions)")
            fshape, fdtype = npz_member_shape(p, "frames")
            ashape, _ = npz_member_shape(p, "actions")
            if len(fshape) != 5 or fshape[-1] != 3 or fdtype != np.uint8:
                raise SystemExit(f"{p}: frames must be [n, T, H, W, 3] uint8, got {fshape} {fdtype}")
            if len(ashape) != 3 or ashape[0] != fshape[0] or ashape[1] != fshape[1] - 1:
                raise SystemExit(f"{p}: actions must be [n, T-1, A] for frames {fshape}, got {ashape}")
            items.append({"path": p, "kind": "shard", "n": fshape[0], "t": fshape[1], "h": fshape[2], "w": fshape[3], "action_dim": ashape[2], "meta_path": find_meta(p, args.meta)})
    elif args.frames:
        spec, _, key = args.frames.partition(":")
        p = Path(spec).resolve()
        if p.suffix == ".npz":
            fshape, fdtype = npz_member_shape(p, key or "frames")
        else:
            arr = np.load(p, mmap_mode="r")
            fshape, fdtype = tuple(arr.shape), arr.dtype
        if len(fshape) != 5 or fshape[-1] != 3 or fdtype != np.uint8:
            raise SystemExit(f"{p}: frames must be [n, T, H, W, 3] uint8, got {fshape} {fdtype}")
        action_dim = args.action_dim
        if args.actions:
            a = np.load(args.actions, mmap_mode="r") if Path(args.actions).suffix == ".npy" else np.load(args.actions)["actions"]
            if a.shape[:2] != (fshape[0], fshape[1] - 1):
                raise SystemExit(f"actions {a.shape} do not match frames {fshape}")
            action_dim = int(a.shape[2])
        items.append({"path": p, "kind": "frames", "key": key or "frames", "n": fshape[0], "t": fshape[1], "h": fshape[2], "w": fshape[3], "action_dim": action_dim,
                      "meta_path": args.meta, "actions_path": Path(args.actions).resolve() if args.actions else None})
    else:
        raise SystemExit("--shards or --frames is required")
    ts = {it["t"] for it in items}
    ads = {it["action_dim"] for it in items}
    if len(ts) != 1 or len(ads) != 1:
        raise SystemExit(f"inconsistent T {ts} or action_dim {ads} across shards")
    return items


def load_item(item: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], bool]:
    """Load one input fully (frames uint8, actions float32, meta list, actions_are_zero)."""

    p = item["path"]
    if item["kind"] == "shard":
        with np.load(p) as z:
            frames = np.asarray(z["frames"])
            actions = np.asarray(z["actions"], dtype=np.float32)
        zero = False
    else:
        if p.suffix == ".npz":
            with np.load(p) as z:
                frames = np.asarray(z[item["key"]])
        else:
            frames = np.load(p)
        if item["actions_path"] is not None:
            ap = item["actions_path"]
            actions = np.asarray(np.load(ap) if ap.suffix == ".npy" else np.load(ap)["actions"], dtype=np.float32)
            zero = False
        else:
            actions = np.zeros((item["n"], item["t"] - 1, item["action_dim"]), dtype=np.float32)
            zero = True
    meta = load_meta_list(item["meta_path"], item["n"], p)
    return frames, actions, meta, zero


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("input")
    src.add_argument("--shards", nargs="+", default=None, help="glob(s) of generator .npz shards (frames, actions) with meta JSON next to them")
    src.add_argument("--frames", default=None, help="fallback: single frames array PATH[:key] (.npy/.npz), [n, T, H, W, 3] uint8")
    src.add_argument("--actions", default=None, help="with --frames: actions array (.npy or .npz['actions']); zeros if absent")
    src.add_argument("--action-dim", type=int, default=2, help="with --frames and no --actions")
    src.add_argument("--meta", type=Path, default=None, help="explicit meta JSON (applies to every input)")
    src.add_argument("--hazard-key", default=None, help="meta field to use for the bool hazard label (default: auto)")
    src.add_argument("--limit", type=int, default=0, help="stop after this many clips (smoke)")
    enc = ap.add_argument_group("encoder (pilot loader, DROID JEPA-WM)")
    enc.add_argument("--repo", type=Path, required=True)
    enc.add_argument("--config", type=Path, required=True, help="DROID eval yaml")
    enc.add_argument("--checkpoint", type=Path, required=True, help="jepa_wm_droid.pth.tar")
    enc.add_argument("--model-name", default="jepa_wm_droid")
    enc.add_argument("--device", default="cuda:0")
    enc.add_argument("--batch-frames", type=int, default=32, help="frames per encoder call (VRAM knob)")
    out = ap.add_argument_group("output")
    out.add_argument("--out", type=Path, required=True)
    out.add_argument("--pool", choices=POOLS, default="none")
    out.add_argument("--dtype", choices=DTYPES, default="float16")
    out.add_argument("--split-seed", type=int, default=0)
    out.add_argument("--split-names", default="discovery,confirmation", help="names for hash<128 / hash>=128 (e.g. train,val)")
    out.add_argument("--resume", action="store_true", help="skip shards already recorded in <out>/progress.json")
    out.add_argument("--project-n", type=int, default=12000, help="clip count for the projected size table")
    chk = ap.add_argument_group("verification")
    chk.add_argument("--verify-clips", type=int, default=2, help="clips of the first batch to cross-check batched/clip/single encoding")
    chk.add_argument("--verify-atol", type=float, default=1e-3, help="fp32 batched vs whole-clip vs single-frame encoding (observed max 1.9e-5)")
    chk.add_argument("--verify-rtol", type=float, default=1e-4)
    chk.add_argument("--self-check", type=int, default=2, help="per shard: re-encode this many random stored frames via encode_frames and compare to the memmap on disk")
    chk.add_argument("--self-check-atol", type=float, default=1e-4, help="absolute allowance on top of half-ulp for the encoder's fp32 batch-shape nondeterminism")
    args = ap.parse_args(argv)
    args.split_names = tuple(args.split_names.split(","))
    if len(args.split_names) != 2:
        raise SystemExit("--split-names needs two comma-separated names")

    t_start = time.time()
    items = resolve_inputs(args)
    n_total = sum(it["n"] for it in items)
    if args.limit:
        n_total = min(n_total, args.limit)
    t_frames, action_dim = items[0]["t"], items[0]["action_dim"]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_name = FEATURE_FILE[args.dtype]
    feat_path = out_dir / feat_name
    g_out = grid_after(args.pool)
    shape = (n_total, t_frames, g_out, g_out, DIM)
    store_dtype = np.float16 if args.dtype == "float16" else np.uint16

    progress_path = out_dir / "progress.json"
    progress: dict[str, Any] = {"done": {}, "shape": list(shape), "dtype": args.dtype, "pool": args.pool}
    if args.resume and progress_path.exists() and feat_path.exists():
        prev = json.loads(progress_path.read_text(encoding="utf-8"))
        if prev.get("shape") == list(shape) and prev.get("dtype") == args.dtype and prev.get("pool") == args.pool:
            progress = prev
            mm = np.load(feat_path, mmap_mode="r+")
            print(f"resume: {len(progress['done'])} shards already done")
        else:
            raise SystemExit(f"--resume: existing progress {prev.get('shape')}/{prev.get('dtype')}/{prev.get('pool')} != {list(shape)}/{args.dtype}/{args.pool}")
    else:
        mm = np.lib.format.open_memmap(feat_path, mode="w+", dtype=store_dtype, shape=shape)
    print(json.dumps({"n_clips": n_total, "T": t_frames, "grid": g_out, "dim": DIM, "action_dim": action_dim, "pool": args.pool, "dtype": args.dtype,
                      "features_file": str(feat_path), "bytes": int(np.prod(shape)) * 2, "n_inputs": len(items)}))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.empty(0, device=device)
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    model, _ = load_model(args.repo, args.config, args.checkpoint, args.model_name, str(device))
    load_s = time.time() - t0
    grid_model = int(getattr(model, "grid_size", GRID))
    if grid_model != GRID:
        raise SystemExit(f"model grid {grid_model} != {GRID}")

    clips: list[dict[str, Any]] = [None] * n_total  # type: ignore[list-item]
    actions_all = np.zeros((n_total, t_frames - 1, action_dim), dtype=np.float32)
    shard_meta: list[dict[str, Any]] = []
    verify: dict[str, Any] | None = None
    rounding: dict[str, Any] | None = None
    self_checks: list[dict[str, Any]] = []
    zero_actions = False
    frames_done = 0
    enc_seconds = 0.0
    clip_batch = max(1, args.batch_frames // t_frames)
    rng = np.random.default_rng(args.split_seed)
    start = 0
    for item in items:
        p = item["path"]
        n_here = item["n"]
        if start >= n_total:
            break
        n_use = min(n_here, n_total - start)
        key = str(p)
        t_shard = time.time()
        sha = sha256_file(p)
        if key in progress["done"] and progress["done"][key]["sha256"] == sha and progress["done"][key]["n"] == n_use:
            skipped = True  # resume: actions + meta only (npz members load lazily; frames untouched)
            if item["kind"] == "shard":
                with np.load(p) as z:
                    actions = np.asarray(z["actions"], dtype=np.float32)
                meta, zero = load_meta_list(item["meta_path"], n_here, p), False
            else:
                _, actions, meta, zero = load_item(item)
            frames, actions = None, actions[:n_use]
        else:
            frames, actions, meta, zero = load_item(item)
            frames, actions = frames[:n_use], actions[:n_use]
            skipped = False
        zero_actions = zero_actions or zero
        actions_all[start : start + n_use] = actions
        for i in range(n_use):
            clips[start + i] = clip_record(meta[i], p, i, start + i, args)
        if not skipped:
            for b0 in range(0, n_use, clip_batch):
                b1 = min(n_use, b0 + clip_batch)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                te = time.time()
                raw = encode_batch(model, frames[b0:b1], device)  # [b, T, G, G, D] fp32
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                enc_seconds += time.time() - te
                if verify is None and args.verify_clips > 0:
                    verify = verify_encoding(model, frames[b0:b1], raw, device, args.verify_clips, args.verify_atol, args.verify_rtol)
                    print(f"verify: {json.dumps(verify)}")
                b, t = raw.shape[:2]
                pooled = pool_tokens(raw.reshape(b * t, GRID, GRID, DIM), args.pool).reshape(b, t, g_out, g_out, DIM)
                stored = to_storage(pooled, args.dtype)
                if rounding is None:
                    rounding = rounding_check(stored, pooled, args.dtype)
                    print(f"rounding: {json.dumps(rounding)}")
                mm[start + b0 : start + b1] = stored
                frames_done += b * t
            mm.flush()
            if args.self_check > 0:
                for _ in range(args.self_check):
                    ci, ti = int(rng.integers(n_use)), int(rng.integers(t_frames))
                    z = encode_frames(model, np.ascontiguousarray(frames[ci, ti : ti + 1]), device)[0, 0, 0].float()  # [G, G, D]
                    ref = pool_tokens(z.unsqueeze(0), args.pool)[0]
                    disk = np.load(feat_path, mmap_mode="r")[start + ci, ti]
                    refn = ref.cpu().numpy()
                    err = np.abs(from_storage(disk, args.dtype) - refn)
                    # half-ulp of the storage dtype + the fp32 encoder's batch-shape nondeterminism
                    # (measured 1.9e-5 max on the 4090 with TF32 off; --self-check-atol 1e-4)
                    bound = HALF_ULP_REL[args.dtype] * np.abs(refn) + args.self_check_atol
                    excess = err - bound
                    ok = bool(np.all(excess <= 0))
                    rec = {"shard": p.name, "clip": start + ci, "t": ti, "max_abs_err": float(err.max()), "max_excess_over_bound": float(excess.max()),
                           "n_over_bound": int((excess > 0).sum()), "ok": ok}
                    self_checks.append(rec)
                    if not ok:
                        raise AssertionError(f"self-check failed: encode_frames(stored frame) != stored feature: {rec}")
            progress["done"][key] = {"sha256": sha, "n": n_use, "start": start, "end": start + n_use}
            progress_path.write_text(json.dumps(progress, indent=1) + "\n", encoding="utf-8")
        shard_meta.append({"path": str(p), "sha256": sha, "n": n_use, "start": start, "end": start + n_use, "meta_path": str(item["meta_path"]) if item["meta_path"] else None,
                           "meta_sha256": sha256_file(item["meta_path"]) if item["meta_path"] else None, "skipped_resume": skipped, "seconds": time.time() - t_shard})
        rate = (n_use * t_frames) / max(1e-9, time.time() - t_shard)
        print(json.dumps({"shard": p.name, "clips": [start, start + n_use], "skipped": skipped, "shard_s": round(time.time() - t_shard, 2), "frames_per_s_end_to_end": round(rate, 1)}), flush=True)
        start += n_use
        del frames
    if start != n_total:
        raise RuntimeError(f"wrote {start} clips, expected {n_total}")
    mm.flush()
    del mm

    # A resumed run carries the verification records of the runs that actually encoded.
    prev_runs: list[dict[str, Any]] = []
    meta_path = out_dir / "features.meta.json"
    if args.resume and meta_path.exists():
        prev = json.loads(meta_path.read_text(encoding="utf-8"))
        verify = verify or prev.get("verify_encoding")
        rounding = rounding or prev.get("rounding_check")
        self_checks = list(prev.get("self_checks") or []) + self_checks
        prev_runs = list(prev.get("previous_runs") or []) + [{"timing": prev.get("timing"), "peak_vram_gb": prev.get("peak_vram_gb"), "shards_encoded": [s["path"] for s in prev.get("shards", []) if not s.get("skipped_resume")]}]

    np.save(out_dir / "actions.npy", actions_all)
    (out_dir / "clips.json").write_text(json.dumps(clips, indent=1) + "\n", encoding="utf-8")
    total_s = time.time() - t_start
    hazard_frac = float(np.mean([c["hazard"] for c in clips])) if clips else float("nan")
    split_counts = {name: int(sum(c["split"] == name for c in clips)) for name in args.split_names}
    peak_vram = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else None
    try:
        commit = subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        commit = None
    meta_out = {
        "producer": "scripts/cgs_pilot/precompute_dinov3_features.py",
        "encoder": {"model_name": args.model_name, "loader": "model_action_sensitivity.load_model + encode_frames (EncPredWM.encode)", "config": str(args.config), "config_sha256": sha256_file(args.config),
                    "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256_file(args.checkpoint), "repo": str(args.repo), "repo_commit": commit,
                    "grid": GRID, "dim": DIM, "normalize_reps": bool(getattr(model, "normalize_reps", False)), "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32), "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32)},
        "features_file": feat_name, "storage_dtype": str(np.dtype(store_dtype)), "dtype": args.dtype, "pool": args.pool,
        "decode": "float16: as is; bfloat16: torch.from_numpy(u16.view(np.int16)).view(torch.bfloat16).float()  (see load_features/from_storage)",
        "N": n_total, "T": t_frames, "grid": g_out, "D": DIM, "action_dim": action_dim, "frame_hw": [items[0]["h"], items[0]["w"]],
        "bytes_features": int(np.prod(shape)) * 2, "actions_zero_filled": zero_actions,
        "hazard_fraction": hazard_frac, "split_seed": args.split_seed, "split_rule": "sha256(f'{split_seed}:{seed}')[0] < 128 -> " + args.split_names[0] + " else " + args.split_names[1],
        "split_counts": split_counts, "n_seed_from_meta": int(sum(c["seed_from_meta"] for c in clips)), "hazard_key": args.hazard_key,
        "shards": shard_meta,
        "verify_encoding": verify, "rounding_check": rounding, "self_checks": self_checks, "previous_runs": prev_runs,
        "timing": {"total_s": total_s, "model_load_s": load_s, "encoder_s": enc_seconds, "frames_encoded": frames_done,
                   "frames_per_s_encoder": frames_done / enc_seconds if enc_seconds > 0 else None,
                   "frames_per_s_end_to_end_excl_load": frames_done / max(1e-9, total_s - load_s) if frames_done else None,
                   "batch_frames": args.batch_frames, "clip_batch": clip_batch},
        "peak_vram_gb": peak_vram,
        "size_table": size_table(t_frames, DIM, args.project_n),
        "env": {"torch": torch.__version__, "numpy": np.__version__, "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "python": sys.version.split()[0],
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS")},
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    meta_path.write_text(json.dumps(meta_out, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    summary = {k: meta_out[k] for k in ("N", "T", "grid", "D", "dtype", "pool", "bytes_features", "hazard_fraction", "split_counts", "peak_vram_gb")}
    summary["timing"] = meta_out["timing"]
    summary["size_table_gb_per_1k"] = {k: round(v["gb_per_1k_clips"], 3) for k, v in meta_out["size_table"].items()}
    print("DONE " + json.dumps(summary, default=str))
    return meta_out


if __name__ == "__main__":
    main()
