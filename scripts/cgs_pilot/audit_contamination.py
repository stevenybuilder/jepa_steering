#!/usr/bin/env python3
"""Pre-probe stimulus audits: domain shift, action novelty, within-set duplicates.

Why domain shift, not contamination. The ``jepa_wm_droid`` checkpoint was trained
on DROID only (vendored config ``datasets: [DROID]``, ``override_datasets: true``;
JEPA-WM, arXiv:2512.24497, reports no RoboCasa finetuning). RoboCasa renders
therefore cannot be memorised training frames; the real risk is that they lie
far outside the training distribution, in which case predictor behaviour on them
is extrapolation and any "mechanism" is suspect.

(a) Domain-shift audit. Each stimulus context frame is embedded with the model's
    own frozen DINOv3 ViT-L/16 encoder (mean-pooled patch tokens) and its
    nearest-neighbour distance to a DROID reference subsample is reported
    relative to the DROID-to-DROID (leave-one-episode-out) nearest-neighbour
    distance, following the feature-space NN protocol of Somepalli et al.,
    "Diffusion Art or Digital Forgery?", CVPR 2023 (arXiv:2212.03860) and the
    ratio-to-in-distribution-NN normalisation of Rahman et al., WACV 2025
    (frame-level nearest-neighbour novelty relative to in-distribution NN). A ratio near 1
    means the stimuli are as close to DROID as DROID frames are to each other;
    ratios well above 1 quantify the shift. Reference: ``droid_reference.py``
    (``droid_100`` RLDS sample, ``exterior_image_2_left``). If the reference
    cache is absent the audit is marked ``deferred`` with the exact command.

(b) Action-novelty audit. Stimulus 3x7 action chunks vs DROID chunks (JEPA-WM
    delta-action convention, see ``droid_reference.py``) under the released
    planner scaling ``max_norms = [0.1, 0.75]`` (dims 0-5 / 0.1, dim 6 / 0.75):
    nearest-neighbour Euclidean distance and DTW distance (Sakoe & Chiba 1978),
    each relative to DROID-to-DROID NN. Plus: within-pair action identity across
    hazard levels, leave-one-scene-out 1-NN hazard-from-actions accuracy (must be
    at chance), and an explicit flag when all scenes share identical action
    sequences (no unseen-action control exists yet).

(c) Within-stimulus duplicate audit across seeds (pixel signature of context
    frames, hazard centroid distance).

Latents come from ``latent_cache.py`` (``--cache``); DROID frame latents are
computed once with ``--repo/--config/--checkpoint`` into ``--droid-latents``.

``--domain driving`` (v0.8): the DROID reference is not applicable (the driving
predictor is trained by us on MetaDrive clips) and is skipped; instead
``--training-reference`` (npz with ``pooled`` [N, D] mean-pooled DINOv3 latents
of training context frames, ``clip_id`` [N] group ids, optionally
``action_chunks`` [M, T, A] and ``chunk_clip`` [M]) drives the same NN /
nearest-training-clip audit (the v0.6 "nearest-training-clip audit"). Actions
are 2-D (steer, throttle) and are compared unscaled; hazard-from-actions chance
is 1 / n_levels (4 levels); hazard centroids come from ``hazard_centroid_xy``
(``target_centroid_xy`` still accepted).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from protocol import canonical_json, sha256_file  # noqa: E402
from stats_utils import finite  # noqa: E402
from token_groups import DOMAINS  # noqa: E402

MAX_NORMS = np.array([0.1] * 6 + [0.75], dtype=np.float64)
DROID_DEFERRED_CMD = (
    "/opt/conda/bin/python code/cgs_pilot/droid_reference.py --cache-dir /root/cgs-pilot/reference/droid_100"
)


# --------------------------------------------------------------------------- #
# Pure-numpy helpers (tested on CPU)
# --------------------------------------------------------------------------- #


def pixel_signature(frames: np.ndarray, size: int = 32) -> np.ndarray:
    frames = np.asarray(frames)
    if frames.ndim == 3:
        frames = frames[None]
    n, h, w, _ = frames.shape
    gray = frames[..., :3].astype(np.float32).mean(axis=-1)
    bh, bw = h // size, w // size
    gray = gray[:, : bh * size, : bw * size].reshape(n, size, bh, size, bw).mean(axis=(2, 4))
    flat = gray.reshape(n, -1)
    flat = flat - flat.mean(axis=1, keepdims=True)
    return flat / np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-6)


def pairwise_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d2 = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * a @ b.T
    return np.sqrt(np.maximum(d2, 0.0))


def nearest_other_group(features: np.ndarray, groups: list | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = pairwise_l2(features, features)
    g = np.asarray(groups)
    d[g[:, None] == g[None, :]] = np.inf
    idx = d.argmin(axis=1)
    return d[np.arange(len(d)), idx], idx


def nearest_reference(features: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = pairwise_l2(features, reference)
    idx = d.argmin(axis=1)
    return d[np.arange(len(d)), idx], idx


def loso_nn_accuracy(features: np.ndarray, labels: np.ndarray, scene_ids: list[str]) -> float:
    dist, idx = nearest_other_group(features, scene_ids)
    labels = np.asarray(labels)
    valid = np.isfinite(dist)
    return float((labels[idx[valid]] == labels[valid]).mean()) if valid.any() else float("nan")


def summarize(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size), "min": float(x.min()), "p05": float(np.percentile(x, 5)), "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)), "max": float(x.max()), "mean": float(x.mean()),
    }


def scale_actions(chunks: np.ndarray, scale: np.ndarray | None = MAX_NORMS) -> np.ndarray:
    """[..., 7] -> divide dims 0-5 by 0.1 and dim 6 by 0.75 (released max_norms); ``scale=None`` leaves actions unscaled."""

    chunks = np.asarray(chunks, dtype=np.float64)
    if scale is None:
        return chunks
    if chunks.shape[-1] != len(scale):
        raise ValueError(f"action dim {chunks.shape[-1]} does not match the scaling vector of length {len(scale)}")
    return chunks / scale


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Classic DTW (Sakoe & Chiba 1978) between two [T, D] sequences, Euclidean local cost."""

    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    n, m = len(a), len(b)
    cost = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    return float(acc[n, m])


def domain_shift_ratio(stim: np.ndarray, ref: np.ndarray, ref_groups: np.ndarray) -> dict[str, Any]:
    """NN distance of stimuli to reference relative to leave-one-group-out reference NN distance."""

    d_stim, idx = nearest_reference(stim, ref)
    d_ref, _ = nearest_other_group(ref, ref_groups)
    d_ref = d_ref[np.isfinite(d_ref)]
    base = float(np.median(d_ref)) if d_ref.size else float("nan")
    return {
        "stimulus_to_reference_nn": summarize(d_stim),
        "reference_to_reference_loo_nn": summarize(d_ref),
        "ratio_median": float(np.median(d_stim) / base) if base and np.isfinite(base) else float("nan"),
        "ratio_per_item": (d_stim / base).tolist() if base and np.isfinite(base) else [],
        "fraction_stimuli_beyond_reference_p95": (
            float((d_stim > np.percentile(d_ref, 95)).mean()) if d_ref.size else float("nan")
        ),
        "nearest_reference_index": idx.tolist(),
    }


def action_audit(actions: np.ndarray, hazard: np.ndarray, level: np.ndarray, scene_ids: list[str], chance: float = 0.5) -> dict[str, Any]:
    flat = actions.reshape(len(actions), -1).astype(np.float64)
    dist, _ = nearest_other_group(flat, scene_ids)
    by_scene: dict[str, dict[tuple[int, int], np.ndarray]] = {}
    for i, s in enumerate(scene_ids):
        by_scene.setdefault(s, {})[(int(hazard[i]), int(level[i]))] = flat[i]
    max_within = 0.0
    for cells in by_scene.values():
        for a in (0, 1):
            if (0, a) in cells and (1, a) in cells:
                max_within = max(max_within, float(np.abs(cells[(0, a)] - cells[(1, a)]).max()))
    distinct = {a: len({tuple(np.round(flat[i], 6)) for i in range(len(flat)) if level[i] == a}) for a in (0, 1)}
    acc_h = loso_nn_accuracy(flat, hazard, scene_ids)
    return {
        "nearest_other_scene_action_distance": summarize(dist),
        "max_abs_action_diff_across_hazard_within_pair": max_within,
        "actions_identical_across_hazard": bool(max_within == 0.0),
        "distinct_action_sequences_per_level": {str(k): int(v) for k, v in distinct.items()},
        "all_scenes_share_identical_actions": bool(all(v == 1 for v in distinct.values())),
        "unseen_action_control_present": bool(any(v > 1 for v in distinct.values())),
        "loso_nn_accuracy_hazard_from_actions": acc_h,
        "loso_nn_accuracy_action_level_from_actions": loso_nn_accuracy(flat, level, scene_ids),
        "chance_hazard": chance,
        "hazard_from_actions_at_chance": bool(np.isfinite(acc_h) and abs(acc_h - chance) <= 0.15),
    }


def action_novelty(stim_chunks: np.ndarray, ref_chunks: np.ndarray, ref_groups: np.ndarray, max_ref: int, seed: int, scale: np.ndarray | None = MAX_NORMS) -> dict[str, Any]:
    rng = np.random.RandomState(seed)
    if len(ref_chunks) > max_ref:
        keep = rng.choice(len(ref_chunks), max_ref, replace=False)
        ref_chunks, ref_groups = ref_chunks[keep], ref_groups[keep]
    s = scale_actions(stim_chunks, scale)
    r = scale_actions(ref_chunks, scale)
    uniq_idx = np.unique(np.round(s.reshape(len(s), -1), 6), axis=0, return_index=True)[1]
    euclid = domain_shift_ratio(s.reshape(len(s), -1), r.reshape(len(r), -1), ref_groups)
    # DTW only for the distinct stimulus chunks (they are few) against the reference.
    dtw_nn = []
    for i in uniq_idx:
        dtw_nn.append(min(dtw_distance(s[i], r[j]) for j in range(len(r))))
    ref_dtw = []
    sub = rng.choice(len(r), min(60, len(r)), replace=False)
    for j in sub:
        others = [k for k in range(len(r)) if ref_groups[k] != ref_groups[j]]
        if others:
            ref_dtw.append(min(dtw_distance(r[j], r[k]) for k in rng.choice(others, min(200, len(others)), replace=False)))
    base = float(np.median(ref_dtw)) if ref_dtw else float("nan")
    return {
        "scaling": "dims 0-5 / 0.1, dim 6 / 0.75 (released max_norms)" if scale is not None else "unscaled (driving: steer, throttle)",
        "euclidean": {k: v for k, v in euclid.items() if k != "nearest_reference_index"},
        "dtw": {
            "distinct_stimulus_chunks": int(len(uniq_idx)),
            "stimulus_to_reference_nn": summarize(np.asarray(dtw_nn)),
            "reference_to_reference_loo_nn_subsample": summarize(np.asarray(ref_dtw)),
            "ratio_median": float(np.median(dtw_nn) / base) if dtw_nn and np.isfinite(base) and base else float("nan"),
        },
        "n_reference_chunks_used": int(len(r)),
    }


def duplicate_audit(context_sig: np.ndarray, scene_ids: list[str], centroids: np.ndarray | None, pixel_threshold: float, centroid_threshold_px: float) -> dict[str, Any]:
    dist, idx = nearest_other_group(context_sig, scene_ids)
    flagged = [
        {"cell_index": int(i), "nearest_cell_index": int(idx[i]), "pixel_signature_distance": float(dist[i])}
        for i in np.where(dist <= pixel_threshold)[0]
    ]
    result = {
        "nearest_other_scene_context_distance": summarize(dist),
        "pixel_threshold": pixel_threshold,
        "near_duplicate_context_pairs": flagged,
        "n_near_duplicate_context_cells": len(flagged),
    }
    if centroids is not None:
        cd, _ = nearest_other_group(np.asarray(centroids, dtype=np.float64), scene_ids)
        result["nearest_other_scene_hazard_centroid_px"] = summarize(cd)
        result["n_cells_with_hazard_centroid_within_threshold"] = int((cd <= centroid_threshold_px).sum())
        result["centroid_threshold_px"] = centroid_threshold_px
    return result


# --------------------------------------------------------------------------- #
# I/O and model-side helpers
# --------------------------------------------------------------------------- #


def load_cells(artifacts: Path) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    rows = [json.loads(ln) for ln in (artifacts / "manifest.jsonl").read_text().splitlines() if ln.strip()]
    context, actions = [], []
    for r in rows:
        with np.load(artifacts / r["artifact"]) as cell:
            context.append(np.asarray(cell["context_frames"][-1]))
            actions.append(np.asarray(cell["model_actions"], dtype=np.float32))
    return rows, np.stack(context), np.stack(actions)


def encode_pooled(model, frames: np.ndarray, device, batch: int = 8) -> np.ndarray:
    import torch

    from model_action_sensitivity import encode_frames

    outs = []
    for i in range(0, len(frames), batch):
        z = encode_frames(model, frames[i : i + batch], device).float()  # [1, T, 16, 16, D]
        outs.append(z.reshape(z.shape[0], z.shape[1], -1, z.shape[-1]).mean(dim=2)[0].cpu().numpy())
    return np.concatenate(outs, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=None, help="latent_cache.py npz (stimulus latents)")
    ap.add_argument("--droid-reference", type=Path, default=Path("/root/cgs-pilot/reference/droid_100/droid_100_reference.npz"))
    ap.add_argument("--droid-latents", type=Path, default=None, help="npz cache of pooled DROID frame latents (built if model args given)")
    ap.add_argument("--repo", type=Path)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--model-name", default="jepa_wm_droid")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-reference-chunks", type=int, default=3000)
    ap.add_argument("--pixel-dup-threshold", type=float, default=0.05)
    ap.add_argument("--centroid-dup-threshold-px", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--domain", choices=list(DOMAINS), default="egg")
    ap.add_argument("--training-reference", type=Path, default=None, help="driving: npz with pooled [N,D], clip_id [N] (+ action_chunks [M,T,A], chunk_clip [M])")
    args = ap.parse_args()

    rows, context, actions = load_cells(args.artifacts)
    scene_ids = [r["pair_id"] for r in rows]
    hazard = np.asarray([int(r["hazard"]) for r in rows])
    level = np.asarray([int(r["candidate_action"]) for r in rows])
    centroids = np.asarray([r.get("target_centroid_xy", r.get("hazard_centroid_xy", [np.nan, np.nan])) for r in rows], dtype=np.float64)
    driving = args.domain == "driving"

    report: dict[str, Any] = {
        "artifacts": str(args.artifacts),
        "manifest_sha256": sha256_file(args.artifacts / "manifest.jsonl"),
        "n_cells": len(rows),
        "n_scenes": len(set(scene_ids)),
        "framing": (
            "jepa_wm_droid trained on DROID only (config datasets: [DROID]; arXiv:2512.24497): RoboCasa frame "
            "contamination is impossible; this audit quantifies domain shift and action novelty instead."
        ) if not driving else (
            "driving predictor trained by us on MetaDrive clips (protocol v0.8): the factorial stimuli are held-out seeds "
            "of the same simulator, so the risk is trajectory copying from the nearest training clip; this audit reports "
            "nearest-training-clip distance (frames, actions) relative to training-to-training NN when --training-reference is given."
        ),
    }
    if driving:
        report["domain"] = "driving"
        report["hazard_levels_present"] = sorted(int(h) for h in set(hazard.tolist()))
    chance = 1.0 / max(len(set(hazard.tolist())), 1) if driving else 0.5
    report["action_audit"] = action_audit(actions, hazard, level, scene_ids, chance)
    report["duplicate_audit"] = duplicate_audit(
        pixel_signature(context), scene_ids, centroids if np.isfinite(centroids).all() else None,
        args.pixel_dup_threshold, args.centroid_dup_threshold_px,
    )

    # ---- domain shift (needs stimulus latents + DROID reference) ----
    stim_pooled = None
    if args.cache is not None and args.cache.exists():
        from latent_cache import load_cache

        c = load_cache(args.cache)
        ctx = c["context"].astype(np.float32)  # [n, 16, 16, D]
        stim_pooled = ctx.reshape(len(ctx), -1, ctx.shape[-1]).mean(1)
        if list(c["cell_id"]) != [r["cell_id"] for r in rows]:
            raise SystemExit("cache cell order does not match manifest")

    droid: dict[str, Any] = {}
    if driving:
        droid = {"status": "not_applicable", "reason": "domain driving: the predictor is trained on MetaDrive clips, not DROID; see training_reference_audit"}
        train: dict[str, Any] = {}
        if args.training_reference is None or not args.training_reference.exists():
            train = {"status": "deferred", "reason": "no --training-reference npz (pooled [N,D], clip_id [N], optional action_chunks/chunk_clip)"}
        else:
            with np.load(args.training_reference, allow_pickle=False) as z:
                ref_pooled = np.asarray(z["pooled"], dtype=np.float32)
                ref_clip = np.asarray(z["clip_id"])
                ref_chunks = np.asarray(z["action_chunks"]) if "action_chunks" in z.files else None
                ref_chunk_clip = np.asarray(z["chunk_clip"]) if "chunk_clip" in z.files else None
            train = {"status": "run", "n_reference_frames": int(len(ref_pooled)), "n_reference_clips": int(len(set(ref_clip.tolist())))}
            if ref_chunks is not None and ref_chunk_clip is not None and ref_chunks.shape[-1] == actions.shape[-1]:
                train["action_novelty"] = action_novelty(actions, ref_chunks, ref_chunk_clip, args.max_reference_chunks, args.seed, scale=None)
            if stim_pooled is None:
                train["frame_domain_shift"] = {"status": "deferred", "reason": "need stimulus latents (--cache)"}
            else:
                ds = domain_shift_ratio(stim_pooled, ref_pooled, ref_clip)
                ds["encoder"] = "DINOv3 ViT-L/16 (frozen encoder), mean-pooled patch tokens"
                ds["protocol"] = "nearest-training-clip NN distance relative to leave-one-clip-out training NN (Somepalli et al. CVPR 2023; Rahman et al. WACV 2025)"
                nn_idx = np.asarray(ds.pop("nearest_reference_index"))
                ds["nearest_training_clip_per_cell"] = {rows[i]["cell_id"]: str(ref_clip[j]) for i, j in enumerate(nn_idx.tolist())}
                train["frame_domain_shift"] = ds
        report["training_reference_audit"] = train
    elif not args.droid_reference.exists():
        droid["status"] = "deferred"
        droid["reason"] = f"reference cache missing: {args.droid_reference}"
        droid["command"] = DROID_DEFERRED_CMD
    else:
        with np.load(args.droid_reference) as z:
            ref_chunks = z["action_chunks"]
            ref_chunk_ep = z["chunk_episode"]
            ref_frame_ep = z["frame_episode"]
            ref_meta = json.loads(str(z["meta"]))
            ref_frames = z["frames"] if (args.droid_latents is None or not args.droid_latents.exists()) and args.repo else None
        droid["reference_meta"] = ref_meta
        droid["action_novelty"] = action_novelty(actions, ref_chunks, ref_chunk_ep, args.max_reference_chunks, args.seed)
        ref_lat = None
        if args.droid_latents is not None and args.droid_latents.exists():
            with np.load(args.droid_latents) as z:
                ref_lat = z["pooled"]
        elif ref_frames is not None and args.repo and args.config and args.checkpoint:
            import torch

            from model_action_sensitivity import load_model

            device = torch.device(args.device if torch.cuda.is_available() else "cpu")
            model, _ = load_model(args.repo, args.config, args.checkpoint, args.model_name, str(device))
            ref_lat = encode_pooled(model, ref_frames, device)
            if stim_pooled is None:
                stim_pooled = encode_pooled(model, context, device)
            if args.droid_latents is not None:
                args.droid_latents.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(args.droid_latents, pooled=ref_lat, frame_episode=ref_frame_ep)
        if ref_lat is None or stim_pooled is None:
            droid["frame_domain_shift"] = {
                "status": "deferred",
                "reason": "need stimulus latents (--cache) and DROID latents (--droid-latents or model args)",
            }
        else:
            ds = domain_shift_ratio(stim_pooled, ref_lat, ref_frame_ep)
            ds["encoder"] = "DINOv3 ViT-L/16 (JEPA-WM frozen encoder), mean-pooled patch tokens"
            ds["protocol"] = "Somepalli et al. CVPR 2023 feature-space NN; ratio to leave-one-episode-out DROID NN (Rahman et al. WACV 2025)"
            droid["frame_domain_shift"] = ds
        droid["status"] = "run"
    report["droid_reference_audit"] = droid

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json.loads(canonical_json(finite(report))), indent=2) + "\n")
    brief = {
        "n_cells": report["n_cells"],
        "n_scenes": report["n_scenes"],
        "all_scenes_share_identical_actions": report["action_audit"]["all_scenes_share_identical_actions"],
        "hazard_from_actions_at_chance": report["action_audit"]["hazard_from_actions_at_chance"],
        "near_duplicates": report["duplicate_audit"]["n_near_duplicate_context_cells"],
        "droid_status": droid.get("status"),
        "frame_domain_shift_ratio_median": (report.get("training_reference_audit", {}) if driving else droid).get("frame_domain_shift", {}).get("ratio_median"),
        "action_euclid_ratio_median": droid.get("action_novelty", {}).get("euclidean", {}).get("ratio_median"),
        "action_dtw_ratio_median": droid.get("action_novelty", {}).get("dtw", {}).get("ratio_median"),
    }
    print(json.dumps(finite(brief), indent=2))


if __name__ == "__main__":
    main()
