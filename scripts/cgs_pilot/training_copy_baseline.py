#!/usr/bin/env python
"""Training-set interaction-copy baseline (memorisation control; pooled level).

`vla_learnings.md` rule 2 and `cross model design jepa.md` F4: the other-scene copy-delta baseline cannot tell whether a
self-trained predictor *memorised training futures*, because the training set is not factorial.  This control asks the
question directly at the level the training reference permits (mean-pooled DINOv3 latents per frame, all TRAIN clips):

    for every factorial cell, find the nearest TRAINING clip (context pooled latent + action chunk), copy that clip's
    pooled future delta (last future frame - context frame) as the "prediction", form the hazard x action DiD of those
    copies per scene, and score it against the true pooled DiD exactly as the model's pooled DiD is scored.

If the copier's DiD cosine/NMSE is as good as the model's, the behavioural PASS is compatible with retrieval of training
futures; if the copier is at chance while the model is not, memorisation of training clips is not the explanation.
Pooled level only (the reference stores no patch tokens), so this complements, not replaces, the token-level retrieval
baselines in retrieval_baseline.py.  Descriptive; no threshold is preregistered for it.

Inputs: --artifacts <eval dir with manifest.jsonl>, --cache <latent_cache.npz> (context/target/prediction arrays),
--training-reference <training_reference.npz: pooled [N_frames, D], clip_id, frame_index, action_chunks [M, T, A], chunk_clip>.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

CELL_ORDER = [(0, 0), (0, 1), (1, 0), (1, 1)]  # (hazard, action): H0A0, H0A1, H1A0, H1A1
ACTION_CHUNKS = {0: [0.0, -1.0], 1: [0.0, 0.5]}  # v0.7/v0.8 factorial chunks (steer, throttle), constant over the 3 model steps


def did(f: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    return (f[(1, 1)] - f[(1, 0)]) - (f[(0, 1)] - f[(0, 0)])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def nmse(pred: np.ndarray, true: np.ndarray) -> float:
    d = float(true @ true)
    return float(((pred - true) @ (pred - true)) / d) if d > 0 else float("nan")


def sign_flip_p(x: np.ndarray, n_perm: int = 5000, seed: int = 0) -> float:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    obs = x.mean()
    flips = rng.choice([-1.0, 1.0], size=(n_perm, len(x)))
    null = (flips * x).mean(1)
    return float((np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (n_perm + 1))


def boot_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    m = x[idx].mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def summarize(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    lo, hi = boot_ci(x)
    return {"median": float(np.nanmedian(x)), "mean": float(np.nanmean(x)), "ci_low": lo, "ci_high": hi, "n": int(np.isfinite(x).sum())}


def pooled_frames(arr: np.ndarray) -> np.ndarray:
    """[N, ..., tokens, D] -> [N, D] mean over tokens of the LAST frame/step."""
    a = np.asarray(arr)
    while a.ndim > 3:
        a = a[:, -1]
    return a.astype(np.float32).mean(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--training-reference", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seeds-file", type=Path, default=None, help="restrict to these scene seeds (discovery)")
    ap.add_argument("--hazard-levels", type=int, nargs="*", default=[1, 3], help="in-lane levels scored as H1 (1 pedestrian, 3 cone)")
    ap.add_argument("--null-level", type=int, default=0, help="off-path level used as H0 (0 sidewalk; 2 = H0' mirror pose)")
    ap.add_argument("--k", type=int, nargs="*", default=[1, 5], help="nearest-neighbour set sizes (average of the k copied deltas)")
    ap.add_argument("--action-weight", type=float, default=1.0, help="weight of the action-chunk distance (in units of the latent distance scale)")
    ap.add_argument("--model-label", default="")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.artifacts.joinpath("manifest.jsonl").read_text().splitlines() if l.strip()]
    keep = None
    if args.seeds_file and args.seeds_file.exists():
        keep = {int(x) for x in args.seeds_file.read_text().split()}
    cache = np.load(args.cache, allow_pickle=True)
    ctx = pooled_frames(cache["context"])
    true = pooled_frames(cache["target"])
    pred = pooled_frames(cache["prediction"])
    assert len(rows) == len(ctx) == len(true) == len(pred), (len(rows), ctx.shape, true.shape, pred.shape)

    ref = np.load(args.training_reference, allow_pickle=True)
    pooled, clip_id, frame_index = ref["pooled"].astype(np.float32), ref["clip_id"].astype(str), ref["frame_index"].astype(int)
    chunks, chunk_clip = ref["action_chunks"].astype(np.float32), ref["chunk_clip"].astype(str)
    by_clip: dict[str, dict[int, int]] = defaultdict(dict)
    for i, (c, f) in enumerate(zip(clip_id, frame_index)):
        by_clip[c][int(f)] = i
    chunk_of = {c: chunks[i] for i, c in enumerate(chunk_clip)}
    clips = [c for c in by_clip if 0 in by_clip[c] and max(by_clip[c]) >= 1 and c in chunk_of]
    T_last = {c: max(by_clip[c]) for c in clips}
    X0 = np.stack([pooled[by_clip[c][0]] for c in clips])                    # context frame of each train clip
    DZ = np.stack([pooled[by_clip[c][T_last[c]]] - pooled[by_clip[c][0]] for c in clips])  # pooled future delta
    A = np.stack([chunk_of[c].reshape(-1) for c in clips])                    # flattened action chunk
    lat_scale = float(np.median(np.linalg.norm(X0[1:] - X0[:-1], axis=1)) ** 2) or 1.0
    act_scale = float(np.median(np.linalg.norm(A[1:] - A[:-1], axis=1)) ** 2) or 1.0

    scenes: dict[str, dict[tuple[int, int], int]] = defaultdict(dict)
    for i, r in enumerate(rows):
        if keep is not None and int(r["seed"]) not in keep:
            continue
        scenes[r["pair_id"]][(int(r["hazard"]), int(r["candidate_action"]))] = i

    out: dict[str, Any] = {"model_label": args.model_label, "artifacts": str(args.artifacts), "cache": str(args.cache),
                           "training_reference": str(args.training_reference), "n_train_clips": len(clips), "n_scenes": len(scenes),
                           "level": "pooled (mean over DINOv3 patch tokens, last future step); training reference stores pooled frames only",
                           "nn_metric": f"||ctx - clip_ctx||^2 / {lat_scale:.4g} + {args.action_weight} * ||chunk - clip_chunk||^2 / {act_scale:.4g}",
                           "interpretation_scope": "memorisation control at the pooled level: a copier of the nearest TRAINING clip's future delta; descriptive, no preregistered threshold; complements retrieval_baseline.py",
                           "identities": {}}
    for h1 in args.hazard_levels:
        per_k: dict[str, dict[str, list[float]]] = {f"copy_k{k}": defaultdict(list) for k in args.k}
        model_stats: dict[str, list[float]] = defaultdict(list)
        nn_dist: list[float] = []
        used = 0
        for pid, cells in scenes.items():
            need = {(0, 0): (args.null_level, 0), (0, 1): (args.null_level, 1), (1, 0): (h1, 0), (1, 1): (h1, 1)}
            if not all(v in cells for v in need.values()):
                continue
            used += 1
            f_true = {k: true[cells[v]] - ctx[cells[v]] for k, v in need.items()}
            f_pred = {k: pred[cells[v]] - ctx[cells[v]] for k, v in need.items()}
            d_true, d_pred = did(f_true), did(f_pred)
            model_stats["cosine"].append(cosine(d_pred, d_true)); model_stats["nmse"].append(nmse(d_pred, d_true))
            for k in args.k:
                f_copy = {}
                for key, v in need.items():
                    i = cells[v]
                    a = np.tile(np.asarray(ACTION_CHUNKS[int(rows[i]["candidate_action"])], dtype=np.float32), (chunks.shape[1], 1)).reshape(-1)
                    dist = ((X0 - ctx[i]) ** 2).sum(1) / lat_scale + args.action_weight * ((A - a) ** 2).sum(1) / act_scale
                    nn = np.argsort(dist)[:k]
                    if k == args.k[0] and key == (1, 1):
                        nn_dist.append(float(math.sqrt(max(dist[nn[0]], 0.0))))
                    f_copy[key] = DZ[nn].mean(0)
                d_copy = did(f_copy)
                per_k[f"copy_k{k}"]["cosine"].append(cosine(d_copy, d_true)); per_k[f"copy_k{k}"]["nmse"].append(nmse(d_copy, d_true))
                per_k[f"copy_k{k}"]["cosine_copy_vs_model"].append(cosine(d_copy, d_pred))
        res: dict[str, Any] = {"n_scenes_used": used, "model": {m: summarize(np.array(v)) for m, v in model_stats.items()}, "nn_distance_h1a1": summarize(np.array(nn_dist))}
        for name, st in per_k.items():
            res[name] = {m: summarize(np.array(v)) for m, v in st.items()}
            diff = np.array(model_stats["cosine"]) - np.array(st["cosine"])
            res[name]["model_minus_copy_cosine"] = {**summarize(diff), "sign_flip_p": sign_flip_p(diff)}
            diffn = np.array(st["nmse"]) - np.array(model_stats["nmse"])
            res[name]["copy_minus_model_nmse"] = {**summarize(diffn), "sign_flip_p": sign_flip_p(diffn)}
        out["identities"][f"h1_level{h1}_vs_level{args.null_level}"] = res
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1) + "\n")
    brief = {k: {"model_cos": round(v["model"]["cosine"]["median"], 3), "copy_k1_cos": round(v["copy_k1"]["cosine"]["median"], 3) if "copy_k1" in v else None,
                 "model_nmse": round(v["model"]["nmse"]["median"], 3), "copy_k1_nmse": round(v["copy_k1"]["nmse"]["median"], 3) if "copy_k1" in v else None,
                 "p_model_minus_copy_cos": v.get("copy_k1", {}).get("model_minus_copy_cosine", {}).get("sign_flip_p")} for k, v in out["identities"].items()}
    print("TRAINING_COPY_BASELINE", json.dumps(brief))


if __name__ == "__main__":
    main()
