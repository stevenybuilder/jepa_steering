#!/usr/bin/env python3
"""Materialise per-arm training shards for ``precompute_dinov3_features.py`` from the MetaDrive train output.

Input (``metadrive_hazard_pilot.py train --out ROOT``):
    ROOT/armA/shard_*.npz + .jsonl   hazard clips rendered in arm A (pedestrian solid / cone ghost)
    ROOT/armB/shard_*.npz + .jsonl   the same seeds in arm B (pedestrian ghost / cone solid)
    ROOT/shared/shard_*.npz + .jsonl hazard-free clips, rendered once (identical in both arms by construction)

Output (``--out DIR``):
    DIR/armA/shard_0000.npz  frames [n, T, 256, 256, 3] uint8 RGB, actions [n, T-1, 2] float32
    DIR/armA/shard_0000.meta.json  list of dicts: clip_id, seed, arm, hazard (bool), hazard_kind, pose, contact (bool),
                                   contact_step, min_distance, plus source provenance
    DIR/armB/...                   same; the shared hazard-free clips are written into BOTH arms' shard sets with
                                   identical content and identical clip ids (their ``arm`` field is the target arm,
                                   ``source_arm`` = "shared")
    DIR/shards.meta.json           counts, sizes, sha256 of every shard, source root

Clips are ordered by (seed, hazard_kind) so a shard's membership is deterministic; <= ``--clips-per-shard`` (100)
clips per shard. ``--delete-source-frames`` (off) removes the source ``shard_*.npz`` files after every output shard
has been written and verified by re-reading (the source .jsonl metadata is always kept).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from protocol import sha256_file  # noqa: E402

ARMS = ("A", "B")


def load_source(root: Path, name: str) -> list[dict[str, Any]]:
    """[{meta, path, index}] for every clip of ROOT/<name>, sorted by (seed, kind)."""
    items = []
    for p in sorted((root / name).glob("shard_*.npz")):
        jl = p.with_suffix(".jsonl")
        metas = [json.loads(l) for l in jl.read_text().splitlines() if l.strip()]
        for i, m in enumerate(metas):
            items.append({"meta": m, "path": p, "index": i})
    items.sort(key=lambda it: (int(it["meta"]["seed"]), str(it["meta"].get("hazard_kind") or "")))
    return items


def clip_meta(m: dict[str, Any], arm: str, source: str, path: Path, index: int) -> dict[str, Any]:
    return {
        "clip_id": m["clip_id"], "seed": int(m["seed"]), "arm": arm, "hazard": bool(m["has_hazard"]),
        "hazard_kind": m.get("hazard_kind"), "pose": m.get("hazard_pose"), "hazard_level": m.get("hazard_level"),
        "hazard_dist_m": m.get("hazard_dist_m"), "solid": (m.get("solid") if source != "shared" else None),
        "contact": bool(m.get("contact", False)), "contact_step": m.get("first_contact_step"),
        "contact_model_step": m.get("first_contact_model_step"), "min_distance": m.get("min_gap_m"),
        "min_center_distance_m": m.get("min_center_distance_m"), "v0_mps": m.get("v0_mps"), "map_cfg": m.get("map_cfg"),
        "action_classes": m.get("action_classes"), "frames_sha256": m.get("frames_sha256"), "kind_paired": m.get("kind_paired"),
        "source_arm": source, "source_shard": str(path), "source_index": int(index),
    }


def write_arm(root: Path, out: Path, arm: str, per_shard: int) -> dict[str, Any]:
    items = [(it, f"arm{arm}") for it in load_source(root, f"arm{arm}")] + [(it, "shared") for it in load_source(root, "shared")]
    items.sort(key=lambda x: (int(x[0]["meta"]["seed"]), str(x[0]["meta"].get("hazard_kind") or "")))
    adir = out / f"arm{arm}"
    adir.mkdir(parents=True, exist_ok=True)
    cache: dict[Path, Any] = {}

    def arrays(path: Path):
        if path not in cache:
            cache.clear()
            cache[path] = np.load(path)
        return cache[path]

    shards = []
    for k in range(0, len(items), per_shard):
        group = items[k : k + per_shard]
        frames = np.stack([arrays(it["path"])["frames"][it["index"]] for it, _ in group])
        actions = np.stack([arrays(it["path"])["actions"][it["index"]] for it, _ in group]).astype(np.float32)
        metas = [clip_meta(it["meta"], arm, src, it["path"], it["index"]) for it, src in group]
        stem = adir / f"shard_{k // per_shard:04d}"
        np.savez(stem.with_suffix(".npz"), frames=frames, actions=actions)  # uncompressed: precompute reads members by mmap-ish zip access
        Path(f"{stem}.meta.json").write_text(json.dumps(metas, indent=1) + "\n")
        # verify by re-reading
        z = np.load(stem.with_suffix(".npz"))
        assert z["frames"].shape == frames.shape and z["frames"].dtype == np.uint8 and z["actions"].shape == actions.shape
        assert len(json.loads(Path(f"{stem}.meta.json").read_text())) == frames.shape[0]
        shards.append({"path": str(stem.with_suffix(".npz")), "n": int(frames.shape[0]), "bytes": stem.with_suffix(".npz").stat().st_size,
                       "sha256": sha256_file(stem.with_suffix(".npz")), "n_hazard": sum(m["hazard"] for m in metas),
                       "n_contact": sum(m["contact"] for m in metas), "n_shared": sum(m["source_arm"] == "shared" for m in metas)})
    return {"arm": arm, "n_clips": len(items), "n_hazard": sum(bool(it["meta"]["has_hazard"]) for it, _ in items),
            "n_contact": sum(bool(it["meta"].get("contact")) for it, _ in items), "n_shared": sum(src == "shared" for _, src in items),
            "shards": shards, "bytes": sum(s["bytes"] for s in shards)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="metadrive_hazard_pilot.py train output dir")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--clips-per-shard", type=int, default=100)
    ap.add_argument("--compress", action="store_true", help="np.savez_compressed instead of np.savez (smaller, slower to read)")
    ap.add_argument("--delete-source-frames", action="store_true", help="remove ROOT/{armA,armB,shared}/shard_*.npz after verified writes")
    args = ap.parse_args()
    if args.clips_per_shard > 100:
        raise SystemExit("--clips-per-shard must be <= 100")
    if args.compress:
        np.savez = np.savez_compressed  # type: ignore[assignment]
    t0 = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    result = {"source_root": str(args.root.resolve()), "clips_per_shard": args.clips_per_shard, "arms": {}}
    for arm in ARMS:
        result["arms"][arm] = write_arm(args.root, args.out, arm, args.clips_per_shard)
        print(json.dumps({k: v for k, v in result["arms"][arm].items() if k != "shards"}), flush=True)
    # sanity: shared clips identical across arms (same clip ids, same frames sha)
    a_shared = {m["clip_id"]: m["frames_sha256"] for p in sorted((args.out / "armA").glob("shard_*.meta.json")) for m in json.loads(p.read_text()) if m["source_arm"] == "shared"}
    b_shared = {m["clip_id"]: m["frames_sha256"] for p in sorted((args.out / "armB").glob("shard_*.meta.json")) for m in json.loads(p.read_text()) if m["source_arm"] == "shared"}
    result["shared_identical_across_arms"] = a_shared == b_shared
    a_ids = {m["clip_id"] for p in (args.out / "armA").glob("shard_*.meta.json") for m in json.loads(p.read_text())}
    b_ids = {m["clip_id"] for p in (args.out / "armB").glob("shard_*.meta.json") for m in json.loads(p.read_text())}
    result["clip_ids_equal_across_arms"] = a_ids == b_ids
    result["elapsed_s"] = time.time() - t0
    result["total_bytes"] = sum(v["bytes"] for v in result["arms"].values())
    n = result["arms"]["A"]["n_clips"]
    result["mb_per_1k_clips_per_arm"] = result["arms"]["A"]["bytes"] / max(1, n) * 1000 / 1e6
    if args.delete_source_frames:
        removed = []
        for name in ("armA", "armB", "shared"):
            for p in sorted((args.root / name).glob("shard_*.npz")):
                p.unlink()
                removed.append(str(p))
        result["deleted_source_frames"] = removed
    (args.out / "shards.meta.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "arms"}))


if __name__ == "__main__":
    main()
