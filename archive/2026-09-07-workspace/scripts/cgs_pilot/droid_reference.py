#!/usr/bin/env python3
"""Download and cache a small public DROID reference subset (``droid_100``).

``gs://gresearch/robotics/droid_100`` is the 100-episode RLDS sample released
with DROID (Khazatsky et al., 2024, arXiv:2403.12945). It is public over HTTPS
(``https://storage.googleapis.com/gresearch/robotics/droid_100/1.0.0/``), ~2 GB
in 31 TFRecord shards. The JEPA-WM DROID checkpoint was trained on DROID only
(vendored config ``datasets: [DROID]``; arXiv:2512.24497 reports no RoboCasa
finetuning), so this subset is the right *domain* reference for the RoboCasa
stimuli: not a contamination check, a domain-shift check.

No TensorFlow on the pilot host, so the shards are parsed with a minimal
hand-written TFRecord + ``tf.train.Example`` protobuf wire reader. Each shard is
streamed, its frames/actions extracted, and the shard deleted before the next
one is fetched, so peak disk use stays ~200 MB (the largest shard).

Cache layout ``<cache_dir>/droid_100_reference.npz``:

- ``frames``   uint8 [N, 256, 256, 3]: ``exterior_image_2_left`` (the camera view
  JEPA-WM trains on for DROID) resized from 180x320 with the same bilinear resize
  the stimulus pipeline uses; subsampled to at most ``--max-frames``.
- ``frame_episode`` int32 [N], ``frame_step`` int32 [N].
- ``action_chunks`` float32 [M, 3, 7]: 3-step chunks of JEPA-WM-style DROID
  actions: the per-step delta of the RLDS ``action`` (target cartesian pose 6-D +
  gripper) across ``--action-stride`` source steps (15 Hz / 4 = 3.75 Hz, the
  released ``fps: 4``), with the angle wrap used by
  ``app/plan_common/datasets/droid_dset.py::_get_delta``; ``chunk_episode`` int32 [M].
- ``trans_frame_t`` / ``trans_frame_t1`` uint8 [K, 256, 256, 3] and
  ``trans_actions`` float32 [K, 3, 7]: paired transitions (frame before the chunk,
  frame after it) for the retrieval baseline; ``trans_episode`` int32 [K].
- ``meta`` json string with provenance (shard list, counts, subsampling).
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import struct
import sys
import urllib.request
from pathlib import Path
from typing import Any, Iterator

import numpy as np

DROID100_HTTPS = "https://storage.googleapis.com/gresearch/robotics/droid_100/1.0.0/"
DROID100_GS = "gs://gresearch/robotics/droid_100/1.0.0/"
SHARD_TEMPLATE = "r2d2_faceblur-train.tfrecord-{i:05d}-of-00031"
N_SHARDS = 31
IMAGE_KEY = "steps/observation/exterior_image_2_left"
ACTION_KEY = "steps/action"
ACTION_DIM = 7


# --------------------------------------------------------------------------- #
# Minimal protobuf wire-format reader for tf.train.Example
# --------------------------------------------------------------------------- #


def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7


def _iter_fields(buf: bytes) -> Iterator[tuple[int, int, bytes | int]]:
    """Yield (field_number, wire_type, payload) for one message."""

    pos = 0
    n = len(buf)
    while pos < n:
        tag, pos = _read_varint(buf, pos)
        field, wt = tag >> 3, tag & 7
        if wt == 0:
            val, pos = _read_varint(buf, pos)
            yield field, wt, val
        elif wt == 1:
            yield field, wt, buf[pos : pos + 8]
            pos += 8
        elif wt == 2:
            length, pos = _read_varint(buf, pos)
            yield field, wt, buf[pos : pos + length]
            pos += length
        elif wt == 5:
            yield field, wt, buf[pos : pos + 4]
            pos += 4
        else:  # pragma: no cover
            raise ValueError(f"unsupported wire type {wt}")


def _parse_feature(buf: bytes) -> tuple[str, Any]:
    """Feature{bytes_list=1, float_list=2, int64_list=3}; each list has repeated value=1."""

    for field, _, payload in _iter_fields(buf):
        if field == 1:
            return "bytes", [p for f, w, p in _iter_fields(payload) if f == 1]
        if field == 2:
            vals: list[float] = []
            for f, w, p in _iter_fields(payload):
                if f != 1:
                    continue
                if w == 2:  # packed
                    vals.extend(struct.unpack(f"<{len(p) // 4}f", p))
                else:
                    vals.append(struct.unpack("<f", p)[0])
            return "float", vals
        if field == 3:
            ivals: list[int] = []
            for f, w, p in _iter_fields(payload):
                if f != 1:
                    continue
                if w == 2:
                    pos = 0
                    while pos < len(p):
                        v, pos = _read_varint(p, pos)
                        ivals.append(v)
                else:
                    ivals.append(p)
            return "int64", ivals
    return "empty", []


def parse_example(buf: bytes, wanted: set[str] | None = None) -> dict[str, tuple[str, Any]]:
    """Parse tf.train.Example bytes into {key: (kind, values)}."""

    out: dict[str, tuple[str, Any]] = {}
    for field, _, features in _iter_fields(buf):  # Example.features = 1
        if field != 1:
            continue
        for f2, _, entry in _iter_fields(features):  # Features.feature = 1 (map entry)
            if f2 != 1:
                continue
            key, value = None, None
            for f3, _, p in _iter_fields(entry):
                if f3 == 1:
                    key = p.decode("utf-8")
                elif f3 == 2:
                    value = p
            if key is None or value is None:
                continue
            if wanted is not None and key not in wanted:
                continue
            out[key] = _parse_feature(value)
    return out


def iter_tfrecords(path: Path) -> Iterator[bytes]:
    """TFRecord: uint64 length, uint32 masked crc(length), data, uint32 masked crc(data)."""

    with path.open("rb") as fh:
        while True:
            header = fh.read(12)
            if len(header) < 12:
                return
            (length,) = struct.unpack("<Q", header[:8])
            data = fh.read(length)
            fh.read(4)
            if len(data) < length:
                return
            yield data


# --------------------------------------------------------------------------- #
# Download + extract
# --------------------------------------------------------------------------- #


def download(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url, timeout=120) as resp, dest.open("wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)


def decode_resize(jpeg: bytes, size: int) -> np.ndarray:
    from PIL import Image

    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    return np.asarray(img.resize((size, size), Image.BILINEAR), dtype=np.uint8)


def jepa_wm_delta_actions(raw: np.ndarray, stride: int) -> np.ndarray:
    """JEPA-WM DROID action: delta of RLDS ``action`` across ``stride`` steps, angles wrapped."""

    sub = raw[::stride]
    delta = sub[1:] - sub[:-1]
    rot = delta[:, 3:6]
    rot[rot > np.pi] -= 2 * np.pi
    rot[rot < -np.pi] += 2 * np.pi
    delta[:, 3:6] = rot
    return delta.astype(np.float32)


def extract_shard(
    shard: Path,
    frame_stride: int,
    action_stride: int,
    chunk_len: int,
    image_size: int,
    episode_offset: int,
    transitions_per_episode: int,
    rng: np.random.RandomState,
) -> dict[str, list]:
    out: dict[str, list] = {k: [] for k in ("frames", "frame_ids", "chunks", "chunk_eps", "tf_t", "tf_t1", "t_actions", "t_eps")}
    ep = episode_offset
    for rec in iter_tfrecords(shard):
        ex = parse_example(rec, {IMAGE_KEY, ACTION_KEY})
        _, images = ex.get(IMAGE_KEY, ("empty", []))
        kind_a, acts = ex.get(ACTION_KEY, ("empty", []))
        if kind_a == "float" and len(acts) % ACTION_DIM == 0 and len(images) >= (chunk_len + 1) * action_stride:
            raw = np.asarray(acts, dtype=np.float64).reshape(-1, ACTION_DIM)
            delta = jepa_wm_delta_actions(raw, action_stride)  # delta[i] spans source steps i*stride -> (i+1)*stride
            n_chunks = len(delta) - chunk_len + 1
            for s in range(n_chunks):
                out["chunks"].append(delta[s : s + chunk_len])
                out["chunk_eps"].append(ep)
            if transitions_per_episode > 0 and n_chunks > 0:
                for s in rng.choice(n_chunks, min(transitions_per_episode, n_chunks), replace=False):
                    t0, t1 = int(s * action_stride), int((s + chunk_len) * action_stride)
                    if t1 < len(images):
                        out["tf_t"].append(decode_resize(images[t0], image_size))
                        out["tf_t1"].append(decode_resize(images[t1], image_size))
                        out["t_actions"].append(delta[s : s + chunk_len])
                        out["t_eps"].append(ep)
        for t in range(0, len(images), max(1, frame_stride)):
            out["frames"].append(decode_resize(images[t], image_size))
            out["frame_ids"].append((ep, t))
        ep += 1
    out["next_episode"] = ep
    return out


def build_reference(
    cache_dir: Path,
    max_frames: int,
    frame_stride: int,
    action_stride: int,
    chunk_len: int,
    image_size: int,
    shards: list[int],
    keep_shards: bool,
    seed: int,
    transitions_per_episode: int = 6,
    max_transitions: int = 600,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_dir / "_shards"
    tmp.mkdir(exist_ok=True)
    rng = np.random.RandomState(seed)
    acc: dict[str, list] = {k: [] for k in ("frames", "frame_ids", "chunks", "chunk_eps", "tf_t", "tf_t1", "t_actions", "t_eps")}
    ep = 0
    used = []
    for i in shards:
        name = SHARD_TEMPLATE.format(i=i)
        dest = tmp / name
        if not dest.exists():
            free = shutil.disk_usage(cache_dir).free
            if free < 600 * (1 << 20):
                raise RuntimeError(f"refusing to download {name}: only {free / 1e9:.2f} GB free")
            print(f"downloading {name}", flush=True)
            download(DROID100_HTTPS + name, dest)
        part = extract_shard(dest, frame_stride, action_stride, chunk_len, image_size, ep, transitions_per_episode, rng)
        ep = part.pop("next_episode")
        for k, v in part.items():
            acc[k].extend(v)
        used.append(name)
        print(f"{name}: episodes so far {ep}, frames {len(acc['frames'])}, chunks {len(acc['chunks'])}, transitions {len(acc['tf_t'])}", flush=True)
        if not keep_shards:
            dest.unlink()
    if not acc["frames"]:
        raise RuntimeError("no frames extracted")
    frames = np.stack(acc["frames"])
    ids = np.asarray(acc["frame_ids"], dtype=np.int32)
    if len(frames) > max_frames:
        keep = np.sort(rng.choice(len(frames), max_frames, replace=False))
        frames, ids = frames[keep], ids[keep]
    tf_t = np.stack(acc["tf_t"]) if acc["tf_t"] else np.zeros((0, image_size, image_size, 3), np.uint8)
    tf_t1 = np.stack(acc["tf_t1"]) if acc["tf_t1"] else np.zeros((0, image_size, image_size, 3), np.uint8)
    t_actions = np.stack(acc["t_actions"]).astype(np.float32) if acc["t_actions"] else np.zeros((0, chunk_len, ACTION_DIM), np.float32)
    t_eps = np.asarray(acc["t_eps"], dtype=np.int32)
    if len(tf_t) > max_transitions:
        keep = np.sort(rng.choice(len(tf_t), max_transitions, replace=False))
        tf_t, tf_t1, t_actions, t_eps = tf_t[keep], tf_t1[keep], t_actions[keep], t_eps[keep]
    out = cache_dir / "droid_100_reference.npz"
    meta = {
        "source": DROID100_GS,
        "https": DROID100_HTTPS,
        "shards": used,
        "episodes": int(ep),
        "image_key": IMAGE_KEY,
        "action_key": ACTION_KEY,
        "frame_stride": frame_stride,
        "action_stride": action_stride,
        "chunk_len": chunk_len,
        "image_size": image_size,
        "n_frames": int(len(frames)),
        "n_action_chunks": int(len(acc["chunks"])),
        "n_transitions": int(len(tf_t)),
        "subsample_seed": seed,
        "action_semantics": (
            "JEPA-WM DROID convention: per-step delta of RLDS 'action' (target cartesian pose 6-D + gripper) "
            "across action_stride source steps with angle wrap (droid_dset._get_delta)."
        ),
    }
    np.savez_compressed(
        out,
        frames=frames,
        frame_episode=ids[:, 0],
        frame_step=ids[:, 1],
        action_chunks=np.stack(acc["chunks"]).astype(np.float32),
        chunk_episode=np.asarray(acc["chunk_eps"], dtype=np.int32),
        trans_frame_t=tf_t,
        trans_frame_t1=tf_t1,
        trans_actions=t_actions,
        trans_episode=t_eps,
        meta=json.dumps(meta),
    )
    if not keep_shards:
        shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps(meta, indent=2))
    return out


def load_reference(path: Path) -> dict[str, Any]:
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d["meta"]))
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=Path("/root/cgs-pilot/reference/droid_100"))
    ap.add_argument("--max-frames", type=int, default=5000)
    ap.add_argument("--frame-stride", type=int, default=15, help="15 Hz source -> ~1 frame/s")
    ap.add_argument("--action-stride", type=int, default=4, help="15 Hz source; JEPA-WM DROID uses fps 4")
    ap.add_argument("--chunk-len", type=int, default=3)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--shards", type=int, nargs="*", default=list(range(N_SHARDS)))
    ap.add_argument("--keep-shards", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--transitions-per-episode", type=int, default=6)
    ap.add_argument("--max-transitions", type=int, default=600)
    args = ap.parse_args()
    build_reference(
        args.cache_dir, args.max_frames, args.frame_stride, args.action_stride, args.chunk_len, args.image_size,
        args.shards, args.keep_shards, args.seed, args.transitions_per_episode, args.max_transitions,
    )


if __name__ == "__main__":
    sys.exit(main())
