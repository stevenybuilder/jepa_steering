#!/usr/bin/env python3
"""Frozen, source-split TRAIN-only Push calibration; no planner or held rollouts."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import time

INDICES = (0, 5, 10, 15, 20, 25, 30)
CHECKPOINT_SHA = "9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""): h.update(block)
    return h.hexdigest()


def save_json(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")


def frozen_rows():
    return [{"source_id": i, "split": "fit" if i < 100 else "validation" if i < 125 else "confirmation",
             "clip_offset": 0, "observed_raw_indices": list(INDICES)} for i in range(150)]


def validate_manifest(manifest):
    if manifest.get("dataset_split") != "train" or manifest.get("rows") != frozen_rows():
        raise ValueError("Only fixed first150 TRAIN source IDs with100/25/25 split and offset0")
    if manifest.get("checkpoint_sha256") != CHECKPOINT_SHA:
        raise ValueError("Wrong frozen Push checkpoint")


def physical_q(states):
    import torch
    if states.shape[-1] != 7: raise ValueError("Expected agentXY,blockXY,angle,agentVelocityXY")
    return torch.cat([states[..., :4], states[..., 4:5].sin(), states[..., 4:5].cos()], -1)


def prepare(args):
    # Freeze the source IDs and split before opening any source tensor/video.
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    if args.dataset.name != "train": raise ValueError("Dataset must be official TRAIN directory")
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.manifest, args.output / "manifest.json")
    import torch
    torch.set_num_threads(1)
    started = time.monotonic()
    with (args.dataset / "seq_lengths.pkl").open("rb") as f: lengths = pickle.load(f)
    if len(lengths) < 150 or any(int(lengths[i]) < 31 for i in range(150)):
        raise ValueError("A predeclared source cannot provide the fixed clip; no automatic replacement")
    names = ("states.pth", "velocities.pth", "rel_actions.pth", "seq_lengths.pkl")
    source_hashes = {name: sha(args.dataset / name) for name in names}
    raw = torch.load(args.dataset / "states.pth", map_location="cpu", weights_only=True, mmap=True)
    vel = torch.load(args.dataset / "velocities.pth", map_location="cpu", weights_only=True, mmap=True)
    act = torch.load(args.dataset / "rel_actions.pth", map_location="cpu", weights_only=True, mmap=True)
    states = torch.cat([raw[:150, :31].float(), vel[:150, :31].float()], -1).contiguous()
    actions = (act[:150, :30].float() / 100.0).contiguous()
    rows = []
    videos = args.output / "videos"
    videos.mkdir()
    for row in manifest["rows"]:
        source = row["source_id"]
        video = args.dataset / "obses" / f"episode_{source:03d}.mp4"
        target = videos / video.name
        shutil.copyfile(video, target)
        rows.append({**row, "source_length": int(lengths[source]), "video": str(target.relative_to(args.output)),
                     "video_sha256": sha(target)})
    path = args.output / "train_first150.pt"
    torch.save({"states": states, "raw_actions": actions, "source_ids": torch.arange(150)}, path)
    save_json(args.output / "DONE.json", {"complete": True, "stage": "CPU fixed TRAIN clip subset only",
              "dataset_split": "train", "manifest_sha256": sha(args.manifest), "source_root": str(args.dataset),
              "source_sha256": source_hashes, "subset_path": path.name, "subset_sha256": sha(path),
              "rows": rows, "seconds": time.monotonic() - started, "model_execution": False,
              "held_scripted_rollouts_accessed": False,
              "precomputed_tokens_reused": False,
              "token_reuse_reason": "No verified frozen visual-encoder/preprocess provenance for tokens.pth"})
    print(json.dumps({"event": "train_subset_complete", "sources": 150}), flush=True)


def first_tensor(value):
    import torch
    if isinstance(value, torch.Tensor): return value
    if isinstance(value, (tuple, list)):
        for x in value:
            try: return first_tensor(x)
            except TypeError: pass
    raise TypeError("Predictor block did not return a tensor")


def capture_one(wm, frames, states, actions, block):
    import torch
    visual = torch.as_tensor(frames).permute(0, 3, 1, 2).unsqueeze(0)
    # Official Push proprio is agentXY+agent velocity, not block position.
    proprio = torch.cat([states[:, :2], states[:, 5:7]], -1).unsqueeze(0)
    captured = []
    handle = block.register_forward_hook(lambda _m, _i, out: captured.append(first_tensor(out).detach().clone()))
    try:
        with torch.inference_mode():
            encoded = wm.encode({"visual": visual, "proprio": proprio})
            normalized = wm.preprocessor.normalize_actions(actions.reshape(6, 5, 2)).reshape(6, 1, 10).to(wm.device, dtype=torch.float32)
            embedded = wm.model.encode_act(normalized)
            predicted, _, predicted_proprio = wm.model.forward_pred(
                encoded["visual"][0, :6].unsqueeze(1), embedded,
                encoded["proprio"][0, :6].unsqueeze(1))
    finally:
        handle.remove()
    if len(captured) != 1: raise RuntimeError("Expected exactly one vectorized predictor call")
    p3 = captured[0]
    if p3.shape[0] != 6: raise RuntimeError(f"Unexpected predictor layout {p3.shape}")
    pooled = p3.reshape(6, -1, p3.shape[-1]).float().mean(1)
    q = physical_q(states)
    output = {"encoded_visual_pooled": encoded["visual"].reshape(7, -1, 384).float().mean(1).cpu(),
              "predictor_p3_pooled": pooled.cpu(),
              "predicted_visual_pooled": predicted.reshape(6, -1, 384).float().mean(1).cpu(),
              "encoded_proprio": encoded["proprio"].float().cpu(),
              "predicted_proprio": predicted_proprio.float().cpu(),
              "states": states.cpu(), "proprio": proprio.cpu(), "q_current": q[:-1].cpu(), "q_next": q[1:].cpu(),
              "raw_actions": actions.cpu(), "normalized_action_chunks": normalized[:, 0].cpu()}
    if not all(torch.isfinite(x).all().item() for x in output.values()): raise RuntimeError("Nonfinite capture")
    return output, list(p3.shape)


def parameter_sha(model):
    import torch
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def capture(args):
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    receipt = json.loads((args.inputs / "DONE.json").read_text())
    if not receipt.get("complete") or receipt["manifest_sha256"] != sha(args.manifest): raise ValueError("Input manifest not verified")
    if receipt["subset_sha256"] != sha(args.inputs / receipt["subset_path"]): raise ValueError("Subset SHA mismatch before load")
    if [r["source_id"] for r in receipt["rows"]] != list(range(150)): raise ValueError("Wrong source IDs")
    if sha(args.checkpoint) != CHECKPOINT_SHA: raise ValueError("Checkpoint SHA mismatch")
    for row in receipt["rows"]:
        if sha(args.inputs / row["video"]) != row["video_sha256"]: raise ValueError("Video SHA mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.manifest, args.output / "manifest.json")
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from model_loader import load_headless
    torch.set_num_threads(2)
    torch.manual_seed(90505)
    started = time.monotonic()
    wm, _, provenance = load_headless(args.repo, model_name="jepa_wm_pusht", checkpoint_override=args.checkpoint)
    predictor = wm.model.predictor
    block_attribute = "predictor_blocks" if hasattr(predictor, "predictor_blocks") else "blocks"
    blocks = getattr(predictor, block_attribute)
    block = blocks[3]
    before = parameter_sha(wm)
    bank = torch.load(args.inputs / receipt["subset_path"], map_location="cpu", weights_only=True)
    outputs = []
    for row in receipt["rows"]:
        source = row["source_id"]
        with imageio.get_reader(args.inputs / row["video"], format="ffmpeg") as video:
            frames = np.stack([video.get_data(i) for i in INDICES])
        state = bank["states"][source, list(INDICES)]
        action = bank["raw_actions"][source]
        result, shape = capture_one(wm, frames, state, action, block)
        if source == 0:
            repeated, _ = capture_one(wm, frames, state, action, block)
            if not all(torch.equal(result[k], repeated[k]) for k in result): raise RuntimeError("Exact repeated capture failed")
        result["meta"] = {**row, "checkpoint_sha256": CHECKPOINT_SHA, "input_done_sha256": sha(args.inputs / "DONE.json"),
                          "manifest_sha256": sha(args.manifest), "predictor_module": "model.predictor." + block_attribute + ".3",
                          "predictor_raw_shape": shape, "xy_units": "native simulator pixels (512 arena)",
                          "angle_targets": "sin(theta),cos(theta)", "prediction_semantics": "latent prediction, not native physical XYZ output"}
        path = args.output / f"source-{source:03d}.pt"
        torch.save(result, path)
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        if not all(torch.equal(result[k], loaded[k]) for k in result if k != "meta"): raise RuntimeError("Save/reload mismatch")
        outputs.append({"source_id": source, "split": row["split"], "path": path.name, "sha256": sha(path), "bytes": path.stat().st_size})
        print(json.dumps({"event": "calibration_source_complete", "source_id": source, "split": row["split"]}), flush=True)
    if parameter_sha(wm) != before: raise RuntimeError("Frozen parameters changed")
    save_json(args.output / "DONE.json", {"complete": True, "sources": 150, "observed_states": 1050, "transitions": 900,
              "outputs": outputs, "split_counts": {"fit": 100, "validation": 25, "confirmation": 25},
              "input_done_sha256": sha(args.inputs / "DONE.json"), "manifest_sha256": sha(args.manifest),
              "script_sha256": sha(__file__), "loader_sha256": sha(Path(__file__).with_name("model_loader.py")),
              "checkpoint_sha256": CHECKPOINT_SHA, "native_config_sha256": sha(provenance["config"]),
              "provenance": provenance, "seconds": time.monotonic() - started, "parameter_sha256": before,
              "no_cem_or_simulator_rollout": True, "held_scripted_outcomes_accessed": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "capture"])
    for name in ("manifest", "output", "dataset", "inputs", "repo", "checkpoint"):
        parser.add_argument("--" + name, type=Path, required=name in ("manifest", "output"))
    args = parser.parse_args()
    (prepare if args.mode == "prepare" else capture)(args)


if __name__ == "__main__": main()
