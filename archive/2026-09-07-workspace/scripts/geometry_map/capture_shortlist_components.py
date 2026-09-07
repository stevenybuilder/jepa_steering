#!/usr/bin/env python3
"""Frozen-model, discovery-only attention/MLP and embedding capture.

The first manifest episode is the integrity canary and supplies the only full
spatial frame. Attention/MLP module outputs precede residual addition, including
any outer layer-scale/drop-path. No head importance or causal effect is measured.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

SHORTLIST = {"encoder": [6], "predictor": [2, 3, 5]}
COMPONENTS = {"attention_output": "attn", "mlp_output": "mlp"}
FRAME_INDICES = tuple(range(0, 100, 5))


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Immutable output exists: {path}")
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def selected_manifest(rows, parquet_name):
    selected = [row for row in rows if row["parquet"] == parquet_name and row["task"] == "mw-reach-wall"
                and row["split"] == "discovery" and 0 <= int(row["episode"]) < 50]
    selected.sort(key=lambda row: (int(row["seed"]), int(row["episode"])))
    keys = [(int(row["seed"]), int(row["episode"])) for row in selected]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("Empty or duplicate discovery manifest")
    for seed in {seed for seed, _ in keys}:
        if {ep for s, ep in keys if s == seed} != set(range(50)):
            raise ValueError("Require predeclared complete discovery episode set 0..49 per available seed")
    return selected


def tensor_output(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for entry in value:
            try:
                return tensor_output(entry)
            except TypeError:
                pass
    raise TypeError("No tensor in component output")


def component_modules(blocks):
    modules = {}
    for family, indices in SHORTLIST.items():
        for index in indices:
            block = blocks[family][index]
            for name, attribute in COMPONENTS.items():
                module = getattr(block, attribute, None)
                if not isinstance(module, torch.nn.Module):
                    raise ValueError(f"Missing explicit {family}.{index}.{attribute} hook module")
                modules[f"{family}.{index}.{name}"] = module
    return modules


def verify_predictor_contract(model):
    if getattr(model, "pred_type", None) != "AdaLN" or getattr(model.predictor, "proprio_encoding", None) != "feature":
        raise ValueError("Capture requires the released AdaLN predictor with proprio feature conditioning")


def pool_component(value, family):
    if value.ndim != 3:
        raise ValueError(f"Expected component [samples,tokens,channels], got {tuple(value.shape)}")
    expected_samples, expected_tokens, expected_channels = (20, 257, 384) if family == "encoder" else (19, 256, 400)
    if tuple(value.shape) != (expected_samples, expected_tokens, expected_channels):
        raise ValueError(f"Unexpected {family} token semantics: {tuple(value.shape)}")
    spatial = value[:, 1:] if family == "encoder" else value
    return spatial.float().mean(dim=1).half().cpu(), spatial


def assert_finite_tensors(tree):
    for key, value in tree.items():
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError(f"Non-finite or non-tensor capture {key}")


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        digest.update(name.encode())
        data = value.detach().cpu().contiguous()
        digest.update(str((tuple(data.shape), data.dtype)).encode())
        digest.update(data.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def capture_episode(wm, row, modules, retain_spatial):
    from canary_capture import decode_selected
    frames = decode_selected(row["video"], FRAME_INDICES)
    states = np.asarray(row["states"], dtype=np.float32)
    actions = np.asarray(row["actions"], dtype=np.float32)
    visual = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0)
    proprio = torch.from_numpy(states[list(FRAME_INDICES), :4]).unsqueeze(0)
    captured, counts = {}, {name: 0 for name in modules}
    internal_embeddings = {}
    handles = []

    def record(name):
        def hook(_module, _inputs, output):
            counts[name] += 1
            captured[name] = tensor_output(output).detach().clone()
        return hook

    for name, module in modules.items():
        handles.append(module.register_forward_hook(record(name)))
    if wm.model.predictor.action_encoder_inpred:
        def record_internal_action(_module, _inputs, output):
            if "action_embedding" in internal_embeddings:
                raise ValueError("Action embedding encoder unexpectedly called twice")
            internal_embeddings["action_embedding"] = tensor_output(output).detach().clone()
        handles.append(wm.model.predictor.action_encoder.register_forward_hook(record_internal_action))
    try:
        encoded = wm.encode({"visual": visual, "proprio": proprio})
        chunks = torch.from_numpy(actions[:95]).reshape(19, 5, 4)
        normalized = wm.preprocessor.normalize_actions(chunks).reshape(19, 1, 20).to(wm.device, dtype=torch.float32)
        action_embedding = wm.model.encode_act(normalized)
        predicted_visual, _, predicted_proprio = wm.model.forward_pred(
            encoded["visual"][0, :19].unsqueeze(1), action_embedding,
            encoded["proprio"][0, :19].unsqueeze(1))
    finally:
        for handle in handles:
            handle.remove()
    if any(count != 1 for count in counts.values()):
        raise ValueError(f"Unexpected repeated/missing component hook calls: {counts}")
    pooled, spatial, component_shapes = {}, {}, {}
    for name, value in captured.items():
        family = name.split(".")[0]
        pooled[name], patches = pool_component(value, family)
        component_shapes[name] = list(value.shape)
        if retain_spatial:
            spatial[name] = patches[0].half().cpu()
    embeddings = {
        "action_embedding": internal_embeddings.get("action_embedding", action_embedding).detach().half().cpu(),
        "action_conditioning_input": action_embedding.detach().half().cpu(),
        "encoded_proprio": encoded["proprio"].detach().half().cpu(),
        "predicted_proprio": predicted_proprio.detach().half().cpu(),
        "encoded_visual_pooled": encoded["visual"].reshape(20, -1, 384).float().mean(dim=1).half().cpu(),
        "predicted_visual_pooled": predicted_visual.reshape(19, -1, 384).float().mean(dim=1).half().cpu(),
    }
    if retain_spatial:
        spatial["encoded_visual"] = encoded["visual"].reshape(20, 256, 384)[0].half().cpu()
        spatial["predicted_visual"] = predicted_visual.reshape(19, 256, 384)[0].half().cpu()
    raw = {"frame_index": torch.tensor(FRAME_INDICES), "transition_index": torch.arange(19),
           "raw_action_chunks": chunks.cpu(), "normalized_action_chunks": normalized.cpu(),
           "raw_proprio": proprio.cpu()}
    for values in (pooled, embeddings, spatial, raw):
        assert_finite_tensors(values)
    return {"pooled_components": pooled, "embeddings": embeddings, "representative_spatial": spatial,
            "alignment": raw, "component_shapes": component_shapes, "hook_call_counts": counts}, frames[0]


def equal_capture(a, b):
    for group in ("pooled_components", "embeddings", "representative_spatial", "alignment"):
        if a[group].keys() != b[group].keys() or not all(torch.equal(a[group][k], b[group][k]) for k in a[group]):
            return False
    return True


def file_receipt(path, **meta):
    return {"path": path.name, "sha256": file_sha(path), "bytes": path.stat().st_size, **meta}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    from canary_capture import block_list
    from model_loader import load_headless_metaworld
    import pyarrow.parquet as pq
    from PIL import Image

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite nonempty immutable capture root")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    selected = selected_manifest(source_rows, args.parquet.name)
    expected_source = {row["parquet_sha256"] for row in selected}
    actual_source = file_sha(args.parquet)
    if expected_source != {actual_source}:
        raise ValueError("Original parquet differs from frozen manifest SHA")
    semantics = {
        "phase": "offline_teacher_forced_one_step", "weights": "frozen/eval/no edits",
        "encoder_sample_index": "0..19: raw observed frame 5*i",
        "predictor_sample_index": "0..18: raw observed frame 5*i with recorded raw actions [5*i:5*i+5]",
        "imagined_timestep": 1, "context_length": 1,
        "candidate_identity": "The single recorded TD-MPC2 action chunk per sample; no candidate search or CEM ranking in this capture",
        "encoder_tokens": "token0 CLS excluded from pooling; token1..256 row-major16x16 spatial patches",
        "predictor_tokens": "256 row-major16x16 spatial positions;384 visual plus16 proprio channels. Actions condition AdaLN modulation/gates, not extra spatial tokens",
        "components": "Raw attention-module or MLP-module output before outer AdaLN output gates, layer-scale, drop-path and residual addition; not per-head attribution",
        "action_embedding": "Actual action-encoder output, including an internal predictor.action_encoder hook when action_encoder_inpred is true; action_conditioning_input is separately retained",
        "pooled_components": "Mean over256 spatial tokens at every sampled observation/one-step prediction",
        "full_spatial_selection": {"seed": selected[0]["seed"], "episode": selected[0]["episode"], "raw_frame": 0, "transition": 0},
        "figure_use": "Authentic source-frame background for population fixed-grid CV scores; not single-frame attribution",
        "missing": ["causal component importance", "multiple imagined rollout steps", "CEM costs/ranks for these TD-MPC2 actions"]}
    frozen = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "shortlist": SHORTLIST, "components": list(COMPONENTS), "selected_rows": selected,
              "selection_rule": "All discovery IDs0..49 available in the one pre-existing official parquet; no outcome screening",
              "source_manifest_sha256": file_sha(args.manifest), "parquet_sha256": actual_source,
              "script_sha256": file_sha(Path(__file__)), "semantics": semantics}
    manifest_path = args.output_dir / "capture_manifest.json"
    save_json(manifest_path, frozen)
    print(json.dumps({"event": "manifest_frozen", "pid": os.getpid(), "episodes": len(selected),
                      "seeds": sorted({row["seed"] for row in selected}), "manifest_sha256": file_sha(manifest_path)}), flush=True)
    # Predicate filters are applied before converting any video/state/action row to Python.
    seeds = sorted({int(row["seed"]) for row in selected})
    table = pq.read_table(args.parquet, columns=["task", "seed", "episode", "video", "states", "actions"],
                          filters=[("task", "=", "mw-reach-wall"), ("seed", "in", seeds), ("episode", ">=", 0), ("episode", "<", 50)])
    lookup = {(int(table["seed"][index].as_py()), int(table["episode"][index].as_py())): index for index in range(len(table))}
    if set(lookup) != {(int(row["seed"]), int(row["episode"])) for row in selected}:
        raise ValueError("Filtered original table does not match frozen discovery manifest")
    wm, _preprocessor, provenance = load_headless_metaworld(args.repo, args.device)
    wm.eval()
    wm.requires_grad_(False)
    verify_predictor_contract(wm.model)
    encoder_path, encoders = block_list(wm.model.encoder, ("blocks", "base_model.blocks", "trunk.blocks", "backbone.blocks", "model.blocks"), 12, "encoder")
    predictor_path, predictors = block_list(wm.model.predictor, ("predictor_blocks", "blocks"), 6, "predictor")
    modules = component_modules({"encoder": encoders, "predictor": predictors})
    module_provenance = {}
    for name, module in modules.items():
        source = Path(inspect.getfile(type(module)))
        module_provenance[name] = {"class": f"{type(module).__module__}.{type(module).__qualname__}",
                                   "source_path": str(source), "source_sha256": file_sha(source)}
    parameter_before = parameter_hash(wm)
    model_receipt = {**provenance, "checkpoint_sha256": file_sha(Path(provenance["checkpoint"])),
                     "config_sha256": file_sha(Path(provenance["config"])),
                     "parameter_sha256_before": parameter_before, "all_requires_grad_false": not any(p.requires_grad for p in wm.parameters()),
                     "encoder_block_path": encoder_path, "predictor_block_path": predictor_path,
                     "hook_modules": module_provenance,
                     "predictor_class": type(wm.model.predictor).__qualname__,
                     "predictor_source_sha256": file_sha(Path(inspect.getfile(type(wm.model.predictor)))),
                     "action_encoder_inpred": wm.model.predictor.action_encoder_inpred,
                     "helper_source_sha256": {name: file_sha(Path(__file__).parent / name) for name in ("canary_capture.py", "model_loader.py")}}
    outputs = [file_receipt(manifest_path)]
    started = time.monotonic()
    for position, manifest_row in enumerate(selected):
        key = int(manifest_row["seed"]), int(manifest_row["episode"])
        index = lookup[key]
        row = {name: table[name][index].as_py() for name in table.column_names}
        capture, frame = capture_episode(wm, row, modules, retain_spatial=position == 0)
        if position == 0:
            repeated, _ = capture_episode(wm, row, modules, retain_spatial=True)
            if not equal_capture(capture, repeated):
                raise ValueError("One-episode canary failed exact repeated-capture equality")
            del repeated
        capture["meta"] = {**manifest_row, "model": "jepa_wm_metaworld", "capture_manifest_sha256": file_sha(manifest_path),
                            "schema_version": 1, "semantics": semantics, "model_provenance": model_receipt}
        path = args.output_dir / f"seed-{key[0]}-episode-{key[1]:03d}.pt"
        torch.save(capture, path.with_suffix(".pt.tmp"))
        path.with_suffix(".pt.tmp").replace(path)
        receipt = file_receipt(path, seed=key[0], episode=key[1])
        # Verify serializability and required tensor groups before continuing.
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        if not equal_capture(capture, loaded):
            raise ValueError("Saved capture differs from in-memory tensor data")
        outputs.append(receipt)
        if position == 0:
            image_path = args.output_dir / "representative_frame.png"
            Image.fromarray(frame).save(image_path)
            image_receipt = file_receipt(image_path, seed=key[0], episode=key[1], raw_frame_index=0)
            image_provenance = {**image_receipt, "parquet": str(args.parquet), "parquet_sha256": actual_source,
                                "original_parquet_row": int(manifest_row["row"]), "source_video_sha256": hashlib.sha256(row["video"]["bytes"]).hexdigest(),
                                "image_shape": list(frame.shape), "figure_use": semantics["figure_use"]}
            provenance_path = args.output_dir / "representative_frame.json"
            save_json(provenance_path, image_provenance)
            outputs.extend([image_receipt, file_receipt(provenance_path)])
            canary = {"complete": True, "exact_repeat_equal": True, "episode": key[1], "seed": key[0],
                      "capture": receipt, "component_shapes": capture["component_shapes"],
                      "embedding_shapes": {name: list(value.shape) for name, value in capture["embeddings"].items()},
                      "representative_spatial_shapes": {name: list(value.shape) for name, value in capture["representative_spatial"].items()},
                      "model_provenance": model_receipt, "manifest_sha256": file_sha(manifest_path)}
            canary_path = args.output_dir / "CANARY_DONE.json"
            save_json(canary_path, canary)
            outputs.append(file_receipt(canary_path))
            print(json.dumps({"event": "canary_passed", "pid": os.getpid(), **receipt}), flush=True)
        print(json.dumps({"event": "episode_saved", "completed": position + 1, **receipt}), flush=True)
    parameter_after = parameter_hash(wm)
    if parameter_after != parameter_before:
        raise ValueError("Frozen model parameter hash changed during capture")
    done = {"complete": True, "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "episode_count": len(selected), "seed_episode_counts": {str(seed): sum(row["seed"] == seed for row in selected) for seed in seeds},
            "observed_states": 20 * len(selected), "one_step_transitions": 19 * len(selected),
            "duration_seconds": time.monotonic() - started, "model_provenance": model_receipt,
            "parameter_sha256_after": parameter_after, "parameters_unchanged": True,
            "manifest_sha256": file_sha(manifest_path), "script_sha256": file_sha(Path(__file__)),
            "semantics": semantics, "outputs": outputs, "causal_component_importance": "not_measured"}
    save_json(args.output_dir / "DONE.json", done)
    print(json.dumps({"event": "complete", "episodes": len(selected), "output_dir": str(args.output_dir),
                      "done_sha256": file_sha(args.output_dir / "DONE.json")}), flush=True)


if __name__ == "__main__":
    main()
