#!/usr/bin/env python3
"""Capture pooled or targeted-token JEPA-WM activations for manifest rows."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

from canary_capture import block_list, decode_selected, first_tensor
from model_loader import load_headless_metaworld
from protocol import file_sha256, reach_wall_labels, write_json_atomic


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"Empty manifest {path}")
    return rows


def table_row(table: Any, index: int) -> dict[str, Any]:
    return {name: table.column(name)[index].as_py() for name in table.column_names}


@torch.no_grad()
def capture_episode(
    wm: torch.nn.Module,
    row: dict[str, Any],
    encoder_blocks: list[torch.nn.Module],
    predictor_blocks: list[torch.nn.Module],
    encoder_ids: list[int],
    predictor_ids: list[int],
    mode: str,
    projection_dim: int,
) -> dict[str, Any]:
    frame_indices = tuple(range(0, 100, 5))
    frames = decode_selected(row["video"], frame_indices)
    states = np.asarray(row["states"], dtype=np.float32)
    actions = np.asarray(row["actions"], dtype=np.float32)
    rewards = np.asarray(row["rewards"], dtype=np.float32)
    visual = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0)
    proprio = torch.from_numpy(states[list(frame_indices), :4]).unsqueeze(0)
    captures: dict[str, torch.Tensor] = {}
    handles = []

    def hook(name: str):
        def record(_module: torch.nn.Module, _args: tuple[Any, ...], output: Any) -> None:
            captures[name] = first_tensor(output).detach()

        return record

    for index, block in zip(encoder_ids, encoder_blocks):
        handles.append(block.register_forward_hook(hook(f"encoder.{index}")))
    for index, block in zip(predictor_ids, predictor_blocks):
        handles.append(block.register_forward_hook(hook(f"predictor.{index}")))
    try:
        encoded = wm.encode({"visual": visual, "proprio": proprio})
        transition_count = len(frame_indices) - 1
        raw_action = torch.from_numpy(actions[: transition_count * 5]).reshape(transition_count, 5, 4)
        normalized_action = wm.preprocessor.normalize_actions(raw_action).reshape(transition_count, 1, -1)
        normalized_action = normalized_action.to(wm.device, dtype=torch.float32)
        current_visual = encoded["visual"][0, :transition_count].unsqueeze(1)
        current_proprio = encoded["proprio"][0, :transition_count].unsqueeze(1)
        action_features = wm.model.encode_act(normalized_action)
        predicted_visual, _, predicted_proprio = wm.model.forward_pred(
            current_visual, action_features, current_proprio
        )
    finally:
        for handle in handles:
            handle.remove()

    def reduce_tokens(value: torch.Tensor, family: str, block_id: int) -> torch.Tensor:
        if mode == "tokens":
            return value
        if mode == "pooled":
            return value.mean(dim=1)
        generator = torch.Generator(device="cpu").manual_seed(
            1729 + (0 if family == "encoder" else 10000) + block_id
        )
        projection = torch.randn(value.shape[-1], projection_dim, generator=generator, dtype=torch.float32)
        projection = projection.div(projection_dim**0.5).to(value.device, dtype=value.dtype)
        return value @ projection

    encoder_values = []
    encoder_cls = []
    for index in encoder_ids:
        value = captures[f"encoder.{index}"]
        encoder_cls.append(value[:, 0].to(torch.float16).cpu())
        patches = value[:, 1:]
        encoder_values.append(reduce_tokens(patches, "encoder", index).to(torch.float16).cpu())
    predictor_values = []
    for index in predictor_ids:
        value = captures[f"predictor.{index}"]
        predictor_values.append(reduce_tokens(value, "predictor", index).to(torch.float16).cpu())

    encoded_pooled = encoded["visual"].reshape(20, -1, 384).mean(dim=1).to(torch.float16).cpu()
    predicted_pooled = predicted_visual.reshape(transition_count, -1, 384).mean(dim=1).to(torch.float16).cpu()
    predicted_proprio_pooled = predicted_proprio.reshape(transition_count, -1, predicted_proprio.shape[-1]).mean(dim=1)
    return {
        "features": {
            "encoder": torch.stack(encoder_values, dim=1),
            "encoder_cls": torch.stack(encoder_cls, dim=1),
            "predictor": torch.stack(predictor_values, dim=1),
            "encoded_final": encoded_pooled,
            "predicted_final": predicted_pooled,
            "predicted_proprio": predicted_proprio_pooled.to(torch.float16).cpu(),
        },
        "labels": {key: torch.from_numpy(value) for key, value in reach_wall_labels(states, actions, rewards).items()},
        "mode": mode,
        "block_ids": {"encoder": encoder_ids, "predictor": predictor_ids},
        "projection": {
            "dimension": projection_dim if mode == "projected" else None,
            "seed_rule": "1729 + family_offset(encoder=0,predictor=10000) + block_id",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("pooled", "tokens", "projected"), default="pooled")
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--encoder-blocks", type=int, nargs="+")
    parser.add_argument("--predictor-blocks", type=int, nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shards", type=int, nargs="+")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    manifest_rows = read_manifest(args.manifest)
    if args.shards is not None:
        selected_shards = set(args.shards)
        manifest_rows = [row for row in manifest_rows if int(row["shard"]) in selected_shards]
    if args.limit is not None:
        manifest_rows = manifest_rows[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wm, _preprocessor, provenance = load_headless_metaworld(args.repo, device=args.device)
    encoder_path, all_encoder_blocks = block_list(
        wm.model.encoder,
        ("blocks", "base_model.blocks", "trunk.blocks", "backbone.blocks", "model.blocks"),
        12,
        "encoder",
    )
    predictor_path, all_predictor_blocks = block_list(wm.model.predictor, ("predictor_blocks", "blocks"), 6, "predictor")
    encoder_ids = args.encoder_blocks if args.encoder_blocks is not None else list(range(12))
    predictor_ids = args.predictor_blocks if args.predictor_blocks is not None else list(range(6))
    if len(set(encoder_ids)) != len(encoder_ids) or not set(encoder_ids) <= set(range(12)):
        raise ValueError(f"Invalid encoder blocks {encoder_ids}")
    if len(set(predictor_ids)) != len(predictor_ids) or not set(predictor_ids) <= set(range(6)):
        raise ValueError(f"Invalid predictor blocks {predictor_ids}")
    encoder_blocks = [all_encoder_blocks[index] for index in encoder_ids]
    predictor_blocks = [all_predictor_blocks[index] for index in predictor_ids]
    tables: dict[str, Any] = {}
    verified_inputs: set[str] = set()
    outputs = []
    started = time.monotonic()
    for position, manifest_row in enumerate(manifest_rows):
        filename = manifest_row["parquet"]
        source = args.data_dir / filename
        if filename not in verified_inputs:
            if file_sha256(source) != manifest_row["parquet_sha256"]:
                raise RuntimeError(f"Input hash mismatch for {source}")
            verified_inputs.add(filename)
        if filename not in tables:
            tables[filename] = pq.read_table(
                source, columns=["task", "seed", "episode", "video", "states", "actions", "rewards"]
            )
        data_row = table_row(tables[filename], int(manifest_row["row"]))
        observed = (data_row["task"], int(data_row["seed"]), int(data_row["episode"]))
        expected = (manifest_row["task"], manifest_row["seed"], manifest_row["episode"])
        if observed != expected:
            raise RuntimeError(f"Manifest row mismatch: observed={observed}, expected={expected}")
        payload = capture_episode(
            wm,
            data_row,
            encoder_blocks,
            predictor_blocks,
            encoder_ids,
            predictor_ids,
            args.mode,
            args.projection_dim,
        )
        payload["meta"] = manifest_row | {
            "schema_version": 1,
            "model": "jepa_wm_metaworld",
            "model_provenance": provenance,
            "encoder_block_path": encoder_path,
            "predictor_block_path": predictor_path,
        }
        output = args.output_dir / f"seed-{manifest_row['seed']}-episode-{manifest_row['episode']:03d}.pt"
        temporary = output.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, output)
        outputs.append(
            {
                "path": output.name,
                "sha256": file_sha256(output),
                "bytes": output.stat().st_size,
                "seed": manifest_row["seed"],
                "episode": manifest_row["episode"],
            }
        )
        print(json.dumps({"event": "episode", "completed": position + 1, **outputs[-1]}), flush=True)

    receipt = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete": len(outputs) == len(manifest_rows),
        "mode": args.mode,
        "encoder_blocks": encoder_ids,
        "predictor_blocks": predictor_ids,
        "projection_dim": args.projection_dim if args.mode == "projected" else None,
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "episode_count": len(outputs),
        "shards": sorted({int(row["shard"]) for row in manifest_rows}),
        "duration_seconds": time.monotonic() - started,
        "outputs": outputs,
    }
    write_json_atomic(args.output_dir / "DONE.json", receipt)
    print(json.dumps({key: value for key, value in receipt.items() if key != "outputs"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
