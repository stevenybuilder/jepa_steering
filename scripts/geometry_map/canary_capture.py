#!/usr/bin/env python3
"""One-real-sample integrity canary for the released MetaWorld JEPA-WM.

The canary verifies the exact checkpoint and official data path before any burst
compute is rented.  It captures every encoder and action-conditioned predictor
block twice, checks deterministic equality, and emits a compact receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import imageio
import numpy as np
import pyarrow.parquet as pq
import torch

from model_loader import load_headless_metaworld


def sha256(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return first_tensor(item)
            except TypeError:
                pass
    if isinstance(value, dict):
        for item in value.values():
            try:
                return first_tensor(item)
            except TypeError:
                pass
    raise TypeError(f"No tensor in hook output of type {type(value)!r}")


def block_list(root: torch.nn.Module, paths: Iterable[str], expected: int, label: str) -> tuple[str, list[torch.nn.Module]]:
    for path in paths:
        value: Any = root
        try:
            for part in path.split("."):
                value = getattr(value, part)
        except AttributeError:
            continue
        if isinstance(value, (torch.nn.ModuleList, list, tuple)) and len(value) == expected:
            return path, list(value)
    candidates = [(name, len(module)) for name, module in root.named_modules() if isinstance(module, torch.nn.ModuleList)]
    raise RuntimeError(f"Could not identify {expected} {label} blocks; ModuleLists={candidates}")


def decode_selected(video: dict[str, Any], indices: Iterable[int]) -> np.ndarray:
    indices = tuple(int(index) for index in indices)
    wanted = set(indices)
    video_bytes = video.get("bytes") if isinstance(video, dict) else None
    if not video_bytes:
        raise RuntimeError(f"Expected embedded MP4 bytes, got {type(video)!r}")
    reader = imageio.get_reader(io.BytesIO(video_bytes), format="mp4")
    output = []
    try:
        for index, frame in enumerate(reader):
            if index in wanted:
                output.append((index, frame))
            if index == max(indices):
                break
    finally:
        reader.close()
    found = {index: frame for index, frame in output}
    if set(found) != wanted:
        raise RuntimeError(f"Decoded indices {sorted(found)}, expected {sorted(wanted)}")
    return np.stack([found[index] for index in indices])


def load_row(path: Path, row_index: int) -> dict[str, Any]:
    table = pq.read_table(path, columns=["task", "seed", "episode", "video", "states", "actions"])
    if not 0 <= row_index < table.num_rows:
        raise IndexError(row_index)
    return {name: table.column(name)[row_index].as_py() for name in table.column_names}


def parameter_signature(module: torch.nn.Module) -> list[tuple[str, int, int, float, float]]:
    signature = []
    for name, parameter in module.named_parameters():
        data = parameter.detach()
        signature.append(
            (
                name,
                data.numel(),
                int(data._version),
                float(data.double().sum().cpu()),
                float(data.double().square().sum().cpu()),
            )
        )
    return signature


def run_once(
    wm: torch.nn.Module,
    visual: torch.Tensor,
    proprio: torch.Tensor,
    action: torch.Tensor,
    encoder_blocks: list[torch.nn.Module],
    predictor_blocks: list[torch.nn.Module],
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    captures: dict[str, torch.Tensor] = {}
    handles = []

    def hook(name: str):
        def record(_module: torch.nn.Module, _args: tuple[Any, ...], output: Any) -> None:
            captures[name] = first_tensor(output).detach().cpu().clone()

        return record

    for index, block in enumerate(encoder_blocks):
        handles.append(block.register_forward_hook(hook(f"encoder.block.{index}")))
    for index, block in enumerate(predictor_blocks):
        handles.append(block.register_forward_hook(hook(f"predictor.block.{index}")))
    try:
        z_init = wm.encode({"visual": visual, "proprio": proprio})
        normalized_action = wm.preprocessor.normalize_actions(action.cpu()).reshape(1, 1, -1)
        normalized_action = normalized_action.to(wm.device, dtype=torch.float32)
        rollout = wm.unroll(z_init, normalized_action.permute(1, 0, 2))
    finally:
        for handle in handles:
            handle.remove()
    return captures, z_init["visual"].detach().cpu(), rollout["visual"].detach().cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="Official jepa-wms checkout")
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    row = load_row(args.parquet, args.row)
    if row["task"] != "mw-reach-wall":
        raise RuntimeError(f"Canary row is {row['task']!r}, not 'mw-reach-wall'")
    # Evaluation starts from one observed frame; ctxt_window=2 is the maximum
    # sliding history after autoregressive predictions have begun.
    frames = decode_selected(row["video"], indices=(0,))
    states = np.asarray(row["states"], dtype=np.float32)
    actions = np.asarray(row["actions"], dtype=np.float32)
    visual = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0)
    proprio = torch.from_numpy(states[0:1, :4]).unsqueeze(0)
    action = torch.from_numpy(actions[0:5]).unsqueeze(0)

    wm, _preprocessor, model_provenance = load_headless_metaworld(args.repo, device=args.device)
    encoder_path, encoder_blocks = block_list(
        wm.model.encoder,
        ("blocks", "base_model.blocks", "trunk.blocks", "backbone.blocks", "model.blocks"),
        expected=12,
        label="encoder",
    )
    predictor_path, predictor_blocks = block_list(
        wm.model.predictor,
        ("predictor_blocks", "blocks"),
        expected=6,
        label="predictor",
    )

    before = parameter_signature(wm)
    first, encoded_first, rollout_first = run_once(
        wm, visual, proprio, action, encoder_blocks, predictor_blocks
    )
    second, encoded_second, rollout_second = run_once(
        wm, visual, proprio, action, encoder_blocks, predictor_blocks
    )
    after = parameter_signature(wm)

    expected_sites = [f"encoder.block.{i}" for i in range(12)] + [f"predictor.block.{i}" for i in range(6)]
    missing = sorted(set(expected_sites) - set(first))
    repeat_max_abs = {
        key: float((first[key] - second[key]).abs().max()) for key in expected_sites if key in first and key in second
    }
    receipt = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete": not missing
        and first.keys() == second.keys()
        and max(repeat_max_abs.values(), default=float("inf")) == 0.0
        and torch.equal(encoded_first, encoded_second)
        and torch.equal(rollout_first, rollout_second)
        and before == after,
        "model": "jepa_wm_metaworld",
        "checkpoint_source": "official checkpoint and constructors; optional RGB decoder omitted",
        "model_provenance": model_provenance,
        "repo": str(args.repo.resolve()),
        "parquet": str(args.parquet.resolve()),
        "parquet_sha256": sha256(args.parquet),
        "row": args.row,
        "task": row["task"],
        "seed": int(row["seed"]),
        "episode": int(row["episode"]),
        "input_shapes": {
            "visual": list(visual.shape),
            "proprio": list(proprio.shape),
            "action": list(action.shape),
        },
        "encoder_block_path": encoder_path,
        "predictor_block_path": predictor_path,
        "sites": {key: list(value.shape) for key, value in first.items()},
        "missing_sites": missing,
        "repeat_max_abs": repeat_max_abs,
        "encoded_shape": list(encoded_first.shape),
        "rollout_shape": list(rollout_first.shape),
        "encoded_repeat_equal": torch.equal(encoded_first, encoded_second),
        "rollout_repeat_equal": torch.equal(rollout_first, rollout_second),
        "parameters_unchanged": before == after,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(wm.device),
    }
    output = args.output_dir / "DONE.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not receipt["complete"]:
        raise SystemExit("Canary failed; see DONE.json")


if __name__ == "__main__":
    main()
