#!/usr/bin/env python3
"""Causally test whether predictor blocks mediate action-conditioned outcomes.

For each fixed observation, six action sequences are predicted.  A source action's
block residual is mixed into a different target action, while the input observation
and target action stay fixed.  A matched orientation-scrambled delta is the sham.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyarrow.parquet as pq
import torch

from canary_capture import block_list, decode_selected, first_tensor
from model_loader import load_headless_metaworld
from protocol import file_sha256, write_json_atomic


ACTION_NAMES = ("recorded", "hold", "plus_x", "minus_x", "plus_z", "minus_z")


def action_candidates(recorded: np.ndarray, strength: float) -> np.ndarray:
    candidates = np.zeros((6, 5, 4), dtype=np.float32)
    candidates[0] = recorded
    candidates[:, :, 3] = recorded[:, 3]
    candidates[2, :, 0] = strength
    candidates[3, :, 0] = -strength
    candidates[4, :, 2] = strength
    candidates[5, :, 2] = -strength
    return candidates


def metrics(
    target: torch.Tensor,
    source: torch.Tensor,
    changed: torch.Tensor,
    goal: torch.Tensor,
) -> dict[str, float]:
    target = target.float().reshape(-1)
    source = source.float().reshape(-1)
    changed = changed.float().reshape(-1)
    goal = goal.float().reshape(-1)
    reference = source - target
    effect = changed - target
    denominator = float(reference.square().sum())
    transfer = float(torch.dot(effect, reference) / max(denominator, 1e-12))
    cosine = float(torch.dot(effect, reference) / max(float(effect.norm() * reference.norm()), 1e-12))
    target_cost = float((target - goal).square().mean())
    source_cost = float((source - goal).square().mean())
    changed_cost = float((changed - goal).square().mean())
    cost_denominator = source_cost - target_cost
    return {
        "transfer_fraction": transfer,
        "effect_cosine": cosine,
        "effect_norm": float(effect.norm()),
        "reference_norm": float(reference.norm()),
        "target_goal_cost": target_cost,
        "source_goal_cost": source_cost,
        "changed_goal_cost": changed_cost,
        "goal_cost_transfer": (changed_cost - target_cost) / cost_denominator
        if abs(cost_denominator) > 1e-12
        else float("nan"),
    }


@torch.no_grad()
def run_snapshot(
    wm: torch.nn.Module,
    data_row: dict[str, Any],
    snapshot: dict[str, Any],
    block_modules: dict[int, torch.nn.Module],
    betas: list[float],
    strength: float,
) -> dict[str, Any]:
    frame = int(snapshot["frame_index"])
    frames = decode_selected(data_row["video"], (frame, frame + 5, 95))
    states = np.asarray(data_row["states"], dtype=np.float32)
    actions = np.asarray(data_row["actions"], dtype=np.float32)
    visual = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0)
    proprio = torch.from_numpy(states[[frame, frame + 5, 95], :4]).unsqueeze(0)
    encoded = wm.encode({"visual": visual, "proprio": proprio})
    current_visual = encoded["visual"][:, 0:1]
    current_proprio = encoded["proprio"][:, 0:1]
    realized_next = encoded["visual"][:, 1]
    goal = encoded["visual"][:, 2]

    raw_candidates = action_candidates(actions[frame : frame + 5], strength)
    raw_tensor = torch.from_numpy(raw_candidates)
    normalized = wm.preprocessor.normalize_actions(raw_tensor).reshape(6, 1, -1)
    normalized = normalized.to(wm.device, dtype=torch.float32)
    captures: dict[int, torch.Tensor] = {}
    handles = []
    for block_id, module in block_modules.items():
        def record(_module: torch.nn.Module, _args: tuple[Any, ...], output: Any, *, block: int = block_id) -> None:
            captures[block] = first_tensor(output).detach().clone()

        handles.append(module.register_forward_hook(record))
    try:
        baseline, _, _ = wm.model.forward_pred(
            current_visual.expand(6, *current_visual.shape[1:]),
            wm.model.encode_act(normalized),
            current_proprio.expand(6, *current_proprio.shape[1:]),
        )
    finally:
        for handle in handles:
            handle.remove()
    baseline = baseline[:, -1].detach()
    pairs = [(source, target) for source in range(6) for target in range(6) if source != target]
    source_index = torch.tensor([source for source, _target in pairs], device=wm.device)
    target_index = torch.tensor([target for _source, target in pairs], device=wm.device)
    repeated_actions = normalized[target_index]
    batch = len(pairs)
    rows = []

    def patched_forward(block_id: int, replacement: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        handle = block_modules[block_id].register_forward_hook(
            lambda _module, _args, output: replacement(first_tensor(output))
        )
        try:
            output, _, _ = wm.model.forward_pred(
                current_visual.expand(batch, *current_visual.shape[1:]),
                wm.model.encode_act(repeated_actions),
                current_proprio.expand(batch, *current_proprio.shape[1:]),
            )
        finally:
            handle.remove()
        return output[:, -1].detach()

    for block_id in sorted(block_modules):
        source_activation = captures[block_id][source_index]
        target_activation = captures[block_id][target_index]
        delta = source_activation - target_activation
        generator = torch.Generator(device="cpu").manual_seed(
            1000003 * int(snapshot["seed"])
            + 1009 * int(snapshot["episode"])
            + 17 * int(snapshot["model_step"])
            + block_id
        )
        permutation = torch.randperm(delta.shape[-1], generator=generator).to(delta.device)
        signs = torch.randint(0, 2, (delta.shape[-1],), generator=generator, dtype=torch.float32)
        signs = (2 * signs - 1).to(delta.device, dtype=delta.dtype)
        sham_delta = delta[..., permutation] * signs
        sham_cosine = float(
            (delta * sham_delta).sum() / max(float(delta.norm() * sham_delta.norm()), 1e-12)
        )
        for beta in betas:
            patched = patched_forward(block_id, lambda output, b=beta: output + b * delta)
            sham = patched_forward(block_id, lambda output, b=beta: output + b * sham_delta)
            for pair_index, (source, target) in enumerate(pairs):
                common = {
                    "block": block_id,
                    "beta": beta,
                    "source": ACTION_NAMES[source],
                    "target": ACTION_NAMES[target],
                    "activation_delta_norm": float(delta[pair_index].float().norm()),
                    "sham_cosine": sham_cosine,
                }
                rows.append(
                    common
                    | {"arm": "patch"}
                    | metrics(baseline[target], baseline[source], patched[pair_index], goal[0])
                )
                rows.append(
                    common
                    | {"arm": "orientation_sham"}
                    | metrics(baseline[target], baseline[source], sham[pair_index], goal[0])
                )
    recorded_error = float((baseline[0] - realized_next[0]).square().mean())
    baseline_goal_cost = {
        name: float((baseline[index] - goal[0]).square().mean()) for index, name in enumerate(ACTION_NAMES)
    }
    return {
        "snapshot": snapshot,
        "action_names": ACTION_NAMES,
        "action_strength": strength,
        "recorded_one_step_prediction_mse": recorded_error,
        "baseline_goal_cost": baseline_goal_cost,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blocks", type=int, nargs="+", default=[2, 3, 5])
    parser.add_argument("--betas", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    parser.add_argument("--action-strength", type=float, default=0.5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    snapshots = [json.loads(line) for line in args.snapshot_manifest.read_text().splitlines() if line.strip()]
    if args.seeds is not None:
        selected_seeds = set(args.seeds)
        snapshots = [row for row in snapshots if int(row["seed"]) in selected_seeds]
    if args.limit is not None:
        snapshots = snapshots[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wm, _preprocessor, provenance = load_headless_metaworld(args.repo, device=args.device)
    _path, all_blocks = block_list(wm.model.predictor, ("predictor_blocks", "blocks"), 6, "predictor")
    block_modules = {block: all_blocks[block] for block in args.blocks}
    tables: dict[str, Any] = {}
    verified: set[str] = set()
    outputs = []
    started = time.monotonic()
    for index, snapshot in enumerate(snapshots):
        filename = snapshot["parquet"]
        path = args.data_dir / filename
        if filename not in verified:
            if file_sha256(path) != snapshot["parquet_sha256"]:
                raise RuntimeError(f"Input hash mismatch {path}")
            verified.add(filename)
        if filename not in tables:
            tables[filename] = pq.read_table(
                path, columns=["task", "seed", "episode", "video", "states", "actions"]
            )
        table = tables[filename]
        data_row = {name: table.column(name)[int(snapshot["row"])].as_py() for name in table.column_names}
        observed = (data_row["task"], int(data_row["seed"]), int(data_row["episode"]))
        expected = (snapshot["task"], snapshot["seed"], snapshot["episode"])
        if observed != expected:
            raise RuntimeError(f"Snapshot row mismatch {observed} != {expected}")
        result = run_snapshot(wm, data_row, snapshot, block_modules, args.betas, args.action_strength)
        output = args.output_dir / f"seed-{snapshot['seed']}-episode-{snapshot['episode']:03d}.json"
        write_json_atomic(output, result)
        outputs.append({"path": output.name, "sha256": file_sha256(output)})
        print(json.dumps({"event": "snapshot", "completed": index + 1, "path": output.name}), flush=True)
    receipt = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete": len(outputs) == len(snapshots),
        "scope": "discovery-only fixed-observation action counterfactuals",
        "model_provenance": provenance,
        "snapshot_manifest": str(args.snapshot_manifest),
        "snapshot_manifest_sha256": file_sha256(args.snapshot_manifest),
        "blocks": args.blocks,
        "betas": args.betas,
        "action_strength": args.action_strength,
        "snapshot_count": len(outputs),
        "seeds": sorted({int(snapshot["seed"]) for snapshot in snapshots}),
        "duration_seconds": time.monotonic() - started,
        "outputs": outputs,
    }
    write_json_atomic(args.output_dir / "DONE.json", receipt)
    print(json.dumps({key: value for key, value in receipt.items() if key != "outputs"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
