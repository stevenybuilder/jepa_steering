"""Fit and freeze a vision-action coupling operator without opening development outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from . import VENDOR_COMMIT
from .backends import JepaBackend
from .benchmark import batches, filter_reviewed_development
from .interventions import validate_frozen_protocol, validate_operator_bank, window_key
from .protocol import sha256, validate_manifest, write_json


FIT_SEED = 2026090701
PERMUTATION_SEED = 2026090702
RANDOM_CONTROL_SEED = 2026090703
HORIZON = 3
BLOCK = 3


def select_fit_rows(rows: list[dict], task: str, max_lineage_groups: int, seed: int) -> list[dict]:
    """Choose one deterministic representative row per fit lineage group."""
    candidates = [row for row in rows if row["split"] == "fit" and row["task"] == task]
    by_group: dict[str, list[dict]] = {}
    for row in candidates:
        by_group.setdefault(row["lineage_group"], []).append(row)
    if not by_group:
        raise ValueError(f"No fit lineages found for task: {task}")

    def digest(label: str) -> bytes:
        return hashlib.sha256(f"{seed}:{label}".encode()).digest()

    representatives = [
        min(group_rows, key=lambda row: digest(row["trajectory_id"]))
        for group_rows in by_group.values()
    ]
    representatives.sort(key=lambda row: digest(row["lineage_group"]))
    if max_lineage_groups:
        representatives = representatives[:max_lineage_groups]
    return representatives


class NativeCouplingCapture:
    """Capture the native H3 predictor visual field and P3 AdaLN condition."""

    def __init__(self, predictor, horizon: int = HORIZON, block: int = BLOCK):
        self.predictor = predictor
        self.horizon_target = horizon
        self.block_index = block
        self.horizon = 0
        self.visual: torch.Tensor | None = None
        self.condition: torch.Tensor | None = None
        self.handles = []

    def __enter__(self):
        blocks = self.predictor.predictor_blocks
        if len(blocks) != 6 or not 0 <= self.block_index < len(blocks):
            raise ValueError("Pinned capture contract requires six predictor blocks")
        self.handles.append(self.predictor.register_forward_pre_hook(self._predictor_input))
        self.handles.append(blocks[self.block_index].register_forward_pre_hook(
            self._block_input, with_kwargs=True))
        return self

    def _predictor_input(self, module, args):
        del module
        self.horizon += 1
        if len(args) != 3:
            raise ValueError("Pinned predictor must receive visual, action, and proprio inputs")
        if self.horizon == self.horizon_target:
            if self.visual is not None:
                raise RuntimeError("H3 visual site executed more than once")
            self.visual = args[0][:, -1].detach().float().cpu().clone()
        return None

    def _block_input(self, module, args, kwargs):
        del module, kwargs
        if self.horizon != self.horizon_target:
            return None
        if len(args) < 2:
            raise ValueError("Pinned predictor block must receive x and condition inputs")
        if self.condition is not None:
            raise RuntimeError("H3 P3 condition site executed more than once")
        self.condition = args[1][:, -1].detach().float().cpu().clone()
        return None

    def __exit__(self, exc_type, exc, traceback):
        del exc, traceback
        for handle in self.handles:
            handle.remove()
        if exc_type is None and (
                self.horizon != 6 or self.visual is None or self.condition is None):
            raise RuntimeError("Native capture did not observe exactly one H3 visual/P3 pair")


def _unit(value: torch.Tensor, label: str) -> torch.Tensor:
    norm = value.double().norm()
    if not torch.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"Degenerate {label}")
    return value / norm.to(value.dtype)


def robust_score_scale(scores: torch.Tensor) -> float:
    centered = scores - scores.median()
    scale = centered.abs().median().double() / 0.6744897501960817
    if not torch.isfinite(scale) or scale <= 1e-12:
        scale = scores.double().std(unbiased=True)
    if not torch.isfinite(scale) or scale <= 1e-12:
        raise ValueError("Coupled score scale is degenerate")
    return float(scale)


def fit_coupled_directions(
    visual: torch.Tensor,
    condition: torch.Tensor,
    iterations: int = 12,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Fit a deterministic rank-one maximum-covariance pair by power iteration."""
    if visual.ndim < 2 or condition.ndim != 2 or visual.shape[0] != condition.shape[0]:
        raise ValueError("Visual and condition captures require a shared sample dimension")
    if visual.shape[0] < 3 or iterations < 1:
        raise ValueError("At least three samples and one solver iteration are required")
    visual_shape = tuple(visual.shape[1:])
    x = visual.reshape(visual.shape[0], -1).float()
    y = condition.float()
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("Non-finite native activations")
    initial = int(y.square().sum(0).argmax())
    action_score = _unit(y[:, initial], "initial action score")
    visual_direction = action_direction = None
    for _ in range(iterations):
        visual_direction = _unit(x.T.mv(action_score), "visual covariance direction")
        visual_score = _unit(x.mv(visual_direction), "visual score")
        action_direction = _unit(y.T.mv(visual_score), "action covariance direction")
        action_score = _unit(y.mv(action_direction), "action score")
    visual_direction = _unit(x.T.mv(action_score), "final visual direction")
    visual_scores = x.mv(visual_direction)
    action_direction = _unit(y.T.mv(_unit(visual_scores, "final visual score")),
                             "final action direction")
    action_scores = y.mv(action_direction)
    covariance = float((visual_scores.double() * action_scores.double()).mean())
    if covariance < 0:
        visual_direction = -visual_direction
        visual_scores = -visual_scores
        covariance = -covariance
    # Resolve the otherwise arbitrary joint sign without consulting any outcome.
    pivot = int(action_direction.abs().argmax())
    if action_direction[pivot] < 0:
        visual_direction = -visual_direction
        action_direction = -action_direction
        visual_scores = -visual_scores
        action_scores = -action_scores
    diagnostics = {
        "algorithm": "rank_one_cross_covariance_power_iteration",
        "iterations": iterations,
        "samples": visual.shape[0],
        "visual_shape": list(visual_shape),
        "condition_shape": list(condition.shape[1:]),
        "cross_covariance": covariance,
        "visual_score_robust_sigma": robust_score_scale(visual_scores),
        "action_score_robust_sigma": robust_score_scale(action_scores),
        "visual_direction_l2": float(visual_direction.double().norm()),
        "action_direction_l2": float(action_direction.double().norm()),
        "sign_rule": "largest_absolute_action_coordinate_positive",
    }
    return visual_direction.reshape(visual_shape), action_direction, diagnostics


def orthogonal_random_control(direction: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    flat_direction = _unit(direction.flatten().float(), "reference direction")
    random = torch.randn(flat_direction.shape, generator=generator)
    random = random - torch.dot(random, flat_direction) * flat_direction
    return _unit(random, "orthogonal random control").reshape_as(direction)


def permute_visual_direction(direction: torch.Tensor, seed: int) -> torch.Tensor:
    if direction.ndim < 3 or tuple(direction.shape[-3:-1]) != (16, 16):
        raise ValueError(f"Expected final visual grid dimensions 16x16; got {tuple(direction.shape)}")
    prefix, width = direction.shape[:-3], direction.shape[-1]
    patches = direction.reshape(*prefix, 256, width)
    permutation = torch.randperm(256, generator=torch.Generator().manual_seed(seed))
    return patches[..., permutation, :].reshape_as(direction)


def make_protocol(
    task: str,
    manifest_sha256: str,
    checkpoint_sha256: str,
    fit_receipt_sha256: str,
    visual_scale: float,
    action_scale: float,
    frozen_at: str,
) -> dict:
    visual = {"site": "predictor_visual", "horizon": HORIZON,
              "tensor": "visual_direction", "scale": visual_scale}
    action = {"site": "block_condition", "horizon": HORIZON, "block": BLOCK,
              "tensor": "action_direction", "scale": action_scale}
    permuted = {**visual, "tensor": "permuted_visual_direction"}
    random_visual = {**visual, "tensor": "random_visual_direction"}
    random_action = {**action, "tensor": "random_action_direction"}
    protocol = {
        "schema_version": 1,
        "status": "frozen",
        "category": "vision_action_coupling",
        "vendor_commit": VENDOR_COMMIT,
        "manifest_sha256": manifest_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "fit_receipt_sha256": fit_receipt_sha256,
        "fit_split": "fit",
        "evaluation_split": "development",
        "tasks": [task],
        "hypothesis": (
            "A fit-only maximum-covariance visual/action-condition edit at H3/P3 "
            "produces a nonadditive forecast response beyond either component alone."
        ),
        "dose_budget": {
            "rule": "each component is 0.1 fit-split robust score sigma along a unit vector",
            "visual_delivered_l2": visual_scale,
            "action_condition_delivered_l2": action_scale,
            "joint_policy": "retain both single-component doses",
            "matched_random_policy": "orthogonal unit directions with identical per-site doses",
            "permutation_policy": "fixed permutation of 256 visual patches preserves visual L2",
            "outcome_tuning": False,
        },
        "frozen_at": frozen_at,
        "development_analysis_plan": {
            "resampling_unit": "lineage_group",
            "task_pooling": False,
            "bootstrap_replicates": 10000,
            "bootstrap_seed": 2026090704,
            "interval": "two-sided percentile 95%",
            "primary_forecast_endpoint": "proprio_mse_h6",
            "primary_mechanism_endpoint": "visual_output_interaction_mse_h6",
            "role": "variance/effect-size estimation before a separate confirmation freeze",
            "arm_selection": False,
            "confirmatory_multiplicity_and_smallest_useful_effect": "freeze after development",
        },
        "arms": [
            {"name": "native", "edits": []},
            {"name": "zero_dose", "edits": [{**visual, "scale": 0.}, {**action, "scale": 0.}]},
            {"name": "visual_only", "edits": [visual]},
            {"name": "action_condition_only", "edits": [action]},
            {"name": "joint", "edits": [visual, action]},
            {"name": "permuted_visual", "edits": [permuted]},
            {"name": "permuted_joint", "edits": [permuted, action]},
            {"name": "matched_random", "edits": [random_visual, random_action]},
        ],
        "primary_contrasts": [
            {"name": "joint_vs_visual_only", "candidate": "joint", "control": "visual_only"},
            {"name": "joint_vs_action_condition_only", "candidate": "joint",
             "control": "action_condition_only"},
            {"name": "joint_vs_matched_random", "candidate": "joint", "control": "matched_random"},
            {"name": "joint_vs_permuted_joint", "candidate": "joint", "control": "permuted_joint"},
        ],
    }
    validate_frozen_protocol(protocol)
    return protocol


def make_operator_bank(
    protocol_sha256: str,
    evaluation_rows: list[dict],
    global_tensors: dict[str, torch.Tensor],
) -> dict:
    rows = {}
    for row in evaluation_rows:
        for start in row["starts"]:
            meta = {"trajectory_id": row["trajectory_id"], "lineage_group": row["lineage_group"],
                    "task": row["task"], "start": start, "split": "development"}
            rows[window_key(meta)] = {**meta, "tensors": {}}
    return {
        "schema_version": 1,
        "protocol_sha256": protocol_sha256,
        "global_tensors": {key: value.detach().float().cpu().contiguous()
                           for key, value in global_tensors.items()},
        "rows": rows,
    }


def _model_versions(model) -> dict[str, int]:
    versions = {f"parameter:{name}": value._version for name, value in model.named_parameters()}
    versions.update({f"buffer:{name}": value._version for name, value in model.named_buffers()})
    return versions


@torch.inference_mode()
def capture_fit_activations(args, backend, fit_rows):
    visual, condition, metadata = [], [], []
    for meta, frames, proprio, raw_actions in batches(
            fit_rows, args.batch_size, backend, args.data_root, pin_memory=True):
        encoded = backend.encode(frames, proprio)
        context = backend.context(encoded)
        actions = backend.normalize_actions(raw_actions)
        with NativeCouplingCapture(backend.predictor) as capture:
            backend.predict(context, actions)
        visual.append(capture.visual)
        condition.append(capture.condition)
        metadata.extend(meta)
        del encoded, context, actions
    if not metadata:
        raise ValueError("No fit windows were captured")
    return torch.cat(visual), torch.cat(condition), metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--exposure-registry", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--max-fit-lineage-groups", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dose-fraction", type=float, default=0.1)
    parser.add_argument("--solver-iterations", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if (args.max_fit_lineage_groups < 3 or args.batch_size < 1 or
            not 0 < args.dose_fraction <= 1 or args.solver_iterations < 1 or
            not args.device.startswith("cuda:")):
        parser.error("Invalid fit count, batch size, dose, solver iterations, or CUDA device")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        started = time.perf_counter()
        manifest_hash = sha256(args.manifest)
        if sha256(args.checkpoint) != args.checkpoint_sha256:
            raise ValueError("Checkpoint checksum mismatch")
        rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
        validate_manifest(rows)
        datasets = {row["dataset"] for row in rows if row["task"] == args.task}
        if len(datasets) != 1:
            raise ValueError("One task must resolve to exactly one dataset")
        registry = json.loads(args.exposure_registry.read_text())
        if registry.get("manifest_sha256") != manifest_hash:
            raise ValueError("Exposure registry is not bound to the manifest")
        fit_rows = select_fit_rows(rows, args.task, args.max_fit_lineage_groups, FIT_SEED)
        evaluation_rows = [row for row in filter_reviewed_development(rows, registry, manifest_hash)
                           if row["split"] == "development" and row["task"] == args.task]
        if not evaluation_rows:
            raise ValueError("No reviewed development rows are registered for this task")
        write_json(args.output / "fit_selection.json", fit_rows)
        setup_started = time.perf_counter()
        backend = JepaBackend(
            args.vendor, args.checkpoint, args.checkpoint_sha256, next(iter(datasets)),
            args.device, precision="float32", allow_tf32=False,
        )
        setup_seconds = time.perf_counter() - setup_started
        versions = _model_versions(backend.model)
        capture_started = time.perf_counter()
        visual, condition, metadata = capture_fit_activations(args, backend, fit_rows)
        capture_seconds = time.perf_counter() - capture_started
        if versions != _model_versions(backend.model):
            raise RuntimeError("Frozen model parameter or buffer versions changed during fitting")
        visual_direction, action_direction, diagnostics = fit_coupled_directions(
            visual, condition, args.solver_iterations)
        visual_scale = args.dose_fraction * diagnostics["visual_score_robust_sigma"]
        action_scale = args.dose_fraction * diagnostics["action_score_robust_sigma"]
        if not all(math.isfinite(value) and value > 0 for value in (visual_scale, action_scale)):
            raise ValueError("Fit-derived dose is not finite and positive")
        global_tensors = {
            "visual_direction": visual_direction,
            "action_direction": action_direction,
            "permuted_visual_direction": permute_visual_direction(
                visual_direction, PERMUTATION_SEED),
            "random_visual_direction": orthogonal_random_control(
                visual_direction, RANDOM_CONTROL_SEED),
            "random_action_direction": orthogonal_random_control(
                action_direction, RANDOM_CONTROL_SEED + 1),
        }
        frozen_at = datetime.now(timezone.utc).isoformat()
        receipt = {
            "schema_version": 1,
            "status": "fit_only_complete",
            "category": "vision_action_coupling",
            "task": args.task,
            "dataset": next(iter(datasets)),
            "vendor_commit": VENDOR_COMMIT,
            "manifest_sha256": manifest_hash,
            "checkpoint_sha256": args.checkpoint_sha256,
            "fit_split": "fit",
            "fit_seed": FIT_SEED,
            "representative_policy": "one deterministic released row per lineage group",
            "fit_trajectory_ids": [row["trajectory_id"] for row in fit_rows],
            "fit_lineage_groups": [row["lineage_group"] for row in fit_rows],
            "fit_lineage_group_count": len(fit_rows),
            "fit_window_count": len(metadata),
            "fit_window_registry_sha256": hashlib.sha256(json.dumps(
                metadata, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "development_outcomes_accessed": False,
            "holdout_outcomes_accessed": False,
            "holdout_metadata_accessed_for_manifest_validation_only": True,
            "development_metadata_used_only_to_preallocate_bank_rows": True,
            "target_site": {"horizon": HORIZON, "predictor_block": BLOCK,
                            "visual": "newest native imagined visual field",
                            "action": "newest native AdaLN action condition"},
            "solver": diagnostics,
            "dose_fraction": args.dose_fraction,
            "visual_delivered_l2": visual_scale,
            "action_condition_delivered_l2": action_scale,
            "permutation_seed": PERMUTATION_SEED,
            "random_control_seed": RANDOM_CONTROL_SEED,
            "strict_fp32": True,
            "allow_tf32": False,
            "model_parameters_and_buffers_unchanged": True,
            "model_setup_seconds": setup_seconds,
            "capture_seconds": capture_seconds,
            "created_at": frozen_at,
            "backend": backend.provenance,
            "environment": {"python": platform.python_version(), "torch": torch.__version__,
                            "numpy": np.__version__, "cuda": torch.version.cuda,
                            "device": str(backend.device),
                            "gpu": torch.cuda.get_device_name(backend.device)},
        }
        write_json(args.output / "fit_receipt.json", receipt)
        fit_receipt_hash = sha256(args.output / "fit_receipt.json")
        protocol = make_protocol(
            args.task, manifest_hash, args.checkpoint_sha256, fit_receipt_hash,
            visual_scale, action_scale, frozen_at,
        )
        write_json(args.output / "protocol.json", protocol)
        protocol_hash = sha256(args.output / "protocol.json")
        bank = make_operator_bank(protocol_hash, evaluation_rows, global_tensors)
        torch.save(bank, args.output / "operator_bank.pt")
        selected_meta = [
            {"trajectory_id": row["trajectory_id"], "lineage_group": row["lineage_group"],
             "task": row["task"], "start": start, "split": "development"}
            for row in evaluation_rows for start in row["starts"]
        ]
        validate_operator_bank(bank, protocol_hash, selected_meta)
        write_json(args.output / "DONE.json", {
            "status": "vision_action_coupling_protocol_frozen",
            "task": args.task,
            "fit_receipt_sha256": fit_receipt_hash,
            "protocol_sha256": protocol_hash,
            "operator_bank_sha256": sha256(args.output / "operator_bank.pt"),
            "fit_selection_sha256": sha256(args.output / "fit_selection.json"),
            "fit_lineage_groups": len(fit_rows),
            "fit_windows": len(metadata),
            "registered_development_rows": len(evaluation_rows),
            "registered_development_windows": len(selected_meta),
            "seconds": time.perf_counter() - started,
        })
        print(json.dumps(json.loads((args.output / "DONE.json").read_text())), flush=True)
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"status": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
