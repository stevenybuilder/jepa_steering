"""Frozen intervention protocols and shape-checked predictor hooks."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


# Predictor blocks are zero-indexed. The singleton arms make the depth-locality
# claim exhaustive over the six-block predictor; B2+B3 remains the archived,
# evidence-derived intermediate-zone composite and all-six is the global arm.
LAYER_MECHANISM_ARM_BLOCKS = {
    **{f"single_block{block}": {block} for block in range(6)},
    "intermediate_blocks2_3": {2, 3},
    "all_six_blocks": set(range(6)),
}
LAYER_RANDOM_CONTROL_FOR = {
    arm: f"matched_random_{arm}" for arm in LAYER_MECHANISM_ARM_BLOCKS
}
LAYER_ARM_BLOCKS = {
    **LAYER_MECHANISM_ARM_BLOCKS,
    **{
        LAYER_RANDOM_CONTROL_FOR[arm]: blocks
        for arm, blocks in LAYER_MECHANISM_ARM_BLOCKS.items()
    },
}
RANK_ARM_RANKS = {
    "rank1": 1,
    "rank4": 4,
    "rank8": 8,
    "matched_random_rank1": 1,
    "matched_random_rank4": 4,
    "matched_random_rank8": 8,
}
SPATIAL_RANDOM_POSITION_ARMS = {
    "random_position_one_patch",
    "random_position_contiguous_group",
    "random_position_equal_size_scattered_group",
}


CATEGORY_ARMS = {
    "vision_action_coupling": {
        "native", "zero_dose", "visual_only", "action_condition_only", "joint",
        "joint_equal_standardized_energy", "permuted_visual", "permuted_joint",
        "matched_random", "matched_random_equal_standardized_energy",
    },
    "action_response_geometry": {
        "native", "zero_dose", "equal_anchor_linear", "cubic", "projected_cubic",
        "reflected_curvature", "matched_random",
    },
    "imagined_time_routing": {
        "native", "zero_dose", "constant_gate", "memoryless_gate",
        "hmm_filtered_gate", "matched_random",
    },
    "distribution_spatial": {
        "native", "zero_dose", "one_patch", "contiguous_group",
        "equal_size_scattered_group", "all_patches",
        "matched_random_one_patch", "matched_random_contiguous_group",
        "matched_random_equal_size_scattered_group", "matched_random_all_patches",
        *SPATIAL_RANDOM_POSITION_ARMS,
    },
    "distribution_layer": {"native", "zero_dose", *LAYER_ARM_BLOCKS},
    "operator_rank": {"native", "zero_dose", *RANK_ARM_RANKS},
}
EDIT_SITES = {"predictor_visual", "block_condition", "block_output"}


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _is_git_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(c in "0123456789abcdef" for c in value)


def validate_frozen_protocol(protocol: dict) -> None:
    if protocol.get("schema_version") != 1 or protocol.get("status") != "frozen":
        raise ValueError("Intervention protocol must be schema v1 with status=frozen")
    category = protocol.get("category")
    if category not in CATEGORY_ARMS:
        raise ValueError(f"Unknown intervention category: {category}")
    if protocol.get("fit_split") != "fit" or protocol.get("evaluation_split") != "development":
        raise ValueError("Intervention protocols must fit on fit and evaluate on development")
    for field in ("manifest_sha256", "checkpoint_sha256", "fit_receipt_sha256"):
        if not _is_sha256(protocol.get(field)):
            raise ValueError(f"Protocol requires a lowercase SHA256: {field}")
    if not _is_git_sha(protocol.get("vendor_commit")):
        raise ValueError("Protocol requires the pinned vendor Git commit")
    if (not isinstance(protocol.get("tasks"), list) or not protocol["tasks"] or
            any(not isinstance(task, str) or not task for task in protocol["tasks"])):
        raise ValueError("Frozen protocol requires a nonempty task list")
    if len(protocol["tasks"]) != len(set(protocol["tasks"])):
        raise ValueError("Frozen protocol task names must be unique")
    for field in ("hypothesis", "dose_budget", "frozen_at"):
        if not protocol.get(field):
            raise ValueError(f"Frozen protocol is missing {field}")
    arms = protocol.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ValueError("Frozen protocol requires an arm registry")
    names = [arm.get("name") for arm in arms]
    if len(names) != len(set(names)) or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Arm names must be unique nonempty strings")
    missing = CATEGORY_ARMS[category] - set(names)
    if missing:
        raise ValueError(f"Frozen {category} protocol is missing arms: {sorted(missing)}")
    extra = set(names) - CATEGORY_ARMS[category]
    if extra:
        raise ValueError(f"Frozen {category} protocol has unregistered arms: {sorted(extra)}")
    if next(arm for arm in arms if arm["name"] == "native").get("edits", []):
        raise ValueError("Native arm cannot contain an edit")
    for arm in arms:
        edits = arm.get("edits", [])
        if not isinstance(edits, list):
            raise ValueError(f"Arm edits must be a list: {arm['name']}")
        if arm["name"] != "native" and not edits:
            raise ValueError(f"Non-native arm has no registered edit: {arm['name']}")
        for edit in edits:
            site = edit.get("site")
            horizon = edit.get("horizon")
            scale = edit.get("scale")
            if site not in EDIT_SITES:
                raise ValueError(f"Unknown edit site: {site}")
            if not isinstance(horizon, int) or not 1 <= horizon <= 6:
                raise ValueError("Edit horizon must be an integer in 1..6")
            if not isinstance(edit.get("tensor"), str) or not edit["tensor"]:
                raise ValueError("Every edit requires an operator tensor key")
            if not isinstance(scale, (int, float)) or not math.isfinite(scale):
                raise ValueError("Every edit requires a finite numeric scale")
            if site == "predictor_visual" and "block" in edit:
                raise ValueError("predictor_visual edits do not name a predictor block")
            if site != "predictor_visual" and (
                    not isinstance(edit.get("block"), int) or not 0 <= edit["block"] < 6):
                raise ValueError("Block edits require a block index in 0..5")
            if site != "block_output" and any(key in edit for key in ("token_start", "token_end")):
                raise ValueError("Only block_output edits can declare a token slice")
            if site == "block_output":
                for key in ("token_start", "token_end"):
                    if key in edit and edit[key] is not None and not isinstance(edit[key], int):
                        raise ValueError("Block-output token bounds must be integers or null")
    zero_edits = next(arm for arm in arms if arm["name"] == "zero_dose")["edits"]
    if any(edit["scale"] != 0. for edit in zero_edits):
        raise ValueError("zero_dose edits must use exactly zero scale")

    def hook_key(edit):
        return (
            edit["site"], edit["horizon"], edit.get("block"),
            edit.get("token_start"), edit.get("token_end"),
        )

    active_sites = {
        hook_key(edit) for arm in arms if arm["name"] not in {"native", "zero_dose"}
        for edit in arm["edits"]
    }
    zero_sites = {hook_key(edit) for edit in zero_edits}
    if active_sites - zero_sites:
        raise ValueError("zero_dose does not exercise every registered hook location")

    if category == "distribution_layer":
        active_scopes = set()
        operator_ranks = set()
        for arm in arms:
            if arm["name"] not in LAYER_ARM_BLOCKS:
                continue
            edits = arm["edits"]
            if any(edit["site"] == "predictor_visual" for edit in edits):
                raise ValueError("Layer-distribution arms require block-local edit sites")
            actual_blocks = {edit["block"] for edit in edits}
            expected_blocks = LAYER_ARM_BLOCKS[arm["name"]]
            if actual_blocks != expected_blocks:
                raise ValueError(
                    f"Layer arm {arm['name']} requires zero-indexed blocks "
                    f"{sorted(expected_blocks)}; found {sorted(actual_blocks)}"
                )
            active_scopes.update(
                (edit["site"], edit["horizon"], edit.get("token_start"),
                 edit.get("token_end"))
                for edit in edits
            )
            operator_ranks.add(arm.get("operator_rank"))
        if len(active_scopes) != 1:
            raise ValueError(
                "Layer-distribution arms must share one edit site, horizon, and spatial scope"
            )
        if len(operator_ranks) != 1 or not all(
                isinstance(rank, int) and not isinstance(rank, bool) and rank > 0
                for rank in operator_ranks):
            raise ValueError(
                "Layer-distribution arms require one shared positive operator_rank"
            )
        budget = protocol["dose_budget"]
        if budget.get("energy_rule") != "equal_total_delivered_squared_l2_per_arm":
            raise ValueError("Layer distribution requires equal total delivered squared L2 per arm")
        if budget.get("capacity_rule") != "fixed_total_direct_sum_rank_per_arm":
            raise ValueError("Layer distribution requires fixed total direct-sum rank per arm")
        if budget.get("random_control_rule") != "same_support_rank_spectrum_and_energy":
            raise ValueError("Layer random controls must match support, rank, spectrum, and energy")

    if category == "operator_rank":
        rank_scopes = set()
        for arm in arms:
            if arm["name"] not in RANK_ARM_RANKS:
                continue
            expected_rank = RANK_ARM_RANKS[arm["name"]]
            if arm.get("operator_rank") != expected_rank:
                raise ValueError(
                    f"Rank arm {arm['name']} requires operator_rank={expected_rank}"
                )
            rank_scopes.update(
                (edit["site"], edit["horizon"], edit.get("block"),
                 edit.get("token_start"), edit.get("token_end"))
                for edit in arm["edits"]
            )
        if len(rank_scopes) != 1:
            raise ValueError(
                "Operator-rank arms must share one edit site, block, horizon, and spatial scope"
            )
        budget = protocol["dose_budget"]
        if budget.get("energy_rule") != "equal_total_delivered_squared_l2_across_ranks":
            raise ValueError("Operator-rank arms require equal total delivered squared L2")
        if budget.get("random_control_rule") != "same_support_rank_spectrum_and_energy":
            raise ValueError("Rank random controls must match support, rank, spectrum, and energy")
    contrasts = protocol.get("primary_contrasts")
    if not isinstance(contrasts, list) or not contrasts:
        raise ValueError("Frozen protocol requires predeclared primary contrasts")
    for contrast in contrasts:
        if (not contrast.get("name") or contrast.get("candidate") not in names or
                contrast.get("control") not in names or contrast["candidate"] == contrast["control"]):
            raise ValueError("Invalid primary contrast")


def validate_operator_bank(bank: dict, protocol_sha256: str, selected_meta: list[dict]) -> None:
    if bank.get("schema_version") != 1 or bank.get("protocol_sha256") != protocol_sha256:
        raise ValueError("Operator bank is not bound to the frozen protocol")
    rows = bank.get("rows")
    if not isinstance(rows, dict):
        raise ValueError("Operator bank rows must be a mapping")
    for meta in selected_meta:
        key = window_key(meta)
        row = rows.get(key)
        if not isinstance(row, dict):
            raise ValueError(f"Operator bank is missing selected window: {key}")
        if row.get("trajectory_id") != meta["trajectory_id"] or row.get("start") != meta["start"]:
            raise ValueError(f"Operator bank row identity mismatch: {key}")
        if row.get("lineage_group") != meta.get("lineage_group"):
            raise ValueError(f"Operator bank row lineage mismatch: {key}")
        if row.get("split") != "development":
            raise ValueError(f"Operator bank row is not development-only: {key}")
        if not isinstance(row.get("tensors", {}), dict):
            raise ValueError(f"Operator bank row tensors must be a mapping: {key}")
    if not isinstance(bank.get("global_tensors", {}), dict):
        raise ValueError("Operator bank global_tensors must be a mapping")


def window_key(meta: dict) -> str:
    return f"{meta['trajectory_id']}:{meta['start']}"


@dataclass
class CompiledEdit:
    site: str
    horizon: int
    block: int | None
    token_start: int | None
    token_end: int | None
    delta: torch.Tensor
    delivered_l2: list[float] | None = None
    applications: int = 0
    realized_l2: torch.Tensor | None = None


def _resolve_tensor(bank: dict, row: dict, name: str) -> torch.Tensor:
    value = row.get("tensors", {}).get(name)
    if value is None:
        value = bank.get("global_tensors", {}).get(name)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Missing tensor value in operator bank: {name}")
    if value.requires_grad:
        raise ValueError(f"Operator tensor unexpectedly requires gradients: {name}")
    return value.detach().cpu()


def compile_edits(protocol: dict, bank: dict, meta: list[dict], device: torch.device) -> list[CompiledEdit]:
    """Compile all window×arm edits into dense, arm-batched tensors."""
    arms = protocol["arms"]
    effective_batch = len(meta) * len(arms)
    groups: dict[tuple, list[torch.Tensor | None]] = {}
    for window_index, metadata in enumerate(meta):
        row = bank["rows"][window_key(metadata)]
        for arm_index, arm in enumerate(arms):
            output_index = window_index * len(arms) + arm_index
            for edit in arm.get("edits", []):
                key = (
                    edit["site"], edit["horizon"], edit.get("block"),
                    edit.get("token_start"), edit.get("token_end"),
                )
                slots = groups.setdefault(key, [None] * effective_batch)
                tensor = _resolve_tensor(bank, row, edit["tensor"])
                factor = float(edit["scale"])
                if "gate" in edit:
                    gate = _resolve_tensor(bank, row, edit["gate"])
                    if gate.numel() != 1:
                        raise ValueError(f"Edit gate must be scalar: {edit['gate']}")
                    factor *= float(gate.item())
                value = tensor * factor
                if slots[output_index] is not None:
                    if slots[output_index].shape != value.shape:
                        raise ValueError("Edits sharing a hook site have incompatible shapes")
                    value = slots[output_index] + value
                slots[output_index] = value
    compiled = []
    for key, slots in groups.items():
        template = next((value for value in slots if value is not None), None)
        if template is None:
            raise ValueError("Compiled edit group has no tensor")
        if any(value is not None and value.shape != template.shape for value in slots):
            raise ValueError("Operator tensors differ in shape within a hook site")
        staged = torch.stack([
            torch.zeros_like(template) if value is None else value for value in slots
        ])
        delta = staged.to(device, non_blocking=True)
        # This is a dense reduction over the same arm-batched tensor consumed by
        # the predictor. Keeping it on the CPU dominated full-scale execution
        # through thread-pool and memory-bandwidth contention across GPU shards.
        delivered_l2 = delta.double().flatten(1).norm(dim=1).cpu().tolist()
        compiled.append(CompiledEdit(*key, delta=delta, delivered_l2=delivered_l2))
    return compiled


class PredictorIntervention:
    """Apply dense arm-batched edits at registered predictor sites."""

    def __init__(self, predictor: torch.nn.Module, edits: list[CompiledEdit], expected_horizons: int = 6):
        self.predictor = predictor
        self.blocks = predictor.predictor_blocks
        if len(self.blocks) != 6:
            raise ValueError(f"Pinned intervention contract requires six predictor blocks; found {len(self.blocks)}")
        self.edits = edits
        self.expected_horizons = expected_horizons
        self.horizon = 0
        self.handles = []

    def __enter__(self):
        self.handles.append(self.predictor.register_forward_pre_hook(self._predictor_input))
        for index, block in enumerate(self.blocks):
            self.handles.append(block.register_forward_pre_hook(
                lambda module, args, kwargs, index=index: self._block_input(index, args, kwargs),
                with_kwargs=True,
            ))
            self.handles.append(block.register_forward_hook(
                lambda module, args, output, index=index: self._block_output(index, output)
            ))
        return self

    @staticmethod
    def _replace(target: torch.Tensor, delta: torch.Tensor, label: str) -> torch.Tensor:
        if target.shape != delta.shape:
            raise ValueError(f"{label} shape mismatch: target={tuple(target.shape)} delta={tuple(delta.shape)}")
        return target + delta.to(dtype=target.dtype)

    def _matching(self, site: str, block: int | None = None) -> list[CompiledEdit]:
        return [
            edit for edit in self.edits
            if edit.site == site and edit.horizon == self.horizon and edit.block == block
        ]

    def _predictor_input(self, module, args):
        del module
        self.horizon += 1
        if len(args) != 3:
            raise ValueError("Pinned predictor must receive visual, action, and proprio inputs")
        matching = self._matching("predictor_visual")
        if not matching:
            return None
        visual, actions, proprio = args
        changed = visual.clone()
        for edit in matching:
            changed[:, -1] = self._replace(
                changed[:, -1], edit.delta, f"predictor_visual/H{self.horizon}")
            edit.applications += 1
        return changed, actions, proprio

    def _block_input(self, block: int, args, kwargs):
        matching = self._matching("block_condition", block)
        if not matching:
            return None
        if len(args) < 2:
            raise ValueError("Pinned predictor block must receive x and condition inputs")
        x, condition, *tail = args
        changed = condition.clone()
        for edit in matching:
            changed[:, -1] = self._replace(
                changed[:, -1], edit.delta, f"block_condition/P{block}/H{self.horizon}")
            edit.applications += 1
        return (x, changed, *tail), kwargs

    def _block_output(self, block: int, output):
        matching = self._matching("block_output", block)
        if not matching:
            return output
        if not isinstance(output, torch.Tensor):
            raise ValueError("Pinned predictor block output must be one tensor")
        changed = output.clone()
        for edit in matching:
            if edit.token_start is None and edit.token_end is None:
                changed = self._replace(
                    changed, edit.delta, f"block_output/P{block}/H{self.horizon}")
            else:
                token_slice = slice(edit.token_start, edit.token_end)
                before = changed[:, token_slice]
                replacement = self._replace(
                    before, edit.delta,
                    f"block_output/P{block}/H{self.horizon}/tokens",
                )
                edit.realized_l2 = (replacement - before).float().flatten(1).norm(dim=1)
                changed[:, token_slice] = replacement
            edit.applications += 1
        return changed

    def __exit__(self, exc_type, exc, traceback):
        del exc, traceback
        for handle in self.handles:
            handle.remove()
        if exc_type is None:
            if self.horizon != self.expected_horizons:
                raise RuntimeError(
                    f"Expected {self.expected_horizons} predictor calls, observed {self.horizon}")
            missed = [
                (edit.site, edit.horizon, edit.block) for edit in self.edits
                if edit.applications != 1
            ]
            if missed:
                raise RuntimeError(f"Registered edits did not execute exactly once: {missed}")
