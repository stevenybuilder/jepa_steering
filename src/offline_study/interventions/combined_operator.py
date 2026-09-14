"""The planned additive coupling/rank recipe and its fixed drop-one checks.

Reuse native-derived source edits exactly; do not recompute a support operator on
an already coupled rollout or move an edit to an outcome-selected layer.
"""
from __future__ import annotations

import torch

from offline_study.interventions.interventions import CompiledEdit, compile_edits, window_key
from offline_study.interventions.support_operator import prepare_support


# name, source coupling arm, source rank arm. Order is part of the freeze.
RECIPE_ARMS = (
    ("native", "native", "native"),
    ("zero_dose", "zero_dose", "zero_dose"),
    ("coupling_only", "joint_equal_standardized_energy", "native"),
    ("rank4_only", "native", "rank4"),
    ("combined", "joint_equal_standardized_energy", "rank4"),
    ("combined_rank1", "joint_equal_standardized_energy", "rank1"),
    ("matched_random_coupling", "matched_random_equal_standardized_energy", "native"),
    ("matched_random_rank4", "native", "matched_random_rank4"),
    ("matched_random_combined", "matched_random_equal_standardized_energy", "matched_random_rank4"),
    ("matched_random_combined_rank1", "matched_random_equal_standardized_energy", "matched_random_rank1"),
)
PAIRS = (
    ("combined", "native"), ("combined", "matched_random_combined"),
    ("combined", "coupling_only"), ("combined", "rank4_only"),
    ("combined", "combined_rank1"),
    ("coupling_only", "native"), ("coupling_only", "matched_random_coupling"),
    ("rank4_only", "native"), ("rank4_only", "matched_random_rank4"),
    ("combined_rank1", "native"), ("zero_dose", "native"),
    ("combined_rank1", "matched_random_combined_rank1"),
    ("matched_random_coupling", "native"), ("matched_random_rank4", "native"),
    ("matched_random_combined", "native"),
    ("matched_random_combined_rank1", "native"),
)


def remap_edits(edits, source_names, column, batch):
    indices = [source_names.index(row[column]) for row in RECIPE_ARMS]
    result = []
    for edit in edits:
        if edit.delta.shape[0] != batch * len(source_names):
            raise ValueError("Source edit arm coverage changed")
        fields = edit.delta.reshape(batch, len(source_names), *edit.delta.shape[1:])
        delta = fields[:, indices].flatten(0, 1).contiguous()
        result.append(CompiledEdit(edit.site, edit.horizon, edit.block, edit.token_start, edit.token_end,
            delta, delta.double().flatten(1).norm(dim=1).cpu().tolist()))
    return result


def prepare_combined(backend, context, actions, metadata, coupling_protocol, coupling_bank, rank_protocol, rank_bank, return_sources=False):
    if coupling_protocol["category"] != "vision_action_coupling" or rank_protocol["category"] != "operator_rank":
        raise ValueError("Only the registered coupling/rank composition is implemented")
    # Both donors are constructed from the SAME unedited input and recorded
    # actions. prepare_support retains ALL source rank probes/degeneracy checks.
    support, diagnostics = prepare_support(backend, context, actions, rank_protocol, rank_bank)
    # Source global tensors are unchanged. Membership of this NEW planned stage
    # is verified by combined_contract, not by rewriting the old 29-row bank.
    static_bank = {"global_tensors": coupling_bank["global_tensors"],
                   "rows": {window_key(row): {"tensors": {}} for row in metadata}}
    coupling = compile_edits(coupling_protocol, static_bank, metadata, backend.device)
    batch = actions.shape[1]
    result = remap_edits(coupling, [a["name"] for a in coupling_protocol["arms"]], 1, batch)
    result += remap_edits(support, [a["name"] for a in rank_protocol["arms"]], 2, batch)
    keys = [(e.site, e.horizon, e.block, e.token_start, e.token_end) for e in result]
    if len(keys) != len(set(keys)):
        raise ValueError("Overlapping source hooks need an explicit composition rule")
    if return_sources:
        return result, diagnostics, {1: (coupling, [a["name"] for a in coupling_protocol["arms"]]),
                                     2: (support, [a["name"] for a in rank_protocol["arms"]])}
    return result, diagnostics
