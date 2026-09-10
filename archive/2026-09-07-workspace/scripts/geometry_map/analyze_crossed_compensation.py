#!/usr/bin/env python3
"""Exact crossed forecast decomposition; opposition is not proof of intent.

Rows are imagined horizons. The four forecasts hold starting context fixed and
cross model intervention (off/on) with selected plan (baseline/edited).
No fitted coordinate readout is required: use native objective-weighted vectors.
"""
from __future__ import annotations

import numpy as np


def native_metric_vectors(visual, proprio, alpha=0.1):
    """Flatten each horizon so squared distance equals native per-step cost.

    Inputs exclude the observed context and contain one selected plan, not a
    candidate batch. Visual/proprio feature layouts may differ across models.
    This is MSE(visual) + alpha*MSE(proprio), NOT pooled-feature MSE.
    """
    visual, proprio = np.asarray(visual, dtype=np.float64), np.asarray(proprio, dtype=np.float64)
    if visual.ndim < 2 or proprio.ndim < 2 or len(visual) != len(proprio):
        raise ValueError("Aligned horizon-first visual and proprio arrays required")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("Nonnegative finite native proprio weight required")
    if not np.isfinite(visual).all() or not np.isfinite(proprio).all():
        raise ValueError("Nonfinite forecast")
    v, p = visual.reshape(len(visual), -1), proprio.reshape(len(proprio), -1)
    if not v.shape[1] or not p.shape[1]:
        raise ValueError("Empty feature axis")
    return np.concatenate((v / np.sqrt(v.shape[1]), p * np.sqrt(alpha / p.shape[1])), axis=1)


def crossed_decomposition(base_base, edit_base, base_edit, edit_edit, goal=None):
    """Separate fixed-action edit, native action-selection effect, interaction.

    base_base = F_0(a_0); edit_base = F_e(a_0)
    base_edit = F_0(a_e); edit_edit = F_e(a_e)
    Their total difference is exactly edit_bias + action_effect + interaction.
    Positive cancellation_projection means action selection opposes edit bias;
    it does NOT identify compensation uniquely (optimizer disruption can too).
    """
    arrays = [np.asarray(v, dtype=np.float64) for v in (base_base, edit_base, base_edit, edit_edit)]
    shape = arrays[0].shape
    if len(shape) != 2 or not all(v.shape == shape for v in arrays) or not all(shape):
        raise ValueError("Four aligned [horizon,feature] forecasts required")
    if not all(np.isfinite(v).all() for v in arrays):
        raise ValueError("Nonfinite crossed forecast")
    bb, eb, be, ee = arrays
    bias, action = eb - bb, be - bb
    interaction = ee - eb - be + bb
    total = ee - bb
    residual = total - (bias + action + interaction)
    scale = max(1.0, *(float(np.max(np.abs(v))) for v in arrays))
    if np.max(np.abs(residual)) > 32 * np.finfo(np.float64).eps * scale:
        raise AssertionError("Crossed decomposition failed numerical identity")
    target = None
    if goal is not None:
        target = np.asarray(goal, dtype=np.float64)
        if target.shape == (shape[1],):
            target = np.broadcast_to(target, shape)
        if target.shape != shape or not np.isfinite(target).all():
            raise ValueError("Goal must match features, optionally repeated by horizon")
    rows = []
    for h in range(len(bb)):
        b2, a2 = float(bias[h] @ bias[h]), float(action[h] @ action[h])
        dot = float(bias[h] @ action[h])
        floor = 1e-12 * max(1.0, float(bb[h] @ bb[h]))
        row = {
            "imagined_step": h + 1,
            "fixed_action_edit_norm": float(np.sqrt(b2)),
            "native_action_selection_effect_norm": float(np.sqrt(a2)),
            "interaction_norm": float(np.linalg.norm(interaction[h])),
            "total_forecast_change_norm": float(np.linalg.norm(total[h])),
            "edit_action_dot": dot,
            "edit_action_cosine": dot / np.sqrt(b2 * a2) if min(b2, a2) > floor else None,
            "cancellation_projection": -dot / b2 if b2 > floor else None,
            "numerically_resolved_edit": b2 > floor,
            "decomposition_max_abs_error": float(np.max(np.abs(residual[h]))),
        }
        if target is not None:
            costs = [float(np.sum((v[h] - target[h]) ** 2)) for v in arrays]
            cbb, ceb, cbe, cee = costs
            row.update(
                base_model_base_plan_cost=cbb,
                edited_model_base_plan_cost=ceb,
                base_model_edited_plan_cost=cbe,
                edited_model_edited_plan_cost=cee,
                edited_model_selection_gain=ceb - cee,
                base_model_selection_gain=cbb - cbe,
                fixed_action_cost_bias=ceb - cbb,
                cost_interaction=cee - ceb - cbe + cbb,
                total_cost_change=cee - cbb,
            )
        rows.append(row)
    return {
        "rows": rows,
        "identity_max_abs_error": float(np.max(np.abs(residual))),
        "interpretation": "Exact crossed finite differences; native objective uses terminal row only. Opposition or apparent edited-model improvement is compatible with compensation but also optimizer disruption; require signed/sham controls, actual action consequences, and new starts.",
        "physical_effect_measured_by_this_function": False,
    }
