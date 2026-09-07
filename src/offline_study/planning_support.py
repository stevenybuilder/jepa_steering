"""Transfer unchanged H6 support corrections to the official two-context planner.

The offline target is H6-specific. At shortened horizons, return the true native
rollout: do not fabricate an H5 target or append fictitious candidate actions.
This adapter does not select a candidate or authorize prospective outcomes.
"""
from __future__ import annotations

from dataclasses import replace

import torch

from .backends import JepaBackend
from .interventions import PredictorIntervention
from .planning_intervention import static_edits
from .support_operator import prepare_support


TRANSFER_POLICY = {
    "planning_context": 2,
    "input_context_frames": 1,
    "response_horizon": 6,
    "edit_horizon": 3,
    "short_horizon_policy": "true native; no support or coupling edit below H6",
    "native_shadow": "official planning unroll on unchanged candidate actions",
    "target_and_fit": "unchanged BF16-primary bank evaluated in strict FP32",
    "common_degeneracy": "retain all source rank probes and all source rank arms",
    "candidate_chunk_size": 8,
    "chunking_scope": "response-operator construction only; actual native and edited forecasts retain the full candidate batch",
    "candidate_reordering_or_padding": False,
    "outcome_dependent_execution": False,
}


def context_values(context, batch, start, stop):
    """Materialize candidate identities before response-probe replication."""
    if not 0 <= start < stop <= batch:
        raise ValueError("Invalid disjoint candidate slice")
    result = {}
    for key in ("visual", "proprio"):
        value = context[key]
        if value.shape[0] not in (1, batch) or value.shape[1] != 1:
            raise ValueError("Require the official single observed-frame planning context")
        result[key] = value.expand(batch, *value.shape[1:])[start:stop].contiguous()
    return result


def select_context(context, batch, start, stop):
    from tensordict import TensorDict
    # Validate layout, but preserve the official singleton broadcast/strides.
    # Materializing a singleton before unroll can change FP32 kernel selection.
    values = context_values(context, batch, start, stop)
    return TensorDict({k: context[k] if context[k].shape[0] == 1 else v
                       for k, v in values.items()}, batch_size=[])


class SupportRolloutBackend:
    """Keep the shared initial context singleton during candidate/probe expansion."""
    def __init__(self, backend):
        self.backend, self.device, self.predictor = backend, backend.device, backend.predictor

    def predict(self, context, actions):
        return self.backend.predict(context, actions)

    def expand_context(self, context, repeats):
        if all(context[k].shape[0] == 1 for k in ("visual", "proprio")):
            return context  # Upstream unroll broadcasts to the action batch itself.
        return self.backend.expand_context(context, repeats)


def selected_support_fields(edits, names, arm, batch):
    if names.count(arm) != 1:
        raise ValueError("Unknown frozen rank arm")
    index = names.index(arm)
    result = []
    for edit in edits:
        if edit.delta.shape[0] != batch * len(names):
            raise ValueError("Source candidate-by-arm layout changed")
        delta = edit.delta.reshape(batch, len(names), *edit.delta.shape[1:])[:, index].contiguous()
        result.append(replace(edit, delta=delta,
            delivered_l2=delta.double().flatten(1).norm(dim=1).cpu().tolist(), applications=0, realized_l2=None))
    return result


class PlanningSupportIntervention:
    def __init__(self, backend, protocol, bank, arm, coupling=None):
        if type(backend) is not JepaBackend or backend.model.ctxt_window != 2:
            raise ValueError("Require the original two-context planning backend")
        if backend.precision != "float32" or backend.allow_tf32:
            raise ValueError("The standalone planner is fixed at strict FP32")
        if protocol["category"] != "operator_rank" or arm not in {a["name"] for a in protocol["arms"]}:
            raise ValueError("Only unchanged registered rank arms are supported")
        config = protocol["support_operator"]
        if config["response_horizon"] != 6 or config["edit_horizon"] != 3:
            raise ValueError("This target is defined only at H6 with the edit at H3")
        if coupling is not None and (arm in ("native", "zero_dose") or
                coupling[2] not in ("joint_equal_standardized_energy", "matched_random_equal_standardized_energy")):
            raise ValueError("Only the fixed equal-budget combined components are supported")
        self.backend, self.protocol, self.bank, self.arm, self.coupling = backend, protocol, bank, arm, coupling
        self.calls, self.energy = 0, []

    @torch.no_grad()
    def __call__(self, context, act_suffix=None, **kwargs):
        if act_suffix is None or kwargs:
            raise ValueError("Require explicit planning actions with the frozen default unroll options")
        horizon, batch, _ = act_suffix.shape
        if not 1 <= horizon <= 6 or batch < 1:
            raise ValueError("Invalid planning action shape")
        self.calls += 1
        if horizon < 6 or self.arm in ("native", "zero_dose"):
            # A genuine native model call, not a zero-filled active hook path.
            result = self.backend.predict(context, act_suffix)
            self.energy.append({"horizon": horizon, "candidates": batch, "edited_candidates": 0,
                "requested_squared_l2_sum": 0., "realized_squared_l2_sum": 0., "response_probe_rollouts": 0})
            return result
        from tensordict import TensorDict
        # Probe construction can be bounded by candidate slices. The ACTUAL
        # forecast must keep the official full candidate batch: GPU GEMM choices
        # can otherwise change FP32 predictions even without an intervention.
        staged = {}
        record = {"horizon": horizon, "candidates": batch, "edited_candidates": 0,
                  "requested_squared_l2_sum": 0., "realized_squared_l2_sum": 0.,
                  "response_probe_rollouts": 0, "native_shadow_rollouts": 0,
                  "final_forecast_candidate_batch": batch, "final_forecasts_chunked": False}
        size = TRANSFER_POLICY["candidate_chunk_size"]
        for start in range(0, batch, size):
            stop = min(start + size, batch)
            z = select_context(context, batch, start, stop)
            actions = act_suffix[:, start:stop].contiguous()
            all_edits, diagnostics = prepare_support(SupportRolloutBackend(self.backend), z, actions, self.protocol, self.bank)
            fields = selected_support_fields(all_edits, [a["name"] for a in self.protocol["arms"]], self.arm, stop - start)
            if self.coupling is not None:
                coupling_protocol, coupling_bank, coupling_arm = self.coupling
                fields += static_edits(coupling_protocol, coupling_bank, coupling_arm, stop - start, horizon, self.backend.device)
            keys = [(e.site, e.block, e.horizon, e.token_start, e.token_end) for e in fields]
            if len(set(keys)) != len(keys):
                raise ValueError("Combined planning hooks overlap without a composition rule")
            for key, field in zip(keys, fields):
                staged.setdefault(key, []).append(field)
            record["response_probe_rollouts"] += sum(r["response_probe_rollouts"] for r in diagnostics)
            record["native_shadow_rollouts"] += sum(r["native_shadow_rollouts"] for r in diagnostics)
        fields = []
        for values in staged.values():
            delta = torch.cat([e.delta for e in values], dim=0).contiguous()
            if len(delta) != batch:
                raise ValueError("Missing or duplicated constructed candidate edits")
            fields.append(replace(values[0], delta=delta,
                delivered_l2=delta.double().flatten(1).norm(dim=1).cpu().tolist(), applications=0, realized_l2=None))
        active = torch.stack([e.delta.flatten(1).ne(0).any(1) for e in fields]).any(0)
        record["edited_candidates"] = int(active.sum())
        # Decide zero-treatment dispatch BEFORE forecasting. Both paths retain
        # the full population; this is not an after-the-fact parity correction.
        native = self.backend.predict(context, act_suffix) if not active.all() else None
        if active.any():
            with PredictorIntervention(self.backend.predictor, fields):
                changed = self.backend.predict(context, act_suffix)
            if native is None:
                result = changed
            else:
                result = TensorDict({key: torch.where(active.reshape(1, batch, *([1] * (changed[key].ndim - 2))),
                    changed[key], native[key]) for key in ("visual", "proprio")}, batch_size=[])
            for edit in fields:
                if edit.realized_l2 is None or not torch.isfinite(edit.realized_l2).all():
                    raise ValueError("Missing/nonfinite delivered planning energy")
                record["requested_squared_l2_sum"] += sum(x*x for x in edit.delivered_l2)
                record["realized_squared_l2_sum"] += float(edit.realized_l2.double().square().sum())
        else:
            result = native
        if any(not torch.isfinite(result[key]).all() for key in ("visual", "proprio")):
            raise ValueError("Nonfinite planning predictions; do not discard candidates")
        self.energy.append(record)
        return result
