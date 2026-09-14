"""Transfer a frozen static coupling arm to the official planning unroll.

No dataset or outcome is read here. Episode access and candidate selection are
separate launch gates. Dynamic H6 response operators require a different adapter;
they must not silently inherit an invented rule for shortened planning horizons.
"""
import torch

from offline_study.interventions.interventions import CompiledEdit, PredictorIntervention


def static_edits(protocol, bank, arm_name, batch, horizon, device):
    if protocol["category"] != "vision_action_coupling":
        raise ValueError("Dynamic geometry/support planning transfer is not validated by this adapter")
    arms = [arm for arm in protocol["arms"] if arm["name"] == arm_name]
    if len(arms) != 1 or batch < 1 or not 1 <= horizon <= 6:
        raise ValueError("Unknown frozen arm or invalid planning dimensions")
    grouped = {}
    for edit in arms[0]["edits"]:
        if "gate" in edit:
            raise ValueError("Static planning adapter cannot introduce temporal routing")
        if edit["horizon"] > horizon:
            continue  # The registered edit time is outside this truncated rollout.
        if float(edit["scale"]) == 0.:
            continue  # A true no-op must not change striding by cloning a context.
        tensor = bank["global_tensors"][edit["tensor"]]
        if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
            raise ValueError("Invalid frozen operator tensor")
        value = tensor.detach().to(device).float() * float(edit["scale"])
        key = (edit["site"], edit["horizon"], edit.get("block"), edit.get("token_start"), edit.get("token_end"))
        grouped[key] = grouped.get(key, torch.zeros_like(value)) + value
    result = []
    for key, value in grouped.items():
        delta = value[None].expand(batch, *value.shape).contiguous()
        energy = delta.double().flatten(1).norm(dim=1).cpu().tolist()
        result.append(CompiledEdit(*key, delta=delta, delivered_l2=energy))
    return result


class StaticPlanningIntervention:
    """Callable compatible with the unchanged official CEMPlanner.unroll API."""
    def __init__(self, backend, protocol, bank, arm):
        self.backend, self.protocol, self.bank, self.arm = backend, protocol, bank, arm
        self.calls = 0
        self.energy = []

    @torch.no_grad()
    def __call__(self, context, act_suffix=None, **kwargs):
        if act_suffix is None:
            raise ValueError("Planning actions are required")
        horizon, batch, _ = act_suffix.shape
        edits = static_edits(self.protocol, self.bank, self.arm, batch, horizon, self.backend.device)
        if not edits:
            with self.backend.autocast():
                result = self.backend.model.unroll(context.clone(), act_suffix=act_suffix, **kwargs)
            self.energy.append({"horizon": horizon, "candidates": batch,
                                "requested_squared_l2_mean": 0., "realized_squared_l2_mean": 0.})
            self.calls += 1
            return result
        with self.backend.autocast(), PredictorIntervention(self.backend.predictor, edits, expected_horizons=horizon):
            result = self.backend.model.unroll(context.clone(), act_suffix=act_suffix, **kwargs)
        if any(edit.realized_l2 is None for edit in edits):
            raise ValueError("Missing delivered planning energy measurement")
        # Scalar summaries only, not all candidate activation tensors or outcomes.
        self.energy.append({"horizon": horizon, "candidates": batch,
            "requested_squared_l2_mean": sum(sum(x*x for x in e.delivered_l2) / batch for e in edits),
            "realized_squared_l2_mean": sum(e.realized_l2.square().mean().item() for e in edits)})
        self.calls += 1
        return result
