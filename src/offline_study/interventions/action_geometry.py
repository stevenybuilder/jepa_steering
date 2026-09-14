"""Four-donor interpolation with an omitted recorded-action center."""
from __future__ import annotations

import torch

from offline_study.interventions.interventions import CompiledEdit, PredictorIntervention


ANCHORS = (-1., -.5, .5, 1.)
GEOMETRY_ARMS = ("equal_anchor_linear", "cubic", "projected_cubic", "reflected_curvature")


def central_estimates(donors: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Port of archived estimator, batched as [window, four donors, ...].

    No central activation is accepted. At this symmetric coordinate cubic equals
    quadratic least squares; it does not uniquely identify third-order dynamics.
    Invalid endpoint lines are flagged, never dropped from population summaries.
    """
    if donors.ndim < 3 or donors.shape[1] != 4 or not donors.is_floating_point():
        raise ValueError("Expected floating-point [batch, four donors, ...]")
    if not torch.isfinite(donors).all():
        raise ValueError("Nonfinite donor activation")
    original_dtype = donors.dtype
    work = donors.double()
    origin = work[:, 0]
    centered = work - origin[:, None]
    linear = origin + centered.mean(1)
    cubic = origin + (2. / 3.) * (centered[:, 1] + centered[:, 2]) - centered[:, 3] / 6.
    chord = centered[:, 3]
    chord_square = chord.square().flatten(1).sum(1)
    path_square = centered.square().flatten(2).sum(2).max(1).values
    valid = chord_square > torch.maximum(path_square * 1e-12, chord_square.new_tensor(1e-20))
    coefficient = ((cubic - origin) * chord).flatten(1).sum(1) / chord_square.clamp_min(1e-20)
    shape = (-1,) + (1,) * (origin.ndim - 1)
    projected = origin + coefficient.reshape(shape) * chord
    # A closed endpoint path supplies no line. Make its entire intervention pair
    # a zero-dose observation and preserve its flag in the full population.
    projected = torch.where(valid.reshape(shape), projected, origin)
    values = {"equal_anchor_linear": linear, "cubic": cubic,
              "projected_cubic": projected, "reflected_curvature": 2 * projected - cubic}
    return {name: value.to(original_dtype) for name, value in values.items()}, valid


class BlockCapture:
    def __init__(self, predictor, block=3, horizon=3):
        self.predictor, self.block, self.target = predictor, block, horizon
        self.horizon, self.value = 0, None
        self.handles = []

    def __enter__(self):
        if len(self.predictor.predictor_blocks) != 6:
            raise ValueError("Expected six predictor blocks")
        self.handles = [self.predictor.register_forward_pre_hook(self._input),
                        self.predictor.predictor_blocks[self.block].register_forward_hook(self._output)]
        return self

    def _input(self, module, args):
        self.horizon += 1

    def _output(self, module, args, output):
        if self.horizon == self.target:
            if self.value is not None or output.ndim != 3 or output.shape[1] < 256:
                raise ValueError("Unexpected native predictor output shape/call count")
            self.value = output[:, -256:].detach().clone()

    def __exit__(self, exc_type, exc, traceback):
        for handle in self.handles:
            handle.remove()
        if exc_type is None and (self.horizon != 6 or self.value is None):
            raise RuntimeError("Capture must observe H3 exactly once in a six-step rollout")


def donor_actions(actions, direction, radius):
    """Perturb only H3 in normalized units; preserve all recorded recipients."""
    if actions.ndim != 3 or actions.shape[0] != 6 or direction.shape != actions.shape[-1:]:
        raise ValueError("Invalid H6 action sequence or action direction")
    if not 0 < radius <= 1 or not torch.isfinite(direction).all():
        raise ValueError("Invalid fixed normalized action radius/direction")
    expanded = actions.repeat_interleave(5, dim=1)
    coordinates = actions.new_tensor((*ANCHORS, 0.)).repeat(actions.shape[1])
    expanded[2] += radius * coordinates[:, None] * direction[None]
    return expanded


@torch.inference_mode()
def collect_donors(backend, context, actions, direction, radius):
    with BlockCapture(backend.predictor) as capture:
        predictions = backend.predict(backend.expand_context(context, 5),
                                      donor_actions(actions, direction, radius))
    batch = actions.shape[1]
    # Fit and evaluate interpolation in FP32 even when predictor inference uses BF16.
    captured = capture.value.float().reshape(batch, 5, *capture.value.shape[1:])
    native_predictions = {key: value[:, 4::5].clone() for key, value in predictions.items()}
    return captured[:, :4], captured[:, 4], native_predictions


def matched_deltas(estimates, native, valid, random_direction, dose):
    residuals = {name: value - native for name, value in estimates.items()}
    norms = {name: value.flatten(1).norm(dim=1) for name, value in residuals.items()}
    eligible = valid.clone()
    for norm in norms.values():
        eligible &= torch.isfinite(norm) & (norm > 1e-10)
    shape = (-1,) + (1,) * (native.ndim - 1)
    shared_dose = eligible.to(native.dtype) * dose
    deltas = {name: value * (shared_dose / norms[name].clamp_min(1e-10)).reshape(shape)
              for name, value in residuals.items()}
    deltas["matched_random"] = random_direction[None] * shared_dose.reshape(shape)
    deltas["zero_dose"] = torch.zeros_like(native)
    deltas["native"] = torch.zeros_like(native)
    return deltas, eligible


def compile_dynamic(deltas, arm_names):
    delta = torch.stack([deltas[name] for name in arm_names], dim=1).flatten(0, 1)
    return [CompiledEdit("block_output", 3, 3, -256, None, delta,
                         delta.double().flatten(1).norm(dim=1).cpu().tolist())]


@torch.inference_mode()
def prepare_geometry(backend, context, actions, protocol, bank):
    """Capture donors once and retain raw fidelity separately from equal-dose edits."""
    config = protocol["geometry"]
    if (config["anchors"] != list(ANCHORS) or config["omitted_coordinate"] != 0.
            or config["normalized_action_radius"] != .1):
        raise ValueError("Geometry runtime differs from the registered anchor contract")
    direction = bank["global_tensors"]["action_direction"].to(backend.device)
    random = bank["global_tensors"]["random_direction"].to(backend.device)
    donors, native, native_predictions = collect_donors(
        backend, context, actions, direction, config["normalized_action_radius"])
    estimates, valid = central_estimates(donors)
    deltas, eligible = matched_deltas(estimates, native, valid, random, config["delivered_l2"])
    names = [arm["name"] for arm in protocol["arms"]]
    compiled = compile_dynamic(deltas, names)
    # Separate raw reconstruction pass: these residuals have unequal norms and
    # cannot be used as the budget-matched efficacy measurements.
    raw_names = ["native", *GEOMETRY_ARMS]
    raw_deltas = {name: value - native for name, value in estimates.items()}
    raw_deltas["native"] = torch.zeros_like(native)
    raw_edits = compile_dynamic(raw_deltas, raw_names)
    with PredictorIntervention(backend.predictor, raw_edits):
        raw_predictions = backend.predict(backend.expand_context(context, len(raw_names)),
                                          actions.repeat_interleave(len(raw_names), dim=1))
    labels, columns = ["endpoint_line_valid", "common_energy_edit_eligible"], [valid.float(), eligible.float()]
    for name in GEOMETRY_ARMS:
        labels.append(name + "/omitted_activation_mse")
        columns.append((estimates[name] - native).square().flatten(1).mean(1))
        for modality in ("visual", "proprio"):
            for horizon in (3, 6):
                forecast = raw_predictions[modality][horizon, raw_names.index(name)::len(raw_names)]
                difference = forecast - native_predictions[modality][horizon]
                labels.append(name + f"/raw_native_{modality}_fidelity_mse_h{horizon}")
                columns.append(difference.square().flatten(1).mean(1))
    diagnostics = [dict(zip(labels, values, strict=True))
                   for values in torch.stack(columns, dim=1).cpu().tolist()]
    return compiled, diagnostics
