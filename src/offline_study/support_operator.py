"""Fixed PCA-coordinate H6 correction for the registered support/rank sweeps."""
from __future__ import annotations

import itertools
import torch

from .interventions import CompiledEdit, PredictorIntervention, LAYER_MECHANISM_ARM_BLOCKS


SEED = 2026090720
CATEGORIES = ("operator_rank", "distribution_layer", "distribution_spatial")


class NativeFieldCapture:
    def __init__(self, predictor):
        self.predictor, self.horizon, self.values, self.handles = predictor, 0, {}, []

    def __enter__(self):
        if len(self.predictor.predictor_blocks) != 6:
            raise ValueError("Expected the six-block pinned predictor")
        self.handles.append(self.predictor.register_forward_pre_hook(self._input))
        for block, module in enumerate(self.predictor.predictor_blocks):
            self.handles.append(module.register_forward_hook(
                lambda module, args, output, block=block: self._output(block, output)))
        return self

    def _input(self, module, args):
        self.horizon += 1

    def _output(self, block, output):
        if self.horizon == 3:
            if block in self.values or output.ndim != 3 or output.shape[1] < 256:
                raise ValueError("Unexpected native field shape or duplicate hook")
            self.values[block] = output[:, -256:].detach().clone()

    def __exit__(self, kind, exc, traceback):
        for handle in self.handles:
            handle.remove()
        if kind is None and (self.horizon != 6 or set(self.values) != set(range(6))):
            raise ValueError("Capture did not see each H3 field exactly once")


def principal_basis(values, rank):
    """Small sample-space Gram avoids a dense feature-by-feature covariance."""
    if values.ndim != 2 or values.shape[0] <= rank or not torch.isfinite(values).all():
        raise ValueError("PCA requires enough finite rows")
    mean = values.mean(0)
    centered = values - mean
    gram = centered @ centered.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram.double())
    eigenvalues = eigenvalues[-rank:].flip(0)
    vectors = eigenvectors[:, -rank:].flip(1).to(values.dtype)
    if eigenvalues[-1] <= max(1e-12, float(eigenvalues[0]) * 1e-8):
        raise ValueError("Registered PCA rank is numerically deficient")
    raw = (vectors.T @ centered) / eigenvalues.sqrt().to(values.dtype)[:, None]
    basis = torch.linalg.qr(raw.T, mode="reduced").Q.T
    # Fix signs without outcomes or an unstable eigenvector sign convention.
    pivots = basis.abs().argmax(1)
    signs = basis[torch.arange(rank, device=basis.device), pivots].sign()
    basis = basis * signs[:, None]
    return mean, basis


def fit_target(features, errors):
    mean, basis = principal_basis(features, 3)
    score = (features - mean) @ basis.T
    scale = score.std(0, unbiased=False).clamp_min(1e-12)
    design = torch.cat([torch.ones_like(score[:, :1]), score / scale], 1)
    weight = torch.linalg.pinv(design.double()) @ errors.double()
    return {"mean": mean, "basis": basis, "scale": scale, "weight": weight.float()}


def predict_target(features, target):
    score = ((features - target["mean"]) @ target["basis"].T) / target["scale"]
    return torch.cat([torch.ones_like(score[:, :1]), score], 1) @ target["weight"]


def spatial_positions(field, errors):
    _, error_basis = principal_basis(errors, 8)
    scores = (errors - errors.mean(0)) @ error_basis.T
    centered = field - field.mean(0)
    covariance = centered.flatten(1).T @ scores / len(field)
    strength = covariance.reshape(256, field.shape[-1], 8).square().sum((1, 2))
    regions = [list((r + dr) * 16 + c + dc for dr in range(4) for dc in range(4))
               for r in range(13) for c in range(13)]
    best_region = max(range(len(regions)), key=lambda i: float(strength[regions[i]].sum()))
    ranked = sorted(range(256), key=lambda i: (-float(strength[i]), i))
    scattered = [[(r + 4 * dr) * 16 + c + 4 * dc for dr in range(4) for dc in range(4)]
                 for r in range(4) for c in range(4)]
    best_scattered = max(range(16), key=lambda i: float(strength[scattered[i]].sum()))
    generator = torch.Generator().manual_seed(SEED)
    random_region = int(torch.randint(len(regions), (), generator=generator))
    return {"one_patch": ranked[:1], "contiguous_group": regions[best_region],
            "equal_size_scattered_group": scattered[best_scattered], "all_patches": list(range(256)),
            "random_position_one_patch": [int(torch.randint(256, (), generator=generator))],
            "random_position_contiguous_group": regions[random_region],
            "random_position_equal_size_scattered_group": scattered[int(torch.randint(16, (), generator=generator))]}


def supported_basis(fields, blocks, positions, rank):
    data = fields[:, blocks][:, :, positions]
    _, basis = principal_basis(data.flatten(1), rank)
    return {"blocks": blocks, "positions": positions,
            "basis": basis.reshape(rank, len(blocks), len(positions), fields.shape[-1])}


def random_basis(reference, seed):
    basis = reference["basis"]
    generator = torch.Generator().manual_seed(seed)
    value = torch.randn((basis[0].numel(), len(basis)), generator=generator, dtype=torch.float32)
    value = torch.linalg.qr(value.to(basis.device), mode="reduced").Q.T.reshape_as(basis)
    return {**reference, "basis": value}


def fit_bases(fields, errors):
    positions = spatial_positions(fields[:, 3], errors)
    all_positions = list(range(256))
    rank = supported_basis(fields, [3], all_positions, 8)
    bases = {"rank": rank, "rank_random": random_basis(rank, SEED + 1)}
    maps = {"operator_rank": {}}
    for k in (1, 4, 8):
        maps["operator_rank"][f"rank{k}"] = ("rank", k)
        maps["operator_rank"][f"matched_random_rank{k}"] = ("rank_random", k)
    maps["distribution_layer"] = {}
    for index, (name, blocks) in enumerate(LAYER_MECHANISM_ARM_BLOCKS.items()):
        if name == "single_block3":
            key, random_key = "rank", "rank_random"
        else:
            key, random_key = name, "random_" + name
            bases[key] = supported_basis(fields, sorted(blocks), all_positions, 1)
            bases[random_key] = random_basis(bases[key], SEED + 10 + index)
        maps["distribution_layer"][name] = (key, 1)
        maps["distribution_layer"]["matched_random_" + name] = (random_key, 1)
    maps["distribution_spatial"] = {}
    for index, (name, selected) in enumerate(positions.items()):
        if name == "all_patches":
            key, random_key = "rank", "rank_random"
        else:
            key, random_key = "spatial_" + name, "random_spatial_" + name
            bases[key] = supported_basis(fields, [3], selected, 1)
            if not name.startswith("random_position_"):
                bases[random_key] = random_basis(bases[key], SEED + 30 + index)
        maps["distribution_spatial"][name] = (key, 1)
        if not name.startswith("random_position_"):
            maps["distribution_spatial"]["matched_random_" + name] = (random_key, 1)
    return bases, maps, positions


def expand_basis(spec, count):
    basis = spec["basis"][:count]
    value = basis.new_zeros(count, 6, 256, basis.shape[-1])
    for j, block in enumerate(spec["blocks"]):
        value[:, block, spec["positions"]] = basis[:, j]
    return value


def compile_fields(fields, active_blocks):
    """fields: window x arm x block x patch x channel, already energy matched."""
    result = []
    for block in sorted(active_blocks):
        delta = fields[:, :, block].flatten(0, 1).contiguous()
        result.append(CompiledEdit("block_output", 3, block, -256, None, delta,
                                   delta.flatten(1).norm(dim=1).cpu().tolist()))
    return result


def solve_correction(response, target, basis):
    gram = response @ response.transpose(-1, -2)
    damping = .01 * gram.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-18)
    regularized = gram + damping[:, None, None] * torch.eye(len(basis), device=gram.device)
    rhs = (response @ target[:, :, None]).squeeze(-1)
    coefficient = torch.linalg.solve(regularized.double(), rhs.double()).float()
    delta = coefficient @ basis.flatten(1)
    return delta.reshape(len(target), *basis.shape[1:])


@torch.inference_mode()
def prepare_support(backend, context, actions, protocol, bank):
    config = protocol["support_operator"]
    if "_support_device" not in bank:
        def stage(value):
            if isinstance(value, torch.Tensor):
                return value.to(backend.device)
            if isinstance(value, dict):
                return {k: stage(v) for k, v in value.items()}
            return value
        bank["_support_device"] = stage(bank["support_bank"])
    fitted = bank["_support_device"]
    mapping = fitted["maps"][protocol["category"]]
    with NativeFieldCapture(backend.predictor) as capture:
        backend.predict(context, actions)
    target = predict_target(capture.values[3].flatten(1), fitted["target"])
    batch = actions.shape[1]
    required = {}
    for key, rank in mapping.values():
        required[key] = max(required.get(key, 0), rank)
    expanded = {key: expand_basis(fitted["bases"][key], rank) for key, rank in required.items()}
    probes = [(key, j, sign) for key, rank in required.items()
              for j in range(rank) for sign in (-1, 1)]
    outputs = {}
    radius = config["response_radius"]
    for start in range(0, len(probes), config["probe_chunk_size"]):
        chunk = probes[start:start + config["probe_chunk_size"]]
        delta = torch.stack([expanded[key][j] * sign * radius for key, j, sign in chunk])
        blocks = set(itertools.chain.from_iterable(fitted["bases"][key]["blocks"] for key, _, _ in chunk))
        edits = compile_fields(delta[None].expand(batch, -1, -1, -1, -1), blocks)
        with PredictorIntervention(backend.predictor, edits):
            predictions = backend.predict(backend.expand_context(context, len(chunk)),
                                          actions.repeat_interleave(len(chunk), dim=1))
        final = predictions["visual"][6].reshape(batch, len(chunk), -1)
        for index, probe in enumerate(chunk):
            outputs[probe] = final[:, index].clone()
    responses = {key: torch.stack([(outputs[(key, j, 1)] - outputs[(key, j, -1)]) / (2 * radius)
                                  for j in range(rank)], 1) for key, rank in required.items()}
    del outputs
    deltas = {name: solve_correction(responses[key][:, :rank], target, expanded[key][:rank])
              for name, (key, rank) in mapping.items()}
    norms = torch.stack([value.flatten(1).norm(dim=1) for value in deltas.values()], 1)
    if not torch.isfinite(norms).all():
        raise ValueError("Nonfinite support correction; no observations silently excluded")
    eligible = (norms > 1e-10).all(1)
    shape = (-1, 1, 1, 1)
    for index, name in enumerate(deltas):
        deltas[name] *= (eligible.float() * config["delivered_l2"] / norms[:, index].clamp_min(1e-10)).reshape(shape)
    template = next(iter(deltas.values()))
    deltas["native"], deltas["zero_dose"] = torch.zeros_like(template), torch.zeros_like(template)
    names = [arm["name"] for arm in protocol["arms"]]
    fields = torch.stack([deltas[name] for name in names], 1)
    blocks = set(itertools.chain.from_iterable(fitted["bases"][key]["blocks"] for key in required))
    diagnostics = [{"common_energy_edit_eligible": float(value),
                    "response_probe_rollouts": len(probes), "native_shadow_rollouts": 1}
                   for value in eligible.cpu().tolist()]
    return compile_fields(fields, blocks), diagnostics
