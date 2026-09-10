"""Exact H1/H2 common-prefix reuse inside one H3 response-probe construction.

No cross-CEM, cross-candidate, precision, batch-layout or intervention change.
This is whole-predictor prefix memoization, not an attention KV cache. Equality
checks and immutable snapshots are intentional safeguards; measure net speed.
"""
from __future__ import annotations

import torch

from .support_operator import prepare_support
from .planning_support import PlanningSupportIntervention


def snapshot(value):
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided or any(s == 0 and n > 1 for s, n in zip(value.stride(), value.shape)):
            raise ValueError("Unsupported overlapping/layout cache tensor")
        result = torch.empty_strided(value.shape, value.stride(), device=value.device, dtype=value.dtype)
        result.copy_(value)
        return result
    if isinstance(value, tuple):
        return tuple(snapshot(v) for v in value)
    if isinstance(value, list):
        return [snapshot(v) for v in value]
    if isinstance(value, dict):
        return {k: snapshot(v) for k, v in value.items()}
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ValueError("Unsupported predictor argument/output type")


def identical(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, torch.Tensor):
        return (a.shape == b.shape and a.stride() == b.stride() and a.dtype == b.dtype and
                a.device == b.device and torch.equal(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(identical(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(identical(x, y) for x, y in zip(a, b))
    return a == b


class PrefixMemo:
    """Only wrap one native-six-step + fixed H3 probe call sequence."""
    def __init__(self, predictor):
        self.predictor = predictor
        self.calls = self.hits = self.misses = self.unsupported = 0
        self.cache = {}

    def __enter__(self):
        if any(module.training for module in self.predictor.modules()) or torch.is_grad_enabled():
            raise ValueError("Prefix caching requires eval mode and disabled autograd")
        self.original = self.predictor.forward
        self.versions = [(name, p._version) for name, p in self.predictor.named_parameters()]
        self.predictor.forward = self.forward
        return self

    def forward(self, *args, **kwargs):
        self.calls += 1
        horizon = (self.calls - 1) % 6 + 1
        if horizon > 2:
            return self.original(*args, **kwargs)
        inputs = (args, kwargs)
        old = self.cache.get(horizon)
        if old is not None and identical(inputs, old[0]):
            self.hits += 1
            # Do not hand a downstream in-place operation the cached storage.
            return snapshot(old[1])
        self.misses += 1
        self.cache.pop(horizon, None)
        try:
            before = snapshot(inputs)
        except ValueError:
            self.unsupported += 1
            return self.original(*args, **kwargs)
        result = self.original(*args, **kwargs)
        # Refuse caching if forward changes its inputs or layout is unsupported.
        if identical(inputs, before):
            try:
                self.cache[horizon] = (before, snapshot(result))
            except ValueError:
                self.unsupported += 1
        return result

    def __exit__(self, kind, exc, traceback):
        self.predictor.forward = self.original
        self.cache.clear()
        if kind is None and (self.calls % 6 or self.versions !=
                [(name, p._version) for name, p in self.predictor.named_parameters()]):
            raise ValueError("Incomplete six-step probe sequence or predictor weights changed")


@torch.inference_mode()
def cached_prepare_support(backend, context, actions, protocol, bank, counters=None):
    cfg = protocol["support_operator"]
    if (protocol["category"] != "operator_rank" or cfg["response_horizon"] != 6 or
            cfg["edit_horizon"] != 3 or actions.shape[0] != 6):
        raise ValueError("Prefix reuse is validated only for fixed H3/H6 rank probes")
    if any(module._forward_hooks or module._forward_pre_hooks for module in backend.predictor.modules()):
        raise ValueError("No external hooks may depend on omitted prefix block calls")
    cpu_rng = torch.get_rng_state()
    gpu_rng = torch.cuda.get_rng_state(backend.device) if backend.device.type == "cuda" else None
    with PrefixMemo(backend.predictor) as memo:
        result = prepare_support(backend, context, actions, protocol, bank)
    if not torch.equal(cpu_rng, torch.get_rng_state()) or (gpu_rng is not None and
            not torch.equal(gpu_rng, torch.cuda.get_rng_state(backend.device))):
        raise ValueError("Response construction unexpectedly consumed RNG")
    if counters is not None:
        counters.append({"logical_predictor_calls": memo.calls, "cached_prefix_calls": memo.hits,
            "computed_predictor_calls": memo.calls - memo.hits, "prefix_misses": memo.misses,
            "unsupported_layouts": memo.unsupported})
    return result


class CachedPlanningSupportIntervention(PlanningSupportIntervention):
    """Opt-in engineering path; experiment use requires the GPU parity receipt."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix_counters = []

    def _prepare_support(self, backend, context, actions):
        return cached_prepare_support(backend, context, actions, self.protocol, self.bank,
                                      self.prefix_counters)
