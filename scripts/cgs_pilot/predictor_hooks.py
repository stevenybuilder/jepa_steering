#!/usr/bin/env python3
"""Forward-hook capture and activation-patching primitives for the JEPA-WM
AdaLN predictor (``app.plan_common.models.AdaLN_vit.VisionTransformerAdaLN``).

Verified module layout (jepa-wms @ 13cf1d9c, ``jepa_wm_droid``):

``EncPredWM`` (hubconf model)
  ``.model``            VideoWM
    ``.predictor``      VisionTransformerAdaLN
      ``.predictor_embed``          Linear(1024 -> 1024), applied to [B,T,1,16,16,D]
      ``.predictor_blocks.{0..11}`` FWAdaLNBlock
        ``.norm1`` ``.attn`` (RoPEAttention) ``.norm2`` ``.mlp`` (fc1/fc2) ``.adaLN_modulation``
      ``.predictor_norm`` ``.predictor_proj``

Token layout inside a block: ``x`` is ``[B, T*256, D]`` with ``T`` = frames in the
rolling window (1 at imagined step 0, then ``ctxt_window``=2), tokens ordered frame
by frame; the LAST 256 tokens belong to the most recent frame, whose block outputs
become the next predicted latent.  There is no action token: the action enters
only through ``adaLN_modulation(z)`` with ``z = action_encoder(actions)`` of shape
``[B, T, D]``, broadcast over the 256 tokens of each frame (shift/scale/gate for
attention and MLP).  A block-causal attention mask (``local_window_time=3``) means
earlier-frame tokens never see later frames, so per-step capture of the last-frame
tokens loses nothing.

``EncPredWM.unroll`` calls ``VideoWM.forward_pred`` -> ``predictor.forward`` once per
imagined step, so a forward-pre-hook on the predictor counts imagined steps.

Hook points per block ``L`` (site id ``L{L:02d}.{hook}``):

- ``resid_pre``   block input residual
- ``attn_out``    attention output before AdaLN gate
- ``mlp_in``      MLP input (modulated norm2 output)
- ``mlp_out``     MLP output before AdaLN gate
- ``resid_post``  block output residual
- ``adaln``       AdaLN modulation vector ``[B, T, 6D]`` (token group = last frame, 1 "token")

plus the special site ``embed`` (``predictor_embed`` output, i.e. block-0 input
before flattening).

Everything here is model-agnostic given ``predictor``, ``n_spatial_tokens`` and an
``unroll`` callable, so the CPU tests use a tiny fake predictor.

Patches may target a token subset (``(index, values)``) so that spatial token
groups (egg / gripper / corridor) can be written without replacing the whole
residual stream; whole-256-token residual patches amount to scene replacement and
are kept as a reference only (arXiv:2309.16042 on patching best practices;
arXiv:2511.04638 on off-manifold edits; arXiv:2606.27510 on multi-site
interaction of patches).
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import torch


HOOK_POINTS = ("resid_pre", "attn_out", "mlp_in", "mlp_out", "resid_post", "adaln")
COMPONENT_HOOKS = ("attn_out", "mlp_out")
EMBED_SITE = "embed"
_HOOK_RANK = {"resid_pre": 0, "attn_out": 1, "mlp_in": 2, "mlp_out": 3, "resid_post": 5, "adaln": 4}

ZERO_SHOT_SCOPE = (
    "Checkpoint jepa_wm_droid is DROID-only (vendored training config "
    "droid_4fpcs_fps4_r256_dv3vitl_asp1_pred_AdaLN_depth12_noprop_repro_2roll_4n.yaml: datasets: [DROID], "
    "override_datasets: true); the RoboCasa contact stimuli are zero-shot for this model."
)


def site_order_key(site: "Site", step: int = 0) -> tuple[int, int, int]:
    """Temporal-then-depth ordering with resid_pre < attn_out < mlp_in < mlp_out < resid_post
    and resid_post(L) == resid_pre(L+1); the embed site precedes block 0."""

    if site.layer < 0:
        return (step, 0, -1)
    if site.hook == "resid_post":
        return (step, site.layer + 1, 0)
    return (step, site.layer, _HOOK_RANK[site.hook])
_SITE_RE = re.compile(r"^L(\d{2})\.(resid_pre|attn_out|mlp_in|mlp_out|resid_post|adaln)$")


@dataclass(frozen=True)
class Site:
    layer: int  # -1 for the embed site
    hook: str

    @property
    def site_id(self) -> str:
        return EMBED_SITE if self.layer < 0 else f"L{self.layer:02d}.{self.hook}"

    @staticmethod
    def parse(site_id: str) -> "Site":
        if site_id == EMBED_SITE:
            return Site(-1, EMBED_SITE)
        match = _SITE_RE.match(site_id)
        if not match:
            raise ValueError(f"bad site id {site_id!r}")
        return Site(int(match.group(1)), match.group(2))

    @property
    def group_tokens(self) -> str:
        return "mod" if self.hook == "adaln" else "last"


def all_sites(n_layers: int, hooks: Iterable[str] = HOOK_POINTS, include_embed: bool = True) -> list[Site]:
    sites = [Site(-1, EMBED_SITE)] if include_embed else []
    for layer in range(n_layers):
        for hook in hooks:
            if hook not in HOOK_POINTS:
                raise ValueError(f"unknown hook {hook}")
            sites.append(Site(layer, hook))
    return sites


def n_layers_of(predictor: torch.nn.Module) -> int:
    return len(predictor.predictor_blocks)


def _group_slice(tensor: torch.Tensor, site: Site, n_spatial: int) -> tuple[torch.Tensor, int]:
    """Return ``(view [B, n_group, d], n_group)`` of the last-frame token group."""

    batch = tensor.shape[0]
    flat = tensor.reshape(batch, -1, tensor.shape[-1])
    n_group = 1 if site.hook == "adaln" else n_spatial
    return flat[:, -n_group:], n_group


def _module_for(predictor: torch.nn.Module, site: Site) -> tuple[torch.nn.Module, str]:
    """Return ``(module, kind)`` where kind is ``pre`` (forward-pre-hook on args[0])
    or ``post`` (forward hook on output)."""

    if site.layer < 0:
        return predictor.predictor_embed, "post"
    block = predictor.predictor_blocks[site.layer]
    return {
        "resid_pre": (block, "pre"),
        "attn_out": (block.attn, "post"),
        "mlp_in": (block.mlp, "pre"),
        "mlp_out": (block.mlp, "post"),
        "resid_post": (block, "post"),
        "adaln": (block.adaLN_modulation, "post"),
    }[site.hook]


class PredictorRecorder:
    """Capture last-frame activations at sites for every imagined step.

    ``acts[site_id][step]`` -> tensor ``[B, n_group, d]`` (detached; on ``store_device``).
    """

    def __init__(
        self,
        predictor: torch.nn.Module,
        sites: Iterable[Site],
        n_spatial: int,
        store_device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ) -> None:
        self.predictor = predictor
        self.sites = list(sites)
        self.n_spatial = n_spatial
        self.store_device = torch.device(store_device)
        self.dtype = dtype
        self.acts: dict[str, dict[int, torch.Tensor]] = {s.site_id: {} for s in self.sites}
        self.step = -1
        self._handles: list[Any] = []

    def _store(self, site: Site, tensor: torch.Tensor) -> None:
        view, _ = _group_slice(tensor, site, self.n_spatial)
        out = view.detach()
        if self.dtype is not None:
            out = out.to(self.dtype)
        self.acts[site.site_id][self.step] = out.to(self.store_device).clone()

    def __enter__(self) -> "PredictorRecorder":
        def count_step(module, args, kwargs):
            self.step += 1

        self._handles.append(self.predictor.register_forward_pre_hook(count_step, with_kwargs=True))
        for site in self.sites:
            module, kind = _module_for(self.predictor, site)
            if kind == "pre":
                self._handles.append(
                    module.register_forward_pre_hook(
                        lambda m, args, kwargs, site=site: self._store(site, args[0]), with_kwargs=True
                    )
                )
            else:
                self._handles.append(
                    module.register_forward_hook(lambda m, args, out, site=site: self._store(site, out))
                )
        return self

    def __exit__(self, *exc: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @property
    def n_steps(self) -> int:
        return self.step + 1


class PredictorPatcher:
    """Replace last-frame activations at ``(site_id, step)`` with supplied tensors.

    ``patches[(site_id, step)]`` -> either a tensor ``[B, n_group, d]`` (whole token
    group of the last frame) or a tuple ``(index, values)`` with ``index`` a 1-D
    LongTensor of token positions within the last frame and ``values``
    ``[B, len(index), d]``.  Values are already the donor content to write, so
    every control is expressed by what the caller puts here.
    """

    def __init__(self, predictor: torch.nn.Module, patches: dict[tuple[str, int], torch.Tensor], n_spatial: int) -> None:
        self.predictor = predictor
        self.patches = dict(patches)
        self.n_spatial = n_spatial
        self.step = -1
        self.applied: list[tuple[str, int]] = []
        self._handles: list[Any] = []

    def _apply(self, site: Site, tensor: torch.Tensor) -> torch.Tensor:
        key = (site.site_id, self.step)
        donor = self.patches.get(key)
        if donor is None:
            return tensor
        view, n_group = _group_slice(tensor, site, self.n_spatial)
        out = tensor.clone()
        flat = out.reshape(out.shape[0], -1, out.shape[-1])
        if isinstance(donor, tuple):
            index, values = donor
            index = torch.as_tensor(index, dtype=torch.long, device=out.device)
            if values.shape[0] != view.shape[0] or values.shape[1] != index.numel() or values.shape[2] != view.shape[2]:
                raise ValueError(f"patch values {tuple(values.shape)} incompatible with index {index.numel()} at {key}")
            if index.numel() == 0:
                self.applied.append(key)
                return out
            if int(index.max()) >= n_group or int(index.min()) < 0:
                raise ValueError(f"token index out of range for group size {n_group} at {key}")
            start = flat.shape[1] - n_group
            flat[:, start + index] = values.to(dtype=out.dtype, device=out.device)
        else:
            if tuple(donor.shape) != tuple(view.shape):
                raise ValueError(f"patch shape {tuple(donor.shape)} != site shape {tuple(view.shape)} at {key}")
            flat[:, -n_group:] = donor.to(dtype=out.dtype, device=out.device)
        self.applied.append(key)
        return out

    def __enter__(self) -> "PredictorPatcher":
        def count_step(module, args, kwargs):
            self.step += 1

        self._handles.append(self.predictor.register_forward_pre_hook(count_step, with_kwargs=True))
        sites = {Site.parse(site_id) for site_id, _ in self.patches}
        for site in sites:
            module, kind = _module_for(self.predictor, site)
            if kind == "pre":

                def pre_hook(m, args, kwargs, site=site):
                    return (self._apply(site, args[0]), *args[1:]), kwargs

                self._handles.append(module.register_forward_pre_hook(pre_hook, with_kwargs=True))
            else:
                self._handles.append(
                    module.register_forward_hook(lambda m, args, out, site=site: self._apply(site, out))
                )
        return self

    def __exit__(self, *exc: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


# --------------------------------------------------------------------------- #
# Control constructions.  Each returns the tensor to WRITE at the site, so the
# patcher stays a dumb replacement primitive.
# --------------------------------------------------------------------------- #


def norm_matched_random(base: torch.Tensor, donor: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """``base + r`` with ``r`` an isotropic random direction whose Frobenius norm equals
    ``||donor - base||`` over the patched slice (per batch element)."""

    delta = donor - base
    rand = torch.randn(delta.shape, generator=generator, device=delta.device, dtype=delta.dtype)
    dims = tuple(range(1, delta.ndim))
    scale = delta.norm(dim=dims, keepdim=True) / rand.norm(dim=dims, keepdim=True).clamp_min(1e-12)
    return base + rand * scale


def sham(base: torch.Tensor) -> torch.Tensor:
    return base.clone()


def time_shifted(donor_acts: dict[int, torch.Tensor], target_step: int) -> dict[int, torch.Tensor]:
    """Map each candidate source step != target to the donor activation from that step."""

    return {s: a for s, a in donor_acts.items() if s != target_step and a.shape == donor_acts[target_step].shape}


def unrelated_site(site: Site, n_layers: int, offset: int) -> Site:
    """Same hook, different layer: the donor content is written at a wrong layer."""

    if site.layer < 0:
        return Site(offset % n_layers, "resid_pre")
    return Site((site.layer + offset) % n_layers, site.hook)


# --------------------------------------------------------------------------- #
# Running the wrapper with hooks
# --------------------------------------------------------------------------- #


def unroll_from_latents(
    wm: Any,
    z_context: torch.Tensor,
    actions: torch.Tensor,
    recorder: PredictorRecorder | None = None,
    patcher: PredictorPatcher | None = None,
) -> torch.Tensor:
    """Run ``wm.unroll`` and return the predicted latents ``[T, B, ...]`` (context stripped)."""

    stack = contextlib.ExitStack()
    with stack:
        if recorder is not None:
            stack.enter_context(recorder)
        if patcher is not None:
            stack.enter_context(patcher)
        with torch.inference_mode():
            rollout = wm.unroll(z_context, act_suffix=actions)
    return rollout[int(z_context.shape[1]):]


def predictor_of(wm: Any) -> torch.nn.Module:
    return wm.model.predictor


def spatial_tokens_of(wm: Any) -> int:
    return int(wm.grid_size) ** 2
