#!/usr/bin/env python3
"""Cross-token relational action-Jacobian sonar (design doc "JEPA rep geometry.md",
sections 1, 2.1 and 4/E1): the predictor's differentiable action sensitivity as a
model-native, coordinate-free "ping".

Instead of two coarse candidate actions, we probe the JEPA-WM predictor with the
Jacobian of its one-step latent forecast with respect to the 7-D action,

    J(z, a) = dP_1(z, a) / da            (columns = action dims)

evaluated at each cell's context latent ``z`` and first action ``a``, restricted
to the egg+gripper+corridor token region (masks) and to the full frame.  Per cell
we report ``||J||_F``, the responses ``J a_hat`` to the gentle / aggressive unit
action directions, and the singular spectrum of ``J`` (numerical rank,
participation ratio).

Relational statistics per scene (same ``a``):
``dJ_hazard = J(H1) - J(H0)``, ``dJ_null = J(H0') - J(H0)``,
``relational = dJ_hazard - dJ_null``; Frobenius norms, the ratio
``||dJ_hazard|| / ||J(H0)||``, and cross-scene consistency of the token-pooled
``vec(dJ)`` by leave-one-scene-out projection (``cgs_stats``).  Mixed-derivative
check: the directional derivative of ``J`` along ``delta_h = z(H1) - z(H0)``,
``d^2P/(da dz) . delta_h``, obtained as a small-step central difference of the
forward-mode Jacobian (nested forward-AD levels are not available), compared to
the unit-step finite difference ``dJ_hazard``.

Site mediation: with forward-AD hooks at every site (layer x {attn_out, mlp_out,
resid_post}, token groups) one dual forward pass per action dim yields the
site's action Jacobian ``dh_s/da e_i`` at every site simultaneously; the
downstream sensitivity ``(dP/dh_s)`` is obtained by injecting that tangent at
the site (and nowhere else) in a second dual pass, so
``M_s = (dP/dh_s)(dh_s/da)`` is the part of ``J`` that flows through ``h_s``.
Hazard dependence: ``||M_s(H1) - M_s(H0) - (M_s(H0') - M_s(H0))||`` per site.
For a residual stream this is a cut, so ``M_{resid_post(L)}`` equals the total
``J`` when every action path enters at or before ``L`` and the group is the whole
frame; increments across layers then sum to the total.

Refinements (design doc 2.1): (a) pattern/value split - ``dJ`` recomputed with
the attention probabilities detached (value path only); the difference is the
attention-pattern term (AtP*-style Q/K correction in derivative form); (b)
principal angles between ``col J_g(z_H1)`` and ``col J_g(z_H0)`` (rank <= 7
subspaces; Grassmann distance and top angle) as the primary rotation statistic,
with ``S_F = ||dJ_rel||_F / ||J_g(z_H0)||_F``; (c) the projected statistic
``<dJ_rel (a1 - a0), I_true>`` against the true-future interaction vector of the
cell npz; (d) central-difference checks of the directional derivative along
``a1 - a0`` at eps in {1e-3, 1e-2, 1e-1}; (e) a sign-randomized same-construction
null (per-token sign flips of ``delta_h`` on the egg tokens) next to the isotropic
random null; (f) finite-difference confirmation ``a0 -> a1`` on the same tokens:
per site the self-donor action patch (site activation from the a1 run written
into the a0 run) is compared with the Jacobian prediction ``M_s (a1 - a0)``
(cosine / sign), and at the total level ``P_g(a1) - P_g(a0)`` vs ``J_g (a1 - a0)``.
E1 thresholds: ``S_F >= 0.05``, top principal angle >= 10 deg, max-T p < 0.05 over
sites against the H0', random and sign-randomized nulls.

Statistics (unit = scene): sign-flip over scenes of the relational norm minus a
null from random token-space perturbations of the egg tokens with the norm of
``delta_h``; sign-flip max-T on the LOSO consistency (equivalent to within-scene
hazard-label permutation, which flips the sign of ``dJ``); max-T across sites for
the mediation map; scene-clustered bootstrap CIs.  Step 0 (context -> P_1) is
primary; predictions at steps 2-3 (still differentiated w.r.t. the first
action) are autoregressive.

Outputs ``<out>/jacobian_sonar.json`` (per scene and pooled) and
``<out>/site_mediation.npz``.

References: Sussillo & Barak (2013, Neural Computation) linearisation /
Jacobian analysis of learned recurrent dynamics; Lusch, Kutz & Brunton (2018,
Nature Communications) linear (Koopman) embeddings of latent dynamics; Zhang &
Nanda (arXiv:2309.16042) activation-patching best practices; COAST
(arXiv:2605.17144) donor-free subspace interventions; arXiv:2510.00845 on
mediation-score variance (band, not index, claims).
TODO: cite the design doc "JEPA rep geometry.md" once written.

``--domain driving`` (protocol v0.8): region = hazard U corridor tokens and route =
corridor tokens via the egg-name aliases of ``token_groups.py`` (``egg`` ->
``hazard``, ``gripper`` -> empty, ``corridor`` -> corridor); hazard level 3 (cone
on the same path) adds ``dJ_object = J(H3) - J(H0)`` and the identity contrast
``dJ_hazard - dJ_object`` (``identity_frobenius_region``, ``S_F_route_identity``,
principal angles ``object``) next to the H0' null.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.autograd.forward_ad as fwAD

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cgs_stats import cluster_bootstrap, cv_projection, json_safe, sign_flip_maxt  # noqa: E402
from localize_interaction import IDENTITY_CELLS, NULL_CELLS, encode_cell, identity_pairs, identity_status, merge_manifests, null_factor_pairs, null_factor_status, select_pairs  # noqa: E402
from model_action_sensitivity import load_model  # noqa: E402
from predictor_hooks import ZERO_SHOT_SCOPE, Site, _module_for, all_sites, n_layers_of, predictor_of, site_order_key, spatial_tokens_of  # noqa: E402
from protocol import CELL_ORDER, OBJECT_HAZARD, canonical_json, sha256_file  # noqa: E402
from token_groups import DOMAINS, CellGroups, alias_note, load_cell_groups, set_domain, union  # noqa: E402


MEDIATION_HOOKS = ("attn_out", "mlp_out", "resid_post")
REGION_GROUPS = ("egg", "gripper", "corridor")


_DETACH_PATTERN = [False]


@contextlib.contextmanager
def detach_attention_pattern():
    """Within this context the attention probabilities carry no forward-mode tangent
    (value path only); the difference to the full Jacobian is the attention-pattern term."""

    _DETACH_PATTERN[0] = True
    try:
        yield
    finally:
        _DETACH_PATTERN[0] = False


def _math_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    """Pure-tensor scaled dot-product attention (supports forward-mode AD; fused kernels do not)."""

    d = query.shape[-1]
    s = (1.0 / d**0.5) if scale is None else scale
    scores = (query @ key.transpose(-2, -1)) * s
    L, S = query.shape[-2], key.shape[-2]
    if is_causal:
        causal = torch.ones(L, S, dtype=torch.bool, device=query.device).tril()
        scores = scores.masked_fill(~causal, float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask
    weights = torch.softmax(scores, dim=-1)
    if _DETACH_PATTERN[0]:
        weights = fwAD.unpack_dual(weights).primal
    if dropout_p > 0.0:
        weights = torch.nn.functional.dropout(weights, p=dropout_p)
    return weights @ value


@contextlib.contextmanager
def _sdpa_math():
    """Route ``F.scaled_dot_product_attention`` through the pure-tensor implementation
    (``sdpa_kernel(MATH)`` does not cover the CPU flash kernel, which has no forward-AD rule)."""

    F = torch.nn.functional
    original = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = _math_sdpa
    try:
        yield
    finally:
        F.scaled_dot_product_attention = original


# --------------------------------------------------------------------------- #
# Forward-AD hooks
# --------------------------------------------------------------------------- #


class TangentHooks:
    """Capture the forward-mode tangent of the last-frame tokens at sites, and/or
    inject a tangent at one site (primal kept, upstream tangent discarded)."""

    def __init__(self, predictor: torch.nn.Module, sites: list[Site], n_spatial: int, inject: tuple[str, torch.Tensor] | None = None) -> None:
        self.predictor = predictor
        self.sites = sites
        self.n_spatial = n_spatial
        self.inject = inject
        self.tangents: dict[str, torch.Tensor] = {}
        self.step = -1
        self.target_step = 0
        self._handles: list[Any] = []

    def _capture(self, site: Site, out: torch.Tensor) -> None:
        if self.step != self.target_step:
            return
        tangent = fwAD.unpack_dual(out).tangent
        if tangent is None:
            return
        flat = tangent.reshape(tangent.shape[0], -1, tangent.shape[-1])
        n = 1 if site.hook == "adaln" else self.n_spatial
        self.tangents[site.site_id] = flat[:, -n:].detach().clone()

    def _inject(self, site: Site, out: torch.Tensor) -> torch.Tensor:
        if self.step != self.target_step:
            return out
        primal = fwAD.unpack_dual(out).primal
        tangent = torch.zeros_like(primal)
        flat = tangent.reshape(tangent.shape[0], -1, tangent.shape[-1])
        n = 1 if site.hook == "adaln" else self.n_spatial
        flat[:, -n:] = self.inject[1].to(primal.dtype)
        return fwAD.make_dual(primal, tangent)

    def __enter__(self) -> "TangentHooks":
        def count(module, args, kwargs):
            self.step += 1

        self._handles.append(self.predictor.register_forward_pre_hook(count, with_kwargs=True))
        for site in self.sites:
            module, kind = _module_for(self.predictor, site)
            if kind != "post":
                raise ValueError("tangent capture only at output hooks (attn_out, mlp_out, resid_post, adaln)")
            self._handles.append(module.register_forward_hook(lambda m, a, out, site=site: self._capture(site, out)))
        if self.inject is not None:
            site = Site.parse(self.inject[0])
            module, _ = _module_for(self.predictor, site)
            self._handles.append(module.register_forward_hook(lambda m, a, out, site=site: self._inject(site, out)))
        return self

    def __exit__(self, *exc: Any) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def _predict(wm: Any, z_ctx: torch.Tensor, actions: torch.Tensor, k: int) -> torch.Tensor:
    """Prediction at imagined step ``k`` (0 = P_1 from the context), ``[n_tokens, D]``."""

    rollout = wm.unroll(z_ctx, act_suffix=actions[: k + 1])
    pred = rollout[int(z_ctx.shape[1]) + k]
    return pred.reshape(pred.shape[0], -1, pred.shape[-1])[0]


def jacobian_forward_ad(
    wm: Any,
    predictor: torch.nn.Module,
    z_ctx: torch.Tensor,
    actions: torch.Tensor,
    k: int,
    n_spatial: int,
    capture_sites: list[Site] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """``J`` ``[n_tokens, D, A]`` of the step-k prediction w.r.t. the FIRST action, plus per-site
    tangents ``[A, n_spatial, D]`` at the step-k predictor call."""

    A = actions.shape[-1]
    cols, site_cols = [], {}
    with _sdpa_math():
        for i in range(A):
            tangent = torch.zeros_like(actions)
            tangent[0, :, i] = 1.0
            with fwAD.dual_level():
                a_dual = fwAD.make_dual(actions, tangent)
                hooks = TangentHooks(predictor, capture_sites or [], n_spatial)
                hooks.target_step = k
                with hooks:
                    pred = _predict(wm, z_ctx, a_dual, k)
                cols.append(fwAD.unpack_dual(pred).tangent.detach().clone())
                for sid, tg in hooks.tangents.items():
                    site_cols.setdefault(sid, []).append(tg[0])
    J = torch.stack(cols, dim=-1)
    return J, {sid: torch.stack(v, dim=0) for sid, v in site_cols.items()}


def downstream_response(wm: Any, predictor: torch.nn.Module, z_ctx: torch.Tensor, actions: torch.Tensor, k: int, n_spatial: int, site_id: str, tangent: torch.Tensor) -> torch.Tensor:
    """``(dP_k / dh_site) . tangent`` with ``tangent`` ``[n_spatial, D]`` injected at the site only."""

    with _sdpa_math(), fwAD.dual_level():
        hooks = TangentHooks(predictor, [], n_spatial, inject=(site_id, tangent.unsqueeze(0)))
        hooks.target_step = k
        with hooks:
            pred = _predict(wm, z_ctx, actions, k)
        out = fwAD.unpack_dual(pred).tangent
    return torch.zeros_like(pred) if out is None else out.detach().clone()


# --------------------------------------------------------------------------- #
# Per-cell summaries
# --------------------------------------------------------------------------- #


def spectrum(J: torch.Tensor) -> dict[str, Any]:
    """Singular values of ``J`` reshaped to ``[n_out, A]`` via the A x A Gram."""

    M = J.reshape(-1, J.shape[-1]).double()
    ev = torch.linalg.eigvalsh(M.T @ M).clamp_min(0.0).flip(0)
    s = ev.sqrt()
    s2 = ev
    tol = float(s.max().item()) * max(M.shape) * 1e-7 if s.numel() else 0.0
    return {
        "singular_values": [float(v) for v in s],
        "numerical_rank": int((s > tol).sum().item()),
        "participation_ratio": float((s2.sum() ** 2 / (s2**2).sum().clamp_min(1e-30)).item()),
        "top1_energy_fraction": float((s2[0] / s2.sum().clamp_min(1e-30)).item()) if s2.numel() else float("nan"),
    }


def cell_summary(J: torch.Tensor, region: np.ndarray, gentle: torch.Tensor, aggressive: torch.Tensor) -> dict[str, Any]:
    def summarize(Jx: torch.Tensor) -> dict[str, Any]:
        return {
            "frobenius": float(Jx.norm().item()),
            "response_gentle_norm": float((Jx @ gentle).norm().item()),
            "response_aggressive_norm": float((Jx @ aggressive).norm().item()),
            "response_cosine_gentle_vs_aggressive": float(torch.nn.functional.cosine_similarity((Jx @ gentle).flatten(), (Jx @ aggressive).flatten(), dim=0).item()),
            **spectrum(Jx),
        }

    idx = torch.as_tensor(region, dtype=torch.long, device=J.device)
    return {"full": summarize(J), "region": summarize(J[idx]), "region_size": int(len(region))}


def pooled(J: torch.Tensor, region: np.ndarray) -> np.ndarray:
    idx = torch.as_tensor(region, dtype=torch.long, device=J.device)
    return J[idx].mean(dim=0).flatten().detach().cpu().double().numpy()


def principal_angles(J1: torch.Tensor, J0: torch.Tensor) -> dict[str, Any]:
    """Principal angles (deg) between col(J1) and col(J0), both reshaped to [n_out, A]."""

    def basis(J: torch.Tensor) -> torch.Tensor:  # numerically significant left singular vectors only
        U, S, _ = torch.linalg.svd(J.reshape(-1, J.shape[-1]).double(), full_matrices=False)
        keep = S > S.max() * 1e-6 if S.numel() else S > 0
        return U[:, keep]

    Q1, Q0 = basis(J1), basis(J0)
    if Q1.shape[1] == 0 or Q0.shape[1] == 0:
        return {"angles_deg": [], "top_angle_deg": float("nan"), "grassmann_distance_rad": float("nan"), "chordal_distance": float("nan"), "rank_1": int(Q1.shape[1]), "rank_0": int(Q0.shape[1])}
    s = torch.linalg.svdvals(Q1.T @ Q0).clamp(-1.0, 1.0)
    theta = torch.rad2deg(torch.arccos(s))
    return {
        "angles_deg": [float(v) for v in theta],
        "top_angle_deg": float(theta.max().item()),
        "grassmann_distance_rad": float(torch.deg2rad(theta).norm().item()),
        "chordal_distance": float(torch.sqrt((1 - s**2).sum()).item()),
        "rank_1": int(Q1.shape[1]),
        "rank_0": int(Q0.shape[1]),
    }


def projected_statistic(dJ: torch.Tensor, a_diff: torch.Tensor, I_true: torch.Tensor, region: np.ndarray) -> dict[str, float]:
    """``<dJ (a1 - a0), I_true>`` on the region tokens (raw, cosine, and as a fraction of ||I_true||^2)."""

    idx = torch.as_tensor(region, dtype=torch.long, device=dJ.device)
    resp = (dJ[idx] @ a_diff).flatten().double()
    tgt = I_true[idx].flatten().double()
    inner = float((resp @ tgt).item())
    return {
        "inner": inner,
        "cosine": float((resp @ tgt / (resp.norm() * tgt.norm()).clamp_min(1e-12)).item()),
        "fraction_of_I_true": inner / max(float((tgt @ tgt).item()), 1e-12),
        "response_norm": float(resp.norm().item()),
        "I_true_norm": float(tgt.norm().item()),
    }


def sign_randomized_perturbation(delta: torch.Tensor, egg_tokens: np.ndarray, generator: torch.Generator) -> torch.Tensor:
    """Same-construction null: ``delta_h`` with an independent random sign per egg token."""

    out = torch.zeros_like(delta)
    idx = torch.as_tensor(egg_tokens, dtype=torch.long, device=delta.device)
    signs = (torch.randint(0, 2, (len(egg_tokens), 1), generator=generator, device=delta.device) * 2 - 1).to(delta.dtype)
    out[idx] = delta[idx] * signs
    return out


def central_difference_check(wm: Any, z: torch.Tensor, actions: torch.Tensor, k: int, J: torch.Tensor, direction: torch.Tensor, region: np.ndarray, eps_list: tuple[float, ...]) -> dict[str, Any]:
    """Relative error of the forward-AD directional derivative ``J v`` vs central differences at several eps."""

    v = direction / direction.norm().clamp_min(1e-12)
    idx = torch.as_tensor(region, dtype=torch.long, device=J.device)
    jv = (J @ v)[idx]
    out = {}
    with torch.no_grad():
        for eps in eps_list:
            ap, am = actions.clone(), actions.clone()
            ap[0, 0] += eps * v
            am[0, 0] -= eps * v
            fd = ((_predict(wm, z, ap, k) - _predict(wm, z, am, k)) / (2.0 * eps))[idx]
            out[f"eps_{eps:g}"] = {"relative_error": float(((fd - jv).norm() / jv.norm().clamp_min(1e-12)).item()),
                                   "cosine": float(torch.nn.functional.cosine_similarity(fd.flatten(), jv.flatten(), dim=0).item())}
    return out


def self_donor_site_patch(wm: Any, predictor: torch.nn.Module, z: torch.Tensor, a0: torch.Tensor, a1: torch.Tensor, k: int, n_spatial: int, site_id: str, gidx: np.ndarray | None) -> torch.Tensor:
    """Finite-difference ``a0 -> a1`` through one site: write the site's a1 activation (group tokens) into the a0 run."""

    from predictor_hooks import PredictorPatcher, PredictorRecorder

    site = Site.parse(site_id)
    rec = PredictorRecorder(predictor, [site], n_spatial, store_device=z.device)
    with torch.no_grad(), rec:
        _predict(wm, z, a1, k)
    donor = rec.acts[site_id][k]
    payload = donor if gidx is None or donor.shape[1] == 1 else (torch.as_tensor(gidx, dtype=torch.long, device=donor.device), donor[:, torch.as_tensor(gidx, device=donor.device)])
    patcher = PredictorPatcher(predictor, {(site_id, k): payload}, n_spatial)
    with torch.no_grad(), patcher:
        p_patched = _predict(wm, z, a0, k)
    with torch.no_grad():
        p_base = _predict(wm, z, a0, k)
    return p_patched - p_base


def random_token_perturbation(delta: torch.Tensor, egg_tokens: np.ndarray, generator: torch.Generator) -> torch.Tensor:
    """Random direction supported on the egg tokens with ``||.|| = ||delta||`` (orthogonalised against delta)."""

    out = torch.zeros_like(delta)
    idx = torch.as_tensor(egg_tokens, dtype=torch.long, device=delta.device)
    r = torch.randn((len(egg_tokens), delta.shape[-1]), generator=generator, device=delta.device, dtype=delta.dtype)
    d = delta[idx]
    if d.norm() > 0:
        r = r - (r * d).sum() / (d.norm() ** 2) * d
    r = r / r.norm().clamp_min(1e-12) * delta.norm()
    out[idx] = r
    return out


# --------------------------------------------------------------------------- #
# Per-scene evaluation
# --------------------------------------------------------------------------- #


def evaluate_scene(
    wm: Any,
    predictor: torch.nn.Module,
    n_spatial: int,
    latents: dict[tuple[int, int], dict[str, torch.Tensor]],
    groups: dict[tuple[int, int], CellGroups],
    sites: list[Site],
    site_groups: list[str],
    steps: list[int],
    mediation_steps: list[int],
    generator: torch.Generator,
    hvp_eps: float = 0.1,
    fd_eps: tuple[float, ...] = (1e-3, 1e-2, 1e-1),
    value_path: bool = True,
) -> dict[str, Any]:
    keys = list(latents)
    has_null = all(k in latents for k in NULL_CELLS)
    has_ident = all(k in latents for k in IDENTITY_CELLS)
    D = latents[keys[0]]["z_context"].shape[-1]

    def z_of(k):  # [1, 1, 1, G, G, D] context latent as a plain tensor
        return latents[k]["z_context"].detach().clone()

    def z_flat(k):
        return z_of(k).reshape(-1, D)

    a_of = {k: latents[k]["actions"].detach().clone() for k in keys}  # [T, 1, A]
    gentle = a_of[(0, 0)][0, 0] / a_of[(0, 0)][0, 0].norm().clamp_min(1e-12)
    aggressive = a_of[(0, 1)][0, 0] / a_of[(0, 1)][0, 0].norm().clamp_min(1e-12)

    def region(frame: int) -> np.ndarray:
        return union(*(groups[k].group(frame, g) for k in keys for g in REGION_GROUPS))

    def egg_tokens(frame: int) -> np.ndarray:
        return union(*(groups[k].group(frame, "egg") for k in keys))

    def route(frame: int) -> np.ndarray:
        return union(*(groups[k].group(frame, g) for k in keys for g in ("gripper", "corridor")))

    def I_true_at(k_step: int) -> torch.Tensor:  # true-future interaction vector at the predicted frame, [n_tokens, D]
        f = {k: latents[k]["z_future"][0, k_step].reshape(-1, D).float() for k in CELL_ORDER}
        return f[(1, 1)] - f[(1, 0)] - f[(0, 1)] + f[(0, 0)]

    a_diff = (a_of[(0, 1)][0, 0] - a_of[(0, 0)][0, 0]).detach()
    out: dict[str, Any] = {"cells": {}, "steps": {}, "null_factor": has_null}
    if has_ident:
        out["identity_factor"] = True
    for k_step in steps:
        out_frame = k_step + 1
        in_frame = k_step
        R = region(out_frame)
        G_route = route(out_frame)
        I_true = I_true_at(k_step)
        step_out: dict[str, Any] = {"region_size": int(len(R)), "route_size": int(len(G_route)), "cells": {}, "relational": {}, "mediation": {}}
        capture = sites if k_step in mediation_steps else []
        J: dict[Any, torch.Tensor] = {}
        site_tan: dict[Any, dict[str, torch.Tensor]] = {}
        z_cache: dict[Any, torch.Tensor] = {k: z_of(k) for k in keys}
        a_cache: dict[Any, torch.Tensor] = {k: a_of[k] for k in keys}
        # real cells
        for k in keys:
            J[k], site_tan[k] = jacobian_forward_ad(wm, predictor, z_cache[k], a_cache[k], k_step, n_spatial, capture)
            step_out["cells"][f"h{k[0]}a{k[1]}"] = cell_summary(J[k], R, gentle, aggressive)
        # null cells on z(H0,A): isotropic random delta (||.|| = ||z(H1,A) - z(H0,A)||, egg tokens) and sign-randomized delta_h
        rand_keys = []
        for a in (0, 1):
            delta = z_flat((1, a)) - z_flat((0, a))
            for kind, fn in (("rand", random_token_perturbation), ("sign", sign_randomized_perturbation)):
                dr = fn(delta, egg_tokens(in_frame), generator)
                zr = (z_flat((0, a)) + dr).reshape(z_of((0, a)).shape)
                kr = (kind, a)
                rand_keys.append(kr)
                z_cache[kr], a_cache[kr] = zr, a_of[(0, a)]
                J[kr], site_tan[kr] = jacobian_forward_ad(wm, predictor, zr, a_of[(0, a)], k_step, n_spatial, capture)
                step_out["cells"][f"{kind}_a{a}"] = {**cell_summary(J[kr], R, gentle, aggressive), "delta_norm": float(delta.norm().item())}
        # value-path Jacobians (attention probabilities detached)
        Jv: dict[Any, torch.Tensor] = {}
        if value_path:
            with detach_attention_pattern():
                for k in list(keys) + rand_keys:
                    Jv[k], _ = jacobian_forward_ad(wm, predictor, z_cache[k], a_cache[k], k_step, n_spatial, [])
        # relational per action
        rel_vectors = {}
        for a in (0, 1):
            dJh = J[(1, a)] - J[(0, a)]
            dJr = J[("rand", a)] - J[(0, a)]
            entry = {
                "dJ_hazard_frobenius_region": float(dJh[torch.as_tensor(R, device=dJh.device)].norm().item()),
                "dJ_hazard_frobenius_full": float(dJh.norm().item()),
                "dJ_random_frobenius_region": float(dJr[torch.as_tensor(R, device=dJr.device)].norm().item()),
                "J_h0_frobenius_region": float(J[(0, a)][torch.as_tensor(R, device=dJh.device)].norm().item()),
            }
            entry["ratio_dJ_hazard_over_J_h0"] = entry["dJ_hazard_frobenius_region"] / max(entry["J_h0_frobenius_region"], 1e-12)
            rel = dJh
            if has_null:
                dJn = J[(2, a)] - J[(0, a)]
                rel = dJh - dJn
                entry["dJ_null_frobenius_region"] = float(dJn[torch.as_tensor(R, device=dJn.device)].norm().item())
                entry["relational_frobenius_region"] = float(rel[torch.as_tensor(R, device=rel.device)].norm().item())
                entry["cosine_dJ_hazard_vs_dJ_null"] = float(torch.nn.functional.cosine_similarity(dJh.flatten(), dJn.flatten(), dim=0).item())
            entry["relational_minus_random"] = (entry.get("relational_frobenius_region", entry["dJ_hazard_frobenius_region"]) - entry["dJ_random_frobenius_region"])
            Ridx = torch.as_tensor(R, device=dJh.device)
            Gidx = torch.as_tensor(G_route, device=dJh.device) if len(G_route) else Ridx
            if has_ident:  # v0.8 identity contrast: pedestrian-induced vs object-induced change of the action Jacobian
                dJo = J[(OBJECT_HAZARD, a)] - J[(0, a)]
                ident = dJh - dJo
                entry["dJ_object_frobenius_region"] = float(dJo[Ridx].norm().item())
                entry["identity_frobenius_region"] = float(ident[Ridx].norm().item())
                entry["cosine_dJ_hazard_vs_dJ_object"] = float(torch.nn.functional.cosine_similarity(dJh.flatten(), dJo.flatten(), dim=0).item())
                entry["S_F_route_object"] = float(dJo[Gidx].norm().item()) / max(float(J[(0, a)][Gidx].norm().item()), 1e-12)
                entry["S_F_route_identity"] = float(ident[Gidx].norm().item()) / max(float(J[(0, a)][Gidx].norm().item()), 1e-12)
                entry["identity_minus_random"] = entry["identity_frobenius_region"] - entry["dJ_random_frobenius_region"]
            dJs = J[("sign", a)] - J[(0, a)]
            entry["dJ_sign_frobenius_region"] = float(dJs[Ridx].norm().item())
            entry["relational_minus_sign"] = entry.get("relational_frobenius_region", entry["dJ_hazard_frobenius_region"]) - entry["dJ_sign_frobenius_region"]
            # E1 primary statistics on the ROUTE tokens (gripper + corridor); region versions kept above
            entry["S_F_route"] = float(rel[Gidx].norm().item()) / max(float(J[(0, a)][Gidx].norm().item()), 1e-12)
            entry["S_F_route_random"] = float(dJr[Gidx].norm().item()) / max(float(J[(0, a)][Gidx].norm().item()), 1e-12)
            entry["S_F_route_sign"] = float(dJs[Gidx].norm().item()) / max(float(J[(0, a)][Gidx].norm().item()), 1e-12)
            entry["principal_angles_route"] = {
                "hazard": principal_angles(J[(1, a)][Gidx], J[(0, a)][Gidx]),
                "random": principal_angles(J[("rand", a)][Gidx], J[(0, a)][Gidx]),
                "sign": principal_angles(J[("sign", a)][Gidx], J[(0, a)][Gidx]),
                **({"null": principal_angles(J[(2, a)][Gidx], J[(0, a)][Gidx])} if has_null else {}),
                **({"object": principal_angles(J[(OBJECT_HAZARD, a)][Gidx], J[(0, a)][Gidx])} if has_ident else {}),
            }
            entry["top_angle_minus_random_deg"] = entry["principal_angles_route"]["hazard"]["top_angle_deg"] - entry["principal_angles_route"]["random"]["top_angle_deg"]
            entry["top_angle_minus_sign_deg"] = entry["principal_angles_route"]["hazard"]["top_angle_deg"] - entry["principal_angles_route"]["sign"]["top_angle_deg"]
            entry["projected_I_true"] = {
                "relational": projected_statistic(rel, a_diff, I_true, G_route if len(G_route) else R),
                "hazard": projected_statistic(dJh, a_diff, I_true, G_route if len(G_route) else R),
                "random": projected_statistic(dJr, a_diff, I_true, G_route if len(G_route) else R),
                "sign": projected_statistic(dJs, a_diff, I_true, G_route if len(G_route) else R),
            }
            if value_path:
                dJh_v = Jv[(1, a)] - Jv[(0, a)]
                rel_v = dJh_v - (Jv[(2, a)] - Jv[(0, a)]) if has_null else dJh_v
                pattern = rel - rel_v
                entry["pattern_value_split_route"] = {
                    "value_path_relational_frobenius": float(rel_v[Gidx].norm().item()),
                    "pattern_term_frobenius": float(pattern[Gidx].norm().item()),
                    "pattern_fraction": float(pattern[Gidx].norm().item()) / max(float(rel[Gidx].norm().item()), 1e-12),
                    "cosine_value_vs_full": float(torch.nn.functional.cosine_similarity(rel_v[Gidx].flatten(), rel[Gidx].flatten(), dim=0).item()),
                    "S_F_route_value_path": float(rel_v[Gidx].norm().item()) / max(float(Jv[(0, a)][Gidx].norm().item()), 1e-12),
                }
            entry["central_difference"] = central_difference_check(wm, z_cache[(0, a)], a_cache[(0, a)], k_step, J[(0, a)], a_diff, G_route if len(G_route) else R, fd_eps)
            # total-level finite-difference confirmation a0 -> a1 (relational FD vs dJ_rel (a1 - a0))
            with torch.no_grad():
                fd = {h: _predict(wm, z_cache[(h, 0)], a_cache[(h, 1)], k_step) - _predict(wm, z_cache[(h, 0)], a_cache[(h, 0)], k_step) for h in ((0, 1, 2) if has_null else (0, 1))}
            fd_rel = (fd[1] - fd[0]) - ((fd[2] - fd[0]) if has_null else 0.0)
            pred_rel = rel @ a_diff
            entry["fd_confirmation_route"] = {
                "cosine_fd_vs_jacobian": float(torch.nn.functional.cosine_similarity(fd_rel[Gidx].flatten(), pred_rel[Gidx].flatten(), dim=0).item()),
                "fd_norm": float(fd_rel[Gidx].norm().item()),
                "jacobian_prediction_norm": float(pred_rel[Gidx].norm().item()),
                "sign_agreement": float((torch.sign(fd_rel[Gidx]) == torch.sign(pred_rel[Gidx])).float().mean().item()),
            }
            # mixed-derivative check (small-step central difference along delta_h vs the unit step)
            delta = (z_flat((1, a)) - z_flat((0, a))).reshape(z_of((0, a)).shape)
            Jp, _ = jacobian_forward_ad(wm, predictor, z_of((0, a)) + hvp_eps * delta, a_of[(0, a)], k_step, n_spatial, [])
            Jm, _ = jacobian_forward_ad(wm, predictor, z_of((0, a)) - hvp_eps * delta, a_of[(0, a)], k_step, n_spatial, [])
            hvp = (Jp - Jm) / (2.0 * hvp_eps)  # d/dt J(z + t delta) at t=0  ==  d^2P/(da dz) . delta
            entry["mixed_derivative"] = {
                "eps": hvp_eps,
                "hvp_frobenius_region": float(hvp[torch.as_tensor(R, device=hvp.device)].norm().item()),
                "cosine_hvp_vs_dJ_hazard": float(torch.nn.functional.cosine_similarity(hvp.flatten(), dJh.flatten(), dim=0).item()),
                "relative_error_hvp_vs_dJ_hazard": float(((hvp - dJh).norm() / dJh.norm().clamp_min(1e-12)).item()),
            }
            rel_vectors[a] = {"dJ_hazard": pooled(dJh, R), "relational": pooled(rel, R), "dJ_random": pooled(dJr, R)}
            step_out["relational"][f"a{a}"] = entry
        step_out["pooled_vectors"] = rel_vectors  # consumed by the pooled statistics, not serialised
        # site mediation
        if k_step in mediation_steps and sites:
            med: dict[str, Any] = {}
            for site in sites:
                sid = site.site_id
                if sid not in site_tan[(0, 0)]:
                    continue
                for gname in site_groups:
                    gidx = None if gname == "all" else union(*(groups[k].group(in_frame, gname) for k in keys))
                    if gidx is not None and len(gidx) == 0:
                        continue
                    M: dict[Any, torch.Tensor] = {}
                    for k in keys + rand_keys:
                        cols = []
                        for i in range(site_tan[k][sid].shape[0]):
                            tg = site_tan[k][sid][i].clone()
                            if gidx is not None:
                                mask = torch.zeros(tg.shape[0], dtype=torch.bool, device=tg.device)
                                mask[torch.as_tensor(gidx, device=tg.device)] = True
                                tg[~mask] = 0.0
                            cols.append(downstream_response(wm, predictor, z_cache[k], a_cache[k], k_step, n_spatial, sid, tg))
                        M[k] = torch.stack(cols, dim=-1)
                    Ridx = torch.as_tensor(R, device=M[(0, 0)].device)
                    per_a = {}
                    for a in (0, 1):
                        dMh = M[(1, a)] - M[(0, a)]
                        dMr = M[("rand", a)] - M[(0, a)]
                        dMs = M[("sign", a)] - M[(0, a)]
                        rel = dMh - (M[(2, a)] - M[(0, a)]) if has_null else dMh
                        # finite-difference confirmation through this site: self-donor action patch a0 -> a1 on the same tokens
                        fd_site = {h: self_donor_site_patch(wm, predictor, z_cache[(h, 0)], a_cache[(h, 0)], a_cache[(h, 1)], k_step, n_spatial, sid, gidx) for h in ((0, 1, 2) if has_null else (0, 1))}
                        fd_site_rel = (fd_site[1] - fd_site[0]) - ((fd_site[2] - fd_site[0]) if has_null else 0.0)
                        pred_site_rel = rel @ a_diff
                        per_a[f"a{a}"] = {
                            "M_h0_frobenius_region": float(M[(0, a)][Ridx].norm().item()),
                            "dM_hazard_frobenius_region": float(dMh[Ridx].norm().item()),
                            "dM_random_frobenius_region": float(dMr[Ridx].norm().item()),
                            "relational_frobenius_region": float(rel[Ridx].norm().item()),
                            "fraction_of_total_relational": float(rel[Ridx].norm().item()) / max(step_out["relational"][f"a{a}"].get("relational_frobenius_region", step_out["relational"][f"a{a}"]["dJ_hazard_frobenius_region"]), 1e-12),
                            "fraction_of_total_J": float(M[(0, a)][Ridx].norm().item()) / max(step_out["relational"][f"a{a}"]["J_h0_frobenius_region"], 1e-12),
                            "relational_minus_random": float(rel[Ridx].norm().item() - dMr[Ridx].norm().item()),
                            "dM_sign_frobenius_region": float(dMs[Ridx].norm().item()),
                            "relational_minus_sign": float(rel[Ridx].norm().item() - dMs[Ridx].norm().item()),
                            "S_F_site": float(rel[Ridx].norm().item()) / max(float(M[(0, a)][Ridx].norm().item()), 1e-12),
                            "top_angle_deg": principal_angles(M[(1, a)][Ridx], M[(0, a)][Ridx])["top_angle_deg"],
                            "top_angle_random_deg": principal_angles(M[("rand", a)][Ridx], M[(0, a)][Ridx])["top_angle_deg"],
                            "top_angle_sign_deg": principal_angles(M[("sign", a)][Ridx], M[(0, a)][Ridx])["top_angle_deg"],
                            "cosine_M_vs_J_h0": float(torch.nn.functional.cosine_similarity(M[(0, a)][Ridx].flatten(), J[(0, a)][Ridx].flatten(), dim=0).item()),
                            "fd_confirmation": {
                                "cosine_fd_vs_jacobian": float(torch.nn.functional.cosine_similarity(fd_site_rel[Ridx].flatten(), pred_site_rel[Ridx].flatten(), dim=0).item()),
                                "sign_agreement": float((torch.sign(fd_site_rel[Ridx]) == torch.sign(pred_site_rel[Ridx])).float().mean().item()),
                                "fd_norm": float(fd_site_rel[Ridx].norm().item()),
                                "jacobian_prediction_norm": float(pred_site_rel[Ridx].norm().item()),
                            },
                        }
                    med[f"{sid}|{gname}"] = per_a
            step_out["mediation"] = med
        out["steps"][str(k_step)] = step_out
    return out


# --------------------------------------------------------------------------- #
# Pooled statistics
# --------------------------------------------------------------------------- #


def pooled_statistics(per_scene: dict[str, dict[str, Any]], steps: list[int], sites: list[Site], site_groups: list[str], n_boot: int, n_perm: int, rng: np.random.Generator, alpha: float) -> dict[str, Any]:
    pair_ids = sorted(per_scene)
    out: dict[str, Any] = {}
    for k_step in steps:
        s = str(k_step)
        entry: dict[str, Any] = {"autoregressive": k_step >= 1, "n_scenes": len(pair_ids)}
        for a in (0, 1):
            rows = [per_scene[p]["steps"][s]["relational"][f"a{a}"] for p in pair_ids if s in per_scene[p]["steps"]]
            stats = {}
            for field in ("dJ_hazard_frobenius_region", "dJ_random_frobenius_region", "dJ_sign_frobenius_region", "dJ_null_frobenius_region", "relational_frobenius_region",
                          "ratio_dJ_hazard_over_J_h0", "relational_minus_random", "relational_minus_sign", "S_F_route", "S_F_route_random", "S_F_route_sign",
                          "top_angle_minus_random_deg", "top_angle_minus_sign_deg",
                          "dJ_object_frobenius_region", "identity_frobenius_region", "cosine_dJ_hazard_vs_dJ_object", "S_F_route_object", "S_F_route_identity", "identity_minus_random"):
                vals = np.asarray([r.get(field, np.nan) for r in rows], dtype=float)
                if np.all(np.isnan(vals)):
                    continue
                stats[field] = cluster_bootstrap(vals, n_boot, rng)
            hv = np.asarray([r["mixed_derivative"]["relative_error_hvp_vs_dJ_hazard"] for r in rows])
            stats["mixed_derivative_relative_error"] = cluster_bootstrap(hv, n_boot, rng)
            for nested, field in (("principal_angles_route", None), ("projected_I_true", None), ("pattern_value_split_route", None), ("fd_confirmation_route", None)):
                if nested in rows[0]:
                    flat = {}
                    for r in rows:
                        for kk, vv in r[nested].items():
                            if isinstance(vv, dict):
                                for k2, v2 in vv.items():
                                    if isinstance(v2, (int, float)):
                                        flat.setdefault(f"{kk}.{k2}", []).append(v2)
                            elif isinstance(vv, (int, float)):
                                flat.setdefault(kk, []).append(vv)
                    stats[nested] = {kk: cluster_bootstrap(np.asarray(v, dtype=float), n_boot, rng) for kk, v in flat.items()}
            if "central_difference" in rows[0]:
                stats["central_difference"] = {eps: {"relative_error": cluster_bootstrap(np.asarray([r["central_difference"][eps]["relative_error"] for r in rows]), n_boot, rng)} for eps in rows[0]["central_difference"]}
            # sign-flip: relational norm minus each null norm, S_F and top-angle differences (paired within scene)
            for field in ("relational_minus_random", "relational_minus_sign", "top_angle_minus_random_deg", "top_angle_minus_sign_deg", "identity_minus_random"):
                if not all(field in r for r in rows):
                    continue
                diff = np.asarray([r[field] for r in rows]).reshape(-1, 1)
                if len(diff) >= 2:
                    res = sign_flip_maxt({field: diff}, n_perm, rng, statistic=lambda v, sg: v[:, 0] if sg is None else v[:, 0] * sg)
                    stats[f"{field}_test"] = {**res["sites"][field], "n_perm": res["n_perm"], "exact": res["exact"]}
            sf = stats.get("S_F_route", {}).get("point")
            ta = stats.get("principal_angles_route", {}).get("hazard.top_angle_deg", {}).get("point")
            stats["E1_total_level"] = {"S_F_ge_0.05": bool(sf is not None and sf >= 0.05), "top_angle_ge_10deg": bool(ta is not None and ta >= 10.0),
                                       "beats_random_p": stats.get("relational_minus_random_test", {}).get("p_raw"), "beats_sign_p": stats.get("relational_minus_sign_test", {}).get("p_raw")}
            # LOSO consistency of pooled vec(dJ) across scenes (sign-flip == hazard-label permutation)
            for kind in ("dJ_hazard", "relational", "dJ_random"):
                mats = np.stack([per_scene[p]["steps"][s]["pooled_vectors"][a][kind] for p in pair_ids])
                proj = cv_projection(mats)
                if len(pair_ids) >= 2:
                    res = sign_flip_maxt({kind: mats}, n_perm, rng)
                    proj.update(res["sites"][kind])
                proj.pop("per_scene", None)
                stats[f"loso_{kind}"] = proj
            entry[f"a{a}"] = stats
        # mediation map with max-T across sites (relational minus random, per site|group)
        med_keys = sorted({k for p in pair_ids for k in per_scene[p]["steps"][s]["mediation"]})
        if med_keys:
            med_stats: dict[str, Any] = {}
            for a in (0, 1):
                tests = {}
                for field in ("relational_minus_random", "relational_minus_sign"):
                    vectors = {}
                    for key in med_keys:
                        vals = np.asarray([per_scene[p]["steps"][s]["mediation"][key][f"a{a}"][field] for p in pair_ids if key in per_scene[p]["steps"][s]["mediation"]])
                        if len(vals) == len(pair_ids):
                            vectors[key] = vals.reshape(-1, 1)
                    tests[field] = sign_flip_maxt(vectors, n_perm, rng, statistic=lambda v, sg: v[:, 0] if sg is None else v[:, 0] * sg) if vectors else {"sites": {}}
                for key in med_keys:
                    rows_k = [per_scene[p]["steps"][s]["mediation"][key][f"a{a}"] for p in pair_ids if key in per_scene[p]["steps"][s]["mediation"]]
                    summ = {
                        "fraction_of_total_relational": cluster_bootstrap(np.asarray([r["fraction_of_total_relational"] for r in rows_k]), n_boot, rng),
                        "relational_frobenius_region": cluster_bootstrap(np.asarray([r["relational_frobenius_region"] for r in rows_k]), n_boot, rng),
                        "S_F_site": cluster_bootstrap(np.asarray([r["S_F_site"] for r in rows_k]), n_boot, rng),
                        "top_angle_deg": cluster_bootstrap(np.asarray([r["top_angle_deg"] for r in rows_k]), n_boot, rng),
                        "fd_cosine": cluster_bootstrap(np.asarray([r["fd_confirmation"]["cosine_fd_vs_jacobian"] for r in rows_k]), n_boot, rng),
                        "fd_sign_agreement": cluster_bootstrap(np.asarray([r["fd_confirmation"]["sign_agreement"] for r in rows_k]), n_boot, rng),
                        "relational_minus_random_test": tests["relational_minus_random"]["sites"].get(key),
                        "relational_minus_sign_test": tests["relational_minus_sign"]["sites"].get(key),
                    }
                    p_r = (summ["relational_minus_random_test"] or {}).get("p_maxt_fwer")
                    p_s = (summ["relational_minus_sign_test"] or {}).get("p_maxt_fwer")
                    summ["E1_site"] = bool(summ["S_F_site"]["point"] >= 0.05 and summ["top_angle_deg"]["point"] >= 10.0
                                           and p_r is not None and p_r == p_r and p_r < alpha and p_s is not None and p_s == p_s and p_s < alpha)
                    med_stats.setdefault(key, {})[f"a{a}"] = summ
                med_stats.setdefault("_permutation", {})[f"a{a}"] = {f: {k: v for k, v in res.items() if k != "sites"} for f, res in tests.items()}
            ranked = sorted((k for k in med_keys), key=lambda k: -(med_stats[k]["a1"]["fraction_of_total_relational"]["point"] or -np.inf))
            entry["mediation"] = med_stats
            entry["mediation_ranked_by_fraction_a1"] = ranked[:20]
            entry["E1_sites_a1"] = [k for k in med_keys if med_stats[k]["a1"]["E1_site"]]
            entry["E1_sites_a0"] = [k for k in med_keys if med_stats[k]["a0"]["E1_site"]]
        out[s] = entry
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="jepa_wm_droid")
    parser.add_argument("--model-label", default="JEPA-WM DROID action-Jacobian sonar")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sites", nargs="*", default=None, help="mediation sites; default = all layers x {attn_out, mlp_out, resid_post}")
    parser.add_argument("--hooks", nargs="*", default=list(MEDIATION_HOOKS))
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--groups", nargs="*", default=["gripper_corridor", "egg", "all"], help="token groups for the site tangents")
    parser.add_argument("--steps", type=int, nargs="*", default=[0], help="prediction index k (0 = P_1 from the context); k >= 1 is autoregressive")
    parser.add_argument("--mediation-steps", type=int, nargs="*", default=[0])
    parser.add_argument("--no-mediation", action="store_true")
    parser.add_argument("--hvp-eps", type=float, default=0.1)
    parser.add_argument("--fd-eps", type=float, nargs="*", default=[1e-3, 1e-2, 1e-1])
    parser.add_argument("--no-value-path", action="store_true", help="skip the attention-pattern/value split")
    parser.add_argument("--seeds-file", type=Path, default=None)
    parser.add_argument("--allow-calibration-seeds", action="store_true")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-perm", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--domain", choices=list(DOMAINS), default="egg", help="driving = v0.8 MetaDrive scenes (hazard/corridor groups, level-3 identity contrast)")
    args = parser.parse_args()
    set_domain(args.domain)

    t0 = time.time()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    wm, _ = load_model(args.repo, args.config, args.checkpoint, args.model_name, str(device))
    predictor = predictor_of(wm)
    for p in wm.parameters():
        p.requires_grad_(False)
    n_layers = n_layers_of(predictor)
    n_spatial = spatial_tokens_of(wm)
    if args.no_mediation:
        sites: list[Site] = []
    elif args.sites:
        sites = [Site.parse(s) for s in args.sites]
    else:
        layers = n_layers if args.max_layers is None else min(n_layers, args.max_layers)
        sites = all_sites(layers, args.hooks, include_embed=False)

    pairs = select_pairs(merge_manifests(args.artifacts), args.seeds_file, args.allow_calibration_seeds)
    if not pairs:
        raise SystemExit("no complete quartets selected")
    null_status = null_factor_status(pairs)
    null_pairs = set(null_factor_pairs(pairs))
    ident_status = identity_status(pairs) if args.domain == "driving" else "none"
    ident_pairs = set(identity_pairs(pairs)) if args.domain == "driving" else set()

    per_scene: dict[str, dict[str, Any]] = {}
    for pair_id in sorted(pairs):
        cell_keys = list(CELL_ORDER) + (list(NULL_CELLS) if pair_id in null_pairs else []) + (list(IDENTITY_CELLS) if pair_id in ident_pairs else [])
        latents = {key: encode_cell(wm, *pairs[pair_id][key], device) for key in cell_keys}
        groups = {key: load_cell_groups(*pairs[pair_id][key]) for key in cell_keys}
        per_scene[pair_id] = evaluate_scene(wm, predictor, n_spatial, latents, groups, sites, args.groups, args.steps, args.mediation_steps, generator, args.hvp_eps,
                                            tuple(args.fd_eps), not args.no_value_path)
        print(f"[jacobian] scene {pair_id} done ({time.time() - t0:.1f}s)", file=sys.stderr)

    pooled_stats = pooled_statistics(per_scene, args.steps, sites, args.groups, args.n_boot, args.n_perm, rng, args.alpha)
    # site_mediation.npz: [n_scenes, n_keys, 2 actions, 4 stats]
    med_keys = sorted({k for p in per_scene.values() for s in p["steps"].values() for k in s["mediation"]})
    args.output.mkdir(parents=True, exist_ok=True)
    if med_keys:
        stats_names = ("M_h0_frobenius_region", "dM_hazard_frobenius_region", "dM_random_frobenius_region", "dM_sign_frobenius_region", "relational_frobenius_region", "S_F_site", "top_angle_deg")
        for k_step in args.mediation_steps:
            arr = np.full((len(per_scene), len(med_keys), 2, len(stats_names)), np.nan)
            for i, pid in enumerate(sorted(per_scene)):
                med = per_scene[pid]["steps"].get(str(k_step), {}).get("mediation", {})
                for j, key in enumerate(med_keys):
                    for a in (0, 1):
                        if key in med:
                            arr[i, j, a] = [med[key][f"a{a}"][n] for n in stats_names]
            np.savez(args.output / (f"site_mediation.npz" if k_step == 0 else f"site_mediation_s{k_step}.npz"), values=arr, scenes=np.array(sorted(per_scene)), keys=np.array(med_keys), stats=np.array(stats_names))
    seeds = sorted({int(pairs[pid][CELL_ORDER[0]][1]["seed"]) for pid in pairs})
    for p in per_scene.values():
        for s in p["steps"].values():
            s.pop("pooled_vectors", None)
    out = {
        "tool": "action_jacobian_sonar (draft)",
        "model_label": args.model_label, "model_name": args.model_name,
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256_file(args.checkpoint), "config_sha256": sha256_file(args.config),
        "artifacts": [str(p) for p in args.artifacts], "pair_ids": sorted(pairs), "seeds": seeds, "n_scenes": len(pairs),
        "null_factor_status": null_status, "n_null_scenes": len(null_pairs),
        **({"domain": args.domain, "token_group_source": alias_note(args.domain), "identity_factor": ident_status, "n_identity_scenes": len(ident_pairs)} if args.domain != "egg" else {}),
        "sites": [s.site_id for s in sites], "site_groups": args.groups, "steps": args.steps, "mediation_steps": args.mediation_steps,
        "hvp_eps": args.hvp_eps, "n_boot": args.n_boot, "n_perm": args.n_perm, "alpha": args.alpha,
        "pooled": pooled_stats,
        "per_scene": per_scene,
        "interpretation_scope": ("Differentiable action sensitivity of the latent forecast (no donor, no candidate-action pair). Relational = hazard-induced "
                                 "change of the action Jacobian minus the matched off-path (H0') change. Step 0 is primary; k >= 1 is autoregressive. "
                                 + ZERO_SHOT_SCOPE + {"none": " No H0' cells.", "partial": f" H0' cells for {len(null_pairs)}/{len(pairs)} scenes.", "full": ""}[null_status]),
        "runtime_s": time.time() - t0, "torch_version": torch.__version__,
    }
    (args.output / "jacobian_sonar.json").write_text(canonical_json(json_safe(out)) + "\n")
    brief = {s: {a: {**{k: v.get("point") if isinstance(v, dict) else v for k, v in pooled_stats[s][a].items()
                         if k in ("S_F_route", "S_F_route_random", "S_F_route_sign", "relational_frobenius_region", "dJ_random_frobenius_region", "dJ_sign_frobenius_region")},
                      "top_angle_deg": pooled_stats[s][a].get("principal_angles_route", {}).get("hazard.top_angle_deg", {}).get("point"),
                      "E1": pooled_stats[s][a].get("E1_total_level")} for a in ("a0", "a1")} for s in pooled_stats}
    print(json.dumps(json_safe({"n_scenes": len(pairs), "runtime_s": out["runtime_s"], "brief": brief,
                                "E1_sites_a1_s0": pooled_stats.get("0", {}).get("E1_sites_a1", []),
                                "top_mediation_s0": pooled_stats.get("0", {}).get("mediation_ranked_by_fraction_a1", [])[:8]}), indent=1))


if __name__ == "__main__":
    main()
