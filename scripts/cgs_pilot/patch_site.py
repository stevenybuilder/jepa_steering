#!/usr/bin/env python3
"""Step 11: bidirectional activation patching with real counterfactual donors,
token-group granularity, controls, off-manifold and INT diagnostics, and a
band-level site-selection rule.

Donors are always REAL latent states from the matched cell of the same scene
(arXiv:2309.16042).  For hazard patches the donor differs from the receiver only
in hazard (same candidate action): ``(0,A) <- (1,A)`` [safe->unsafe] and
``(1,A) <- (0,A)`` [unsafe->safe].  Action-only donors ``(H,0) <-> (H,1)`` are the
action-availability positive control.  Patches write the donor's activation at
one predictor site (``layer x hook``), one imagined step, and one token group
(``token_groups.py``: egg / gripper / corridor / gripper_corridor / all).

Controls written at the same site/step/group unless stated:

- ``sham``              donor = receiver
- ``random``            isotropic direction with the donor-receiver norm (``--n-random`` draws)
- ``time_shift``        donor activation from a different imagined step
- ``unrelated_site``    same donor content at the same hook ``--unrelated-offset`` layers away
- ``main_effect_only``  H1 cell with the OTHER candidate action as donor (hazard appearance
                        without the receiver's route)
- ``wrong_group``       donor tokens of the other group (egg <-> gripper_corridor) written
                        into the target group's positions (cyclic assignment)
- ``null_factor``       H0' donor (hazard == 2, matched off-path placement) when present

Endpoint (final imagined step, last-frame latent):

- ``recovery_region``   ``1 - ||p'_R - p_donor_R|| / ||p_base_R - p_donor_R||`` on the
                        egg+gripper token region R of the final frame
- ``recovery_all``      the same on all 256 tokens (reference)
- ``unsafe_future_delta``  change in RMS distance of the region to the true (H1,A1)
                        future latent (negative = moved toward the unsafe future)
- ``patch_effect_did``  ``recovery_region(receiver A=1) - recovery_region(receiver A=0)``
                        per hazard direction and scene (the relational signature)
- ``cem_l2_cost_delta`` change of the CEM planning objective ``mean ||p - goal||^2`` with the
                        receiver's own true final latent as goal
- ``corruption``        ``d(p'_R, t_donor_R) / d(p_donor_R, t_donor_R)``
- ``mahalanobis``       diagonal-covariance Mahalanobis score (per dim, /sqrt(d)) of the written
                        tokens against the clean per-site token distribution of the scene,
                        and the receiver's own tokens' score (arXiv:2511.04638)
- INT (arXiv:2606.27510): recovery of site X alone, Y alone and X+Y together
  (``--int-pairs X+Y``), reported as ``int = XY - X - Y``

Modes: ``one_shot`` (single step; step 0 is the clean primary analysis, steps 1-2
report persistence of a one-shot edit) and ``persistent`` (the site is patched at
every step with the donor's activation at that step).

Candidate rule: only token-group patches (groups other than ``all``) or
``attn_out``/``mlp_out`` component patches count as candidates; whole-256-token
residual patches and ``resid_post`` at the last layer are trivially scene
replacement and are reported as ``reference`` only; ``adaln`` sites are the action
availability positive control and never relational candidates.

Partial null factor: when only a subset of scenes carries ``h2`` cells the
``null_factor`` control is run on that subset (``n_null_scenes``, exact sign-flip
on donor-minus-null recovery), and liveness uses it whenever at least one scene
has it; ``--allow-missing-null-factor`` only governs the zero-h2 case.
``--focus-sites`` runs the full battery (all directions, one-shot and persistent,
INT with L06.attn_out / L08.attn_out / L05.mlp_out) on a short list and emits
per-scene recovery and patch-effect DiD tables.

Site selection (arXiv:2510.00845: report a band, not an index): order sites by
``(step, layer, hook)`` with resid_pre < attn_out < mlp_in < mlp_out < resid_post
and resid_post(L) == resid_pre(L+1).  A site is *interaction-live* when the H-only
patch-effect DiD at ``gripper_corridor`` is significant under sign-flip max-T
(p < ``--alpha``) AND the H0' null donor recovers < 10 % (median).  ``selected`` =
earliest live site with >= 30 % median recovery in both hazard directions; the
reported BAND is the maximal contiguous run of live sites containing it, with
per-scene stability.  ``--seeds-file`` freezes the band on discovery seeds;
``--confirm --band-file`` evaluates a frozen band on confirmation seeds without any
selection.

``--domain driving`` (protocol v0.8, MetaDrive): token groups from ``hazard_mask`` /
``corridor_mask`` with the egg names as aliases (``egg`` -> ``hazard``,
``gripper_corridor`` -> ``corridor``); the endpoint region is hazard U corridor
(``token_groups.REGION_GROUPS``); the primary group is ``hazard_corridor``
(alias ``gripper_corridor``). Hazard level 3 (cone on the same path) adds the
``identity`` donor (``(0,A) <- (3,A)``): its recovery next to the pedestrian donor
is the identity x action contrast (``donor_minus_identity_test``); it is reported
and never enters liveness or ``controls_below_threshold`` (under arm B the object
donor is expected to recover).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cgs_stats import cluster_bootstrap, json_safe, sign_flip_maxt  # noqa: E402
from localize_interaction import (  # noqa: E402
    CALIBRATION_SEEDS,
    IDENTITY_CELLS,
    NULL_CELLS,
    canon,
    encode_cell,
    identity_pairs,
    identity_status,
    merge_manifests,
    null_factor_pairs,
    null_factor_status,
    rms,
    select_pairs,
)
from model_action_sensitivity import load_model  # noqa: E402
from predictor_hooks import (  # noqa: E402
    COMPONENT_HOOKS,
    HOOK_POINTS,
    ZERO_SHOT_SCOPE,
    PredictorPatcher,
    PredictorRecorder,
    Site,
    all_sites,
    n_layers_of,
    norm_matched_random,
    predictor_of,
    site_order_key,
    spatial_tokens_of,
    unrelated_site,
    unroll_from_latents,
)
from protocol import CELL_ORDER, OBJECT_HAZARD, canonical_json, sha256_file  # noqa: E402
from token_groups import DOMAINS, PRIMARY_GROUP as DOMAIN_PRIMARY_GROUP, REGION_GROUPS, CellGroups, alias_note, candidate_groups, frame_for_step, load_cell_groups, set_domain, summarize, union  # noqa: E402


HAZARD_DIRECTIONS = {
    "hazard_safe_to_unsafe": lambda a: ((0, a), (1, a)),
    "hazard_unsafe_to_safe": lambda a: ((1, a), (0, a)),
}
ACTION_DIRECTIONS = {
    "action_gentle_to_aggr": lambda h: ((h, 0), (h, 1)),
    "action_aggr_to_gentle": lambda h: ((h, 1), (h, 0)),
}
DIRECTIONS = {**HAZARD_DIRECTIONS, **ACTION_DIRECTIONS}
CONTROLS = ("sham", "random", "time_shift", "unrelated_site", "main_effect_only", "wrong_group", "null_factor")
IDENTITY_CONTROL = "identity"  # driving only: object-on-path donor, reported next to the null donor, never a liveness criterion
OTHER_GROUP = {"egg": "gripper_corridor", "gripper_corridor": "egg", "gripper": "egg", "corridor": "egg", "all": "egg", "background": "egg",
               "hazard": "corridor", "hazard_corridor": "hazard"}
PRIMARY_GROUP = "gripper_corridor"
PRIMARY_DIRECTION = "hazard_safe_to_unsafe"
_DOMAIN = ["egg"]


def _domain() -> str:
    return _DOMAIN[0]


def _kinds() -> tuple[str, ...]:
    return ("donor",) + CONTROLS + ((IDENTITY_CONTROL,) if _domain() == "driving" else ())


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def endpoint(pred: torch.Tensor) -> torch.Tensor:
    last = pred[-1]
    return last.reshape(last.shape[0], -1, last.shape[-1])[0].float()


def _sel(x: torch.Tensor, index: np.ndarray | None) -> torch.Tensor:
    if index is None or len(index) == 0:
        return x
    return x[torch.as_tensor(index, dtype=torch.long, device=x.device)]


def patch_metrics(
    p_patched: torch.Tensor,
    p_base: torch.Tensor,
    p_donor: torch.Tensor,
    t_base: torch.Tensor,
    t_donor: torch.Tensor,
    t_unsafe: torch.Tensor,
    z_ctx: torch.Tensor,
    region: np.ndarray | None,
) -> dict[str, float]:
    """All inputs are ``[n_tokens, d]`` final-frame latents; ``region`` indexes the egg+gripper tokens."""

    def rec(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, idx: np.ndarray | None) -> tuple[float, float, float]:
        gap = rms(_sel(b, idx) - _sel(c, idx))
        residual = rms(_sel(a, idx) - _sel(c, idx))
        return 1.0 - residual / max(gap, 1e-12), gap, residual

    recovery_all, gap_all, _ = rec(p_patched, p_base, p_donor, None)
    recovery_region, gap_region, residual_region = rec(p_patched, p_base, p_donor, region)
    cost = lambda p: float(torch.mean((p - t_base) ** 2).item())  # noqa: E731  CEM L2 objective, receiver goal
    return {
        "recovery_region": float(recovery_region),
        "recovery_all": float(recovery_all),
        "gap_region_rms": float(gap_region),
        "gap_all_rms": float(gap_all),
        "residual_region_rms": float(residual_region),
        "unsafe_future_delta": float(rms(_sel(p_patched, region) - _sel(t_unsafe, region)) - rms(_sel(p_base, region) - _sel(t_unsafe, region))),
        "target_shift": float(
            (rms(_sel(p_patched, region) - _sel(t_donor, region)) - rms(_sel(p_patched, region) - _sel(t_base, region)))
            - (rms(_sel(p_base, region) - _sel(t_donor, region)) - rms(_sel(p_base, region) - _sel(t_base, region)))
        ),
        "displacement_delta": float(rms(p_patched - z_ctx) - rms(p_base - z_ctx)),
        "cem_l2_cost_delta": cost(p_patched) - cost(p_base),
        "corruption": float(rms(_sel(p_patched, region) - _sel(t_donor, region)) / max(rms(_sel(p_donor, region) - _sel(t_donor, region)), 1e-12)),
        "edit_rms": float(rms(p_patched - p_base)),
    }


def mahalanobis_stats(samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Diagonal mean/variance from ``[n, d]`` clean token activations."""

    mu = samples.float().mean(dim=0)
    var = samples.float().var(dim=0, unbiased=False).clamp_min(1e-8)
    return mu, var


def mahalanobis_score(x: torch.Tensor, mu: torch.Tensor, var: torch.Tensor) -> float:
    """Mean over tokens of sqrt(mean_d z^2): ~1 for in-distribution tokens."""

    z2 = ((x.float() - mu) ** 2 / var).mean(dim=-1)
    return float(torch.sqrt(z2).mean().item())


def cyclic_assign(source: torch.Tensor, n_target: int) -> torch.Tensor:
    """``[B, k, d]`` donor tokens -> ``[B, n_target, d]`` by cyclic reuse."""

    if source.shape[1] == 0 or n_target == 0:
        return source[:, :0]
    idx = torch.arange(n_target, device=source.device) % source.shape[1]
    return source[:, idx]


# --------------------------------------------------------------------------- #
# Per-pair evaluation
# --------------------------------------------------------------------------- #


def _patch_value(acts: torch.Tensor, index: np.ndarray | None, values: torch.Tensor | None = None):
    """Build the patcher payload for site activations ``[B, n_tok, d]`` restricted to ``index``."""

    if acts.shape[1] == 1 or index is None:  # adaln or whole group
        return acts if values is None else values
    idx = torch.as_tensor(index, dtype=torch.long, device=acts.device)
    return (idx, acts[:, idx] if values is None else values)


def evaluate_pair(
    wm: Any,
    predictor: torch.nn.Module,
    n_spatial: int,
    n_layers: int,
    latents: dict[tuple[int, int], dict[str, torch.Tensor]],
    groups: dict[tuple[int, int], CellGroups],
    sites: list[Site],
    steps: list[int] | None,
    group_names: list[str],
    n_random: int,
    unrelated_offset: int,
    generator: torch.Generator,
    directions: list[str],
    mode: str = "one_shot",
    frame_offset: int = 0,
    int_pairs: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Return ``{site_id: {step: {group: {direction: {"fixed_<lvl>": {...}}}}}}`` plus ``"_int"``."""

    keys = [k for k in latents]
    null_present = all(k in latents for k in NULL_CELLS)
    ident_present = all(k in latents for k in IDENTITY_CELLS)
    record = {s.site_id: s for s in sites}
    for s in sites:
        u = unrelated_site(s, n_layers, unrelated_offset)
        record.setdefault(u.site_id, u)
    for x_id, y_id in int_pairs or []:
        for sid in (x_id, y_id):
            record.setdefault(sid, Site.parse(sid))
    clean: dict[tuple[int, int], dict[str, Any]] = {}
    for key in keys:
        rec = PredictorRecorder(predictor, record.values(), n_spatial, store_device=latents[key]["z_context"].device)
        pred = unroll_from_latents(wm, latents[key]["z_context"], latents[key]["actions"], recorder=rec)
        clean[key] = {"acts": rec.acts, "p": endpoint(pred), "n_steps": rec.n_steps}
    n_steps = clean[keys[0]]["n_steps"]
    use_steps = list(range(n_steps)) if steps is None else [s for s in steps if s < n_steps]
    step_list: list[Any] = ["persistent"] if mode == "persistent" else use_steps

    # clean per-site token distribution for Mahalanobis (all cells, steps, tokens of this scene)
    maha: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for sid in record:
        samples = torch.cat([clean[k]["acts"][sid][s][0] for k in keys for s in range(n_steps)], dim=0)
        maha[sid] = mahalanobis_stats(samples)

    def target(key: tuple[int, int]) -> torch.Tensor:
        z = latents[key]["z_future"][:, -1]
        return z.reshape(-1, z.shape[-1]).float()

    def ctx(key: tuple[int, int]) -> torch.Tensor:
        z = latents[key]["z_context"][:, -1]
        return z.reshape(-1, z.shape[-1]).float()

    def tokens(gname: str, step: Any, cells: tuple[tuple[int, int], ...]) -> np.ndarray | None:
        if gname == "all":
            return None
        frame = frame_for_step(0 if step == "persistent" else int(step), frame_offset)
        return union(*(groups[k].group(frame, gname) for k in cells))

    region_names = REGION_GROUPS[_domain()]

    def region_for(cells: tuple[tuple[int, int], ...]) -> np.ndarray:
        frame = n_steps  # final predicted frame
        return union(*(groups[k].group(frame, g) for k in cells for g in region_names))

    t_unsafe = target((1, 1))

    def run_patches(receiver: tuple[int, int], patches: dict) -> torch.Tensor:
        patcher = PredictorPatcher(predictor, patches, n_spatial)
        pred = unroll_from_latents(wm, latents[receiver]["z_context"], latents[receiver]["actions"], patcher=patcher)
        if len(patcher.applied) != len(patches):
            raise RuntimeError(f"patch not applied: requested {sorted(patches)} applied {patcher.applied}")
        return endpoint(pred)

    def donor_payload(sid: str, step: Any, donor: tuple[int, int], index: np.ndarray | None, values_fn=None) -> dict:
        step_ids = use_steps if step == "persistent" else [int(step)]
        payload = {}
        for s in step_ids:
            acts = clean[donor]["acts"][sid][s]
            payload[(sid, s)] = _patch_value(acts, index) if values_fn is None else _patch_value(acts, index, values_fn(s, acts))
        return payload

    out: dict[str, Any] = {}
    for site in sites:
        sid = site.site_id
        u = unrelated_site(site, n_layers, unrelated_offset)
        for step in step_list:
            for gname in group_names:
                if site.hook == "adaln" and gname != "all":
                    continue  # modulation vector has no spatial extent
                for direction in directions:
                    for lvl in (0, 1):
                        receiver, donor = DIRECTIONS[direction](lvl)
                        if receiver not in latents or donor not in latents:
                            continue
                        index = tokens(gname, step, (receiver, donor))
                        if index is not None and len(index) == 0 and site.hook != "adaln":
                            continue
                        region = region_for((receiver, donor))
                        metric_args = (clean[receiver]["p"], clean[donor]["p"], target(receiver), target(donor), t_unsafe, ctx(receiver), region)
                        mu, var = maha[sid]
                        s0 = use_steps[0] if step == "persistent" else int(step)
                        base_act = clean[receiver]["acts"][sid][s0]
                        donor_act = clean[donor]["acts"][sid][s0]
                        written_base = base_act[:, :] if index is None or base_act.shape[1] == 1 else base_act[:, torch.as_tensor(index, device=base_act.device)]
                        written_donor = donor_act[:, :] if index is None or donor_act.shape[1] == 1 else donor_act[:, torch.as_tensor(index, device=donor_act.device)]
                        cell_out: dict[str, Any] = {"n_tokens_written": int(written_donor.shape[1]), "region_size": int(len(region))}

                        def run(patches: dict, written: torch.Tensor | None = None) -> dict[str, float]:
                            m = patch_metrics(run_patches(receiver, patches), *metric_args)
                            if written is not None:
                                m["mahalanobis_written"] = mahalanobis_score(written[0], mu, var)
                                m["mahalanobis_receiver"] = mahalanobis_score(written_base[0], mu, var)
                            return m

                        cell_out["donor"] = run(donor_payload(sid, step, donor, index), written_donor)
                        cell_out["sham"] = run(donor_payload(sid, step, receiver, index))
                        cell_out["random"] = [
                            run(donor_payload(sid, step, receiver, index, values_fn=lambda s, acts, _g=generator: _rand_values(clean, receiver, donor, sid, s, index, _g)), None)
                            for _ in range(n_random)
                        ]
                        if step != "persistent":
                            cell_out["time_shift"] = [
                                run({(sid, int(step)): _patch_value(clean[donor]["acts"][sid][s2], index)})
                                for s2 in range(n_steps)
                                if s2 != int(step) and clean[donor]["acts"][sid][s2].shape == donor_act.shape
                            ]
                        if u.site_id != sid and clean[donor]["acts"][u.site_id][s0].shape == donor_act.shape:
                            cell_out["unrelated_site"] = run(donor_payload(u.site_id, step, donor, index))
                            cell_out["unrelated_site"]["site_id"] = u.site_id
                        if direction in HAZARD_DIRECTIONS:
                            other = (donor[0], 1 - donor[1])
                            if other in latents:
                                cell_out["main_effect_only"] = run(donor_payload(sid, step, other, index))
                            if null_present and receiver[0] == 0:
                                cell_out["null_factor"] = run(donor_payload(sid, step, (2, receiver[1]), index))
                            if ident_present and receiver[0] == 0:
                                cell_out[IDENTITY_CONTROL] = run(donor_payload(sid, step, (OBJECT_HAZARD, receiver[1]), index))
                        if index is not None and site.hook != "adaln":
                            other_idx = tokens(OTHER_GROUP.get(gname, "egg"), step, (receiver, donor))
                            if other_idx is not None and len(other_idx) > 0:
                                cell_out["wrong_group"] = run(
                                    donor_payload(sid, step, donor, index, values_fn=lambda s, acts, oi=other_idx, n=len(index): cyclic_assign(acts[:, torch.as_tensor(oi, device=acts.device)], n))
                                )
                        out.setdefault(sid, {}).setdefault(str(step), {}).setdefault(gname, {}).setdefault(direction, {})[f"fixed_{lvl}"] = cell_out

    # INT diagnostic: X alone, Y alone, X+Y at step 0, primary group/direction, receiver A=1
    if int_pairs:
        receiver, donor = DIRECTIONS[PRIMARY_DIRECTION](1)
        # primary group, falling back to the first non-empty requested group (bbox fallback has no gripper tokens)
        ordered = ([PRIMARY_GROUP] if PRIMARY_GROUP in group_names else []) + [g for g in group_names if g != PRIMARY_GROUP]
        gname, index = ordered[0], tokens(ordered[0], 0, (receiver, donor))
        for cand in ordered:
            idx = tokens(cand, 0, (receiver, donor))
            if idx is None or len(idx) > 0:
                gname, index = cand, idx
                break
        region = region_for((receiver, donor))
        metric_args = (clean[receiver]["p"], clean[donor]["p"], target(receiver), target(donor), t_unsafe, ctx(receiver), region)
        int_out = []
        for x_id, y_id in int_pairs:
            if x_id not in record or y_id not in record:
                continue
            if index is not None and len(index) == 0:
                continue
            px = {(x_id, 0): _patch_value(clean[donor]["acts"][x_id][0], index)}
            py = {(y_id, 0): _patch_value(clean[donor]["acts"][y_id][0], index)}
            rx = patch_metrics(run_patches(receiver, px), *metric_args)["recovery_region"]
            ry = patch_metrics(run_patches(receiver, py), *metric_args)["recovery_region"]
            rxy = patch_metrics(run_patches(receiver, {**px, **py}), *metric_args)["recovery_region"]
            int_out.append({"x": x_id, "y": y_id, "group": gname, "recovery_x": rx, "recovery_y": ry, "recovery_xy": rxy, "int": rxy - rx - ry})
        out["_int"] = int_out
    return out


def _rand_values(clean, receiver, donor, sid, step, index, generator):
    base = clean[receiver]["acts"][sid][step]
    don = clean[donor]["acts"][sid][step]
    if index is not None and base.shape[1] != 1:
        idx = torch.as_tensor(index, device=base.device)
        base, don = base[:, idx], don[:, idx]
    return norm_matched_random(base, don, generator)


# --------------------------------------------------------------------------- #
# Aggregation and selection
# --------------------------------------------------------------------------- #


def _vals(entry: Any, stat: str) -> list[float]:
    if entry is None:
        return []
    if isinstance(entry, list):
        return [e[stat] for e in entry if stat in e]
    return [entry[stat]] if stat in entry else []


def is_candidate(site: Site, gname: str, n_layers: int) -> tuple[bool, str]:
    if site.hook == "adaln":
        return False, "action_availability_control"
    if gname == "all" and site.hook not in COMPONENT_HOOKS:
        return False, "reference_whole_residual"
    if site.hook == "resid_post" and site.layer == n_layers - 1:
        return False, "reference_last_layer_resid"
    if gname in candidate_groups(_domain()) or site.hook in COMPONENT_HOOKS:
        return True, "candidate"
    return False, "reference"


def aggregate(
    per_pair: dict[str, dict[str, Any]],
    sites: list[Site],
    n_layers: int,
    n_boot: int,
    n_perm: int,
    rng: np.random.Generator,
    recovery_threshold: float,
    control_threshold: float,
    alpha: float,
    allow_missing_null: bool,
    confirm: bool = False,
) -> dict[str, Any]:
    pair_ids = sorted(per_pair)
    results: list[dict[str, Any]] = []
    did_vectors: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}  # (step, group, direction) -> site -> [n,1]

    for site in sites:
        sid = site.site_id
        steps = sorted({step for pid in pair_ids for step in per_pair[pid].get(sid, {})}, key=lambda s: (s == "persistent", str(s)))
        for step in steps:
            groups = sorted({g for pid in pair_ids for g in per_pair[pid][sid].get(step, {})})
            for gname in groups:
                cand, role = is_candidate(site, gname, n_layers)
                entry: dict[str, Any] = {
                    "site_id": sid, "layer": site.layer, "hook": site.hook, "step": step, "group": gname,
                    "order_key": list(site_order_key(site, 0 if step == "persistent" else int(step))),
                    "is_candidate": cand, "role": role, "directions": {},
                }
                for direction in DIRECTIONS:
                    rows = [per_pair[pid][sid][step][gname].get(direction) for pid in pair_ids]
                    rows = [r for r in rows if r]
                    if not rows:
                        continue
                    summary: dict[str, Any] = {}
                    for kind in _kinds():
                        per_scene = []
                        for r in rows:
                            v = [x for l in (0, 1) for x in _vals(r.get(f"fixed_{l}", {}).get(kind), "recovery_region")]
                            per_scene.append(float(np.mean(v)) if v else np.nan)
                        arr = np.asarray(per_scene)
                        if np.all(np.isnan(arr)):
                            continue
                        summary[kind] = {
                            "recovery_region": cluster_bootstrap(arr, n_boot, rng),
                            "recovery_region_median": float(np.nanmedian(arr)),
                            "n_scenes": int(np.sum(~np.isnan(arr))),
                            "per_scene": {pid: (None if np.isnan(v) else float(v)) for pid, v in zip(pair_ids, arr)},
                        }
                        if kind in ("null_factor", IDENTITY_CONTROL):
                            donor_arr = np.asarray([summary["donor"]["per_scene"][pid] if summary["donor"]["per_scene"][pid] is not None else np.nan for pid in pair_ids])
                            diff = donor_arr - arr
                            keep = ~np.isnan(diff)
                            summary[kind]["n_null_scenes" if kind == "null_factor" else "n_identity_scenes"] = int(keep.sum())
                            if keep.sum() >= 2:
                                res = sign_flip_maxt({sid: diff[keep].reshape(-1, 1)}, n_perm, rng, statistic=lambda v, s: v[:, 0] if s is None else v[:, 0] * s)
                                summary[kind]["donor_minus_null_test" if kind == "null_factor" else "donor_minus_identity_test"] = {**res["sites"][sid], "n_scenes": res["n_scenes"], "n_perm": res["n_perm"], "exact": res["exact"]}
                        if kind == "donor":
                            for stat in ("recovery_all", "unsafe_future_delta", "target_shift", "cem_l2_cost_delta", "corruption", "edit_rms", "mahalanobis_written", "mahalanobis_receiver"):
                                vals = np.asarray([float(np.mean(v)) if (v := [x for l in (0, 1) for x in _vals(r.get(f"fixed_{l}", {}).get("donor"), stat)]) else np.nan for r in rows])
                                summary[kind][stat] = cluster_bootstrap(vals, n_boot, rng)
                            # patch-effect DiD: receiver A=1 minus receiver A=0 (hazard directions)
                            if direction in HAZARD_DIRECTIONS:
                                did = []
                                for r in rows:
                                    r1 = _vals(r.get("fixed_1", {}).get("donor"), "recovery_region")
                                    r0 = _vals(r.get("fixed_0", {}).get("donor"), "recovery_region")
                                    did.append(r1[0] - r0[0] if r1 and r0 else np.nan)
                                did = np.asarray(did)
                                summary[kind]["patch_effect_did"] = cluster_bootstrap(did, n_boot, rng)
                                summary[kind]["patch_effect_did_per_scene"] = {pid: (None if np.isnan(v) else float(v)) for pid, v in zip(pair_ids, did)}
                                did_vectors.setdefault((str(step), gname, direction), {})[sid] = did.reshape(-1, 1)
                    entry["directions"][direction] = summary
                results.append(entry)

    # sign-flip max-T over sites within (step, group, direction) on the patch-effect DiD
    def scalar_stat(v: np.ndarray, signs: np.ndarray | None) -> np.ndarray:
        x = v[:, 0]
        return x if signs is None else x * signs

    perm_out: dict[str, Any] = {}
    for (step, gname, direction), per_site in did_vectors.items():
        clean_sites = {sid: v for sid, v in per_site.items() if not np.any(np.isnan(v))}
        if not clean_sites:
            continue
        res = sign_flip_maxt(clean_sites, n_perm, rng, statistic=scalar_stat)
        perm_out[f"{step}|{gname}|{direction}"] = {k: v for k, v in res.items() if k != "sites"}
        for entry in results:
            if str(entry["step"]) == step and entry["group"] == gname and direction in entry["directions"] and entry["site_id"] in res["sites"]:
                entry["directions"][direction]["donor"]["patch_effect_did_test"] = res["sites"][entry["site_id"]]

    # liveness + selection at the primary group
    for entry in results:
        d = entry["directions"]
        prim = d.get(PRIMARY_DIRECTION, {}).get("donor", {})
        test = prim.get("patch_effect_did_test", {})
        p = test.get("p_maxt_fwer")
        sig = bool(p is not None and not np.isnan(p) and p < alpha)
        null_ctrl = d.get(PRIMARY_DIRECTION, {}).get("null_factor", {})
        null_med = null_ctrl.get("recovery_region_median")
        null_ok = (null_med is not None and null_med < control_threshold) or (null_med is None and allow_missing_null)
        entry["n_null_scenes"] = int(null_ctrl.get("n_null_scenes", 0))
        entry["null_factor_status"] = "missing" if null_med is None else ("pass" if null_med < control_threshold else "fail")
        entry["interaction_live"] = bool(entry["is_candidate"] and _is_primary(entry["group"]) and sig and null_ok)
        if _domain() == "driving":
            ident = d.get(PRIMARY_DIRECTION, {}).get(IDENTITY_CONTROL, {})
            entry["identity_donor_recovery_median"] = ident.get("recovery_region_median")
            entry["n_identity_scenes"] = int(ident.get("n_identity_scenes", 0))
        both = [d.get(k, {}).get("donor", {}).get("recovery_region_median", -1.0) for k in HAZARD_DIRECTIONS]
        entry["recovery_both_directions_ok"] = bool(all(b >= recovery_threshold for b in both))
        entry["controls_below_threshold"] = bool(
            all(d.get(PRIMARY_DIRECTION, {}).get(k, {}).get("recovery_region_median", 0.0) < control_threshold for k in CONTROLS if k in d.get(PRIMARY_DIRECTION, {}))
        )

    selection: dict[str, Any] = {"selected": None, "band": [], "rule": "earliest interaction-live site with >=threshold recovery in both hazard directions; band = maximal contiguous run of live sites containing it"}
    if not confirm:
        primary = sorted((e for e in results if _is_primary(e["group"]) and e["is_candidate"]), key=lambda e: tuple(e["order_key"]))
        for step in sorted({str(e["step"]) for e in primary}):
            chain = [e for e in primary if str(e["step"]) == step]
            for i, e in enumerate(chain):
                if e["interaction_live"] and e["recovery_both_directions_ok"]:
                    lo = i
                    while lo - 1 >= 0 and chain[lo - 1]["interaction_live"]:
                        lo -= 1
                    hi = i
                    while hi + 1 < len(chain) and chain[hi + 1]["interaction_live"]:
                        hi += 1
                    band = chain[lo : hi + 1]
                    selection["selected"] = {"site_id": e["site_id"], "step": e["step"], "group": PRIMARY_GROUP}
                    selection["band"] = [
                        {
                            "site_id": b["site_id"], "step": b["step"],
                            "per_scene_stability": _stability(b),
                            "recovery_median": b["directions"][PRIMARY_DIRECTION]["donor"]["recovery_region_median"],
                        }
                        for b in band
                    ]
                    break
            if selection["selected"]:
                break
    return {"site_results": results, "permutation": perm_out, "selection": selection}


def _is_primary(group: str) -> bool:
    """The primary group under the current domain, accepting the egg-name alias (driving)."""

    return canon(group, _domain()) == canon(PRIMARY_GROUP, _domain())


def focus_tables(per_pair: dict[str, dict[str, Any]], focus: list[str]) -> dict[str, Any]:
    """Per-scene recovery (donor and every control) and patch-effect DiD for focus sites."""

    pair_ids = sorted(per_pair)
    out: dict[str, Any] = {}
    for sid in focus:
        steps = sorted({s for pid in pair_ids for s in per_pair[pid].get(sid, {})}, key=lambda s: (s == "persistent", str(s)))
        for step in steps:
            groups = sorted({g for pid in pair_ids for g in per_pair[pid][sid].get(step, {})})
            for gname in groups:
                directions = sorted({dname for pid in pair_ids for dname in per_pair[pid][sid][step].get(gname, {})})
                for direction in directions:
                    table = []
                    for pid in pair_ids:
                        cell = per_pair[pid].get(sid, {}).get(step, {}).get(gname, {}).get(direction)
                        if not cell:
                            continue
                        row: dict[str, Any] = {"pair_id": pid}
                        for lvl in ("fixed_0", "fixed_1"):
                            c = cell.get(lvl, {})
                            for kind in _kinds():
                                v = _vals(c.get(kind), "recovery_region")
                                if v:
                                    row[f"{lvl}_{kind}"] = float(np.mean(v))
                            if "donor" in c:
                                row[f"{lvl}_unsafe_future_delta"] = c["donor"]["unsafe_future_delta"]
                                row[f"{lvl}_cem_l2_cost_delta"] = c["donor"]["cem_l2_cost_delta"]
                        if "fixed_1_donor" in row and "fixed_0_donor" in row:
                            row["patch_effect_did"] = row["fixed_1_donor"] - row["fixed_0_donor"]
                        table.append(row)
                    out.setdefault(sid, {}).setdefault(str(step), {}).setdefault(gname, {})[direction] = table
    return out


def _stability(entry: dict[str, Any]) -> dict[str, Any]:
    per = entry["directions"][PRIMARY_DIRECTION]["donor"].get("patch_effect_did_per_scene", {})
    vals = [v for v in per.values() if v is not None]
    return {"fraction_scenes_positive_did": (float(np.mean([v > 0 for v in vals])) if vals else None), "n_scenes": len(vals)}


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
    parser.add_argument("--model-label", default="JEPA-WM DROID site patching")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sites", nargs="*", default=None)
    parser.add_argument("--hooks", nargs="*", default=list(COMPONENT_HOOKS))
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--steps", type=int, nargs="*", default=None)
    parser.add_argument("--mode", choices=["one_shot", "persistent", "both"], default="one_shot")
    parser.add_argument("--groups", nargs="*", default=[PRIMARY_GROUP, "egg", "all"])
    parser.add_argument("--group-frame-offset", type=int, default=0)
    parser.add_argument("--directions", nargs="*", default=list(DIRECTIONS))
    parser.add_argument("--n-random", type=int, default=3)
    parser.add_argument("--unrelated-offset", type=int, default=6)
    parser.add_argument("--int-pairs", nargs="*", default=None, help="e.g. L03.mlp_out+L05.mlp_out")
    parser.add_argument("--focus-sites", nargs="*", default=None,
                        help="full battery on these sites: all directions, one-shot AND persistent, INT with L06.attn_out/L08.attn_out/L05.mlp_out")
    parser.add_argument("--focus-int-partners", nargs="*", default=["L06.attn_out", "L08.attn_out", "L05.mlp_out"])
    # donor-free conceptor intervention (conceptor_patch.py, COAST arXiv:2605.17144)
    parser.add_argument("--conceptor-dir", type=Path, default=None, help="dir with <site>__s<step>__<group>.npz conceptors; switches to conceptor mode")
    parser.add_argument("--conceptor-mode", nargs="+", choices=["strengthen", "suppress", "failure_suppress"], default=["strengthen"])
    parser.add_argument("--beta", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    parser.add_argument("--conceptor-key", default="C_safety", help="C_safety | C_int | C_action")
    parser.add_argument("--band-sites", nargs="*", default=None, help="apply the conceptor at these sites simultaneously (distributed-pathway test)")
    parser.add_argument("--equivalence-margin", type=float, default=0.05, help="TOST margin on non-target recovery_unsafe")
    parser.add_argument("--recovery-threshold", type=float, default=0.30)
    parser.add_argument("--control-threshold", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--allow-missing-null-factor", action="store_true")
    parser.add_argument("--seeds-file", type=Path, default=None)
    parser.add_argument("--allow-calibration-seeds", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--band-file", type=Path, default=None)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-perm", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--domain", choices=list(DOMAINS), default="egg", help="driving = v0.8 MetaDrive scenes (hazard/corridor groups, level-3 identity donor)")
    args = parser.parse_args()
    _DOMAIN[0] = set_domain(args.domain)

    if args.confirm and args.band_file is None:
        raise SystemExit("--confirm requires --band-file")
    t0 = time.time()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    wm, _ = load_model(args.repo, args.config, args.checkpoint, args.model_name, str(device))
    predictor = predictor_of(wm)
    n_layers = n_layers_of(predictor)
    n_spatial = spatial_tokens_of(wm)

    if args.conceptor_dir is not None:
        from conceptor_patch import run as run_conceptor

        run_conceptor(args, wm, predictor, n_layers, n_spatial, device, t0)
        return

    band: dict[str, Any] | None = None
    if args.confirm:
        band = json.loads(args.band_file.read_text())
        sites = [Site.parse(b["site_id"]) for b in band["band"]]
        steps = sorted({int(b["step"]) for b in band["band"] if b["step"] != "persistent"}) or None
        group_names = [band.get("group", PRIMARY_GROUP)]
        mode = "persistent" if any(b["step"] == "persistent" for b in band["band"]) else "one_shot"
    elif args.focus_sites:
        sites = [Site.parse(s) for s in args.focus_sites]
        steps, group_names, mode = args.steps, args.groups, "both"
    else:
        if args.sites:
            sites = [Site.parse(s) for s in args.sites]
        else:
            layers = n_layers if args.max_layers is None else min(n_layers, args.max_layers)
            sites = all_sites(layers, args.hooks, include_embed=False)
        steps, group_names, mode = args.steps, args.groups, args.mode
    directions = list(DIRECTIONS) if args.focus_sites else args.directions
    int_pairs = [tuple(p.split("+")) for p in (args.int_pairs or []) if "+" in p]
    if args.focus_sites:
        for s in sites:
            for partner in args.focus_int_partners:
                if partner != s.site_id and (s.site_id, partner) not in int_pairs:
                    int_pairs.append((s.site_id, partner))

    pairs = select_pairs(merge_manifests(args.artifacts), args.seeds_file, args.allow_calibration_seeds)
    if not pairs:
        raise SystemExit("no complete quartets selected (calibration seeds are excluded by default)")
    null_status = null_factor_status(pairs)
    null_pairs = set(null_factor_pairs(pairs))
    null_present = null_status != "none"
    ident_status = identity_status(pairs) if args.domain == "driving" else "none"
    ident_pairs = set(identity_pairs(pairs)) if args.domain == "driving" else set()
    modes = ["one_shot", "persistent"] if mode == "both" else [mode]

    per_pair: dict[str, dict[str, Any]] = {}
    group_sources: set[str] = set()
    for pair_id in sorted(pairs):
        cell_keys = list(CELL_ORDER) + (list(NULL_CELLS) if pair_id in null_pairs else []) + (list(IDENTITY_CELLS) if pair_id in ident_pairs else [])
        latents = {key: encode_cell(wm, *pairs[pair_id][key], device) for key in cell_keys}
        groups = {key: load_cell_groups(*pairs[pair_id][key]) for key in cell_keys}
        group_sources |= {g.source for g in groups.values()}
        merged: dict[str, Any] = {}
        for m in modes:
            res = evaluate_pair(
                wm, predictor, n_spatial, n_layers, latents, groups, sites, steps, group_names, args.n_random,
                args.unrelated_offset, generator, directions, m, args.group_frame_offset, int_pairs if m == modes[0] else None,
            )
            for sid, by_step in res.items():
                if sid == "_int":
                    merged["_int"] = by_step
                    continue
                merged.setdefault(sid, {}).update(by_step)
        per_pair[pair_id] = merged
        print(f"[patch] pair {pair_id} done ({time.time() - t0:.1f}s)", file=sys.stderr)

    agg = aggregate(
        {pid: {k: v for k, v in r.items() if k != "_int"} for pid, r in per_pair.items()},
        sites, n_layers, args.n_boot, args.n_perm, rng, args.recovery_threshold, args.control_threshold, args.alpha,
        args.allow_missing_null_factor, confirm=args.confirm,
    )
    seeds = sorted({int(pairs[pid][CELL_ORDER[0]][1]["seed"]) for pid in pairs})
    out = {
        "model_label": args.model_label,
        "model_name": args.model_name,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "config_sha256": sha256_file(args.config),
        "artifacts": [str(p) for p in args.artifacts],
        "pair_ids": sorted(pairs),
        "seeds": seeds,
        "n_scenes": len(pairs),
        "calibration_seeds_included": bool(set(seeds) & set(CALIBRATION_SEEDS)),
        "confirmation_run": args.confirm,
        "band_file": str(args.band_file) if args.band_file else None,
        "sites": [s.site_id for s in sites],
        "n_sites": len(sites),
        "steps": steps,
        "mode": mode,
        "groups": group_names,
        "token_groups": "masks" if group_sources == {"masks"} else ("bbox_fallback" if group_sources == {"bbox_fallback"} else "mixed"),
        "group_frame_offset": args.group_frame_offset,
        "null_factor_present": null_present,
        "null_factor_status": null_status,
        "n_null_scenes": len(null_pairs),
        "null_scene_pair_ids": sorted(null_pairs),
        **({"domain": args.domain, "token_group_source": alias_note(args.domain), "region_groups": list(REGION_GROUPS[args.domain]),
            "primary_group_native": DOMAIN_PRIMARY_GROUP[args.domain], "identity_factor": ident_status, "n_identity_scenes": len(ident_pairs),
            "identity_scene_pair_ids": sorted(ident_pairs)} if args.domain != "egg" else {}),
        "focus_sites": args.focus_sites,
        "int_pairs": [list(p) for p in int_pairs],
        "n_random": args.n_random,
        "unrelated_offset": args.unrelated_offset,
        "gate": {
            "recovery_threshold": args.recovery_threshold,
            "control_threshold": args.control_threshold,
            "alpha": args.alpha,
            "primary_group": PRIMARY_GROUP,
            "primary_direction": PRIMARY_DIRECTION,
            "liveness": f"H-only patch-effect DiD at {PRIMARY_GROUP} significant under sign-flip max-T AND H0' null donor median recovery < control_threshold"
            + (" (driving: gripper_corridor = corridor alias; the level-3 identity donor is reported, not a liveness criterion)" if args.domain == "driving" else ""),
            "allow_missing_null_factor": args.allow_missing_null_factor,
        },
        **agg,
        "int_diagnostic": {pid: r.get("_int", []) for pid, r in per_pair.items()},
        "focus_tables": focus_tables({pid: {k: v for k, v in r.items() if k != "_int"} for pid, r in per_pair.items()}, args.focus_sites) if args.focus_sites else None,
        "per_pair": {pid: {k: v for k, v in r.items() if k != "_int"} for pid, r in per_pair.items()},
        "interpretation_scope": (
            f"Latent-forecast endpoint on the {'+'.join(REGION_GROUPS[args.domain])} region; contact/clearance and fresh candidate-ranking "
            "endpoints require a readout fitted on discovery data and are not evaluated here. "
            + ZERO_SHOT_SCOPE
            + (" Token groups are bbox fallbacks (no masks): gripper/corridor groups are empty, so the primary "
               "gripper_corridor analysis is not available." if "bbox_fallback" in group_sources else "")
            + {"none": " No H0' null-factor cells: liveness cannot be established"
                       + (" (allow_missing_null_factor set: treated as satisfied)." if args.allow_missing_null_factor else "."),
               "partial": f" H0' null-factor cells exist for {len(null_pairs)}/{len(pairs)} scenes: the null donor control and liveness use that subset.",
               "full": ""}[null_status]
        ),
        "runtime_s": time.time() - t0,
        "torch_version": torch.__version__,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "patch_results.json").write_text(canonical_json(json_safe(out)) + "\n")
    if not args.confirm and agg["selection"]["selected"]:
        (args.output / "band.json").write_text(json.dumps({"group": PRIMARY_GROUP, "selected": agg["selection"]["selected"], "band": agg["selection"]["band"],
                                                            "frozen_on_seeds": seeds, "null_factor_status": null_status, "n_null_scenes": len(null_pairs)}, indent=2))
    brief = []
    for e in agg["site_results"]:
        d = e["directions"]
        row = {"site": e["site_id"], "step": e["step"], "group": e["group"], "role": e["role"], "live": e["interaction_live"]}
        for k in DIRECTIONS:
            if k in d and "donor" in d[k]:
                row[k[:12]] = round(d[k]["donor"]["recovery_region_median"], 3)
        brief.append(row)
    print(json.dumps({"n_scenes": len(pairs), "selection": agg["selection"]["selected"], "band_len": len(agg["selection"]["band"]), "sites": brief[:60]}, indent=1))


if __name__ == "__main__":
    main()
