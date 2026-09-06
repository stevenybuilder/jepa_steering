#!/usr/bin/env python3
"""Donor-free conceptor interventions (COAST, arXiv:2605.17144) on the JEPA-WM
predictor, as an intervention arm of the sonar search.

Conceptor files (written by ``geometry_conceptor.py``)::

    <conceptor_dir>/<site>__s<step>__<group>.npz
        C_safety, C_int, C_action, C_random_matched : d x d
        mean : d      alpha : scalar      quota : scalar     (optional: C_failure, mean_diff)

At the hooked site/step, for the tokens of the group, the centred activation is
transformed by a d x d operator ``M`` (row-vector convention ``h' = mean + (h - mean) M``):

- ``strengthen``        ``M = (1 - beta) I + beta C``
- ``suppress``          ``M = I - beta C``
- ``failure_suppress``  suppress with ``C_failure`` (falls back to ``C_int``)

The operator is applied to EVERY cell of a scene (H0, H1 and H0' x A) so that
selectivity can be measured: the endpoint metrics (recovery toward the true
(H1,A1) future on the egg+gripper region, ``unsafe_future_delta``,
``cem_l2_cost_delta``, ``corruption``) are aggregated separately for H1 cells
(target) and for H0 / H0' cells (non-target), which must stay inside an
equivalence margin (ratio + TOST-style check with the scene-clustered CI).

Controls at the same site/beta: ``random_matched`` (``C_random_matched``,
matched spectrum), ``rank_one`` (additive mean-difference edit of equal norm),
``wrong_site`` (layer + offset, same hook), ``wrong_step``, ``wrong_group``.
Off-manifold diagnostics: diagonal Mahalanobis and nearest-neighbour distance of
the written tokens to the clean tokens of the other cells.

Band mode (``--band-sites``): the conceptor of each listed site is applied at all
of them simultaneously (distributed-pathway hypothesis) and reported next to the
single-site effects.  One-shot vs persistent as in ``patch_site.py``.  Sign-flip
tests over scenes on selectivity (H1 effect minus H0 / H0' effect) with max-T
across sites x beta within a (mode, group, step).

References: arXiv:2605.17144 (COAST), arXiv:2309.16042, arXiv:2511.04638,
arXiv:2510.00845.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cgs_stats import cluster_bootstrap, json_safe, sign_flip_maxt  # noqa: E402
from localize_interaction import NULL_CELLS, encode_cell, merge_manifests, null_factor_pairs, null_factor_status, rms, select_pairs  # noqa: E402
from patch_site import PRIMARY_GROUP, endpoint, mahalanobis_score, mahalanobis_stats  # noqa: E402
from predictor_hooks import ZERO_SHOT_SCOPE, PredictorPatcher, PredictorRecorder, Site, n_layers_of, predictor_of, site_order_key, spatial_tokens_of, unrelated_site, unroll_from_latents  # noqa: E402
from protocol import CELL_ORDER, canonical_json, sha256_file  # noqa: E402
from token_groups import CellGroups, frame_for_step, load_cell_groups, union  # noqa: E402


MODES = ("strengthen", "suppress", "failure_suppress")
CONTROLS = ("random_matched", "rank_one", "wrong_site", "wrong_step", "wrong_group")
OTHER_GROUP = {"egg": "gripper_corridor", "gripper_corridor": "egg", "gripper": "egg", "corridor": "egg", "background": "egg", "all": "egg"}


# --------------------------------------------------------------------------- #
# Operators
# --------------------------------------------------------------------------- #


def operator(C: torch.Tensor, beta: float, mode: str) -> torch.Tensor:
    eye = torch.eye(C.shape[0], dtype=C.dtype, device=C.device)
    if mode == "strengthen":
        return (1.0 - beta) * eye + beta * C
    if mode in ("suppress", "failure_suppress"):
        return eye - beta * C
    raise ValueError(f"unknown mode {mode}")


def apply_operator(h: torch.Tensor, M: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    """``h`` [..., d] rows -> ``mean + (h - mean) M``."""

    return mean + (h.to(M.dtype) - mean) @ M


def rank_one_edit(h: torch.Tensor, direction: torch.Tensor, target_norm: torch.Tensor, sign: float) -> torch.Tensor:
    """Additive edit along the unit ``direction`` with per-batch Frobenius norm ``target_norm``."""

    u = direction / direction.norm().clamp_min(1e-12)
    k = h.shape[1]
    step = sign * target_norm.reshape(-1, 1, 1) / np.sqrt(max(k, 1))
    return h + step * u.reshape(1, 1, -1)


def conceptor_key_for(mode: str, key: str, available: set[str]) -> str:
    if mode == "failure_suppress":
        return "C_failure" if "C_failure" in available else "C_int"
    return key


def load_conceptor(path: Path, device: torch.device) -> dict[str, Any]:
    with np.load(path) as handle:
        out: dict[str, Any] = {}
        for k in handle.files:
            arr = np.asarray(handle[k])
            if arr.ndim >= 1 and k.startswith("C_") or k in ("mean", "mean_diff"):
                out[k] = torch.as_tensor(arr, dtype=torch.float32, device=device)
            else:
                out[k] = float(arr) if arr.ndim == 0 else arr.tolist()
    return out


def conceptor_path(directory: Path, site_id: str, step: int, group: str) -> Path:
    return directory / f"{site_id}__s{step}__{group}.npz"


def nearest_neighbour_distance(x: torch.Tensor, bank: torch.Tensor) -> float:
    """Mean over rows of ``x`` [n, d] of the L2 distance to the nearest row of ``bank`` [m, d]."""

    if bank.shape[0] == 0 or x.shape[0] == 0:
        return float("nan")
    d = torch.cdist(x.float(), bank.float())
    return float(d.min(dim=1).values.mean().item())


# --------------------------------------------------------------------------- #
# Endpoint metrics per cell (no donor)
# --------------------------------------------------------------------------- #


def _sel(x: torch.Tensor, index: np.ndarray | None) -> torch.Tensor:
    if index is None or len(index) == 0:
        return x
    return x[torch.as_tensor(index, dtype=torch.long, device=x.device)]


def cell_metrics(p_new: torch.Tensor, p_base: torch.Tensor, t_own: torch.Tensor, t_unsafe: torch.Tensor, z_ctx: torch.Tensor, region: np.ndarray) -> dict[str, float]:
    gap = rms(_sel(p_base, region) - _sel(t_unsafe, region))
    residual = rms(_sel(p_new, region) - _sel(t_unsafe, region))
    cost = lambda p: float(torch.mean((p - t_own) ** 2).item())  # noqa: E731
    return {
        "recovery_unsafe": float(1.0 - residual / max(gap, 1e-12)),
        "unsafe_future_delta": float(residual - gap),
        "cem_l2_cost_delta": cost(p_new) - cost(p_base),
        "corruption": float(rms(_sel(p_new, region) - _sel(t_own, region)) / max(rms(_sel(p_base, region) - _sel(t_own, region)), 1e-12)),
        "displacement_delta": float(rms(p_new - z_ctx) - rms(p_base - z_ctx)),
        "edit_rms": float(rms(p_new - p_base)),
    }


# --------------------------------------------------------------------------- #
# Per-scene evaluation
# --------------------------------------------------------------------------- #


def evaluate_scene(
    wm: Any,
    predictor: torch.nn.Module,
    n_spatial: int,
    n_layers: int,
    latents: dict[tuple[int, int], dict[str, torch.Tensor]],
    groups: dict[tuple[int, int], CellGroups],
    conceptors: dict[tuple[str, int, str], dict[str, Any]],
    sites: list[Site],
    band: list[Site] | None,
    steps: list[int],
    group_names: list[str],
    betas: list[float],
    mode: str,
    key: str,
    persistent: bool,
    wrong_site_offset: int,
    frame_offset: int,
    controls: tuple[str, ...] = CONTROLS,
) -> dict[str, Any]:
    """Return ``{config_id: {"cells": {cell_id: {"main": m, "<control>": m}}, "meta": {...}}}``."""

    keys = list(latents)
    record = {s.site_id: s for s in sites}
    for s in sites:
        u = unrelated_site(s, n_layers, wrong_site_offset)
        record.setdefault(u.site_id, u)
    for s in band or []:
        record.setdefault(s.site_id, s)
    clean: dict[tuple[int, int], dict[str, Any]] = {}
    for k in keys:
        rec = PredictorRecorder(predictor, record.values(), n_spatial, store_device=latents[k]["z_context"].device)
        pred = unroll_from_latents(wm, latents[k]["z_context"], latents[k]["actions"], recorder=rec)
        clean[k] = {"acts": rec.acts, "p": endpoint(pred), "n_steps": rec.n_steps}
    n_steps = clean[keys[0]]["n_steps"]
    use_steps = [s for s in steps if s < n_steps]
    maha = {sid: mahalanobis_stats(torch.cat([clean[k]["acts"][sid][s][0] for k in keys for s in range(n_steps)], dim=0)) for sid in record}

    def tokens(gname: str, step: int, cells) -> np.ndarray | None:
        if gname == "all":
            return None
        return union(*(groups[k].group(frame_for_step(step, frame_offset), gname) for k in cells))

    def region_for(cells) -> np.ndarray:
        return union(*(union(groups[k].group(n_steps, "egg"), groups[k].group(n_steps, "gripper")) for k in cells))

    def target(k):
        z = latents[k]["z_future"][:, -1]
        return z.reshape(-1, z.shape[-1]).float()

    def ctx(k):
        z = latents[k]["z_context"][:, -1]
        return z.reshape(-1, z.shape[-1]).float()

    t_unsafe = target((1, 1))
    region = region_for(tuple(keys))

    def run(cell, patches: dict) -> torch.Tensor:
        patcher = PredictorPatcher(predictor, patches, n_spatial)
        pred = unroll_from_latents(wm, latents[cell]["z_context"], latents[cell]["actions"], patcher=patcher)
        if len(patcher.applied) != len(patches):
            raise RuntimeError(f"patch not applied at {sorted(patches)}")
        return endpoint(pred)

    def payload(cell, sid: str, step: int, index: np.ndarray | None, values: torch.Tensor):
        acts = clean[cell]["acts"][sid][step]
        if index is None or acts.shape[1] == 1:
            return values
        return (torch.as_tensor(index, dtype=torch.long, device=acts.device), values)

    def written(cell, sid: str, step: int, index: np.ndarray | None) -> torch.Tensor:
        acts = clean[cell]["acts"][sid][step]
        if index is None or acts.shape[1] == 1:
            return acts
        return acts[:, torch.as_tensor(index, dtype=torch.long, device=acts.device)]

    def bank_for(cell, sid: str, step: int) -> torch.Tensor:
        return torch.cat([clean[k]["acts"][sid][step][0] for k in keys if k != cell], dim=0)

    def conceptor_for(sid: str, step: int, gname: str) -> dict[str, Any] | None:
        return conceptors.get((sid, step, gname)) or conceptors.get((sid, 0, gname))

    def mean_diff_direction(sid: str, step: int, gname: str, index: np.ndarray | None, con: dict[str, Any]) -> torch.Tensor:
        if "mean_diff" in con:
            return con["mean_diff"]
        h1 = torch.cat([written(k, sid, step, index)[0] for k in keys if k[0] == 1], dim=0).mean(dim=0)
        h0 = torch.cat([written(k, sid, step, index)[0] for k in keys if k[0] == 0], dim=0).mean(dim=0)
        return h1 - h0

    def evaluate(cell, edits: list[tuple[str, int, np.ndarray | None, str]], beta: float, op_mode: str, C_key: str, control_kind: str | None = None) -> dict[str, float]:
        """``edits``: list of (site_id, step, index, conceptor_site_step_group_key) to apply together."""

        patches: dict = {}
        diag: dict[str, float] = {}
        for sid, step, index, cgroup in edits:
            con_sid, con_step = cgroup
            con = conceptor_for(con_sid, con_step, gname_ref[0])
            if con is None:
                return {}
            C = con[C_key]
            base = written(cell, sid, step, index)
            if control_kind == "rank_one":
                # equal-norm additive edit along the mean-difference direction
                M = operator(C, beta, op_mode)
                edited = apply_operator(base, M, con["mean"])
                norm = (edited - base).flatten(1).norm(dim=1)
                sign = 1.0 if op_mode == "strengthen" else -1.0
                values = rank_one_edit(base, mean_diff_direction(sid, step, gname_ref[0], index, con), norm, sign)
            else:
                M = operator(C, beta, op_mode)
                values = apply_operator(base, M, con["mean"])
            patches[(sid, step)] = payload(cell, sid, step, index, values.to(base.dtype))
            if not diag:
                mu, var = maha[sid]
                diag = {
                    "mahalanobis_written": mahalanobis_score(values[0], mu, var),
                    "mahalanobis_receiver": mahalanobis_score(base[0], mu, var),
                    "nn_distance_written": nearest_neighbour_distance(values[0], bank_for(cell, sid, step)),
                    "nn_distance_receiver": nearest_neighbour_distance(base[0], bank_for(cell, sid, step)),
                    "activation_edit_rms": rms(values - base),
                }
        p_new = run(cell, patches)
        m = cell_metrics(p_new, clean[cell]["p"], target(cell), t_unsafe, ctx(cell), region)
        m.update(diag)
        return m

    out: dict[str, Any] = {}
    gname_ref = [group_names[0]]
    configs: list[tuple[str, list[Site]]] = [(s.site_id, [s]) for s in sites]
    if band:
        configs.append(("band:" + "+".join(s.site_id for s in band), list(band)))
    for cfg_id, cfg_sites in configs:
        for step in use_steps:
            step_ids = use_steps if persistent else [step]
            if persistent and step != use_steps[0]:
                continue
            step_label = "persistent" if persistent else str(step)
            for gname in group_names:
                gname_ref[0] = gname
                index = tokens(gname, step, tuple(keys))
                if index is not None and len(index) == 0:
                    continue
                avail = {k for s in cfg_sites for st in step_ids for k in (conceptor_for(s.site_id, st, gname) or {})}
                C_key = conceptor_key_for(mode, key, avail)
                if not all(conceptor_for(s.site_id, st, gname) for s in cfg_sites for st in step_ids):
                    continue
                for beta in betas:
                    cid = f"{cfg_id}|s{step_label}|{gname}|{mode}|b{beta}"
                    cells_out: dict[str, Any] = {}
                    for cell in keys:
                        cell_name = f"h{cell[0]}a{cell[1]}"
                        edits = [(s.site_id, st, tokens(gname, st, tuple(keys)), (s.site_id, st)) for s in cfg_sites for st in step_ids]
                        res = {"main": evaluate(cell, edits, beta, mode, C_key)}
                        if "random_matched" in controls and "C_random_matched" in avail:
                            res["random_matched"] = evaluate(cell, edits, beta, mode, "C_random_matched")
                        if "rank_one" in controls:
                            res["rank_one"] = evaluate(cell, edits, beta, mode, C_key, control_kind="rank_one")
                        if "wrong_site" in controls and len(cfg_sites) == 1:
                            u = unrelated_site(cfg_sites[0], n_layers, wrong_site_offset)
                            if u.site_id != cfg_sites[0].site_id:
                                res["wrong_site"] = evaluate(cell, [(u.site_id, st, tokens(gname, st, tuple(keys)), (cfg_sites[0].site_id, st)) for st in step_ids], beta, mode, C_key)
                                res["wrong_site"]["site_id"] = u.site_id
                        if "wrong_step" in controls and not persistent and len(use_steps) > 1:
                            other = [st for st in use_steps if st != step][0]
                            res["wrong_step"] = evaluate(cell, [(s.site_id, other, tokens(gname, other, tuple(keys)), (s.site_id, step)) for s in cfg_sites], beta, mode, C_key)
                            res["wrong_step"]["step"] = other
                        if "wrong_group" in controls and gname != "all":
                            og = OTHER_GROUP.get(gname, "egg")
                            oidx = tokens(og, step, tuple(keys))
                            if oidx is not None and len(oidx) > 0:
                                res["wrong_group"] = evaluate(cell, [(s.site_id, st, tokens(og, st, tuple(keys)), (s.site_id, st)) for s in cfg_sites for st in step_ids], beta, mode, C_key)
                                res["wrong_group"]["group"] = og
                        cells_out[cell_name] = {k: v for k, v in res.items() if v}
                    out[cid] = {
                        "cells": cells_out,
                        "meta": {"config": cfg_id, "sites": [s.site_id for s in cfg_sites], "band": len(cfg_sites) > 1, "step": step_label, "group": gname,
                                 "mode": mode, "beta": beta, "conceptor_key": C_key, "n_tokens": int(len(index)) if index is not None else n_spatial,
                                 "quota": (conceptor_for(cfg_sites[0].site_id, step, gname) or {}).get("quota"),
                                 "alpha": (conceptor_for(cfg_sites[0].site_id, step, gname) or {}).get("alpha")},
                    }
    return out


# --------------------------------------------------------------------------- #
# Aggregation: selectivity, equivalence, sign-flip max-T
# --------------------------------------------------------------------------- #


def scene_effects(cells: dict[str, Any], kind: str, stat: str = "recovery_unsafe") -> dict[str, float]:
    def mean_over(prefix: str) -> float:
        v = [cells[c][kind][stat] for c in cells if c.startswith(prefix) and kind in cells[c] and stat in cells[c][kind]]
        return float(np.mean(v)) if v else float("nan")

    target = mean_over("h1")
    h0 = mean_over("h0")
    h2 = mean_over("h2")
    return {"target_h1": target, "nontarget_h0": h0, "nontarget_h2": h2, "selectivity_h0": target - h0, "selectivity_h2": target - h2,
            "ratio_h0": (h0 / target) if abs(target) > 1e-12 else float("nan"), "ratio_h2": (h2 / target) if abs(target) > 1e-12 else float("nan")}


def aggregate(per_scene: dict[str, dict[str, Any]], n_boot: int, n_perm: int, rng: np.random.Generator, margin: float, alpha: float, stat: str = "recovery_unsafe") -> dict[str, Any]:
    pair_ids = sorted(per_scene)
    cids = sorted({cid for pid in pair_ids for cid in per_scene[pid]})
    results: dict[str, Any] = {}
    sel_vectors: dict[tuple[str, str], dict[str, np.ndarray]] = {}  # (family, which) -> cid -> [n,1]
    for cid in cids:
        metas = [per_scene[pid][cid]["meta"] for pid in pair_ids if cid in per_scene[pid]]
        meta = metas[0]
        kinds = sorted({k for pid in pair_ids if cid in per_scene[pid] for c in per_scene[pid][cid]["cells"].values() for k in c})
        entry: dict[str, Any] = {"meta": meta, "kinds": {}}
        for kind in kinds:
            rows = {pid: scene_effects(per_scene[pid][cid]["cells"], kind, stat) for pid in pair_ids if cid in per_scene[pid]}
            summary: dict[str, Any] = {"n_scenes": len(rows)}
            for field in ("target_h1", "nontarget_h0", "nontarget_h2", "selectivity_h0", "selectivity_h2", "ratio_h0", "ratio_h2"):
                arr = np.asarray([rows[pid][field] for pid in rows])
                summary[field] = cluster_bootstrap(arr, n_boot, rng)
                summary[field]["per_scene"] = {pid: (None if np.isnan(rows[pid][field]) else rows[pid][field]) for pid in rows}
            # TOST-style equivalence for non-target effects: bootstrap CI inside [-margin, margin]
            for which in ("nontarget_h0", "nontarget_h2"):
                ci = summary[which]
                summary[f"equivalent_{which}"] = bool(ci["ci_low"] is not None and ci["ci_low"] > -margin and ci["ci_high"] < margin)
            for extra in ("cem_l2_cost_delta", "corruption", "unsafe_future_delta", "mahalanobis_written", "mahalanobis_receiver", "nn_distance_written", "nn_distance_receiver", "activation_edit_rms"):
                vals = np.asarray([scene_effects(per_scene[pid][cid]["cells"], kind, extra)["target_h1"] for pid in rows])
                summary[f"target_{extra}"] = cluster_bootstrap(vals, n_boot, rng)
            entry["kinds"][kind] = summary
            if kind == "main":
                fam = (meta["mode"], meta["group"], meta["step"])
                for which in ("selectivity_h0", "selectivity_h2"):
                    v = np.asarray([rows[pid][which] for pid in rows])
                    if not np.all(np.isnan(v)):
                        sel_vectors.setdefault((str(fam), which), {})[cid] = v[~np.isnan(v)].reshape(-1, 1)
        results[cid] = entry
    perm: dict[str, Any] = {}
    for (fam, which), per_cid in sel_vectors.items():
        n = {c: v.shape[0] for c, v in per_cid.items()}
        common = {c: v for c, v in per_cid.items() if v.shape[0] == max(n.values())}
        if not common:
            continue
        res = sign_flip_maxt(common, n_perm, rng, statistic=lambda v, s: v[:, 0] if s is None else v[:, 0] * s)
        perm[f"{fam}|{which}"] = {k: v for k, v in res.items() if k != "sites"}
        for cid, test in res["sites"].items():
            results[cid]["kinds"]["main"][f"{which}_test"] = {**test, "n_scenes": res["n_scenes"], "n_perm": res["n_perm"], "exact": res["exact"]}
    for cid, entry in results.items():
        main = entry["kinds"].get("main", {})
        t0 = main.get("selectivity_h0_test", {})
        entry["selective"] = bool(t0.get("p_maxt_fwer") is not None and t0["p_maxt_fwer"] == t0["p_maxt_fwer"] and t0["p_maxt_fwer"] < alpha
                                  and main.get("equivalent_nontarget_h0", False))
        entry["controls_selective"] = {k: bool(v.get("selectivity_h0", {}).get("ci_low") is not None and v["selectivity_h0"]["ci_low"] > 0) for k, v in entry["kinds"].items() if k != "main"}
    # band vs single comparison
    band_vs_single = []
    for cid, entry in results.items():
        if entry["meta"]["band"]:
            singles = {}
            for sid in entry["meta"]["sites"]:
                scid = cid.replace(entry["meta"]["config"], sid, 1)
                if scid in results:
                    singles[sid] = results[scid]["kinds"]["main"]["target_h1"]["point"]
            band_vs_single.append({"band": cid, "band_target_h1": entry["kinds"]["main"]["target_h1"]["point"], "band_selectivity_h0": entry["kinds"]["main"]["selectivity_h0"]["point"],
                                   "singles_target_h1": singles, "sum_singles": float(np.nansum(list(singles.values()))) if singles else None})
    return {"results": results, "permutation": perm, "band_vs_single": band_vs_single}


# --------------------------------------------------------------------------- #
# Entry point (called from patch_site.py when --conceptor-dir is given)
# --------------------------------------------------------------------------- #


def run(args: Any, wm: Any, predictor: torch.nn.Module, n_layers: int, n_spatial: int, device: torch.device, t0: float) -> None:
    rng = np.random.default_rng(args.seed)
    sites = [Site.parse(s) for s in (args.sites or [])]
    band = [Site.parse(s) for s in (args.band_sites or [])] or None
    if not sites and not band:
        raise SystemExit("--conceptor-dir requires --sites and/or --band-sites")
    steps = args.steps if args.steps else [0]
    group_names = args.groups
    pairs = select_pairs(merge_manifests(args.artifacts), args.seeds_file, args.allow_calibration_seeds)
    if not pairs:
        raise SystemExit("no complete quartets selected")
    null_status = null_factor_status(pairs)
    null_pairs = set(null_factor_pairs(pairs))
    # conceptors
    conceptors: dict[tuple[str, int, str], dict[str, Any]] = {}
    for s in list(sites) + list(band or []):
        for st in range(3):
            for g in group_names:
                pth = conceptor_path(args.conceptor_dir, s.site_id, st, g)
                if pth.exists():
                    conceptors[(s.site_id, st, g)] = load_conceptor(pth, device)
    if not conceptors:
        raise SystemExit(f"no conceptor files found under {args.conceptor_dir} for the requested sites/groups")
    modes = list(args.conceptor_mode)
    per_scene: dict[str, dict[str, Any]] = {}
    for pair_id in sorted(pairs):
        cell_keys = list(CELL_ORDER) + (list(NULL_CELLS) if pair_id in null_pairs else [])
        latents = {key: encode_cell(wm, *pairs[pair_id][key], device) for key in cell_keys}
        groups = {key: load_cell_groups(*pairs[pair_id][key]) for key in cell_keys}
        merged: dict[str, Any] = {}
        for mode in modes:
            for persistent in ([False, True] if args.mode == "both" else [args.mode == "persistent"]):
                merged.update(evaluate_scene(wm, predictor, n_spatial, n_layers, latents, groups, conceptors, sites, band, steps, group_names, list(args.beta),
                                             mode, args.conceptor_key, persistent, args.unrelated_offset, args.group_frame_offset))
        per_scene[pair_id] = merged
        print(f"[conceptor] pair {pair_id} done ({time.time() - t0:.1f}s)", file=sys.stderr)
    agg = aggregate(per_scene, args.n_boot, args.n_perm, rng, args.equivalence_margin, args.alpha)
    seeds = sorted({int(pairs[pid][CELL_ORDER[0]][1]["seed"]) for pid in pairs})
    out = {
        "intervention": "conceptor",
        "model_label": args.model_label, "model_name": args.model_name,
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256_file(args.checkpoint), "config_sha256": sha256_file(args.config),
        "artifacts": [str(p) for p in args.artifacts], "pair_ids": sorted(pairs), "seeds": seeds, "n_scenes": len(pairs),
        "null_factor_status": null_status, "n_null_scenes": len(null_pairs),
        "conceptor_dir": str(args.conceptor_dir), "conceptor_files": sorted(f"{k[0]}__s{k[1]}__{k[2]}" for k in conceptors),
        "modes": modes, "betas": list(args.beta), "conceptor_key": args.conceptor_key, "sites": [s.site_id for s in sites],
        "band_sites": [s.site_id for s in band] if band else None, "steps": steps, "mode": args.mode, "groups": group_names,
        "equivalence_margin": args.equivalence_margin, "alpha": args.alpha, "wrong_site_offset": args.unrelated_offset,
        "primary_stat": "recovery_unsafe (toward the true (H1,A1) future on the egg+gripper region); selectivity = H1 effect minus H0/H0' effect",
        **agg,
        "per_scene": per_scene,
        "interpretation_scope": ("Donor-free conceptor (COAST) intervention on the latent forecast; selectivity requires H0/H0' cells inside the equivalence margin. "
                                 + ZERO_SHOT_SCOPE + {"none": " No H0' cells.", "partial": f" H0' cells for {len(null_pairs)}/{len(pairs)} scenes.", "full": ""}[null_status]),
        "runtime_s": time.time() - t0, "torch_version": torch.__version__,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "conceptor_results.json").write_text(canonical_json(json_safe(out)) + "\n")
    brief = [{"config": cid, "target_h1": round(e["kinds"]["main"]["target_h1"]["point"], 4), "nontarget_h0": round(e["kinds"]["main"]["nontarget_h0"]["point"], 4),
              "selective": e["selective"], "p_maxt": e["kinds"]["main"].get("selectivity_h0_test", {}).get("p_maxt_fwer")} for cid, e in agg["results"].items()]
    print(json.dumps(json_safe({"n_scenes": len(pairs), "n_configs": len(agg["results"]), "band_vs_single": agg["band_vs_single"], "configs": brief[:40]}), indent=1))
